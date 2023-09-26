package ckzg4844

// #cgo CFLAGS: -I${SRCDIR}/../../src
// #cgo CFLAGS: -I${SRCDIR}/blst_headers
// #ifndef FIELD_ELEMENTS_PER_BLOB
// #define FIELD_ELEMENTS_PER_BLOB 4096
// #endif
// #include "c_kzg_4844.c"
import "C"

import (
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
	"unsafe"

	// So its functions are available during compilation.
	_ "github.com/supranational/blst/bindings/go"
)

const (
	BytesPerBlob         = C.BYTES_PER_BLOB
	BytesPerCommitment   = C.BYTES_PER_COMMITMENT
	BytesPerFieldElement = C.BYTES_PER_FIELD_ELEMENT
	BytesPerProof        = C.BYTES_PER_PROOF
	FieldElementsPerBlob = C.FIELD_ELEMENTS_PER_BLOB
)

type (
	Bytes32       [32]byte
	Bytes48       [48]byte
	KZGCommitment Bytes48
	KZGProof      Bytes48
	Blob          [BytesPerBlob]byte
	Sample        []Bytes32
)

var (
	loaded     = false
	settings   = C.KZGSettings{}
	ErrBadArgs = errors.New("bad arguments")
	ErrError   = errors.New("unexpected error")
	ErrMalloc  = errors.New("malloc failed")
	errorMap   = map[C.C_KZG_RET]error{
		C.C_KZG_OK:      nil,
		C.C_KZG_BADARGS: ErrBadArgs,
		C.C_KZG_ERROR:   ErrError,
		C.C_KZG_MALLOC:  ErrMalloc,
	}
	ErrInvalidDataLength  = errors.New("invalid data length")
	ErrInvalidSampleCount = errors.New("invalid sample count")
	ErrInvalidSampleSize  = errors.New("invalid sample size")
	ErrInvalidBlobCount   = errors.New("invalid blob count")
)

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

// makeErrorFromRet translates an (integral) return value, as reported
// by the C library, into a proper Go error. If there is no error, this
// will return nil.
func makeErrorFromRet(ret C.C_KZG_RET) error {
	err, ok := errorMap[ret]
	if !ok {
		panic(fmt.Sprintf("unexpected return value: %v", ret))
	}
	return err
}

///////////////////////////////////////////////////////////////////////////////
// Unmarshal Functions
///////////////////////////////////////////////////////////////////////////////

func (b *Bytes32) UnmarshalText(input []byte) error {
	inputStr := string(input)
	if strings.HasPrefix(inputStr, "0x") {
		inputStr = strings.TrimPrefix(inputStr, "0x")
	}
	bytes, err := hex.DecodeString(inputStr)
	if err != nil {
		return err
	}
	if len(bytes) != len(b) {
		return ErrBadArgs
	}
	copy(b[:], bytes)
	return nil
}

func (b *Bytes48) UnmarshalText(input []byte) error {
	inputStr := string(input)
	if strings.HasPrefix(inputStr, "0x") {
		inputStr = strings.TrimPrefix(inputStr, "0x")
	}
	bytes, err := hex.DecodeString(inputStr)
	if err != nil {
		return err
	}
	if len(bytes) != len(b) {
		return ErrBadArgs
	}
	copy(b[:], bytes)
	return nil
}

func (b *Blob) UnmarshalText(input []byte) error {
	inputStr := string(input)
	if strings.HasPrefix(inputStr, "0x") {
		inputStr = strings.TrimPrefix(inputStr, "0x")
	}
	bytes, err := hex.DecodeString(inputStr)
	if err != nil {
		return err
	}
	if len(bytes) != len(b) {
		return ErrBadArgs
	}
	copy(b[:], bytes)
	return nil
}

///////////////////////////////////////////////////////////////////////////////
// Internal Helper Functions
///////////////////////////////////////////////////////////////////////////////

func chunk(data []Bytes32) ([]Sample, error) {
	if len(data) != GetDataCount() {
		return []Sample{}, ErrInvalidDataLength
	}
	sampleSize := GetSampleSize()
	sampleCount := GetSampleCount()
	samples := make([]Sample, sampleCount)
	for i := 0; i < sampleCount; i++ {
		samples[i] = data[i*sampleSize : (i+1)*sampleSize]
	}
	return samples, nil
}

func chunk2d(data []Bytes32) ([][]Sample, error) {
	if len(data) != GetSampleCount()*GetDataCount() {
		return [][]Sample{}, ErrInvalidDataLength
	}
	samples := make([][]Sample, GetSampleCount())
	for i := range samples {
		var err error
		samples[i], err = chunk(data[i*GetDataCount() : (i+1)*GetDataCount()])
		if err != nil {
			return [][]Sample{}, ErrInvalidDataLength
		}
	}
	return samples, nil
}

func flatten(samples []Sample) ([]Bytes32, error) {
	if len(samples) != GetSampleCount() {
		return []Bytes32{}, ErrInvalidSampleCount
	}
	sampleSize := GetSampleSize()
	for _, sample := range samples {
		if len(sample) != sampleSize {
			return []Bytes32{}, ErrInvalidSampleSize
		}
	}
	dataCount := GetDataCount()
	data := make([]Bytes32, dataCount)
	for i := 0; i < dataCount; i++ {
		data[i] = samples[i/sampleSize][i%sampleSize]
	}
	return data, nil
}

func flatten2d(samples [][]Sample) ([]Bytes32, error) {
	if len(samples) != GetSampleCount() {
		return []Bytes32{}, ErrInvalidSampleCount
	}
	data := make([]Bytes32, GetSampleCount()*GetDataCount())
	for i, row := range samples {
		rowData, err := flatten(row)
		if err != nil {
			return []Bytes32{}, err
		}
		copy(data[i*GetDataCount():(i+1)*GetDataCount()], rowData)
	}
	return data, nil

}

///////////////////////////////////////////////////////////////////////////////
// Configuration Functions
///////////////////////////////////////////////////////////////////////////////

func GetDataCount() int {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	return int(settings.max_width)
}

func GetSampleSize() int {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	return int(settings.sample_size)
}

func GetSampleCount() int {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	return int(settings.sample_count)
}

func GetBlobCount() int {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	return int(settings.blob_count)
}

func GetNullSample() Sample {
	nullVal := Bytes32{
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
	}
	nullSample := make(Sample, GetSampleSize())
	for i := range nullSample {
		nullSample[i] = nullVal
	}
	return nullSample
}

///////////////////////////////////////////////////////////////////////////////
// Interface Functions
///////////////////////////////////////////////////////////////////////////////

/*
LoadTrustedSetup is the binding for:

	C_KZG_RET load_trusted_setup(
	    KZGSettings *out,
	    const uint8_t *g1_bytes,
	    size_t n1,
	    const uint8_t *g2_bytes,
	    size_t n2);
*/
func LoadTrustedSetup(g1Bytes, g2Bytes []byte) error {
	if loaded {
		panic("trusted setup is already loaded")
	}
	if len(g1Bytes)%C.BYTES_PER_G1 != 0 {
		panic(fmt.Sprintf("len(g1Bytes) is not a multiple of %v", C.BYTES_PER_G1))
	}
	if len(g2Bytes)%C.BYTES_PER_G2 != 0 {
		panic(fmt.Sprintf("len(g2Bytes) is not a multiple of %v", C.BYTES_PER_G2))
	}
	numG1Elements := len(g1Bytes) / C.BYTES_PER_G1
	numG2Elements := len(g2Bytes) / C.BYTES_PER_G2
	ret := C.load_trusted_setup(
		&settings,
		*(**C.uint8_t)(unsafe.Pointer(&g1Bytes)),
		(C.size_t)(numG1Elements),
		*(**C.uint8_t)(unsafe.Pointer(&g2Bytes)),
		(C.size_t)(numG2Elements))
	if ret == C.C_KZG_OK {
		loaded = true
	}
	return makeErrorFromRet(ret)
}

/*
LoadTrustedSetupFile is the binding for:

	C_KZG_RET load_trusted_setup_file(
	    KZGSettings *out,
	    FILE *in);
*/
func LoadTrustedSetupFile(trustedSetupFile string) error {
	if loaded {
		panic("trusted setup is already loaded")
	}
	cTrustedSetupFile := C.CString(trustedSetupFile)
	defer C.free(unsafe.Pointer(cTrustedSetupFile))
	cMode := C.CString("r")
	defer C.free(unsafe.Pointer(cMode))
	fp := C.fopen(cTrustedSetupFile, cMode)
	if fp == nil {
		panic("error reading trusted setup")
	}
	ret := C.load_trusted_setup_file(&settings, fp)
	C.fclose(fp)
	if ret == C.C_KZG_OK {
		loaded = true
	}
	return makeErrorFromRet(ret)
}

/*
FreeTrustedSetup is the binding for:

	void free_trusted_setup(
	    KZGSettings *s);
*/
func FreeTrustedSetup() {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	C.free_trusted_setup(&settings)
	loaded = false
}

/*
BlobToKZGCommitment is the binding for:

	C_KZG_RET blob_to_kzg_commitment(
	    KZGCommitment *out,
	    const Blob *blob,
	    const KZGSettings *s);
*/
func BlobToKZGCommitment(blob Blob) (KZGCommitment, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	commitment := KZGCommitment{}
	ret := C.blob_to_kzg_commitment(
		(*C.KZGCommitment)(unsafe.Pointer(&commitment)),
		(*C.Blob)(unsafe.Pointer(&blob)),
		&settings)
	return commitment, makeErrorFromRet(ret)
}

/*
ComputeKZGProof is the binding for:

	C_KZG_RET compute_kzg_proof(
	    KZGProof *proof_out,
	    Bytes32 *y_out,
	    const Blob *blob,
	    const Bytes32 *z_bytes,
	    const KZGSettings *s);
*/
func ComputeKZGProof(blob Blob, zBytes Bytes32) (KZGProof, Bytes32, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	proof := KZGProof{}
	y := Bytes32{}
	ret := C.compute_kzg_proof(
		(*C.KZGProof)(unsafe.Pointer(&proof)),
		(*C.Bytes32)(unsafe.Pointer(&y)),
		(*C.Blob)(unsafe.Pointer(&blob)),
		(*C.Bytes32)(unsafe.Pointer(&zBytes)),
		&settings)
	return proof, y, makeErrorFromRet(ret)
}

/*
ComputeBlobKZGProof is the binding for:

	C_KZG_RET compute_blob_kzg_proof(
	    KZGProof *out,
	    const Blob *blob,
	    const Bytes48 *commitment_bytes,
	    const KZGSettings *s);
*/
func ComputeBlobKZGProof(blob Blob, commitmentBytes Bytes48) (KZGProof, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	proof := KZGProof{}
	ret := C.compute_blob_kzg_proof(
		(*C.KZGProof)(unsafe.Pointer(&proof)),
		(*C.Blob)(unsafe.Pointer(&blob)),
		(*C.Bytes48)(unsafe.Pointer(&commitmentBytes)),
		&settings)
	return proof, makeErrorFromRet(ret)
}

/*
VerifyKZGProof is the binding for:

	C_KZG_RET verify_kzg_proof(
	    bool *out,
	    const Bytes48 *commitment_bytes,
	    const Bytes32 *z_bytes,
	    const Bytes32 *y_bytes,
	    const Bytes48 *proof_bytes,
	    const KZGSettings *s);
*/
func VerifyKZGProof(commitmentBytes Bytes48, zBytes, yBytes Bytes32, proofBytes Bytes48) (bool, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	var result C.bool
	ret := C.verify_kzg_proof(
		&result,
		(*C.Bytes48)(unsafe.Pointer(&commitmentBytes)),
		(*C.Bytes32)(unsafe.Pointer(&zBytes)),
		(*C.Bytes32)(unsafe.Pointer(&yBytes)),
		(*C.Bytes48)(unsafe.Pointer(&proofBytes)),
		&settings)
	return bool(result), makeErrorFromRet(ret)
}

/*
VerifyBlobKZGProof is the binding for:

	C_KZG_RET verify_blob_kzg_proof(
	    bool *out,
	    const Blob *blob,
	    const Bytes48 *commitment_bytes,
	    const Bytes48 *proof_bytes,
	    const KZGSettings *s);
*/
func VerifyBlobKZGProof(blob Blob, commitmentBytes, proofBytes Bytes48) (bool, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	var result C.bool
	ret := C.verify_blob_kzg_proof(
		&result,
		(*C.Blob)(unsafe.Pointer(&blob)),
		(*C.Bytes48)(unsafe.Pointer(&commitmentBytes)),
		(*C.Bytes48)(unsafe.Pointer(&proofBytes)),
		&settings)
	return bool(result), makeErrorFromRet(ret)
}

/*
VerifyBlobKZGProofBatch is the binding for:

	C_KZG_RET verify_blob_kzg_proof_batch(
	    bool *out,
	    const Blob *blobs,
	    const Bytes48 *commitments_bytes,
	    const Bytes48 *proofs_bytes,
	    const KZGSettings *s);
*/
func VerifyBlobKZGProofBatch(blobs []Blob, commitmentsBytes, proofsBytes []Bytes48) (bool, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	if len(blobs) != len(commitmentsBytes) || len(blobs) != len(proofsBytes) {
		return false, ErrBadArgs
	}

	var result C.bool
	ret := C.verify_blob_kzg_proof_batch(
		&result,
		*(**C.Blob)(unsafe.Pointer(&blobs)),
		*(**C.Bytes48)(unsafe.Pointer(&commitmentsBytes)),
		*(**C.Bytes48)(unsafe.Pointer(&proofsBytes)),
		(C.size_t)(len(blobs)),
		&settings)
	return bool(result), makeErrorFromRet(ret)
}

/*
GetSamplesAndProofs is the binding for:

	C_KZG_RET get_samples_and_proofs(
	    Bytes32 *data,
	    KZGProof *proofs,
	    const Blob *blob,
	    const KZGSettings *s);
*/
func GetSamplesAndProofs(blob Blob) ([]Sample, []KZGProof, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	data := make([]Bytes32, GetDataCount())
	proofs := make([]KZGProof, GetSampleCount())
	err := makeErrorFromRet(C.get_samples_and_proofs(
		*(**C.Bytes32)(unsafe.Pointer(&data)),
		*(**C.KZGProof)(unsafe.Pointer(&proofs)),
		(*C.Blob)(unsafe.Pointer(&blob)),
		&settings))
	if err != nil {
		return []Sample{}, []KZGProof{}, err
	}
	samples, err := chunk(data)
	if err != nil {
		return []Sample{}, []KZGProof{}, err
	}
	return samples, proofs, nil
}

func Get2dSamples(blobs []Blob) ([][]Sample, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	if len(blobs) != GetBlobCount() {
		return [][]Sample{}, ErrInvalidBlobCount
	}
	data := make([]Bytes32, 2*GetBlobCount()*GetDataCount())
	err := makeErrorFromRet(C.get_2d_samples(
		*(**C.Bytes32)(unsafe.Pointer(&data)),
		*(**C.Blob)(unsafe.Pointer(&blobs)),
		&settings))
	samples := make([][]Sample, 2*GetBlobCount())
	for i := range samples {
		samples[i], err = chunk(data[i*GetDataCount() : (i+1)*GetDataCount()])
	}
	return samples, err
}

/*
SamplesToBlob is the binding for:

	C_KZG_RET samples_to_blob(
	    Blob *blob,
	    const Bytes32 *data,
	    const KZGSettings *s);
*/
func SamplesToBlob(samples []Sample) (Blob, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	blob := Blob{}
	data, err := flatten(samples)
	if err != nil {
		return blob, err
	}
	ret := C.samples_to_blob(
		(*C.Blob)(unsafe.Pointer(&blob)),
		*(**C.Bytes32)(unsafe.Pointer(&data)),
		&settings)
	return blob, makeErrorFromRet(ret)
}

/*
RecoverSamples is the binding for:

	C_KZG_RET recover_samples(
	    Bytes32 *recovered,
	    const Bytes32 *data,
	    const KZGSettings *s);
*/
func RecoverSamples(samples []Sample) ([]Sample, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	partialData, err := flatten(samples)
	if err != nil {
		return []Sample{}, err
	}
	recoveredData := make([]Bytes32, GetDataCount())
	err = makeErrorFromRet(C.recover_samples(
		*(**C.Bytes32)(unsafe.Pointer(&recoveredData)),
		*(**C.Bytes32)(unsafe.Pointer(&partialData)),
		&settings))
	if err != nil {
		return []Sample{}, err
	}
	recovered, err := chunk(recoveredData)
	if err != nil {
		return []Sample{}, err
	}
	return recovered, nil
}

/*
Recover2dSamples is the binding for:

	C_KZG_RET recover_samples(
	    Bytes32 *recovered,
	    const Bytes32 *data,
	    const KZGSettings *s);
*/
func Recover2dSamples(samples [][]Sample) ([][]Sample, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	partialData, err := flatten2d(samples)
	if err != nil {
		return [][]Sample{}, err
	}
	recoveredData := make([]Bytes32, GetSampleCount()*GetDataCount())
	err = makeErrorFromRet(C.recover_2d_samples(
		*(**C.Bytes32)(unsafe.Pointer(&recoveredData)),
		*(**C.Bytes32)(unsafe.Pointer(&partialData)),
		&settings))
	if err != nil {
		return [][]Sample{}, err
	}
	recovered, err := chunk2d(recoveredData)
	if err != nil {
		return [][]Sample{}, err
	}
	return recovered, nil
}

/*
VerifySampleProof is the binding for:

	C_KZG_RET verify_sample_proof(
	    bool *ok,
	    const Bytes48 *commitment_bytes,
	    const Bytes48 *proof_bytes,
	    const Bytes32 *data,
	    size_t index,
	    const KZGSettings *s);
*/
func VerifySampleProof(commitment, proof Bytes48, sample Sample, index int) (bool, error) {
	if !loaded {
		panic("trusted setup isn't loaded")
	}
	if len(sample) != GetSampleSize() {
		return false, ErrBadArgs
	}
	var result C.bool
	ret := C.verify_sample_proof(
		&result,
		(*C.Bytes48)(unsafe.Pointer(&commitment)),
		(*C.Bytes48)(unsafe.Pointer(&proof)),
		*(**C.Bytes32)(unsafe.Pointer(&sample)),
		(C.size_t)(index),
		&settings)
	return bool(result), makeErrorFromRet(ret)
}
