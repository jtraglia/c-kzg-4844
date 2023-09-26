package ckzg4844

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v3"
)

func TestMain(m *testing.M) {
	err := LoadTrustedSetupFile("../../src/trusted_setup.txt")
	if err != nil {
		panic("failed to load trusted setup")
	}
	defer FreeTrustedSetup()
	code := m.Run()
	os.Exit(code)
}

///////////////////////////////////////////////////////////////////////////////
// Helper Functions
///////////////////////////////////////////////////////////////////////////////

func getRandFieldElement(seed int64) Bytes32 {
	rand.Seed(seed)
	bytes := make([]byte, 31)
	_, err := rand.Read(bytes)
	if err != nil {
		panic("failed to get random field element")
	}

	// This leaves the first byte in fieldElementBytes as
	// zero, which guarantees it's a canonical field element.
	var fieldElementBytes Bytes32
	copy(fieldElementBytes[1:], bytes)
	return fieldElementBytes
}

func getRandBlob(seed int64) Blob {
	var blob Blob
	for i := 0; i < BytesPerBlob; i += BytesPerFieldElement {
		fieldElementBytes := getRandFieldElement(seed + int64(i))
		copy(blob[i:i+BytesPerFieldElement], fieldElementBytes[:])
	}
	return blob
}

func deleteSamples(samples []Sample, i int) []Sample {
	partialSamples := make([]Sample, GetSampleCount())
	for j := 0; j < GetSampleCount(); j++ {
		if j%i == 0 {
			partialSamples[j] = GetNullSample()
		} else {
			partialSamples[j] = samples[j]
		}
	}
	return partialSamples
}

func getPartialSamples(samples [][]Sample) [][]Sample {
	type Pair[T, U any] struct {
		First  T
		Second U
	}

	partialSamples := make([][]Sample, GetSampleCount())
	indices := make([]Pair[int, int], GetSampleCount()*GetSampleCount())
	for i := 0; i < 2*GetBlobCount(); i++ {
		partialSamples[i] = make([]Sample, GetSampleCount())
		for j := 0; j < GetSampleCount(); j++ {
			indices[i*GetSampleCount()+j] = Pair[int, int]{i, j}
			partialSamples[i][j] = samples[i][j]
		}
	}

	/* Mark the first 25% of shuffled indices as missing */
	rand.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
	count := len(indices) / 4
	for _, index := range indices[:count] {
		partialSamples[index.First][index.Second] = GetNullSample()
	}

	return partialSamples
}

func getRow(samples [][]Sample, row int) []Sample {
	return samples[row]
}

func getColumn(samples [][]Sample, column int) []Sample {
	result := make([]Sample, len(samples[0]))
	for i, row := range samples {
		result[i] = row[column]
	}
	return result
}

///////////////////////////////////////////////////////////////////////////////
// Reference Tests
///////////////////////////////////////////////////////////////////////////////

var (
	testDir                      = "../../tests"
	blobToKZGCommitmentTests     = filepath.Join(testDir, "blob_to_kzg_commitment/*/*/*")
	computeKZGProofTests         = filepath.Join(testDir, "compute_kzg_proof/*/*/*")
	computeBlobKZGProofTests     = filepath.Join(testDir, "compute_blob_kzg_proof/*/*/*")
	verifyKZGProofTests          = filepath.Join(testDir, "verify_kzg_proof/*/*/*")
	verifyBlobKZGProofTests      = filepath.Join(testDir, "verify_blob_kzg_proof/*/*/*")
	verifyBlobKZGProofBatchTests = filepath.Join(testDir, "verify_blob_kzg_proof_batch/*/*/*")
)

func TestBlobToKZGCommitment(t *testing.T) {
	type Test struct {
		Input struct {
			Blob string `yaml:"blob"`
		}
		Output *Bytes48 `yaml:"output"`
	}

	tests, err := filepath.Glob(blobToKZGCommitmentTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var blob Blob
			err = blob.UnmarshalText([]byte(test.Input.Blob))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			commitment, err := BlobToKZGCommitment(blob)
			if err == nil {
				require.NotNil(t, test.Output)
				require.Equal(t, test.Output[:], commitment[:])
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestComputeKZGProof(t *testing.T) {
	type Test struct {
		Input struct {
			Blob string `yaml:"blob"`
			Z    string `yaml:"z"`
		}
		Output *[]string `yaml:"output"`
	}

	tests, err := filepath.Glob(computeKZGProofTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var blob Blob
			err = blob.UnmarshalText([]byte(test.Input.Blob))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var z Bytes32
			err = z.UnmarshalText([]byte(test.Input.Z))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			proof, y, err := ComputeKZGProof(blob, z)
			if err == nil {
				require.NotNil(t, test.Output)
				var expectedProof Bytes48
				err = expectedProof.UnmarshalText([]byte((*test.Output)[0]))
				require.NoError(t, err)
				require.Equal(t, expectedProof[:], proof[:])
				var expectedY Bytes32
				err = expectedY.UnmarshalText([]byte((*test.Output)[1]))
				require.NoError(t, err)
				require.Equal(t, expectedY[:], y[:])
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestComputeBlobKZGProof(t *testing.T) {
	type Test struct {
		Input struct {
			Blob       string `yaml:"blob"`
			Commitment string `yaml:"commitment"`
		}
		Output *Bytes48 `yaml:"output"`
	}

	tests, err := filepath.Glob(computeBlobKZGProofTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var blob Blob
			err = blob.UnmarshalText([]byte(test.Input.Blob))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var commitment Bytes48
			err = commitment.UnmarshalText([]byte(test.Input.Commitment))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			proof, err := ComputeBlobKZGProof(blob, commitment)
			if err == nil {
				require.NotNil(t, test.Output)
				require.Equal(t, test.Output[:], proof[:])
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestVerifyKZGProof(t *testing.T) {
	type Test struct {
		Input struct {
			Commitment string `yaml:"commitment"`
			Z          string `yaml:"z"`
			Y          string `yaml:"y"`
			Proof      string `yaml:"proof"`
		}
		Output *bool `yaml:"output"`
	}

	tests, err := filepath.Glob(verifyKZGProofTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var commitment Bytes48
			err = commitment.UnmarshalText([]byte(test.Input.Commitment))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var z Bytes32
			err = z.UnmarshalText([]byte(test.Input.Z))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var y Bytes32
			err = y.UnmarshalText([]byte(test.Input.Y))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var proof Bytes48
			err = proof.UnmarshalText([]byte(test.Input.Proof))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			valid, err := VerifyKZGProof(commitment, z, y, proof)
			if err == nil {
				require.NotNil(t, test.Output)
				require.Equal(t, *test.Output, valid)
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestVerifyBlobKZGProof(t *testing.T) {
	type Test struct {
		Input struct {
			Blob       string `yaml:"blob"`
			Commitment string `yaml:"commitment"`
			Proof      string `yaml:"proof"`
		}
		Output *bool `yaml:"output"`
	}

	tests, err := filepath.Glob(verifyBlobKZGProofTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var blob Blob
			err = blob.UnmarshalText([]byte(test.Input.Blob))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var commitment Bytes48
			err = commitment.UnmarshalText([]byte(test.Input.Commitment))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			var proof Bytes48
			err = proof.UnmarshalText([]byte(test.Input.Proof))
			if err != nil {
				require.Nil(t, test.Output)
				return
			}

			valid, err := VerifyBlobKZGProof(blob, commitment, proof)
			if err == nil {
				require.NotNil(t, test.Output)
				require.Equal(t, *test.Output, valid)
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestVerifyBlobKZGProofBatch(t *testing.T) {
	type Test struct {
		Input struct {
			Blobs       []string `yaml:"blobs"`
			Commitments []string `yaml:"commitments"`
			Proofs      []string `yaml:"proofs"`
		}
		Output *bool `yaml:"output"`
	}

	tests, err := filepath.Glob(verifyBlobKZGProofBatchTests)
	require.NoError(t, err)
	require.True(t, len(tests) > 0)

	for _, testPath := range tests {
		t.Run(testPath, func(t *testing.T) {
			testFile, err := os.Open(testPath)
			require.NoError(t, err)
			test := Test{}
			err = yaml.NewDecoder(testFile).Decode(&test)
			require.NoError(t, testFile.Close())
			require.NoError(t, err)

			var blobs []Blob
			for _, b := range test.Input.Blobs {
				var blob Blob
				err = blob.UnmarshalText([]byte(b))
				if err != nil {
					require.Nil(t, test.Output)
					return
				}
				blobs = append(blobs, blob)
			}

			var commitments []Bytes48
			for _, c := range test.Input.Commitments {
				var commitment Bytes48
				err = commitment.UnmarshalText([]byte(c))
				if err != nil {
					require.Nil(t, test.Output)
					return
				}
				commitments = append(commitments, commitment)
			}

			var proofs []Bytes48
			for _, p := range test.Input.Proofs {
				var proof Bytes48
				err = proof.UnmarshalText([]byte(p))
				if err != nil {
					require.Nil(t, test.Output)
					return
				}
				proofs = append(proofs, proof)
			}

			valid, err := VerifyBlobKZGProofBatch(blobs, commitments, proofs)
			if err == nil {
				require.NotNil(t, test.Output)
				require.Equal(t, *test.Output, valid)
			} else {
				require.Nil(t, test.Output)
			}
		})
	}
}

func TestSampleProof(t *testing.T) {
	blob := getRandBlob(0)

	commitment, err := BlobToKZGCommitment(blob)
	require.NoError(t, err)
	samples, proofs, err := GetSamplesAndProofs(blob)
	require.NoError(t, err)

	for i := range proofs[:] {
		ok, err := VerifySampleProof(Bytes48(commitment), Bytes48(proofs[i]), samples[i], i)
		require.NoError(t, err)
		require.True(t, ok)
	}
}

func Test2d(t *testing.T) {
	/* Generate some random blobs */
	blobs := make([]Blob, GetBlobCount())
	for i := range blobs {
		blobs[i] = getRandBlob(int64(i))
	}

	/* Get a 2d array of samples for the blobs */
	samples, err := Get2dSamples(blobs[:])
	require.NoError(t, err)

	/* Copy samples so we mark some as missing */
	partialSamples := make([][]Sample, len(samples))
	for i, row := range samples {
		partialSamples[i] = make([]Sample, len(row))
		copy(partialSamples[i], samples[i])
	}

	/* Mark 25% of them as missing */
	for i, row := range samples {
		for j := range row {
			if i%2 == 0 && j%2 == 0 {
				partialSamples[i][j] = GetNullSample()
			}
		}
	}

	/* Recover all of rows */
	for i := range partialSamples {
		row := getRow(samples, i)
		partialRow := getRow(partialSamples, i)
		recovered, err := RecoverSamples(partialRow)
		require.NoError(t, err)

		for j, sample := range row {
			for k := range sample {
				require.Equal(t, row[j][k], recovered[j][k])
			}
		}
	}

	/* Recover all of columns */
	for i := range partialSamples[0] {
		column := getColumn(samples, i)
		partialColumn := getColumn(partialSamples, i)
		recovered, err := RecoverSamples(partialColumn)
		require.NoError(t, err)

		for j, sample := range column {
			for k := range sample {
				require.Equal(t, column[j][k], recovered[j][k])
			}
		}
	}
}

func Test2dRecover(t *testing.T) {
	/* Generate some random blobs */
	blobs := make([]Blob, GetBlobCount())
	for i := range blobs {
		blobs[i] = getRandBlob(int64(i))
	}

	/* Get a 2d array of samples for the blobs */
	samples, err := Get2dSamples(blobs[:])
	require.NoError(t, err)

	/* Mark 25% of them as missing */
	partialSamples := getPartialSamples(samples)

	/* Recover data */
	recovered, err := Recover2dSamples(partialSamples)
	require.NoError(t, err)

	/* Ensure recovered matches original */
	require.Equal(t, len(samples), len(recovered))
	for i := range samples {
		require.Equal(t, len(samples[i]), len(recovered[i]))
		for j := range samples[i] {
			require.Equal(t, samples[i][j], recovered[i][j])
		}
	}
}

func Test2dRecoverFirstRowIsMissing(t *testing.T) {
	/* Generate some random blobs */
	blobs := make([]Blob, GetBlobCount())
	for i := range blobs {
		blobs[i] = getRandBlob(int64(i))
	}

	/* Get a 2d array of samples for the blobs */
	samples, err := Get2dSamples(blobs[:])
	require.NoError(t, err)

	/* Copy samples so we mark some as missing */
	partialSamples := make([][]Sample, len(samples))
	for i, row := range samples {
		partialSamples[i] = make([]Sample, len(row))
		copy(partialSamples[i], samples[i])
	}

	/* Mark the first 75% samples in the first row as null */
	l := (len(partialSamples[0]) / 4) * 3
	for j := range partialSamples[0][:l] {
		partialSamples[0][j] = GetNullSample()
	}

	/* Recover data */
	recovered, err := Recover2dSamples(partialSamples)
	require.NoError(t, err)

	/* Ensure recovered matches original */
	require.Equal(t, len(samples), len(recovered))
	for i := range samples {
		require.Equal(t, len(samples[i]), len(recovered[i]))
		for j := range samples[i] {
			require.Equal(t, samples[i][j], recovered[i][j])
		}
	}
}

func TestRecoverNoMissing(t *testing.T) {
	blob := getRandBlob(0)
	samples, _, err := GetSamplesAndProofs(blob)
	require.NoError(t, err)
	recovered, err := RecoverSamples(samples)
	require.NoError(t, err)
	require.Equal(t, recovered, samples)
}

///////////////////////////////////////////////////////////////////////////////
// Benchmarks
///////////////////////////////////////////////////////////////////////////////

func Benchmark2dRecover(b *testing.B) {
	length := GetBlobCount()
	blobs := make([]Blob, length)
	samples, err := Get2dSamples(blobs)
	require.NoError(b, err)
	partialSamples := getPartialSamples(samples)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, err := Recover2dSamples(partialSamples)
		require.Nil(b, err)
	}
}

func Benchmark(b *testing.B) {
	length := GetBlobCount()
	blobs := make([]Blob, length)
	commitments := make([]Bytes48, length)
	proofs := make([]Bytes48, length)
	fields := make([]Bytes32, length)
	samples := make([][]Sample, length)
	sampleProofs := make([][]KZGProof, length)
	partialSamples := make([][]Sample, length)

	for i := 0; i < length; i++ {
		blob := getRandBlob(int64(i))
		commitment, err := BlobToKZGCommitment(blob)
		require.NoError(b, err)
		proof, err := ComputeBlobKZGProof(blob, Bytes48(commitment))
		require.NoError(b, err)

		blobs[i] = blob
		commitments[i] = Bytes48(commitment)
		proofs[i] = Bytes48(proof)
		fields[i] = getRandFieldElement(int64(i))
		samples[i], sampleProofs[i], err = GetSamplesAndProofs(blobs[i])
		require.NoError(b, err)
		partialSamples[i] = deleteSamples(samples[i], 2)
	}

	b.Run("BlobToKZGCommitment", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _ = BlobToKZGCommitment(blobs[0])
		}
	})

	b.Run("ComputeKZGProof", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _, _ = ComputeKZGProof(blobs[0], fields[0])
		}
	})

	b.Run("ComputeBlobKZGProof", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _ = ComputeBlobKZGProof(blobs[0], commitments[0])
		}
	})

	b.Run("VerifyKZGProof", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _ = VerifyKZGProof(commitments[0], fields[0], fields[1], proofs[0])
		}
	})

	b.Run("VerifyBlobKZGProof", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _ = VerifyBlobKZGProof(blobs[0], commitments[0], proofs[0])
		}
	})

	for i := 1; i <= len(blobs); i *= 2 {
		b.Run(fmt.Sprintf("VerifyBlobKZGProofBatch(count=%v)", i), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_, _ = VerifyBlobKZGProofBatch(blobs[:i], commitments[:i], proofs[:i])
			}
		})
	}

	b.Run("GetSamplesAndProofs", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, _, err := GetSamplesAndProofs(blobs[0])
			require.Nil(b, err)
		}
	})

	b.Run("Get2dSamples", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, err := Get2dSamples(blobs)
			require.Nil(b, err)
		}
	})

	b.Run("SamplesToBlob", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			_, err := SamplesToBlob(samples[0])
			require.Nil(b, err)
		}
	})

	for i := 2; i <= 8; i *= 2 {
		percentMissing := (1.0 / float64(i)) * 100
		partial := deleteSamples(samples[0], i)
		b.Run(fmt.Sprintf("RecoverSamples(missing=%2.1f%%)", percentMissing), func(b *testing.B) {
			for n := 0; n < b.N; n++ {
				_, err := RecoverSamples(partial)
				require.Nil(b, err)
			}
		})
	}

	b.Run("VerifySampleProof", func(b *testing.B) {
		for n := 0; n < b.N; n++ {
			ok, err := VerifySampleProof(commitments[0], Bytes48(sampleProofs[0][0]), samples[0][0], 0)
			require.Nil(b, err)
			require.True(b, ok)
		}
	})
}
