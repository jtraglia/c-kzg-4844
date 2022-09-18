import atexit
import ckzg

def int_from_uint64s(tup):
    res, mult = 0, 1
    for x in tup:
        res += mult * x
        mult *= 2 ** 64
    return res

def eval_poly(coeffs, x):
    res, mult = 0, 1
    for c in coeffs:
        res += mult * c
        mult *= x
    return res

c1 = ckzg.fr_from_uint64s((12,13,0,0))
c2 = ckzg.fr_from_uint64(2)
c3 = ckzg.fr_from_uint64s((1,0,0,0))
c4 = ckzg.fr_sub(c2, c3)

assert ckzg.fr_is_one(c4)

assert ckzg.fr_equal(c3, c4)

coeffs = [c1, c2, c3, c4]
cfa = ckzg.frArray(len(coeffs))
for i, c in enumerate(coeffs):
    cfa[i] = c

ret, pptr = ckzg.new_poly_with_coeffs(cfa.cast(), len(coeffs))
assert ret == 0

p = ckzg.polyp_frompointer(pptr).value()
assert p.length == 4

pcoeffs = ckzg.frArray_frompointer(p.coeffs)
assert ckzg.fr_to_uint64s(pcoeffs[1]) == (2, 0, 0, 0)

max_scale = 4

ret, fs = ckzg.new_fft_settings(max_scale)
assert ret == 0

secret_s = ckzg.blst_scalar_from_uint64((29,3,1,4))

num_secrets = 2 ** max_scale

g1s = ckzg.g1Array(num_secrets)
g2s = ckzg.g2Array(num_secrets)

ckzg.generate_trusted_setup(g1s.cast(), g2s.cast(), secret_s, num_secrets)

ret, ks = ckzg.new_kzg_settings(g1s.cast(), g2s.cast(), num_secrets, fs)
assert ret == 0

ret, p_l = ckzg.new_poly_l_from_poly(p, ks)
assert ret == 0

ret, y_l = ckzg.eval_poly_l(p_l, c2, fs)
assert ret == 0

y = ckzg.eval_poly(p, c2)
assert ckzg.fr_equal(y, y_l)

py_coeffs = [int_from_uint64s(ckzg.fr_to_uint64s(c)) for c in coeffs]

y_p = eval_poly(py_coeffs, int_from_uint64s(ckzg.fr_to_uint64s(c2)))
assert int_from_uint64s(ckzg.fr_to_uint64s(y)) == y_p

def cleanup():
    ckzg.free_poly(pptr)
    ckzg.free_poly_l(p_l)
    ckzg.free_fft_settings(fs)
    ckzg.free_kzg_settings(ks)
atexit.register(cleanup)
