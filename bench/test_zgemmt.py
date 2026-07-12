# Unit test for the zgemmt shim: every uplo/transa/transb combo, sizes spanning the
# 192-column block boundary, beta=0 must-not-read-C semantics, padded ldas.
import ctypes
import numpy as np

lib = ctypes.CDLL("./libzgemmt_fix.so")
zgemmt = lib.zgemmt_

rng = np.random.default_rng(0)
fails = 0

def run(uplo, ta, tb, n, k, beta, pad=0):
    global fails
    An = (n, k) if ta == "N" else (k, n)
    Bn = (k, n) if tb == "N" else (n, k)
    lda, ldb, ldc = An[0] + pad, Bn[0] + pad, n + pad
    A = np.zeros((lda, An[1]), dtype=np.complex128, order="F")
    B = np.zeros((ldb, Bn[1]), dtype=np.complex128, order="F")
    C = np.zeros((ldc, n), dtype=np.complex128, order="F")
    A[:An[0]] = rng.standard_normal(An) + 1j * rng.standard_normal(An)
    B[:Bn[0]] = rng.standard_normal(Bn) + 1j * rng.standard_normal(Bn)
    C[:n] = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    alpha = 1.3 - 0.7j
    C0 = C.copy()
    tri = np.triu_indices(n) if uplo == "U" else np.tril_indices(n)
    if beta == 0:  # BLAS beta=0: C's referenced triangle must not be read
        Cn = C[:n, :n].copy()
        Cn[tri] = np.nan
        C[:n, :n] = Cn
    opA = {"N": A[:n, :k] if ta == "N" else None, "T": A[:k, :n].T if ta != "N" else None,
           "C": A[:k, :n].conj().T if ta == "C" else None}[ta]
    opB = {"N": B[:k, :n] if tb == "N" else None, "T": B[:n, :k].T if tb != "N" else None,
           "C": B[:n, :k].conj().T if tb == "C" else None}[tb]
    full = alpha * (opA @ opB) + (beta * C0[:n, :n] if beta != 0 else 0)
    zgemmt(ctypes.c_char_p(uplo.encode()), ctypes.c_char_p(ta.encode()), ctypes.c_char_p(tb.encode()),
           ctypes.byref(ctypes.c_int(n)), ctypes.byref(ctypes.c_int(k)),
           (ctypes.c_double * 2)(alpha.real, alpha.imag), A.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ctypes.c_int(lda)),
           B.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ctypes.c_int(ldb)),
           (ctypes.c_double * 2)(beta.real if beta else 0.0, beta.imag if beta else 0.0),
           C.ctypes.data_as(ctypes.c_void_p), ctypes.byref(ctypes.c_int(ldc)),
           ctypes.c_size_t(1), ctypes.c_size_t(1), ctypes.c_size_t(1))
    got = C[:n, :n]
    err = np.max(np.abs(got[tri] - full[tri])) / max(1e-300, np.max(np.abs(full[tri])))
    anti = (np.tril_indices(n, -1) if uplo == "U" else np.triu_indices(n, 1))
    untouched = np.array_equal(got[anti], C0[:n, :n][anti])
    ok = err < 1e-13 and untouched and np.all(np.isfinite(got[tri]))
    if not ok:
        fails += 1
    print(f"uplo={uplo} ta={ta} tb={tb} n={n} k={k} beta={beta} pad={pad}: "
          f"relerr={err:.2e} untouched={untouched} {'OK' if ok else 'FAIL'}")

for uplo in "UL":
    for ta in "NTC":
        for tb in "NTC":
            run(uplo, ta, tb, 317, 71, 0.9 + 0.2j)
run("U", "N", "T", 400, 100, 0)          # beta=0 NaN semantics
run("L", "N", "T", 400, 100, 0)
run("U", "T", "N", 191, 3, 1.0 + 0j, pad=5)   # just under block size, padded ld
run("L", "C", "C", 193, 64, 0.5 - 0.5j, pad=3)  # just over block size
run("U", "N", "N", 1, 1, 0.7 + 0j)       # degenerate
print("ALL_PASS" if fails == 0 else f"{fails} FAILURES")
