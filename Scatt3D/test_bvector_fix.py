# Unit test for the postProcessing.py b-vector fix (the ONE imaging-side code change).
# Proves: (1) on reciprocal (simulated) S-matrices the fix is a no-op — zero regression
# risk for every all-simulation result; (2) on non-reciprocal (measured) S-matrices the
# old indexing injected exactly the reference reciprocity error into transmission rows.
import numpy as np

Nf, Na = 5, 4
rng = np.random.default_rng(1)


def build_b(S_dut, S_ref, transposed_ref):
    b = np.zeros(Nf * Na * Na, dtype=complex)
    i = 0
    for nf in range(Nf):
        for m in range(Na):
            for n in range(Na):
                if transposed_ref:
                    b[i] = S_dut[nf, m, n] - S_ref[nf, n, m]  # old (buggy) line
                else:
                    b[i] = S_dut[nf, m, n] - S_ref[nf, m, n]  # fixed line
                i += 1
    return b


# Case 1: reciprocal S (what the FEM produces) -> old and new must agree exactly
S = rng.normal(size=(Nf, Na, Na)) + 1j * rng.normal(size=(Nf, Na, Na))
S_ref = 0.5 * (S + np.transpose(S, (0, 2, 1)))
S_dut = S_ref + 1e-3 * (rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape))
assert np.array_equal(build_b(S_dut, S_ref, True), build_b(S_dut, S_ref, False)), \
    "fix must be a no-op on reciprocal data"

# Case 2: non-reciprocal measured-like S_ref -> old line corrupts transmission rows by
# exactly the reciprocity asymmetry, new line does not
asym = 1e-4 * (rng.normal(size=S.shape) + 1j * rng.normal(size=S.shape))
S_ref_meas = S_ref + asym - np.transpose(asym, (0, 2, 1))
b_old = build_b(S_dut, S_ref_meas, True)
b_new = build_b(S_dut, S_ref_meas, False)
diff = (b_old - b_new).reshape(Nf, Na, Na)
expected = S_ref_meas - np.transpose(S_ref_meas, (0, 2, 1))  # injected spurious term
assert np.allclose(diff, expected), "old-vs-new difference must equal the reciprocity error"
assert np.allclose(np.diagonal(diff, axis1=1, axis2=2), 0), "reflection rows unaffected"
assert np.max(np.abs(diff)) > 0, "non-reciprocal case must actually differ"

print("B-VECTOR FIX UNIT TEST PASSED")
print(f"  reciprocal (sim) data: old == new exactly (no-op, zero regression)")
print(f"  non-reciprocal (measured-like) data: old line injected spurious term with "
      f"max |err| = {np.max(np.abs(expected)):.3e} into transmission rows; fixed line does not")
