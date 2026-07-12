"""Standalone synthetic qs.h5 + matching npz, for MECHANICAL validation only of
measurement_diagnostics.py's --qs code path (SVD/projection/plotting). The real
e2e_stockoutput-qs.h5 was left incomplete/corrupt by a killed FEM run (only a
few of 32 sensitivity rows written), so this does NOT validate the physics --
it only proves the file-parsing/SVD/plot code runs and produces sane numbers.
"""
import os
import h5py
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(W, "bench/e2e_work/case_synthqs/data3D")
os.makedirs(OUTDIR, exist_ok=True)

Nf, Na, Ncells = 2, 4, 5000
SEED = 99

rng = np.random.default_rng(SEED)
fvec = np.array([8.0e9, 12.0e9])
S_ref = np.tile(np.eye(4, dtype=complex) * (0.999 + 0.001j), (Nf, 1, 1))
S_ref += 1e-3 * (rng.standard_normal((Nf, 4, 4)) + 1j * rng.standard_normal((Nf, 4, 4)))
S_dut = S_ref + 1e-4 * (rng.standard_normal((Nf, 4, 4)) + 1j * rng.standard_normal((Nf, 4, 4)))

np.savez(os.path.join(OUTDIR, "e2e_synthqsoutput.npz"), fvec=fvec, S_ref=S_ref, S_dut=S_dut)

# low-rank-plus-noise rows so SVD/projection give non-degenerate (but not physically
# meaningful) numbers -- purely to exercise the code path.
basis = rng.standard_normal((5, Ncells))
with h5py.File(os.path.join(OUTDIR, "e2e_synthqsoutput-qs.h5"), "w") as f:
    grp_r = f.create_group("Function/real_f")
    grp_i = f.create_group("Function/imag_f")
    for i in range(Nf * Na * Na):
        wr, wi = rng.standard_normal(5), rng.standard_normal(5)
        row_r = wr @ basis + 0.05 * rng.standard_normal(Ncells)
        row_i = wi @ basis + 0.05 * rng.standard_normal(Ncells)
        grp_r.create_dataset(str(i), data=row_r)
        grp_i.create_dataset(str(i), data=row_i)

print(f"wrote {OUTDIR}/e2e_synthqsoutput.npz, e2e_synthqsoutput-qs.h5 "
      f"({Nf*Na*Na} rows x {Ncells} cells)")
