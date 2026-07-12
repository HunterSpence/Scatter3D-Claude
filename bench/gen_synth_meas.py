"""Generate synthetic measurement CSVs from a completed FEM run, with a KNOWN
per-channel calibration error H_mn baked in, for closed-loop validation of
measurement_diagnostics.py.

S_meas(f) = H_mn * S_true(f) + noise,  H fixed per channel (not per frequency).

Output format matches measurement_diagnostics.load_meas_folder exactly:
np.loadtxt(dtype=complex, delimiter=',', skiprows=3) then transpose, so each
FILE ROW is one frequency point: [freq, S(i=0,j=0), S(1,0), S(2,0), S(3,0),
S(0,1), S(1,1), ... S(3,3)] (i fastest, then j) -> 1 + 16 = 17 columns.
"""
import os
import numpy as np

W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIM_NPZ = os.path.join(W, "bench/e2e_work/case_stock/data3D/e2e_stockoutput.npz")
OUT_REF = os.path.join(W, "bench/e2e_work/synth_ref")
OUT_DUT = os.path.join(W, "bench/e2e_work/synth_dut")
NGRID = 21
NOISE = 1e-6
SEED = 1234

os.makedirs(OUT_REF, exist_ok=True)
os.makedirs(OUT_DUT, exist_ok=True)

d = np.load(SIM_NPZ)
fvec, S_ref, S_dut = d["fvec"], d["S_ref"], d["S_dut"]

grid = np.linspace(fvec.min(), fvec.max(), NGRID)  # endpoints == fvec exactly


def interp_S(S):
    """Linear-interpolate a (nf,4,4) complex array onto `grid` (nf==2 -> straight line)."""
    out = np.empty((NGRID, 4, 4), dtype=complex)
    for i in range(4):
        for j in range(4):
            out[:, i, j] = np.interp(grid, fvec, S[:, i, j].real) + \
                           1j * np.interp(grid, fvec, S[:, i, j].imag)
    return out


S_ref_grid = interp_S(S_ref)
S_dut_grid = interp_S(S_dut)

rng = np.random.default_rng(SEED)
mag = rng.uniform(0.5, 2.0, size=(4, 4))
phase = rng.uniform(-np.pi, np.pi, size=(4, 4))
H = mag * np.exp(1j * phase)
np.savez(os.path.join(os.path.dirname(OUT_REF), "synth_H.npz"), H=H)


def add_noise(S):
    noise = NOISE * (rng.standard_normal(S.shape) + 1j * rng.standard_normal(S.shape))
    return H[None, :, :] * S + noise


S_ref_meas = add_noise(S_ref_grid)
S_dut_meas = add_noise(S_dut_grid)


def write_csv(path, freqs, S):
    lines = [
        "# synthetic measurement (gen_synth_meas.py)",
        "# H-calibration-corrupted, noise=1e-6",
        "# freq, S(i=0,j=0), S(1,0), S(2,0), S(3,0), S(0,1), ... S(3,3)",
    ]
    for k, f in enumerate(freqs):
        vals = [str(complex(f, 0.0))] + [str(S[k, i, j]) for j in range(4) for i in range(4)]
        lines.append(",".join(vals))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


write_csv(os.path.join(OUT_REF, "angle0.00.csv"), grid, S_ref_meas)
write_csv(os.path.join(OUT_DUT, "angle0.00.csv"), grid, S_dut_meas)

print(f"grid: {NGRID} pts, {grid[0]/1e9:.3f}-{grid[-1]/1e9:.3f} GHz, "
      f"sim fvec at indices {[int(np.argmin(np.abs(grid - f))) for f in fvec]}")
print(f"H magnitude range: {np.abs(H).min():.3f}-{np.abs(H).max():.3f}")
print(f"wrote {OUT_REF}/angle0.00.csv, {OUT_DUT}/angle0.00.csv, synth_H.npz")
