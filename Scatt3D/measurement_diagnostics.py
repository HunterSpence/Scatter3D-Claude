# encoding: utf-8
"""
Standalone diagnostics for the Scatt3D measurement pipeline.

Answers, from the measured data alone (no FEM required):
  1. Is the differential defect signal |dS| = |S_dut - S_ref| above the measurement
     error floor (reciprocity error + drift)? If not, no inversion can work.
  2. How badly did the (now fixed) transposed-reference-index bug corrupt b?
  3. How far apart are measurement and simulation per channel (the complex
     calibration factors the current phase-only correction ignores)?
  4. Does the measured data vector b actually live in the range space of the
     simulated sensitivity matrix A? (TSVD projection test: if it does not,
     the image is guaranteed to be noise regardless of truncation choice.)

Inputs use the exact formats of the existing pipeline:
  - measured folders with angle{angle:.2f}.csv (np.loadtxt complex CSV, 3 header
    rows; after transpose row 0 = frequency, S[i][j] = row[1 + i + j*4])
  - simulation output: <problemName>output.npz (keys fvec, S_ref[, S_dut])
  - sensitivity rows:  <problemName>output-qs.h5 (Function/real_f/<i>, imag_f)

Usage examples:
  python measurement_diagnostics.py --ref MEAS/ref --dut MEAS/dut
  python measurement_diagnostics.py --ref MEAS/ref --dut MEAS/dut --sim data3D/myrun
  python measurement_diagnostics.py --ref MEAS/ref --dut MEAS/dut --sim data3D/myrun --qs
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

C0 = 299792458.0


def load_meas_folder(folder, angles=None):
    """Returns (angles, freqs, S[nAngle, nFreq, 4, 4]) from a folder of angle*.csv files."""
    if angles is None:
        files = sorted(glob.glob(os.path.join(folder, "angle*.csv")))
        angles = sorted(float(re.search(r"angle(-?[\d.]+)\.csv", os.path.basename(f)).group(1)) for f in files)
    if not angles:
        raise FileNotFoundError(f"no angle*.csv files in {folder}")
    S_all = []
    freqs = None
    for angle in angles:
        fname = os.path.join(folder, f"angle{angle:.2f}.csv")
        Sdata = np.transpose(np.loadtxt(fname, dtype=complex, delimiter=",", skiprows=3))
        if freqs is None:
            freqs = np.real(Sdata[0])
        S = np.zeros((len(freqs), 4, 4), dtype=complex)
        for i in range(4):
            for j in range(4):
                S[:, i, j] = Sdata[1 + i + j * 4, :]
        S_all.append(S)
    return np.array(angles), freqs, np.array(S_all)


def db(x):
    return 20 * np.log10(np.maximum(np.abs(x), 1e-300))


def reciprocity_check(name, S):
    """Reciprocity error |S_mn - S_nm| is a floor on trustable transmission signal:
    the FEM model is exactly reciprocal, so any measured asymmetry is pure error."""
    errs, sigs = [], []
    for m in range(4):
        for n in range(m + 1, 4):
            errs.append(np.abs(S[:, :, m, n] - S[:, :, n, m]).ravel())
            sigs.append(np.abs(0.5 * (S[:, :, m, n] + S[:, :, n, m])).ravel())
    errs, sigs = np.concatenate(errs), np.concatenate(sigs)
    print(f"[reciprocity] {name}: median|S_mn-S_nm| = {np.median(errs):.3e} "
          f"({np.median(db(errs)):.1f} dB), 95th pct = {np.percentile(errs, 95):.3e}; "
          f"median relative to |S_mn| = {np.median(errs / np.maximum(sigs, 1e-300)):.2%}")
    return np.median(errs)


def signal_check(S_ref, S_dut, recip_floor):
    """Is the defect signal above the error floor?"""
    dS = S_dut - S_ref
    refl = np.abs(np.array([dS[:, :, m, m] for m in range(4)])).ravel()
    trans = np.abs(np.array([dS[:, :, m, n] for m in range(4) for n in range(4) if m != n])).ravel()
    print(f"[signal] median|dS| transmission = {np.median(trans):.3e} ({np.median(db(trans)):.1f} dB), "
          f"reflection = {np.median(refl):.3e} ({np.median(db(refl)):.1f} dB)")
    ratio = np.median(trans) / max(recip_floor, 1e-300)
    print(f"[signal] defect-signal / reciprocity-error-floor = {ratio:.2f}")
    if ratio < 3:
        print("[signal] *** VERDICT: differential signal is at or below the measurement error "
              "floor - no inversion can succeed until |dS| is raised (bigger contrast/defect, "
              "more averaging, less drift) or the floor is lowered. ***")
    else:
        print("[signal] verdict: signal is above the reciprocity floor - imaging is not "
              "hopeless on SNR grounds alone; look at calibration/model mismatch next.")
    return dS


def bug_impact(S_ref, S_dut):
    """Quantifies what the transposed-index bug (b = S_dut[m,n] - S_ref[n,m]) injected."""
    spurious, signal = [], []
    for m in range(4):
        for n in range(4):
            if m == n:
                continue
            spurious.append(np.abs(S_ref[:, :, m, n] - S_ref[:, :, n, m]).ravel())
            signal.append(np.abs(S_dut[:, :, m, n] - S_ref[:, :, m, n]).ravel())
    spurious, signal = np.concatenate(spurious), np.concatenate(signal)
    print(f"[bug] spurious term injected by the old transposed indexing: median = "
          f"{np.median(spurious):.3e}; genuine dS median = {np.median(signal):.3e}; "
          f"median spurious/signal = {np.median(spurious) / max(np.median(signal), 1e-300):.2f}")
    if np.median(spurious) > 0.5 * np.median(signal):
        print("[bug] *** the old b-vector bug injected error comparable to or larger than the "
              "defect signal in transmission rows - re-run imaging with the fixed indexing. ***")


def nearest_freq_idx(meas_freqs, sim_freqs):
    return np.array([int(np.argmin(np.abs(meas_freqs - f))) for f in sim_freqs])


def calibration_factors(meas_freqs, S_meas_ref, sim_npz, outdir):
    """Per-channel complex factor c_mn(f) = S_sim_ref/S_meas_ref - what a proper
    calibration must supply and the current phase-only correction ignores."""
    data = np.load(sim_npz)
    sim_f, S_sim = data["fvec"], data["S_ref"]
    idx = nearest_freq_idx(meas_freqs, sim_f)
    Sm = S_meas_ref[idx]
    c = S_sim / np.where(np.abs(Sm) > 1e-300, Sm, np.nan)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for m in range(4):
        for n in range(4):
            style = "-" if m == n else "--"
            axes[0].plot(sim_f / 1e9, db(c[:, m, n]), style, label=f"c{m}{n}" if m <= n else None)
            axes[1].plot(sim_f / 1e9, np.unwrap(np.angle(c[:, m, n])), style)
    axes[0].set_ylabel("|c| (dB)"); axes[1].set_ylabel("phase(c) (rad)"); axes[1].set_xlabel("f (GHz)")
    axes[0].set_title("Calibration factor c = S_sim_ref / S_meas_ref per channel\n"
                      "(flat curves = correctable; wild structure = model/measurement mismatch)")
    axes[0].legend(fontsize=6, ncol=4)
    fig.savefig(os.path.join(outdir, "calibration_factors.png"), dpi=150)
    print(f"[cal] |c| spread across channels/freq: {np.nanmedian(db(c)):.1f} dB median, "
          f"{np.nanpercentile(db(c), 5):.1f}..{np.nanpercentile(db(c), 95):.1f} dB (5th-95th pct)")
    for m in range(4):
        ph = np.unwrap(np.angle(c[:, m, m]))
        if len(sim_f) > 1:
            slope = np.polyfit(sim_f, ph, 1)[0]
            tau = -slope / (2 * np.pi)
            print(f"[cal] antenna {m}: reflection-phase slope -> electrical delay {tau*1e12:.1f} ps "
                  f"(~{tau*C0*1000/2:.1f} mm one-way reference-plane offset)")
    np.savez(os.path.join(outdir, "calibration_factors.npz"), c=c, fvec=sim_f)
    print(f"[cal] factors saved to {outdir}/calibration_factors.npz - apply as "
          "S_meas_calibrated = c * S_meas before differencing.")
    return c, idx


def projection_test(qs_prefix, sim_npz, meas_freqs, S_meas_ref, S_meas_dut, outdir, cal=None):
    """Does measured b lie in the range space of A? Compare against simulated-dut b."""
    import h5py
    data = np.load(sim_npz)
    sim_f, S_sim_ref = data["fvec"], data["S_ref"]
    S_sim_dut = data["S_dut"] if "S_dut" in data.files else None
    Nf, Na = len(sim_f), S_sim_ref.shape[1]
    idx = nearest_freq_idx(meas_freqs, sim_f)
    qs_path = qs_prefix + "output-qs.h5"
    rows, b_meas, b_sim = [], [], []
    try:
        with h5py.File(qs_path, "r") as f:
            real_f, imag_f = f["Function"]["real_f"], f["Function"]["imag_f"]
            missing = [i for i in range(Nf * Na * Na)
                       if str(i) not in real_f or str(i) not in imag_f]
            if missing:
                sys.exit(f"[projection] {qs_path}: missing {len(missing)}/{Nf * Na * Na} "
                          f"sensitivity rows (indices {missing}) - qs.h5 looks incomplete "
                          "(FEM run interrupted mid-write?). Re-run to completion, then retry --qs.")
            for nf in range(Nf):
                for m in range(Na):
                    for n in range(Na):
                        i = nf * Na * Na + m * Na + n
                        q = np.array(real_f[str(i)]).squeeze() + 1j * np.array(imag_f[str(i)]).squeeze()
                        rows.append(q)
                        dSm = S_meas_dut[idx[nf], m, n] - S_meas_ref[idx[nf], m, n]
                        if cal is not None:
                            dSm = cal[nf, m, n] * dSm
                        b_meas.append(dSm)
                        if S_sim_dut is not None:
                            b_sim.append(S_sim_dut[nf, m, n] - S_sim_ref[nf, m, n])
    except (OSError, RuntimeError, KeyError) as e:
        sys.exit(f"[projection] cannot read {qs_path}: {e} - file is missing/corrupt/incomplete "
                  "(e.g. FEM run killed mid-write). Re-run to completion, then retry --qs.")
    A = np.array(rows)
    b_meas = np.array(b_meas)
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    def in_space_fraction(b):
        proj = np.abs(U.conj().T @ b) ** 2
        return np.cumsum(proj) / np.dot(b.conj(), b).real
    fm = in_space_fraction(b_meas)
    ks = np.arange(1, len(s) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, fm, label="measured b")
    if b_sim:
        ax.plot(ks, in_space_fraction(np.array(b_sim)), label="simulated b")
    ax2 = ax.twinx(); ax2.semilogy(ks, s / s[0], "k:", alpha=0.5); ax2.set_ylabel("sigma_k/sigma_1")
    ax.set_xlabel("retained singular vectors k"); ax.set_ylabel("fraction of ||b||^2 captured")
    ax.legend(); ax.set_title("Range-space projection test")
    fig.savefig(os.path.join(outdir, "projection_test.png"), dpi=150)
    k10 = max(1, int(0.1 * len(s)))
    print(f"[projection] measured b energy in top-10% singular subspace: {fm[k10-1]:.2%}"
          + (f"; simulated b: {in_space_fraction(np.array(b_sim))[k10-1]:.2%}" if b_sim else ""))
    print("[projection] interpretation: if simulated b concentrates in the leading subspace but "
          "measured b spreads evenly (like noise), the failure is model/data mismatch "
          "(calibration, geometry, environment), NOT just low SNR.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ref", required=True, help="folder with reference-object angle*.csv")
    p.add_argument("--dut", required=True, help="folder with DUT angle*.csv")
    p.add_argument("--sim", default=None, help="simulation problemName prefix (loads <sim>output.npz)")
    p.add_argument("--qs", action="store_true", help="also run the A-matrix projection test (loads <sim>output-qs.h5)")
    p.add_argument("--apply-cal", action="store_true", help="apply per-channel calibration factors in the projection test")
    p.add_argument("--angles", type=float, nargs="*", default=None)
    p.add_argument("--out", default="diagnostics_out")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    ang_r, f_r, S_ref = load_meas_folder(args.ref, args.angles)
    ang_d, f_d, S_dut = load_meas_folder(args.dut, args.angles)
    if not np.array_equal(ang_r, ang_d) or not np.allclose(f_r, f_d):
        print("WARNING: ref and dut folders have different angles or frequency grids")
    print(f"loaded {len(ang_r)} angles x {len(f_r)} freqs "
          f"({f_r[0]/1e9:.2f}-{f_r[-1]/1e9:.2f} GHz)")

    floor_r = reciprocity_check("ref", S_ref)
    reciprocity_check("dut", S_dut)
    signal_check(S_ref, S_dut, floor_r)
    bug_impact(S_ref, S_dut)

    cal = None
    if args.sim:
        cal, _ = calibration_factors(f_r, S_ref[0], args.sim + "output.npz", args.out)
    if args.qs:
        if not args.sim:
            sys.exit("--qs requires --sim")
        projection_test(args.sim, args.sim + "output.npz", f_r, S_ref[0], S_dut[0],
                        args.out, cal=cal if args.apply_cal else None)
    print(f"done - plots in {args.out}/")


if __name__ == "__main__":
    main()
