# Self-check for measurement_diagnostics.py on synthetic data with known properties.
import os
import subprocess
import sys
import tempfile

import numpy as np

NF, NA = 51, 4
RECIP_ERR = 1e-4     # injected reciprocity asymmetry
DEFECT_SIG = 3e-3    # injected defect delta-S on transmission channels


def write_folder(folder, S, freqs):
    os.makedirs(folder, exist_ok=True)
    rows = np.zeros((NF, 17), dtype=complex)
    rows[:, 0] = freqs
    for i in range(4):
        for j in range(4):
            rows[:, 1 + i + j * 4] = S[:, i, j]
    with open(os.path.join(folder, "angle0.00.csv"), "w") as f:
        f.write("h1\nh2\nh3\n")
        for r in rows:
            f.write(",".join(str(z) for z in r) + "\n")


def main():
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp()
    freqs = np.linspace(5e9, 7e9, NF)
    base = 0.1 * np.exp(1j * rng.uniform(0, 2 * np.pi, (NF, NA, NA)))
    base = 0.5 * (base + np.transpose(base, (0, 2, 1)))  # reciprocal baseline
    asym = RECIP_ERR * np.exp(1j * rng.uniform(0, 2 * np.pi, (NF, NA, NA)))
    S_ref = base + asym - np.transpose(asym, (0, 2, 1))  # known asymmetry
    dS = DEFECT_SIG * np.exp(1j * rng.uniform(0, 2 * np.pi, (NF, NA, NA)))
    S_dut = S_ref + dS
    write_folder(os.path.join(tmp, "ref"), S_ref, freqs)
    write_folder(os.path.join(tmp, "dut"), S_dut, freqs)

    out = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "measurement_diagnostics.py"),
         "--ref", os.path.join(tmp, "ref"), "--dut", os.path.join(tmp, "dut"),
         "--out", os.path.join(tmp, "diag")],
        capture_output=True, text=True)
    print(out.stdout, out.stderr)
    assert out.returncode == 0
    # reciprocity floor should be ~2*RECIP_ERR (difference of two asym draws), well under 1e-3
    line = [l for l in out.stdout.splitlines() if "[reciprocity] ref" in l][0]
    floor = float(line.split("median|S_mn-S_nm| = ")[1].split(" ")[0])
    assert 0.5 * RECIP_ERR < floor < 10 * RECIP_ERR, floor
    # signal median should be ~DEFECT_SIG and verdict should NOT be hopeless (ratio >> 3)
    line = [l for l in out.stdout.splitlines() if "[signal] median|dS|" in l][0]
    sig = float(line.split("transmission = ")[1].split(" ")[0])
    assert 0.5 * DEFECT_SIG < sig < 2 * DEFECT_SIG, sig
    assert any("verdict: signal is above" in l for l in out.stdout.splitlines())
    print("SELF-TEST PASSED")


if __name__ == "__main__":
    main()
