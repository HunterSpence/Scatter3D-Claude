#!/bin/bash
# Final benchmark cases: dense-grid sweep-vs-direct, and larger-problem BLR memory.
exec > /work/bench/final_cases_container.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
cd /work/bench
echo "=== DENSE GRID (Nf=6, 9.95-10.05 GHz, 20 MHz steps): MODIFIED direct ==="
SCATT3D_SRC=/work/Scatt3D timeout 1500 python3 bench_driver.py 0.5 1 6 '{}' densefast 9.95e9:10.05e9 2>&1 | grep -E 'BENCH_RESULT|Traceback|error code' | head -4
echo "=== DENSE GRID: MODIFIED sweep ==="
SCATT3D_SRC=/work/Scatt3D timeout 1500 python3 bench_driver.py 0.5 1 6 '{"sweep_mode": true}' densesweep 9.95e9:10.05e9 2>&1 | grep -E 'BENCH_RESULT|re-anchoring|Traceback|error code' | head -8
echo "=== LARGE PROBLEM (h=0.25 deg=1, Nf=1): MODIFIED direct ==="
SCATT3D_SRC=/work/Scatt3D timeout 1500 python3 bench_driver.py 0.25 1 1 '{}' bigfast 2>&1 | grep -E 'BENCH_RESULT|Traceback|error code|NaN' | head -4
echo "=== LARGE PROBLEM: MODIFIED + BLR 1e-6 ==="
SCATT3D_SRC=/work/Scatt3D timeout 1500 python3 bench_driver.py 0.25 1 1 '{"blr_tol": 1e-6}' bigblr 2>&1 | grep -E 'BENCH_RESULT|Traceback|error code|NaN' | head -4
echo "=== EQUIVALENCE ==="
python3 - <<'EOF'
import numpy as np
for a, b in [("densefast", "densesweep"), ("bigfast", "bigblr")]:
    try:
        Sa = np.load(f"S_{a}.npz")["S_ref"]; Sb = np.load(f"S_{b}.npz")["S_ref"]
        print(f"{b} vs {a}: max|dS| = {np.max(np.abs(Sa-Sb)):.3e}")
    except Exception as e:
        print(a, b, "compare failed:", e)
EOF
echo FINAL_CASES_DONE
