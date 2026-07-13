#!/bin/bash
# Stock legs compare against an unmodified upstream tree: from the mounted repo root,
#   git clone https://github.com/Wojoxiw/Scatt3D upstream-Scatt3D
# (giving /work/upstream-Scatt3D/Scatt3D); the modified tree is this repo at /work/Scatt3D.
# Head-to-head: stock upstream vs modified solver (direct / BLR / sweep). Run inside container from /work/bench.
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
set -e
H=${1:-0.5}; DEG=${2:-1}; NF=${3:-3}
rm -rf data3D S_*.npz
echo "=== STOCK (upstream, LinearProblem re-factorizes every solve) ==="
SCATT3D_SRC=/work/upstream-Scatt3D/Scatt3D timeout 1200 python3 bench_driver.py $H $DEG $NF '{}' stock 2>&1 | grep -E 'BENCH_RESULT|1st solution|NaN|Traceback|Error' | head -5
echo "=== MODIFIED direct (factorize once per frequency) ==="
SCATT3D_SRC=/work/Scatt3D timeout 1200 python3 bench_driver.py $H $DEG $NF '{}' fast 2>&1 | grep -E 'BENCH_RESULT|1st solution|NaN|Traceback|Error' | head -5
echo "=== MODIFIED + BLR (tol 1e-6) ==="
SCATT3D_SRC=/work/Scatt3D timeout 1200 python3 bench_driver.py $H $DEG $NF '{"blr_tol": 1e-6}' blr 2>&1 | grep -E 'BENCH_RESULT|1st solution|NaN|Traceback|Error' | head -5
echo "=== MODIFIED sweep mode (anchor LU + FGMRES) ==="
SCATT3D_SRC=/work/Scatt3D timeout 1200 python3 bench_driver.py $H $DEG $NF '{"sweep_mode": true}' sweep 2>&1 | grep -E 'BENCH_RESULT|1st solution|Sweep mode|NaN|Traceback|Error' | head -8
echo "=== S-PARAMETER EQUIVALENCE vs STOCK ==="
python3 - <<'EOF'
import numpy as np
ref = np.load('S_stock.npz')['S_ref']
for tag in ['fast', 'blr', 'sweep']:
    try:
        S = np.load(f'S_{tag}.npz')['S_ref']
        denom = np.max(np.abs(ref))
        print(f"{tag}: max|dS| = {np.max(np.abs(S-ref)):.3e}  (rel to max|S|={denom:.3e}: {np.max(np.abs(S-ref))/denom:.3e})")
    except FileNotFoundError:
        print(f"{tag}: MISSING")
EOF
echo HEADTOHEAD_DONE
