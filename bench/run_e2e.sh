#!/bin/bash
# Stock legs compare against an unmodified upstream tree: from the mounted repo root,
#   git clone https://github.com/Wojoxiw/Scatt3D upstream-Scatt3D
# (giving /work/upstream-Scatt3D/Scatt3D); the modified tree is this repo at /work/Scatt3D.
# End-to-end imaging verification: stock vs branch (direct/BLR/sweep). Self-logging.
# Env knobs: E2E_H (mesh h, default 0.25 = lambda/4), E2E_NF (freqs, default 2),
#            E2E_TIMEOUT (seconds per case, default 4500).
# NOTE: upstream readSol/create_interpolation_data makes the sensitivity build VERY slow
# single-node on fine meshes (see SOLVER_IMPROVEMENTS.md) - use coarse E2E_H and a long
# E2E_TIMEOUT on a box that can run unattended.
exec > /work/bench/e2e_container.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
H=${E2E_H:-0.25}
NF=${E2E_NF:-2}
T=${E2E_TIMEOUT:-4500}
cd /work/bench
rm -rf e2e_work && mkdir e2e_work && cd e2e_work

run_case() {
  local tag=$1 solver=$2 src=$3
  mkdir -p case_$tag && cd case_$tag
  cp ../../e2e_driver.py .
  SCATT3D_SRC=$src timeout $T python3 e2e_driver.py $tag "$solver" $H $NF 2>&1 | grep -E 'E2E_RESULT|re-anchoring|Traceback|NaN|error code' | head -8
  cd ..
}
echo "=== wave 1: stock + branch (parallel), H=$H NF=$NF timeout=$T ==="
( run_case stock '{}' /work/upstream-Scatt3D/Scatt3D ) &
( run_case branch '{}' /work/Scatt3D ) &
wait
echo "=== wave 2: blr + sweep (parallel) ==="
( run_case blr '{"blr_tol": 1e-6}' /work/Scatt3D ) &
( run_case sweep '{"sweep_mode": true}' /work/Scatt3D ) &
wait
echo "=== E2E COMPARISON ==="
python3 - <<'PYEOF'
import numpy as np
ref = np.load("case_stock/e2e_stock_S.npz")
for tag in ["branch", "blr", "sweep"]:
    try:
        d = np.load(f"case_{tag}/e2e_{tag}_S.npz")
        dr = np.max(np.abs(d["S_ref"] - ref["S_ref"]))
        dd = np.max(np.abs(d["S_dut"] - ref["S_dut"]))
        print(f"{tag}: max|dS_ref|={dr:.3e} max|dS_dut|={dd:.3e}")
    except Exception as e:
        print(tag, "compare failed:", e)
PYEOF
echo E2E_DONE
