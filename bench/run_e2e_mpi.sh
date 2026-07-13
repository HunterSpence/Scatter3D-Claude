#!/bin/bash
# Stock legs compare against an unmodified upstream tree: from the mounted repo root,
#   git clone https://github.com/Wojoxiw/Scatt3D upstream-Scatt3D
# (giving /work/upstream-Scatt3D/Scatt3D); the modified tree is this repo at /work/Scatt3D.
# End-to-end imaging verification with MPI sim phase + single-rank postprocessing.
exec > /work/bench/e2e_mpi.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
RANKS=${E2E_RANKS:-8}
H=${E2E_H:-0.25}
NF=${E2E_NF:-2}
cd /work/bench
rm -rf e2e_work && mkdir e2e_work && cd e2e_work

run_case() {
  local tag=$1 solver=$2 src=$3
  mkdir -p case_$tag && cd case_$tag
  cp ../../e2e_driver.py .
  echo "--- $tag sim phase ($(date -u +%H:%M:%SZ)) ---"
  SCATT3D_SRC=$src timeout 21600 mpirun -np $RANKS \
    python3 e2e_driver.py $tag "$solver" $H $NF sim 2>&1 | grep -E 're-anchoring|Traceback|NaN|error code' | head -6
  echo "--- $tag post phase ($(date -u +%H:%M:%SZ)) ---"
  SCATT3D_SRC=$src timeout 7200 python3 e2e_driver.py $tag "$solver" $H $NF post 2>&1 | grep -E 'E2E_RESULT|Traceback|error' | head -4
  cd ..
}
run_case stock '{}' /work/upstream-Scatt3D/Scatt3D
run_case branch '{}' /work/Scatt3D
run_case blr '{"blr_tol": 1e-6}' /work/Scatt3D
run_case sym '{"symmetric": true}' /work/Scatt3D
run_case sweep '{"sweep_mode": true}' /work/Scatt3D
echo "=== E2E COMPARISON ==="
python3 - <<'EOF'
import numpy as np
ref = np.load("case_stock/e2e_stock_S.npz")
for tag in ["branch", "blr", "sym", "sweep"]:
    try:
        d = np.load(f"case_{tag}/e2e_{tag}_S.npz")
        print(f"{tag}: max|dS_ref|={np.max(np.abs(d['S_ref']-ref['S_ref'])):.3e} "
              f"max|dS_dut|={np.max(np.abs(d['S_dut']-ref['S_dut'])):.3e}")
    except Exception as e:
        print(tag, "compare failed:", e)
EOF
echo E2E_MPI_DONE
