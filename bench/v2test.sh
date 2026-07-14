#!/bin/bash
# v2 validation battery: smoke (899k) -> cableport regression -> full workload
# battery at 2.26M/Nf=22 (BEFORE stock-LU vs batching vs final stack vs fp32-max
# vs ICNTL(27) rider) -> S-parameter cross-checks.
exec > /work/bench/v2test.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
cd /work/bench
mkdir -p v2_work && cd v2_work
cp ../bench_driver.py .

run() { # tag np h Nf solver_json
  local tag=$1 np=$2 h=$3 nf=$4 js=$5
  echo "=== $tag np=$np h=$h Nf=$nf ($(date -u +%H:%M:%SZ)) ==="
  SCATT3D_SRC=/work/Scatt3D timeout 14400 mpirun -np "$np" python3 bench_driver.py "$h" 3 "$nf" "$js" "$tag" > "case_$tag.log" 2>&1
  echo "EXIT $? tag=$tag"
  grep -E 'BENCH_RESULT|MUMPS factor memory|Adaptive sweep: solved|batch_rhs requested|Traceback|NaN' "case_$tag.log" | head -6
  rm -rf data3D/bench${tag}*
}

# ── smoke at 899k ──
run SMK_sym      8 0.5 22 '{"symmetric": true}'
run SMK_symbatch 8 0.5 22 '{"symmetric": true, "batch_rhs": true}'
run SMK_final    8 0.5 22 '{"symmetric": true, "batch_rhs": true, "adaptive_sweep": 1e-4}'
python3 - <<'PY'
import numpy as np
a = np.load("S_SMK_sym.npz")["S_ref"]
for t in ("SMK_symbatch", "SMK_final"):
    b = np.load(f"S_{t}.npz")["S_ref"]
    print(f"SMOKE_ACCURACY {t} vs sym: max|dS|={np.max(np.abs(a-b)):.3e}")
PY

# ── cableport regression (untouched single-antenna path) ──
echo "=== CABLEPORT regression ($(date -u +%H:%M:%SZ)) ==="
cd /work/bench && CP_COAXS="3" bash run_cableport.sh > /work/bench/v2_work/cableport_v2.log 2>&1
grep -E 'VERDICT|coax' /work/bench/v2_work/cableport_v2.log | tail -6
cd /work/bench/v2_work

# ── full battery at 2.26M / Nf=22 (his workload shape) ──
run WL_LU        8 0.31 22 '{}'
run WL_symbatch  8 0.31 22 '{"symmetric": true, "batch_rhs": true}'
run WL_final     8 0.31 22 '{"symmetric": true, "batch_rhs": true, "adaptive_sweep": 1e-4}'
run WL_fp32max   8 0.31 22 '{"symmetric": true, "mat_mumps_icntl_7": 3, "blr_tol": 1e-6, "mat_mumps_icntl_37": 1, "pc_precision": "single", "ksp_type": "fgmres", "ksp_rtol": 1e-12, "ksp_max_it": 100, "adaptive_sweep": 1e-4}'
run WL_icntl27   8 0.31 22 '{"symmetric": true, "batch_rhs": true, "mat_mumps_icntl_27": 64}'

python3 - <<'PY'
import numpy as np
ref = np.load("S_WL_LU.npz")["S_ref"]
for t in ("WL_symbatch", "WL_final", "WL_fp32max", "WL_icntl27"):
    try:
        b = np.load(f"S_{t}.npz")["S_ref"]
        print(f"WL_ACCURACY {t} vs LU-full: max|dS|={np.max(np.abs(ref-b)):.3e}")
    except Exception as e:
        print(f"WL_ACCURACY {t}: {e}")
PY
echo V2TEST_DONE
