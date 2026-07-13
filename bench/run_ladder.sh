#!/bin/bash
# Memory-target ladder (friend's acceptance test): degree-3, increasing DOFs,
# LU vs symmetric LDL^T vs symmetric+BLR. Metric: MUMPS factor memory (INFOG 16/17)
# + accuracy vs LU at each size. Runs under MPI. Self-logging.
exec > /work/bench/ladder.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
RANKS=${LADDER_RANKS:-8}
cd /work/bench
rm -rf ladder_work && mkdir ladder_work && cd ladder_work
cp ../bench_driver.py .
for H in 0.5 0.35 0.28 0.22; do
  for CFG in 'lu|{}' 'sym|{"symmetric": true}' 'symblr|{"symmetric": true, "blr_tol": 1e-6}'; do
    T="${CFG%%|*}"; S="${CFG#*|}"
    echo "=== h=$H deg=3 cfg=$T ($(date -u +%H:%M:%SZ)) ==="
    SCATT3D_SRC=/work/Scatt3D timeout 10800 mpirun -np $RANKS \
      python3 bench_driver.py $H 3 1 "$S" "L${T}h${H}" 2>&1 \
      | grep -E 'BENCH_RESULT|MUMPS factor|1st solution|NaN|Traceback|error code|Killed|out of memory|INFOG\(1\)' | head -8
    rm -rf data3D/benchL* ## free the solution files, keep the mesh cache out of the way
  done
  python3 - <<PY
import numpy as np
try:
    a = np.load("S_Lluh$H.npz")["S_ref"]
    for t in ["sym", "symblr"]:
        try:
            b = np.load(f"S_L{t}h$H.npz")["S_ref"]
            print(f"ACCURACY h=$H {t} vs lu: max|dS|={np.max(np.abs(a-b)):.3e}")
        except Exception as e:
            print(f"ACCURACY h=$H {t}: no data ({e})")
except Exception as e:
    print(f"ACCURACY h=$H: no LU baseline ({e})")
PY
done
echo LADDER_DONE
