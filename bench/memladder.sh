#!/bin/bash
# Memory-vs-DOF dispute ladder (2026-07-14).
# Settles: (a) does LU factor memory vs DOFs follow a line on ONE geometry at fixed ranks,
# (b) do more ranks cost more memory (per metric), (c) coax-vs-sphere per-DOF constant.
# Metrics per case: MUMPS INFOG(16/17) estimates + INFOG(21/22) effective (the doc's metric)
# AND whole-process RSS sum over ranks (the friend's memTimeEstimation.py metric).
exec > /work/bench/memladder.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
cd /work/bench
mkdir -p memladder_work && cd memladder_work
cp ../bench_driver.py .

run() { # tag np h solver_json
  local tag=$1 np=$2 h=$3 js=$4
  echo "=== $tag np=$np h=$h ($(date -u +%H:%M:%SZ)) ==="
  if [ "$np" = 1 ]; then
    SCATT3D_SRC=/work/Scatt3D timeout 14400 python3 bench_driver.py "$h" 3 1 "$js" "$tag" > "case_$tag.log" 2>&1
  else
    SCATT3D_SRC=/work/Scatt3D timeout 14400 mpirun -np "$np" python3 bench_driver.py "$h" 3 1 "$js" "$tag" > "case_$tag.log" 2>&1
  fi
  echo "EXIT $? tag=$tag"
  grep -E 'BENCH_RESULT|MUMPS factor memory|Killed|out of memory|INFOG\(1\)|Traceback' "case_$tag.log" | head -6
  rm -rf data3D/bench${tag}*
}

# 1) sphere LU + sym lines, fixed np=8 (same geometry family the 2.8M cert used)
for H in 0.6 0.5 0.42 0.35 0.31 0.28; do
  run "LUh${H}n8" 8 "$H" '{}'
  run "SYMh${H}n8" 8 "$H" '{"symmetric": true}'
done

# 2) rank A/B at h=0.5 (np=8 already covered by the ladder)
for NP in 1 4 16; do
  run "LUh0.5n${NP}" "$NP" 0.5 '{}'
done

# 3) sphere at ~545k dofs, single rank — direct comparison to the doc's coax 545k point
run "LUh0.48n1" 1 0.48 '{}'

# 4) coax idx3 h=1/3.5 deg3 (the doc's 545k coax), single rank, LU
echo "=== COAX3 lu np=1 h=1/3.5 ($(date -u +%H:%M:%SZ)) ==="
SCATT3D_SRC=/work/Scatt3D timeout 14400 python3 ../cableport_validate.py 3 '{}' memlu 0.2857142857142857 3 1 > case_coax3lu.log 2>&1
echo "EXIT $? tag=coax3lu"
grep -E 'CABLEPORT_RESULT|MUMPS factor memory|ndofs|Traceback' case_coax3lu.log | head -6

echo MEMLADDER_DONE
