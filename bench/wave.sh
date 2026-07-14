#!/bin/bash
# Wave-2 fast items: (a) log_view phase split / symbolic-reuse check,
# (b) matrix dump + MatMatSolve batching probe, (c) basix condensable-DOF count,
# (d) PML-thickness economy sweep with Mie accuracy gate.
exec > /work/bench/wave.log 2>&1
source /usr/local/bin/dolfinx-complex-mode 2>/dev/null
cd /work/bench
mkdir -p wave_work && cd wave_work
cp ../bench_driver.py .

echo "=== (a) LOGVIEW sym h=0.42 Nf=6 np=8 ($(date -u +%H:%M:%SZ)) ==="
PETSC_OPTIONS="-log_view :/work/bench/wave_work/logview_sym.txt" \
  SCATT3D_SRC=/work/Scatt3D timeout 7200 mpirun -np 8 python3 bench_driver.py 0.42 3 6 '{"symmetric": true}' PHASES > case_phases.log 2>&1
echo "EXIT $?"
grep -E 'BENCH_RESULT|MUMPS factor memory' case_phases.log | head -3
grep -E 'MatCholFctrSym|MatCholFctrNum|MatSolve |MatAssemblyEnd|SNESSolve|KSPSolve ' /work/bench/wave_work/logview_sym.txt | head -8

echo "=== (b) matrix dump for MatMatSolve probe ($(date -u +%H:%M:%SZ)) ==="
mkdir -p /work/bench/cableport
if [ ! -f /work/bench/cableport/A545k.bin ]; then
  PETSC_OPTIONS="-theScatteringProblem_ksp_view_mat binary:/work/bench/cableport/A545k.bin" \
    SCATT3D_SRC=/work/Scatt3D timeout 3600 python3 ../cableport_validate.py 3 '{}' dump 0.2857142857142857 3 1 > case_dump.log 2>&1
  echo "dump EXIT $?"
fi
ls -la /work/bench/cableport/
echo "=== (b) MATMAT probe ($(date -u +%H:%M:%SZ)) ==="
timeout 3600 python3 ../matmat_bench.py > case_matmat.log 2>&1
echo "EXIT $?"
grep MATMAT case_matmat.log

echo "=== (c) basix deg-3 N1curl condensable-DOF count ==="
python3 - <<'PY'
import basix
import numpy as np
import dolfinx.mesh as dm
from mpi4py import MPI
e = basix.create_element(basix.ElementFamily.N1E, basix.CellType.tetrahedron, 3)
ned = e.num_entity_dofs
per_edge = ned[1][0]; per_face = ned[2][0]; per_cell = ned[3][0]
print("BASIX per-entity dofs deg3 N1E: edge", per_edge, "face", per_face, "cell-interior", per_cell)
msh = dm.create_unit_cube(MPI.COMM_WORLD, 20, 20, 20, dm.CellType.tetrahedron)
msh.topology.create_entities(1); msh.topology.create_entities(2)
nE = msh.topology.index_map(1).size_global; nF = msh.topology.index_map(2).size_global
nC = msh.topology.index_map(3).size_global
tot = nE*per_edge + nF*per_face + nC*per_cell
print(f"BASIX mesh E={nE} F={nF} C={nC} -> dofs total={tot}, "
      f"cell-interior {nC*per_cell} ({100*nC*per_cell/tot:.1f}%), "
      f"face {nF*per_face} ({100*nF*per_face/tot:.1f}%), edge {nE*per_edge} ({100*nE*per_edge/tot:.1f}%)")
PY

echo "=== (d) PML thickness sweep, Mie gate ($(date -u +%H:%M:%SZ)) ==="
for T in 0.5 0.35 0.25 0.15 0.1; do
  SCATT3D_SRC=/work/Scatt3D timeout 7200 mpirun -np 8 python3 ../pml_bench.py $T > case_pml$T.log 2>&1
  echo "PML $T EXIT $?"
  grep -E 'PML_RESULT|Traceback' case_pml$T.log | head -3
  rm -rf data3D/pml*
done
echo WAVE_DONE
