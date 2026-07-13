#!/usr/bin/env python3
# p-coarse feasibility spike, step 1: dump the assembled operator at one frequency.
# Usage: python3 spike_dump.py <fem_degree> <out.bin>
# Same coax-3 case as the certification (h=1/3.5), but geometric mesh order FIXED at 1
# so the deg-3 and deg-1 FE spaces live on the *identical* mesh.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import sys
import numpy as np

degree = int(sys.argv[1])
outfile = sys.argv[2]
# dump the system matrix at the first KSP solve
os.environ["PETSC_OPTIONS"] = f"-theScatteringProblem_ksp_view_mat binary:{outfile}"

import dolfinx.fem.petsc  # noqa: F401  (scatteringProblem assumes driver imported this)
from mpi4py import MPI
from scipy.constants import pi

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
import scatteringProblem

comm = MPI.COMM_WORLD
epsr1, epsr2, d, L = [2.1 * (1 - 0.01j), 12.1 * (1 - 0.01j), 1e-3, 8e-3]  # coax 3
h = 1 / 3.5
freqs = np.linspace(5.4e9, 7.2e9, 1)
runName = f"spike_deg{degree}"

os.makedirs("data3D", exist_ok=True)
refMesh = meshMaker.MeshInfo(
    comm, f"data3D/{runName}mesh.msh", viewGMSH=False, reference=True, verbosity=2,
    h=h, N_antennas=1, order=1, antenna_type="coaxTest", object_geom=None,
    domain_geom=None, object_height=L, defect_height=d, antenna_epsrs=[epsr1, epsr2])

prob = scatteringProblem.Scatt3DProblem(
    comm, refMesh, MPInum=comm.size, name=runName, fem_degree=degree, verbosity=2,
    freqs=freqs, antenna_mat_epsrs=[epsr1, epsr2], dataFolder="data3D/",
    computeBoth=False, makeOptVects=False, E_ref_anim=False,
    solver_settings={"symmetric": True})
print(f"SPIKE_DUMP_OK degree={degree} ndofs={prob.FEMmesh_ref.ndofs} file={outfile}")
