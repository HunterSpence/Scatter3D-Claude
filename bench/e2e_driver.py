# End-to-end imaging verification driver: full pipeline (mesh -> FEM solves ->
# sensitivity q-vectors -> TSVD reconstruction) with a quantitative error metric.
# Usage: SCATT3D_SRC=<tree> python3 e2e_driver.py <tag> [solver_json] [h] [Nf]
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import json, sys
import numpy as np
import dolfinx.fem.petsc  # scatteringProblem assumes caller imported this
from mpi4py import MPI

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/upstream-Scatt3D/Scatt3D"))
import meshMaker
import scatteringProblem
import postProcessing

comm = MPI.COMM_WORLD
tag = sys.argv[1]
solver = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
h = float(sys.argv[3]) if len(sys.argv) > 3 else 1 / 6
Nf = int(sys.argv[4]) if len(sys.argv) > 4 else 2

folder = "data3D/"
runName = "e2e_" + tag
os.makedirs(folder, exist_ok=True)
# testFullExample geometry (waveguide antennas, cylinder object + defect), small h
mesh_settings = {"h": h, "N_antennas": 4, "order": 1,
                 "object_offset": np.array([.15, .1, 0]), "viewGMSH": False,
                 "defect_offset": np.array([-.04, .17, .01]), "defect_radius": 0.175,
                 "defect_height": 0.3, "antenna_type": "waveguide"}
refMesh = meshMaker.MeshInfo(comm, folder + runName + "mesh.msh", reference=False,
                             verbosity=2, **mesh_settings)
prob = scatteringProblem.Scatt3DProblem(
    comm, refMesh, MPInum=comm.size, name=runName, fem_degree=1, verbosity=2,
    Nf=Nf, dataFolder=folder, computeBoth=True, E_ref_anim=False,
    makeOptVects=True, solver_settings=solver)
if comm.rank == 0:
    np.savez(f"e2e_{tag}_S.npz", S_ref=prob.S_ref, S_dut=prob.S_dut, fvec=prob.fvec)
errs = postProcessing.solveFromQs(folder + runName, solutionName="",
                                  onlyAPriori=True, plotSs=False, returnResults=[3])
print(f"E2E_RESULT tag={tag} reconstruction_err={errs[0]:.12e} solver={json.dumps(solver)}")
