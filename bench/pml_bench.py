# PML-thickness economy probe: friend's testSphereScattering geometry (Mie ground
# truth), deg 2, sweep PML_thickness; report ndofs, factor memory, far-field error.
# Usage: python3 pml_bench.py <PML_thickness>
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import sys
import numpy as np
import dolfinx.fem.petsc
from mpi4py import MPI
from timeit import default_timer as timer

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
import scatteringProblem

comm = MPI.COMM_WORLD
T = float(sys.argv[1])
h = 1 / 12
deg = 2

os.makedirs("data3D", exist_ok=True)
t0 = timer()
refMesh = meshMaker.MeshInfo(
    comm, reference=True, viewGMSH=False, verbosity=2, N_antennas=0,
    object_radius=.33, domain_radius=.9, PML_thickness=T, h=h,
    domain_geom="sphere", object_geom="sphere", FF_surface=True, order=deg)
freqs = np.linspace(10e9, 12e9, 1)
prob = scatteringProblem.Scatt3DProblem(
    comm, refMesh, verbosity=2, name=f"pml{T}", dataFolder="data3D/", MPInum=comm.size,
    makeOptVects=True, excitation="planewave", freqs=freqs,
    material_epsrs=[2.0 * (1 - 0.01j)], fem_degree=deg,
    solver_settings={"symmetric": True})
t1 = timer()
# his own Mie-comparison path prints 'Forward-scattering intensity relative error: ...'
prob.calcFarField(reference=True, compareToMie=True, showPlots=False,
                  returnConvergenceVals=False)
nd = getattr(prob, "ndofs", None) or getattr(getattr(prob, "FEMmesh_ref", None), "ndofs", "?")
fmem = getattr(prob, "lastFactorMemMB", None)
if comm.rank == 0:
    print(f"PML_RESULT T={T} ndofs={nd} solve_s={t1-t0:.1f} "
          f"mumps_mem_mb={list(fmem) if fmem else None}")
