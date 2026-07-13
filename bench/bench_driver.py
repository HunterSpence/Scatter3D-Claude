# Benchmark driver for Scatt3D solver experiments.
# Usage (in container): SCATT3D_SRC=/work/<tree>/Scatt3D python3 bench_driver.py [h] [degree] [Nf] [solver_json] [tag]
# Replicates runScatt3D.testRun geometry (sphere object, 3 antennas) without plots.
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import json, sys
import numpy as np
import dolfinx.fem.petsc  # scatteringProblem assumes caller imported this
from mpi4py import MPI
from timeit import default_timer as timer
import resource

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
import scatteringProblem

comm = MPI.COMM_WORLD
h = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
deg = int(sys.argv[2]) if len(sys.argv) > 2 else 1
Nf = int(sys.argv[3]) if len(sys.argv) > 3 else 1
solver_settings = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
tag = sys.argv[5] if len(sys.argv) > 5 else "run"
extra = {}
if len(sys.argv) > 6:  # optional dense frequency band "f1:f2"
    f1, f2 = (float(x) for x in sys.argv[6].split(":"))
    extra["freqs"] = np.linspace(f1, f2, Nf)

os.makedirs("data3D", exist_ok=True)
t0 = timer()
refMesh = meshMaker.MeshInfo(
    comm, "data3D/benchmesh.msh", reference=True, viewGMSH=False, verbosity=2,
    h=h, object_geom="sphere", domain_radius=0.8, domain_height=0.46,
    dome_height=0.22, PML_thickness=0.1, antenna_bounding_box_offset=0.05,
    object_radius=0.2, N_antennas=3, order=deg)
t1 = timer()
prob = scatteringProblem.Scatt3DProblem(
    comm, refMesh, verbosity=2, MPInum=comm.size, name="bench" + tag,
    dataFolder="data3D/", computeBoth=False, Nf=Nf, fem_degree=deg,
    E_ref_anim=False, makeOptVects=False, solver_settings=solver_settings, **extra)
t2 = timer()
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
## aggregate RSS across ranks: the sum is the operational "machine memory" number —
## a single rank's RSS understates MPI runs
mem_sum = comm.allreduce(mem, op=MPI.SUM)
mem_max = comm.allreduce(mem, op=MPI.MAX)
ndofs = getattr(prob, "ndofs", None) or getattr(getattr(prob, "FEMmesh_ref", None), "ndofs", "?")
if comm.rank == 0:
    np.savez(f"S_{tag}.npz", S_ref=prob.S_ref, fvec=prob.fvec)
    fmem = getattr(prob, "lastFactorMemMB", None)
    print(f"BENCH_RESULT tag={tag} h={h} deg={deg} Nf={Nf} ndofs={ndofs} "
          f"mesh_s={t1-t0:.1f} solve_s={t2-t1:.1f} maxrss_gb={mem:.2f} "
          f"rss_sum_gb={mem_sum:.2f} rss_max_gb={mem_max:.2f} "
          f"mumps_mem_mb={list(fmem) if fmem else None} "
          f"solver={json.dumps(solver_settings)}")
