# Cable-port solver validation: coax transmission vs analytic transmission-line theory.
# Improved version of upstream runScatt3D2.py cablePortTest/cablePortRMSError:
#   - one (coax, solver_config) per PROCESS so a solver crash cannot kill the whole matrix
#   - compares every solver config against BOTH theory and the LU baseline (max |dS|)
#   - records MUMPS factor memory (INFOG 16/17) so LU vs LDLT memory shows up at toy scale
#   - machine-readable JSON line output; aggregation + PASS/FAIL verdict in run_cableport.sh
# Usage (in container, complex mode):
#   SCATT3D_SRC=/work/Scatt3D python3 cableport_validate.py <coax_idx> '<solver_json>' [tag] [h] [degree] [nf]
import os
os.environ["OMP_NUM_THREADS"] = "1"
import matplotlib
matplotlib.use("Agg")
import json, sys
import numpy as np
import dolfinx.fem.petsc  # scatteringProblem assumes the driver imported this
from mpi4py import MPI
from scipy.constants import c as c0, mu_0 as mu0, epsilon_0 as eps0, pi
from timeit import default_timer as timer
import resource

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
import scatteringProblem

eta0 = np.sqrt(mu0 / eps0)

# upstream cablePortRMSError coax set, verbatim: [epsr1, epsr2, d, L]
COAXS = [
    [2.1 * (1 - 0.01j), 2.1 * (1 - 1j), 1e-3, 1e-3],
    [5.1 * (1 - 0.001j), 8.1 * (1 - 0.01j), 4e-3, 2e-3],
    [2.7 * (1 - 0.01j), 2.1 * (1 - 0.01j), 3e-3, 1e-3],
    [2.1 * (1 - 0.01j), 12.1 * (1 - 0.01j), 1e-3, 8e-3],
    [2.1 * (1 - 0.01j), 23.1 * (1 - 0.04j), 6e-3, 3e-3],
    [1.3 * (1 - 0.01j), 2.1 * (1 - 0.8j), 0.5e-3, 0.4e-3],
    [2.1 * (1 - 0.01j), 8.1 * (1 - 0.01j), 1.1e-3, 0.1e-3],
    [2.1 * (1 - 0.01j), 12.1 * (1 - 0.1j), 3e-3, 1e-3],
    [2.1 * (1 - 0.1j), 2.1 * (1 - 0.01j), 1e-3, 10e-3],
    [75 * (1 - 0.01j), 76.1 * (1 - 0.05j), 8e-3, 2e-3],
]

comm = MPI.COMM_WORLD
coax_idx = int(sys.argv[1])
solver_settings = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
tag = sys.argv[3] if len(sys.argv) > 3 else "run"
h = float(sys.argv[4]) if len(sys.argv) > 4 else 1 / 3.5
degree = int(sys.argv[5]) if len(sys.argv) > 5 else 3
nf = int(sys.argv[6]) if len(sys.argv) > 6 else 3

epsr1, epsr2, d, L = COAXS[coax_idx]
freqs = np.linspace(5.4e9, 7.2e9, nf)
runName = f"cableport_c{coax_idx}_{tag}"

os.makedirs("data3D", exist_ok=True)
t0 = timer()
# mesh/problem settings copied from upstream cablePortTest (runScatt3D2.py:216)
refMesh = meshMaker.MeshInfo(
    comm, f"data3D/{runName}mesh.msh", viewGMSH=False, reference=True, verbosity=2,
    h=h, N_antennas=1, order=degree, antenna_type="coaxTest", object_geom=None,
    domain_geom=None, object_height=L, defect_height=d, antenna_epsrs=[epsr1, epsr2])
t1 = timer()
prob = scatteringProblem.Scatt3DProblem(
    comm, refMesh, MPInum=comm.size, name=runName, fem_degree=degree, verbosity=2,
    freqs=freqs, antenna_mat_epsrs=[epsr1, epsr2], dataFolder="data3D/",
    computeBoth=False, makeOptVects=False, E_ref_anim=False,
    solver_settings=solver_settings)
t2 = timer()

k1 = 2 * pi / c0 * freqs * np.sqrt(epsr1)
k2 = 2 * pi / c0 * freqs * np.sqrt(epsr2)
Z1 = eta0 / np.sqrt(epsr1) / (2 * pi) * np.log(refMesh.coax_outr / refMesh.coax_inr)
Z2 = eta0 / np.sqrt(epsr2) / (2 * pi) * np.log(refMesh.coax_outr / refMesh.coax_inr)
theory = ((-1 + Z2 / Z1 * 1j * np.tan(k2 * d)) / (1 + Z2 / Z1 * 1j * np.tan(k2 * d))) * np.exp(-2j * k1 * L)
sim = np.asarray(prob.S_ref).flatten()

if comm.rank == 0:
    mem_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    fmem = getattr(prob, "lastFactorMemMB", None)
    out = {
        "coax": coax_idx, "solver": solver_settings, "tag": tag,
        "h": h, "degree": degree, "nf": nf,
        "S_sim": [[float(s.real), float(s.imag)] for s in sim],
        "S_theory": [[float(s.real), float(s.imag)] for s in theory],
        "mag_relerr": np.abs(np.abs(sim) - np.abs(theory)) / np.abs(theory),
        "phase_err_deg": np.abs(np.unwrap(np.angle(sim)) - np.unwrap(np.angle(theory))) * 180 / pi,
        "factor_mem_mb": list(fmem) if fmem is not None else None,
        "mesh_s": t1 - t0, "solve_s": t2 - t1, "maxrss_gb": mem_gb,
    }
    print("CABLEPORT_RESULT " + json.dumps(out, default=lambda a: np.asarray(a).tolist()))
