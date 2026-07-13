#!/usr/bin/env python3
# Spike round 2: multiplicative two-level (Jacobi pre/post + p-coarse correction),
# judged on TRUE residual. Reuses the dumped A3/A1 and the probe-mesh spaces.
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import dolfinx, dolfinx.fem.petsc
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
from scatteringProblem import FEMmesh

comm = MPI.COMM_WORLD
epsr1, epsr2, d, L = [2.1 * (1 - 0.01j), 12.1 * (1 - 0.01j), 1e-3, 8e-3]
refMesh = meshMaker.MeshInfo(
    comm, "data3D/spike_probe2mesh.msh", viewGMSH=False, reference=True, verbosity=2,
    h=1/3.5, N_antennas=1, order=1, antenna_type="coaxTest", object_geom=None,
    domain_geom=None, object_height=L, defect_height=d, antenna_epsrs=[epsr1, epsr2])
fm3, fm1 = FEMmesh(refMesh, 3, 6), FEMmesh(refMesh, 1, 6)
Pp = dolfinx.fem.petsc.interpolation_matrix(fm1.VSpace, fm3.VSpace); Pp.assemble()

def csr(M):
    i, j, v = M.getValuesCSR(); return sp.csr_matrix((v, j, i), shape=M.getSize())
def load(p):
    return csr(PETSc.Mat().load(PETSc.Viewer().createBinary(p, "r")))

P = csr(Pp); Pt = P.T.tocsr()
A3, A1 = load("A_deg3.bin"), load("A_deg1.bin")
A1lu = spla.splu(A1.tocsc())
diag3 = A3.diagonal()

def make_mult(omega, nsm):
    def apply(r):
        z = np.zeros_like(r)
        for _ in range(nsm):                       # pre-smooth (damped Jacobi)
            z += omega * (r - A3 @ z) / diag3
        z += P @ A1lu.solve(Pt @ (r - A3 @ z))     # coarse correction
        for _ in range(nsm):                       # post-smooth
            z += omega * (r - A3 @ z) / diag3
        return z
    return apply

rng = np.random.default_rng(7)
b = rng.standard_normal(A3.shape[0]) + 1j * rng.standard_normal(A3.shape[0])
bn = np.linalg.norm(b)
for tag, omega, nsm in [("mult w0.5 s1", 0.5, 1), ("mult w0.2 s2", 0.2, 2)]:
    M = spla.LinearOperator(A3.shape, matvec=make_mult(omega, nsm), dtype=complex)
    t0 = time.time()
    x, info = spla.gmres(A3, b, M=M, rtol=1e-8, atol=0.0, restart=100, maxiter=1)
    relres = np.linalg.norm(b - A3 @ x) / bn
    print(f"{tag}: true relres after 100 its = {relres:.2e}  ({time.time()-t0:.0f}s)  "
          + ("CONVERGED" if relres < 1e-7 else "STALLED"))
print("SPIKE2_DONE")
