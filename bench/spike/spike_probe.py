#!/usr/bin/env python3
# p-coarse feasibility spike, step 2: two-level convergence probe.
# Loads A_deg3.bin / A_deg1.bin (same mesh, geometric order 1), builds the N1curl
# deg1->deg3 interpolation P on an identically regenerated mesh, then runs FGMRES on
# A3 with several preconditioners and reports iteration counts.
#
# Question answered: does a rediscretized p-coarse space (deg-1 operator, direct-solved)
# preconditioning the deg-3 operator CONVERGE on this indefinite curl-curl+PML system?
# Known caveat (spike-level): P ignores PEC bc rows; A1/A3 both carry their own bcs.
import os
os.environ["OMP_NUM_THREADS"] = "4"
import sys
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import dolfinx
import dolfinx.fem.petsc
import basix.ufl
from mpi4py import MPI
from petsc4py import PETSc

sys.path.insert(0, os.environ.get("SCATT3D_SRC", "/work/Scatt3D"))
import meshMaker
from scatteringProblem import FEMmesh

comm = MPI.COMM_WORLD
epsr1, epsr2, d, L = [2.1 * (1 - 0.01j), 12.1 * (1 - 0.01j), 1e-3, 8e-3]
h = 1 / 3.5

refMesh = meshMaker.MeshInfo(
    comm, "data3D/spike_probemesh.msh", viewGMSH=False, reference=True, verbosity=2,
    h=h, N_antennas=1, order=1, antenna_type="coaxTest", object_geom=None,
    domain_geom=None, object_height=L, defect_height=d, antenna_epsrs=[epsr1, epsr2])

fm3 = FEMmesh(refMesh, 3, 6)
fm1 = FEMmesh(refMesh, 1, 6)
n3, n1 = fm3.ndofs, fm1.ndofs
print(f"spaces: deg3 {n3} dofs, deg1 {n1} dofs")

t0 = time.time()
Pp = dolfinx.fem.petsc.interpolation_matrix(fm1.VSpace, fm3.VSpace)
Pp.assemble()
print(f"P built in {time.time()-t0:.1f}s, shape {Pp.getSize()}")

def petsc_to_csr(M):
    i, j, v = M.getValuesCSR()
    return sp.csr_matrix((v, j, i), shape=M.getSize())

def load_bin(path):
    viewer = PETSc.Viewer().createBinary(path, "r")
    M = PETSc.Mat().load(viewer)
    return petsc_to_csr(M)

P = petsc_to_csr(Pp)
A3 = load_bin(sys.argv[1] if len(sys.argv) > 1 else "A_deg3.bin")
A1 = load_bin(sys.argv[2] if len(sys.argv) > 2 else "A_deg1.bin")
assert A3.shape[0] == n3 and A1.shape[0] == n1, (A3.shape, A1.shape, n3, n1)
print(f"A3 {A3.shape} nnz {A3.nnz}; A1 {A1.shape} nnz {A1.nnz}")

t0 = time.time()
A1lu = spla.splu(A1.tocsc())
print(f"coarse LU in {time.time()-t0:.1f}s")

Pt = P.T.tocsr()          # plain transpose (complex-symmetric system)
diag3 = A3.diagonal()
assert np.all(np.abs(diag3) > 0)

def coarse(r):
    return P @ A1lu.solve(Pt @ r)

PCS = {
    "none":         None,
    "jacobi":       lambda r: r / diag3,
    "coarse":       coarse,
    "coarse+jac":   lambda r: coarse(r) + 0.5 * r / diag3,
}

rng = np.random.default_rng(7)
b = rng.standard_normal(n3) + 1j * rng.standard_normal(n3)
bn = np.linalg.norm(b)
MAXIT = 150

print(f"\n{'PC':>12} {'its':>5} {'relres':>10} {'time_s':>7}  verdict")
for name, apply in PCS.items():
    M = spla.LinearOperator(A3.shape, matvec=apply, dtype=complex) if apply else None
    resids = []
    t0 = time.time()
    x, info = spla.gmres(A3, b, M=M, rtol=1e-8, atol=0.0, restart=MAXIT, maxiter=1,
                         callback=lambda pr: resids.append(pr), callback_type="pr_norm")
    dt = time.time() - t0
    relres = np.linalg.norm(b - A3 @ x) / bn
    its = len(resids)
    verdict = "CONVERGED" if info == 0 else ("stalled" if relres > 1e-2 else "partial")
    curve = " ".join(f"{r:.1e}" for r in resids[:: max(1, its // 8)][:9])
    print(f"{name:>12} {its:>5} {relres:>10.2e} {dt:>7.1f}  {verdict}  [{curve}]")

print("\nSPIKE_PROBE_DONE")
