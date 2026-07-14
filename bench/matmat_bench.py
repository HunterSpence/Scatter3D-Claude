# Multi-RHS batching probe: N sequential KSPSolve vs one MatMatSolve against the
# same MUMPS LDLT factor. Single rank, dumped 545k coax matrix.
import sys
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
from timeit import default_timer as timer
import numpy as np

comm = PETSc.COMM_WORLD
viewer = PETSc.Viewer().createBinary("/work/bench/cableport/A545k.bin", "r", comm=comm)
A = PETSc.Mat().load(viewer)
viewer.destroy()
A.assemble()
n = A.getSize()[0]
A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
print(f"MATMAT n={n}")

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType("cholesky")
pc.setFactorSolverType("mumps")
t0 = timer()
ksp.setUp()
print(f"MATMAT factor_s={timer()-t0:.2f}")
F = pc.getFactorMatrix()

rng = np.random.default_rng(1)
for N in (4, 9, 32):
    Bnp = (rng.standard_normal((n, N)) + 1j * rng.standard_normal((n, N))).astype(np.complex128)
    b = A.createVecLeft()
    x = A.createVecRight()
    t0 = timer()
    for k in range(N):
        b.setArray(Bnp[:, k])
        ksp.solve(b, x)
    t_seq = timer() - t0
    B = PETSc.Mat().createDense([n, N], array=np.asfortranarray(Bnp), comm=comm)
    B.assemble()
    X = PETSc.Mat().createDense([n, N], comm=comm)
    X.setUp()
    X.assemble()
    t0 = timer()
    F.matSolve(B, X)
    t_blk = timer() - t0
    # correctness: residual of column 0
    x0 = PETSc.Vec().createWithArray(np.ascontiguousarray(X.getDenseArray()[:, 0]), comm=comm)
    r = A.createVecLeft()
    A.mult(x0, r)
    r.axpy(-1.0, PETSc.Vec().createWithArray(np.ascontiguousarray(Bnp[:, 0]), comm=comm))
    rel = r.norm() / np.linalg.norm(Bnp[:, 0])
    print(f"MATMAT N={N} seq_s={t_seq:.2f} block_s={t_blk:.2f} speedup={t_seq/t_blk:.2f} relres_col0={rel:.2e}")
    B.destroy(); X.destroy()
print("MATMAT_DONE")
