# Direct-solver frontier probe: factor the dumped 545k coax matrix with a chosen
# PC type + factor solver package; report factor time, peak RSS, MUMPS INFOG when
# applicable, and forward-solve accuracy. Usage: probe4.py <pctype> <solver> [petsc opts]
import sys
import resource
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np
from timeit import default_timer as timer

pctype, solver = sys.argv[1], sys.argv[2]
comm = PETSc.COMM_WORLD
viewer = PETSc.Viewer().createBinary("/work/bench/cableport/A545k.bin", "r", comm=comm)
A = PETSc.Mat().load(viewer)
viewer.destroy()
A.assemble()
n = A.getSize()[0]
A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

rng = np.random.default_rng(0)
x_ex = rng.standard_normal(n) + 1j * rng.standard_normal(n)
xv = PETSc.Vec().createWithArray(x_ex, comm=comm)
b = A.createVecLeft()
A.mult(xv, b)
x = A.createVecRight()

ksp = PETSc.KSP().create(comm)
ksp.setOperators(A)
ksp.setType("preonly")
pc = ksp.getPC()
pc.setType(pctype)
pc.setFactorSolverType(solver)
ksp.setFromOptions()
t0 = timer()
ksp.setUp()
tf = timer() - t0
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
infog = ""
try:
    F = pc.getFactorMatrix()
    infog = (f"INFOG21={F.getMumpsInfog(21)} INFOG22={F.getMumpsInfog(22)}")
except Exception:
    pass
ksp.solve(b, x)
err = (x - xv).norm() / xv.norm()
print(f"PROBE4_RESULT pc={pctype} solver={solver} n={n} factor_s={tf:.1f} "
      f"rss_before_mb={rss0:.0f} rss_peak_mb={rss1:.0f} {infog} rel_err={err:.2e}")
