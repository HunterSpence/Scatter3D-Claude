#!/usr/bin/env python3
# Mixed-precision factorization probe: fp64 outer FGMRES + single-precision (CMUMPS)
# LDLT factor via PETSc 3.25 -pc_precision single, vs same-build fp64 baseline.
# Usage: python3 mpprobe.py <fp64|fp32> [icntl_7=3] [cntl_7=1e-6] [icntl_37=1] ...
#
# Input matrix /work/bench/cableport/A545k.bin (545k-dof deg-3 coax system) is
# regenerated with (inside the bench container, ~4 min on 16 cores):
#   mkdir -p bench/cableport && cd bench/cableport
#   PETSC_OPTIONS="-theScatteringProblem_ksp_view_mat binary:/work/bench/cableport/A545k.bin" \n#   SCATT3D_SRC=/work/Scatt3D python3 -u /work/bench/cableport_validate.py \n#   3 "{}" dump 0.2857142857142857 3 1
import sys
import resource
import petsc4py
petsc4py.init(sys.argv)
from petsc4py import PETSc
import numpy as np

mode = sys.argv[1]
kv_args = sys.argv[2:]

comm = PETSc.COMM_WORLD
print(f"MPPROBE mode={mode} opts={kv_args} LOADING...")
sys.stdout.flush()

try:
    viewer = PETSc.Viewer().createBinary("/work/bench/cableport/A545k.bin", "r", comm=comm)
    A = PETSc.Mat().load(viewer)
    viewer.destroy()
    A.assemble()
    n = A.getSize()
    print(f"MATRIX_INFO n={n}")
    sys.stdout.flush()

    rng = np.random.default_rng(0)
    x_exact_np = rng.standard_normal(n[0]) + 1j * rng.standard_normal(n[0])
    x_exact = PETSc.Vec().createWithArray(x_exact_np, comm=comm)
    b = A.createVecLeft()
    A.mult(x_exact, b)
    x = A.createVecRight()

    opts = PETSc.Options()
    prefix = "op_"

    def set_opt(key, val):
        opts[prefix + key] = val

    set_opt("pc_factor_mat_solver_type", "mumps")
    set_opt("pc_type", "cholesky")
    A.setOption(PETSc.Mat.Option.SYMMETRIC, True)
    A.setOption(PETSc.Mat.Option.SYMMETRY_ETERNAL, True)

    if mode == "fp32":
        set_opt("pc_precision", "single")
        set_opt("ksp_type", "fgmres")
        set_opt("ksp_rtol", "1e-12")
        set_opt("ksp_max_it", "100")
    elif mode == "fp64":
        set_opt("ksp_type", "preonly")
    else:
        raise SystemExit(f"unknown mode: {mode}")

    for kv in kv_args:
        key, val = kv.split("=", 1)
        set_opt(f"mat_mumps_{key}", val)

    ksp = PETSc.KSP().create(comm=comm)
    ksp.setOptionsPrefix(prefix)
    ksp.setOperators(A)
    ksp.setFromOptions()
    print(f"MPPROBE mode={mode} SOLVING...")
    sys.stdout.flush()
    ksp.solve(b, x)
    its = ksp.getIterationNumber()
    reason = ksp.getConvergedReason()

    err = x.copy()
    err.axpy(-1.0, x_exact)
    rel_err = err.norm(PETSc.NormType.NORM_2) / x_exact.norm(PETSc.NormType.NORM_2)

    # true fp64 residual ||b - Ax|| / ||b||
    r = b.copy()
    tmp = A.createVecLeft()
    A.mult(x, tmp)
    r.axpy(-1.0, tmp)
    rel_res = r.norm(PETSc.NormType.NORM_2) / b.norm(PETSc.NormType.NORM_2)

    print(f"KSP its={its} reason={reason}")
    try:
        F = ksp.getPC().getFactorMatrix()
        for i in [7, 13, 16, 17, 18, 19, 21, 22]:
            print(f"INFOG_{i}={F.getMumpsInfog(i)}")
    except Exception as e:
        print(f"INFOG_UNAVAILABLE {e}")

    maxrss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 ** 2)
    print(f"RSS maxrss={maxrss_gb:.4f} GB")
    print(f"rel_err={rel_err:.6e} rel_res={rel_res:.6e}")
    if reason > 0 and np.isfinite(rel_err) and np.isfinite(rel_res):
        print("CASE_OK")
    else:
        print(f"CASE_FAIL converged_reason={reason} (non-converged or non-finite result)")

except Exception as e:
    print(f"CASE_FAIL {e}")
    sys.exit(1)
