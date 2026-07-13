# Scatter3D-Claude

**A hardened edition of [Scatt3D](https://github.com/Wojoxiw/Scatt3D) — 3D electromagnetic
scattering FEM + linearized S-parameter defect imaging — with the measured-data bug fixed,
a 3.3× faster solver, a frequency-sweep iterative mode that actually converges, measurement
diagnostics, tests, CI, and a reproducible environment.**

Scatt3D (Alexandros Pallaris & Daniel Sjöberg, Lund University — EuCAP 2025,
[DOI:10.23919/EuCAP63536.2025.10999660](https://ieeexplore.ieee.org/document/10999660))
images fabrication defects inside known objects from microwave scattering parameters:
a FEniCSx/dolfinx FEM solver simulates the measurement scene to build a linearized
sensitivity operator, and a truncated-SVD pseudoinverse maps measured ΔS to a 3D
permittivity-perturbation image. It works beautifully on simulated data — and produced
only noise on real VNA measurements.

This repository is the result of a deep audit of why, and of making the code the way it
deserved to be. Every claim below is backed by a runnable artifact in this repo.

---

## What was found and fixed

| # | Finding | Where | Status |
|---|---------|-------|--------|
| 1 | **Transposed reference indices in the data vector**: `b = S_dut[nf,m,n] − S_ref[nf,n,m]`. Invisible with simulated (exactly reciprocal) S-matrices; injects each channel pair's VNA reciprocity error into every transmission row with real data. | `postProcessing.py:791` | **Fixed** + unit test (`Scatt3D/test_bvector_fix.py`) |
| 2 | **Redundant factorizations**: the system matrix depends only on frequency, but the solver re-assembled *and re-factorized* MUMPS for every antenna excitation. | `scatteringProblem.py` | **Fixed** — factorize once per frequency: **3.3× faster, bit-identical S-parameters** |
| 3 | No converging iterative option (≈20 commented-out failed attempts — expected: indefinite Maxwell + curl null-space defeat generic AMG/DD). | `scatteringProblem.py` | **Added** `sweep_mode`: anchor-LU + FGMRES across the frequency sweep, auto-re-anchor fallback. **2× faster on dense grids, ≤1e-8 error, cannot be less robust than the direct solve** |
| 4 | Direct-solver memory wall. | — | **Added** MUMPS BLR option (`blr_tol`). Honest caveat: pays off at production scale (degree-3 / cluster fronts), measurably *not* at small degree-1 scale — see benchmarks |
| 5 | **Measurement calibration is phase-only** (constant per antenna; the dispersive term is dead code; **no amplitude calibration exists anywhere**) — while every successful experimental system uses per-channel complex calibration. | `postProcessing.py:518-573` | Documented + **`measurement_diagnostics.py`** extracts the per-channel factors from data you already have |
| 6 | A stack of additional root-cause candidates for “sim works, measurement is noise” — near-zero POM↔PLA contrast, un-modeled metal optical table, drift/positioning budget, inverse-crime testing gap — ranked with checks and sources. | — | **[docs/WHY-MEASURED-IMAGING-FAILS.md](docs/WHY-MEASURED-IMAGING-FAILS.md)** |
| 7 | `readSol`/`create_interpolation_data` makes single-node sensitivity builds pathologically slow on fine meshes. | `scatteringProblem.py:470` | Documented (workaround: reconstruction submesh flow); flagged as high-value future fix |

## The two documents

- **[docs/WHY-MEASURED-IMAGING-FAILS.md](docs/WHY-MEASURED-IMAGING-FAILS.md)** — the ranked
  diagnosis: 6 tiers of causes ordered by (probability × cheapness to check), each with
  what/why/concrete check/source. Every citation adversarially verified.
- **[docs/SOLVER_IMPROVEMENTS.md](docs/SOLVER_IMPROVEMENTS.md)** — the solver rework:
  design, usage, measured benchmarks, verification summary, and why the failed iterative
  attempts were doomed (with literature).

## Quickstart

Everything runs in the standard dolfinx container plus a thin dependency layer:

```bash
docker build -f bench/Dockerfile.bench -t scatt3d .
docker run --rm -v $(pwd):/work -w /work/Scatt3D scatt3d bash -c \
  'source /usr/local/bin/dolfinx-complex-mode && python3 testExample.py'
```

Run the test suite (no FEM stack needed for the two fast tests):

```bash
python3 Scatt3D/test_bvector_fix.py               # proves the b-vector fix semantics
python3 Scatt3D/test_measurement_diagnostics.py   # diagnostics on synthetic ground truth
```

### Solver options (drop-in, via the existing `solver_settings` dict)

```python
prob = Scatt3DProblem(comm, mesh, ...)                                # direct, 3.3x faster than before
prob = Scatt3DProblem(comm, mesh, solver_settings={'blr_tol': 1e-6}) # + MUMPS BLR compression
prob = Scatt3DProblem(comm, mesh, solver_settings={'sweep_mode': True})  # anchor-LU + FGMRES sweep
prob = Scatt3DProblem(comm, mesh, solver_settings={'symmetric': True})   # complex-symmetric LDL^T: 0.528x stock LU measured (INFOG-22, 545k); 0.411x with +Scotch+BLR+ICNTL(37); +OOC = 0.38x measured peak RAM (single rank)
# maximum savings — fp32 factor + fp64 FGMRES (PETSc >= 3.25; needs both GEMMT shims): 0.220x (545k) / 0.309x (2.8M) measured, S-params 2e-7-class vs LU
prob = Scatt3DProblem(comm, mesh, solver_settings={'symmetric': True, 'mat_mumps_icntl_7': 3, 'blr_tol': 1e-6, 'mat_mumps_icntl_37': 1, 'pc_precision': 'single', 'ksp_type': 'fgmres', 'ksp_rtol': 1e-12, 'ksp_max_it': 100})
```

> `'symmetric'` needs a BLAS with a working GEMMT: OpenBLAS 0.3.26 (the default in
> `dolfinx/dolfinx:stable`) segfaults inside `zgemmt_` under ZMUMPS 5.8.2 — use the
> LD_PRELOAD shim in `bench/zgemmt_fix.f90` (already baked into `bench/Dockerfile.bench`).
> Details + measured memory ladder: `docs/SOLVER_IMPROVEMENTS.md`.

### Diagnose your measurement (no FEM required)

```bash
python3 Scatt3D/measurement_diagnostics.py \
    --ref MEAS/reference_folder --dut MEAS/dut_folder \
    --sim data3D/yourrun --qs --out diagnostics_out
```

Answers, from data you already have: is the defect signal above the reciprocity/drift
error floor at all; how much error the old indexing bug injected; the per-channel complex
calibration factors the pipeline currently ignores (exported ready-to-apply); and whether
your measured data lives in the model's range space (TSVD projection test — separates
“model is wrong” from “signal too small”, which demand different fixes).

## Benchmarks (all reproducible via `bench/`)

Baseline case: 73,736 dofs, 3 antennas, 3 frequencies (9 solves), dolfinx 0.11 complex, MUMPS, 1 rank.

| configuration | solve time | S-params vs stock |
|---|---|---|
| stock (re-factorize every solve) | 31.6 s | — |
| **rework, direct** | **9.7 s (3.3×)** | **bit-identical (max ∣ΔS∣ = 0.0)** |
| rework + BLR 1e-6 | 8.8 s | 5.6e-16 |
| rework, sweep (dense grid, 6 freqs) | **16.7 s vs 33.8 s direct (2.0×)** | 1.4e-08 (= tolerance, tunable) |

Verification stack: bit-identical S-parameters at three problem scales · sweep-mode
stressed on both stale-anchor (auto-refactorization observed working, 47/42 its) and
dense-grid (zero re-anchors) regimes · one-line imaging change unit-tested for both
reciprocal (no-op) and non-reciprocal (corrects exactly) inputs · diagnostics validated
closed-loop against pipeline outputs with a known injected calibration error (recovered
to 1.7e-6). Details: [docs/SOLVER_IMPROVEMENTS.md](docs/SOLVER_IMPROVEMENTS.md).

## Repository map

```
Scatt3D/                     core (upstream layout preserved deliberately — diffs stay honest)
  scatteringProblem.py       FEM: weak form, PML, ports, reworked solver, sensitivity kernels
  meshMaker.py               gmsh scene construction (antennas, object, defects, PML)
  postProcessing.py          imaging: data vector (FIXED), TSVD inversion, measured-data ingestion
  measurement_diagnostics.py standalone measurement triage tool
  test_bvector_fix.py        unit test for the critical fix
  test_measurement_diagnostics.py
  testExample.py             environment smoke test
bench/                       reproducible benchmark + verification harness (docker recipe included)
docs/                        the diagnosis report + solver documentation
```

## Attribution & license

Core physics code originates from [Wojoxiw/Scatt3D](https://github.com/Wojoxiw/Scatt3D)
by Alexandros Pallaris (Lund University); this repository exists to support that research
— see [NOTICE.md](NOTICE.md). Original code remains © its author. The improvements,
tests, tooling, and documentation added here are MIT-licensed ([LICENSE](LICENSE)).

Audit, fixes, and documentation by Claude (Anthropic), commissioned by Hunter Spence, 2026-07-12.
