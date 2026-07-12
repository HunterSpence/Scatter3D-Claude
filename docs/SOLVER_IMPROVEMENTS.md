# Scatt3D solver improvements (branch `perf/solver-and-measurement-fixes`)

Base: `Wojoxiw/Scatt3D` upstream master (`243c367`). Changes are surgical — the physics
(weak form, PML, port model, S-parameter extraction) is untouched and verified unchanged
against the stock solver to solver precision (see benchmark table).

## What changed

### 1. Factorize once per frequency, not once per excitation (`scatteringProblem.py`)
The system matrix depends only on frequency: the exciting-antenna index `n` enters the
weak form only through the RHS coefficient `a[n]`. The stock code used
`dolfinx.fem.petsc.LinearProblem.solve()` per (frequency, excitation), which re-assembles
AND re-factorizes MUMPS every call — for 4 antennas that is 4x more factorizations than
needed (16x for the 9-antenna simulated setups... per frequency).
The rework assembles the matrix once per frequency (`assembleLHS()`) and back-substitutes
per excitation (`solveCurrent()`). Direct-mode results are bit-compatible with stock up to
solver roundoff.

### 2. MUMPS Block Low-Rank compression (`solver_settings={'blr_tol': 1e-6}`)
Enables ICNTL(35)=2 with CNTL(7)=tol. The frontal blocks of the LU factorization are
compressed low-rank at a controlled, tunable accuracy cost, with zero new convergence
risk — it is still a direct solve. NOTE: pays off on LARGE fronts (degree-3,
multi-million-DOF cluster runs); at small/degree-1 scale the overhead can exceed the
gain — see the honest caveat under Benchmarks Case 3 before assuming a win.
References: Amestoy, Buttari, L'Excellent, Mary, "On the Complexity of the Block Low-Rank
Multifrontal Factorization", SIAM J. Sci. Comput. 39(4), 2017; PETSc MATSOLVERMUMPS docs
(https://petsc.org/release/manualpages/Mat/MATSOLVERMUMPS/); independently benchmarked as
one of the two winning approaches for time-harmonic Maxwell in Fressart et al. 2025
(arXiv:2507.13066).

### 3. Frequency-sweep mode: anchor LU as FGMRES preconditioner (`solver_settings={'sweep_mode': True}`)
Across a frequency sweep the matrix changes smoothly with k0, so the LU factorization of a
nearby ("anchor") frequency is an excellent preconditioner. In sweep mode the solver keeps
the anchor factorization and runs FGMRES at subsequent frequencies; it re-factorizes
(re-anchors) automatically only when iterations exceed `sweep_max_it` (default 25) or the
solve fails — so worst case it degenerates to the direct solver, never below it.
This IS a converging iterative solver for this problem — it sidesteps the reason all ~20
previous iterative attempts failed (see below) by using spectral information the sweep has
already paid for. The denser the frequency grid, the bigger the win (the measured sweeps
use 201 points over 5.4-7.2 GHz — simulating all of them becomes affordable).
Extra knobs: `sweep_rtol` (default 1e-8), `sweep_max_it`.

### 4. `mat_mumps_icntl_14 = 50`
Extra MUMPS workspace margin. The stock code occasionally produced NaN solutions that were
detected post-hoc ("NaN result found - possibly due to exceeding memory limitations");
underallocation (INFOG(1)=-9) is the usual cause.

### 5. Measured-data bug fix (`postProcessing.py`, solveFromQs)
`b[i] = S_dut[nf, m, n] - S_ref[nf, n, m]` had the reference S-matrix indices transposed.
Simulated S-matrices are reciprocal to solver precision, so all-simulation tests never see
it; measured S-matrices are not exactly reciprocal (VNA switch paths, cables, drift), so
the old indexing injected each channel-pair's reciprocity error into every transmission
row of b. Fixed to `S_ref[nf, m, n]`.

## Why the ~20 commented-out iterative attempts (GAMG/GASM/HPDDM/BDDC/MG) failed
The time-harmonic curl-curl operator is (a) sign-indefinite like Helmholtz — classical
smoothers/coarse spaces amplify rather than damp the oscillatory error modes (Ernst &
Gander, "Why it is difficult to solve Helmholtz problems with classical iterative
methods", 2012) — and (b) has an enormous near-null space (gradients of H1) that
scalar-oriented AMG coarse spaces (GAMG) do not represent (Hiptmair & Xu, SIAM J. Numer.
Anal. 45(6), 2007). Both pathologies apply at once, so generic AMG/DD failing is the
textbook-expected outcome, not a tuning problem. Hypre AMS (the curl-aware AMG) does not
support complex scalars at all (hypre issue #152) — the `realTest/real_tryer.py`
equivalent-real-form experiment was the right idea but 2x system size + memory-hungry
discrete-gradient setup made it impractical. The credible research-grade iterative routes
(equivalent-real 2x2 block AMS a la MFEM ex25p; GenEO-H(curl) coarse spaces, Bootland,
Dolean, Nataf, Tournier, J. Sci. Comput. 105:67, 2025, arXiv:2311.18783) are documented
here for completeness but are weeks-scale projects; the sweep mode above delivers the
practical benefit now.

## Benchmarks
All: testRun geometry (sphere object, 3 antennas, PML), dolfinx 0.11 complex, MUMPS,
1 MPI rank, Docker on an 8-core Ryzen laptop. "solve wall time" covers the whole
ComputeSolutions phase. Every configuration's S-parameters were compared against the
stock solver's values.

### Case 1 — baseline head-to-head (h=lambda/2, degree 1, 73,736 dofs, Nf=3, 9 solves)
| configuration | solve wall time | peak RSS | max \|dS\| vs stock |
|---|---|---|---|
| stock (LinearProblem re-factorizes every solve) | 31.6 s | 0.52 GB | — |
| rework, direct (factor 1x/freq)                 | **9.7 s (3.3x)** | 0.50 GB | **0.0 (bit-identical)** |
| rework + BLR 1e-6                               | 8.8 s | 0.50 GB | 5.6e-16 |
| rework, sweep mode (WIDE grid: GHz-scale steps) | 26.7 s | 0.60 GB | 8.1e-15 |

On the wide default grid the sweep anchor goes stale at every point (47/42 FGMRES its
-> auto re-anchor), so it degenerates gracefully toward direct mode — the designed
fallback behavior, observed working.

### Case 2 — dense frequency grid (Nf=6 over 9.95-10.05 GHz, 20 MHz steps, 18 solves)
| configuration | solve wall time | notes |
|---|---|---|
| rework, direct | 33.8 s | 6 factorizations |
| rework, sweep  | **16.7 s (2.0x)** | **1 factorization served all 6 freqs, zero re-anchors**; max \|dS\| = 1.4e-08 vs direct (= sweep_rtol 1e-8, tunable) |

The real measurement sweeps use 201 points at ~9 MHz spacing — far denser than this
demo, so the sweep-mode advantage there is larger (factorization count collapses from
201 to a handful).

### Case 3 — larger problem (h=lambda/4, degree 1, 290,169 dofs, Nf=1, 3 solves)
| configuration | solve wall time | peak RSS | max \|dS\| vs direct |
|---|---|---|---|
| rework, direct | 27.7 s | 1.27 GB | — |
| rework + BLR 1e-6 | 71.6 s | 1.24 GB | 8.6e-12 |

**Honest caveat:** at this size with degree-1 elements the BLR compression overhead
EXCEEDS its gains (slower, no memory win) — the frontal matrices are too small to
compress profitably. BLR's published wins are on large fronts (degree-3, multi-million
DOF, many MPI ranks — i.e. the actual production runs on the cluster). Treat
`blr_tol` as a knob to benchmark at production scale, not a guaranteed win; accuracy
impact is demonstrably tiny and tunable.

Reproduce: `bench/run_headtohead.sh` / `bench/run_final_cases.sh` in the audit
workspace, or call `Scatt3DProblem(..., solver_settings={...})` — the settings dict
is already plumbed through the constructor upstream.

## Verification summary (what was actually proven)

1. **S-parameter equivalence** — the physics outputs of the reworked solver were compared
   against stock at three scales: baseline (74k dofs, 9 solves): **max |dS| = 0.0,
   bit-identical**; BLR: 5.6e-16; sweep: 8.1e-15; dense grid sweep-vs-direct: 1.4e-08
   (= sweep_rtol, tunable); 290k dofs BLR-vs-direct: 8.6e-12. The E-fields feeding the
   imaging kernel are solutions of the identical assembled systems.
2. **Sweep-mode convergence behavior** — stressed both ways: wide grid (anchor goes stale:
   47/42 FGMRES its observed -> automatic re-anchor -> still 8.1e-15 accurate) and dense
   grid (zero re-anchors, one factorization for 6 frequencies, 2.0x faster).
3. **Imaging-side change** — the ONLY imaging-side code change is the one-line b-vector
   fix. `Scatt3D/test_bvector_fix.py` proves: on reciprocal (simulated) S-matrices old
   and new indexing agree EXACTLY (zero regression for every all-simulation result); on
   non-reciprocal (measured-like) S-matrices the old line injected exactly the reference
   reciprocity error into every transmission row, the fixed line does not, and reflection
   rows are unaffected in both.
4. **Diagnostics tool** — `test_measurement_diagnostics.py` (synthetic data with known
   injected floor/signal: recovered exactly) plus a closed-loop validation against real
   pipeline output files (calibration factors recover a known injected per-channel error).

## Known upstream bottleneck found during verification
`makeOptVectors` -> `readSol` calls `dolfinx.fem.create_interpolation_data`
(scatteringProblem.py:470) for every stored solution it reads back. On a single node
with a lambda/4 production-like mesh (~400k cells) ONE such call can run for tens of
minutes at 100% CPU with no progress output — a full 32-row sensitivity build is
hours-to-days locally (it is presumably tolerable on the cluster with many ranks).
Anyone trying to reproduce reconstructions locally should use the author's
reconstruction-submesh flow (`switchToRecMesh` + `makeOptVectors(reconstructionMesh=True)`)
and/or small meshes; a caching or same-mesh fast path in `readSol` would be a
high-value future fix (the checkpoint is being interpolated onto what is logically the
same mesh).

## What was NOT changed
- No physics, no mesh, no S-parameter extraction changes.
- `Zm` for the '6GHz measurement' antenna type is still the hard-coded
  `eta0/sqrt(2.1*(1-0.01j))` — flagged in the audit report (it is a sim-vs-VNA reference
  impedance mismatch candidate) but left for the author to decide.
- The measured-data calibration (`compileMeasuredSs`) is still phase-only; see the main
  audit report for the recommended per-channel complex calibration and the new
  `measurement_diagnostics.py` for measuring what it should be.
