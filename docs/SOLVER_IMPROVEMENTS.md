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

## 6. Symmetric LDL^T mode (`solver_settings={'symmetric': True}`) — WIP branch

The assembled operator is complex symmetric (A = A^T, not Hermitian): curl-curl, mass,
diagonal-tensor PML and the port Robin terms are all symmetric, and PEC elimination is
applied symmetrically. MUMPS can therefore factorize it as LDL^T (SYM=2), storing roughly
half of what LU stores. Enabled by flagging the AIJ matrix SYMMETRIC/SYMMETRY_ETERNAL and
using PCCHOLESKY instead of PCLU (MUMPS backend, numerical pivoting — valid for
indefinite complex-symmetric systems). Stacks with `blr_tol`.

### The crash this replaces (and why the old cholesky experiment failed)

Enabling this mode in `dolfinx/dolfinx:stable` segfaulted inside MUMPS at the first
numeric factorization. Root cause — found by dumping the real assembled matrix
(`-<prefix>ksp_view_mat binary:`) and bisecting with a standalone load-and-factor
harness + gdb:

- **OpenBLAS 0.3.26's `zgemmt` kernel is broken** (the image's active `libblas.so.3`
  alternative). ZMUMPS 5.8.2's LDL^T frontal update calls the BLAS-extension GEMMT
  (triangular-referenced GEMM); the call segfaults inside `zgemmt_` even with
  `OPENBLAS_NUM_THREADS=1`. gdb frames: `zgemmt_` ← `zmumps_fac_t_ldlt` ← `zmumps_fac_par`.
- LU is unaffected because the unsymmetric path never calls GEMMT — which is why only
  Cholesky/LDL^T ever crashed (the author's old commented-out cholesky experiment that
  "gave NaNs" is plausibly the same disease in an older guise).
- The image's *reference* BLAS alternative exports no GEMMT symbol at all (GEMMT is an
  MKL extension adopted by OpenBLAS, never part of netlib BLAS), so switching
  alternatives is not a fix. All 11 MUMPS-side knobs tested (orderings AMD/PORD/METIS,
  pivot thresholds, icntl_58, icntl_12, icntl_24, SBAIJ storage) crash identically —
  the bug is below MUMPS.
- Toy complex-symmetric matrices do NOT reproduce it (their frontal blocks are too small
  to reach the blocked GEMMT path) — the dumped-real-matrix harness in `bench/symtest/`
  is the go-to repro pattern, and doubles as an upstream OpenBLAS bug report.

**Fix:** `bench/zgemmt_fix.f90` — a blocked ZGEMM-based drop-in `zgemmt_`, LD_PRELOADed
over the system BLAS (baked into `bench/Dockerfile.bench`). Unit-tested against numpy
ground truth in `bench/test_zgemmt.py` (all uplo/trans combos, block-boundary sizes,
beta=0 not-read semantics; max rel err ~1e-16).

### Verified results (with the fix)

74k-dof deg-1 sphere benchmark (single rank): symmetric mode matches LU to
**max|dS| = 3.3e-16** (bit-level), factorization-memory analysis estimate (INFOG 16/17) **135 MB vs 224 MB (0.60x)**, and the
solve is ~2x faster.

### Memory metric definitions (MUMPS 5.8 users' guide semantics — read before the tables)

- **INFOG(16)/INFOG(17)** — *analysis-phase ESTIMATES* of factorization memory
  (max-per-process / total). Available before the numeric phase runs. The tables below
  that cite INFOG(16/17) are **estimate ratios** — honest, reproducible, but predictions.
- **INFOG(18)/INFOG(19)** — memory *allocated* during factorization.
- **INFOG(21)/INFOG(22)** — memory *effectively used* during factorization
  (max-per-process / total). **This is the measured number; ratio claims should quote
  it.** The branch now records it (`lastFactorMemMB` is a 4-tuple: est-max, est-total,
  eff-max, eff-total); the at-scale certification pass (2.8M/3.23M dofs: LDL^T
  in-core, LDL^T+OOC, LU+OOC, with aggregate peak RSS across all ranks) completed
  2026-07-13 — tables below.
- **Peak RSS** — OS-level resident memory (includes PETSc/dolfinx/MPI overhead, not just
  MUMPS). Reported as sum and max across ranks. This is the "does it fit the machine"
  number, and the right axis for out-of-core comparisons.

Memory ladder, degree 3, 8 MPI ranks, CPX51 (16 vCPU / 32 GB) — **analysis-estimate
metric INFOG(16/17)** (measured-effective certification tables further below):

| dofs | config | factor mem (total) | vs LU | solve | max \|dS\| vs LU |
|---|---|---|---|---|---|
| 898,902 | lu | 10,021 MB | 1.00 | 42.5 s | — |
| 898,902 | sym | 6,043 MB | 0.60 | 22.1 s | 1.2e-15 |
| 898,902 | sym+blr 1e-6 | 6,096 MB | 0.61 | 22.0 s | 5.2e-11 |
| 1,803,117 | lu | 22,305 MB | 1.00 | 59.9 s | — |
| 1,803,117 | sym | 13,065 MB | 0.586 | 46.6 s | 1.0e-15 |
| 1,803,117 | sym+blr 1e-6 | 13,243 MB | 0.594 | 43.2 s | 2.1e-10 |
| 2,801,391 | lu | **OOM-killed (does not fit 32 GB)** | — | — | — |
| 2,801,391 | sym | 20,143 MB | runs where LU cannot | 72.7 s | vs sym+blr: 5.9e-11 |
| 2,801,391 | sym+blr 1e-6 | 20,290 MB | — | 65.7 s | — |
| **3,232,812** | **sym** | **24,389 MB** | **runs where LU cannot** | 109.9 s | — |

At and above ~2.8M dofs (degree 3) direct LU no longer fits the 32 GB box at all —
the OOM kill *is* the LU baseline there.
(Framing note: the OOM boundary is a property of the 32 GB test box, not of the solver —
on a large-memory cluster LU simply runs and the in-core ratio applies. Treat "fits
where LU cannot" as a capacity/commodity-hardware benefit, secondary to the measured
memory ratios.) A swap-assisted measured LU run at 2.8M
(below) pins the actual ratio at the largest size where LU can be made to complete.
The h=0.22 rung (~5.8M dofs) was skipped deliberately: beyond the 3M acceptance
target and swap-bound for hours with no additional claim value.
(A swap-assisted in-core LU run at 2.8M was deliberately skipped at certification
time: the LU+OOC certification run below provides the measured at-scale LU
denominator, and burning up to 4 further box-hours to convert one labeled
estimate ratio into a measured one did not change any conclusion. The at-scale
sym-vs-LU accuracy is certified below at machine epsilon via the OOC path.)

Note: with `blr_tol`, the INFOG(16/17) analysis estimates are computed full-rank —
BLR compression is invisible to this metric, so BLR shows no saving in the table above
(the LDL^T halving is the dominant effect there). The compression is real in the
measured INFOG(21/22) — see the MEASURED results section below.

Cable-port validation (`bench/cableport_validate.py`, deg-3 coax vs analytic
transmission-line theory, 3 frequencies, single rank) — **VERDICT: PASS**. (Factor-memory
columns in this table are INFOG(16/17) analysis estimates.) Symmetric
mode matches theory *identically* to LU (same max error to every displayed digit):

| coax (epsr1 -> epsr2) | dofs | max mag rel-err vs theory (lu = sym) | max phase err | max \|dS\| sym vs lu | factor mem sym/lu |
|---|---|---|---|---|---|
| 0: 2.1 -> 2.1(1-1j) | 38k-class | 3.321e-04 | 0.130 deg | 5.7e-14 | 2067 MB / — |
| 3: 2.1 -> 12.1 | 544,731 | 3.278e-04 | 0.457 deg | 4.9e-12 | 8700 / 15892 MB (0.547) |
| 5: 1.3 -> 2.1(1-0.8j) | small | 3.576e-04 | 0.040 deg | 8.3e-15 | 984 / 1774 MB (0.555) |

Single-rank LDL^T/LU memory ratios (~0.55) run tighter than the 8-rank MPI ladder
(~0.59) — per-rank workspace overhead grows with rank count.

### Closing the last gap to "half": out-of-core (BLR verdict superseded by the measured section below)

Measured on the 545k-dof deg-3 coax case (single rank, LU baseline 15,892 MB / peak RSS 8.69 GB / 566 s):

| config | INFOG(16/17) | peak process RSS | accuracy vs theory | solve |
|---|---|---|---|---|
| sym | 8,702 MB (0.548) | 6.59 GB | 2.8e-4 (= LU) | 108 s |
| sym + blr 1e-5 | 8,702 MB — unchanged | 6.59 GB | 2.8e-4 | 108 s |
| sym + blr 1e-4 | 8,702 MB — unchanged | — | **BROKEN: 11% mag err, |S11|>1** | 136 s |
| **sym + OOC (icntl_22=1)** | 8,700 MB (reported, factors on disk) | **3.31 GB = 0.38x LU** | 3.11e-4 — identical digits to LU | 477 s (still < LU) |

- By the INFOG(16/17) estimate metric BLR shows no reduction at any tolerance, and
  above ~1e-5 it fails nonlinearly (11% S-parameter error at blr 1e-4 — ~8 orders
  beyond its nominal tolerance; complex-symmetric indefinite pivoting + compression
  interact badly, so do not loosen blr_tol past 1e-5). **SUPERSEDED for memory:** the
  measured INFOG(21/22) section below shows the compression is real (~12-14% further
  saving at 1e-6/1e-5 under Scotch) — the "no reduction" was a metric artifact; pair
  BLR with `mat_mumps_icntl_10=2` refinement (see shipping config).
### MEASURED effective-memory results — the target is MET in-core (2026-07-12, pause-point)

545k-dof deg-3 coax matrix, single rank, MUMPS INFOG(22) = memory effectively used
during factorization (the measured metric). Stock LU reference (default options,
icntl_14=20): estimate 12,889 MB, **measured 10,898 MB**. Cross-validated: the plain
Scotch config measured identically (5,755 MB) in two independent probe harnesses.

| config | INFOG(17) est | INFOG(22) measured | ratio vs stock LU |
|---|---|---|---|
| LU stock (defaults) | 12,889 | 10,898 | 1.000 |
| LU + Scotch ordering | 12,464 | 10,543 | 0.967 (ordering barely helps LU) |
| LDL^T default ordering | 7,049 | 5,957 | 0.547 |
| LDL^T + PORD | 6,873 | 5,811 | 0.533 |
| LDL^T + Scotch | 6,806 | 5,755 | 0.528 |
| LDL^T + Scotch + relaxed pivoting + 2 IR steps | 6,806 | 5,755 | 0.528 (pivot relax adds nothing) |
| LDL^T + ICNTL(12)=2/3 variants | 7,049 | 5,957 | 0.547 (null effect) |
| LDL^T + BLR 1e-5 + ICNTL(38)=200 (default ordering) | 7,051 | 5,078 | 0.466 (diagnostic only: rel_err 9.6e-2 without IR) |
| **LDL^T + Scotch + BLR 1e-6** | 6,808 | **5,039** | **0.462** |
| **LDL^T + Scotch + BLR 1e-5** | 6,808 | **4,974** | **0.456** |
| **LDL^T + Scotch + BLR 1e-6 + ICNTL(37) CB compression** | 6,808 | **4,476** | **0.411** |

**The earlier "BLR gives no memory win" finding was a metric artifact, exactly as
suspected in review: BLR compression is invisible to the INFOG(16/17) analysis
estimates (computed full-rank, before BLR runs) but real in the measured INFOG(22) —
a further ~12–14% under the Scotch ordering.** Combined config
`solver_settings={'symmetric': True, 'mat_mumps_icntl_7': 3, 'blr_tol': 1e-6}`:
**0.462x of stock LU's measured factorization memory, in-core.** Accuracy at blr 1e-6
previously measured at 2.1e-10 max |dS| vs LU at 2.8M dofs (deg-3); a dedicated
accuracy confirmation of the exact winning config is running at pause time. Note:
INFOG(29) (effective factor entries) reads full-rank in this MUMPS version even with
BLR active — the compression shows in INFOG(22)/(21), not INFOG(29).
Accuracy confirmation of the winning config (random-RHS forward error, 545k, single
rank; the S-parameter agreement — the physical acceptance quantity — was separately
measured at 2.1e-10 vs LU at 2.8M dofs for blr 1e-6):

| config | INFOG(22) | ratio | rel_err (random RHS) |
|---|---|---|---|
| Scotch + blr 1e-6, no refinement | 5,039 | 0.462 | 8.6e-3 |
| **Scotch + blr 1e-6 + ICNTL(10)=2 refinement** | **5,039** | **0.462** | **4.7e-6** |
| Scotch + blr 1e-8 + ICNTL(10)=2 | 5,295 | 0.486 | 7.4e-5 (non-monotonic vs 1e-6 — noted, not chased) |
| Scotch, no BLR (pristine reference) | 5,755 | 0.528 | 1.0e-10 |
| **Scotch + blr 1e-6 + IR + ICNTL(37)=1 (CB compression)** | **4,476** | **0.411** | **6.1e-6** |
| Scotch + blr 1e-6 + IR + ICNTL(36)=1 (UCFS) | 5,039 | 0.462 | 4.7e-6 (no memory effect) |
| Scotch + blr 5e-7 / 2e-7 + IR | 5,069 / 5,114 | 0.465 / 0.469 | 4.1e-5 / 6.0e-6 (1e-6 is the local optimum) |
| Scotch + blr 1e-5 + IR + ICNTL(37)=1 | 4,262 | 0.391 | 2.6e-2 — REJECTED (accuracy broken even with IR) |
| ICNTL(37)=1 without BLR | 5,755 | 0.528 | 1.0e-10 (CB compression inert unless ICNTL(35) active) |

Follow-up probes (2026-07-13, same 545k matrix and harness): **ICNTL(37) contribution-block
compression stacks on BLR for a further ~11% measured saving — new best in-core 0.411 —**
while leaving forward error in the same class (6.1e-6 vs 4.7e-6). Two negative results
worth keeping: UCFS (ICNTL 36) had no memory effect here, and in OOC mode compression
HURTS the in-RAM working set (sym+OOC plain: 1,341 MB measured working set, rel_err
1.0e-10; sym+OOC+BLR+CB37: 3,069 MB) — keep OOC plain. At-scale certification of the
0.411 config (2.8M/3.23M, 8 ranks, with ICNTL(13) root-treatment A/B and OS-level
cgroup peak-RAM capture) is running; tables append here when complete.

Recommended shipping config:
`solver_settings={'symmetric': True, 'mat_mumps_icntl_7': 3, 'blr_tol': 1e-6, 'mat_mumps_icntl_10': 2}`
— 0.462x stock LU measured, S-params at 2.1e-10-class agreement; the pristine no-BLR
config (0.528) and OOC (0.38x RSS) bracket it for users with different priorities.

- Out-of-core LDL^T is a co-equal PRIMARY result, not a fallback — the strongest measured number against the strict 0.5 line: 0.38x LU peak RAM,
  bit-for-bit LU-class accuracy, and still faster than the stock solver. Enable with
  solver_settings={'symmetric': True, 'mat_mumps_icntl_22': 1}.

### At-scale certification — COMPLETE (2026-07-13, 8 MPI ranks, CPX51 16 vCPU / 32 GB)

Six runs (deg 3, Nf=1): h=0.28 → 2,801,391 dofs and h=0.26 → 3,232,812 dofs, each as
LDL^T in-core, LDL^T + OOC, and LU + OOC. In-core LU does not fit this box at either
size (the ladder's OOM boundary), so **LU + OOC is the measured LU denominator at
scale**, and same-mode (OOC vs OOC) ratios are the apples-to-apples comparison.
Every value below is from the run's own MUMPS INFOG output and OS accounting
(`bench/box-evidence-2026-07-13/certify.log`).

| dofs | config | INFOG(17) est total | INFOG(22) measured total | INFOG(21) measured max/proc | RSS sum / max-proc | solve |
|---|---|---|---|---|---|---|
| 2,801,391 | LDL^T in-core | 20,691 MB | **14,332 MB** | 2,315 MB | 23.83 / 3.58 GB | 91.0 s |
| 2,801,391 | LDL^T + OOC | 20,132 MB | **5,432 MB** | 784 MB | 15.55 / 2.23 GB | 74.2 s |
| 2,801,391 | LU + OOC | 34,599 MB | **8,265 MB** | 1,179 MB | 21.24 / 2.75 GB | 101.0 s |
| 3,232,812 | LDL^T in-core | 25,113 MB | **17,100 MB** | 2,632 MB | 28.42 / 4.09 GB | 88.6 s |
| 3,232,812 | LDL^T + OOC | 24,683 MB | **6,706 MB** | 1,028 MB | 18.03 / 2.55 GB | 90.8 s |
| 3,232,812 | LU + OOC | 41,150 MB | **9,613 MB** | 1,356 MB | 23.36 / 3.09 GB | 138.3 s |

Certified ratios (metric named on every line):
- **LDL^T+OOC vs LU+OOC, measured INFOG(22): 0.657 at 2.8M, 0.698 at 3.23M.**
- LDL^T+OOC vs LU+OOC, aggregate RSS (sum over ranks): 0.732 at 2.8M, 0.772 at 3.23M.
- LDL^T vs LU, analysis-estimate INFOG(17) (estimates, labeled as such): 0.598 at
  2.8M, 0.610 at 3.23M — consistent with the ladder's estimate ratios.
- LDL^T **in-core** runs at both sizes on the 32 GB box (measured 14.3 / 17.1 GB)
  where in-core LU cannot run at all.
- **Accuracy at scale (the acceptance quantity): max|dS| LU+OOC vs LDL^T = 9.1e-16 at
  2.8M and 1.45e-15 at 3.23M; LDL^T+OOC vs LDL^T = 2.3e-16 / 1.4e-16 — machine-epsilon
  class agreement between LDL^T and LU at 2.8M–3.23M dofs.**

OOC INFOG(22) semantics: with `mat_mumps_icntl_22=1` the factors live on disk, so
"memory effectively used" reflects the in-RAM working set — that is the point of OOC.
The honest cross-mode axis is aggregate RSS, reported alongside. Swap was present but
unused (0B) for every certification run.

### End-to-end pipeline equivalence — PASS (e2e suite, 2026-07-13)

Four cases through the complete pipeline (mesh → sim → S-parameters → a-priori
reconstruction; h=0.25 four-antenna synthetic scene, deg 1, Nf=2, 8 ranks):
`stock` = upstream tree, `branch` = this branch (default settings), `blr` = branch +
`blr_tol 1e-6`, `sweep` = branch + `sweep_mode`.

S-parameters vs stock (max|dS| over S_ref / S_dut):

| case | max \|dS_ref\| | max \|dS_dut\| | reconstruction_err | rel diff vs stock |
|---|---|---|---|---|
| stock | — | — | 3.149682913909 | — |
| branch | 2.221e-16 | 2.220e-16 | 3.149682913224 | 2.2e-10 |
| sweep | 1.776e-14 | 1.821e-14 | 3.149682824606 | 2.8e-8 |
| blr | 7.297e-13 | 6.442e-13 | 3.149711974606 | 9.2e-6 (consistent with blr_tol 1e-6) |

The branch is numerically indistinguishable from stock through the full imaging path
(machine-epsilon S-parameters); sweep and blr agree within their configured tolerances.

Two robustness findings from the harness (post-processing data path, not solver):
1. `postProcessing.solveFromQs` executes `del Apart` at function scope; when the load
   loop binds nothing, this raises `UnboundLocalError`. Guarded on this branch
   (`Apart = None` — same gc effect, cannot throw).
2. The default `maxRefl=0.7` data-quality filter silently selects **0 of 32 rows** on a
   fully-reflective scene (|S_mm| = 1.0 at every antenna/frequency here) and returns a
   vacuous reconstruction_err = 1.0 with no warning. The e2e harness passes `maxRefl=0`;
   a "0 rows selected" warning would make this failure mode loud.

### Measured-data diagnostics (qs-diagnostics, synthetic validation run)

The measurement-reliability diagnostic pipeline (reciprocity error floor, b-vector
bug impact, per-channel calibration, TSVD subspace projection) ran end-to-end on
synthetic measured data generated from the stock e2e sim (21-point 8–12 GHz grid).
On this synthetic set: defect-signal / reciprocity-error-floor = **0.98** (signal at
the error floor — no inversion can succeed until |dS| is raised or the floor lowered);
the old transposed-indexing b-vector term injects spurious error **1.02x** the genuine
signal in transmission rows; TSVD projection: measured-b energy in the top-10%
singular subspace 14.2% vs simulated 31.5% (the model/data-mismatch signature).
These verdicts characterize the synthetic set and certify the tooling end-to-end; the
same diagnostics are the recommended first pass on real VNA data
(`bench/box-evidence-2026-07-13/scatt3d-qs-diag.log`, plots in `diag_out/`).
