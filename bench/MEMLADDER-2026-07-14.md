# Memory-vs-DOF ladder — independent re-measurement (2026-07-14)

One fresh Hetzner CPX51 (16 vCPU / 32 GB), this repo's `bench/Dockerfile.bench` image,
`bench/memladder.sh`. Motivated by three review questions: does factor memory scale
"in a straight line" with DOFs; does more MPI ranks mean more memory; and is the
coax-vs-sphere per-DOF difference real. All memory below is measured, with the metric
named: **factor** = MUMPS INFOG(22) "memory effectively used" summed over ranks;
**RSS** = peak resident set summed over ranks (the whole job, what
`memTimeEstimation.runTimesMems` records as `memCost`).

## 1. Same geometry, fixed ranks: LU vs LDLT ladder

`testRun` sphere geometry, degree 3, Nf=1, 8 ranks:

| DOFs | LU factor | LDLT factor | ratio | LU→LDLT solve time | max dS vs LU |
|---|---|---|---|---|---|
| 626,088 | 5,147 MB | 3,205 MB | 0.62 | 37.5 → 16.5 s * | 4.4e-16 |
| 898,902 | 7,283 MB | 4,336 MB | 0.60 | 25.7 → 22.4 s | 1.1e-15 |
| 1,231,185 | 9,863 MB | 5,939 MB | 0.60 | 35.8 → 29.3 s | 5.6e-16 |
| 1,803,117 | 15,619 MB | 9,148 MB | 0.59 | 61.1 → 46.0 s | 7.9e-16 |
| 2,258,487 | 19,495 MB | 11,580 MB | 0.59 | 74.1 → 60.6 s | 4.5e-16 |
| 2,801,391 | 24,021 MB | 14,035 MB | 0.58 | 111.0 → 74.1 s | 8.9e-16 |

\* first case in the session; LU time includes one-time JIT compilation — treat its
timing (not memory) as an outlier.

Per-DOF factor memory rises 8.2 → 8.6 MB/kDOF across the range: mildly superlinear,
as sparse-direct theory predicts. The 2.8M LU row independently reproduces the
certification table (24,021 vs 24,646 MB, −2.5%; solve 111 vs 120 s).

## 2. Rank count vs memory (same problem, stock LU, 898,902 DOFs)

| ranks | factor | RSS (whole job) | solve |
|---|---|---|---|
| 1 | 6,348 MB | 8.3 GB | 126.2 s |
| 4 | 6,605 MB | 10.6 GB | 41.5 s |
| 8 | 7,283 MB | 12.3 GB | 25.7 s |
| 16 | 8,997 MB | 15.4 GB | 16.4 s |

More ranks means more total memory on both metrics (+42% factor, +86% RSS from
1→16 ranks) buying a 7.7x speedup. On many-rank cluster jobs the per-rank overhead
dominates whole-job memory, which is why a factor-level change (LDLT) moves job RSS
much less than it moves the factor itself.

## 3. Geometry sets the per-DOF constant

Single rank, stock LU, same box, same day:

| case | DOFs | factor | per kDOF |
|---|---|---|---|
| coax cable-port (idx 3, h=1/3.5, deg 3) | 544,731 | 10,912 MB | 20.0 MB |
| testRun sphere (h=0.48, deg 3) | 966,117 | 6,823 MB | 7.1 MB |

The coax matrix costs **2.8x more memory per DOF** than the sphere matrix — memory
vs DOFs cannot be read across different geometries. (The 10,912 MB coax number also
reproduces the SOLVER-UPDATE table exactly, from a freshly regenerated matrix.)

## 4. Nf=22 workload shape (2,258,487 DOFs, 8 ranks, LDLT)

| config | factor | RSS (whole job) | total solve |
|---|---|---|---|
| `{'symmetric': True}` | 11,636 MB | 20.1 GB | 1,104 s |
| `{'symmetric': True, 'sweep_mode': True}` | 11,556 MB | 25.2 GB | 1,915 s |

**Negative result — do not chase:** at this scale `sweep_mode` is a net LOSS with a
plain LDLT factor (1.7x slower, slightly more memory, despite zero re-anchors): the
LDLT refactorization is cheap enough (~60 s) that FGMRES-iterating on a stale anchor
costs more than just refactorizing per frequency. sweep_mode's measured win remains
the fp32-anchor configuration at laptop scale (see SOLVER_IMPROVEMENTS.md); with fast
fp64 factors at multi-million DOFs, prefer plain per-frequency factorization.

## Reproduce

```bash
docker build -f bench/Dockerfile.bench -t scatt3d-bench bench/
docker run --rm -v $(pwd):/work -w /work/bench scatt3d-bench bash memladder.sh
# results stream to bench/memladder.log; per-case logs in bench/memladder_work/
```
