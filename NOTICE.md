# Notice — provenance and licensing

## Upstream origin

The core simulation and imaging code in `Scatt3D/` derives from
**[Wojoxiw/Scatt3D](https://github.com/Wojoxiw/Scatt3D)** by **Alexandros Pallaris**
(Lund University, Dept. of Electrical and Information Technology; advisor Daniel Sjöberg),
base commit `243c367`. The underlying method is published as:

> A. Pallaris, D. Sjöberg, "Microwave Reconstruction of Fabrication Defects in Known
> Objects Using Scattering Parameter Sensitivities," EuCAP 2025,
> DOI:10.23919/EuCAP63536.2025.10999660

The upstream repository carries **no license file**; the original code therefore remains
**© Alexandros Pallaris, all rights reserved**, included here in support of that research
collaboration. If you are the upstream author and want the licensing, attribution, or the
existence of this repository changed in any way — say the word and it will be done.

## What is new in this repository (MIT-licensed, see LICENSE)

- The measured-data bug fix in `postProcessing.py` (transposed reference indices) and its
  unit test `Scatt3D/test_bvector_fix.py`
- The solver rework in `scatteringProblem.py` (factorize-once-per-frequency, MUMPS BLR
  option, anchor-LU + FGMRES sweep mode)
- `Scatt3D/measurement_diagnostics.py` + `Scatt3D/test_measurement_diagnostics.py`
- The `bench/` harness, `docs/` (diagnosis report + solver documentation), CI workflow,
  and this repository's packaging/README

## Diff transparency

The `Scatt3D/` directory intentionally preserves the upstream file layout so that
`diff -r` against upstream `243c367` shows exactly what changed and nothing else.
