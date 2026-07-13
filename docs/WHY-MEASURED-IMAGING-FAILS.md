# Why Measured-Data Imaging Fails on Scatt3D — Ranked Diagnosis and Fix List

**Prepared for:** Alexandros Pallaris (Lund University, advisor Daniel Sjöberg)
**Subject:** EuCAP 2025 / URSI EMTS 2025 linearized S-parameter imaging pipeline (repo `Wojoxiw/Scatt3D`)
**Date:** 2026-07-12

## TL;DR

Simulated data reconstructs a clean defect image; real measured data reconstructs pure noise, even restricted to well-matched frequencies, transmission-only, or reflection-only subsets. Failing everywhere uniformly is the signature of a **systematic model/calibration mismatch**, not insufficient SNR. Three actions this week:

1. **Confirm the b-vector reciprocity bug is gone and re-run.** `postProcessing.py:791` differenced `S_dut[m,n]` against the *transposed* `S_ref[n,m]` — invisible with numerically-reciprocal simulated data, but it injects raw VNA forward/reverse asymmetry into every transmission row of real data. Already fixed in this repository — use it as your new baseline, not upstream `master`.
2. **Run `Scatt3D/measurement_diagnostics.py`** (new on the same branch): reciprocity floor vs. ΔS signal, old-bug impact quantifier, and a TSVD range-space projection test that tells you whether the surviving noise is model-mismatch (fixable) or SNR (needs more contrast/less drift).
3. **Plot calibrated-measured vs. simulated S-parameters** for the reference object — magnitude AND phase, all 16 channels, vs. frequency and rotation angle. The single most informative diagnostic here, and cheap.

Everything else — calibration architecture, material contrast, the un-modeled optical table, protocol drift, inversion hygiene — is real and worth doing, but should be run in the order given: cheap decisive tests before expensive ones.

## How to read this

Ranked by **(probability of being a real cause) × (cost to check)**. Tier 1 = high-probability, near-free. Tier 6 = legitimate refinements to apply *after* the data lands in the model's range space — no regularization tuning fixes a forward-model mismatch. Citations marked CONFIRMED/PLAUSIBLE by an adversarial verification pass; unsourced claims marked `[general knowledge]`. Four corrections from that pass are reflected throughout: an EuCAP 2022 calibration paper is correctly attributed to Origlia et al. (not Meaney); a claimed Monte-Carlo antenna-position study in a 2026 Sensors paper was dropped (doesn't exist); Fhager et al. 2013 is cited for antenna-*length* modeling error, not Gaussian position perturbations; and the `maxRefl` frequency gate is described accurately — in the measured path, the "S_ref" it gates against is already the measured reference, not simulated, so it's not an independent simulated-vs-measured check.

---

## TIER 1 — Confirmed code bugs & decisive cheap diagnostics (this week)

### 1.1 The b-vector reciprocity bug

**What:** `postProcessing.py:791`: `b[i] = S_dut[nf,m,n] - S_ref[nf,n,m]` — reference term indexed with m,n **swapped** vs. the DUT term. The sensitivity kernel `q ∝ dot(E_n, E_m)` (`scatteringProblem.py:1135`) is symmetric under m↔n, making this the *only* reciprocity-sensitive line in the pipeline.

**Why:** FEM-simulated S is reciprocal to machine precision, so the swap is a no-op in simulation — exactly why "fully simulated → clean image." Real VNA S is only approximately reciprocal (switch asymmetry, cable flex, drift between forward/reverse sweeps), so every transmission row differences the DUT's forward path against the reference's *reverse* path — discarding the common-mode cancellation ΔS depends on and replacing it with raw hardware asymmetry. Explains why transmission-only still failed. A no-op for reflection rows (m==n) — consistent with, not contradicted by, reflection-only also failing.

**Check/fix:** Already fixed in this repository (`HunterSpence/Scatter3D-Claude`, `main`). Confirm the fix and re-run real-data reconstruction from it; expect transmission-channel noise to drop noticeably if this was a major contributor.

**Source:** First-hand audit, `postProcessing.py:791` vs. `scatteringProblem.py:1135/1145` (CONFIRMED).

### 1.2 Run `measurement_diagnostics.py` — the decisive discriminator

**What:** New tool on the fix branch computing: reciprocity floor vs. ΔS signal, an old-bug impact quantifier, per-channel complex calibration factors (`c_mn = S_sim_ref/S_meas_ref`, exported `.npz`), and a TSVD range-space projection test.

**Why:** For `Ax=b`, TSVD reconstructs `x_k = Σ(u_i^H b/σ_i)v_i` — whether this can work depends on whether b lies (up to noise) in `span{u_1..u_k}` (the discrete Picard condition). Projecting the working simulated ΔS, the real measured ΔS, and a pure-noise vector (scaled to your repeat-measurement floor) onto the retained subspace and comparing energy fractions `‖P_k b‖²/‖b‖²` cleanly separates "data isn't in the model's range space" (model-mismatch — no regularization fixes this) from "data is in range space but drowned in noise" (SNR/averaging problem).

**Check/fix:** Run against existing collected data before writing anything new. If measured-data energy ≈ noise-only energy, stop tuning regularization and go straight to Tiers 2–5.

**Source:** Tool in this repository (first-hand); Hansen, P.C., *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM 1998 (CONFIRMED — Picard condition/TSVD filter factors).

### 1.3 Overlay calibrated-measured vs. simulated S-parameters — the single most informative plot

**What:** For the reference object only, plot `|S_mn|` (dB) and unwrapped phase (degrees), measured vs. simulated, all 16 channels, full 5–7 GHz sweep, several rotation angles. Do this before touching DUT data.

**Why:** Every calibration paper surveyed validates exactly this way. Persistent disagreement (>~3 dB / >~20°, outside deep nulls) after calibration implicates the forward model (antenna feed, mesh, port impedance); good agreement but a still-noisy DUT reconstruction points to contrast/environment (Tier 3–4) instead. Fastest way to separate a calibration bug from a solver/mesh bug — both explain "sim works, real data is noise."

**Check/fix:** 4×4 grid of overlays; flag channels exceeding a simple error threshold. Cross-check with a quick λ/4→λ/6→λ/8 mesh-refinement on simulated-only data to rule out discretization error.

**Source:** Ostadrahimi, M., Mojabi, P., Zakaria, A., LoVetri, J. et al., "Analysis of Incident Field Modeling and Incident/Scattered Field Calibration Techniques in Microwave Tomography," IEEE AWPL vol. 10, pp. 900–903, 2011, https://ieeexplore.ieee.org/document/6008622 (CONFIRMED); Kasper, M. et al., "S-parameter calibration procedure for multiport microwave imaging systems," arXiv:1905.00963 (CONFIRMED); Geffrin, Sabouroux, Eyraud, Inverse Problems 21, S117 (2005) (PLAUSIBLE).

### 1.4 Validate the CSV→S-parameter port mapping via reciprocity

**What:** `compileMeasuredSs` (`postProcessing.py:518-573`) unpacks S-parameters via `S[l][i][j] = Sdata[i+j*4, l]` — a hard-coded, unverified assumption about VNA export column order. No header check or reciprocity/passivity assertion exists anywhere.

**Why:** If the real export order differs, transmit/receive pairs are silently permuted — indistinguishable from "pure noise," and compounds badly with 1.1.

**Check/fix:** Nearly free — compute `S_meas[m,n]` vs. `S_meas[n,m]` for the reference object across all channels/frequencies. If the pattern of large/small values doesn't match physical intuition, suspect the column mapping, not just calibration.

**Source:** First-hand audit, `postProcessing.py:550` — flagged unverified in three independent passes (CONFIRMED as a code fact; mapping correctness itself unverified).

---

## TIER 2 — Calibration (the structural gap)

### 2.1 Per-channel complex calibration from the reference pair

**What:** Compute `c_mn(f) = S_sim,ref,mn(f)/S_meas,ref,mn(f)` per antenna pair and frequency (you already have both halves), apply multiplicatively to measured DUT data before differencing.

**Why:** Model each channel as `S_meas = H·S_true` (H = the unknown transfer function of cable/connector/feed mismatch/reference-plane offset, common to both scans). Differencing cancels additive artifacts but **not** a multiplicative H: `ΔS_meas = H·ΔS_true`. The pipeline has zero amplitude calibration and only a constant phase correction (2.2), so H rides straight into the SVD input. `c_mn = 1/H` by construction, so applying it recovers `ΔS_true` exactly.

**Check/fix:** Plot `|c_mn(f)|`/`arg(c_mn(f))` — should be smooth vs. frequency if it's a fixed instrumentation factor. Apply and re-run; if the defect appears, hypothesis confirmed. If `c_mn` varies with rotation angle, that's mechanical (Tier 5), not fixed calibration.

**Source:** Ostadrahimi et al. 2011 (CONFIRMED — "scattered-field calibration"); Fresnel single-complex-factor calibration (PLAUSIBLE); Kasper et al. arXiv:1905.00963 (CONFIRMED — ECal + unknown-thru variant).

### 2.2 What ΔS subtraction does and doesn't cancel

**What:** Current pipeline applies only a constant, non-dispersive per-antenna phase offset (`postProcessing.py:552-560`); the dispersive/linear branch exists but is dead, gated `if(False)` (line 555). **No amplitude calibration anywhere.**

**Why:** Additive common-mode errors cancel in `S_dut − S_ref`; multiplicative errors don't. A few-mm cable-length mismatch produces a phase error growing linearly across 5–7 GHz — not removable by a constant-only correction, and not fixable by frequency-subsetting. Using each antenna's own S_ii (which carries real defect-reflection signal) as calibration ground truth confounds signal with calibration — a specific, testable reason reflection-only also failed, despite 1.1 being a no-op for reflection rows.

**Check/fix:** Re-derive the linear phase term properly, or replace the whole scheme with 2.1's `c_mn`. Add an explicit per-channel amplitude scale factor.

**Source:** First-hand audit, `postProcessing.py:519-571` (CONFIRMED); Ostadrahimi et al. 2011 (CONFIRMED).

### 2.3 Reference-plane mismatch

**What:** A VNA cal (even full SOLT/TRL/ECal) only guarantees accuracy at the calibration-standard plane — typically the cable end. Everything beyond it (connector, feed, 3D-printed holder) imposes an unknown transfer function the cal never touches, and the FEM's idealized port is a different reference plane entirely.

**Why:** With custom 3D-printed feeds, this is exactly where the simulated feed model is most likely to diverge from the real launch — the textbook reason raw measured S can't be compared to simulation without a transfer step (what 2.1's `c_mn` corrects for).

**Check/fix:** Confirm where the cal plane sits vs. where the FEM defines its port; move the cal plane (ECal/port extension) to match, or correct fully via `c_mn`.

**Source:** Keysight/Agilent AN 1287-3, "Applying Error Correction to Network Analyzer Measurements" (CONFIRMED); Kasper et al. (CONFIRMED).

### 2.4 How every successful system does this

**What:** Fresnel (Geffrin, Sabouroux, Eyraud): anechoic chamber + incident-field calibration. Dartmouth (Meaney, Paulsen): monopole antennas in a lossy coupling liquid, cal plane engineered to match the CAD model's port. Manitoba (Ostadrahimi, Mojabi, Zakaria, LoVetri, Gilmore): both incident- and scattered-field calibration, publishing raw + calibrated data so inversion can be tested against calibration in isolation. PoliTo (Origlia, Rodriguez-Duarte, Tobon Vasquez, Vipiana, EuCAP 2022): simulated-vs-measured S-matching calibration, structurally similar to 2.1's `c_mn`.

**Why:** Every one has (a) differential measurement (already present here), (b) an explicit calibration standard/procedure, and (c) a controlled or explicitly-modeled environment (Tier 4). This setup has only (a).

**Check/fix:** Implement 2.1 first (cheapest); consider moving the antenna feed reference plane to match the FEM port (Dartmouth-style) as a permanent fix.

**Source:** Ostadrahimi et al./Gilmore et al. (CONFIRMED); Geffrin & Sabouroux (PLAUSIBLE); Meaney/Paulsen body of work, PMC3757128/PMC3759252 (PLAUSIBLE — the coupling liquid's primary published rationale is impedance matching/penetration, multipath suppression at most secondary — don't overclaim); Origlia, Rodriguez-Duarte, Tobon Vasquez, Vipiana, EuCAP 2022, IEEE doc 9769081 (CONFIRMED, corrected authorship — not Meaney).

---

## TIER 3 — Signal magnitude & materials

### 3.1 Contrast reality check: solid PLA ≈ POM

**What:** POM reported roughly εr 2.8–3.0 at GHz (exact table cell not independently pulled — approximate); solid/high-infill 3D-printed PLA measures εr≈2.75–2.96 — **solid PLA and POM are near dielectric twins**. Effective PLA permittivity drops substantially with lower infill (roughly 2.4–2.5 at 20% infill down further at sparser infill; numbers vary by print settings/pattern).

**Why:** The assumed sim contrast (Δεr≈0.07, ~2.6%) is already small. If the real insert is solid/high-infill, true contrast could be near zero; if low-infill, contrast could be larger but qualitatively different — an unmodeled air-gap at the print/bore interface rather than a smooth material swap. Either breaks the assumption behind the simulated "success." A thin high-contrast air shell at an interface is a classic linearization-breaker for first-order Born even at modest volume-averaged contrast.

**Check/fix:** Determine actual slicer settings (infill %, shells, pattern), or back it out by weighing the part vs. solid-CAD volume. Measure a coupon if possible. Compare assumed vs. real Δεr — a >2× difference matters at this signal level.

**Source:** Felicio, J.M., Fernandes, C.A., Costa, J.R., IEEE doc 7843900 (CONFIRMED); infill-vs-permittivity trend corroborated across two independent sources (CONFIRMED trend; exact numbers vary, don't over-quote); Riddle, Baker-Jarvis, Krupka, IEEE Trans. MTT 51(3):727-733, 2003 (PLAUSIBLE).

### 3.2 Expected |ΔS| vs. the reciprocity/drift noise floor

**What:** Weak-scatterer estimate: Δεr≈0.15, ~1 cm³ defect, k≈126 rad/m at 6 GHz → fractional perturbation `k²·V·Δεr/(4π)` lands around −75 to −115 dB relative to nominal channel level. Comparable to or below realistic VNA dynamic range/repeatability — and far below what session drift and mm-scale positioning error can hold steady.

**Why:** Explains the "every mitigation failed identically" pattern exactly as expected if true signal is simply smaller than a broadband, all-paths error floor.

**Check/fix:** **Decisive test:** re-measure the reference object twice with a full remove-replace-and-rotate cycle between, touching nothing else. If `|S_repeat2 − S_repeat1|` ≳ predicted `|ΔS_defect|`, the defect is invisible to this rig regardless of other fixes — the fix is experiment design (Tier 5), not software.

**Source:** First-principles estimate `[general knowledge]`, cross-checked against the paper's own λ/10 resolution claim (Tier 6.1); permittivity sources as 3.1.

### 3.3 Validation ladder: metal rod → air hole → PLA insert

**What:** Every working system validates on a high-contrast, easy target before a weak dielectric defect. This jumped straight to one of the hardest cases (near-twin dielectrics).

**Why:** A metal target's scattered field sits orders of magnitude above the noise/drift/calibration floor. If that case also fails, the bug is definitively in geometry/port-mapping/calibration — not sensitivity. Cleanly separates "pipeline broken" from "sensitivity too low."

**Check/fix:** Fabricate a metal rod/foil cylinder sized like the current defect, same location, re-run the identical pipeline. If it works, step down in contrast before returning to POM/PLA.

**Source:** Chiu, C.-C. et al., Int. J. RF Microwave CAE 22(5), 2012, DOI:10.1002/mmce.20623 (CONFIRMED); Fresnel database graded-difficulty design (PLAUSIBLE).

---

## TIER 4 — Un-modeled physics

### 4.1 The optical table is an unmodeled near-perfect metal mirror at 6 GHz

**What:** Surface-impedance calc (Rs=√(πfμ0μr/σ)) gives ≥98% power reflectivity at 6 GHz for typical table constructions (stainless ~99.9%, carbon-steel ~98.9%). Styrofoam is RF-transparent (εr≈1.03–1.06) and shields nothing. `meshMaker.py`'s only domain options (`'domedCyl'`, `'sphere'`, lines 159-182) are closed, PML-truncated free-space domains — no table/floor/ground plane anywhere.

**Why:** Simulated E_ref (and the sensitivity kernel) is computed in idealized free space; real fields are dominated by a strong close-range reflector. Since the object rotates relative to the fixed table, object–table coupling changes per angle, injecting an angle-dependent systematic component absent from the kernel — matching "all mitigations failed identically."

**Check/fix:** Compare reference-object-only real vs. FEM-simulated S-parameters directly (before ΔS) — a substantial mismatch even here confirms environment, not DUT contrast, as primary. Physically test with RF absorber under/around the riser. Cheaper: time-gate (4.2).

**Source:** Surface-impedance calc `[general knowledge/textbook EM]`; corroborated by mesh-geometry code, `meshMaker.py:159-182` (CONFIRMED code fact).

### 4.2 Room multipath: absorber, time-gating, or model the table

**What:** Two valid strategies: suppress physically (absorber, or a lossy coupling medium as at Dartmouth) or model explicitly (as Manitoba does for its chamber). This setup does neither.

**Why:** Static, angle-independent background (antenna–table–antenna, no object) *does* cancel in `S_dut(θ)−S_ref(θ)` at matching angles — a real strength, don't over-fix this. What doesn't cancel: object-state-dependent coupling (the object's near-field interacting with the table, differing between ref/DUT because their local fields differ near the defect) — absent from the idealized model.

**Check/fix:** Time-gate a subset (IFFT, window the direct path, IFFT back) — improvement implicates multipath. Measure the empty fixture at all angles to isolate background; check whether reconstruction "noise" has periodic structure tied to the rotation-stage period.

**Source:** Keysight/Agilent AN 1287-12, "Time Domain Analysis Using a Network Analyzer" (correct citation for gating, replacing an earlier misattribution to AN 1287-3); Manitoba chamber-modeling, Ostadrahimi/Gilmore et al. (CONFIRMED); ISAR zero-Doppler clutter literature `[adjacent-domain analogue, self-flagged]`.

### 4.3 Rotation breaks the environment-invariance assumption

**What:** Rotating the target with fixed transceivers is the standard turntable/ISAR duality to a synthetic antenna ring — valid only if the surrounding scene is absent or rotationally symmetric. Here the table, foam, and holders are fixed and azimuthally asymmetric.

**Why:** Angle-matched-per-session clutter cancels (4.2's strength); object-state-dependent near-field coupling to the fixed environment doesn't, and is absent from a PML-truncated free-space sim — producing angle-dependent signal outside the model's range space (ties to 1.2).

**Check/fix:** Check whether reconstruction "noise" correlates with the rotation-stage mechanical period rather than being unstructured.

**Source:** ISAR/turntable duality `[general knowledge]`; free-space differential imaging literature `[adjacent-domain, self-flagged]`.

### 4.4 PLA antenna holders in the reactive near field

**What:** PLA (εr≈2.7–2.96, loss tangent ~0.01–0.05) sits within a wavelength (~5 cm at 6 GHz) of the radiating aperture — inside the reactive near field where dielectric loading shifts resonance, match, and phase center.

**Why:** If the FEM omits the holder or guesses its permittivity, simulated near-field is systematically wrong for every voxel/channel simultaneously — not fixable by frequency-subsetting.

**Check/fix:** Confirm the FEM mesh includes actual holder geometry and a measured (not guessed) PLA permittivity. Ablate: simulate antenna S11 with/without the holder, compare shift to the ~−75 to −115 dB signal level (3.2).

**Source:** Catarinucci et al., IET Microw. Antennas Propag., 2017 (PLAUSIBLE); Felicio et al. (CONFIRMED).

---

## TIER 5 — Protocol & hardware

### 5.1 VNA drift, warm-up, interleaving

**What:** VNA drift with cable/connector/component temperature after cal; published thermal-drift studies show stabilization only after ~30 min–3 h warm-up (most relevant study is mmWave/THz, not 5–7 GHz benchtop — treat as an analogue, not a hard number).

**Why:** Given 3.2's estimate of sub-0.01%-amplitude / fraction-of-a-degree stability needed between sessions, ordinary bench drift over a swap-and-rewarm interval is a serious candidate, and drift doesn't average out with more sweeps.

**Check/fix:** Re-measure the reference object's full sweep twice back-to-back with a matching time gap, untouched — isolates drift from repositioning (same test as 3.2). If significant, interleave DUT/reference measurements per angle instead of two separate sweeps.

**Source:** Bystrov, Wang, Gardner et al., MDPI Metrology 2(2), 2022 (CONFIRMED content, different hardware regime); Eyraud, Geffrin, Litman, Sabouroux, Giovannini, Applied Physics Letters 89, 244104 (2006) (CONFIRMED).

### 5.2 Positioning/rotation-stage tolerance vs. λ/20

**What:** At 5–7 GHz, free-space λ≈43–60 mm (λ/20≈2.1–3.0 mm); inside POM (εr≈2.9), λ≈25–35 mm (λ/20≈1.3–1.8 mm). A ~1 mm stage/jig error (matching the stated fabrication tolerance) gives ~6–8° two-way phase error per mm — comparable to or larger than the defect's expected perturbation.

**Why:** Unlike drift, this is genuinely angle-dependent — changes shape every 15° step, not absorbable by one calibration constant. Combined with non-repeatable ref/DUT swap placement, directly violates the differencing design's assumption of identical geometry between scans.

**Check/fix:** Rotate the reference object through a full sweep, return to 0°, re-measure — bounds stage repeatability directly. Verify swap uses a repeatable jig, not manual placement.

**Source:** Bourqui, J., Sill, J., Fear, E., Int. J. Biomedical Imaging, 2012, PMC3348648 (CONFIRMED, corrected author list); Newell, M.H., IEEE Trans. Antennas Propag. 36(6), 1988 (CONFIRMED); "Low-Cost Turntable Designed for RF Phased Array Antenna Active Element Pattern Measurement," arXiv (CONFIRMED).

### 5.3 Antenna as-built vs. as-designed: the author's own sims already show this breaks reconstruction

**What:** `runScatt3D.py`'s own experiments include `testRunDifferentDUTAntennas` (logged "unsuccessful reconstruction") and `forPaper` sweep entries `'patch2percentsmaller'`, `'patchepsr4.2'` — 2% antenna-model perturbations, tested in simulation.

**Why:** In-repo evidence, not literature inference: reconstruction collapses under perturbations far smaller than a real ~1 mm tolerance across four separately-printed holders would produce. Directly corroborates 2.3, 4.4, 5.4.

**Check/fix:** Re-run these existing sim experiments as a quantitative sensitivity study — at what perturbation magnitude does the clean image degrade to noise? That's your real antenna-model accuracy budget, and it's cheap since the code already exists.

**Source:** First-hand audit, `runScatt3D.py` (CONFIRMED, in-repo fact).

### 5.4 Per-antenna normalization and Zm impedance mismatch

**What:** (a) Port-mode field `Ep` normalized using only antenna 0's face integral (`scatteringProblem.py:628-639`), applied uniformly to all 4 antennas, assuming identical fabrication. (b) `Zm` for `'6GHz measurement'` hard-coded to `eta0/sqrt(2.1*(1-0.01j))` (`scatteringProblem.py:954`) — an assumed dielectric's wave impedance (~260 Ω), not the real 50 Ω VNA reference, and not derived from actual coax geometry.

**Why:** (a) can't represent real per-antenna coupling variation given a ~1 mm tolerance across 4 holders. (b) means the FEM's own generalized S-parameters — including the `compileMeasuredSs` phase fit and `maxRefl` gate — reference a different impedance than the physical VNA, a systematic offset independent of fabrication tolerance.

**Check/fix:** Lower priority than Tiers 1–4 (systematic biases, not obviously "pure noise" on their own) — fix once bigger issues resolved: derive Zm from actual coax geometry; consider per-antenna normalization if holder variation is measurable.

**Source:** First-hand audit, `scatteringProblem.py:628-639`, `:949-954` (CONFIRMED).

---

## TIER 6 — Inversion hygiene

### 6.1 Truncation-rank selection has no ground-truth-free method

**What:** Primary solver (`np.linalg.pinv(A_ap, rcond=rcond)`, `postProcessing.py:1040-1047`) uses hard-coded `rcond=10**-1.85`, tuned once via `reconstructionError()` (lines 79-125), which needs the ground-truth `epsr_dut` — usable only in simulation.

**Why:** Optimized against noiseless sim data, reused verbatim for real data without re-deriving against the real noise floor. Per 1.2's logic: if data isn't in the model's range space at all, no truncation choice fixes that — apply this only after 1.2 confirms above-noise energy exists.

**Check/fix:** Ground-truth-free selection: Morozov discrepancy (needs a noise estimate, e.g. from 3.2/5.1), L-curve (no noise estimate, robust to correlated errors, ambiguous corner risk), or GCV (automatic, can misbehave under correlated noise). Cross-check the knee against the paper's own ~λ/10 resolution claim.

**Source:** Hansen, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM 1998 (CONFIRMED); Hansen & O'Leary, SIAM J. Sci. Comput. 14(6), 1993 (CONFIRMED, corrected co-author); Pallaris & Sjöberg, EuCAP 2025 (CONFIRMED).

### 6.2 Row weighting/whitening across reflection vs. transmission

**What:** Plain TSVD is optimal only with comparable, uncorrelated per-row noise. Reflection rows: large-magnitude/high-SNR, low defect information. Transmission rows: more defect information, worse SNR. The already-tried transmission-only/reflection-only subsetting is a blunt version of proper weighting.

**Why:** Un-normalized stacking lets the largest-magnitude rows dominate the SVD basis — not necessarily the most defect-informative ones. Evaluate after 1.2's projection test, since whitening changes what "in range" means.

**Check/fix:** Estimate per-channel noise floor from repeat measurements (already collected per 3.2/5.1); build `W=diag(1/σ_i)`, apply to A and b (or use relative scaling `ΔS_ij/|S_ij,ref|`, standard in CSI/DBIM). Re-run 1.2 on the whitened system.

**Source:** Aster, Borchers, Thurber, *Parameter Estimation and Inverse Problems*, Elsevier `[general knowledge, standard]` (CONFIRMED).

### 6.3 Tikhonov/positivity/support constraints — helpful, not a substitute for fixing the model

**What:** Tikhonov (`σ_i²/(σ_i²+λ²)` filters) is a smoothed TSVD — changes the noise/resolution tradeoff, not whether data is explainable by the model. Support constraints (restrict to the known object volume/insert region) can genuinely help here since object geometry is fully known (fabrication QA, not blind discovery).

**Why:** Doesn't explain the failure — it's hygiene. If 1.2 shows measured ΔS has little energy in the model's range space, no smoothing/constraint rescues it; a still-noisy constrained result confirms model mismatch, not insufficient priors.

**Check/fix:** Try Tikhonov (L-curve λ) plus a hard support constraint; if the result doesn't materially change from plain TSVD on the same failing data, stop tuning and revisit Tiers 1–5.

**Source:** Hansen, *Rank-Deficient and Discrete Ill-Posed Problems* (CONFIRMED — Tikhonov/TSVD equivalence).

### 6.4 Inverse-crime-free robustness testing

**What:** The "simulation works" test almost certainly uses the same mesh/element order/PML/antenna model to both generate synthetic data and build the inversion Jacobian — the textbook inverse crime (Colton & Kress). A clean recovery proves internal consistency, not real-world robustness.

**Why:** Directly explains "perfect on simulated DUT, noise on real data" without anything exotic — the test was structurally incapable of catching modeling error.

**Check/fix:** Perturb ONE variable only in the data-generating simulation (1 mm antenna shift, different mesh, air-gap shell, antenna orientation), keep the inversion model fixed, add realistic noise, re-run the same TSVD pipeline. Rank which perturbation kills the image fastest — quantifies, cheaply and in silico, how much each Tier 3–5 hypothesis actually matters here.

**Source:** Colton, D., Kress, R., *Inverse Acoustic and Electromagnetic Scattering Theory*, Springer (CONFIRMED); Wirgin, A., "The inverse crime," arXiv:math-ph/0401050 (2004) (CONFIRMED); Kaipio, Somersalo, *Statistical and Computational Inverse Problems*, Springer 2005 `[well-established]`.

---

## Appendix A — Code landmines

| Landmine | Location | Status |
|---|---|---|
| **b-vector transposed reference indices** — `S_dut[m,n] − S_ref[n,m]` | `postProcessing.py:791` | **Fixed** in this repository; top root-cause candidate (1.1). |
| **`SparamMeas` ordering landmine** — docstring says freqs-then-angles; working code needs angles-then-freqs | `postProcessing.py:614-618` | Documentation trap, not a runtime bug per se. |
| **DUT calibration chaining** — reference phase-corrected against FEM sim; DUT then phase-corrected against the *already-corrected measured* reference, not the original sim | `postProcessing.py:617-618` | Compounds residual error into the second correction. Decouple regardless of calibration scheme used. |
| **Nearest-neighbor frequency snap**, no interpolation, warns only above 1 Hz mismatch | `postProcessing.py:538-544` | Low risk if grids align as designed — confirm they do. |
| **Zm hard-coded** to an assumed dielectric wave impedance, not 50 Ω | `scatteringProblem.py:954` | See 5.4. |
| **Single-antenna `normFactor`** applied to all 4 antennas | `scatteringProblem.py:628-639` | See 5.4. |
| **`maxRefl` frequency gate** — gates on `|S_ref[nf,m,m]|`. In the measured path, `S_ref` here is already the phase-corrected *measured* reference (reassigned earlier, see DUT-cal-chaining above) — not an independent sim-vs-measured check, and should not be described as one. | `postProcessing.py:731-734` | Corrected per adversarial verification; an earlier characterization of this line was inaccurate. |
| **CSV column mapping** `S[l][i][j]=Sdata[i+j*4,l]` — unverified VNA export-order assumption | `postProcessing.py:550` | See 1.4 — cheap to validate via reciprocity, not yet confirmed against a real export. |

## Appendix B — Full source list

**Own work / theory:** Pallaris, A. & Sjöberg, D., "Microwave Reconstruction of Fabrication Defects in Known Objects Using Scattering Parameter Sensitivities," EuCAP 2025, DOI:10.23919/EuCAP63536.2025.10999660 (2D, fully-simulated feasibility study — no measured-data validation published). Pallaris, A. & Sjöberg, D., "3D Simulation Code Using Parallel Processing...," URSI EMTS 2025, DOI:10.46620/URSIEMTS25/RXOJ6945 (3D code's only published validation is also self-simulated on both sides — origin of the data equation used throughout). Nikolova, N.K., *Introduction to Microwave Imaging*, Cambridge Univ. Press, 2017, DOI:10.1017/9781316084267, Ch. 4.4.

**Calibration:** Geffrin, Sabouroux, Eyraud, Inverse Problems 21, S117 (2005); Geffrin & Sabouroux, Inverse Problems 25, 024001 (2009); Ostadrahimi, Mojabi, Zakaria, LoVetri et al., IEEE AWPL 10:900-903, 2011, https://ieeexplore.ieee.org/document/6008622; Gilmore, Mojabi, Zakaria, Ostadrahimi et al., IEEE Antennas Propag. Mag., 2011; Kasper, Ragulskis, Gramse, Kienberger, arXiv:1905.00963; Origlia, Rodriguez-Duarte, Tobon Vasquez, Vipiana, EuCAP 2022, IEEE doc 9769081 (corrected authorship, not Meaney); Ferrero, Pisani, IEEE MGWL 2(12):505-507, 1992; Belkebir et al., IEEE TAP, 2014, DOI:10.1109/TAP.2014.2308534; Keysight/Agilent AN 1287-3 and AN 1287-12; Eyraud, Geffrin, Litman, Sabouroux, Giovannini, Applied Physics Letters 89, 244104 (2006); Kwon et al., Radio Science, 2018, DOI:10.1002/2017RS006399.

**Validation ladder / antenna fidelity:** Chiu et al., Int. J. RF Microwave CAE 22(5), 2012, DOI:10.1002/mmce.20623; Fresnel free-space scattering database, fresnel.fr/3Ddirect/database.php; Meaney, Paulsen et al. Dartmouth body of work, PMC3757128, PMC3759252; Bourqui, Sill, Fear, Int. J. Biomedical Imaging, 2012, PMC3348648; Newell, IEEE Trans. Antennas Propag. 36(6), 1988; "Low-Cost Turntable Designed for RF Phased Array Antenna Active Element Pattern Measurement," arXiv.

**Materials:** Riddle, Baker-Jarvis, Krupka, IEEE Trans. MTT 51(3):727-733, 2003; Felicio, Fernandes, Costa, IEEE doc 7843900; "Evaluating the Relationship Between Relative Permittivity and Infill Density in 3D Printed Dielectric Slabs," IEEE doc 10838558; "Evaluation of Relative Permittivity and Loss Factor of 3D Printing Materials for RF Applications," Processes (MDPI) 10(9):1881, 2022; Catarinucci et al., IET Microw. Antennas Propag., 2017.

**Solver background (cross-reference — see companion solver document):** Ernst, Gander, "Why it is Difficult to Solve Helmholtz Problems with Classical Iterative Methods," 2012; Hiptmair, Xu, SIAM J. Numer. Anal. 45(6), 2007; Fressart et al., arXiv:2507.13066 (2025); Bootland, Dolean, Nataf, Tournier, J. Sci. Comput. 105, 67 (2025), arXiv:2311.18783; hypre GitHub issue #152; PETSc `MATSOLVERMUMPS` docs (ICNTL(35)/CNTL(7)).

**Inversion math:** Hansen, *Rank-Deficient and Discrete Ill-Posed Problems*, SIAM 1998; Hansen & O'Leary, SIAM J. Sci. Comput. 14(6), 1993; Colton, Kress, *Inverse Acoustic and Electromagnetic Scattering Theory*, Springer; Wirgin, arXiv:math-ph/0401050 (2004); Kaipio, Somersalo, *Statistical and Computational Inverse Problems*, Springer 2005; Aster, Borchers, Thurber, *Parameter Estimation and Inverse Problems*, Elsevier.

**First-hand code audit (this repo):** `Wojoxiw/Scatt3D`: `postProcessing.py` (lines 79-148, 519-618, 731-794, 1040-1276), `scatteringProblem.py` (lines 628-639, 949-993, 1042-1150), `meshMaker.py` (lines 159-182, 587-673), `runScatt3D.py` (`testRunDifferentDUTAntennas`, `forPaper` sweep) — audited against upstream `master`; fixes in this repository (b-vector fix, `measurement_diagnostics.py`, MUMPS BLR/anchor-LU+FGMRES solver work).
