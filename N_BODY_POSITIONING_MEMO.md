# N-body literature-positioning memo

**Purpose:** position the low-pericenter causal-handle result in the stellar-dynamics literature
**without overclaiming.** Not paper prose. No `paper.tex` edits, no new simulations.
**Reference caveat:** the works in §6 were grounded via a literature search (June 2026); authors,
years, and venues are as confirmed there, but **exact volume/page/DOI must be verified on NASA ADS
before any go into the manuscript.** This is a positioning scaffold, not a final bibliography.

---

## 1. What the paper actually proves

- **An interventional, not correlational, design.** From one initial condition we build matched
  arms — speed-preserving velocity rotations (radialize toward centre; shell-local variants;
  tangentialize; v_t-rescale) plus a magnitude-matched random-rotation **sham** — integrate all to a
  common horizon, and read effects **within pair as (arm − sham)**. KE/virial drift held < 10⁻²;
  paired-bootstrap CIs; profile-correct pericenter potential. *This is the methodological core.*
- **Low-pericenter dose predicts inner-mass response.** The t₀ increase in the low-pericenter
  fraction, Δf_peri(<r_c), causally predicts the peak central-mass response ΔM(<r_c): per-pair
  regression slope CI excludes 0 in every concentrated cell; arm-mean corr 0.86–0.99. The chain is
  **time-ordered** (M significant at ~5 steps, C₈ at ~10) and **scale-matched**.
- **Profile dependence decomposes as dose × slope.** Plummer (core) is ~2× stronger than Hernquist
  (cusp); at r_c=0.1 this is **dose ×1.44 × slope ×1.36 = ×1.96**, the measured response ratio
  exactly. The effect is **radius-specific** (×1.60 / ×1.96 / ×1.22 at r_c = 0.05 / 0.10 / 0.20,
  peaking near r_c ≈ a/2) — **not** a scale-free law.
- **Negative controls.** uniform3d and bimodal3d show the mechanism *absent*: ~5× smaller response
  and a non-positive low-pericenter predictor. This rules out "any radialization concentrates."
- **Kill-test-survived specificity.** Across a 34-cell battery (Hernquist + Plummer, N=512–4096,
  four softenings, 100 pairs/cell), 25/26 concentrated cells pass and **zero kill tests fire**:
  global angular-momentum summaries and outer-shell responses do **not** carry the signal as cleanly
  as low-pericenter dose; the sham is null; N-robust to 4096 (response ratio N=4096/N=512: Hernquist
  ×0.92, Plummer ×1.01) and the N=4096 cell reproduces an independent pilot exactly.
- **Disclosed borderline (state it maturely).** *"One concentrated cell narrowly missed the
  preregistered locality ratio threshold, 2.77 versus 3.0, while passing all other criteria and
  triggering no kill test. We therefore report the strict Boolean result separately from the
  scientific verdict."*

## 2. What it does NOT prove (scope guards)

- **Not a loss-cone model.** No central sink / SMBH, no true loss-cone boundary, no
  angular-momentum diffusion into a sink, **no TDE or EMRI rates.** The response here is *fast and
  collisionless* (orbit rearrangement over a few dynamical times), not relaxation-driven repopulation.
- **Not a collisional-relaxation / Fokker-Planck study.** Finite-N relaxation is *controlled for*
  (matched arms at equal N) and *bounded* (N-robustness test), not modelled.
- **Not cosmological.** Stylized isolated profiles, softened direct summation, N ≤ 4096; no halos,
  mergers, or baryonic physics.
- **No thermodynamic / N→∞ statement.** And the anisotropy claim is **scoped**: *"in this controlled
  battery, global angular-momentum summaries and outer-shell responses do not carry the causal signal
  as cleanly as low-pericenter dose"* — **not** "anisotropy is never causal in stellar dynamics."

## 3. Literature bridges (neighbours, not claimed applications)

- **Velocity anisotropy ↔ central structure / anisotropic models.** Osipkov–Merritt models (Osipkov
  1979; Merritt 1985) are isotropic in the core and radially anisotropic in the envelope — the
  natural radial-anisotropy state our interventions perturb. Our result *separates cause from
  signature*: anisotropy is largely a downstream summary here, while the low-pericenter dose is the
  handle. Bridge, not equivalence.
- **Radial-orbit instability (ROI).** Polyachenko & Shukhman (1981); Merritt & Aguilar (1985);
  Barnes (1985). Our radializing intervention pushes toward radial anisotropy, the same axis ROI
  concerns — but ROI is a *global symmetry-breaking instability*, whereas ours is a *controlled,
  dose-graded causal handle on central mass*. Useful contrast: distinguish "instability of radial
  systems" from "central-concentration response to added radial/low-pericenter orbits."
- **Core collapse / gravothermal evolution.** Antonov (1962); Lynden-Bell & Wood (1968); Cohn
  (1980); Spitzer (1987). Both concern growth of central concentration — but gravothermal collapse is
  *slow and collisional* (~15 relaxation times), while our effect is *fast and collisionless*. Frame
  as a complementary, mechanism-level handle, not a collapse model.
- **Loss-cone dynamics — ANALOGY ONLY.** Frank & Rees (1976); Lightman & Shapiro (1977); Cohn &
  Kulsrud (1978); Magorrian & Tremaine (1999); Merritt (2013). Both concern the *population of
  low-pericenter orbits*. *"The mechanism is adjacent to loss-cone thinking because both concern the
  population of low-pericenter orbits, but our setup does not include a central sink or a true
  loss-cone boundary."* Do **not** claim loss-cone repopulation, rates, or empty/full-cone regimes.
- **Cusp–core / orbital structure.** Plummer (1911); Hernquist (1990); Dehnen (1993); NFW (1996,
  1997); de Blok (2010, review). Our profile dependence (core vs cusp) connects to cusp–core
  structure: the handle's leverage differs between a flat core and a 1/r cusp, decomposing into dose
  headroom (deep-pericenter accessibility) and a potential-shape response slope.

## 4. Best astrophysical framing

> **Causal isolation of low-pericenter accessibility as the driver of inner-concentration response**
> — i.e. controlled tests of whether central concentration is driven by *orbit-family accessibility*
> rather than by *downstream anisotropy summaries*.

Loss-cone, core-collapse, and cusp–core become **motivating neighbours**, not claimed applications.
The two genuinely novel contributions to lead with: (i) the **interventional matched-pair design**
(counterfactual causal isolation in collisionless N-body — rare in stellar dynamics, where most work
is correlational or instability-based); (ii) the **dose × slope decomposition** of profile
dependence at a target radius.

## 5. Journal fit

| journal | fit | what it would require |
|---|---|---|
| **New Astronomy** | **Best fit.** Controlled computational astrophysics / methodology. | Clean methods + reproducibility (have it), honest scope, the framing in §4. |
| **MNRAS** | Possible. | Strong literature positioning (this memo), polished figures (have them), a clear stellar-dynamics motivation tied to §3. |
| **ApJ** | Possible. | A sharper, *direct* astrophysical-system hook — which we deliberately avoid overclaiming; would need more than the stylized setup. |
| **ApJ Letters** | Not first choice. | Only if the result is made very short and very consequential; the result is strong but stylized. |

**Recommendation:** target **New Astronomy** first, framed as a methodological + mechanism result;
MNRAS as the fallback if the literature positioning and figures are strengthened. Do **not** lead
with ApJL.

## 6. Required references (anchor works — verify exact details on ADS)

**Profile models (the simulated systems):**
- **Plummer (1911), MNRAS** — the Plummer (cored) model; one of our two profiles.
- **Hernquist (1990), ApJ** — the Hernquist (cuspy) model; our other profile.
- **Dehnen (1993), MNRAS** — γ-models generalizing Hernquist (γ=1); for profile-family context.
- **Navarro, Frenk & White (1996; 1997), ApJ** — NFW cusp; cosmological-halo cusp–core context.

**Anisotropy & orbital structure:**
- **Osipkov (1979); Merritt (1985)** — Osipkov–Merritt anisotropic models (isotropic core, radial
  envelope); the natural anisotropy state our interventions perturb; supports "anisotropy as
  signature."
- **Polyachenko & Shukhman (1981); Merritt & Aguilar (1985); Barnes (1985)** — radial-orbit
  instability; the radial-anisotropy axis our radialization moves along (contrast, not application).

**Central concentration / collisional evolution:**
- **Antonov (1962); Lynden-Bell & Wood (1968), MNRAS; Cohn (1980), ApJ; Spitzer (1987), book** —
  gravothermal catastrophe / core collapse; complementary (slow, collisional) route to central
  concentration to contrast with our fast collisionless handle.

**Loss-cone (analogy only):**
- **Frank & Rees (1976), MNRAS; Lightman & Shapiro (1977), ApJ; Cohn & Kulsrud (1978), ApJ** —
  foundational loss-cone / low-pericenter-orbit theory; empty vs full loss cone (cite as the
  conceptual neighbour, explicitly not applied).
- **Magorrian & Tremaine (1999), MNRAS; Merritt (2013), "Dynamics and Evolution of Galactic Nuclei"
  (Princeton)** — loss-cone in real galactic nuclei; the boundary we do **not** model.

**Cusp–core & foundations:**
- **de Blok (2010), Advances in Astronomy (review)** — the cusp–core problem; motivates the
  core-vs-cusp profile dependence we quantify.
- **Binney & Tremaine, "Galactic Dynamics" (2nd ed., 2008, Princeton)** — the standard reference for
  pericenters/effective potential, anisotropy, and orbit families used throughout.

*(Methodological note: interventional/counterfactual matched-pair design appears to have little
direct precedent in collisionless stellar dynamics — worth a short literature check during writing to
confirm novelty rather than assert it.)*
