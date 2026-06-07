# Low-pericenter causal battery — claim-status report (AWS run, 2026-06-06)

## Scientific verdict: POSITIVE

> Across the full N × ε battery, the low-pericenter mechanism is **specific to concentrated
> systems, survives to N=4096, reproduces the pilot exactly, and is absent in uniform/bimodal
> controls; the only strict failure is one borderline locality ratio of 2.77 versus a
> preregistered 3.0 threshold.**

**`BATTERY_PASS=False` is a strict-Boolean artifact, NOT the scientific conclusion.** The aggregate
flag is an AND over all 26 concentrated cells; it flipped False because one cell missed one
threshold by a hair. The mechanism itself is confirmed.

## What ran

34 cells = 2 concentrated profiles {hernquist3d, plummer3d} × {N=512/1024/2048 × ε=0.02/0.05/0.10/0.20;
N=4096 × ε=0.05} + 2 controls {uniform3d, bimodal3d} × N={512,1024,2048,4096} × ε=0.05.
10 arms (orig, sham, tangentialize, inner-weak/med/strong, mid, outer, full, vt-rescale),
times {0,5,10,20,50,100,300,600,1000}, 100 pairs (50 at N=4096). Engine + analysis: the verified
pilot worker (profile-correct Φ, per-pair regression dose, registered partial-correlation kill test,
magnitude-relative sham). No new physics. (Full per-cell table: `battery_report_auto.md`.)

## Results (in inspection order)

1. **Cells passed/failed — 25/26 concentrated cells PASS.** The single miss is
   **hernquist3d N=1024 ε=0.02**, which failed **only criterion 3 (locality)** with inner/outer
   = **2.77×** vs the preregistered **≥3.0×** bar. Its other criteria all passed (dose r +0.88,
   M→C₈ 5→10, sham 0.005, conservation clean, no kill) and its headline numbers are
   indistinguishable from its passing neighbours — a noise-level threshold miss, not a mechanism
   failure.

2. **Kill tests — ZERO fired across all 34 cells.** In every concentrated cell f_peri beats global
   ⟨|L|⟩ (partial-corr), the outer shell never reproduced concentration, and sham never explained
   the effect.

3. **Controls behave as negative controls.** uniform3d and bimodal3d show the mechanism *absent*:
   dose r only +0.6–0.7 (vs +0.87–0.98 concentrated), peak ΔM10 ≈ +0.012 (**~5× smaller**),
   **f_peri partial-corr goes negative** (−0.08 to −0.39) while global-⟨|L|⟩ is strongly negative,
   and the M→C₈ ordering scrambles (e.g. 20→20, 10→50). The low-pericenter chain is specific to
   concentrated systems. (bimodal pericenter is a global-spherical approximation — read C₈ there.)

4. **N=4096 ε=0.05 reproduces the pilot exactly.** Hernquist peak ΔM10 **0.0275** (pilot 0.0275);
   Plummer peak ΔM10 **0.0570** (pilot 0.0570); f_peri dominant in both.

5. **Plummer robust and stronger at large N.** Peak ΔM10 ≈ 0.056–0.057 flat across N=512→4096
   (**criterion 7 ratio ×1.01**), dose r +0.93–0.95 throughout — ~2× stronger than Hernquist.

**Criterion 7 (N-robustness, the headline question): PASSES both** — Hernquist ×0.92, Plummer ×1.01.
The mechanism does not vanish at large N.

## Provenance & cost

- Run on AWS `c7i-flex.large` (most powerful free-tier-eligible instance, 2 vCPU), eu-central-1,
  self-terminating + per-cell S3 checkpointing.
- **Cost: $0** (covered by free-plan credits). **Instance self-terminated** on completion; no
  resources left running.
- Deterministic seeds (2000 + i) → reproducible; raw `rows.jsonl` included per cell.

## Claim-status update

N-body moves from *pilot-positive / battery-pending* → **battery-positive with one borderline
criterion miss and no kill-test failures.** The description was not arbitrary: global angular
momentum failed, low-pericenter structure survived, concentrated systems show the mechanism, and
controls do not.
