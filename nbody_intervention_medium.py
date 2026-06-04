#!/usr/bin/env python3
"""
nbody_intervention_medium.py — stress-test the N-body causal anisotropy handle
==============================================================================

The pilot found: initial velocity anisotropy causally modulates future anisotropy (~23%
memory) with energy/virial preserved.  This run tries to BREAK that handle across four axes:

  1. θ-scaling   — does the effect grow with intervention strength? (θ ∈ {5,10,20,30}°, ε=0.05)
  2. ε-robustness— does it survive softening changes? (ε ∈ {0.02,0.05,0.10}°, θ=20°)
  3. target-specificity — β vs σ_r, Q, phase-space entropy S, clustering C₈
  4. trivial-memory baseline — FREE-STREAMING (gravity off): if the imposed Δβ persists just as
     well ballistically, the "memory" is trivial, not a relaxation effect.

Matched-pair design (orig / intervention / sham), same conservation checks.  Cusps; N=1024;
100 pairs/cell.  No AWS.  Does not touch paper.tex.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init, angular_momentum_3d, total_momentum
from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, obs_coarse_var, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from coarse_grain_pilot import _spearman

OUTDIR = "outputs/nbody_intervention_medium"
DT, STEPS = 0.005, 600
TARGETS = ["beta", "sigr", "Q", "S", "C8"]

# (family, eps, theta_deg) — θ20/ε0.05 hernquist is shared by both sweeps (computed once)
CELLS = [
    ("hernquist3d", 0.05, 5), ("hernquist3d", 0.05, 10),
    ("hernquist3d", 0.05, 20), ("hernquist3d", 0.05, 30),
    ("hernquist3d", 0.02, 20), ("hernquist3d", 0.10, 20),
    ("plummer3d", 0.05, 20),
]


def _kin_obs(pos, vel, cfg):
    """β, σ_r, phase-space entropy S, C₈ — no potential evaluation (cheap)."""
    center = np.mean(pos, axis=0)
    r, v_r, v_t = psf.decompose(pos, vel, center)
    inside = r <= psf.PS_RMAX
    sigr = float(np.std(v_r[inside])) if np.sum(inside) > 2 else float("nan")
    sigt = float(np.sqrt(np.mean(v_t[inside] ** 2))) if np.sum(inside) > 2 else float("nan")
    beta = 1.0 - sigt ** 2 / (2.0 * sigr ** 2) if sigr and sigr > 1e-9 else float("nan")
    return {"beta": beta, "sigr": sigr, "S": psf._phase_entropy(r, v_r),
            "C8": obs_coarse_var(pos, cfg, 8, False)}


def _cfg(fam, eps, seed):
    return StressConfig(model="direct_isolated", init=fam, seed=seed, n=1024, steps=STEPS,
                        eps=eps, box_size=2.0, k_fine=16, plummer_a=0.20)


def _worker(task):
    fam, eps, theta_deg, seed = task
    theta = math.radians(theta_deg)
    cfg = _cfg(fam, eps, seed); sc = get_simconfig(cfg); mass = 1.0 / 1024
    pos0, vel0 = get_initial_conditions(cfg)
    center = np.mean(pos0, axis=0)
    vels = {"orig": vel0, "int": intervene_anisotropy(pos0, vel0, theta, center),
            "sham": sham_rotation(pos0, vel0, theta, seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        o0 = _kin_obs(pos0, v, cfg)
        ke0 = 0.5 * mass * float(np.sum(v * v))
        r0 = psf.relaxation_observables(pos0, v, cfg)            # for Q0 (PE)
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted({0, STEPS}), True)
        o1 = _kin_obs(snaps[STEPS][0], snaps[STEPS][1], cfg)
        r1 = psf.relaxation_observables(snaps[STEPS][0], snaps[STEPS][1], cfg)  # Q1
        out[name] = {**{f"{k}0": o0[k] for k in o0}, **{f"{k}1": o1[k] for k in o1},
                     "Q0": r0["Q"], "Q1": r1["Q"], "ke0": ke0,
                     "L0": float(np.linalg.norm(angular_momentum_3d(pos0, v, mass))),
                     "p0": float(np.linalg.norm(total_momentum(v, mass)))}
        if name in ("int", "sham"):                              # free-streaming (gravity OFF)
            pos_free = pos0 + (STEPS * DT) * v
            of = _kin_obs(pos_free, v, cfg)
            out[name].update({f"{k}1_free": of[k] for k in of})
    return fam, eps, theta_deg, out


def _paired(vals, seed=7):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size < 5:
        return float("nan"), float("nan"), float("nan"), a.size
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(2000, len(a)))].mean(axis=1)
    return float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), a.size


def analyse(by_cell):
    cells = {}
    for key, rows in by_cell.items():
        fam, eps, th = key
        d = {"family": fam, "eps": eps, "theta_deg": th, "n_pairs": len(rows), "targets": {}}
        for t in TARGETS:
            causal = [r["int"][f"{t}1"] - r["sham"][f"{t}1"] for r in rows]
            d["targets"][t] = {"causal_int_minus_sham": _paired(causal)}
        # β specifics
        imp = [r["int"]["beta0"] - r["orig"]["beta0"] for r in rows]
        causb = [r["int"]["beta1"] - r["sham"]["beta1"] for r in rows]
        free = [r["int"]["beta1_free"] - r["sham"]["beta1_free"] for r in rows]
        shamimp = [r["sham"]["beta0"] - r["orig"]["beta0"] for r in rows]
        sham_t1 = [r["sham"]["beta1"] - r["orig"]["beta1"] for r in rows]
        d["imposed_dbeta0"] = _paired(imp)
        d["sham_dbeta0"] = _paired(shamimp)
        d["sham_minus_orig_beta1"] = _paired(sham_t1)
        d["causal_beta1"] = _paired(causb)
        d["free_causal_beta1"] = _paired(free)
        im, cm = d["imposed_dbeta0"][0], d["causal_beta1"][0]
        fm = d["free_causal_beta1"][0]
        d["persistence_grav"] = float(cm / im) if abs(im) > 1e-9 else float("nan")
        d["persistence_free"] = float(fm / im) if abs(im) > 1e-9 else float("nan")
        d["orig_beta0_mean"] = float(np.mean([r["orig"]["beta0"] for r in rows]))
        d["orig_beta1_mean"] = float(np.mean([r["orig"]["beta1"] for r in rows]))
        # conservation (int vs orig at t0)
        d["dKE_rel"] = float(np.median([abs(r["int"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
        d["dQ_rel"] = float(np.median([abs(r["int"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))
        d["dL"] = float(np.median([r["int"]["L0"] - r["orig"]["L0"] for r in rows]))
        d["dp"] = float(np.median([r["int"]["p0"] - r["orig"]["p0"] for r in rows]))
        cells[f"{fam}|eps={eps:g}|th={th}"] = d
    return cells


def verdict(cells):
    hern = lambda th: cells.get(f"hernquist3d|eps=0.05|th={th}")
    th_list = [5, 10, 20, 30]
    th_cells = [hern(t) for t in th_list if hern(t)]
    imp_th = [c["imposed_dbeta0"][0] for c in th_cells]
    eff_th = [c["causal_beta1"][0] for c in th_cells]
    sp_imp = _spearman(np.array(th_list[:len(th_cells)], float), np.array(imp_th))
    sp_eff = _spearman(np.array(th_list[:len(th_cells)], float), np.array(eff_th))

    eps_list = [0.02, 0.05, 0.10]
    eps_cells = [cells.get(f"hernquist3d|eps={e:g}|th=20") for e in eps_list]
    eff_eps = [(c["eps"], c["causal_beta1"]) for c in eps_cells if c]

    def ci_pos(p):
        return math.isfinite(p[1]) and (p[1] > 0 or p[2] < 0)
    n_cells = len(cells)
    n_eff = sum(1 for c in cells.values() if ci_pos(c["causal_beta1"]))
    sham_ok = all(not ci_pos(c["sham_dbeta0"]) for c in cells.values())
    cons_ok = all(c["dKE_rel"] < 1e-2 and c["dQ_rel"] < 1e-2 for c in cells.values())
    th_monotonic = sp_imp > 0.8 and sp_eff > 0.5
    # free-streaming comparison (median persistence across cells)
    pg = float(np.median([c["persistence_grav"] for c in cells.values() if math.isfinite(c["persistence_grav"])]))
    pf = float(np.median([c["persistence_free"] for c in cells.values() if math.isfinite(c["persistence_free"])]))
    nontrivial = abs(pg) > abs(pf) + 0.05    # gravitational memory exceeds ballistic baseline

    if not cons_ok:
        dec = "INVALID — KE/Q not preserved in some cell."
    elif not sham_ok:
        dec = "KILLED — sham imposes a comparable Δβ₀ in some cell; the handle is not anisotropy-specific."
    elif th_monotonic and n_eff >= int(0.7 * n_cells) and nontrivial:
        dec = ("CONFIRMED — the causal anisotropy handle scales with θ (Spearman θ↔imposed="
               f"{sp_imp:+.2f}, θ↔effect={sp_eff:+.2f}), the effect CI excludes 0 in {n_eff}/{n_cells} "
               f"cells, sham stays null, KE/Q preserved, and gravitational persistence ({pg:.0%}) "
               f"exceeds the free-streaming baseline ({pf:.0%}). Controlled anisotropy perturbations "
               "causally modulate relaxation outcomes. No AWS.")
    elif th_monotonic and n_eff >= int(0.7 * n_cells):
        dec = ("CONFIRMED-BUT-TRIVIAL-MEMORY — the effect scales with θ and is robust, but "
               f"gravitational persistence ({pg:.0%}) is NOT clearly above the free-streaming "
               f"baseline ({pf:.0%}); the 'memory' may be ballistic carryover rather than a "
               "relaxation effect. Interpret cautiously.")
    elif n_eff >= int(0.7 * n_cells):
        dec = ("REGIME-SPECIFIC / FRAGILE — effect present in most cells but θ-scaling is weak "
               f"(Spearman θ↔effect={sp_eff:+.2f}); downgrade from a clean handle.")
    else:
        dec = f"NOT CONFIRMED — effect CI excludes 0 in only {n_eff}/{n_cells} cells; the pilot was fragile."

    return {
        "theta_spearman_imposed": sp_imp, "theta_spearman_effect": sp_eff,
        "theta_imposed_dbeta0": dict(zip(th_list, imp_th)),
        "theta_effect_beta1": dict(zip(th_list, eff_th)),
        "eps_effect_beta1": {f"{e:g}": p for e, p in eff_eps},
        "n_effect_cells": f"{n_eff}/{n_cells}",
        "sham_null": sham_ok, "conservation_ok": cons_ok,
        "persistence_grav_median": pg, "persistence_free_median": pf,
        "nontrivial_vs_freestream": nontrivial,
        "aws_needed": False, "decision": dec,
    }


def run(workers):
    os.makedirs(OUTDIR, exist_ok=True)
    seeds = [2000 + i for i in range(100)]
    tasks = [(fam, eps, th, s) for (fam, eps, th) in CELLS for s in seeds]
    by_cell = {}
    print(f"Medium confirmation: {len(CELLS)} cells × 100 pairs = {len(tasks)} matched triples")
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(True,)) as ex:
        futs = [ex.submit(_worker, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
            fam, eps, th, out = fut.result()
            by_cell.setdefault((fam, eps, th), []).append(out)
    cells = analyse(by_cell)
    summ = {"cells": cells, "verdict": verdict(cells)}
    _write(cells, summ, by_cell)
    return summ


def _write(cells, summ, by_cell):
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "eps", "theta_deg", "n_pairs", "imposed_dbeta0", "causal_beta1",
                    "causal_beta1_lo", "causal_beta1_hi", "persistence_grav", "persistence_free",
                    "sham_dbeta0", "dKE_rel", "dQ_rel", "dL",
                    "causal_sigr", "causal_Q", "causal_S", "causal_C8",
                    "orig_beta0", "orig_beta1"])
        for c in cells.values():
            w.writerow([c["family"], c["eps"], c["theta_deg"], c["n_pairs"],
                        f"{c['imposed_dbeta0'][0]:.4f}", f"{c['causal_beta1'][0]:.4f}",
                        f"{c['causal_beta1'][1]:.4f}", f"{c['causal_beta1'][2]:.4f}",
                        f"{c['persistence_grav']:.4f}", f"{c['persistence_free']:.4f}",
                        f"{c['sham_dbeta0'][0]:.4f}", f"{c['dKE_rel']:.2e}", f"{c['dQ_rel']:.2e}",
                        f"{c['dL']:.4f}",
                        f"{c['targets']['sigr']['causal_int_minus_sham'][0]:.4f}",
                        f"{c['targets']['Q']['causal_int_minus_sham'][0]:.4f}",
                        f"{c['targets']['S']['causal_int_minus_sham'][0]:.4f}",
                        f"{c['targets']['C8']['causal_int_minus_sham'][0]:.4f}",
                        f"{c['orig_beta0_mean']:.4f}", f"{c['orig_beta1_mean']:.4f}"])
    _figures(cells, summ)
    _report(cells, summ)
    print(f"[outputs] → {OUTDIR}/")


def _figures(cells, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    th = [5, 10, 20, 30]
    tc = [cells.get(f"hernquist3d|eps=0.05|th={t}") for t in th]
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    # effect & imposed vs θ
    ax[0].errorbar(th, [c["causal_beta1"][0] for c in tc],
                   yerr=[[c["causal_beta1"][0] - c["causal_beta1"][1] for c in tc],
                         [c["causal_beta1"][2] - c["causal_beta1"][0] for c in tc]],
                   marker="o", capsize=4, label="causal β effect")
    ax[0].plot(th, [c["imposed_dbeta0"][0] for c in tc], "s--", color="gray", label="imposed Δβ₀")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("θ (deg)"); ax[0].set_ylabel("Δβ")
    ax[0].set_title("Effect & imposed vs θ"); ax[0].legend(fontsize=8)
    # persistence: grav vs free
    ax[1].plot(th, [c["persistence_grav"] for c in tc], "o-", label="gravity")
    ax[1].plot(th, [c["persistence_free"] for c in tc], "x--", color="C3", label="free-streaming")
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("θ (deg)"); ax[1].set_ylabel("persistence β(t₁)/imposed")
    ax[1].set_title("Memory: gravity vs ballistic"); ax[1].legend(fontsize=8)
    # effect vs eps
    eps = [0.02, 0.05, 0.10]
    ec = [cells.get(f"hernquist3d|eps={e:g}|th=20") for e in eps]
    ax[2].errorbar(eps, [c["causal_beta1"][0] for c in ec],
                   yerr=[[c["causal_beta1"][0] - c["causal_beta1"][1] for c in ec],
                         [c["causal_beta1"][2] - c["causal_beta1"][0] for c in ec]],
                   marker="o", capsize=4, color="C2")
    ax[2].axhline(0, color="k", lw=0.6); ax[2].set_xlabel("ε"); ax[2].set_ylabel("causal β effect")
    ax[2].set_title("Effect vs ε (θ=20°)")
    fig.tight_layout()
    os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "figures", "fig_medium_confirmation.pdf")); plt.close(fig)


def _report(cells, summ):
    v = summ["verdict"]
    band = ("🟢 CONFIRMED" if v["decision"].startswith("CONFIRMED -") or v["decision"].startswith("CONFIRMED —")
            else "🟡 CAUTION" if "TRIVIAL" in v["decision"] or "REGIME" in v["decision"] or "FRAGILE" in v["decision"]
            else "🔴 NOT CONFIRMED" if v["decision"].startswith("NOT") or v["decision"].startswith("KILLED")
            else "⚪")
    L = ["# N-body Causal Anisotropy Handle — Medium Confirmation\n"]
    L.append("Matched-pair (orig/int/sham), N=1024, 100 pairs/cell, horizon t₁=600. Handle: "
             "speed-preserving radial anisotropy rotation. **Goal: try to break the handle.**\n")
    L.append(f"## Verdict — {band}\n\n> **{v['decision']}**\n")
    L.append("## The four stress tests\n")
    L.append(f"1. **θ-scaling:** Spearman(θ, imposed Δβ₀) = {v['theta_spearman_imposed']:+.2f}; "
             f"Spearman(θ, effect) = {v['theta_spearman_effect']:+.2f}. "
             f"effect by θ: {{ {', '.join(f'{k}°:{val:+.3f}' for k,val in v['theta_effect_beta1'].items())} }}.")
    L.append(f"2. **ε-robustness:** causal β effect by ε: "
             f"{{ {', '.join(f'{k}:{p[0]:+.3f}[{p[1]:+.3f},{p[2]:+.3f}]' for k,p in v['eps_effect_beta1'].items())} }}.")
    L.append(f"3. **Sham null:** {v['sham_null']}.  **Conservation OK:** {v['conservation_ok']}.  "
             f"effect CI>0 in {v['n_effect_cells']} cells.")
    L.append(f"4. **Trivial-memory baseline:** gravitational persistence (median) "
             f"{v['persistence_grav_median']:.0%} vs **free-streaming** {v['persistence_free_median']:.0%} "
             f"→ {'non-trivial (gravity sustains the imposed anisotropy beyond ballistic carryover)' if v['nontrivial_vs_freestream'] else 'NOT clearly above ballistic baseline — memory may be trivial'}.\n")
    L.append("## Per-cell\n")
    L.append("| family | ε | θ | imposed Δβ₀ | causal β [CI] | persist grav/free | sham Δβ₀ | ΔKE/KE | causal σr | causal Q | causal S | causal C₈ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells.values():
        t = c["targets"]
        L.append(f"| {c['family']} | {c['eps']:g} | {c['theta_deg']}° | {c['imposed_dbeta0'][0]:+.3f} | "
                 f"{c['causal_beta1'][0]:+.3f} [{c['causal_beta1'][1]:+.3f},{c['causal_beta1'][2]:+.3f}] | "
                 f"{c['persistence_grav']:.0%}/{c['persistence_free']:.0%} | {c['sham_dbeta0'][0]:+.3f} | "
                 f"{c['dKE_rel']:.1e} | {t['sigr']['causal_int_minus_sham'][0]:+.3f} | "
                 f"{t['Q']['causal_int_minus_sham'][0]:+.3f} | {t['S']['causal_int_minus_sham'][0]:+.3f} | "
                 f"{t['C8']['causal_int_minus_sham'][0]:+.3f} |")
    L.append(f"\n*Context: unperturbed β relaxes 0→~{cells.get('hernquist3d|eps=0.05|th=20',{'orig_beta1_mean':0})['orig_beta1_mean']:.2f} "
             "(radial-anisotropy attractor); the handle is a modulation on top.*\n")
    with open(os.path.join(OUTDIR, "medium_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Medium confirmation of the N-body anisotropy handle (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    summ = run(args.workers)
    print("\n" + summ["verdict"]["decision"])


if __name__ == "__main__":
    main()
