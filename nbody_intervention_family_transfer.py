#!/usr/bin/env python3
"""
nbody_intervention_family_transfer.py — does the anisotropy handle transfer beyond Hernquist?
=============================================================================================

The confirmed handle (radial velocity-anisotropy rotation) modulates future relaxation in
Hernquist.  Before inventing new handles, test whether the SAME handle transfers to a second
cusp family with different core/relaxation geometry: Plummer.

Matrix: {Hernquist, Plummer} × ε∈{0.02,0.05,0.10}, θ=20°, N=1024, 100 matched pairs/cell.
Reuses the medium-confirmation cells where they already exist (Hernquist ×3 ε, Plummer ε=0.05)
and runs only the missing Plummer cells (ε=0.02, 0.10).  Same paired design, same controls,
same free-streaming baseline.  No AWS.  paper.tex untouched.
"""
from __future__ import annotations

import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
import nbody_intervention_medium as med

OUTDIR = "outputs/nbody_intervention_family_transfer"
MED_SUMMARY = "outputs/nbody_intervention_medium/summary.json"
FAMILIES = ["hernquist3d", "plummer3d"]
EPS = [0.02, 0.05, 0.10]
THETA = 20


def _key(fam, eps):
    return f"{fam}|eps={eps:g}|th={THETA}"


def _ci_pos(p):
    return p is not None and math.isfinite(p[1]) and (p[1] > 0 or p[2] < 0)


def gather_cells(workers):
    # reuse medium-confirmation cells where available
    existing = {}
    if os.path.exists(MED_SUMMARY):
        existing = json.load(open(MED_SUMMARY)).get("cells", {})
    cells, missing = {}, []
    for fam in FAMILIES:
        for eps in EPS:
            k = _key(fam, eps)
            if k in existing:
                cells[k] = existing[k]
            else:
                missing.append((fam, eps))
    print(f"[family-transfer] reusing {len(cells)} cells; running {len(missing)} missing: {missing}")

    if missing:
        seeds = [2000 + i for i in range(100)]
        tasks = [(fam, eps, THETA, s) for (fam, eps) in missing for s in seeds]
        by_cell = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(True,)) as ex:
            futs = [ex.submit(med._worker, t) for t in tasks]
            for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
                fam, eps, th, out = fut.result()
                by_cell.setdefault((fam, eps, th), []).append(out)
        for k, v in med.analyse(by_cell).items():
            cells[k] = v
    return cells


def family_verdict(cells):
    res = {}
    for fam in FAMILIES:
        cs = [cells.get(_key(fam, e)) for e in EPS]
        cs = [c for c in cs if c]
        n_pos = sum(_ci_pos(c["causal_beta1"]) for c in cs)
        sign_ok = all(np.sign(c["causal_beta1"][0]) == np.sign(c["imposed_dbeta0"][0]) for c in cs)
        sham_ok = all(not _ci_pos(c["sham_dbeta0"]) for c in cs)
        cons_ok = all(c["dKE_rel"] < 1e-2 and c["dQ_rel"] < 1e-2 for c in cs)
        pg = float(np.median([c["persistence_grav"] for c in cs]))
        pf = float(np.median([c["persistence_free"] for c in cs]))
        c8_pos = sum(_ci_pos(c["targets"]["C8"]["causal_int_minus_sham"]) for c in cs)
        passed = (n_pos >= 2 and sign_ok and sham_ok and cons_ok)
        res[fam] = {
            "n_beta_ci_pos": f"{n_pos}/{len(cs)}", "sign_ok": sign_ok, "sham_null": sham_ok,
            "conservation_ok": cons_ok, "persistence_grav": pg, "persistence_free": pf,
            "nontrivial_vs_free": abs(pg) > abs(pf),
            "c8_responds": f"{c8_pos}/{len(cs)}",
            "beta_effect_by_eps": {f"{e:g}": cells[_key(fam, e)]["causal_beta1"]
                                   for e in EPS if _key(fam, e) in cells},
            "passed": passed,
        }
    h, p = res["hernquist3d"], res["plummer3d"]
    if h["passed"] and p["passed"]:
        dec = ("ROBUST CUSP-FAMILY HANDLE — velocity anisotropy is a causal handle on future "
               f"relaxation in BOTH Hernquist and Plummer (β CI>0 in {h['n_beta_ci_pos']} / "
               f"{p['n_beta_ci_pos']} ε cells, sham null, KE/Q preserved, gravity persistence "
               f"{p['persistence_grav']:.0%} {'>' if p['nontrivial_vs_free'] else '≈'} free-streaming "
               f"{p['persistence_free']:.0%}). The handle transfers across cusp families. No AWS.")
    elif h["passed"] and not p["passed"]:
        dec = ("REGIME-SPECIFIC — Hernquist passes but Plummer does NOT "
               f"(β CI>0 in only {p['n_beta_ci_pos']} ε cells). The anisotropy handle is likely "
               "tied to cusp/radial-orbit relaxation specifics, not a general cusp-family effect.")
    elif p["passed"]:
        dec = ("CONDITIONAL — Plummer passes; revisit Hernquist criteria. Handle present but "
               "family-dependent in strength.")
    else:
        dec = "NOT TRANSFERRED — the handle does not pass cleanly in Plummer."

    c8_family_specific = (int(h["c8_responds"].split("/")[0]) > 0) != (int(p["c8_responds"].split("/")[0]) > 0)
    return {"by_family": res, "c8_family_specific": c8_family_specific,
            "aws_needed": False, "decision": dec}


def _write(cells, summ):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump({"cells": cells, "verdict": summ}, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["family", "eps", "theta", "imposed_dbeta0", "causal_beta1", "ci_lo", "ci_hi",
                    "persistence_grav", "persistence_free", "sham_dbeta0", "dKE_rel", "dQ_rel",
                    "causal_sigr", "causal_Q", "causal_S", "causal_C8", "orig_beta0", "orig_beta1"])
        for fam in FAMILIES:
            for e in EPS:
                c = cells.get(_key(fam, e))
                if not c:
                    continue
                t = c["targets"]
                w.writerow([fam, e, THETA, f"{c['imposed_dbeta0'][0]:.4f}", f"{c['causal_beta1'][0]:.4f}",
                            f"{c['causal_beta1'][1]:.4f}", f"{c['causal_beta1'][2]:.4f}",
                            f"{c['persistence_grav']:.4f}", f"{c['persistence_free']:.4f}",
                            f"{c['sham_dbeta0'][0]:.4f}", f"{c['dKE_rel']:.2e}", f"{c['dQ_rel']:.2e}",
                            f"{t['sigr']['causal_int_minus_sham'][0]:.4f}",
                            f"{t['Q']['causal_int_minus_sham'][0]:.4f}",
                            f"{t['S']['causal_int_minus_sham'][0]:.4f}",
                            f"{t['C8']['causal_int_minus_sham'][0]:.4f}",
                            f"{c['orig_beta0_mean']:.4f}", f"{c['orig_beta1_mean']:.4f}"])
    _figures(cells, summ)
    _report(cells, summ)
    print(f"[outputs] → {OUTDIR}/")


def _figures(cells, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"hernquist3d": "C0", "plummer3d": "C1"}
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for fam in FAMILIES:
        cs = [cells.get(_key(fam, e)) for e in EPS]
        m = [c["causal_beta1"][0] if c else np.nan for c in cs]
        lo = [c["causal_beta1"][0] - c["causal_beta1"][1] if c else 0 for c in cs]
        hi = [c["causal_beta1"][2] - c["causal_beta1"][0] if c else 0 for c in cs]
        ax[0].errorbar(EPS, m, yerr=[lo, hi], marker="o", capsize=4, color=colors[fam], label=fam)
        ax[1].plot(EPS, [c["persistence_grav"] if c else np.nan for c in cs], "o-", color=colors[fam], label=f"{fam} grav")
        ax[1].plot(EPS, [c["persistence_free"] if c else np.nan for c in cs], "x--", color=colors[fam], alpha=0.6, label=f"{fam} free")
    ax[0].axhline(0, color="k", lw=0.6); ax[0].set_xlabel("ε"); ax[0].set_ylabel("causal β effect")
    ax[0].set_title("β effect vs ε by family"); ax[0].legend(fontsize=8)
    ax[1].axhline(0, color="k", lw=0.6); ax[1].set_xlabel("ε"); ax[1].set_ylabel("persistence")
    ax[1].set_title("Persistence: gravity vs free-streaming"); ax[1].legend(fontsize=7)
    # secondary-target response map
    tg = ["beta", "sigr", "Q", "S", "C8"]
    rows_lbl = [f"{f[:4]} ε{e:g}" for f in FAMILIES for e in EPS]
    M = np.full((len(rows_lbl), len(tg)), np.nan)
    for i, (f, e) in enumerate([(f, e) for f in FAMILIES for e in EPS]):
        c = cells.get(_key(f, e))
        if not c:
            continue
        for j, t in enumerate(tg):
            p = c["causal_beta1"] if t == "beta" else c["targets"][t]["causal_int_minus_sham"]
            M[i, j] = 1.0 if _ci_pos(p) else 0.0
    ax[2].imshow(M, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax[2].set_xticks(range(len(tg))); ax[2].set_xticklabels(tg)
    ax[2].set_yticks(range(len(rows_lbl))); ax[2].set_yticklabels(rows_lbl, fontsize=8)
    ax[2].set_title("Target responds (CI>0)?")
    fig.tight_layout()
    os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "figures", "fig_family_transfer.pdf")); plt.close(fig)


def _report(cells, summ):
    v = summ; bf = v["by_family"]
    dec = v["decision"]
    band = ("🟢 ROBUST CUSP-FAMILY" if dec.startswith("ROBUST")
            else "🟡 REGIME-SPECIFIC" if dec.startswith("REGIME") or dec.startswith("CONDITIONAL")
            else "🔴 NOT TRANSFERRED")
    L = ["# N-body Anisotropy Handle — Family Transfer (Hernquist → Plummer)\n"]
    L.append("Same handle / sham / controls as the confirmation. {Hernquist, Plummer} × "
             "ε∈{0.02,0.05,0.10}, θ=20°, N=1024, 100 matched pairs/cell.\n")
    L.append(f"## Verdict — {band}\n\n> **{dec}**\n")
    L.append("## By family\n")
    for fam in FAMILIES:
        r = bf[fam]
        L.append(f"- **{fam}** — β CI>0 in {r['n_beta_ci_pos']} ε cells; sign_ok={r['sign_ok']}; "
                 f"sham_null={r['sham_null']}; conservation_ok={r['conservation_ok']}; "
                 f"persistence grav/free = {r['persistence_grav']:.0%}/{r['persistence_free']:.0%} "
                 f"({'non-trivial' if r['nontrivial_vs_free'] else 'NOT > ballistic'}); "
                 f"C₈ responds in {r['c8_responds']} cells; **passed={r['passed']}**.")
    L.append(f"\n**C₈ response family-specific:** {v['c8_family_specific']}.\n")
    L.append("## Per-cell\n")
    L.append("| family | ε | imposed Δβ₀ | causal β [CI] | persist grav/free | sham Δβ₀ | ΔKE/KE | "
             "causal σr | causal Q | causal S | causal C₈ | orig β 0→1 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for fam in FAMILIES:
        for e in EPS:
            c = cells.get(_key(fam, e))
            if not c:
                continue
            t = c["targets"]
            L.append(f"| {fam} | {e:g} | {c['imposed_dbeta0'][0]:+.3f} | "
                     f"{c['causal_beta1'][0]:+.3f} [{c['causal_beta1'][1]:+.3f},{c['causal_beta1'][2]:+.3f}] | "
                     f"{c['persistence_grav']:.0%}/{c['persistence_free']:.0%} | {c['sham_dbeta0'][0]:+.3f} | "
                     f"{c['dKE_rel']:.1e} | {t['sigr']['causal_int_minus_sham'][0]:+.3f} | "
                     f"{t['Q']['causal_int_minus_sham'][0]:+.3f} | {t['S']['causal_int_minus_sham'][0]:+.3f} | "
                     f"{t['C8']['causal_int_minus_sham'][0]:+.3f} | {c['orig_beta0_mean']:+.2f}→{c['orig_beta1_mean']:+.2f} |")
    L.append("")
    with open(os.path.join(OUTDIR, "family_transfer_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Anisotropy handle family-transfer test (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()
    cells = gather_cells(args.workers)
    summ = family_verdict(cells)
    _write(cells, summ)
    print("\n" + summ["decision"])


if __name__ == "__main__":
    main()
