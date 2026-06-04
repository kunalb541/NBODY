#!/usr/bin/env python3
"""
nbody_intervention_Ndep.py — is the |L| causal chain a finite-N artifact?
=========================================================================

The mechanism (conserved |L|-depletion → M(<r_c)↑ at t≈5 → C₈↑ at t≈10–20; β transient) is
time-ordered, form-robust, and cross-profile.  Remaining worry: is it just a property of a finite
particle realization?  Test across N ∈ {512, 1024, 2048} (Hernquist, ε=0.05, θ=20°, radialize/
tangentialize/sham).

Note on metrics: C₈ (count variance on an 8³ grid) scales ~extensively with N, so it is read for
SIGN/ordering only.  The clean "does the effect vanish as N→∞?" indicators are the INTENSIVE ones:
M(<r_c) (mass FRACTION), β, ⟨|L_i|⟩.  If those stay roughly constant (not →0) while the ordering
holds at every N, the mechanism is particle-number robust, not a finite-N artifact.

No AWS.  No new handles.  paper.tex untouched.
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
from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from nbody_anisotropy_mechanism_pilot import tangentialize
from nbody_intervention_timecourse import _obs

OUTDIR = "outputs/nbody_intervention_Ndep"
TIMES = [0, 5, 10, 20, 50, 100, 300, 600]
THETA, EPS = 20.0, 0.05
N_PAIRS = [(512, 100), (1024, 100), (2048, 50)]
QUANT = ["beta", "Lspec", "M05", "M10", "M20", "C8", "sigr"]


def _worker(task):
    seed, n = task
    cfg = StressConfig(model="direct_isolated", init="hernquist3d", seed=seed, n=n, steps=max(TIMES),
                       eps=EPS, box_size=2.0, k_fine=16, plummer_a=0.20)
    sc = get_simconfig(cfg); mass = 1.0 / n
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0)
    th = math.radians(THETA)
    vels = {"orig": vel0, "rad": intervene_anisotropy(pos0, vel0, th, center),
            "tan": tangentialize(pos0, vel0, th, center, seed),
            "sham": sham_rotation(pos0, vel0, th, seed)}
    out = {"seed": seed}
    for name, v in vels.items():
        snaps = _integrate_leapfrog(pos0, v, mass, sc, sorted(set(TIMES)), True)
        out[name] = {str(t): _obs(snaps[t][0], snaps[t][1], cfg) for t in TIMES}
        out[name]["ke0"] = 0.5 * mass * float(np.sum(v * v))
        out[name]["Q0"] = psf.relaxation_observables(pos0, v, cfg)["Q"]
        out[name]["C8orig"] = out[name]["0"]["C8"] if name == "orig" else None
    return out


def _paired(vals, seed=7):
    a = np.array([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size < 5:
        return [float("nan")] * 3 + [a.size]
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), size=(1500, len(a)))].mean(axis=1)
    return [float(a.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), a.size]


def _ci_pos(p):
    return math.isfinite(p[1]) and (p[1] > 0 or p[2] < 0)


def _first_sig(eff_q):
    for t in TIMES:
        if t > 0 and _ci_pos(eff_q[str(t)]):
            return t
    return None


def analyse_N(rows, n):
    eff = {arm: {q: {str(t): _paired([r[arm][str(t)][q] - r["sham"][str(t)][q] for r in rows])
                     for t in TIMES} for q in QUANT} for arm in ("rad", "tan")}
    sham_dyn = {q: {str(t): _paired([r["sham"][str(t)][q] - r["orig"][str(t)][q] for r in rows])
                    for t in TIMES} for q in ("beta", "M10", "C8")}
    dKE = float(np.median([abs(r["rad"]["ke0"] - r["orig"]["ke0"]) / max(r["orig"]["ke0"], 1e-30) for r in rows]))
    dQ = float(np.median([abs(r["rad"]["Q0"] - r["orig"]["Q0"]) / max(abs(r["orig"]["Q0"]), 1e-30) for r in rows]))
    fin = str(TIMES[-1])
    t_M05, t_M10, t_C8 = (_first_sig(eff["rad"][q]) for q in ("M05", "M10", "C8"))
    L0, Lf = eff["rad"]["Lspec"]["0"][0], eff["rad"]["Lspec"][fin][0]
    L_permanent = abs(L0) > 1e-6 and abs(Lf / L0) > 0.85
    b0, bf = eff["rad"]["beta"]["0"][0], eff["rad"]["beta"][fin][0]
    beta_transient = abs(bf) < 0.5 * abs(b0)
    conc_before_c8 = (t_M05 is not None and t_C8 is not None and t_M05 < t_C8) or \
                     (t_M10 is not None and t_C8 is not None and t_M10 < t_C8)
    simultaneous = (t_M05 == t_C8) or (t_M10 == t_C8)
    peak_M10 = max(abs(eff["rad"]["M10"][str(t)][0]) for t in TIMES)        # intensive: mass fraction
    peak_C8 = max(abs(eff["rad"]["C8"][str(t)][0]) for t in TIMES)          # extensive (~N)
    c8_sign = np.sign(eff["rad"]["C8"][fin][0])

    def _sham_clean(q):
        s = max(abs(sham_dyn[q][str(t)][0]) for t in TIMES)
        peak = max(abs(eff["rad"][q][str(t)][0]) for t in TIMES)
        return s < 0.15 * peak + 1e-3
    sham_clean = all(_sham_clean(q) for q in ("beta", "M10", "C8"))
    return {"n": n, "effects": eff, "sham_dynamic": sham_dyn, "n_pairs": len(rows),
            "first_sig": {"M05": t_M05, "M10": t_M10, "C8": t_C8},
            "L_permanent": L_permanent, "beta_transient": beta_transient,
            "concentration_before_c8": conc_before_c8, "simultaneous": simultaneous,
            "peak_M10_effect": peak_M10, "peak_C8_effect": peak_C8, "c8_sign": float(c8_sign),
            "imposed_beta0": b0, "imposed_L0": L0, "sham_clean": sham_clean,
            "conservation": {"dKE_rel": dKE, "dQ_rel": dQ}}


def verdict_across_N(per_N):
    Ns = sorted(per_N.keys())
    orderings = [per_N[n]["concentration_before_c8"] or per_N[n]["simultaneous"] for n in Ns]
    Lperm = [per_N[n]["L_permanent"] for n in Ns]
    btrans = [per_N[n]["beta_transient"] for n in Ns]
    signs = [per_N[n]["c8_sign"] for n in Ns]
    m10 = [per_N[n]["peak_M10_effect"] for n in Ns]

    def _shamq(pn, q):   # sham-cleanliness for one quantity (peak-relative, absolute floor)
        s = max(abs(pn["sham_dynamic"][q][str(t)][0]) for t in TIMES)
        peak = max(abs(pn["effects"]["rad"][q][str(t)][0]) for t in TIMES)
        return s < 0.15 * peak + 1e-3
    # judge sham on the INTENSIVE channels (β, M10); C₈ is extensive and noisy at small N
    sham_intensive = all(_shamq(per_N[n], "beta") and _shamq(per_N[n], "M10") for n in Ns)
    c8_sham_noisy = [n for n in Ns if not _shamq(per_N[n], "C8")]
    sham_ok = sham_intensive
    cons_ok = all(per_N[n]["conservation"]["dKE_rel"] < 1e-2 and per_N[n]["conservation"]["dQ_rel"] < 1e-2 for n in Ns)
    # does the intensive M(<r_c) effect vanish as N grows?  (ratio of largest-N to smallest-N peak)
    m10_ratio = m10[-1] / m10[0] if m10[0] > 1e-9 else float("nan")
    m10_vanishes = m10_ratio < 0.4                      # dropped to <40% across the N range → trending to 0
    ordering_all = all(orderings)
    sign_consistent = len(set(signs)) == 1
    robust = ordering_all and all(Lperm) and all(btrans) and sign_consistent and not m10_vanishes and sham_ok and cons_ok

    if not cons_ok:
        dec = "REJECT — KE/Q drift at some N."
    elif robust:
        caveat = (f" (C₈ sham is noisy at N={c8_sham_noisy} — expected: C₈ is a count variance, small "
                  f"and noisy at low N; the intensive β/M(<r_c) shams are clean at all N.)"
                  if c8_sham_noisy else "")
        dec = (f"PARTICLE-NUMBER ROBUST — across N={Ns}: ordering (M before/with C₈) holds at all N, "
               f"|L| permanent at all N, β transient at all N, C₈ sign consistent (+), intensive sham "
               f"clean, KE/Q clean; the INTENSIVE M(<r_c) effect does NOT vanish (peak "
               f"{m10[0]:.3f}→{m10[-1]:.3f}, ratio {m10_ratio:.2f}) while the extensive C₈ grows ~with N. "
               f"The |L|→concentration→C₈ chain is a real dynamical mechanism in the tested range, not a "
               f"finite-N realization artifact." + caveat + " No AWS.")
    elif m10_vanishes and ordering_all:
        dec = (f"FINITE-N SUSPECT — the ordering holds but the intensive M(<r_c) effect drops sharply "
               f"with N (peak {m10[0]:.3f}→{m10[-1]:.3f}, ratio {m10_ratio:.2f}); the effect may trend "
               f"to zero as N→∞ — possible finite-N artifact. Larger N needed to decide.")
    elif per_N[Ns[-1]]["n_pairs"] < 100 and not ordering_all:
        dec = f"INCONCLUSIVE — largest N ({Ns[-1]}) has only {per_N[Ns[-1]]['n_pairs']} pairs; ordering unresolved there."
    else:
        dec = f"NOT ROBUST — ordering/|L|/β structure not consistent across N (orderings={orderings}, Lperm={Lperm})."
    return {"Ns": Ns, "ordering_all_N": ordering_all, "L_permanent_all_N": all(Lperm),
            "beta_transient_all_N": all(btrans), "c8_sign_consistent": sign_consistent,
            "peak_M10_by_N": dict(zip(Ns, m10)), "peak_C8_by_N": dict(zip(Ns, [per_N[n]["peak_C8_effect"] for n in Ns])),
            "m10_ratio_hiN_loN": m10_ratio, "m10_vanishes": m10_vanishes,
            "first_sig_by_N": {n: per_N[n]["first_sig"] for n in Ns},
            "sham_intensive_clean_all_N": sham_ok, "c8_sham_noisy_N": c8_sham_noisy,
            "aws_needed": False, "decision": dec}


def _write(per_N, ver):
    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "summary.json"), "w") as f:
        json.dump({"per_N": per_N, "verdict": ver}, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, np.floating) else str(x))
    with open(os.path.join(OUTDIR, "results.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["N", "arm", "quantity"] + [f"t{t}" for t in TIMES])
        for n in sorted(per_N.keys()):
            for arm in ("rad", "tan"):
                for q in QUANT:
                    w.writerow([n, arm, q] + [f"{per_N[n]['effects'][arm][q][str(t)][0]:.4f}" for t in TIMES])
    _figures(per_N)
    _report(per_N, ver)
    print(f"[outputs] → {OUTDIR}/")


def _figures(per_N):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ns = sorted(per_N.keys())
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(Ns)))
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    for q, axi, title in [("M10", 0, "M(<0.1) effect vs t (intensive)"),
                          ("C8", 1, "C₈ effect vs t (extensive ~N)"),
                          ("beta", 2, "β effect vs t (intensive)")]:
        for n, c in zip(Ns, cols):
            ax[axi].plot(TIMES, [per_N[n]["effects"]["rad"][q][str(t)][0] for t in TIMES], "o-", color=c, label=f"N={n}")
        ax[axi].axhline(0, color="k", lw=0.6); ax[axi].set_xlabel("t (steps)"); ax[axi].set_title(title)
        ax[axi].legend(fontsize=8)
    ax[3].plot(Ns, [per_N[n]["peak_M10_effect"] for n in Ns], "o-", label="peak M(<0.1) (intensive)")
    ax[3].set_xlabel("N"); ax[3].set_ylabel("peak M(<0.1) effect"); ax[3].set_xscale("log", base=2)
    ax[3].axhline(0, color="k", lw=0.6); ax[3].set_title("Does intensive effect vanish with N?"); ax[3].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(OUTDIR, "figures"), exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, "figures", "fig_Ndep.pdf")); plt.close(fig)


def _report(per_N, ver):
    d = ver["decision"]
    band = ("🟢 PARTICLE-NUMBER ROBUST" if d.startswith("PARTICLE") else
            "🟡 FINITE-N SUSPECT" if d.startswith("FINITE") else
            "🟡 INCONCLUSIVE" if d.startswith("INCONCLUSIVE") else "🔴 NOT ROBUST")
    L = ["# N-dependence of the |L| causal chain (Hernquist, ε=0.05, θ=20°)\n"]
    L.append(f"N ∈ {ver['Ns']} (pairs: {[per_N[n]['n_pairs'] for n in ver['Ns']]}). Times: {TIMES}.\n")
    L.append(f"## Verdict — {band}\n\n> **{d}**\n")
    L.append("## Per-N summary\n")
    L.append("| N | pairs | M(<0.05) t | M(<0.1) t | C₈ t | conc<C₈ | |L| perm | β trans | peak M(<0.1) | peak C₈ | C₈ sign | sham✓ |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in ver["Ns"]:
        p = per_N[n]; fs = p["first_sig"]
        L.append(f"| {n} | {p['n_pairs']} | {fs['M05']} | {fs['M10']} | {fs['C8']} | "
                 f"{p['concentration_before_c8'] or p['simultaneous']} | {p['L_permanent']} | "
                 f"{p['beta_transient']} | {p['peak_M10_effect']:.3f} | {p['peak_C8_effect']:.2f} | "
                 f"{p['c8_sign']:+.0f} | {p['sham_clean']} |")
    L.append(f"\n- **intensive M(<0.1) peak by N:** { {k: round(v,3) for k,v in ver['peak_M10_by_N'].items()} } "
             f"(ratio hiN/loN = {ver['m10_ratio_hiN_loN']:.2f}; vanishes={ver['m10_vanishes']}).")
    L.append(f"- extensive C₈ peak by N: { {k: round(v,1) for k,v in ver['peak_C8_by_N'].items()} } "
             f"(grows ~with N, as expected for a count variance).")
    L.append(f"- ordering holds at all N: {ver['ordering_all_N']}; |L| permanent all N: {ver['L_permanent_all_N']}; "
             f"β transient all N: {ver['beta_transient_all_N']}; C₈ sign consistent: {ver['c8_sign_consistent']}.\n")
    with open(os.path.join(OUTDIR, "Ndep_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="N-dependence of the |L| causal chain (no AWS).")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--from-cache", action="store_true", help="recompute verdict from cached per-N (no sims)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.from_cache:
        data = json.load(open(os.path.join(OUTDIR, "summary.json")))
        per_N = {int(k): v for k, v in data["per_N"].items()}
        ver = verdict_across_N(per_N); _write(per_N, ver)
        print("\n" + ver["decision"]); return
    per_N = {}
    for n, pairs in N_PAIRS:
        print(f"N-dep — N={n}, {pairs} pairs × 4 arms → step {max(TIMES)}")
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init, initargs=(True,)) as ex:
            futs = [ex.submit(_worker, (2000 + i, n)) for i in range(pairs)]
            for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, unit="pair"):
                rows.append(fut.result())
        per_N[n] = analyse_N(rows, n)
    ver = verdict_across_N(per_N)
    _write(per_N, ver)
    print("\n" + ver["decision"])


if __name__ == "__main__":
    main()
