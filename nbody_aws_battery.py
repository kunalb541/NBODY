#!/usr/bin/env python3
"""
nbody_aws_battery.py — checkpointed low-pericenter causal battery (local or cloud)
==================================================================================

Generalizes the VERIFIED pilot worker (`nbody_aws_low_pericenter_pilot.py`) over the preregistered
grid — **no new physics**.  Reuses the pilot's profile-correct potential, per-pair regression dose,
registered partial-correlation kill test, and magnitude-relative sham (CI-only as diagnostic).

Grid (principled N=4096-ε trim — ε-robustness already settled at N≤2048; the only new N=4096
question is finite-N survival, which ε=0.05 answers):
  • primary {hernquist3d, plummer3d}: N∈{512,1024,2048} × ε∈{0.02,0.05,0.10,0.20}; N=4096 × ε=0.05
  • controls {uniform3d (negative), bimodal3d (geometry contrast)}: all N × ε=0.05
  • arms (10): orig · sham · tangentialize · inner-{weak10°,med20°,strong35°} · mid · outer · full · vt-rescale
  • times: preregistered {0,5,10,20,50,100,300,600,1000}
  • pairs: 100 (N≤2048), 50 (N=4096)
→ 34 cells.  Potential: analytic for hern/plum; measured Φ_meas (registered §3a) for the controls
  (no analytic Φ).  bimodal3d caveat: its radial intervention / pericenter use the GLOBAL centre — a
  spherical approximation for a two-clump geometry; its meaningful read is the C₈ geometry contrast.

Checkpointing (per cell = profile/N/ε): rows append to `cell_*/rows.jsonl` incrementally (flush per
pair); a completed cell writes `cell_summary.json`; `manifest.json` records status/seed-range/
timestamp/git-commit/config.  `--resume` skips done cells and continues partial ones from rows.jsonl.

Stop rule: the first cell run is hernquist3d/N1024/ε0.05; if it fails the registered criteria the
run halts before spending the night.  **Does not auto-launch** — `--smoke` validates the machinery;
the full grid runs only when invoked explicitly (see __main__).  No AWS required; paper.tex untouched.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

from nbody_3d import _worker_init
from nbody_stress import StressConfig, get_initial_conditions, get_simconfig, _integrate_leapfrog
import phase_space_coarse_features as psf
from nbody_intervention_pilot import intervene_anisotropy, sham_rotation
from nbody_anisotropy_mechanism_pilot import tangentialize
from nbody_orbital_summary import shell_radialize
from nbody_intervention_Lhandle import vt_scale
from nbody_intervention_timecourse import _obs, _paired, _ci_pos
import nbody_aws_low_pericenter_pilot as P     # verified helpers: _phi_*, _pericenters, _fperi, _lshells, _regress, _slope_ci, _partial_r, _pcorr

OUTROOT = "outputs/nbody_aws_battery"
A, THETA, VT_F = 0.20, 20.0, 0.5
TIMES = [0, 5, 10, 20, 50, 100, 300, 600, 1000]
PRIMARY = ["hernquist3d", "plummer3d"]
CONTROL = ["uniform3d", "bimodal3d"]
EPS_FULL = [0.02, 0.05, 0.10, 0.20]
ARMS = ["orig", "sham", "tan", "inner-weak", "inner-med", "inner-strong", "mid", "outer", "full", "vt"]
INTERV = ["tan", "inner-weak", "inner-med", "inner-strong", "mid", "outer", "full", "vt"]
RCS = P.RCS                                    # [("005","M05",0.05),("01","M10",0.10),("02","M20",0.20)]
STOP_CELL = ("hernquist3d", 1024, 0.05)        # stop-rule / smoke cell


def _jd(x):
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=os.path.dirname(os.path.abspath(__file__))).decode().strip()
    except Exception:
        return "unknown"


def _phi_for(profile, r_eval, r_particles):
    """analytic Φ for the concentrated profiles (exact); measured Φ_meas for the controls (§3a)."""
    if profile in ("hernquist3d", "plummer3d"):
        return P._phi_analytic(r_eval, profile)
    return P._phi_measured(r_eval, r_particles)


def build_grid(pairs_small=100, pairs_4096=50):
    """Stop-rule cell first, then the rest. Each entry = (profile, N, eps, pairs)."""
    cells = [(STOP_CELL[0], STOP_CELL[1], STOP_CELL[2], pairs_small)]
    for prof in PRIMARY:
        for N in [512, 1024, 2048]:
            for eps in EPS_FULL:
                if (prof, N, eps) != STOP_CELL:
                    cells.append((prof, N, eps, pairs_small))
        cells.append((prof, 4096, 0.05, pairs_4096))
    for prof in CONTROL:
        for N in [512, 1024, 2048]:
            cells.append((prof, N, 0.05, pairs_small))
        cells.append((prof, 4096, 0.05, pairs_4096))
    return cells


# ── per-pair worker (generalized verified pilot worker) ─────────────────────────
def _cell_worker(task):
    profile, N, eps, seed = task
    cfg = StressConfig(model="direct_isolated", init=profile, seed=seed, n=N, steps=max(TIMES),
                       eps=eps, box_size=2.0, k_fine=16, plummer_a=A)
    sc = get_simconfig(cfg); mass = 1.0 / N
    pos0, vel0 = get_initial_conditions(cfg); center = np.mean(pos0, axis=0); th = math.radians(THETA)
    vels = {
        "orig": vel0,
        "sham": sham_rotation(pos0, vel0, th, seed),
        "tan": tangentialize(pos0, vel0, th, center, seed),
        "inner-weak": shell_radialize(pos0, vel0, math.radians(10.0), 0.0, 33.333, center),
        "inner-med": shell_radialize(pos0, vel0, math.radians(20.0), 0.0, 33.333, center),
        "inner-strong": shell_radialize(pos0, vel0, math.radians(35.0), 0.0, 33.333, center),
        "mid": shell_radialize(pos0, vel0, th, 33.333, 66.667, center),
        "outer": shell_radialize(pos0, vel0, th, 66.667, 100.0, center),
        "full": intervene_anisotropy(pos0, vel0, th, center),
        "vt": vt_scale(pos0, vel0, VT_F, center),
    }
    r0 = np.linalg.norm(pos0 - center, axis=1)
    phig_a, phia_a = _phi_for(profile, P._RG, r0), _phi_for(profile, r0, r0)
    if profile in ("hernquist3d", "plummer3d"):
        phig_m, phia_m = P._phi_measured(P._RG, r0), P._phi_measured(r0, r0)
    else:
        phig_m, phia_m = phig_a, phia_a
    out = {"profile": profile, "N": N, "eps": eps, "seed": seed}
    ke_orig = 0.5 * mass * float(np.sum(vel0 * vel0))
    orig_snaps = None
    for name, v in vels.items():
        snaps = _integrate_leapfrog(pos0, v, mass, sc, TIMES, True)
        if name == "orig":
            orig_snaps = snaps
        rp_a, L, _ = P._pericenters(pos0, v, center, phig_a, phia_a)
        rp_m, _, _ = P._pericenters(pos0, v, center, phig_m, phia_m)
        out[name] = {"series": {str(t): _obs(snaps[t][0], snaps[t][1], cfg) for t in TIMES},
                     "fperi_a": P._fperi(rp_a), "fperi_m": P._fperi(rp_m),
                     "Lshell": P._lshells(L, r0), "ke0": 0.5 * mass * float(np.sum(v * v))}
    tf = max(TIMES)
    T0 = 0.5 * mass * float(np.sum(orig_snaps[0][1] ** 2)); Q0 = float(psf.relaxation_observables(orig_snaps[0][0], orig_snaps[0][1], cfg)["Q"])
    Tf = 0.5 * mass * float(np.sum(orig_snaps[tf][1] ** 2)); Qf = float(psf.relaxation_observables(orig_snaps[tf][0], orig_snaps[tf][1], cfg)["Q"])
    out["cons"] = {"Q0": Q0, "ke_orig": ke_orig,
                   "E0": T0 * (1.0 - 2.0 / Q0) if abs(Q0) > 1e-9 else float("nan"),
                   "E6": Tf * (1.0 - 2.0 / Qf) if abs(Qf) > 1e-9 else float("nan")}
    return out


# ── analysis (battery TIMES-aware; reuses pilot stat helpers) ────────────────────
def _eff(rows, arm, ref, key):
    return {t: _paired([r[arm]["series"][str(t)][key] - r[ref]["series"][str(t)][key] for r in rows]) for t in TIMES}


def _first_sig(eff):
    for t in TIMES:
        if t > 0 and _ci_pos(eff[t]):
            return t
    return None


def analyse_cell(rows, profile, N, eps):
    conc = profile in ("hernquist3d", "plummer3d")
    tstar = {}
    for fk, mk, _rc in RCS:
        e = _eff(rows, "full", "sham", mk)
        tstar[mk] = max((t for t in TIMES if t > 0), key=lambda t: abs(e[t][0]))

    crit1 = {}
    for fk, mk, _rc in RCS:
        ts = tstar[mk]; pbp = {}; triples = []
        for i, r in enumerate(rows):
            for a in INTERV:
                x = r[a]["fperi_a"][fk] - r["sham"]["fperi_a"][fk]
                y = r[a]["series"][str(ts)][mk] - r["sham"]["series"][str(ts)][mk]
                pbp.setdefault(i, []).append((x, y)); triples.append((a, x, y))
        slope, _b = P._regress([p for v in pbp.values() for p in v]); lo, hi = P._slope_ci(pbp); pr = P._partial_r(triples)
        amx = [np.mean([r[a]["fperi_a"][fk] - r["sham"]["fperi_a"][fk] for r in rows]) for a in INTERV]
        amy = [np.mean([r[a]["series"][str(ts)][mk] - r["sham"]["series"][str(ts)][mk] for r in rows]) for a in INTERV]
        amc = float(np.corrcoef(amx, amy)[0, 1]) if np.std(amx) > 1e-12 else float("nan")
        sp = math.isfinite(slope) and slope > 0 and math.isfinite(lo) and lo > 0
        crit1[fk] = {"tstar": ts, "slope": slope, "slope_ci": [lo, hi], "partial_r": pr,
                     "armmean_corr": amc, "pass": bool(sp), "strict_pass": bool(sp and math.isfinite(pr) and pr >= 0.5)}
    c1 = all(crit1[fk]["pass"] for fk, _m, _r in RCS)

    effM, effC = _eff(rows, "full", "sham", "M10"), _eff(rows, "full", "sham", "C8")
    tM, tC = _first_sig(effM), _first_sig(effC)
    c2 = (tM is not None) and (tC is None or tM <= tC)

    def peak(arm, mk):
        e = _eff(rows, arm, "sham", mk)
        return max((e[t][0] for t in TIMES if t > 0), key=abs)
    dM_in, dM_out = peak("inner-med", "M10"), peak("outer", "M10")
    dLg_in = float(np.mean([r["inner-med"]["Lshell"]["global"] - r["sham"]["Lshell"]["global"] for r in rows]))
    dLg_out = float(np.mean([r["outer"]["Lshell"]["global"] - r["sham"]["Lshell"]["global"] for r in rows]))
    c3 = (dM_in >= 3.0 * dM_out) and (abs(dLg_out) >= abs(dLg_in))

    shamM, shamB = _eff(rows, "sham", "orig", "M10"), _eff(rows, "sham", "orig", "beta")
    ts = tstar["M10"]; svf = abs(shamM[ts][0]) / max(abs(effM[ts][0]), 1e-12)
    c5 = bool(svf < 0.1); c5_strict = bool((not _ci_pos(shamM[ts])) and (not _ci_pos(shamB[ts])))

    ke_drift = float(np.median([max(abs(r[a]["ke0"] - r["cons"]["ke_orig"]) for a in INTERV) / max(r["cons"]["ke_orig"], 1e-30) for r in rows]))
    Edr = [abs(r["cons"]["E6"] - r["cons"]["E0"]) / max(abs(r["cons"]["E0"]), 1e-30)
           for r in rows if math.isfinite(r["cons"]["E0"]) and math.isfinite(r["cons"]["E6"])]
    c6 = ke_drift < 1e-2

    tsM = str(tstar["M10"])
    dfp = np.array([r[a]["fperi_a"]["01"] - r["sham"]["fperi_a"]["01"] for r in rows for a in INTERV])
    dL = np.array([r[a]["Lshell"]["global"] - r["sham"]["Lshell"]["global"] for r in rows for a in INTERV])
    dM = np.array([r[a]["series"][tsM]["M10"] - r["sham"]["series"][tsM]["M10"] for r in rows for a in INTERV])
    pc_f, pc_L = P._pcorr(dfp, dM, dL), P._pcorr(dL, dM, dfp)
    amf = float(np.corrcoef([np.mean([r[a]["fperi_a"]["01"] - r["sham"]["fperi_a"]["01"] for r in rows]) for a in INTERV],
                            [np.mean([r[a]["series"][tsM]["M10"] - r["sham"]["series"][tsM]["M10"] for r in rows]) for a in INTERV])[0, 1])
    amL = float(np.corrcoef([np.mean([r[a]["Lshell"]["global"] - r["sham"]["Lshell"]["global"] for r in rows]) for a in INTERV],
                            [np.mean([r[a]["series"][tsM]["M10"] - r["sham"]["series"][tsM]["M10"] for r in rows]) for a in INTERV])[0, 1])
    kill = {"globalL_beats_fperi": bool(math.isfinite(pc_L) and math.isfinite(pc_f) and pc_L > pc_f),
            "outer_reproduces": bool(dM_out > 0.6 * peak("full", "M10")),
            "sham_explains": bool(svf >= 0.5 or ke_drift > 1e-2)}

    phi_agree = None
    if conc:
        fa = np.array([r["full"]["fperi_a"]["01"] for r in rows]); fm = np.array([r["full"]["fperi_m"]["01"] for r in rows])
        phi_agree = float(np.median(np.abs(fm - fa) / np.maximum(np.abs(fa), 1e-9)))
    cell_pass = bool(c1 and c2 and c3 and c5 and c6 and not any(kill.values())) if conc else None
    return {"profile": profile, "N": N, "eps": eps, "n_pairs": len(rows), "concentrated": conc,
            "bimodal_pericenter_caveat": (profile == "bimodal3d"),
            "crit1": crit1, "crit1_pass": bool(c1),
            "crit2": {"tM": tM, "tC": tC, "pass": bool(c2)},
            "crit3": {"dM_in": dM_in, "dM_out": dM_out, "dLg_in": dLg_in, "dLg_out": dLg_out, "pass": bool(c3)},
            "crit5": {"sham_vs_full": svf, "pass": bool(c5), "ci_only_pass": c5_strict},
            "crit6": {"ke_drift": ke_drift, "E_drift": float(np.median(Edr)) if Edr else float("nan"), "pass": bool(c6)},
            "predictor": {"pc_f": pc_f, "pc_L": pc_L, "amf": amf, "amL": amL},
            "kill": kill, "phi_agree": phi_agree, "peak_dM10_full": peak("full", "M10"),
            "CELL_PASS": cell_pass}


# ── checkpointed cell runner ─────────────────────────────────────────────────────
def run_cell(profile, N, eps, pairs, workers, resume):
    cell_id = f"{profile}_N{N}_eps{eps}"
    cdir = os.path.join(OUTROOT, cell_id); os.makedirs(cdir, exist_ok=True)
    rows_path = os.path.join(cdir, "rows.jsonl")
    rows, done = [], set()
    if resume and os.path.exists(rows_path):
        for line in open(rows_path):
            line = line.strip()
            if line:
                r = json.loads(line); rows.append(r); done.add(r["seed"])
    todo = [2000 + i for i in range(pairs) if (2000 + i) not in done]
    if todo:
        with open(rows_path, "a" if resume else "w") as fout:
            if not resume:
                rows, done = [], set()
            with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(True,)) as ex:
                futs = {ex.submit(_cell_worker, (profile, N, eps, s)): s for s in todo}
                for fut in tqdm(as_completed(futs), total=len(futs), ncols=100, desc=cell_id, unit="pair"):
                    r = fut.result(); rows.append(r)
                    fout.write(json.dumps(r, default=_jd) + "\n"); fout.flush()
    res = analyse_cell(rows, profile, N, eps)
    json.dump(res, open(os.path.join(cdir, "cell_summary.json"), "w"), indent=2, default=_jd)
    return res, len(rows)


def aggregate(cells):
    by = {(c["profile"], c["eps"], c["N"]): c for c in cells}
    crit7 = {}
    for prof in PRIMARY:
        peaks = {N: (by.get((prof, 0.05, N)) or {}).get("peak_dM10_full") for N in [512, 1024, 2048, 4096]}
        if peaks.get(512) and peaks.get(4096):
            ratio = peaks[4096] / peaks[512] if peaks[512] else float("nan")
            crit7[prof] = {"peaks": peaks, "ratio_4096_512": ratio, "pass": bool(math.isfinite(ratio) and ratio > 0.4)}
    conc = [c for c in cells if c.get("concentrated")]
    overall = bool(conc) and all(c["CELL_PASS"] for c in conc) and all(v["pass"] for v in crit7.values())
    return {"crit7_Nrobust": crit7, "n_cells": len(cells),
            "concentrated_pass": sum(1 for c in conc if c["CELL_PASS"]), "concentrated_total": len(conc),
            "BATTERY_PASS": overall}


def write_verdict(cells, agg):
    os.makedirs(OUTROOT, exist_ok=True)
    json.dump({"cells": cells, "aggregate": agg}, open(os.path.join(OUTROOT, "verdict.json"), "w"), indent=2, default=_jd)
    L = ["# Low-pericenter battery — verdict\n",
         f"{agg['n_cells']} cells · arms {ARMS} · times {TIMES}\n",
         f"## {'🟢 BATTERY PASS' if agg['BATTERY_PASS'] else '🟡 see cells'} "
         f"({agg['concentrated_pass']}/{agg['concentrated_total']} concentrated cells pass)\n",
         "| profile | N | ε | CELL_PASS | dose r(arm-mean) | M→C₈ | f_peri vs L (pcorr) | peak ΔM10 |",
         "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        d = c["crit1"]["01"]
        L.append(f"| {c['profile']} | {c['N']} | {c['eps']} | "
                 f"{'✅' if c['CELL_PASS'] else ('—' if c['CELL_PASS'] is None else '❌')} | "
                 f"{d['armmean_corr']:+.2f} | {c['crit2']['tM']}→{c['crit2']['tC']} | "
                 f"{c['predictor']['pc_f']:+.2f}/{c['predictor']['pc_L']:+.2f} | {c['peak_dM10_full']:+.4f} |")
    L.append("\n## Criterion 7 — N-robustness (ε=0.05)")
    for prof, v in agg["crit7_Nrobust"].items():
        L.append(f"- {prof}: peak ΔM10 by N {v['peaks']} → ×{v['ratio_4096_512']:.2f} (>0.4 = {v['pass']})")
    L.append("\n_uniform3d = negative control; bimodal3d = geometry contrast (pericenter is a global-"
             "spherical approximation — read C₈, not f_peri)._")
    open(os.path.join(OUTROOT, "battery_report.md"), "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Checkpointed low-pericenter battery (no auto-launch).")
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--pairs-4096", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny one-cell machinery test (pairs=3), then stop")
    ap.add_argument("--smoke-pairs", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(OUTROOT, exist_ok=True)
    git = _git_commit()
    manifest_path = os.path.join(OUTROOT, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else {}

    if args.smoke:
        prof, N, eps = STOP_CELL
        print(f"SMOKE: one cell {prof} N{N} eps{eps} × {args.smoke_pairs} pairs (resume={args.resume})")
        res, ndone = run_cell(prof, N, eps, args.smoke_pairs, args.workers, args.resume)
        cid = f"{prof}_N{N}_eps{eps}"
        manifest[cid] = {"status": "done" if ndone >= args.smoke_pairs else "partial", "pairs_done": ndone,
                         "pairs_target": args.smoke_pairs, "seed_start": 2000, "seed_end": 2000 + args.smoke_pairs - 1,
                         "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git,
                         "config": {"profile": prof, "N": N, "eps": eps, "smoke": True}}
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        print(f"  pairs_done={ndone}  CELL_PASS={res['CELL_PASS']}  "
              f"dose arm-mean r={res['crit1']['01']['armmean_corr']:+.2f}  M→C8={res['crit2']['tM']}→{res['crit2']['tC']}")
        print(f"  checkpoint → {OUTROOT}/{cid}/rows.jsonl ({ndone} rows), cell_summary.json; manifest updated.")
        return

    # FULL GRID (explicit invocation only). Stop-rule cell first.
    cells = build_grid(args.pairs, args.pairs_4096)
    results = []
    for (prof, N, eps, pairs) in cells:
        cid = f"{prof}_N{N}_eps{eps}"
        if args.resume and manifest.get(cid, {}).get("status") == "done":
            cs = os.path.join(OUTROOT, cid, "cell_summary.json")
            if os.path.exists(cs):
                results.append(json.load(open(cs)))
            print(f"skip {cid} (done)"); continue
        res, ndone = run_cell(prof, N, eps, pairs, args.workers, args.resume)
        results.append(res)
        manifest[cid] = {"status": "done" if ndone >= pairs else "partial", "pairs_done": ndone,
                         "pairs_target": pairs, "seed_start": 2000, "seed_end": 2000 + pairs - 1,
                         "timestamp": datetime.datetime.now().isoformat(timespec="seconds"), "git_commit": git,
                         "config": {"profile": prof, "N": N, "eps": eps}}
        json.dump(manifest, open(manifest_path, "w"), indent=2)
        if (prof, N, eps) == STOP_CELL and not res["CELL_PASS"]:
            print(f"\n🛑 STOP RULE: first cell {cid} failed registered criteria — halting before the full grid.")
            return
    agg = aggregate(results)
    write_verdict(results, agg)
    print(f"\nBATTERY_PASS={agg['BATTERY_PASS']}  → {OUTROOT}/verdict.json, battery_report.md")


if __name__ == "__main__":
    main()
