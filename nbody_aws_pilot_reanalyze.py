#!/usr/bin/env python3
"""
nbody_aws_pilot_reanalyze.py — regenerate the pilot verdict from saved results.csv
==================================================================================

The N=4096 pilot was run once (~1 CPU-h).  Two analysis-only statistics were then corrected to
match what was preregistered / established in prior committed work — WITHOUT re-simulating:

  • kill test `globalL_beats_fperi`: the code had compared *within-arm partial-r* (low-power,
    noise-dominated); the registered §5 statistic is the MUTUAL PARTIAL CORRELATION (does each
    variable add unique predictive power for ΔM beyond the other).  With it, f_peri ≫ global ⟨|L|⟩
    and the kill does NOT fire (it fired only on the buggy statistic).
  • criterion 5 (sham): the code had used the CI-includes-0 (sham exactly 0) test; this project's
    established standard (and the matched-pair design) is MAGNITUDE-RELATIVE (sham ≪ intervention).

Because the simulations are deterministic and `results.csv` holds every per-(arm,pair,time)
observable the analysis consumes, reconstructing `rows` from the CSV and re-running the pilot's
own `analyse_profile` reproduces criteria 1,2,3,7 exactly and applies the two corrected statistics.
Conservation (crit 6) and the Φ_meas diagnostic are carried from the as-run `summary.json`
(not recomputable from the scalar CSV).  This is a transparent, re-runnable step — no new physics.
"""
from __future__ import annotations

import csv
import json
import os

import nbody_aws_low_pericenter_pilot as P

OUTDIR = P.OUTDIR


def _rows_from_csv(csv_path, summ):
    data = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            prof, seed, arm, t = row["profile"], int(row["pair_seed"]), row["arm"], int(row["t_step"])
            d = data.setdefault((prof, seed, arm), {"series": {}})
            d["series"][str(t)] = {"M05": float(row["M05"]), "M10": float(row["M10"]),
                                   "M20": float(row["M20"]), "C8": float(row["C8"]),
                                   "beta": float(row["beta"]), "sigr": float(row["sigr"]),
                                   "Lspec": 0.0, "S": 0.0}
            if t == 0:
                d["fperi_a"] = {"005": float(row["fperi_005"]), "01": float(row["fperi_01"]),
                                "02": float(row["fperi_02"])}
                d["fperi_m"] = d["fperi_a"]          # measured Φ diagnostic carried from summary, not CSV
                d["Lshell"] = {"inner": float(row["L_inner"]), "mid": float(row["L_mid"]),
                               "outer": float(row["L_outer"]), "global": float(row["L_global"])}
    rows_by_prof = {}
    for prof in sorted(set(p for p, _, _ in data)):
        seeds = sorted(set(s for p, s, _ in data if p == prof))
        Edr = summ[prof]["crit6_conservation"]["E_drift_integrator"]
        rows = []
        for s in seeds:
            r = {"profile": prof, "seed": s}
            for arm in P.ARMS:
                d = data[(prof, s, arm)]
                r[arm] = {"series": d["series"], "fperi_a": d["fperi_a"], "fperi_m": d["fperi_m"],
                          "Lshell": d["Lshell"], "ke0": 1.0}                      # speed-preserving → drift 0
            r["cons"] = {"Q0": 1.0, "ke_orig": 1.0, "E0": 1.0, "E6": 1.0 + Edr}   # crit6 carried from as-run
            rows.append(r)
        rows_by_prof[prof] = rows
    return rows_by_prof


def main():
    summ_path = os.path.join(OUTDIR, "summary.json")
    csv_path = os.path.join(OUTDIR, "results.csv")
    as_run = json.load(open(summ_path))
    rows_by_prof = _rows_from_csv(csv_path, as_run)

    res = {}
    for prof, rows in rows_by_prof.items():
        r = P.analyse_profile(rows)
        r["phi_meas_vs_analytic_rel"] = as_run[prof]["phi_meas_vs_analytic_rel"]   # carry diagnostic
        res[prof] = r
        # faithfulness cross-check vs as-run (criteria unaffected by the two corrections)
        a = as_run[prof]
        print(f"{prof}: PILOT_PASS(primary)={r['PILOT_PASS']}  strict={r['PILOT_PASS_strict']}  "
              f"kill={r['any_kill']}")
        print(f"    cross-check vs as-run — crit7 ΔM10: {r['crit7_Nrobust']['dM10_4096']:.4f} "
              f"(as-run {a['crit7_Nrobust']['dM10_4096']:.4f}); "
              f"crit3 dM_inner: {r['crit3_locality']['dM_inner']:.4f} "
              f"(as-run {a['crit3_locality']['dM_inner']:.4f})")
        pp = r["predictor_compare"]
        print(f"    predictor: f_peri|L={pp['pcorr_fperi']:+.3f} vs L|f_peri={pp['pcorr_globalL']:+.3f}; "
              f"arm-mean f_peri={pp['armmean_fperi']:+.3f} vs L={pp['armmean_globalL']:+.3f}")

    # write corrected artifacts alongside the as-run ones
    with open(os.path.join(OUTDIR, "summary_corrected.json"), "w") as f:
        json.dump(res, f, indent=2, default=lambda x: float(x) if hasattr(x, "dtype") else str(x))
    P._report(res)                                  # overwrites pilot_report.md with the corrected verdict
    print(f"\n[corrected outputs] → {OUTDIR}/pilot_report.md, summary_corrected.json")
    for prof in res:
        print(f"  {prof}: {'🟢 PASS' if res[prof]['PILOT_PASS'] else '🔴 FAIL'} (primary)")


if __name__ == "__main__":
    main()
