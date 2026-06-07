#!/usr/bin/env python3
"""
nbody_battery_plummer_vs_hernquist.py — why is Plummer ~2x stronger than Hernquist?
===================================================================================

Reproducible decomposition of the low-pericenter concentration response, reading ONLY the
committed AWS battery rows under `outputs/nbody_aws_battery/`.  NO new simulations.

Causal chain:  Δf_peri  →  ΔM(<0.1).  Decompose the peak response as

    ΔM10  =  slope × dose ,   dose = mean Δf_peri(<0.1) (full − sham, t0),
                              slope = mean ΔM10 / dose ,

and compare Plummer (core) vs Hernquist (cusp): ratios, N- and ε-dependence, and the baseline
deep-pericenter "headroom" f_peri(<0.05).  Writes `outputs/nbody_aws_battery/plummer_vs_hernquist.json`.
"""
from __future__ import annotations

import json
import os

import numpy as np

D = "outputs/nbody_aws_battery"
NS = [512, 1024, 2048, 4096]
EPSS = [0.02, 0.05, 0.1, 0.2]


def _tstar(cell):
    return json.load(open(f"{D}/{cell}/cell_summary.json"))["crit1"]["01"]["tstar"]


def _rows(cell):
    return [json.loads(l) for l in open(f"{D}/{cell}/rows.jsonl")]


def decomp(cell):
    ts = str(_tstar(cell)); R = _rows(cell)
    bfp1 = float(np.mean([r["orig"]["fperi_a"]["01"] for r in R]))
    bfp5 = float(np.mean([r["orig"]["fperi_a"]["005"] for r in R]))
    bM = float(np.mean([r["orig"]["series"]["0"]["M10"] for r in R]))
    dose = float(np.mean([r["full"]["fperi_a"]["01"] - r["sham"]["fperi_a"]["01"] for r in R]))
    dose_deep = float(np.mean([r["full"]["fperi_a"]["005"] - r["sham"]["fperi_a"]["005"] for r in R]))
    dM = float(np.mean([r["full"]["series"][ts]["M10"] - r["sham"]["series"][ts]["M10"] for r in R]))
    slope = dM / dose if abs(dose) > 1e-12 else float("nan")
    return {"cell": cell, "n": len(R), "tstar": int(ts), "base_fp01": bfp1, "base_fp005": bfp5,
            "base_M10": bM, "dose": dose, "dose_deep": dose_deep, "dM10": dM, "slope": slope}


def main():
    rows = {}
    print(f"{'cell':30s} {'n':>4} {'t*':>3} {'base_fp.1':>9} {'base_fp.05':>10} "
          f"{'dose':>7} {'dose_deep':>9} {'ΔM10':>8} {'slope':>7}")
    for prof in ["hernquist3d", "plummer3d"]:
        for N in NS:
            for eps in (EPSS if N != 4096 else [0.05]):
                cell = f"{prof}_N{N}_eps{eps}"
                if not os.path.exists(f"{D}/{cell}/rows.jsonl"):
                    continue
                d = decomp(cell); rows[cell] = d
                print(f"{cell:30s} {d['n']:4d} {d['tstar']:3d} {d['base_fp01']:9.3f} "
                      f"{d['base_fp005']:10.3f} {d['dose']:7.3f} {d['dose_deep']:9.3f} "
                      f"{d['dM10']:8.4f} {d['slope']:7.3f}")

    def agg(prof, key):
        vals = [rows[f"{prof}_N{N}_eps0.05"][key] for N in NS
                if f"{prof}_N{N}_eps0.05" in rows]
        return float(np.mean(vals))

    H = {k: agg("hernquist3d", k) for k in ["dose", "slope", "dM10", "base_fp005"]}
    P = {k: agg("plummer3d", k) for k in ["dose", "slope", "dM10", "base_fp005"]}
    dr, sr, mr = P["dose"] / H["dose"], P["slope"] / H["slope"], P["dM10"] / H["dM10"]
    print("\n=== Plummer / Hernquist (ε=0.05, mean over N) ===")
    print(f"  dose : Hern {H['dose']:.3f}  Plum {P['dose']:.3f}  ratio ×{dr:.2f}")
    print(f"  slope: Hern {H['slope']:.3f}  Plum {P['slope']:.3f}  ratio ×{sr:.2f}")
    print(f"  ΔM10 : Hern {H['dM10']:.4f}  Plum {P['dM10']:.4f}  ratio ×{mr:.2f}")
    print(f"  dose×slope product = ×{dr*sr:.2f}   (vs measured ΔM10 ratio ×{mr:.2f})")
    print(f"  baseline deep f_peri(<0.05): Hern {H['base_fp005']:.3f}  Plum {P['base_fp005']:.3f} "
          f"(cusp pre-loaded; core has headroom)")

    out = {"per_cell": rows,
           "ratios_eps05_meanN": {"dose_ratio_P_over_H": dr, "slope_ratio_P_over_H": sr,
                                  "dM10_ratio_P_over_H": mr, "dose_x_slope_product": dr * sr,
                                  "baseline_deep_fperi_hern": H["base_fp005"],
                                  "baseline_deep_fperi_plum": P["base_fp005"]}}
    json.dump(out, open(f"{D}/plummer_vs_hernquist.json", "w"), indent=2)
    print(f"\n[written] {D}/plummer_vs_hernquist.json")


if __name__ == "__main__":
    main()
