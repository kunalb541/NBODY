#!/usr/bin/env python3
"""
nbody_headroom_scale_analysis.py — is the low-pericenter handle a scale-free headroom law?
==========================================================================================

Reads ONLY committed AWS battery rows under `outputs/nbody_aws_battery/`. NO new simulations.

Tests whether ΔM(<r) ≈ Δf_peri(<r) × H(r), with H set by baseline *unsaturation*, holds across
r_c ∈ {0.05, 0.10, 0.20}. Finding: NO scale-free law.
  • dose advantage  = genuine deep-pericenter headroom (only clear at r<0.05);
  • slope advantage = potential-shape / orbit-deposition effect (NOT unsaturation);
  • overall Plummer/Hernquist advantage is scale-specific, peaking near r ≈ a/2 (a=0.20).
Writes `headroom_scale.json` and `figures/fig5_headroom_scale.png`.
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = "outputs/nbody_aws_battery"
FIG = f"{D}/figures"
os.makedirs(FIG, exist_ok=True)
NS = [512, 1024, 2048, 4096]
TIMES = [5, 10, 20, 50, 100, 300, 600, 1000]
RCS = [("005", "M05", 0.05), ("01", "M10", 0.10), ("02", "M20", 0.20)]


def rows(cell):
    return [json.loads(l) for l in open(f"{D}/{cell}/rows.jsonl")]


def per_rc(cell):
    R = rows(cell); o = {}
    for fk, mk, rc in RCS:
        dose = np.mean([r["full"]["fperi_a"][fk] - r["sham"]["fperi_a"][fk] for r in R])
        eff = [np.mean([r["full"]["series"][str(t)][mk] - r["sham"]["series"][str(t)][mk] for r in R]) for t in TIMES]
        dM = max(eff, key=abs)
        baseM = np.mean([r["orig"]["series"]["0"][mk] for r in R])
        basef = np.mean([r["orig"]["fperi_a"][fk] for r in R])
        o[rc] = {"dose": float(dose), "dM": float(dM),
                 "H": float(dM / dose) if abs(dose) > 1e-12 else float("nan"),
                 "baseM": float(baseM), "basef": float(basef)}
    return o


def main():
    agg = {}
    for prof in ["hernquist3d", "plummer3d"]:
        cells = [per_rc(f"{prof}_N{N}_eps0.05") for N in NS]
        for _, _, rc in RCS:
            agg[(prof, rc)] = {k: float(np.mean([c[rc][k] for c in cells]))
                               for k in ["dose", "dM", "H", "baseM", "basef"]}
    print(f'{"profile":11s} {"r_c":>5} {"dose":>7} {"dM":>8} {"H":>7} {"baseM":>7} {"basef":>7}')
    for prof in ["hernquist3d", "plummer3d"]:
        for _, _, rc in RCS:
            a = agg[(prof, rc)]
            print(f'{prof:11s} {rc:5.2f} {a["dose"]:7.3f} {a["dM"]:8.4f} {a["H"]:7.3f} {a["baseM"]:7.3f} {a["basef"]:7.3f}')

    rcs = [0.05, 0.10, 0.20]
    ratios = {rc: {"dM": agg[("plummer3d", rc)]["dM"] / agg[("hernquist3d", rc)]["dM"],
                   "dose": agg[("plummer3d", rc)]["dose"] / agg[("hernquist3d", rc)]["dose"],
                   "slope": agg[("plummer3d", rc)]["H"] / agg[("hernquist3d", rc)]["H"]} for rc in rcs}
    print("\nPlummer/Hernquist ratios by r_c:")
    for rc in rcs:
        print(f"  r={rc}: dM x{ratios[rc]['dM']:.2f}  dose x{ratios[rc]['dose']:.2f}  slope x{ratios[rc]['slope']:.2f}")
    H = np.array([agg[k]["H"] for k in agg]); bM = np.array([agg[k]["baseM"] for k in agg])
    corrHbM = float(np.corrcoef(H, bM)[0, 1])
    print(f"\ncorr(H, baseM) over 6 pts = {corrHbM:+.2f}  (positive => H tracks geometry/sphere-size, NOT unsaturation)")
    print(f"r=0.1: slope ratio x{ratios[0.10]['slope']:.2f} at ~equal baseM "
          f"(P {agg[('plummer3d',0.10)]['baseM']:.3f} vs H {agg[('hernquist3d',0.10)]['baseM']:.3f}) "
          f"=> slope advantage is NOT unsaturation")

    json.dump({"per_profile_rc": {f"{p}_{rc}": agg[(p, rc)] for p, rc in agg},
               "ratios_by_rc": ratios, "corr_H_baseM": corrHbM},
              open(f"{D}/headroom_scale.json", "w"), indent=2)

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    axs[0].axvspan(0.085, 0.115, color="0.9", zorder=0)
    axs[0].plot(rcs, [ratios[rc]["dM"] for rc in rcs], "o-", label="dM (response)")
    axs[0].plot(rcs, [ratios[rc]["dose"] for rc in rcs], "s--", label="dose df_peri")
    axs[0].plot(rcs, [ratios[rc]["slope"] for rc in rcs], "^--", label="slope H")
    axs[0].axhline(1, color="0.5", lw=0.8)
    axs[0].set_xlabel("r_c"); axs[0].set_ylabel("Plummer / Hernquist")
    axs[0].set_title("Advantage is scale-specific (peaks r~0.1=a/2)")
    axs[0].set_xticks(rcs); axs[0].legend()
    for prof, mk in [("hernquist3d", "o-"), ("plummer3d", "s-")]:
        axs[1].plot(rcs, [agg[(prof, rc)]["H"] for rc in rcs], mk, label=f"H {prof.replace('3d','')}")
    for prof, mk in [("hernquist3d", "o:"), ("plummer3d", "s:")]:
        axs[1].plot(rcs, [agg[(prof, rc)]["baseM"] for rc in rcs], mk, color="0.6",
                    label=f"baseM {prof.replace('3d','')}")
    axs[1].set_xlabel("r_c"); axs[1].set_ylabel("H (slope) and baseline M(<r)")
    axs[1].set_title("H rises WITH baseline mass (geometry, not headroom)")
    axs[1].set_xticks(rcs); axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_headroom_scale.png", dpi=150); plt.close(fig)
    print(f"\n[written] {D}/headroom_scale.json, {FIG}/fig5_headroom_scale.png")


if __name__ == "__main__":
    main()
