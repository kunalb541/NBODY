#!/usr/bin/env python3
"""
make_nbody_battery_figures.py — audit/paper figures from the committed AWS battery.
===================================================================================
Reproducible from committed JSON only (`outputs/nbody_aws_battery/*/cell_summary.json`,
`verdict.json`, `plummer_vs_hernquist.json`). NO new simulations. Writes PNGs to
`outputs/nbody_aws_battery/figures/`.

Figures:
  1. verdict map  — concentrated cells over N x eps, pass/fail, borderline miss marked, 0 kills.
  2. dM10 vs N and vs eps — Hernquist vs Plummer (Plummer ~2x, flat to N=4096).
  3. controls panel — concentrated vs uniform/bimodal (mechanism absent in controls).
  4. dose x slope decomposition — why Plummer is 2x (dose x slope = dM10 ratio).
"""
from __future__ import annotations

import glob
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
EPSS = [0.02, 0.05, 0.1, 0.2]

CELLS = {}
for p in glob.glob(f"{D}/*/cell_summary.json"):
    d = json.load(open(p))
    CELLS[(d["profile"], d["N"], d["eps"])] = d
PVH = json.load(open(f"{D}/plummer_vs_hernquist.json"))
nkills = sum(1 for c in CELLS.values() for k, v in c["kill"].items() if v)


# ── Fig 1: verdict map ──────────────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, prof in zip(axs, ["hernquist3d", "plummer3d"]):
    M = np.full((len(NS), len(EPSS)), np.nan)
    for i, N in enumerate(NS):
        for j, eps in enumerate(EPSS):
            c = CELLS.get((prof, N, eps))
            if c is not None:
                M[i, j] = 1.0 if c["CELL_PASS"] else 0.0
    ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i, N in enumerate(NS):
        for j, eps in enumerate(EPSS):
            c = CELLS.get((prof, N, eps))
            if c is None:
                ax.text(j, i, "n/a", ha="center", va="center", color="0.5", fontsize=8)
            elif c["CELL_PASS"]:
                ax.text(j, i, "PASS", ha="center", va="center", fontsize=8)
            else:
                r = c["crit3"]["dM_in"] / max(c["crit3"]["dM_out"], 1e-9)
                ax.text(j, i, f"FAIL\nloc {r:.2f}\n(vs 3.0)", ha="center", va="center",
                        fontsize=7, fontweight="bold")
    ax.set_xticks(range(len(EPSS))); ax.set_xticklabels(EPSS)
    ax.set_yticks(range(len(NS))); ax.set_yticklabels(NS)
    ax.set_xlabel("eps (softening)"); ax.set_ylabel("N")
    ax.set_title(prof.replace("3d", ""))
fig.suptitle(f"Battery verdict map (concentrated cells) — {nkills} kill tests fired across all 34 cells")
fig.tight_layout()
fig.savefig(f"{FIG}/fig1_verdict_map.png", dpi=150); plt.close(fig)


# ── Fig 2: dM10 vs N and vs eps ─────────────────────────────────────────────────
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for prof, mk in [("hernquist3d", "o-"), ("plummer3d", "s-")]:
    ys = [CELLS[(prof, N, 0.05)]["peak_dM10_full"] for N in NS]
    axs[0].plot(NS, ys, mk, label=prof.replace("3d", ""))
axs[0].set_xscale("log", base=2); axs[0].set_xticks(NS); axs[0].set_xticklabels(NS)
axs[0].set_xlabel("N"); axs[0].set_ylabel("peak dM(<0.1)  [full - sham]")
axs[0].set_title("Response vs N (eps=0.05): Plummer ~2x, flat to N=4096")
axs[0].set_ylim(0, None); axs[0].legend(); axs[0].grid(alpha=0.3)
for prof, mk in [("hernquist3d", "o-"), ("plummer3d", "s-")]:
    ys = [CELLS[(prof, 2048, e)]["peak_dM10_full"] for e in EPSS]
    axs[1].plot(EPSS, ys, mk, label=prof.replace("3d", ""))
axs[1].set_xlabel("eps (softening)"); axs[1].set_ylabel("peak dM(<0.1)")
axs[1].set_title("Response vs eps (N=2048)")
axs[1].set_ylim(0, None); axs[1].legend(); axs[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG}/fig2_dM10_vs_N_eps.png", dpi=150); plt.close(fig)


# ── Fig 3: controls panel ───────────────────────────────────────────────────────
profs = ["hernquist3d", "plummer3d", "uniform3d", "bimodal3d"]
labels = [p.replace("3d", "") for p in profs]
colors = ["C0", "C1", "0.5", "0.7"]
dM10 = [np.mean([CELLS[(p, N, 0.05)]["peak_dM10_full"] for N in NS if (p, N, 0.05) in CELLS]) for p in profs]
pcf = [np.mean([CELLS[(p, N, 0.05)]["predictor"]["pc_f"] for N in NS if (p, N, 0.05) in CELLS]) for p in profs]
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
axs[0].bar(labels, dM10, color=colors)
axs[0].set_ylabel("peak dM(<0.1)"); axs[0].set_title("Concentration response (~5x smaller in controls)")
axs[1].bar(labels, pcf, color=colors); axs[1].axhline(0, color="k", lw=0.8)
axs[1].set_ylabel("f_peri partial-corr with dM (control for global-L)")
axs[1].set_title("Does f_peri predict dM? (>0 = mechanism present)")
fig.suptitle("Controls: low-pericenter mechanism present in concentrated, ABSENT in uniform/bimodal")
fig.tight_layout()
fig.savefig(f"{FIG}/fig3_controls.png", dpi=150); plt.close(fig)


# ── Fig 4: dose x slope decomposition ───────────────────────────────────────────
r = PVH["ratios_eps05_meanN"]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
labels = ["dose\nratio", "slope\nratio", "dose x slope\nproduct", "measured\ndM10 ratio"]
vals = [r["dose_ratio_P_over_H"], r["slope_ratio_P_over_H"], r["dose_x_slope_product"], r["dM10_ratio_P_over_H"]]
bars = ax.bar(labels, vals, color=["C2", "C3", "C4", "k"])
ax.axhline(1.0, color="0.5", ls="--", lw=0.8)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"x{v:.2f}", ha="center", fontsize=10)
ax.set_ylabel("Plummer / Hernquist"); ax.set_ylim(0, max(vals) * 1.18)
ax.set_title("Why Plummer is ~2x stronger: dose x slope = dM10 (product = measured)")
fig.tight_layout()
fig.savefig(f"{FIG}/fig4_dose_slope.png", dpi=150); plt.close(fig)

print("wrote:")
for f in sorted(glob.glob(f"{FIG}/*.png")):
    print(" ", f, f"({os.path.getsize(f)//1024} KB)")
