"""
Script for paper：
  - Fig. 4a  concentrationclass × condition pooled effect（dose-response, yield only）
  - Fig. 4b  particle sizeclass × condition pooled effect（size-response, yield only）
  - Fig. 4c  Non-stress: NPs size × concentration operating windowheatmap
  - Fig. 4d  Stress:     NPs size × concentration operating windowheatmap

per the paper Methods §"Dosage Data Harmonization"：
onlywithusing mass-per-volume / mass-per-soil-mass classconcentrationdata；
kg/ha etc.area-application units excluded(paper line 267）。

Tool: Python + DL random-effects pooling

run：python Fig4_concentration_size_regimes.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "code", "new_figures"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta

if __name__ == "__main__":
    df = load_all()
    sub = df[df["Performance"] == "Yield"]
    sub = sub.dropna(subset=["lnRR", "vi", "Concentration", "NPs_size"])
    sub = sub[sub["vi"] > 0].copy()

    # Concentration bins (mg/L or mg/kg)
    sub["conc_bin"] = pd.cut(sub["Concentration"],
                              bins=[0, 50, 100, 300, 1000, np.inf],
                              labels=["0–50","50–100","100–300","300–1000",">1000"],
                              include_lowest=True)
    sub["size_bin"] = pd.cut(sub["NPs_size"],
                              bins=[0, 20, 30, 60, np.inf],
                              labels=["<20","20–30","30–60",">60"],
                              include_lowest=True)

    # Fig 4a — concentration regimes
    print("\n=== Fig. 4a: Yield vs concentration class ===")
    for cond in ["Non-stress", "Stress"]:
        for cls, g in sub[sub["Condition"]==cond].groupby("conc_bin"):
            if len(g) < 3: continue
            res = dl_meta(g["lnRR"].values, g["vi"].values)
            print(f"  {cond:12s} conc={cls:9s} k={res[10]:3d} "
                  f"{res[2]:+.1f}% [{res[3]:+.1f}, {res[4]:+.1f}]")

    # Fig 4b — size regimes
    print("\n=== Fig. 4b: Yield vs size class ===")
    for cond in ["Non-stress", "Stress"]:
        for cls, g in sub[sub["Condition"]==cond].groupby("size_bin"):
            if len(g) < 3: continue
            res = dl_meta(g["lnRR"].values, g["vi"].values)
            print(f"  {cond:12s} size={cls:8s} k={res[10]:3d} "
                  f"{res[2]:+.1f}% [{res[3]:+.1f}, {res[4]:+.1f}]")

    # Fig 4c/4d — heatmaps of mean yield effect (size x concentration)
    print("\n=== Fig. 4c/4d: size × concentration heatmaps ===")
    for cond in ["Non-stress", "Stress"]:
        d = sub[sub["Condition"]==cond]
        eff = d.groupby(["size_bin", "conc_bin"])["pct_change"].mean().unstack()
        n   = d.groupby(["size_bin", "conc_bin"])["pct_change"].count().unstack()
        print(f"\n--- {cond} mean effect (n>=3 cells only) ---")
        for sz in eff.index:
            row = []
            for cc in eff.columns:
                v = eff.loc[sz, cc]
                k = n.loc[sz, cc] if not pd.isna(n.loc[sz, cc]) else 0
                row.append(f"{v:+.0f}% (n={int(k)})" if k>=3 else "  n<3   ")
            print(f"  size {sz:8s}: " + " | ".join(row))
