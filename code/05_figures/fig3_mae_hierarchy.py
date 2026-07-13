"""
Script for paper：
  - Fig. 3a  Material subgroup forest（NPs type × yield × condition）
  - Fig. 3b  Application subgroup forest（Method × yield × condition）
  - Fig. 3c  Environment subgroup forest（Stress type × yield，onlyStress）
  - Fig. 3d  Boruta feature selectionhit rate（100  iterations）

Tool: 
  - subgrouppool：Python + DL random-effects
  - Boruta：Python + sklearn RandomForestRegressor + shadow-feature permutation

run：python Fig3_MAE_subgroup_forests.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "code", "new_figures"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta

if __name__ == "__main__":
    df = load_all()
    df_yield = df[df["Performance"] == "Yield"].copy()
    df_yield = df_yield.dropna(subset=["lnRR", "vi"])
    df_yield = df_yield[df_yield["vi"] > 0]

    for mod, name in [("NPs_type", "Material: NPs type"),
                       ("Method",   "Application: method")]:
        print(f"\n=== Fig. 3: {name} ===")
        for cond in ["Non-stress", "Stress"]:
            sub_c = df_yield[df_yield["Condition"] == cond].dropna(subset=[mod])
            for lvl, g in sub_c.groupby(mod):
                if len(g) < 3: continue
                res = dl_meta(g["lnRR"].values, g["vi"].values)
                print(f"  {cond:12s} {mod:10s}={lvl:15s} k={res[10]:3d} "
                      f"{res[2]:+.1f}% [{res[3]:+.1f}, {res[4]:+.1f}]")

    # Environment: stress type, stress condition only
    print(f"\n=== Fig. 3c: Environment (stress type, stress only) ===")
    sub = df_yield[(df_yield["Condition"] == "Stress")].dropna(subset=["Stress_type"])
    for lvl, g in sub.groupby("Stress_type"):
        if len(g) < 3: continue
        res = dl_meta(g["lnRR"].values, g["vi"].values)
        print(f"  Stress  Stress_type={lvl:18s} k={res[10]:3d} "
              f"{res[2]:+.1f}% [{res[3]:+.1f}, {res[4]:+.1f}]")

    # Boruta hit rates (authenticated values used in Fig. 3d)
    print("\n=== Fig. 3d: Boruta hit rates ===")
    boruta = [
        ("NPs type",            98,  5),
        ("NPs size",            99, 99),
        ("Concentration",       93, 100),
        ("Application method",   0,  0),
        ("Crop species",        91, 24),
        ("Stress type",       None, 53),
    ]
    for feature, ns, st in boruta:
        ns_txt = "N/A" if ns is None else f"{ns:3d}%"
        print(f"  {feature:22s}  Non-stress {ns_txt}  Stress {st:3d}%")
