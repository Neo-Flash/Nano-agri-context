"""
Script for paper：
  - Fig. 2a  Pot vs Field yield translation gap
  - Fig. 2b  publication bias（8 subgroup's Egger regression −log10 P）
  - Fig. 2c  LOSO robustness（eachsubgroupleave-one-out after leaving out a study poolrange）

Tool: Python + DL random-effects + custom-implemented Egger weighted regression
（per the revised paper Methods §"Assessment of Publication Bias and Robustness"）

run：python Fig2_pot_vs_field_egger_loso.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "code", "new_figures"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta, egger_test, loso

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]

if __name__ == "__main__":
    df = load_all()

    # ── 2a Pot vs Field for yield ──────────────────────────────
    print("\n=== Fig. 2a: Pot vs Field yield ===")
    for cond in ["Non-stress", "Stress"]:
        for exp in ["Pot", "Field"]:
            sub = df[(df["Performance"] == "Yield") &
                     (df["Condition"] == cond) &
                     (df["experiment"] == exp)].dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2: continue
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            print(f"  {cond:12s} {exp:6s} k={res[10]:3d}  "
                  f"{res[2]:+.1f}%  [{res[3]:+.1f}, {res[4]:+.1f}]")

    # ── 2b Egger across 8 subgroups ───────────────────────────
    print("\n=== Fig. 2b: Subgroup Egger ===")
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5: continue
            intercept, p = egger_test(sub["lnRR"].values, sub["vi"].values)
            print(f"  {cond:12s} {ind:25s} k={len(sub):3d}  "
                  f"intercept={intercept:+.3f}  P={p:.2e}")

    # ── 2c LOSO span for each subgroup ────────────────────────
    print("\n=== Fig. 2c: LOSO span ===")
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi", "study_id"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5: continue
            full, arr = loso(sub["lnRR"].values, sub["vi"].values,
                              sub["study_id"].values)
            if len(arr) == 0: continue
            pcts = np.array([x[1] for x in arr], dtype=float)
            print(f"  {cond:12s} {ind:25s}  full={full:+.1f}%  "
                  f"span={pcts.max()-pcts.min():.1f} pp  "
                  f"range=[{pcts.min():+.1f}, {pcts.max():+.1f}]")
