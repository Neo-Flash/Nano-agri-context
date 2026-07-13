"""
Script for paper：
  - Fig. 1a  data structure（observation counts per subgroup）
  - Fig. 1b  four indicators pooled mean effect（DL，8 subgroup）
  - Fig. 1c  Yield  study-levelforest plot + pooled estimate
  - Fig. 1d  95% CI and 95% PI control
  - Table S3 complete 8-row pooled-effects table

Tool: Python + custom-implemented DerSimonian-Laird random-effects pooling
（per the revised paper Methods §"Traditional Meta-Analysis" description）

run：python Fig1_pooled_effects_and_PI.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]

if __name__ == "__main__":
    df = load_all()
    rows = []
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2:
                continue
            theta, se, pct, ci_lo, ci_hi, pi_lo, pi_hi, tau2, Q, I2, k = \
                dl_meta(sub["lnRR"].values, sub["vi"].values)
            rows.append(dict(
                Condition=cond, Indicator=ind, k=k,
                pct=round(pct,1), CI_lo=round(ci_lo,1), CI_hi=round(ci_hi,1),
                PI_lo=round(pi_lo,1), PI_hi=round(pi_hi,1),
                I2=round(I2,1), tau2=round(tau2,4),
            ))
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    # Reproduces all numbers in Fig. 1 and Table S3
