"""
Script for paper：
  - Table S3 in trim-and-fill adjustmentcolumn
  - Results §"Bias-adjusted estimates" section（line 78）

Tool: Python + custom-implementedsimplified trim-and-fill (DL random-effects)

per the paper Methods §"Assessment of Publication Bias and Robustness"：
  trim-and-fill shouldusingto 8 subgroupassessed separately。

run：python TableS3_trim_and_fill.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "code", "new_figures"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]


def trim_and_fill(yi, vi, side="right"):
    """simplified trim-and-fill (R0 + DL re-estimation)."""
    yi = np.asarray(yi); vi = np.asarray(vi)
    res = dl_meta(yi, vi); theta = res[0]
    deviations = yi - theta
    ranks = np.argsort(-deviations) if side == "right" else np.argsort(deviations)
    n = len(yi); k0 = 0
    for _ in range(20):
        if k0 > 0:
            mask = np.ones(n, dtype=bool); mask[ranks[:k0]] = False
            yi_t, vi_t = yi[mask], vi[mask]
        else:
            yi_t, vi_t = yi, vi
        theta_t = dl_meta(yi_t, vi_t)[0]
        deviations_new = yi - theta_t
        if side == "right":
            n_pos = np.sum(deviations_new > 0); n_neg = np.sum(deviations_new <= 0)
            k0_new = max(0, n_pos - n_neg)
        else:
            n_pos = np.sum(deviations_new >= 0); n_neg = np.sum(deviations_new < 0)
            k0_new = max(0, n_neg - n_pos)
        if k0_new == k0: break
        k0 = k0_new
    if k0 > 0:
        mirror_y = 2 * theta_t - yi[ranks[:k0]]
        mirror_v = vi[ranks[:k0]]
        yi_filled = np.concatenate([yi, mirror_y])
        vi_filled = np.concatenate([vi, mirror_v])
    else:
        yi_filled, vi_filled = yi, vi
    res_adj = dl_meta(yi_filled, vi_filled)
    return res_adj[2], k0


if __name__ == "__main__":
    df = load_all()
    print(f"\n{'Condition':12s} {'Indicator':25s} {'k':>3s}  "
          f"{'Original':>10s}  {'Adjusted':>10s}  {'k_imputed':>9s}")
    print("-" * 80)
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5: continue
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            adj_pct, k0 = trim_and_fill(sub["lnRR"].values, sub["vi"].values)
            print(f"{cond:12s} {ind:25s} {res[10]:3d}  "
                  f"{res[2]:+9.1f}%  {adj_pct:+9.1f}%  {k0:>9d}")
