"""Step 01 — Data loading and quality control.

Loads:
  data/normal_data.csv     (non-stress observations)
  data/stress_data.csv     (abiotic-stress observations)

Reports:
  - total observations, study count
  - subgroup sample sizes (4 indicators x 2 conditions)
  - listwise complete-case sample sizes for the ML yield subsets

Expected outputs (matching the paper):
  - 1,107 total observations from 71 independent studies
  - Non-stress yield ML subset:  196 obs from 35 studies
  - Stress     yield ML subset:  277 obs from 35 studies
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "00_shared"))
import numpy as np
import pandas as pd
from _common import load_all

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]


def summary(df):
    df = df[df["vi"] > 0].dropna(subset=["lnRR", "vi"])
    print(f"  Total observations : {len(df)}")
    print(f"  Independent studies: {df['study_id'].nunique()}")
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df.Condition == cond) & (df.Performance == ind)]
            print(f"    {cond:10s} {ind:25s} k = {len(sub):3d}")


def ml_yield_subset(df, condition):
    """Complete-case subset used for machine learning (yield only)."""
    label = "Non-stress" if condition == "normal" else "Stress"
    sub = df[(df.Condition == label) & (df.Performance == "Yield")].copy()
    sub = sub.dropna(subset=["lnRR", "vi"])
    sub = sub[sub.vi > 0]
    features = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop"]
    if condition == "stress" and "Stress_type" in sub.columns:
        features.append("Stress_type")
    sub = sub.dropna(subset=features)
    return sub


if __name__ == "__main__":
    df = load_all()
    print("=== Full dataset ===")
    summary(df)

    print("\n=== ML yield subsets (complete-case, feature filter) ===")
    dn = ml_yield_subset(df, "normal")
    ds = ml_yield_subset(df, "stress")
    print(f"  Non-stress yield: {len(dn)} obs, {dn.study_id.nunique()} studies")
    print(f"  Stress     yield: {len(ds)} obs, {ds.study_id.nunique()} studies")

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    dn.to_csv(os.path.join(out_dir, "ml_yield_non_stress.csv"), index=False)
    ds.to_csv(os.path.join(out_dir, "ml_yield_stress.csv"), index=False)
    print(f"\nML subsets saved to {out_dir}/")
