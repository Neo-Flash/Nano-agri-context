"""
Step 1: Data preparation
- load normal/stress data
- filter Yield observation
- constructfeaturematrix + binaryclassificationtarget (lnRR > 0)
- savehandleafterdata
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(BASE, "data")
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)

def load_and_prepare(csv_name, condition_label):
    df = pd.read_csv(os.path.join(DATA, csv_name), encoding="latin-1")
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()
    if "Crops" in df.columns and "Crop" not in df.columns:
        df = df.rename(columns={"Crops": "Crop"})

    # filter Yield
    sub = df[df["Performance"] == "Yield"].copy()
    sub = sub.dropna(subset=["lnRR", "vi"])
    sub = sub[sub["vi"] > 0]

    # binaryclassificationtarget
    sub["y_binary"] = (sub["lnRR"] > 0).astype(int)

    # multi-class target: strong-positive(>median), weak-positive(0~median), negative(<=0)
    median_pos = sub.loc[sub["lnRR"] > 0, "lnRR"].median()
    sub["y_multi"] = 0  # negative
    sub.loc[(sub["lnRR"] > 0) & (sub["lnRR"] <= median_pos), "y_multi"] = 1  # weak positive
    sub.loc[sub["lnRR"] > median_pos, "y_multi"] = 2  # strong positive

    # feature
    feature_cols = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop"]
    if condition_label == "stress" and "Stress_type" in sub.columns:
        feature_cols.append("Stress_type")

    sub = sub.dropna(subset=feature_cols)
    sub["condition"] = condition_label
    sub["study_id"] = sub["number"].astype(int)

    print(f"\n{'='*50}")
    print(f"Condition: {condition_label}")
    print(f"  Total Yield obs: {len(sub)}")
    print(f"  Positive (lnRR>0): {(sub['y_binary']==1).sum()} ({(sub['y_binary']==1).mean()*100:.1f}%)")
    print(f"  Negative (lnRR<=0): {(sub['y_binary']==0).sum()} ({(sub['y_binary']==0).mean()*100:.1f}%)")
    print(f"  Unique studies: {sub['study_id'].nunique()}")
    print(f"  Features: {feature_cols}")
    print(f"  Multi-class: neg={sum(sub['y_multi']==0)}, weak+={sum(sub['y_multi']==1)}, strong+={sum(sub['y_multi']==2)}")

    return sub, feature_cols

if __name__ == "__main__":
    dn, feat_n = load_and_prepare("normal_data.csv", "normal")
    ds, feat_s = load_and_prepare("stress_data.csv", "stress")

    dn.to_csv(os.path.join(RES, "data_normal_yield.csv"), index=False)
    ds.to_csv(os.path.join(RES, "data_stress_yield.csv"), index=False)

    print(f"\nSaved to {RES}")
    print("Done.")
