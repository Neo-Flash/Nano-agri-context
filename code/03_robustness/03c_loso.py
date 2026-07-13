"""Step 03c — Leave-one-study-out (LOSO) sensitivity analysis.

Iteratively removes each study and recomputes DL pooled effect. Reports:
  - full pooled estimate
  - min, max across LOSO iterations
  - span (max - min) in percentage points
  - flagged studies whose removal shifts effect by > 5 pp

Expected: 6 of 8 subgroups have span < 7 pp; Study 77 (biomass) and
Study 63 (photosynthetic pigment) are notable influential studies.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta, loso

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]

if __name__ == "__main__":
    df = load_all()
    print(f"{'Condition':12s} {'Indicator':25s} {'k':>4s} {'full':>8s} {'span':>7s}")
    print("-" * 62)
    rows = []
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df.Condition == cond) & (df.Performance == ind)]
            sub = sub.dropna(subset=["lnRR", "vi", "study_id"])
            sub = sub[sub.vi > 0]
            if len(sub) < 5: continue
            full, arr = loso(sub.lnRR.values, sub.vi.values, sub.study_id.values)
            if len(arr) == 0: continue
            pcts = np.array([x[1] for x in arr], dtype=float)
            span = pcts.max() - pcts.min()
            print(f"{cond:12s} {ind:25s} {len(sub):4d} {full:+7.1f}% {span:6.1f}pp")
            rows.append(dict(Condition=cond, Indicator=ind, k=len(sub),
                             full_pct=round(full,1), min_pct=round(pcts.min(),1),
                             max_pct=round(pcts.max(),1), span_pp=round(span,1)))
    out = os.path.join(os.path.dirname(__file__), "..", "..", "results", "loso.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved: {out}")
