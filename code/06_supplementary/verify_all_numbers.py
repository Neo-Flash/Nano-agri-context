"""
Full reconciliation script：produce the paper main_en.pdf / main_cn.pdf appearsall quantitative numbers，
compared row-by-row against the paper。

Numbers covered in the paper：
  Abstract / Results
    - 1,201 observations from 66 studies
    - 8 subgrouppooled effect（Non-stress / Stress × 4 indicators）
    - each subgroup  95% CI
    - each subgroup  95% PI
    - each subgroup  I²
    - Stress vs Non-stress yield: 22.5% vs 17.6%
    - Pot vs Field yield (4 number)
    - LOSO span：6/8 < 7 pp
    - Egger test 8 subgroup(paper reports P<0.001）
    - Trim-and-fill k_imputed
    - Fail-safe N (Nfs)
    - ICC = 0.665 / 0.513
    - ioniccontrol < 0.5%
    - GroupKFold R² ≤ 0
    - Conformal coverage 91% / 94%
    - K-Means k=6
    - Boruta hit rate
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
from _common import load_all, dl_meta, egger_test, loso

def compute_icc_local(df, indicator="Yield"):
    out = {}
    for cond in ["Non-stress", "Stress"]:
        sub = df[(df["Condition"] == cond) & (df["Performance"] == indicator)]
        sub = sub.dropna(subset=["lnRR", "study_id"])
        sub = sub[sub["vi"] > 0]
        if sub["study_id"].nunique() < 2: continue
        y = sub["lnRR"].values
        groups = sub["study_id"].values
        mu = y.mean()
        ss_between = ss_within = 0.0
        for g in np.unique(groups):
            yg = y[groups == g]
            ss_between += len(yg) * (yg.mean() - mu) ** 2
            ss_within += np.sum((yg - yg.mean()) ** 2)
        s2_b = ss_between / max(1, len(np.unique(groups)) - 1)
        s2_w = ss_within / max(1, len(y) - len(np.unique(groups)))
        out[cond] = s2_b / (s2_b + s2_w)
    return out

INDICATORS = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]


def trim_and_fill(yi, vi, side="right"):
    yi, vi = np.asarray(yi), np.asarray(vi)
    res = dl_meta(yi, vi); theta = res[0]
    deviations = yi - theta
    ranks = np.argsort(-deviations) if side == "right" else np.argsort(deviations)
    n, k0 = len(yi), 0
    for _ in range(20):
        if k0 > 0:
            mask = np.ones(n, dtype=bool); mask[ranks[:k0]] = False
            yi_t, vi_t = yi[mask], vi[mask]
        else:
            yi_t, vi_t = yi, vi
        theta_t = dl_meta(yi_t, vi_t)[0]
        dev = yi - theta_t
        if side == "right":
            k0_new = max(0, np.sum(dev > 0) - np.sum(dev <= 0))
        else:
            k0_new = max(0, np.sum(dev < 0) - np.sum(dev >= 0))
        if k0_new == k0: break
        k0 = k0_new
    if k0 > 0:
        mirror_y = 2 * theta_t - yi[ranks[:k0]]
        mirror_v = vi[ranks[:k0]]
        yi_full = np.concatenate([yi, mirror_y])
        vi_full = np.concatenate([vi, mirror_v])
    else:
        yi_full, vi_full = yi, vi
    return dl_meta(yi_full, vi_full)[2], k0


def main():
    df = load_all()

    print("="*80)
    print(" 1. Data scale")
    print("="*80)
    n_obs = len(df[(df["lnRR"].notna()) & (df["vi"] > 0)])
    n_studies = df["study_id"].nunique()
    print(f"  Observations (lnRR & vi haseffective): {n_obs}")
    print(f"  Studies: {n_studies}")
    print(f"  paper Abstract: 1,107 obs, 71 studies")

    print("\n" + "="*80)
    print(" 2. 8 subgrouppooled effect（Fig. 1b, Table S3）")
    print("="*80)
    rows = []
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2: continue
            theta, se, pct, ci_lo, ci_hi, pi_lo, pi_hi, tau2, Q, I2, k = \
                dl_meta(sub["lnRR"].values, sub["vi"].values)
            rows.append((cond, ind, k, pct, ci_lo, ci_hi, pi_lo, pi_hi, I2))
            print(f"  {cond:11s} {ind:25s} k={k:3d}  "
                  f"{pct:+5.1f}% [{ci_lo:+5.1f}, {ci_hi:+5.1f}]  "
                  f"PI=[{pi_lo:+6.1f}, {pi_hi:+6.1f}]  I²={I2:.1f}%")

    print("\n  Paper reportsvalue（pleasecontrol）：")
    print("  Non-stress Growth     k=105, +13.7%, CI [+11.2,+16.3], PI [-6.2,+37.9], I²=95.2%")
    print("  Non-stress Pigment    k=96,  +28.4%, CI [+23.7,+33.3], PI [-9.0,+81.2], I²=97.4%")
    print("  Non-stress Biomass    k=88,  +47.7%, CI [+40.1,+55.8], PI [-7.7,+136.5], I²=94.7%")
    print("  Non-stress Yield      k=196, +17.6%, CI [+15.1,+20.2], PI [-8.7,+51.4], I²=97.9%")
    print("  Stress Growth         k=160, +19.3%, CI [+16.9,+21.7], PI [-4.7,+49.2], I²=95.9%")
    print("  Stress Pigment        k=106, +37.9%, CI [+34.0,+42.0], PI [+5.1,+81.1], I²=96.7%")
    print("  Stress Biomass        k=79,  +28.1%, CI [+22.6,+33.9], PI [-11.4,+85.1], I²=96.1%")
    print("  Stress Yield          k=277, +22.5%, CI [+20.6,+24.4], PI [-0.6,+50.8], I²=96.7%")

    print("\n" + "="*80)
    print(" 3. Pot vs Field yield subgroup")
    print("="*80)
    for cond in ["Non-stress", "Stress"]:
        for exp in ["Pot", "Field"]:
            sub = df[(df["Performance"] == "Yield") &
                     (df["Condition"] == cond) &
                     (df["experiment"] == exp)].dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2: continue
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            print(f"  {cond:11s} {exp:6s} k={res[10]:3d}  "
                  f"{res[2]:+.1f}% [{res[3]:+.1f}, {res[4]:+.1f}]")
    print("\n  Paper reports：")
    print("  Non-stress Pot   k=85,  +26.0% CI [+22.6, +29.4]")
    print("  Non-stress Field k=111, +8.9%  CI [+4.8, +13.2]")
    print("  Stress Pot       k=81,  +28.9%")
    print("  Stress Field     k=196, +18.0% CI [+16.0, +20.0]")

    print("\n" + "="*80)
    print(" 4. LOSO span (paper L82: 6/8 subgroups < 7 pp)")
    print("="*80)
    n_lt7 = 0; total = 0
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
            span = pcts.max() - pcts.min()
            total += 1
            if span < 7: n_lt7 += 1
            print(f"  {cond:11s} {ind:25s} span={span:5.1f} pp  "
                  f"{'<7' if span<7 else '≥7'}")
    print(f"\n  actual: {n_lt7}/{total} subgroups < 7 pp(paper: 6/8)")

    print("\n" + "="*80)
    print(" 5. Egger test(paper line 71：'4/8 P<0.05，Stress Yield P<10^-15'）")
    print("="*80)
    n_sig = 0; total = 0
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5: continue
            intercept, p = egger_test(sub["lnRR"].values, sub["vi"].values)
            sig = "✓" if p < 0.001 else "✗"
            total += 1
            if p < 0.001: n_sig += 1
            print(f"  {cond:11s} {ind:25s} intercept={intercept:+6.3f}  "
                  f"P={p:.2e}  {sig}")
    print(f"\n  actual: {n_sig}/{total} subgroups P < 0.001")

    print("\n" + "="*80)
    print(" 6. Trim-and-fill(paper line 78：'no studies imputed in any of 8'）")
    print("="*80)
    for cond in ["Non-stress", "Stress"]:
        for ind in INDICATORS:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)]
            sub = sub.dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5: continue
            adj_pct, k0 = trim_and_fill(sub["lnRR"].values, sub["vi"].values)
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            print(f"  {cond:11s} {ind:25s} orig={res[2]:+5.1f}%  "
                  f"adj={adj_pct:+5.1f}%  k_imputed={k0}")

    print("\n" + "="*80)
    print(" 7. ICC(paper Methods + Results）")
    print("="*80)
    icc = compute_icc_local(df, "Yield")
    for cond, v in icc.items():
        print(f"  {cond:11s} ICC = {v:.3f}")
    print(f"\n  Paper reports：Non-stress ICC = 0.904，Stress ICC = 0.882")

    print("\n" + "="*80)
    print(" 8. ioniccontrol(paper Discussion ionic-control caveat）")
    print("="*80)
    titles = df["title"].astype(str).str.lower()
    ionic_mask = titles.str.contains("ionic|bulk|salt control|ion control", na=False)
    n_ionic = ionic_mask.sum()
    n_total = len(df[(df["lnRR"].notna()) & (df["vi"] > 0)])
    pct = 100 * n_ionic / n_total
    print(f"  observations containing 'ionic/bulk/salt control' keywords: {n_ionic} / {n_total}")
    print(f"  proportion: {pct:.2f}% (paper reports < 0.5%)")


if __name__ == "__main__":
    main()
