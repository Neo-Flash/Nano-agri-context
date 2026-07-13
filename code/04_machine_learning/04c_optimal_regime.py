"""
Step 3: optimalregimerecommend
- foreach crop × condition，find highest-effect  NP type / size / concentration / method combination
- generaterecommendation heatmap
"""
import pandas as pd
import numpy as np
import os, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
FIG = os.path.join(BASE, "figures")

def load_all_yield():
    """load all Yield data (not restricted to complete-case ML features)"""
    DATA = os.path.join(BASE, "data")
    frames = []
    for csv, cond in [("normal_data.csv", "Non-stress"), ("stress_data.csv", "Stress")]:
        df = pd.read_csv(os.path.join(DATA, csv), encoding="latin-1")
        df.columns = df.columns.str.strip()
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].str.strip()
        if "Crops" in df.columns and "Crop" not in df.columns:
            df = df.rename(columns={"Crops": "Crop"})
        sub = df[df["Performance"] == "Yield"].copy()
        sub = sub.dropna(subset=["lnRR", "vi"])
        sub = sub[sub["vi"] > 0]
        sub["Condition"] = cond
        sub["pct_change"] = (np.exp(sub["lnRR"]) - 1) * 100
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

def regime_analysis(df):
    """analysiseachfactor level meaneffect"""
    print("\n" + "="*60)
    print("Optimal Regime Analysis")
    print("="*60)

    factors = ["NPs_type", "Concentration", "Method", "Crop"]
    all_rows = []

    for cond in ["Non-stress", "Stress"]:
        sub = df[df["Condition"] == cond]
        for factor in factors:
            if factor not in sub.columns:
                continue
            grp = sub.groupby(factor).agg(
                mean_pct=("pct_change", "mean"),
                median_pct=("pct_change", "median"),
                n=("pct_change", "count"),
                pos_rate=("lnRR", lambda x: (x > 0).mean() * 100),
                mean_lnRR=("lnRR", "mean"),
                sd_lnRR=("lnRR", "std")
            ).reset_index()
            grp = grp[grp["n"] >= 3]  # ≥ 3 observations
            grp["Condition"] = cond
            grp["Factor"] = factor
            grp = grp.rename(columns={factor: "Level"})
            all_rows.append(grp)

    df_regime = pd.concat(all_rows, ignore_index=True)
    df_regime = df_regime.sort_values(["Condition", "Factor", "mean_pct"], ascending=[True, True, False])
    df_regime.to_csv(os.path.join(RES, "optimal_regime.csv"), index=False)

    # print top combinations
    for cond in ["Non-stress", "Stress"]:
        print(f"\n  --- {cond} ---")
        sub = df_regime[df_regime["Condition"] == cond]
        for factor in factors:
            fsub = sub[sub["Factor"] == factor].head(3)
            if len(fsub) == 0:
                continue
            print(f"    {factor} (top 3 by mean effect):")
            for _, row in fsub.iterrows():
                print(f"      {str(row['Level']):20s}: mean={row['mean_pct']:+.1f}%, "
                      f"pos_rate={row['pos_rate']:.0f}%, n={row['n']:.0f}")

    return df_regime

def plot_regime_heatmap(df, df_regime):
    """generaterecommendation heatmap"""
    print("\n--- Generating regime heatmaps ---")

    # NPs_type × Crop heatmap (by stress/non-stress)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, cond in zip(axes, ["Non-stress", "Stress"]):
        sub = df[df["Condition"] == cond]
        pivot = sub.groupby(["NPs_type", "Crop"])["pct_change"].mean().unstack(fill_value=np.nan)
        # keep only cells with enough data 
        count = sub.groupby(["NPs_type", "Crop"])["pct_change"].count().unstack(fill_value=0)
        pivot[count < 3] = np.nan
        if pivot.empty:
            continue
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Mean effect (%)"})
        ax.set_title(f"{cond}: NPs Type × Crop\n(mean % change, n≥3)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Crop"); ax.set_ylabel("NPs Type")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "regime_heatmap_type_crop.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "regime_heatmap_type_crop.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print("  -> regime_heatmap_type_crop")

    # Method × Crop heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, cond in zip(axes, ["Non-stress", "Stress"]):
        sub = df[df["Condition"] == cond]
        pivot = sub.groupby(["Method", "Crop"])["pct_change"].mean().unstack(fill_value=np.nan)
        count = sub.groupby(["Method", "Crop"])["pct_change"].count().unstack(fill_value=0)
        pivot[count < 3] = np.nan
        if pivot.empty:
            continue
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn", center=0,
                    linewidths=0.5, ax=ax, cbar_kws={"label": "Mean effect (%)"})
        ax.set_title(f"{cond}: Method × Crop\n(mean % change, n≥3)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Crop"); ax.set_ylabel("Method")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "regime_heatmap_method_crop.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "regime_heatmap_method_crop.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print("  -> regime_heatmap_method_crop")

    # positivepositive-effect ratebar chart: NPs_type × Condition
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, cond in enumerate(["Non-stress", "Stress"]):
        sub = df_regime[(df_regime["Condition"] == cond) & (df_regime["Factor"] == "NPs_type")]
        sub = sub.sort_values("pos_rate", ascending=False)
        offset = -0.2 + i * 0.4
        colors = "#2E86AB" if cond == "Non-stress" else "#C73E1D"
        bars = ax.bar(np.arange(len(sub)) + offset, sub["pos_rate"], 0.35,
                      label=cond, color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
        for j, (_, row) in enumerate(sub.iterrows()):
            ax.text(j + offset, row["pos_rate"] + 1, f"n={row['n']:.0f}",
                    ha="center", fontsize=7, rotation=45)
        if i == 0:
            ax.set_xticks(np.arange(len(sub)))
            ax.set_xticklabels(sub["Level"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Positive Effect Rate (%)", fontsize=11, fontweight="bold")
    ax.set_title("Positive Effect Rate by NPs Type and Condition", fontsize=12, fontweight="bold")
    ax.axhline(50, color="grey", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "regime_pos_rate.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "regime_pos_rate.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print("  -> regime_pos_rate")


if __name__ == "__main__":
    df = load_all_yield()
    print(f"Total Yield observations: {len(df)}")
    df_regime = regime_analysis(df)
    plot_regime_heatmap(df, df_regime)
    print("\nOptimal regime analysis done.")
