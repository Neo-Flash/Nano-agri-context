"""
Step 6: 5additionalAImethod
1. CART decision tree
2. PDP + ALE biaseddependencyfigure
3. GAM dose-response curve
4. Apriori association rules
5. Boruta feature selection

memory-friendly: n_jobs=1, gc.collect()
"""
import pandas as pd
import numpy as np
import os, warnings, gc
warnings.filterwarnings("ignore")
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
FIG = os.path.join(BASE, "figures")
DATA = os.path.join(BASE, "data")
np.random.seed(42)
plt.rcParams.update({"font.family":"Arial","font.size":10,"axes.linewidth":1.2,"figure.facecolor":"white"})

# ── Data loading ──
def load_all():
    frames = []
    for csv, cond in [("normal_data.csv","Non-stress"),("stress_data.csv","Stress")]:
        df = pd.read_csv(os.path.join(DATA, csv), encoding="latin-1")
        df.columns = df.columns.str.strip()
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].str.strip()
        if "Crops" in df.columns and "Crop" not in df.columns:
            df = df.rename(columns={"Crops":"Crop"})
        sub = df[df["Performance"]=="Yield"].copy()
        sub = sub.dropna(subset=["lnRR","vi"])
        sub = sub[sub["vi"]>0]
        sub["Condition"] = cond
        sub["pct_change"] = (np.exp(sub["lnRR"])-1)*100
        sub["study_id"] = sub["number"].astype(int)
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)

def encode(df, feat_cols):
    X = df[feat_cols].copy()
    encoders = {}
    for c in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    return X, encoders

# ══════════════════════════════════════════════════════
# 1. CART decision tree
# ══════════════════════════════════════════════════════
def run_cart(df):
    print("\n" + "="*60)
    print("  1. CART Decision Tree")
    print("="*60)

    for cond in ["Non-stress","Stress"]:
        sub = df[df["Condition"]==cond].copy()
        feat = ["NPs_type","NPs_size","Concentration","Method","Crop"]
        if cond == "Stress" and "Stress_type" in sub.columns:
            feat.append("Stress_type")
        sub = sub.dropna(subset=feat)
        X, enc = encode(sub, feat)
        y = sub["lnRR"].values

        # shallow tree (max_depth=4) for interpretability
        tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=10, random_state=42)
        tree.fit(X.values, y)

        # text rules
        rules = export_text(tree, feature_names=feat, decimals=3)
        print(f"\n  --- {cond} (n={len(sub)}) ---")
        print(rules[:1500])

        # saverules
        with open(os.path.join(RES, f"cart_rules_{cond.lower().replace('-','')}.txt"), "w") as f:
            f.write(rules)

        # tree diagram
        fig, ax = plt.subplots(figsize=(20, 10))
        plot_tree(tree, feature_names=feat, filled=True, rounded=True,
                  fontsize=8, ax=ax, impurity=False,
                  label='none', precision=3)
        ax.set_title(f"CART Decision Tree — {cond} Yield (max_depth=4, n={len(sub)})",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"cart_tree_{cond.lower().replace('-','')}.png"),
                    dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(FIG, f"cart_tree_{cond.lower().replace('-','')}.pdf"),
                    dpi=300, bbox_inches="tight")
        plt.close("all")
        print(f"  -> cart_tree_{cond.lower().replace('-','')}")

        # leaf-node statistics
        leaf_ids = tree.apply(X.values)
        unique_leaves = np.unique(leaf_ids)
        print(f"\n  Leaf node summary ({len(unique_leaves)} leaves):")
        leaf_stats = []
        for lid in unique_leaves:
            mask = leaf_ids == lid
            n = mask.sum()
            mean_pct = (np.exp(y[mask].mean())-1)*100
            pos_rate = (y[mask]>0).mean()*100
            leaf_stats.append({"leaf":lid, "n":n, "mean_pct":round(mean_pct,1), "pos_rate":round(pos_rate,1)})
            if n >= 5:
                # find the leaf-node primary feature
                dominant = {}
                for f in feat:
                    if sub[f].dtype == object:
                        mode = sub.iloc[np.where(mask)[0]][f].mode()
                        if len(mode) > 0:
                            dominant[f] = mode.iloc[0]
                print(f"    Leaf {lid}: n={n}, effect={mean_pct:+.1f}%, pos_rate={pos_rate:.0f}%, dominant={dominant}")

        pd.DataFrame(leaf_stats).to_csv(os.path.join(RES, f"cart_leaves_{cond.lower().replace('-','')}.csv"), index=False)
        gc.collect()

# ══════════════════════════════════════════════════════
# 2. PDP + ALE
# ══════════════════════════════════════════════════════
def run_pdp(df):
    print("\n" + "="*60)
    print("  2. Partial Dependence Plots")
    print("="*60)

    for cond in ["Non-stress","Stress"]:
        sub = df[df["Condition"]==cond].copy()
        feat = ["NPs_type","NPs_size","Concentration","Method","Crop"]
        if cond == "Stress" and "Stress_type" in sub.columns:
            feat.append("Stress_type")
        sub = sub.dropna(subset=feat)
        X, enc = encode(sub, feat)
        y = sub["lnRR"].values

        # fit with RF（full dataset，fordescriptive PDP）
        rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
        rf.fit(X.values, y)

        # continuous-variable PDPs: NPs_size, Concentration
        cont_features = ["NPs_size","Concentration"]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, feat_name in zip(axes, cont_features):
            feat_idx = X.columns.get_loc(feat_name)
            pdp_result = partial_dependence(rf, X.values, [feat_idx],
                                            kind="both", grid_resolution=50)
            # ICE lines (individual)
            ice_lines = pdp_result["individual"][0]
            for line in ice_lines[::5]:  # plot every 5th
                ax.plot(pdp_result["grid_values"][0], (np.exp(line)-1)*100,
                       color="lightblue", alpha=0.3, linewidth=0.5)
            # PDP (average)
            ax.plot(pdp_result["grid_values"][0], (np.exp(pdp_result["average"][0])-1)*100,
                   color="#C73E1D", linewidth=2.5, label="PDP (average)")
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
            ax.set_xlabel(feat_name, fontsize=12, fontweight="bold")
            ax.set_ylabel("Partial Dependence (% change)", fontsize=11)
            ax.set_title(f"{cond}: {feat_name}", fontsize=12, fontweight="bold")
            ax.legend(fontsize=9)

        plt.suptitle(f"Partial Dependence + ICE Plots — {cond} Yield",
                     fontsize=14, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"pdp_{cond.lower().replace('-','')}.png"),
                    dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(FIG, f"pdp_{cond.lower().replace('-','')}.pdf"),
                    dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> pdp_{cond.lower().replace('-','')}")

        # classificationvariable PDP: NPs_type, Method, Crop
        cat_features = ["NPs_type","Method","Crop"]
        if cond == "Stress":
            cat_features.append("Stress_type")

        fig, axes = plt.subplots(1, len(cat_features), figsize=(5*len(cat_features), 5))
        if len(cat_features) == 1:
            axes = [axes]
        for ax, feat_name in zip(axes, cat_features):
            feat_idx = X.columns.get_loc(feat_name)
            pdp_result = partial_dependence(rf, X.values, [feat_idx],
                                            kind="average", grid_resolution=50)
            grid = pdp_result["grid_values"][0]
            vals = (np.exp(pdp_result["average"][0])-1)*100

            # Map encoded values back to original labels
            if feat_name in enc:
                labels = enc[feat_name].inverse_transform(grid.astype(int))
            else:
                labels = grid.astype(int).astype(str)

            bars = ax.bar(range(len(grid)), vals, color="#2E86AB", edgecolor="black", linewidth=0.5)
            ax.set_xticks(range(len(grid)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
            ax.set_ylabel("PDP (% change)", fontsize=10)
            ax.set_title(f"{feat_name}", fontsize=11, fontweight="bold")
            for i, v in enumerate(vals):
                ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7)

        plt.suptitle(f"Categorical PDP — {cond} Yield", fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"pdp_cat_{cond.lower().replace('-','')}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> pdp_cat_{cond.lower().replace('-','')}")
        gc.collect()

# ══════════════════════════════════════════════════════
# 3. GAM dose-response curve
# ══════════════════════════════════════════════════════
def run_gam(df):
    print("\n" + "="*60)
    print("  3. GAM Dose-Response Curves")
    print("="*60)

    from scipy.interpolate import UnivariateSpline

    for cond in ["Non-stress","Stress"]:
        sub = df[df["Condition"]==cond].copy()
        sub = sub.dropna(subset=["Concentration","lnRR","NPs_type"])

        # stratified by NPs_type concentration-response curve
        nps_types = sub["NPs_type"].value_counts()
        nps_types = nps_types[nps_types >= 8].index.tolist()  # ≥ 8 observations

        n_types = len(nps_types)
        if n_types == 0:
            print(f"  {cond}: Not enough data per NPs type")
            continue

        cols = min(3, n_types)
        rows = (n_types + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)

        for idx, nps in enumerate(nps_types):
            ax = axes[idx // cols][idx % cols]
            d = sub[sub["NPs_type"]==nps].sort_values("Concentration")
            x = d["Concentration"].values
            y_pct = d["pct_change"].values

            ax.scatter(x, y_pct, s=20, alpha=0.6, color="#2E86AB", edgecolor="black", linewidth=0.3)

            # smoothing spline（ifenough data points）
            if len(x) >= 10:
                try:
                    # withusing LOWESS smooth
                    from statsmodels.nonparametric.smoothers_lowess import lowess
                    smoothed = lowess(y_pct, x, frac=0.5, return_sorted=True)
                    ax.plot(smoothed[:,0], smoothed[:,1], color="#C73E1D", linewidth=2, label="LOWESS")
                except:
                    pass

            ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
            ax.set_xlabel("Concentration (ppm)", fontsize=9)
            ax.set_ylabel("Effect (%)", fontsize=9)
            ax.set_title(f"{nps} (n={len(d)})", fontsize=10, fontweight="bold")
            if len(x) >= 10:
                ax.legend(fontsize=8)

        # hide empty axes
        for idx in range(n_types, rows*cols):
            axes[idx // cols][idx % cols].set_visible(False)

        plt.suptitle(f"Dose-Response Curves by NPs Type — {cond} Yield",
                     fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"gam_dose_{cond.lower().replace('-','')}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> gam_dose_{cond.lower().replace('-','')}")

        # summary: best concentration window
        print(f"\n  {cond} — Optimal concentration windows:")
        for nps in nps_types:
            d = sub[sub["NPs_type"]==nps]
            # grouped by concentration quantile
            bins = [0, 50, 100, 300, 1000, 10000]
            d_copy = d.copy()
            d_copy["conc_bin"] = pd.cut(d_copy["Concentration"], bins=bins, labels=["0-50","50-100","100-300","300-1000",">1000"])
            grp = d_copy.groupby("conc_bin").agg(
                mean_pct=("pct_change","mean"), n=("pct_change","count")
            ).dropna()
            grp = grp[grp["n"]>=2]
            if len(grp) > 0:
                best = grp["mean_pct"].idxmax()
                print(f"    {nps}: best range={best} ({grp.loc[best,'mean_pct']:+.1f}%, n={grp.loc[best,'n']})")
        gc.collect()

# ══════════════════════════════════════════════════════
# 4. Apriori association rules
# ══════════════════════════════════════════════════════
def run_apriori(df):
    print("\n" + "="*60)
    print("  4. Association Rule Mining")
    print("="*60)

    for cond in ["Non-stress","Stress"]:
        sub = df[df["Condition"]==cond].copy()
        feat = ["NPs_type","Method","Crop"]
        if cond == "Stress" and "Stress_type" in sub.columns:
            feat.append("Stress_type")
        sub = sub.dropna(subset=feat + ["lnRR"])

        # binary-valuedeffect: high (top 33%) vs low (bottom 33%)
        q33 = sub["lnRR"].quantile(0.33)
        q67 = sub["lnRR"].quantile(0.67)
        sub["effect_class"] = "medium"
        sub.loc[sub["lnRR"] <= q33, "effect_class"] = "low"
        sub.loc[sub["lnRR"] >= q67, "effect_class"] = "high"

        # concentrationgrouped
        sub["conc_group"] = pd.cut(sub["Concentration"], bins=[0,100,500,10000], labels=["low_conc","mid_conc","high_conc"])

        # particle sizegrouped
        sub["size_group"] = pd.cut(sub["NPs_size"], bins=[0,30,60,1000], labels=["small","medium_size","large"])

        # manual frequent-itemset analysis（no need for mlxtend）
        all_feat = feat + ["conc_group","size_group"]

        print(f"\n  --- {cond} (n={len(sub)}) ---")
        print(f"  Effect thresholds: low <= {(np.exp(q33)-1)*100:.1f}%, high >= {(np.exp(q67)-1)*100:.1f}%")

        # find high-effect combinations
        print(f"\n  TOP HIGH-EFFECT COMBINATIONS:")
        results = []
        for f1 in all_feat:
            for f2 in all_feat:
                if f1 >= f2: continue
                for v1 in sub[f1].dropna().unique():
                    for v2 in sub[f2].dropna().unique():
                        mask = (sub[f1]==v1) & (sub[f2]==v2)
                        n = mask.sum()
                        if n < 5: continue
                        high_rate = (sub.loc[mask,"effect_class"]=="high").mean()
                        mean_pct = sub.loc[mask,"pct_change"].mean()
                        results.append({
                            "rule": f"{f1}={v1} & {f2}={v2}",
                            "n": n, "high_rate": round(high_rate*100,1),
                            "mean_pct": round(mean_pct,1)
                        })

        df_rules = pd.DataFrame(results)
        if len(df_rules) > 0:
            df_rules = df_rules.sort_values("high_rate", ascending=False)
            df_rules.to_csv(os.path.join(RES, f"rules_{cond.lower().replace('-','')}.csv"), index=False)

            # Top 10 high-effect rules
            top = df_rules.head(15)
            for _, row in top.iterrows():
                print(f"    {row['rule']:45s} → high_rate={row['high_rate']:.0f}%, mean={row['mean_pct']:+.1f}%, n={row['n']}")

        # findlow/negativeeffect combination
        print(f"\n  TOP LOW/NEGATIVE-EFFECT COMBINATIONS:")
        if len(df_rules) > 0:
            low_rules = df_rules.sort_values("mean_pct", ascending=True).head(10)
            for _, row in low_rules.iterrows():
                print(f"    {row['rule']:45s} → high_rate={row['high_rate']:.0f}%, mean={row['mean_pct']:+.1f}%, n={row['n']}")

        gc.collect()

# ══════════════════════════════════════════════════════
# 5. Boruta feature selection
# ══════════════════════════════════════════════════════
def run_boruta(df):
    print("\n" + "="*60)
    print("  5. Boruta Feature Selection")
    print("="*60)

    for cond in ["Non-stress","Stress"]:
        sub = df[df["Condition"]==cond].copy()
        feat = ["NPs_type","NPs_size","Concentration","Method","Crop"]
        if cond == "Stress" and "Stress_type" in sub.columns:
            feat.append("Stress_type")
        sub = sub.dropna(subset=feat)
        X, enc = encode(sub, feat)
        y = sub["lnRR"].values

        print(f"\n  --- {cond} (n={len(sub)}, features={len(feat)}) ---")

        # manual Boruta: create shadow features, compare real vs shadow importance
        n_iter = 100
        feature_hits = np.zeros(len(feat))

        rf = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)

        for it in range(n_iter):
            # create shadow features (permute each column)
            X_shadow = X.values.copy()
            for j in range(X_shadow.shape[1]):
                np.random.shuffle(X_shadow[:, j])

            X_combined = np.hstack([X.values, X_shadow])
            rf_temp = RandomForestRegressor(n_estimators=100, max_depth=6,
                                            random_state=42+it, n_jobs=1)
            rf_temp.fit(X_combined, y)

            importances = rf_temp.feature_importances_
            real_imp = importances[:len(feat)]
            shadow_imp = importances[len(feat):]
            max_shadow = shadow_imp.max()

            # realfeatureisnotexceedlargest shadow
            for j in range(len(feat)):
                if real_imp[j] > max_shadow:
                    feature_hits[j] += 1

        # rule: >50% of iterations exceed shadow → confirmed; <20% → rejected; in between → tentative
        print(f"\n  Boruta Results ({n_iter} iterations):")
        boruta_results = []
        for j, f in enumerate(feat):
            hit_rate = feature_hits[j] / n_iter * 100
            if hit_rate > 50:
                status = "CONFIRMED"
            elif hit_rate < 20:
                status = "REJECTED"
            else:
                status = "TENTATIVE"
            print(f"    {f:20s}: hit_rate={hit_rate:.0f}%  → {status}")
            boruta_results.append({"feature":f, "hit_rate":round(hit_rate,1), "status":status})

        pd.DataFrame(boruta_results).to_csv(
            os.path.join(RES, f"boruta_{cond.lower().replace('-','')}.csv"), index=False)

        # visualization
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = []
        for r in boruta_results:
            if r["status"] == "CONFIRMED":
                colors.append("#2E86AB")
            elif r["status"] == "REJECTED":
                colors.append("#C73E1D")
            else:
                colors.append("#F0A500")

        bars = ax.barh(range(len(feat)), [r["hit_rate"] for r in boruta_results],
                       color=colors, edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(feat)))
        ax.set_yticklabels(feat)
        ax.axvline(50, color="green", linestyle="--", linewidth=1, label="Confirmed threshold (50%)")
        ax.axvline(20, color="red", linestyle="--", linewidth=1, label="Rejected threshold (20%)")
        ax.set_xlabel("Hit Rate (%)", fontsize=11)
        ax.set_title(f"Boruta Feature Selection — {cond} Yield", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 105)
        for i, r in enumerate(boruta_results):
            ax.text(r["hit_rate"]+1, i, f"{r['hit_rate']:.0f}% ({r['status']})",
                    va="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"boruta_{cond.lower().replace('-','')}.png"),
                    dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> boruta_{cond.lower().replace('-','')}")
        gc.collect()


# ══════════════════════════════════════════════════════
if __name__ == "__main__":
    df = load_all()
    print(f"Total Yield observations: {len(df)}")

    run_cart(df)
    run_pdp(df)
    run_gam(df)
    run_apriori(df)
    run_boruta(df)

    print("\n" + "="*60)
    print("  ALL 5 ADDITIONAL METHODS COMPLETE")
    print("="*60)
