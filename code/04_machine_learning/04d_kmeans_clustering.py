"""
Step 4: nosupervisedclustering
- K-Means clusteringsearching for M-A-E feature natural grouping
- descriptioneachcluster featureandeffect distribution
- visualization: PCA projection + cluster-feature radar plot
"""
import pandas as pd
import numpy as np
import os, warnings, gc
warnings.filterwarnings("ignore")

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
FIG = os.path.join(BASE, "figures")

np.random.seed(42)

def run_clustering(df, feature_cols, condition):
    print(f"\n{'='*60}")
    print(f"Clustering — {condition}")
    print(f"{'='*60}")

    # Encode
    X_raw = df[feature_cols].copy()
    encoders = {}
    for c in X_raw.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_raw[c] = le.fit_transform(X_raw[c].astype(str))
        encoders[c] = le

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw.values)

    # findoptimal k (silhouette)
    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        sil_scores[k] = sil
        print(f"  k={k}: silhouette={sil:.3f}")

    best_k = max(sil_scores, key=sil_scores.get)
    print(f"  Best k={best_k} (silhouette={sil_scores[best_k]:.3f})")

    # Final clustering
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # ── PCA scatter plot ──
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))
    for c in range(best_k):
        mask = df["cluster"] == c
        mean_effect = df.loc[mask, "lnRR"].mean()
        pct_effect = (np.exp(mean_effect) - 1) * 100
        n = mask.sum()
        pos_rate = (df.loc[mask, "lnRR"] > 0).mean() * 100
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[colors[c]], s=40, alpha=0.6,
                   edgecolor="black", linewidth=0.3,
                   label=f"Cluster {c}: n={n}, effect={pct_effect:+.1f}%, pos={pos_rate:.0f}%")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=11)
    ax.set_title(f"{condition.capitalize()} — K-Means Clustering (k={best_k})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cluster_pca_{condition}.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, f"cluster_pca_{condition}.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print(f"  -> cluster_pca_{condition}")

    # ── Cluster profile ──
    profile_rows = []
    for c in range(best_k):
        mask = df["cluster"] == c
        sub = df[mask]
        row = {"Cluster": c, "Condition": condition, "n": len(sub),
               "mean_lnRR": round(sub["lnRR"].mean(), 4),
               "mean_pct": round((np.exp(sub["lnRR"].mean()) - 1) * 100, 1),
               "pos_rate": round((sub["lnRR"] > 0).mean() * 100, 1)}
        # mode per feature
        for feat in feature_cols:
            mode_val = sub[feat].mode().iloc[0] if len(sub[feat].mode()) > 0 else "N/A"
            row[f"mode_{feat}"] = mode_val
        profile_rows.append(row)
        print(f"\n  Cluster {c} (n={len(sub)}):")
        print(f"    Mean effect: {row['mean_pct']:+.1f}%, Pos rate: {row['pos_rate']:.1f}%")
        for feat in feature_cols:
            vc = sub[feat].value_counts().head(3)
            top3 = ", ".join([f"{v}({c})" for v, c in zip(vc.index, vc.values)])
            print(f"    {feat}: {top3}")

    df_profile = pd.DataFrame(profile_rows)
    df_profile.to_csv(os.path.join(RES, f"cluster_profile_{condition}.csv"), index=False)

    # ── Effect distribution by cluster ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in range(best_k):
        sub = df[df["cluster"] == c]
        pct_vals = (np.exp(sub["lnRR"]) - 1) * 100
        ax.hist(pct_vals, bins=20, alpha=0.5, color=colors[c],
                label=f"Cluster {c} (n={len(sub)}, mean={pct_vals.mean():+.1f}%)",
                edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Effect Size (% change)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"{condition.capitalize()} — Effect Distribution by Cluster",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"cluster_effect_dist_{condition}.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, f"cluster_effect_dist_{condition}.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print(f"  -> cluster_effect_dist_{condition}")

    gc.collect()
    return df, best_k


if __name__ == "__main__":
    dn = pd.read_csv(os.path.join(RES, "data_normal_yield.csv"))
    ds = pd.read_csv(os.path.join(RES, "data_stress_yield.csv"))

    feat_n = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop"]
    feat_s = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop", "Stress_type"]

    dn, k_n = run_clustering(dn, feat_n, "normal")
    ds, k_s = run_clustering(ds, feat_s, "stress")

    print("\nClustering analysis done.")
