"""
Step 2: classificationexperiment
- binaryclassification: lnRR > 0 (haseffective) vs lnRR <= 0 (noeffective/hasharm)
- multi-class: strong-positive / weak-positive / negative effect
- GroupKFold (study-level)
- multiple modelscompare + SHAP
- memory-friendly: n_jobs=1
"""
import pandas as pd
import numpy as np
import os, warnings, gc, json
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, cross_val_predict, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, classification_report,
                             confusion_matrix, balanced_accuracy_score)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
FIG = os.path.join(BASE, "figures")

plt.rcParams.update({"font.family": "Arial", "font.size": 10, "axes.linewidth": 1.2,
                      "figure.facecolor": "white"})

np.random.seed(42)

def encode_features(df, feature_cols):
    """LabelEncode categorical features，return X, encoders"""
    X = df[feature_cols].copy()
    encoders = {}
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    num_cols = [c for c in feature_cols if c not in cat_cols]
    return X.values, X.columns.tolist(), encoders, cat_cols, num_cols


def run_binary_classification(df, feature_cols, condition):
    """binaryclassificationexperiment"""
    print(f"\n{'='*60}")
    print(f"Binary Classification — {condition}")
    print(f"{'='*60}")

    X, col_names, encoders, cat_cols, num_cols = encode_features(df, feature_cols)
    y = df["y_binary"].values
    groups = df["study_id"].values

    n_pos = y.sum()
    n_neg = len(y) - n_pos
    print(f"  Samples: {len(y)} (pos={n_pos}, neg={n_neg}, ratio={n_pos/len(y)*100:.1f}%)")

    # classnotbalancecheck
    if n_neg < 10:
        print(f"  WARNING: Only {n_neg} negative samples — binary classification unreliable for {condition}")
        print(f"  Skipping binary classification, will use multi-class instead.")
        return None, None, None, col_names

    models = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                     random_state=42, n_jobs=1),
        "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                          random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                           random_state=42, n_jobs=1),
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        # cross_val_predict for predictions
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf, n_jobs=1)

        # For AUC, need probabilities
        try:
            y_proba = cross_val_predict(model, X, y, groups=groups, cv=gkf,
                                        method="predict_proba", n_jobs=1)[:, 1]
            auc = roc_auc_score(y, y_proba)
        except:
            auc = np.nan
            y_proba = None

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        f1_w = f1_score(y, y_pred, average="weighted")

        results.append({
            "Model": name, "Condition": condition,
            "Accuracy": round(acc, 3), "Balanced_Acc": round(bal_acc, 3),
            "F1_macro": round(f1, 3), "F1_weighted": round(f1_w, 3),
            "AUC": round(auc, 3) if not np.isnan(auc) else "N/A",
            "Baseline_acc": round(max(n_pos, n_neg) / len(y), 3)
        })
        print(f"    Accuracy={acc:.3f}  Balanced_Acc={bal_acc:.3f}  F1_macro={f1:.3f}  AUC={auc:.3f}")
        print(f"    Baseline (majority): {max(n_pos, n_neg)/len(y):.3f}")
        gc.collect()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RES, f"binary_clf_{condition}.csv"), index=False)

    # usingoptimalmodelperform SHAP
    best_name = df_results.loc[df_results["F1_macro"].idxmax(), "Model"]
    best_model = models[best_name]
    best_model.fit(X, y)
    print(f"\n  Best model for SHAP: {best_name}")

    return best_model, X, y, col_names


def run_multiclass_classification(df, feature_cols, condition):
    """multi-class task: strong-positive / weak-positive / negative"""
    print(f"\n{'='*60}")
    print(f"Multi-class Classification — {condition}")
    print(f"{'='*60}")

    X, col_names, encoders, cat_cols, num_cols = encode_features(df, feature_cols)
    y = df["y_multi"].values
    groups = df["study_id"].values

    class_counts = {i: (y == i).sum() for i in range(3)}
    print(f"  Classes: neg={class_counts[0]}, weak+={class_counts[1]}, strong+={class_counts[2]}")

    models = {
        "RF": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                     random_state=42, n_jobs=1),
        "GBM": GradientBoostingClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                          random_state=42),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=200, max_depth=6, class_weight="balanced",
                                           random_state=42, n_jobs=1),
    }

    gkf = GroupKFold(n_splits=5)
    results = []

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        y_pred = cross_val_predict(model, X, y, groups=groups, cv=gkf, n_jobs=1)

        acc = accuracy_score(y, y_pred)
        bal_acc = balanced_accuracy_score(y, y_pred)
        f1 = f1_score(y, y_pred, average="macro")
        f1_w = f1_score(y, y_pred, average="weighted")

        results.append({
            "Model": name, "Condition": condition,
            "Accuracy": round(acc, 3), "Balanced_Acc": round(bal_acc, 3),
            "F1_macro": round(f1, 3), "F1_weighted": round(f1_w, 3),
            "Baseline_acc": round(max(class_counts.values()) / len(y), 3)
        })
        print(f"    Accuracy={acc:.3f}  Balanced_Acc={bal_acc:.3f}  F1_macro={f1:.3f}")
        print(f"    Baseline: {max(class_counts.values())/len(y):.3f}")

        cm = confusion_matrix(y, y_pred)
        print(f"    Confusion matrix:\n{cm}")
        gc.collect()

    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(RES, f"multi_clf_{condition}.csv"), index=False)

    # SHAP on best
    best_name = df_results.loc[df_results["F1_macro"].idxmax(), "Model"]
    best_model = models[best_name]
    best_model.fit(X, y)
    print(f"\n  Best model for SHAP: {best_name}")

    return best_model, X, y, col_names


def do_shap(model, X, col_names, condition, task_label):
    """SHAP analysis and save figure"""
    print(f"\n  Computing SHAP for {condition} ({task_label})...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # handlemulti-class SHAP (3D array) orbinaryclassification (list of 2)
    if isinstance(shap_values, list):
        sv = shap_values[1] if len(shap_values) == 2 else shap_values[0]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        # multi-class: shape = (n_samples, n_features, n_classes); take mean |SHAP| across classes
        sv = np.mean(np.abs(shap_values), axis=2)  # for bar plot
        sv_full = shap_values  # keep full for summary
    else:
        sv = shap_values

    # save mean |SHAP| values
    sv_2d = sv if sv.ndim == 2 else sv
    df_shap = pd.DataFrame(sv_2d, columns=col_names)
    df_shap.to_csv(os.path.join(RES, f"shap_{task_label}_{condition}.csv"), index=False)

    # SHAP bar plot (mean |SHAP|)
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_shap = np.mean(np.abs(sv_2d), axis=0)
    idx = np.argsort(mean_shap)
    ax.barh(range(len(col_names)), mean_shap[idx], color="#2E86AB", edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(col_names)))
    ax.set_yticklabels([col_names[i] for i in idx])
    ax.set_xlabel("mean |SHAP-value|", fontsize=11)
    ax.set_title(f"Feature Importance (mean |SHAP|) — {condition} ({task_label})",
              fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, f"shap_bar_{task_label}_{condition}.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, f"shap_bar_{task_label}_{condition}.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print(f"    -> shap_bar_{task_label}_{condition}")

    gc.collect()
    return sv


if __name__ == "__main__":
    # Load data
    dn = pd.read_csv(os.path.join(RES, "data_normal_yield.csv"))
    ds = pd.read_csv(os.path.join(RES, "data_stress_yield.csv"))

    feat_n = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop"]
    feat_s = ["NPs_type", "NPs_size", "Concentration", "Method", "Crop", "Stress_type"]

    # ── Binary classification ──
    model_n_bin, X_n, y_n_bin, cols_n = run_binary_classification(dn, feat_n, "normal")
    model_s_bin, X_s, y_s_bin, cols_s = run_binary_classification(ds, feat_s, "stress")

    # SHAP for binary (normal only, stress skipped if too imbalanced)
    if model_n_bin is not None:
        do_shap(model_n_bin, X_n, cols_n, "normal", "binary")

    # ── Multi-class classification ──
    model_n_multi, X_n_m, y_n_m, cols_n_m = run_multiclass_classification(dn, feat_n, "normal")
    model_s_multi, X_s_m, y_s_m, cols_s_m = run_multiclass_classification(ds, feat_s, "stress")

    # SHAP for multi-class
    do_shap(model_n_multi, X_n_m, cols_n_m, "normal", "multi")
    do_shap(model_s_multi, X_s_m, cols_s_m, "stress", "multi")

    # ── comparison figure: model performance ──
    print("\n--- Generating comparison plot ---")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, cond in zip(axes, ["normal", "stress"]):
        # readallResults
        dfs = []
        for task in ["binary", "multi"]:
            fp = os.path.join(RES, f"{task}_clf_{cond}.csv")
            if os.path.exists(fp):
                d = pd.read_csv(fp)
                d["Task"] = task
                dfs.append(d)
        if not dfs:
            continue
        df_all = pd.concat(dfs)
        x = np.arange(len(df_all))
        colors = ["#2E86AB" if t == "binary" else "#C73E1D" for t in df_all["Task"]]
        bars = ax.bar(x, df_all["F1_macro"], color=colors, edgecolor="black", linewidth=0.5, alpha=0.8)
        for i, (_, row) in enumerate(df_all.iterrows()):
            ax.text(i, row["F1_macro"] + 0.01, f"{row['F1_macro']:.3f}", ha="center", fontsize=8)
        # baseline
        baseline = df_all["Baseline_acc"].iloc[0]
        ax.axhline(baseline, color="grey", linestyle="--", linewidth=1, label=f"Majority baseline={baseline:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{row['Task']}\n{row['Model']}" for _, row in df_all.iterrows()],
                           fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("F1 (macro)", fontsize=11)
        ax.set_title(f"{cond.capitalize()} — Classification Performance", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "clf_comparison.png"), dpi=600, bbox_inches="tight")
    plt.savefig(os.path.join(FIG, "clf_comparison.pdf"), dpi=600, bbox_inches="tight")
    plt.close("all")
    print("  -> clf_comparison")

    print("\nClassification experiments done.")
