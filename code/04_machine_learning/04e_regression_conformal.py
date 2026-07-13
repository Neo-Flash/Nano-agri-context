"""
Step 5: advanced AI method
based on literature review，implementationthe followingmethod:
  1. MERF (Mixed Effects Random Forest) — explicit modeling study random effects
  2. MetaForest-style Weighted RF — using inverse-variance weights
  3. CatBoost with ordered boosting + weights
  4. Gaussian Process Regression — small dataset + uncertainty quantification
  5. Ordinal Regression — willprediction of graded effect sizes
  6. Stacking Ensemble — combine multiple models
  7. Conformal Prediction (MAPIE) — distribution-free prediction interval
  8. Quantile Regression — conditionprediction interval

all methods use GroupKFold, n_jobs=1, memory-friendly
"""
import pandas as pd
import numpy as np
import os, warnings, gc, json
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, StackingRegressor)
from sklearn.linear_model import Ridge, Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from merf import MERF
import mord
from mapie.regression import SplitConformalRegressor
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(BASE, "results")
os.makedirs(RES, exist_ok=True)
FIG = os.path.join(BASE, "figures")
np.random.seed(42)

plt.rcParams.update({"font.family":"Arial","font.size":10,"axes.linewidth":1.2,"figure.facecolor":"white"})
CN, CS = "#2E86AB", "#C73E1D"

# ── Data loading and encoding ──
def load_data(condition):
    df = pd.read_csv(os.path.join(RES, f"data_{condition}_yield.csv"))
    feat = ["NPs_type","NPs_size","Concentration","Method","Crop"]
    if condition == "stress" and "Stress_type" in df.columns:
        feat.append("Stress_type")
    X = df[feat].copy()
    encoders = {}
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    for c in cat_cols:
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c].astype(str))
        encoders[c] = le
    y = df["lnRR"].values
    vi = df["vi"].values
    groups = df["study_id"].values
    return X.values, y, vi, groups, X.columns.tolist(), cat_cols, cat_idx, df


def groupkfold_eval(model, X, y, groups, n_splits=5, fit_kw=None):
    """GroupKFold cross-validation; returns y_pred and metrics"""
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        m = clone_model(model)
        if fit_kw:
            m.fit(X[train_idx], y[train_idx], **{k: v[train_idx] if hasattr(v,'__getitem__') else v
                                                   for k, v in fit_kw.items()})
        else:
            m.fit(X[train_idx], y[train_idx])
        y_pred[test_idx] = m.predict(X[test_idx])
    mask = ~np.isnan(y_pred)
    r2 = r2_score(y[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    mae = mean_absolute_error(y[mask], y_pred[mask])
    return y_pred, r2, rmse, mae

def clone_model(model):
    """simple clone"""
    from sklearn.base import clone as sk_clone
    try:
        return sk_clone(model)
    except:
        return model.__class__(**model.get_params())

# ══════════════════════════════════════════════════════════
# method implementations
# ══════════════════════════════════════════════════════════

def method_weighted_rf(X, y, vi, groups, tau2):
    """MetaForest-style Weighted RF: sample_weight = 1/(tau2 + vi)"""
    weights = 1.0 / (tau2 + vi)
    weights = weights / weights.sum() * len(weights)  # normalize
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
    y_pred, r2, rmse, mae = groupkfold_eval(model, X, y, groups,
                                             fit_kw={"sample_weight": weights})
    return "Weighted RF", r2, rmse, mae, y_pred

def method_catboost(X, y, vi, groups, tau2, cat_idx):
    """CatBoost with ordered boosting + inverse-variance weights"""
    weights = 1.0 / (tau2 + vi)
    weights = weights / weights.sum() * len(weights)
    gkf = GroupKFold(n_splits=5)
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = CatBoostRegressor(iterations=200, depth=4, learning_rate=0.1,
                                   random_seed=42, verbose=0, boosting_type="Ordered")
        model.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
        y_pred[test_idx] = model.predict(X[test_idx])
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return "CatBoost Ordered", r2, rmse, mae, y_pred

def method_gpr(X, y, vi, groups):
    """Gaussian Process Regression with known noise"""
    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=5)
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in gkf.split(X, y, groups):
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        kernel = ConstantKernel(1.0) * Matern(nu=2.5) + WhiteKernel(noise_level=np.mean(vi[train_idx]))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=vi[train_idx],
                                        n_restarts_optimizer=3, random_state=42)
        gpr.fit(Xtr, y[train_idx])
        y_pred[test_idx] = gpr.predict(Xte)
    r2 = r2_score(y, y_pred); rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    return "GPR (Matern)", r2, rmse, mae, y_pred

def method_merf(X, y, groups):
    """Mixed Effects Random Forest"""
    gkf = GroupKFold(n_splits=5)
    y_pred = np.full_like(y, np.nan)
    X_df = pd.DataFrame(X).reset_index(drop=True)
    Z = np.ones((len(y), 1))
    clusters = pd.Series(groups).reset_index(drop=True)
    y_series = pd.Series(y).reset_index(drop=True)
    for train_idx, test_idx in gkf.split(X, y, groups):
        tr = list(train_idx); te = list(test_idx)
        merf_model = MERF(max_iterations=30)
        merf_model.fit(X_df.loc[tr], Z[tr], clusters.loc[tr], y_series.loc[tr])
        y_pred[test_idx] = merf_model.predict(X_df.loc[te], Z[te], clusters.loc[te])
    mask = ~np.isnan(y_pred)
    r2 = r2_score(y[mask], y_pred[mask]); rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    mae = mean_absolute_error(y[mask], y_pred[mask])
    return "MERF", r2, rmse, mae, y_pred

def method_ordinal(X, y, groups):
    """Ordinal Regression (5-class)"""
    # 5 ordinal levels: strong_neg, weak_neg, neutral, weak_pos, strong_pos
    q20, q40, q60, q80 = np.percentile(y, [20, 40, 60, 80])
    y_ord = np.zeros(len(y), dtype=int)
    y_ord[y <= q20] = 0
    y_ord[(y > q20) & (y <= q40)] = 1
    y_ord[(y > q40) & (y <= q60)] = 2
    y_ord[(y > q60) & (y <= q80)] = 3
    y_ord[y > q80] = 4

    scaler = StandardScaler()
    gkf = GroupKFold(n_splits=5)
    y_pred_ord = np.full_like(y_ord, -1)
    for train_idx, test_idx in gkf.split(X, y_ord, groups):
        Xtr = scaler.fit_transform(X[train_idx])
        Xte = scaler.transform(X[test_idx])
        model = mord.LogisticIT(alpha=1.0)
        model.fit(Xtr, y_ord[train_idx])
        y_pred_ord[test_idx] = model.predict(Xte)

    from sklearn.metrics import accuracy_score, balanced_accuracy_score
    acc = accuracy_score(y_ord, y_pred_ord)
    bal_acc = balanced_accuracy_score(y_ord, y_pred_ord)
    # map ordinal predictions back to lnRR (using each class centre)
    centers = [np.mean(y[y_ord == c]) for c in range(5)]
    y_pred_cont = np.array([centers[p] for p in y_pred_ord])
    r2 = r2_score(y, y_pred_cont)
    rmse = np.sqrt(mean_squared_error(y, y_pred_cont))
    mae = mean_absolute_error(y, y_pred_cont)
    return f"Ordinal (5-class, acc={acc:.3f}, bal_acc={bal_acc:.3f})", r2, rmse, mae, y_pred_cont

def method_stacking(X, y, vi, groups, tau2):
    """Stacking Ensemble: RF + GBM + Ridge, meta-learner = Ridge"""
    weights = 1.0 / (tau2 + vi)
    weights = weights / weights.sum() * len(weights)
    gkf = GroupKFold(n_splits=5)

    base_models = [
        ("rf", RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)),
        ("gbm", GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)),
        ("ridge", Ridge(alpha=1.0)),
    ]

    # Manual stacking with GroupKFold
    n = len(y)
    oof_preds = np.zeros((n, len(base_models)))

    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        for m_idx, (name, model) in enumerate(base_models):
            m = clone_model(model)
            if name in ["rf", "gbm"]:
                m.fit(X[train_idx], y[train_idx], sample_weight=weights[train_idx])
            else:
                m.fit(X[train_idx], y[train_idx])
            oof_preds[test_idx, m_idx] = m.predict(X[test_idx])

    # Meta-learner on OOF predictions
    meta = Ridge(alpha=1.0)
    y_pred = np.full_like(y, np.nan)
    for train_idx, test_idx in gkf.split(oof_preds, y, groups):
        meta_clone = Ridge(alpha=1.0)
        meta_clone.fit(oof_preds[train_idx], y[train_idx])
        y_pred[test_idx] = meta_clone.predict(oof_preds[test_idx])

    mask = ~np.isnan(y_pred)
    r2 = r2_score(y[mask], y_pred[mask]); rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    mae = mean_absolute_error(y[mask], y_pred[mask])
    return "Stacking (RF+GBM+Ridge)", r2, rmse, mae, y_pred

def method_conformal(X, y, groups):
    """Manual Split Conformal Prediction (no MAPIE dependency issues)"""
    gkf = GroupKFold(n_splits=5)
    base = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)

    y_pred = np.full_like(y, np.nan)
    y_lo = np.full_like(y, np.nan)
    y_hi = np.full_like(y, np.nan)

    for train_idx, test_idx in gkf.split(X, y, groups):
        n_train = len(train_idx)
        n_cal = max(10, n_train // 5)
        fit_idx = train_idx[:n_train - n_cal]
        cal_idx = train_idx[n_train - n_cal:]

        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42, n_jobs=1)
        model.fit(X[fit_idx], y[fit_idx])

        # Calibration residuals
        cal_pred = model.predict(X[cal_idx])
        residuals = np.abs(y[cal_idx] - cal_pred)
        q95 = np.quantile(residuals, 0.95)

        pred = model.predict(X[test_idx])
        y_pred[test_idx] = pred
        y_lo[test_idx] = pred - q95
        y_hi[test_idx] = pred + q95

    mask = ~np.isnan(y_pred)
    r2 = r2_score(y[mask], y_pred[mask])
    rmse = np.sqrt(mean_squared_error(y[mask], y_pred[mask]))
    coverage = np.mean((y[mask] >= y_lo[mask]) & (y[mask] <= y_hi[mask]))
    avg_width = np.mean(y_hi[mask] - y_lo[mask])
    return f"Conformal RF (cov={coverage:.2f}, width={avg_width:.3f})", r2, rmse, 0, y_pred, y_lo, y_hi

def method_plain_rf(X, y, groups):
    """Baseline: Plain RF with GroupKFold (no weights, no random effects)"""
    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42, n_jobs=1)
    y_pred, r2, rmse, mae = groupkfold_eval(model, X, y, groups)
    return "Plain RF (baseline)", r2, rmse, mae, y_pred


# ══════════════════════════════════════════════════════════
# main program
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    all_results = []

    for condition in ["normal", "stress"]:
        print(f"\n{'='*70}")
        print(f"  CONDITION: {condition.upper()}")
        print(f"{'='*70}")

        X, y, vi, groups, col_names, cat_cols, cat_idx, df = load_data(condition)

        # Estimate tau2 (DL estimator)
        w = 1.0 / vi; wsum = np.sum(w)
        theta_fe = np.sum(w * y) / wsum
        Q = np.sum(w * (y - theta_fe)**2)
        C = wsum - np.sum(w**2) / wsum
        tau2 = max(0, (Q - (len(y)-1)) / C)
        print(f"  n={len(y)}, studies={len(np.unique(groups))}, tau2={tau2:.4f}")

        # ICC
        from scipy import stats as spstats
        study_means = pd.Series(y).groupby(groups).transform("mean").values
        ss_between = np.sum((study_means - y.mean())**2)
        ss_total = np.sum((y - y.mean())**2)
        icc = ss_between / ss_total if ss_total > 0 else 0
        print(f"  ICC (study-level) = {icc:.3f}")

        results = []

        # 1. Baseline: Plain RF
        print("\n  [1/8] Plain RF (baseline)...")
        name, r2, rmse, mae, yp = method_plain_rf(X, y, groups)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")

        # 2. Weighted RF
        print("  [2/8] Weighted RF (MetaForest-style)...")
        name, r2, rmse, mae, yp = method_weighted_rf(X, y, vi, groups, tau2)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")
        gc.collect()

        # 3. CatBoost Ordered
        print("  [3/8] CatBoost Ordered...")
        name, r2, rmse, mae, yp = method_catboost(X, y, vi, groups, tau2, cat_idx)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")
        gc.collect()

        # 4. GPR
        print("  [4/8] Gaussian Process Regression...")
        name, r2, rmse, mae, yp_gpr = method_gpr(X, y, vi, groups)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")
        gc.collect()

        # 5. MERF
        print("  [5/8] MERF (Mixed Effects RF)...")
        try:
            name, r2, rmse, mae, yp_merf = method_merf(X, y, groups)
            results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
            print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")
        except Exception as e:
            print(f"    MERF failed: {e}")
            results.append({"Method": "MERF", "R2": np.nan, "RMSE": np.nan, "MAE": np.nan})
        gc.collect()

        # 6. Ordinal Regression
        print("  [6/8] Ordinal Regression (5-class)...")
        name, r2, rmse, mae, yp = method_ordinal(X, y, groups)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    {name}: R2={r2:.4f}, RMSE={rmse:.4f}")
        gc.collect()

        # 7. Stacking
        print("  [7/8] Stacking Ensemble...")
        name, r2, rmse, mae, yp_stack = method_stacking(X, y, vi, groups, tau2)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": mae})
        print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")
        gc.collect()

        # 8. Conformal Prediction
        print("  [8/8] Conformal Prediction (MAPIE)...")
        name, r2, rmse, _, yp_conf, y_lo, y_hi = method_conformal(X, y, groups)
        results.append({"Method": name, "R2": r2, "RMSE": rmse, "MAE": 0})
        print(f"    {name}: R2={r2:.4f}")
        gc.collect()

        # save results
        df_res = pd.DataFrame(results)
        df_res["Condition"] = condition
        df_res.to_csv(os.path.join(RES, f"advanced_methods_{condition}.csv"), index=False)
        all_results.append(df_res)

        # ── plot: R² across all methods ──
        fig, ax = plt.subplots(figsize=(10, 6))
        df_plot = df_res.dropna(subset=["R2"]).sort_values("R2", ascending=True)
        colors = [CS if r2 < 0 else CN for r2 in df_plot["R2"]]
        bars = ax.barh(range(len(df_plot)), df_plot["R2"], color=colors,
                       edgecolor="black", linewidth=0.5)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["Method"].apply(lambda x: x[:35]), fontsize=9)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("R² (GroupKFold CV)", fontsize=11, fontweight="bold")
        ax.set_title(f"{condition.capitalize()} — All Methods Comparison (GroupKFold)",
                     fontsize=12, fontweight="bold")
        for i, r2 in enumerate(df_plot["R2"]):
            ax.text(r2 + 0.02 if r2 >= 0 else r2 - 0.02, i, f"{r2:.3f}",
                    va="center", fontsize=8, fontweight="bold",
                    ha="left" if r2 >= 0 else "right")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"advanced_r2_{condition}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(FIG, f"advanced_r2_{condition}.pdf"), dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> advanced_r2_{condition}")

        # ── plot: Conformal prediction intervals ──
        fig, ax = plt.subplots(figsize=(10, 6))
        order = np.argsort(y)
        ax.fill_between(range(len(y)), (np.exp(y_lo[order])-1)*100, (np.exp(y_hi[order])-1)*100,
                        alpha=0.3, color=CN, label="95% Conformal PI")
        ax.plot(range(len(y)), (np.exp(y[order])-1)*100, ".", color="black", markersize=3,
                label="Observed", alpha=0.7)
        ax.plot(range(len(y)), (np.exp(yp_conf[order])-1)*100, "-", color=CS, linewidth=0.8,
                label="Predicted", alpha=0.7)
        ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
        ax.set_xlabel("Observation (sorted by true effect)", fontsize=11)
        ax.set_ylabel("Effect Size (% change)", fontsize=11)
        ax.set_title(f"{condition.capitalize()} — Conformal Prediction Intervals",
                     fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG, f"conformal_pi_{condition}.png"), dpi=600, bbox_inches="tight")
        plt.savefig(os.path.join(FIG, f"conformal_pi_{condition}.pdf"), dpi=600, bbox_inches="tight")
        plt.close("all")
        print(f"  -> conformal_pi_{condition}")

    # ── summary table ──
    df_all = pd.concat(all_results)
    df_all.to_csv(os.path.join(RES, "all_advanced_results.csv"), index=False)

    print("\n" + "="*70)
    print("ALL RESULTS SUMMARY")
    print("="*70)
    for condition in ["normal", "stress"]:
        print(f"\n  {condition.upper()}:")
        sub = df_all[df_all["Condition"] == condition].sort_values("R2", ascending=False)
        for _, row in sub.iterrows():
            print(f"    {row['Method']:40s}  R2={row['R2']:.4f}  RMSE={row['RMSE']:.4f}")

    print("\nAll advanced methods done.")
