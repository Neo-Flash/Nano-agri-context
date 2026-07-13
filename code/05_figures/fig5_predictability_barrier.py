"""Fig 5 — Predictability barrier.

A: GroupKFold R² distribution — null vs all 8 models (yield, both conditions)
B: ICC variance decomposition (study-level vs within-study)
C: Conformal prediction interval width vs coverage
D: Why prediction fails — schematic missing-context summary
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle
from _common import (load_all, save_fig, panel_label,
                     C_NONSTRESS, C_STRESS, C_NEUTRAL,
                     C_POSITIVE, C_NEGATIVE, C_WARM, C_COOL)


# Authenticated GroupKFold R² results from previous experiments (Table S-ML-1)
R2_DATA = {
    "Non-stress": {
        "Random Forest (RF)": -0.44,
        "XGBoost (XGB)": -0.51,
        "CatBoost (CB)": -0.51,
        "LightGBM (LGBM)": -0.48,
        "Ridge": -0.21,
        "Lasso": -0.19,
        "SVR (RBF)": -0.22,
        "KNN": -0.78,
    },
    "Stress": {
        "Random Forest (RF)": -0.09,
        "XGBoost (XGB)": -0.04,
        "CatBoost (CB)": -0.04,
        "LightGBM (LGBM)": -0.05,
        "Ridge": -0.00,
        "Lasso": +0.01,
        "SVR (RBF)": -0.01,
        "KNN": -0.25,
    },
}

ICC_DATA = {"Non-stress": 0.904, "Stress": 0.882}


def compute_icc(df, indicator="Yield"):
    """Compute ICC = sigma2_between / (sigma2_between + sigma2_within)."""
    res = {}
    for cond in ["Non-stress", "Stress"]:
        sub = df[(df["Condition"] == cond) &
                 (df["Performance"] == indicator)].dropna(
                     subset=["lnRR", "study_id"])
        sub = sub[sub["vi"] > 0]
        groups = sub["study_id"].values
        y = sub["lnRR"].values
        if len(np.unique(groups)) < 2:
            continue
        overall_mean = y.mean()
        ss_between = 0.0
        ss_within = 0.0
        for g in np.unique(groups):
            yg = y[groups == g]
            ss_between += len(yg) * (yg.mean() - overall_mean) ** 2
            ss_within += np.sum((yg - yg.mean()) ** 2)
        sigma2_between = ss_between / max(1, len(np.unique(groups)) - 1)
        sigma2_within = ss_within / max(1, len(y) - len(np.unique(groups)))
        icc = sigma2_between / (sigma2_between + sigma2_within)
        res[cond] = {"icc": icc, "between": sigma2_between,
                     "within": sigma2_within}
    return res


def build():
    df = load_all()
    icc_obs = compute_icc(df, "Yield")
    print("Observed ICC:", {c: round(v["icc"], 3) for c, v in icc_obs.items()})

    # ── Figure ──
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32,
                          left=0.08, right=0.97, top=0.94, bottom=0.08)

    # ── Panel A: GroupKFold R² distribution ──
    axA = fig.add_subplot(gs[0, 0])
    conditions = ["Non-stress", "Stress"]
    x = np.arange(len(conditions))
    width = 0.32

    for i, cond in enumerate(conditions):
        r2_vals = list(R2_DATA[cond].values())
        # Strip plot of all algorithms at single x position
        # Use jitter
        np.random.seed(13 + i)
        jitter = np.random.uniform(-0.10, 0.10, size=len(r2_vals))
        color = C_NONSTRESS if cond == "Non-stress" else C_STRESS
        # Show all 8 R² as small dots
        axA.scatter(np.full(len(r2_vals), x[i] - 0.18) + jitter, r2_vals,
                    s=22, color=color, alpha=0.55, edgecolor="white",
                    linewidth=0.4, zorder=3, label=f"{cond}: 8 algorithms")
        # Best (= max) as larger marker
        best = max(r2_vals)
        axA.scatter([x[i] + 0.18], [best], s=70, color=color, marker="D",
                    edgecolor="black", linewidth=0.8, zorder=4,
                    label=f"{cond}: best model")
        # Annotate best — place ABOVE the marker so the label sits above the
        # null-model dashed line at R^2 = 0
        axA.text(x[i] + 0.22, best + 0.06, f"best $R^2$={best:+.2f}",
                 fontsize=7, va="bottom", color=color)
        # Mean as horizontal line, label placed to the RIGHT to avoid clipping
        mean_r2 = np.mean(r2_vals)
        axA.plot([x[i] - 0.32, x[i] - 0.04], [mean_r2, mean_r2],
                 color=color, linewidth=1.8, zorder=5)
        axA.text(x[i] - 0.02, mean_r2, f"mean={mean_r2:+.2f}",
                 fontsize=6.8, ha="left", va="center", color=color)

    axA.axhline(0, color="black", linewidth=1.0, linestyle="--", zorder=2)
    axA.text(0.03, 0.02, "Null model ($R^2=0$) ← predicting grand mean",
             transform=axA.transAxes, fontsize=7, ha="left", va="bottom",
             style="italic", color="#555")
    axA.set_xticks(x)
    axA.set_xticklabels(conditions)
    axA.set_ylabel(r"GroupKFold $R^2$ (out-of-sample)")
    axA.set_title("All algorithms fail to outperform null model",
                  fontsize=9.5, pad=6)
    axA.set_xlim(-0.6, 1.6)
    axA.set_ylim(-0.95, 0.20)
    # Legend
    handles = [plt.Line2D([0], [0], marker="o", color="gray", lw=0,
                           markersize=4, label="Individual algorithm"),
               plt.Line2D([0], [0], marker="D", color="gray", lw=0,
                           markersize=7, label="Best per condition"),
               plt.Line2D([0], [0], color="gray", linewidth=1.8,
                           label="Mean across algorithms")]
    axA.legend(handles=handles, loc="lower right", fontsize=7)
    panel_label(axA, "a")

    # ── Panel B: ICC variance decomposition ──
    axB = fig.add_subplot(gs[0, 1])
    # Stacked bar: between-study vs within-study variance share
    iccs = [ICC_DATA["Non-stress"], ICC_DATA["Stress"]]
    within = [1 - i for i in iccs]
    bars1 = axB.bar(x, iccs, color=C_NEUTRAL, edgecolor="white",
                    linewidth=0.7, label="Between-study (ICC)")
    bars2 = axB.bar(x, within, bottom=iccs, color="#D9D9D9",
                    edgecolor="white", linewidth=0.7,
                    label="Within-study")
    for j, cond in enumerate(conditions):
        axB.text(x[j], iccs[j] / 2, f"ICC\n={iccs[j]:.2f}",
                 ha="center", va="center", fontsize=9, color="white",
                 fontweight="bold")
        axB.text(x[j], iccs[j] + within[j] / 2, f"{within[j]:.2f}",
                 ha="center", va="center", fontsize=8, color="#555")
    axB.set_xticks(x)
    axB.set_xticklabels(conditions)
    axB.set_ylabel("Fraction of variance in yield log-RR")
    axB.set_title("Variance decomposition: study identity dominates",
                  fontsize=9.5, pad=18)
    axB.set_ylim(0, 1.35)
    axB.legend(loc="upper right", fontsize=7, framealpha=0.95,
               bbox_to_anchor=(1.0, 1.0))
    axB.text(0.02, 0.98, "Cross-study prediction requires within-study\n"
              "variance to dominate. Here it does not.",
              transform=axB.transAxes, fontsize=7, va="top",
              style="italic", color="#555")
    panel_label(axB, "b")

    # ── Panel C: Conformal width vs coverage ──
    axC = fig.add_subplot(gs[1, 0])
    # Authenticated values from previous experiments
    coverage = [0.91, 0.94]
    width_logRR = [1.60, 1.45]  # approx ±0.80
    # Convert width to %
    # interval ±0.80 logRR → from exp(-0.80) to exp(+0.80) → roughly -55% to +123%
    # We'll show the average half-width on log scale and as a typical interval
    x_pts = np.array([0.91, 0.94])
    y_pts = np.array([1.60, 1.45])
    for cond, color, xv, yv, cov_label in zip(
            conditions, [C_NONSTRESS, C_STRESS], x_pts, y_pts,
            ["Non-stress (cov=91%)", "Stress (cov=94%)"]):
        axC.scatter([xv], [yv], s=120, color=color, edgecolor="black",
                    linewidth=0.8, zorder=5, label=cov_label)
        axC.text(xv + 0.003, yv + 0.04, cov_label, fontsize=7, color=color)
    axC.axvline(0.95, color="#888", linestyle="--", linewidth=0.7)
    axC.text(0.951, 0.05, "Target = 95%", fontsize=7, color="#555",
             rotation=90, va="bottom")
    axC.set_xlim(0.86, 0.97)
    axC.set_ylim(0.0, 2.0)
    axC.set_xlabel("Empirical coverage of 95% conformal interval")
    axC.set_ylabel("Mean interval width (log-RR units)")
    axC.set_title("Calibrated, but wide: irreducible uncertainty\n"
                  "≈ ±0.80 log-RR (≈ −55% to +123% on yield)",
                  fontsize=9.5, pad=6)
    panel_label(axC, "c")

    # ── Panel D: Missing-context schematic ──
    axD = fig.add_subplot(gs[1, 1])
    axD.axis("off")
    axD.set_title("Where the predictive signal lives",
                  fontsize=9.5, pad=6)
    # Concentric framework
    # Outer ring: study-level (dominant)
    outer = Wedge((0.5, 0.5), 0.42, 0, 360, width=0.10,
                  facecolor=C_NEUTRAL, edgecolor="white", lw=1.2,
                  transform=axD.transAxes)
    inner = Wedge((0.5, 0.5), 0.30, 0, 360, width=0.13,
                  facecolor="#D9D9D9", edgecolor="white", lw=1.2,
                  transform=axD.transAxes)
    axD.add_patch(outer)
    axD.add_patch(inner)
    # Center
    axD.add_patch(plt.Circle((0.5, 0.5), 0.16, transform=axD.transAxes,
                              facecolor=C_POSITIVE, alpha=0.18,
                              edgecolor="none"))
    # Labels
    axD.text(0.5, 0.5, "Recorded\nM–A–E\nfeatures",
             transform=axD.transAxes, fontsize=8.5,
             ha="center", va="center", fontweight="bold",
             color="#2D7A33")
    axD.text(0.5, 0.78, "Study-level context\n"
              "(soil, climate, cultivar, protocol)",
              transform=axD.transAxes, fontsize=8, ha="center", va="center",
              color="black", fontweight="bold")
    axD.text(0.5, 0.22, "Within-study\nbiological noise",
             transform=axD.transAxes, fontsize=8, ha="center", va="center",
             color="#555")

    # Side annotations: ICC values
    axD.annotate("≈ 51–67% of variance",
                 xy=(0.78, 0.65), xycoords="axes fraction",
                 xytext=(0.98, 0.92), textcoords="axes fraction",
                 fontsize=7.2, color=C_NEUTRAL,
                 arrowprops=dict(arrowstyle="-",
                                 color=C_NEUTRAL, lw=0.7),
                 ha="right")
    axD.annotate("Features explain only\nthe inner ring",
                 xy=(0.36, 0.42), xycoords="axes fraction",
                 xytext=(0.02, 0.22), textcoords="axes fraction",
                 fontsize=7.2, color="#2D7A33",
                 arrowprops=dict(arrowstyle="->",
                                 color="#2D7A33", lw=0.7),
                 ha="left", va="center")
    panel_label(axD, "d", x=0.0, y=1.0)

    save_fig(fig, "fig5_predictability_barrier")


if __name__ == "__main__":
    build()
