"""Fig 1 — Evidence base and headline effects (4 panels).

A: Data structure (study and observation counts)
B: Mean effects across four indicators (stress vs non-stress)
C: Yield response — stress vs non-stress (forest-style)
D: 95% prediction intervals showing uncertainty
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from _common import (load_all, dl_meta, save_fig, panel_label,
                     C_NONSTRESS, C_STRESS, C_NEUTRAL, FIG_DIR)


def build():
    df = load_all()
    print(f"Total observations: {len(df)}")

    indicators = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]
    indicator_labels = ["Growth", "Photosynth.\npigments", "Biomass", "Yield"]

    # Compute pooled estimates per indicator × condition
    rows = []
    for cond in ["Non-stress", "Stress"]:
        for ind in indicators:
            sub = df[(df["Condition"] == cond) & (df["Performance"] == ind)].dropna(
                subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2:
                continue
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            theta, se, pct, ci_lo, ci_hi, pi_lo, pi_hi, tau2, Q, I2, k = res
            rows.append({"Condition": cond, "Indicator": ind, "k": k,
                          "pct": pct, "ci_lo": ci_lo, "ci_hi": ci_hi,
                          "pi_lo": pi_lo, "pi_hi": pi_hi, "I2": I2})
    res_df = pd.DataFrame(rows)
    print(res_df)

    # ── Figure ──
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.08, right=0.97, top=0.95, bottom=0.07)

    # ── Panel A: Data structure ──
    axA = fig.add_subplot(gs[0, 0])
    # Counts per indicator × condition
    counts = (df.groupby(["Performance", "Condition"]).size()
              .unstack(fill_value=0).reindex(indicators))
    x = np.arange(len(indicators))
    width = 0.38
    axA.bar(x - width/2, counts["Non-stress"], width,
            color=C_NONSTRESS, edgecolor="white", linewidth=0.5, label="Non-stress")
    axA.bar(x + width/2, counts["Stress"], width,
            color=C_STRESS, edgecolor="white", linewidth=0.5, label="Stress")
    for i, ind in enumerate(indicators):
        axA.text(i - width/2, counts.loc[ind, "Non-stress"] + 5,
                 f"{counts.loc[ind, 'Non-stress']}", ha="center", fontsize=7)
        axA.text(i + width/2, counts.loc[ind, "Stress"] + 5,
                 f"{counts.loc[ind, 'Stress']}", ha="center", fontsize=7)
    axA.set_xticks(x)
    axA.set_xticklabels(indicator_labels, fontsize=8)
    axA.set_ylabel("Number of observations")
    axA.set_title("Data structure across indicators", fontsize=9.5, pad=6)
    axA.legend(loc="upper left", fontsize=7)
    axA.set_ylim(0, max(counts.values.max() * 1.15, 50))
    panel_label(axA, "a")

    # ── Panel B: Mean effects across four indicators ──
    axB = fig.add_subplot(gs[0, 1])
    y_positions = np.arange(len(indicators))[::-1]
    offset = 0.18
    for cond, color, off in [("Non-stress", C_NONSTRESS, +offset),
                             ("Stress", C_STRESS, -offset)]:
        sub = res_df[res_df["Condition"] == cond].set_index("Indicator").reindex(indicators)
        for j, ind in enumerate(indicators):
            yp = y_positions[j] + off
            if pd.isna(sub.loc[ind, "pct"]):
                continue
            axB.errorbar(sub.loc[ind, "pct"], yp,
                         xerr=[[sub.loc[ind, "pct"] - sub.loc[ind, "ci_lo"]],
                               [sub.loc[ind, "ci_hi"] - sub.loc[ind, "pct"]]],
                         fmt="o", color=color, markersize=5,
                         elinewidth=1.6, capsize=2.5)
            axB.text(sub.loc[ind, "ci_hi"] + 2, yp,
                     f"{sub.loc[ind, 'pct']:+.1f}%",
                     va="center", fontsize=7, color=color)
    axB.axvline(0, color="black", linewidth=0.6, linestyle="--")
    axB.set_yticks(y_positions)
    axB.set_yticklabels(indicator_labels, fontsize=8)
    axB.set_xlabel("Effect size (% change, 95% CI)")
    axB.set_title("Mean effects across indicators", fontsize=9.5, pad=6)
    axB.set_xlim(-15, 75)
    handles = [plt.Line2D([0], [0], color=C_NONSTRESS, marker="o", lw=1.6,
                          label="Non-stress"),
               plt.Line2D([0], [0], color=C_STRESS, marker="o", lw=1.6,
                          label="Stress")]
    axB.legend(handles=handles, loc="lower right", fontsize=7)
    panel_label(axB, "b")

    # ── Panel C: Yield — stress vs non-stress with study-level forest ──
    axC = fig.add_subplot(gs[1, 0])
    yield_sub = df[df["Performance"] == "Yield"].dropna(subset=["lnRR", "vi"])
    yield_sub = yield_sub[yield_sub["vi"] > 0].copy()
    # Aggregate by study × condition
    grp = (yield_sub.groupby(["Condition", "study_id"])
           .agg(lnRR=("lnRR", "mean"), vi=("vi", "mean"),
                pct=("pct_change", "mean"))
           .reset_index())
    # Plot study-level dots (small) + pooled estimate (large)
    np.random.seed(7)
    for cond, color, y_base in [("Non-stress", C_NONSTRESS, 1),
                                 ("Stress", C_STRESS, 0)]:
        d = grp[grp["Condition"] == cond]
        jitter = np.random.uniform(-0.15, 0.15, size=len(d))
        sizes = 8 + 18 / (np.sqrt(d["vi"].values) + 0.05)
        sizes = np.clip(sizes, 6, 50)
        axC.scatter(d["pct"], y_base + jitter, s=sizes, color=color,
                    alpha=0.45, edgecolor="white", linewidth=0.3, zorder=2)
        # Pooled
        sub_full = yield_sub[yield_sub["Condition"] == cond]
        res = dl_meta(sub_full["lnRR"].values, sub_full["vi"].values)
        pct, ci_lo, ci_hi = res[2], res[3], res[4]
        axC.errorbar(pct, y_base, xerr=[[pct - ci_lo], [ci_hi - pct]],
                     fmt="D", color="black", markerfacecolor=color,
                     markersize=10, elinewidth=2.0, capsize=4, zorder=5)
        axC.text(pct, y_base - 0.32, f"Pooled: {pct:+.1f}%",
                 ha="center", fontsize=7.5, fontweight="bold", color=color)

    axC.axvline(0, color="black", linewidth=0.6, linestyle="--")
    axC.set_yticks([0, 1])
    axC.set_yticklabels(["Stress", "Non-stress"], fontsize=9)
    axC.set_xlabel("Yield response (% change)")
    axC.set_title("Yield: stress vs non-stress (each dot = study mean)",
                  fontsize=9.5, pad=6)
    axC.set_xlim(-60, 220)
    axC.set_ylim(-0.6, 1.6)
    panel_label(axC, "c")

    # ── Panel D: 95% PI vs CI — uncertainty ──
    axD = fig.add_subplot(gs[1, 1])
    y_positions = np.arange(len(indicators))[::-1]
    offset = 0.22
    for cond, color, off in [("Non-stress", C_NONSTRESS, +offset),
                             ("Stress", C_STRESS, -offset)]:
        sub = res_df[res_df["Condition"] == cond].set_index("Indicator").reindex(indicators)
        for j, ind in enumerate(indicators):
            yp = y_positions[j] + off
            if pd.isna(sub.loc[ind, "pi_lo"]):
                continue
            # PI as wider thinner bar
            axD.plot([sub.loc[ind, "pi_lo"], sub.loc[ind, "pi_hi"]],
                     [yp, yp], color=color, alpha=0.35, linewidth=5,
                     solid_capstyle="round")
            # CI as thicker line
            axD.plot([sub.loc[ind, "ci_lo"], sub.loc[ind, "ci_hi"]],
                     [yp, yp], color=color, linewidth=2.5,
                     solid_capstyle="butt")
            # Point estimate
            axD.scatter([sub.loc[ind, "pct"]], [yp], color="white",
                        edgecolor=color, s=22, linewidth=1.5, zorder=5)
    axD.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    axD.set_yticks(y_positions)
    axD.set_yticklabels(indicator_labels, fontsize=8)
    axD.set_xlabel("Effect size (%)")
    axD.set_title("95% confidence vs prediction intervals", fontsize=9.5, pad=6)
    axD.set_xlim(-40, 160)
    # Custom legend
    legend_handles = [
        plt.Line2D([0], [0], color=C_NEUTRAL, linewidth=2.5, label="95% CI"),
        plt.Line2D([0], [0], color=C_NEUTRAL, alpha=0.35, linewidth=5,
                   label="95% PI"),
        plt.Line2D([0], [0], marker="o", color="white",
                   markeredgecolor=C_NEUTRAL, markersize=6, linewidth=0,
                   label="Pooled"),
    ]
    axD.legend(handles=legend_handles, loc="lower right", fontsize=7, ncol=1)
    # Footnote inside plot — placed away from title
    axD.text(0.02, 0.05, "PI crossing 0 → a new study could\nplausibly find no benefit",
             transform=axD.transAxes,
             fontsize=6.8, color=C_NEUTRAL, style="italic", va="bottom")
    panel_label(axD, "d")

    save_fig(fig, "fig1_evidence_base")


if __name__ == "__main__":
    build()
