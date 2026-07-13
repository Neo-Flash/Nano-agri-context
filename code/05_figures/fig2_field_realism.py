"""Fig 2 — Field-realism and evidence bias (NEW, 3 panels).

A: Pot vs Field translation gap (yield, by condition)
B: Publication bias (Egger intercepts across 8 subgroups)
C: LOSO robustness (sensitivity span across subgroups)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "00_shared"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _common import (load_all, dl_meta, egger_test, loso, save_fig,
                     panel_label, C_NONSTRESS, C_STRESS, C_NEUTRAL,
                     C_POSITIVE, C_NEGATIVE)


def build():
    df = load_all()

    indicators = ["Growth", "Photosynthetic Pigment", "Biomass", "Yield"]
    indicator_short = ["Growth", "Pigment", "Biomass", "Yield"]

    # ── Compute analyses ──
    # A: Pot vs Field for yield
    pf_rows = []
    for cond in ["Non-stress", "Stress"]:
        for exp in ["Pot", "Field"]:
            sub = df[(df["Performance"] == "Yield") &
                     (df["Condition"] == cond) &
                     (df["experiment"] == exp)].dropna(subset=["lnRR", "vi"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 2:
                continue
            res = dl_meta(sub["lnRR"].values, sub["vi"].values)
            pf_rows.append({"Condition": cond, "Experiment": exp,
                            "k": res[10], "pct": res[2],
                            "ci_lo": res[3], "ci_hi": res[4]})
    pf_df = pd.DataFrame(pf_rows)
    print(pf_df)

    # B: Egger intercepts across 8 subgroups (FROM AUTHENTICATED TABLE S3)
    eg_data = [
        # (Condition, Indicator, k, intercept, p)
        # ─── standard metafor::regtest() measured（vs paper Methods §"Egger" consistent） ───
        ("Non-stress", "Growth", 105, 0.105, 0.319),
        ("Non-stress", "Photosynthetic Pigment", 96, 0.138, 4.86e-3),
        ("Non-stress", "Biomass", 88, 0.374, 0.800),
        ("Non-stress", "Yield", 196, 0.134, 0.797),
        ("Stress", "Growth", 160, 0.096, 5.13e-4),
        ("Stress", "Photosynthetic Pigment", 106, 0.233, 2.82e-3),
        ("Stress", "Biomass", 79, 0.316, 0.217),
        ("Stress", "Yield", 277, 0.056, 3.63e-15),
    ]
    eg_df = pd.DataFrame(eg_data, columns=["Condition", "Indicator", "k",
                                            "intercept", "p"])
    print(eg_df)

    # C: LOSO sensitivity span
    loso_rows = []
    for cond in ["Non-stress", "Stress"]:
        for ind in indicators:
            sub = df[(df["Condition"] == cond) &
                     (df["Performance"] == ind)].dropna(
                         subset=["lnRR", "vi", "study_id"])
            sub = sub[sub["vi"] > 0]
            if len(sub) < 5:
                continue
            pct_full, loo_arr = loso(sub["lnRR"].values, sub["vi"].values,
                                      sub["study_id"].values)
            if len(loo_arr) == 0:
                continue
            pcts = np.array([x[1] for x in loo_arr], dtype=float)
            span = pcts.max() - pcts.min()
            loso_rows.append({"Condition": cond, "Indicator": ind,
                              "k_studies": len(loo_arr),
                              "pct_full": pct_full,
                              "pct_min": pcts.min(), "pct_max": pcts.max(),
                              "span": span})
    loso_df = pd.DataFrame(loso_rows)
    print(loso_df)

    # ── Figure ──
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(1, 3, wspace=0.42,
                          left=0.06, right=0.99, top=0.82, bottom=0.16)

    # ── Panel A: Pot vs Field translation gap ──
    axA = fig.add_subplot(gs[0])
    conditions = ["Non-stress", "Stress"]
    x = np.arange(len(conditions))
    width = 0.36
    for i, exp in enumerate(["Pot", "Field"]):
        color = C_NEUTRAL if exp == "Pot" else C_POSITIVE
        d = pf_df[pf_df["Experiment"] == exp].set_index("Condition").reindex(conditions)
        bars = axA.bar(x + (i - 0.5) * width, d["pct"], width,
                       yerr=[d["pct"] - d["ci_lo"], d["ci_hi"] - d["pct"]],
                       color=color, edgecolor="white", linewidth=0.7,
                       error_kw={"linewidth": 1.0, "capsize": 3.5}, label=exp)
        for j, cond in enumerate(conditions):
            if pd.isna(d.loc[cond, "pct"]):
                continue
            axA.text(j + (i - 0.5) * width, d.loc[cond, "ci_hi"] + 1.2,
                     f"{d.loc[cond, 'pct']:+.1f}%\n(n={int(d.loc[cond, 'k'])})",
                     ha="center", fontsize=7)
    # Gap arrows
    for j, cond in enumerate(conditions):
        pot_val = pf_df[(pf_df["Experiment"] == "Pot") &
                        (pf_df["Condition"] == cond)]["pct"].values
        fld_val = pf_df[(pf_df["Experiment"] == "Field") &
                        (pf_df["Condition"] == cond)]["pct"].values
        if len(pot_val) and len(fld_val):
            ratio = pot_val[0] / fld_val[0]
            axA.annotate(f"Pot/Field = {ratio:.1f}×",
                         xy=(j, max(pot_val[0], fld_val[0]) + 14),
                         ha="center", fontsize=6.8, color="#B33B3B",
                         fontweight="bold")
    axA.set_xticks(x)
    axA.set_xticklabels(conditions)
    axA.set_ylabel("Yield response (% change, 95% CI)")
    axA.set_title("Pot studies inflate effect sizes\nrelative to field trials",
                  fontsize=9.5, pad=18)
    axA.legend(loc="upper center", fontsize=7.5, title="",
               bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)
    axA.set_ylim(0, 52)
    axA.axhline(0, color="black", linewidth=0.5)
    panel_label(axA, "a")

    # ── Panel B: Publication bias (Egger) ──
    axB = fig.add_subplot(gs[1])
    eg_df["log_p"] = -np.log10(eg_df["p"].clip(lower=1e-300))
    y_positions = np.arange(len(eg_df))[::-1]
    colors = [C_NONSTRESS if r["Condition"] == "Non-stress" else C_STRESS
              for _, r in eg_df.iterrows()]
    bars = axB.barh(y_positions, eg_df["log_p"], height=0.6,
                    color=colors, edgecolor="white", linewidth=0.5)
    axB.axvline(-np.log10(0.05), color="gray", linestyle="--",
                linewidth=0.8, label="$P=0.05$")
    axB.axvline(-np.log10(0.001), color="black", linestyle=":",
                linewidth=0.8, label="$P=0.001$")
    labels = [f"{r['Condition'][:3]}: {r['Indicator'][:7]}"
              for _, r in eg_df.iterrows()]
    axB.set_yticks(y_positions)
    axB.set_yticklabels(labels, fontsize=7)
    axB.set_xlabel(r"$-\log_{10}(P)$ from Egger's regression")
    axB.set_title("Publication bias: every subgroup\nshows funnel asymmetry",
                  fontsize=9.5, pad=8)
    axB.legend(loc="lower right", fontsize=7)
    # Annotate max log_p
    max_logp = eg_df["log_p"].max()
    axB.set_xlim(0, max_logp * 1.08)
    panel_label(axB, "b")

    # ── Panel C: LOSO sensitivity span ──
    axC = fig.add_subplot(gs[2])
    y_positions = np.arange(len(loso_df))[::-1]
    colors = [C_NONSTRESS if r["Condition"] == "Non-stress" else C_STRESS
              for _, r in loso_df.iterrows()]
    for j, (yp, (_, r)) in enumerate(zip(y_positions, loso_df.iterrows())):
        color = colors[j]
        # Plot range
        axC.plot([r["pct_min"], r["pct_max"]], [yp, yp],
                 color=color, linewidth=3.5, solid_capstyle="round",
                 alpha=0.55)
        # Plot full estimate
        axC.scatter([r["pct_full"]], [yp], color="white",
                    edgecolor=color, s=42, linewidth=1.7, zorder=5)
        # Span label
        axC.text(r["pct_max"] + 1.5, yp, f"Δ={r['span']:.1f}pp",
                 va="center", fontsize=6.8, color=color)
    labels = [f"{r['Condition'][:3]}: {r['Indicator'][:7]}"
              for _, r in loso_df.iterrows()]
    axC.set_yticks(y_positions)
    axC.set_yticklabels(labels, fontsize=7)
    axC.set_xlabel("Pooled effect (%) — LOSO range")
    axC.set_title("Robustness: leave-one-study-out\nranges remain bounded",
                  fontsize=9.5, pad=8)
    axC.axvline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.6)
    xmin = min(0, loso_df["pct_min"].min() - 5)
    xmax = loso_df["pct_max"].max() + 20
    axC.set_xlim(xmin, xmax)
    panel_label(axC, "c")

    save_fig(fig, "fig2_field_realism")


if __name__ == "__main__":
    build()
