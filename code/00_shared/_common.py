"""Shared utilities: data loading, DL-meta, standard styles"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
DATA_DIR = os.environ.get("NANO_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data"))
FIG_DIR = "/Users/flash/Desktop/project/Revised_Paper/figures"
os.makedirs(FIG_DIR, exist_ok=True)

# ── Nature Sustainability-style colors ──
C_NONSTRESS = "#2E86AB"   # blue
C_STRESS    = "#C73E1D"   # red
C_NEUTRAL   = "#4C5760"   # dark gray
C_POSITIVE  = "#46834D"   # green
C_NEGATIVE  = "#B33B3B"   # red
C_WARM      = "#E89B3F"   # orange
C_COOL      = "#5E72A9"   # slate blue

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
})

np.random.seed(42)


def load_all():
    """load normal + stress data."""
    frames = []
    for f, cond in [("normal_data.csv", "Non-stress"), ("stress_data.csv", "Stress")]:
        df = pd.read_csv(os.path.join(DATA_DIR, f), encoding="latin-1")
        df.columns = df.columns.str.strip()
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].str.strip()
        if "Crops" in df.columns and "Crop" not in df.columns:
            df = df.rename(columns={"Crops": "Crop"})
        if "Stress_type" not in df.columns and "Stress_type " in df.columns:
            df = df.rename(columns={"Stress_type ": "Stress_type"})
        df["Condition"] = cond
        df["pct_change"] = (np.exp(df["lnRR"]) - 1) * 100
        df["study_id"] = df["number"].astype(int)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def dl_meta(y, v):
    """DerSimonian-Laird random-effects.
    Returns: theta, se, pct, ci_lo, ci_hi, pi_lo, pi_hi, tau2, Q, I2, k
    """
    y = np.asarray(y, float); v = np.asarray(v, float)
    mask = (v > 0) & np.isfinite(y) & np.isfinite(v)
    y = y[mask]; v = v[mask]
    if len(y) < 2:
        return tuple([np.nan]*10 + [len(y)])
    w = 1.0/v
    theta_fe = np.sum(w*y)/np.sum(w)
    Q = float(np.sum(w*(y-theta_fe)**2))
    k = len(y)
    C = float(np.sum(w) - np.sum(w**2)/np.sum(w))
    tau2 = max(0.0, (Q - (k-1))/C) if C > 0 else 0.0
    w_star = 1.0/(v + tau2)
    theta = float(np.sum(w_star*y)/np.sum(w_star))
    se = float(1.0/np.sqrt(np.sum(w_star)))
    pct = (np.exp(theta)-1)*100
    ci_lo = (np.exp(theta - 1.96*se) - 1)*100
    ci_hi = (np.exp(theta + 1.96*se) - 1)*100
    from scipy import stats
    if k > 2:
        t_crit = stats.t.ppf(0.975, df=k-2)
        pi_half = t_crit * np.sqrt(tau2 + se**2)
        pi_lo = (np.exp(theta - pi_half) - 1)*100
        pi_hi = (np.exp(theta + pi_half) - 1)*100
    else:
        pi_lo = pi_hi = np.nan
    I2 = max(0.0, (Q - (k-1))/Q*100) if Q > 0 else 0.0
    return theta, se, pct, ci_lo, ci_hi, pi_lo, pi_hi, tau2, Q, I2, k


def egger_test(y, v):
    """Egger's regression test of funnel asymmetry.
    Returns: intercept, p_value
    """
    y = np.asarray(y, float); v = np.asarray(v, float)
    mask = (v > 0) & np.isfinite(y) & np.isfinite(v)
    y = y[mask]; v = v[mask]
    se = np.sqrt(v)
    # Regression of y/se on 1/se
    X = np.column_stack([np.ones(len(y)), 1.0/se])
    Y = y/se
    try:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
        resid = Y - X @ beta
        n = len(Y); p = 2
        sigma2 = np.sum(resid**2) / (n - p)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se_intercept = np.sqrt(cov[0, 0])
        t_stat = beta[0] / se_intercept
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-p))
        return float(beta[0]), float(p_value)
    except Exception:
        return np.nan, np.nan


def loso(y, v, study_id):
    """Leave-one-study-out: returns array of pooled pct estimates."""
    y = np.asarray(y, float); v = np.asarray(v, float)
    study_id = np.asarray(study_id)
    pct_full = dl_meta(y, v)[2]
    studies = np.unique(study_id)
    out = []
    for s in studies:
        mask = study_id != s
        if mask.sum() < 2:
            continue
        pct_loo = dl_meta(y[mask], v[mask])[2]
        out.append((s, pct_loo, pct_loo - pct_full))
    return pct_full, np.array(out, dtype=object)


def save_fig(fig, name):
    p_png = os.path.join(FIG_DIR, f"{name}.png")
    p_pdf = os.path.join(FIG_DIR, f"{name}.pdf")
    fig.savefig(p_png, dpi=600, bbox_inches="tight")
    fig.savefig(p_pdf, dpi=600, bbox_inches="tight")
    print(f"  Saved: {name}.png + {name}.pdf")
    plt.close(fig)


def panel_label(ax, label, x=-0.18, y=1.05, size=12, weight="bold"):
    ax.text(x, y, label, transform=ax.transAxes, fontsize=size,
            fontweight=weight, va="bottom", ha="left")
