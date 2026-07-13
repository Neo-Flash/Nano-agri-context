# Context-Dependency of Nanotechnology in Sustainable Agriculture

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21338243.svg)](https://doi.org/10.5281/zenodo.21338243)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Code and data to reproduce the meta-analysis and machine-learning results reported in:

> *The Context-Dependency of Nanotechnology in Sustainable Agriculture: A Meta-analysis of 1,107 Observations Across Four Staple Crops*

The pipeline pools **1,107 experimental observations from 71 studies** on rice, wheat, maize and soybean, quantifies context-dependent nanoparticle effects on growth, photosynthetic pigments, biomass and yield, and shows that cross-study machine-learning prediction fails (GroupKFold R² ≤ 0; ICC = 0.88–0.90) — indicating that unmeasured study-level context dominates.

---

## Directory structure

```
github/
├── data/                        Raw effect-size datasets (CSV)
│   ├── normal_data.csv          485 non-stress observations
│   └── stress_data.csv          622 stress observations
├── code/
│   ├── 00_shared/               Shared utilities (dl_meta, egger, loso, ...)
│   ├── 01_data_preparation/     Data loading + QC + ML complete-case subsets
│   ├── 02_meta_analysis/        DL pooled effects + rma.mv sensitivity
│   ├── 03_robustness/           Egger (R metafor), trim-and-fill, LOSO, Pot vs Field
│   ├── 04_machine_learning/     GroupKFold regression, classification, clustering, Boruta, conformal
│   ├── 05_figures/              Fig 1–5 generation
│   └── 06_supplementary/        Table S2 fail-safe N, verify_all_numbers.py
├── figures/                     Regenerated main-text figures (PNG + PDF)
├── results/                     Intermediate CSV outputs
├── requirements.txt             Python dependencies
├── R_setup.R                    R dependency installation
├── environment.yml              Optional conda environment
├── run_all.sh                   One-command reproducibility script
├── ZENODO_INSTRUCTIONS.md       How to archive and get a DOI
├── LICENSE                      MIT
└── CITATION.cff                 Machine-readable citation
```

---

## Quick start (one command)

```bash
git clone https://github.com/YOUR_USERNAME/nano-agriculture-meta.git
cd nano-agriculture-meta
bash run_all.sh
```

`run_all.sh` will install Python + R dependencies, run all analyses in order, regenerate every main-text figure, and write a full verification report to `results/verify_all_numbers.txt`.

---

## Manual reproduction (step-by-step)

### Prerequisites

- **Python ≥ 3.9** with `pip`
- **R ≥ 4.0** with `Rscript` on `$PATH`

### 1. Install dependencies

```bash
# Python
pip install -r requirements.txt

# R  (installs metafor, dplyr, tidyr)
Rscript R_setup.R
```

### 2. Run the pipeline in order

| Step | Script | Purpose |
|------|--------|---------|
| 01 | `code/01_data_preparation/01_load_and_check.py` | Load, QC, save ML subsets |
| 02a | `code/02_meta_analysis/02a_pooled_effects_and_PI.py` | Python DL pooled effects + 95% PI |
| 02b | `Rscript code/02_meta_analysis/02b_rma_mv_sensitivity.R` | **R** `rma.mv` multilevel sensitivity |
| 03a | `Rscript code/03_robustness/03a_egger_standard_R.R` | **R** `metafor::regtest()` Egger test |
| 03b | `code/03_robustness/03b_trim_and_fill.py` | Trim-and-fill bias correction |
| 03c | `code/03_robustness/03c_loso.py` | Leave-one-study-out sensitivity |
| 03d | `code/03_robustness/03d_pot_field_loso.py` | Pot vs Field yield subgroup |
| 04a | `code/04_machine_learning/04a_data_preparation.py` | ML-ready complete-case yield subsets |
| 04b | `code/04_machine_learning/04b_classification.py` | Binary/3-class classification |
| 04c | `code/04_machine_learning/04c_optimal_regime.py` | Regime heatmaps + method interactions |
| 04d | `code/04_machine_learning/04d_kmeans_clustering.py` | K-Means clusters (k = 6) |
| 04e | `code/04_machine_learning/04e_regression_conformal.py` | 8 regression algorithms under GroupKFold + conformal PIs |
| 04f | `code/04_machine_learning/04f_boruta_and_extras.py` | Boruta feature selection (100 shadow iterations) |
| 05 | `code/05_figures/fig1_evidence_base.py` … `fig5_predictability_barrier.py` | Regenerate main-text figures |
| 06 | `code/06_supplementary/verify_all_numbers.py` | Numerical audit vs paper text |

### 3. Verify reproduction

```bash
python code/06_supplementary/verify_all_numbers.py
```

Should print **1,107 observations / 71 studies** and match all 8 subgroup pooled effects to the paper.

---

## Key methodological decisions (locked in this repository)

| Design choice | Setting | Rationale |
|---|---|---|
| **Random-effects estimator (primary)** | DerSimonian–Laird (DL) on lnRR, single-level | Closed-form, numerically stable at extreme within-study variance |
| **Random-effects estimator (sensitivity)** | REML in `rma.mv` with `random = list(~1|study_id, ~1|Crop)` | Confirms DL point estimates differ by ≤ 6 pp; qualitative ranking unchanged |
| **95% prediction interval** | `PI = θ̂ ± t_{k−2,0.025} · √(τ̂² + SE²)`; τ̂² from DL | Higgins–Thompson formula |
| **Egger's test** | `metafor::regtest()` in R (random-effects weighting, intercept of standardised effect on precision) | Standard, avoids ad-hoc weighting |
| **Publication-bias correction** | Simplified trim-and-fill with DL re-estimation each loop | k_imputed = 0 in all 8 subgroups; likely reflects extreme I² > 94% |
| **Fail-safe N** | Rosenthal via `metafor::fsn()` (target α = 0.05) | Nfs = 269,068 ≫ 5k + 10 threshold |
| **Cross-validation** | `GroupKFold(n_splits=5)` grouped by study ID | Guarantees no study appears in both train and test |
| **Missing values** | Listwise / complete-case deletion (no imputation, no "Unknown") | Restricts modelling to observations with full M-A-E context |
| **Categorical encoding** | `LabelEncoder` fitted once over full dataset | Not a leak: label IDs carry no train-set statistics; tree/distance models are indifferent to ordinal interpretation |
| **Continuous scaling** | `StandardScaler` fitted **per training fold**, applied `transform`-only to test fold | Prevents test-fold statistics from leaking into fitting |
| **Dose harmonisation** | Only mass-per-volume / mass-per-mass concentrations (mg L⁻¹, mg kg⁻¹, ppm) enter continuous dose ML and meta-regression; kg ha⁻¹ area-application data are excluded | See `Methods §Dosage Data Harmonization` |
| **Random seed** | `numpy.random.seed(42)`; `random_state=42` for scikit-learn | Boruta uses 100 shadow permutations |
| **ICC** | `σ²_between / (σ²_between + σ²_within)` on lnRR, computed per condition on yield subset | 0.904 (non-stress) / 0.882 (stress) |
| **Conformal prediction intervals** | Split conformal, 95% target, exchangeability | Empirical coverage 91% / 94%, width ≈ ±0.80 log RR |

---

## Key numbers reproduced by this code

| Result | Value | Location |
|---|---|---|
| Total observations / studies | **1,107 / 71** | Abstract, Methods §Data Extraction |
| Non-stress yield pooled effect | **+17.6%** (95% CI [15.1, 20.2]; PI [−8.7, +51.4]; k = 196) | Fig 1b, Table S3 |
| Stress yield pooled effect | **+22.5%** (95% CI [20.6, 24.4]; PI [−0.6, +50.8]; k = 277) | Fig 1b, Table S3 |
| Pot vs Field yield gap | +26.0% vs +8.9% (non-stress); +28.9% vs +18.0% (stress) | Fig 2a |
| Egger — Stress Yield | intercept 0.056, P = 3.6 × 10⁻¹⁵ | Table S3 |
| Egger — subgroups significant at P < 0.05 | 4 of 8 | Results §Publication bias |
| ICC (yield) | 0.904 (non-stress) / 0.882 (stress) | Methods §ML, Fig 5b |
| GroupKFold R² (all 8 algorithms) | ≤ 0 for both conditions | Fig 5a, Table S4 |
| Conformal coverage / width | 91% / 94%; ±0.80 log RR | Fig 5c |
| Boruta hit rates (NS) | NPs type 98% · size 99% · concentration 93% · method 0% · crop 91% | Fig 3d |
| Fail-safe N (Rosenthal) | 269,068 (≫ 5k + 10) | Methods §Publication Bias |

---

## Data description

Both CSV files share a common effect-size schema:

| Column | Description |
|---|---|
| `number` | Study ID (integer) |
| `Performance` | One of `Growth`, `Photosynthetic Pigment`, `Biomass`, `Yield` |
| `NPs_type` | Nanoparticle chemical type (Fe, Zn, Si, Ce, Cu, Ti, …) |
| `NPs_size` | Reported particle size (nm) |
| `Concentration` | Applied concentration (mg L⁻¹ / mg kg⁻¹ / ppm; kg ha⁻¹ observations are flagged) |
| `Method` | Application method (Foliar, Seed, Soil) |
| `Crop` (aka `Crops`) | Rice, Wheat, Maize, or Soybean |
| `Stress_type` | (stress dataset only) Drought, Salt, Heavy_Metal, Climatic_Stress, Alkaline_soils, Insect |
| `experiment` | `Pot` or `Field` |
| `lnRR` | Log response ratio, `ln(X_treatment / X_control)` |
| `vi` | Sampling variance of `lnRR` |

Both datasets were extracted from peer-reviewed primary literature published up to December 2024; extraction protocol is described in `Methods §Literature Search and Screening Strategy`.

---

## Software environment

Verified with:
- Python 3.9, `numpy 1.24`, `pandas 2.0`, `scipy 1.10`, `scikit-learn 1.3`, `matplotlib 3.7`, `xgboost 2.0`, `catboost 1.2`, `lightgbm 4.0`, `Boruta 0.3`, `mapie 0.6`
- R 4.3, `metafor 4.4-0`, `dplyr 1.1`, `tidyr 1.3`

Later versions should also work; if not, please open an issue.

---

## License

Code: **MIT** (see `LICENSE`).
Data (`data/*.csv`): released alongside code under **CC BY 4.0** — please cite the original primary studies as well as this compilation.

---

## Citation

If you use this repository, please cite:

```
[Author list]. The Context-Dependency of Nanotechnology in Sustainable Agriculture. [Journal] (Year).
Data & code archived at Zenodo, DOI: TO_BE_ASSIGNED.
```

See `CITATION.cff` for a machine-readable version.

---

## Contact

Open an issue on GitHub for reproducibility questions.
