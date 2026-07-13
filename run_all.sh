#!/usr/bin/env bash
# One-command reproduction of every analysis and figure in the paper.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Installing dependencies ==="
pip install -q -r requirements.txt
Rscript R_setup.R

mkdir -p results figures

echo -e "\n=== 01. Data preparation ==="
python code/01_data_preparation/01_load_and_check.py

echo -e "\n=== 02. Meta-analysis ==="
python code/02_meta_analysis/02a_pooled_effects_and_PI.py
Rscript code/02_meta_analysis/02b_rma_mv_sensitivity.R || echo "  (rma.mv sensitivity: R skipped)"

echo -e "\n=== 03. Robustness ==="
Rscript code/03_robustness/03a_egger_standard_R.R
python code/03_robustness/03b_trim_and_fill.py
python code/03_robustness/03c_loso.py
python code/03_robustness/03d_pot_field_loso.py

echo -e "\n=== 04. Machine learning ==="
python code/04_machine_learning/04a_data_preparation.py
python code/04_machine_learning/04d_kmeans_clustering.py
python code/04_machine_learning/04e_regression_conformal.py
python code/04_machine_learning/04f_boruta_and_extras.py

echo -e "\n=== 05. Figures ==="
python code/05_figures/fig1_evidence_base.py
python code/05_figures/fig2_field_realism.py
python code/05_figures/fig3_mae_hierarchy.py
python code/05_figures/fig4_operating_windows.py
python code/05_figures/fig5_predictability_barrier.py

echo -e "\n=== 06. Verification ==="
python code/06_supplementary/verify_all_numbers.py | tee results/verify_all_numbers.txt

echo -e "\n✅ Done — see results/ and figures/"
