# Fig1_Table_S3_main_meta_analysis.R
#
# usingPurpose：reproducepaper Fig.1、Table S3 in allpooled effect、PI、Egger、LOSO、Trim-and-fill。
# Tool: R + metafor(paper Methods L277 explicitly requires）
# ---------------------------------------------------------------------------
# paper Methods L277：withusing rma() / rma.mv() with study ID + Crop random effects
# paper Methods L253：REML estimator
# paper Methods L259：PI formulain  τ̂² withusing DL estimator
# Methods L271：Egger bysubgrouptest
# paper Methods L273：LOSO sensitivity analysis
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(metafor)
  library(dplyr)
})

DATA_DIR <- "/Users/flash/Desktop/project"
OUT_DIR  <- "/Users/flash/Desktop/project/Revised_Paper/results/paperfiguretablecode/R_outputs"
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# Data loading
load_one <- function(file, cond) {
  df <- read.csv(file.path(DATA_DIR, file), stringsAsFactors = FALSE,
                 fileEncoding = "latin1")
  names(df) <- trimws(names(df))
  ch <- sapply(df, is.character); df[ch] <- lapply(df[ch], trimws)
  if ("Crops" %in% names(df) && !("Crop" %in% names(df)))
    df <- rename(df, Crop = Crops)
  df$Condition <- cond
  df$study_id  <- as.integer(df$number)
  df
}

dn <- load_one("normal_data.csv", "Non-stress")
ds <- load_one("stress_data.csv", "Stress")
df_all <- bind_rows(dn, ds)
df_all <- df_all[!is.na(df_all$lnRR) & df_all$vi > 0, ]
df_all <- df_all[!is.na(df_all$Crop), ]

cat(sprintf("Total observations: %d\n", nrow(df_all)))
cat(sprintf("Studies: %d, Crops: %d\n",
            length(unique(df_all$study_id)),
            length(unique(df_all$Crop))))

# ─────────────────────────────────────────────────────────────────────────
# Section A: 8 subgrouppooled effect(paper Fig.1, Table S3）
# using rma.mv() with study + crop random effects + REML
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== A. 8 subgroup pooled effects (rma.mv + REML + study+crop) ===\n")

results <- data.frame()
indicators <- c("Growth", "Photosynthetic Pigment", "Biomass", "Yield")

for (cond in c("Non-stress", "Stress")) {
  for (perf in indicators) {
    sub <- df_all[df_all$Condition == cond & df_all$Performance == perf, ]
    if (nrow(sub) < 5) next

    # main pool: rma.mv multilevel
    fit_mv <- tryCatch(
      rma.mv(yi = lnRR, V = vi,
             random = list(~ 1 | study_id, ~ 1 | Crop),
             method = "REML", data = sub, sparse = TRUE),
      error = function(e) NULL)
    if (is.null(fit_mv)) next

    # PI: single-level REML + predict(paper L259：τ̂² from DL estimator）
    # strictly per paper: single-level DL gives τ²; metafor does not provide a mixed "REML + DL τ²" form
    # so we use rma() + method="DL" here to give PI (fully consistent with paper L259)
    fit_dl <- rma(yi = sub$lnRR, vi = sub$vi, method = "DL")
    pi <- predict(fit_dl, transf = function(x) (exp(x) - 1) * 100)

    # Egger（per the paper L271：bysubgroup）
    eg <- regtest(fit_dl, model = "lm", predictor = "sei")

    results <- rbind(results, data.frame(
      Condition   = cond,
      Performance = perf,
      k           = fit_mv$k,
      pct_mv_REML = round((exp(coef(fit_mv)) - 1) * 100, 1),
      CI_lo_mv    = round((exp(fit_mv$ci.lb) - 1) * 100, 1),
      CI_hi_mv    = round((exp(fit_mv$ci.ub) - 1) * 100, 1),
      pct_dl      = round((exp(coef(fit_dl)) - 1) * 100, 1),
      CI_lo_dl    = round((exp(fit_dl$ci.lb) - 1) * 100, 1),
      CI_hi_dl    = round((exp(fit_dl$ci.ub) - 1) * 100, 1),
      PI_lo       = round(pi$pi.lb, 1),
      PI_hi       = round(pi$pi.ub, 1),
      I2          = round(fit_dl$I2, 1),
      tau2_DL     = round(fit_dl$tau2, 4),
      egger_int   = round(eg$est, 3),
      egger_z     = round(eg$zval, 3),
      egger_p     = signif(eg$pval, 3)
    ))

    cat(sprintf("  %-12s %-25s k=%3d | rma.mv REML: %+.1f%% [%+.1f, %+.1f] | DL: %+.1f%% [%+.1f, %+.1f] | PI: [%+.1f, %+.1f] | I²=%.1f%% | Egger int=%.3f p=%.2e\n",
                cond, perf, fit_mv$k,
                (exp(coef(fit_mv)) - 1) * 100,
                (exp(fit_mv$ci.lb) - 1) * 100,
                (exp(fit_mv$ci.ub) - 1) * 100,
                (exp(coef(fit_dl)) - 1) * 100,
                (exp(fit_dl$ci.lb) - 1) * 100,
                (exp(fit_dl$ci.ub) - 1) * 100,
                pi$pi.lb, pi$pi.ub,
                fit_dl$I2, eg$est, eg$pval))
  }
}

write.csv(results, file.path(OUT_DIR, "Table_S3_subgroup_meta.csv"), row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────
# Section B: Stress vs Non-stress yieldcompare(paper L67 reports 22.5% vs 17.6%）
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== B. Stress vs Non-stress yield contrast ===\n")

sub_yield <- df_all[df_all$Performance == "Yield", ]
fit_yield <- rma.mv(yi = lnRR, V = vi,
                    mods = ~ Condition,
                    random = list(~ 1 | study_id, ~ 1 | Crop),
                    method = "REML", data = sub_yield, sparse = TRUE)
print(summary(fit_yield))

# ─────────────────────────────────────────────────────────────────────────
# Section C: LOSO(paper L82：6/8 subgroups span < 7 pp）
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== C. LOSO subgroup sensitivity ===\n")

loso_results <- data.frame()
for (cond in c("Non-stress", "Stress")) {
  for (perf in indicators) {
    sub <- df_all[df_all$Condition == cond & df_all$Performance == perf, ]
    if (nrow(sub) < 5) next
    studies <- unique(sub$study_id)
    pcts <- c()
    for (s in studies) {
      rest <- sub[sub$study_id != s, ]
      if (nrow(rest) < 2) next
      fit <- tryCatch(rma(yi = rest$lnRR, vi = rest$vi, method = "DL"),
                      error = function(e) NULL)
      if (is.null(fit)) next
      pcts <- c(pcts, (exp(coef(fit)) - 1) * 100)
    }
    fit_full <- rma(yi = sub$lnRR, vi = sub$vi, method = "DL")
    pct_full <- (exp(coef(fit_full)) - 1) * 100
    loso_results <- rbind(loso_results, data.frame(
      Condition = cond, Performance = perf,
      k_studies = length(pcts),
      full_pct  = round(pct_full, 1),
      min_pct   = round(min(pcts), 1),
      max_pct   = round(max(pcts), 1),
      span_pp   = round(max(pcts) - min(pcts), 1)
    ))
    cat(sprintf("  %-12s %-25s span=%.1f pp (range %.1f–%.1f)\n",
                cond, perf, max(pcts) - min(pcts), min(pcts), max(pcts)))
  }
}
write.csv(loso_results, file.path(OUT_DIR, "LOSO_subgroup.csv"), row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────
# Section D: Trim-and-fill(paper L78：claim "no studies imputed"）
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== D. Trim-and-fill (DL, all 8 subgroups) ===\n")

tf_results <- data.frame()
for (cond in c("Non-stress", "Stress")) {
  for (perf in indicators) {
    sub <- df_all[df_all$Condition == cond & df_all$Performance == perf, ]
    if (nrow(sub) < 5) next
    fit <- rma(yi = sub$lnRR, vi = sub$vi, method = "DL")
    tf  <- trimfill(fit)
    tf_results <- rbind(tf_results, data.frame(
      Condition = cond, Performance = perf,
      k         = fit$k,
      orig_pct  = round((exp(coef(fit)) - 1) * 100, 1),
      adj_pct   = round((exp(coef(tf)) - 1) * 100, 1),
      k_imputed = tf$k0
    ))
    cat(sprintf("  %-12s %-25s orig=%.1f%% → adj=%.1f%% (k_imputed=%d)\n",
                cond, perf,
                (exp(coef(fit)) - 1) * 100,
                (exp(coef(tf)) - 1) * 100,
                tf$k0))
  }
}
write.csv(tf_results, file.path(OUT_DIR, "Trim_and_fill_subgroup.csv"), row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────
# Section E: fail-safe N Nfs(paper L271：reports 269,068）
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== E. Fail-safe N (Rosenthal) ===\n")

# use fsn() with the new signature (positional yi, vi)
nfs <- fsn(df_all$lnRR, df_all$vi)
cat(sprintf("Total k = %d, Rosenthal Nfs = %d (threshold 5k+10 = %d)\n",
            nrow(df_all), nfs$fsnum, 5 * nrow(df_all) + 10))

# ─────────────────────────────────────────────────────────────────────────
# Section F: Pot vs Field yield subgroup(paper L137）
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== F. Pot vs Field yield subgroup ===\n")

pf_results <- data.frame()
for (cond in c("Non-stress", "Stress")) {
  for (exp_type in c("Pot", "Field")) {
    sub <- df_all[df_all$Condition == cond &
                  df_all$Performance == "Yield" &
                  df_all$experiment == exp_type, ]
    if (nrow(sub) < 5) next
    fit <- rma(yi = sub$lnRR, vi = sub$vi, method = "DL")
    pf_results <- rbind(pf_results, data.frame(
      Condition = cond, Experiment = exp_type,
      k        = fit$k,
      pct      = round((exp(coef(fit)) - 1) * 100, 1),
      CI_lo    = round((exp(fit$ci.lb) - 1) * 100, 1),
      CI_hi    = round((exp(fit$ci.ub) - 1) * 100, 1)
    ))
    cat(sprintf("  %-12s %-6s k=%3d: %+.1f%% [%+.1f, %+.1f]\n",
                cond, exp_type, fit$k,
                (exp(coef(fit)) - 1) * 100,
                (exp(fit$ci.lb) - 1) * 100,
                (exp(fit$ci.ub) - 1) * 100))
  }
}
write.csv(pf_results, file.path(OUT_DIR, "Pot_vs_Field_yield.csv"), row.names = FALSE)

# ─────────────────────────────────────────────────────────────────────────
# Section G: Meta-regression (paper Fig.4 dose-response)
# Yield ~ Concentration + NPs_size, fit separately by condition
# exclude area-application units (paper L267): requires a unit column not provided in this dataset
# hereonlyper the paper Methods descriptionrunmodel；ifhas unit columnshouldpre-filter
# ─────────────────────────────────────────────────────────────────────────

cat("\n=== G. Meta-regression: Yield ~ Concentration + NPs_size ===\n")

for (cond in c("Non-stress", "Stress")) {
  sub <- df_all[df_all$Condition == cond &
                df_all$Performance == "Yield" &
                !is.na(df_all$Concentration) &
                !is.na(df_all$NPs_size), ]
  if (nrow(sub) < 20) next
  fit <- rma.mv(yi = lnRR, V = vi,
                mods = ~ Concentration + NPs_size,
                random = list(~ 1 | study_id, ~ 1 | Crop),
                method = "REML", data = sub, sparse = TRUE)
  cat(sprintf("\n--- %s (n = %d) ---\n", cond, nrow(sub)))
  print(coef(summary(fit)))
}

cat("\n\n=== ALL R OUTPUTS SAVED TO ", OUT_DIR, " ===\n")
