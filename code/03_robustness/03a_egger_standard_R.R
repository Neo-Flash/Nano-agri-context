# Fig2b_egger_standard_R.R
# ---------------------------------------------------------------------------
# Use R + metafor::regtest() to perform the standard Egger funnel-plot asymmetry test
# Output: 8 subgroups (4 indicators × 2 conditions): intercept, SE, z, P
# reconciliation：and main_en.tex Fig 2b numberconsistent
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(metafor)
})

DATA_DIR <- file.path(getwd(), "data")
OUT_FILE <- "results/egger_standard_R_results.csv"

# loaddata
dn <- read.csv(file.path(DATA_DIR, "normal_data.csv"),
               stringsAsFactors = FALSE, fileEncoding = "latin1")
ds <- read.csv(file.path(DATA_DIR, "stress_data.csv"),
               stringsAsFactors = FALSE, fileEncoding = "latin1")
names(dn) <- trimws(names(dn)); names(ds) <- trimws(names(ds))

# keep only common columns
common <- intersect(names(dn), names(ds))
dn <- dn[, common]; ds <- ds[, common]

dn$Condition <- "Non-stress"
ds$Condition <- "Stress"
all <- rbind(dn, ds)
ch <- sapply(all, is.character); all[ch] <- lapply(all[ch], trimws)
all <- all[!is.na(all$lnRR) & all$vi > 0, ]

indicators <- c("Growth", "Photosynthetic Pigment", "Biomass", "Yield")

# run 8 subgroup
results <- data.frame()
for (cond in c("Non-stress", "Stress")) {
  for (perf in indicators) {
    sub <- all[all$Condition == cond & all$Performance == perf, ]
    if (nrow(sub) < 5) next
    fit <- rma(yi = sub$lnRR, vi = sub$vi, method = "DL")
    eg  <- regtest(fit, model = "lm", predictor = "sei")
    intercept_val <- as.numeric(eg$fit$coefficients[1])
    se_val        <- summary(eg$fit)$coefficients[1, 2]
    results <- rbind(results, data.frame(
      Condition   = cond,
      Performance = perf,
      k           = fit$k,
      intercept   = round(intercept_val, 3),
      se          = round(se_val,  3),
      z_or_t      = round(eg$zval, 3),
      p_value     = signif(eg$pval, 3),
      significant_at_001 = ifelse(eg$pval < 0.001, "YES", "no")
    ))
    cat(sprintf("  %-12s %-25s k=%3d  intercept=%+6.3f  z=%+6.2f  P=%.3e  %s\n",
                cond, perf, fit$k, intercept_val, eg$zval, eg$pval,
                ifelse(eg$pval < 0.001, "***", ifelse(eg$pval<0.05, "*", "n.s."))))
  }
}

cat("\n=== Summary ===\n")
cat(sprintf("Total subgroups: %d\n", nrow(results)))
cat(sprintf("P < 0.001: %d / %d\n",
            sum(results$p_value < 0.001), nrow(results)))
cat(sprintf("P < 0.05:  %d / %d\n",
            sum(results$p_value < 0.05),  nrow(results)))

write.csv(results, OUT_FILE, row.names = FALSE)
cat(sprintf("\nResults saved to: %s\n", OUT_FILE))
