"""
Script for paper：
  - Methods §"Assessment of Publication Bias and Robustness"
  - Table S2: Fail-safe N (Rosenthal's Nfs)
  - Paper reports Nfs = 269,068, threshold = 5k+10 = 5,545

Tool: R + metafor::fsn() (Rosenthal method)

run：Rscript Nfs_failsafe_R.R
"""

# ── R code ─────────────────────────────────────────────────
R_SCRIPT = '''
suppressPackageStartupMessages(library(metafor))

dn <- read.csv("/Users/flash/Desktop/project/normal data.csv",
               stringsAsFactors = FALSE, fileEncoding = "latin1")
ds <- read.csv("/Users/flash/Desktop/project/stress data.csv",
               stringsAsFactors = FALSE, fileEncoding = "latin1")
names(dn) <- trimws(names(dn))
names(ds) <- trimws(names(ds))

# poolallhaseffectiveobservation
all <- rbind(dn[, c("lnRR", "vi", "Performance")],
             ds[, c("lnRR", "vi", "Performance")])
all <- all[!is.na(all$lnRR) & all$vi > 0, ]

# Rosenthal Nfs
nfs <- fsn(all$lnRR, all$vi)
cat(sprintf("Total k = %d\\n", nrow(all)))
cat(sprintf("Rosenthal Nfs = %d\\n", nfs$fsnum))
cat(sprintf("Threshold 5k+10 = %d\\n", 5 * nrow(all) + 10))
cat(sprintf("Nfs > Threshold? %s\\n",
            ifelse(nfs$fsnum > 5*nrow(all)+10, "YES (robust)", "NO")))
'''

# run (save as .R, then execute with Rscript)
if __name__ == "__main__":
    print(R_SCRIPT)
    print("\n# saved as Nfs.R afterrun：Rscript Nfs.R\n")
