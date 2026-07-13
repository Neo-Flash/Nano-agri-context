# R dependency installer for the meta-analysis pipeline.
# Run once:  Rscript R_setup.R

pkgs <- c("metafor", "dplyr", "tidyr")

installed <- rownames(installed.packages())
to_install <- setdiff(pkgs, installed)

if (length(to_install) > 0) {
  options(repos = c(CRAN = "https://cloud.r-project.org"))
  install.packages(to_install, quiet = TRUE)
}

for (p in pkgs) {
  ok <- requireNamespace(p, quietly = TRUE)
  cat(sprintf("%s : %s\n", p, ifelse(ok, "OK", "MISSING")))
}
