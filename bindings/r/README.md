# tsorder (R) — planned

Thin `extendr` wrapper over `tsorder-core`, tidyverse-friendly, with a pkgdown
site. Target API (to be confirmed):

```r
fit <- order_select(x, method = "pacf")      # or "bic", "aic", "hq", "acf_bartlett", "ensemble", ...
fit$order
autoplot(fit)                                 # statistic/criterion path with threshold line
as_tibble(fit)

fh_order(x, statistic = "pacf", threshold = "bic", c = 1.5, memory = NULL)
ic_order(x, criterion = "bic", max_order = NULL)
ensemble_order(x, statistics = c("pacf", "long_ar_ols"), rule = "majority")
```

Setup: `rextendr::use_extendr()`, vendor Rust deps for CRAN (`src/rust/vendor`), `SystemRequirements: Cargo (Rust's package manager), rustc`.
