# tsorder (R) <img src="man/figures/logo.svg" align="right" height="139" alt="tsorder hex logo" />

Thin [extendr](https://extendr.github.io/) shell over the dependency-free
Rust core `tsorder-core` (no statistical logic lives here — D-01). See the
[package site](https://castlaboratory.github.io/tsorder/r/) and the
[project page](https://castlaboratory.github.io/tsorder/).

```r
# from the repository checkout (needs Rust >= 1.81):
# install.packages("rextendr"); rextendr::document("bindings/r")
devtools::install("bindings/r")

library(tsorder)
x <- arima.sim(list(ar = c(1.2, -0.6)), 400)

fit <- order_select(x)          # PACF-clip, the default
fit$order
ggplot2::autoplot(fit)

ic_order(x, "bic")$order        # classical criteria
svc_order(x)$order              # McMillan degree (robust to observation noise)
integration_order(cumsum(x))$d  # how many differences
```

Cross-validated against R itself in the test suite: the Levinson PACF
matches `stats::ar.yw` to 1e-10, `qn()` matches `robustbase::Qn` to 1e-12,
and `eacf()` matches `TSA::eacf` digit for digit.
