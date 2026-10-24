# tsorder 0.1.0

* First release: information criteria (AIC, AICc, BIC, HQ, FPE, CAT, MDL
  over Yule-Walker, Burg or conditional-OLS fits), Fine-Hwang threshold
  estimators with studentised statistics (PACF, Bartlett ACF, inverse PACF,
  Jirak band, robust PACF), McMillan-degree estimators (Bauer's SVC,
  Hankel singular-value thresholding, sequential rank test), ARMA
  identification (EACF, Hannan-Rissanen with the 1983 correction pass),
  integration order by sequential unit-root testing, and the Q_n scale
  estimator - all in a dependency-free Rust core, with the high-level
  `order_select()` interface.
* Defaults chosen by the reproducible benchmark in the repository
  (poly-log memory; SVC window d = 1).
* Test suite cross-validates against `stats::ar.yw` (1e-10),
  `robustbase::Qn` (1e-12) and `TSA::eacf` (digit for digit), and replays
  a golden fixture shared with the Python binding.
