# tsorder

[![ci](https://github.com/castlaboratory/tsorder/actions/workflows/ci.yml/badge.svg)](https://github.com/castlaboratory/tsorder/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/tsorder-core.svg)](https://crates.io/crates/tsorder-core)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![No dependencies](https://img.shields.io/badge/dependencies-none-brightgreen.svg)

Order selection for time-series models — classical information criteria and
Fine–Hwang threshold estimators with pluggable statistics — in a dependency-free
Rust core with R and Python bindings (bindings: planned).

**Site & API reference:** <https://castlaboratory.github.io/tsorder/>

```text
tsorder/
├── crates/tsorder-core/   Rust core (dependency-free)
├── bindings/r/            R package (extendr, pkgdown) — planned
├── bindings/python/       Python package (PyO3/maturin) — planned
├── benchmark/             benchmark runner and (future) results
└── site/                  landing page; rustdoc is published under /api by CI
```

## Idea in one paragraph

Fine & Hwang (1979) proved the first consistent estimator of system order: fit a
model whose memory `p_n` grows with `n`, zero every coefficient below a vanishing
threshold `η_n`, and report the last surviving lag. It needs no likelihood, no
Gaussianity and no maximum order. It was never studied empirically. We found that
the original threshold is not scale-invariant and degrades with `n`, that
studentizing fixes it, that the PACF is the natural self-studentized statistic for
AR (matching BIC, and beating it on subset AR in small samples), and that the same
recipe extends to MA (Bartlett-studentized ACF) and — in principle, calibration still
open — to ARMA via canonical correlations. For the AR case the rigorous theory
(Gumbel-calibrated bands over a growing Yule–Walker fit, consistent order selection)
is Jirak (2012, *Ann. Statist.*); `tsorder` builds on it rather than competing with
it. `tsorder` packages all of this next to AIC/AICc/BIC/HQ/FPE so the two families
can be compared on equal footing.

## Quick start (Rust)

```rust
use tsorder_core::*;

let x: Vec<f64> = /* your series */;

// Classical
let bic = ic_order(&x, Criterion::Bic, None)?;
println!("BIC order = {}", bic.order);

// Fine–Hwang with the recommended statistic (PACF) and threshold
let fh = fh_order(&x, FhConfig::default())?;
println!("FH order = {} (p_n = {}, tau = {:.2})", fh.order, fh.p_n, fh.tau);

// MA order via Bartlett-studentized ACF
let ma = fh_order(&x, FhConfig { statistic: Statistic::AcfBartlett, ..Default::default() })?;

// Ensemble
let e = ensemble_order(&x, &[
    FhConfig { statistic: Statistic::Pacf, ..Default::default() },
    FhConfig { statistic: Statistic::LongArOls, ..Default::default() },
], EnsembleRule::Majority)?;
```

## Where things stand (2026-10-23)

- Core: AIC/AICc/BIC/HQ/FPE/CAT/MDL over Yule–Walker, Burg or conditional-OLS fits
  (`IcMethod`); Fine–Hwang thresholding with `Pacf`, `LongArYw`, `LongArOls`, `Acf`,
  `AcfBartlett`, `InversePacf` (Cleveland's inverse ACF route), `JirakBand` (exact
  Jirak-2012 studentization) and the experimental `PacfRobust` (Ma–Genton Q_n
  autocorrelations, 25 % breakdown); thresholds `BicLike`, `Lil`, `MaxOfK`, `Gumbel`,
  `Fixed`; memory `⌈2 log n⌉` (the default, picked by the benchmark) or `⌊n^0.4⌋`;
  majority/median ensembles (scheduled for removal before 1.0); autocovariance by FFT
  for large lags; the Q_n scale estimator (O(n log n), matches R's `robustbase::Qn`);
  a `sim` feature with the benchmark's ARMA generator (t innovations, outliers,
  observation noise); McMillan-degree estimators: `svc_order` (Bauer's 2001
  SVC via past×future canonical correlations, his Table 1 reproduced as a regression
  test), `hankel_svt_order` (calibrated singular-value thresholding) and
  `hankel_rank_order` (Robin–Smith's rank test on the autocovariance Hankel, weighted-χ²
  null by exact Imhof inversion, vanishing level); ARMA (p, q) identification by `eacf`
  (Tsay–Tiao's table, matching R's `TSA::eacf` digit for digit, with an automated
  triangle-vertex reading) and `hannan_rissanen` (stages 1–2). Clippy-clean, no
  dependencies, MSRV 1.81.
- Theory positioning: AR thresholding is Jirak (2012); canonical correlations with growing
  window is Bauer (2001); Hankel rank tests are Robin & Smith (2000) and Camba-Méndez &
  Kapetanios (2001). `tsorder` implements all of them under one umbrella; the open questions
  it targets are MA, robustness and Bauer's calibration problem.
- Pre-alpha. The API will still change.

## License

MIT. Authors: André Leite, Raydonal Ospina, Cristiano Ferraz, Hugo Vasconcelos
and Diogo Bezerra (CASTLab/UFPE), with Claude. See `CITATION.cff` for how to cite.
