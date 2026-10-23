# Changelog

## 2026-10-23 — R package (M5, first half)
- `bindings/r`: extendr-based R package `tsorder` — two-level API
  (`order_select()` with print/autoplot over the low-level wrappers), the
  five authors, MIT. R CMD check: 0 errors, 0 warnings, 1 note (extendr's
  known non-API call). Cross-validated against R itself: Levinson PACF =
  `stats::ar.yw` (1e-10), `qn()` = `robustbase::Qn` (1e-12), `eacf()` =
  `TSA::eacf` digit for digit. pkgdown published under /r on the site;
  the Rust core is vendored into the package by `tools/sync-core.sh`
  (CI guards the copy).

## 2026-10-23 — benchmark round 1 (AR/MA)
- `tsorder-bench`: TOML-driven runner (rayon, fixed per-triple seeds, lossless
  aggregated CSV output), designs and R analysis for Q1-Q3, Q5-Q7.
- Headline results: poly-log memory `⌈2 log n⌉` dominates `n^0.4`/`n^0.3`
  (and is the window the available theory covers); ensembles never beat their
  best member on AR/MA; the robust PACF wins big under additive outliers at
  moderate n but its selection degrades for very large n (bias vs shrinking
  threshold); one-step prediction differences across selectors are ≤ 1 %.

## 2026-10-23 — repository
- Published at `castlaboratory/tsorder` with CI (fmt, clippy, tests on linux/macos/windows,
  docs, MSRV 1.81) and a GitHub Pages site (`site/` landing page + rustdoc under `/api`).
- Crate metadata completed: five authors, MIT license, MSRV 1.81, `CITATION.cff`.
- Research material (plan, decision log, theory notes, paper drafts, Python prototypes)
  moved to the private repository `castlaboratory/tsorder-research`.

## 2026-10-23 — day 1 (chat session with Claude)
- Read Hwang (1978), Fine & Hwang (1979), Kwakernaak (1980), Baram & Porat (1988), Bauer (2001).
- Exploratory Python simulations: raw FH threshold degrades with n; studentized
  FH ≈ BIC; PACF-clip ≥ BIC in small samples; raw Hankel SVD fails (scale, separation).
- Rust core `tsorder-core`: autocovariance, Levinson, AIC/AICc/BIC/HQ/FPE, `fh_order` with five
  statistics, thresholds (BicLike, Lil, MaxOfK, Gumbel, Fixed), memory rules, ensembles. 19 tests.
- Review fixes: OLS demeaning, threshold validation, AICc parameter count.
- Novelty searches: Jirak 2012; Robin & Smith 2000; Camba-Méndez & Kapetanios 2001/2008; Bauer 2001;
  García-Hiernaux et al. 2012; Zheng et al. 2025. Positioning and plan rewritten accordingly.
