# Changelog

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
