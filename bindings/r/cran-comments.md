This is a new submission (first release of tsorder).

This resubmission addresses the two points raised in the manual
inspection of 2026-11-08:

* every acronym in the Description field is now spelled out on first use
  (AIC, AICc, BIC, HQ, FPE, CAT, MDL, PACF, ACF, SVC, ARMA);
* the only \dontrun{} example (TSORDER(), the fabletools model
  definition) is now unwrapped: it runs in well under a second and is
  guarded by requireNamespace() on the suggested packages it needs
  (fabletools, tsibble, distributional). No example remains wrapped in
  \dontrun{} or \donttest{}.

## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE "Possibly misspelled words in DESCRIPTION" (incoming checks only;
  the local check is clean): the remaining flagged words are acronyms of
  model-selection criteria and statistics, each spelled out on first use
  (AICc, FPE, MDL, PACF, ACF, SVC, ARMA), and surnames of the authors of
  the implemented methods (Hankel, Hannan, Hwang, Jirak, Parzen, Rissanen).

## Test environments

* local macOS (R 4.6.0), GitHub Actions ubuntu-latest and macos-latest.

## Rust

The package follows the CRAN policy for Rust: SystemRequirements declares
Cargo and rustc (>= 1.81); the tsorder-core crate is vendored inside the
package (src/rust) and all remaining crate dependencies are vendored in
src/rust/vendor.tar.xz (rextendr), so the build needs no network access;
compilation uses at most two jobs.
