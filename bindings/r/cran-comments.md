This is a new submission (first release of tsorder).

This resubmission addresses the two points raised in the manual
inspection:

* every acronym in the Description field is now spelled out on first use
  (AIC, AICc, BIC, HQ, FPE, CAT, MDL, PACF, ACF, SVC, ARMA);
* the only \dontrun{} example (TSORDER(), the fabletools model
  definition) is now unwrapped: it runs in well under a second and is
  guarded by requireNamespace() on the suggested packages it needs
  (fabletools, tsibble, distributional). No example remains wrapped in
  \dontrun{} or \donttest{}.

## R CMD check results

0 errors | 0 warnings | 0 notes (local, --as-cran)

The incoming checks are expected to add two NOTEs:

* new submission;
* "Possibly misspelled words in DESCRIPTION": AICc, FPE, MDL, PACF and SVC
  are acronyms spelled out on first use; AOS is part of a DOI; Hannan,
  Jirak, Parzen and Rissanen are surnames of the authors of the implemented
  methods; Hankel names the Hankel matrix.

## Test environments

* local macOS (R 4.6.0).

## Rust

The package follows the CRAN policy for Rust: SystemRequirements declares
Cargo and rustc (>= 1.81); the tsorder-core crate is vendored inside the
package (src/rust) and all remaining crate dependencies are vendored in
src/rust/vendor.tar.xz (rextendr), so the build needs no network access;
compilation uses at most two jobs.
