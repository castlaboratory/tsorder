This is a new submission (first release of tsorder).

This resubmission fixes the URL flagged by the incoming pre-tests
(https://extendr.github.io/ in README.md replaced by its permanent
redirect target https://extendr.rs/).

## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE "Possibly misspelled words in DESCRIPTION": AICc, FPE and MDL are
  standard acronyms of model-selection criteria; Hankel, Hannan, Hwang and
  Jirak are surnames (of the authors of the implemented methods) and the
  Hankel matrix.

## Test environments

* local macOS (R 4.6.0), GitHub Actions ubuntu-latest and macos-latest.

## Rust

The package follows the CRAN policy for Rust: SystemRequirements declares
Cargo and rustc (>= 1.81); the tsorder-core crate is vendored inside the
package (src/rust) and all remaining crate dependencies are vendored in
src/rust/vendor.tar.xz (rextendr), so the build needs no network access;
compilation uses at most two jobs.
