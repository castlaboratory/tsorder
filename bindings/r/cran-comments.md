## R CMD check results

0 errors | 0 warnings | 1 note

* NOTE "Found non-API call to R: 'R_NamespaceRegistry'": this call is made
  by the extendr runtime (the Rust-R bridge this package is built on), not
  by package code; the extendr project tracks C API compliance upstream.

## Test environments

* local macOS (R 4.6.0), GitHub Actions ubuntu-latest and macos-latest.

## Rust

The package follows the CRAN policy for Rust: SystemRequirements declares
Cargo and rustc (>= 1.81); the tsorder-core crate is vendored inside the
package (src/rust) and the remaining crate dependencies are vendored into
src/rust/vendor.tar.xz with rextendr::vendor_pkgs() before submission, so
the build needs no network access; compilation uses at most two jobs.
