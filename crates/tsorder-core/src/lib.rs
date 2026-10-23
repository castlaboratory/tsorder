//! # tsorder-core
//!
//! Dependency-free core for time-series order selection.
//!
//! Two families of estimators live here, deliberately sharing one data path
//! (sample autocovariance → Levinson–Durbin recursion):
//!
//! * **Classical information criteria** ([`ic_order`]): AIC, AICc, BIC, HQ, FPE,
//!   evaluated over nested AR fits of order `0..=max_order`.
//! * **Fine–Hwang threshold estimators** ([`fh_order`]): fit a model whose memory
//!   `p_n` grows with `n`, "clip" every statistic below a vanishing threshold
//!   `η_n`, and report the index of the last surviving lag. Several pluggable
//!   statistics are supported ([`Statistic`]) as well as an ensemble rule
//!   ([`ensemble_order`]).
//!
//! The mathematical background, the equivalences between the two families and
//! the empirical findings that motivated the design are summarised on the
//! project page: <https://castlaboratory.github.io/tsorder/>.
//!
//! All functions take `&[f64]`, never allocate more than `O(n + K^2)`, and are
//! `no_std`-friendly in spirit (they use `Vec` but nothing else from `std`).

pub mod acov;
pub mod cancorr;
pub mod criteria;
pub mod levinson;
pub mod linalg;
pub mod robust;
pub mod select;
#[cfg(feature = "sim")]
pub mod sim;
pub mod threshold;

pub use acov::{acf, autocovariance};
pub use cancorr::{
    canonical_correlations, canonical_correlations_from_acov, canonical_correlations_stacked,
    svc_order, Cancorr, SvcConfig, SvcPenalty, SvcResult,
};
pub use criteria::{ic_order, ic_order_with, Criterion, IcMethod, IcResult};
pub use levinson::{burg_from_series, levinson, levinson_from_series, Levinson};
pub use robust::{qn, robust_autocorrelation, robust_autocovariance, shrink_to_psd};
pub use select::{
    ensemble_order, fh_order, EnsembleResult, EnsembleRule, FhConfig, FhResult, Statistic,
};
pub use threshold::{polylog_memory, power_memory, Threshold};

/// Library error type.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Series too short for the requested computation.
    TooShort { n: usize, needed: usize },
    /// Non-finite value found in the input.
    NonFinite,
    /// Sample autocovariance matrix is numerically singular.
    Singular,
    /// Invalid parameter.
    Invalid(&'static str),
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TooShort { n, needed } => {
                write!(
                    f,
                    "series of length {n} is too short (need at least {needed})"
                )
            }
            Error::NonFinite => write!(f, "input contains NaN or infinite values"),
            Error::Singular => write!(f, "sample covariance matrix is singular"),
            Error::Invalid(m) => write!(f, "invalid argument: {m}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = core::result::Result<T, Error>;

pub(crate) fn check_input(x: &[f64], needed: usize) -> Result<()> {
    if x.len() < needed {
        return Err(Error::TooShort { n: x.len(), needed });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite);
    }
    Ok(())
}
