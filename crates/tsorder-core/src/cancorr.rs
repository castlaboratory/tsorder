//! Canonical correlations between the past and the future of a stationary
//! series — the weighted (CCA) subspace step of Bauer (2001). The number of
//! non-zero canonical correlations is the state dimension (McMillan degree),
//! which is what the SVC criterion and the Gumbel-thresholded variant read
//! off the estimated correlations.
//!
//! The blocks are built from the sample autocovariances: the joint vector
//! `(x_{t+f-1}, …, x_t, x_{t-1}, …, x_{t-p})` is contiguous, so its covariance
//! is the `(f+p) × (f+p)` Toeplitz matrix of `γ̂(0..f+p-1)`, which the biased
//! estimator keeps positive semi-definite — hence all estimated canonical
//! correlations are guaranteed to lie in `[0, 1]`.

use crate::acov::autocovariance;
use crate::linalg::{singular_values, SquareMatrix};
use crate::{Error, Result};

/// Output of [`canonical_correlations`].
#[derive(Debug, Clone)]
pub struct Cancorr {
    /// Estimated canonical correlations, descending, length `min(f, p)`.
    pub correlations: Vec<f64>,
    pub f: usize,
    pub p: usize,
}

/// Canonical correlations between `(x_t, …, x_{t+f-1})` and
/// `(x_{t-1}, …, x_{t-p})`, from the biased sample autocovariances.
pub fn canonical_correlations(x: &[f64], f: usize, p: usize) -> Result<Cancorr> {
    if f < 1 || p < 1 {
        return Err(Error::Invalid("window sizes f and p must be >= 1"));
    }
    let r = autocovariance(x, f + p - 1)?;
    canonical_correlations_from_acov(&r, f, p)
}

/// Same, from an autocovariance sequence `r[0..=f+p-1]` (exact inputs make
/// exact tests possible).
pub fn canonical_correlations_from_acov(r: &[f64], f: usize, p: usize) -> Result<Cancorr> {
    if r.len() < f + p {
        return Err(Error::TooShort {
            n: r.len(),
            needed: f + p,
        });
    }
    let toeplitz = |m: usize| -> SquareMatrix {
        let mut g = SquareMatrix::zeros(m);
        for i in 0..m {
            for j in 0..m {
                g.set(i, j, r[i.abs_diff(j)]);
            }
        }
        g
    };
    let lf = toeplitz(f).cholesky()?;
    let lp = toeplitz(p).cholesky()?;
    // Γ_fp[i][j] = Cov(x_{t+i}, x_{t-1-j}) = γ(i + j + 1), row-major f × p.
    let mut gfp = vec![0.0; f * p];
    for i in 0..f {
        for j in 0..p {
            gfp[i * p + j] = r[i + j + 1];
        }
    }
    // X = L_f⁻¹ Γ_fp (forward substitution on each column).
    let mut xm = vec![0.0; f * p];
    for j in 0..p {
        for i in 0..f {
            let mut s = gfp[i * p + j];
            for k in 0..i {
                s -= lf.get(i, k) * xm[k * p + j];
            }
            xm[i * p + j] = s / lf.get(i, i);
        }
    }
    // M = X L_p⁻ᵀ: solve L_p mᵀ-row = xᵀ-row for each row of X.
    let mut m = vec![0.0; f * p];
    for i in 0..f {
        for j in 0..p {
            let mut s = xm[i * p + j];
            for k in 0..j {
                s -= lp.get(j, k) * m[i * p + k];
            }
            m[i * p + j] = s / lp.get(j, j);
        }
    }
    let mut correlations = singular_values(&m, f, p);
    correlations.truncate(f.min(p));
    // Clamp tiny numerical excursions above 1.
    for c in correlations.iter_mut() {
        if *c > 1.0 && *c < 1.0 + 1e-10 {
            *c = 1.0;
        }
    }
    Ok(Cancorr { correlations, f, p })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_has_exactly_one_canonical_correlation_equal_to_phi() {
        // Exact autocovariances of AR(1): γ(k) = φ^k / (1 − φ²). The state is
        // one-dimensional, so exactly one non-zero canonical correlation, and
        // its value is φ (best predictor pair x_t ↔ x_{t-1}).
        let phi: f64 = 0.8;
        let r: Vec<f64> = (0..12).map(|k| phi.powi(k) / (1.0 - phi * phi)).collect();
        let cc = canonical_correlations_from_acov(&r, 6, 6).unwrap();
        assert!(
            (cc.correlations[0] - phi).abs() < 1e-10,
            "{:?}",
            cc.correlations
        );
        for c in &cc.correlations[1..] {
            assert!(c.abs() < 1e-8, "{:?}", cc.correlations);
        }
    }

    #[test]
    fn ma1_has_rank_one_past_future_covariance() {
        // MA(1): γ(0) = 1 + θ², γ(1) = θ, 0 beyond — state dimension 1.
        let theta = 0.6f64;
        let mut r = vec![0.0; 12];
        r[0] = 1.0 + theta * theta;
        r[1] = theta;
        let cc = canonical_correlations_from_acov(&r, 6, 6).unwrap();
        assert!(cc.correlations[0] > 0.3);
        for c in &cc.correlations[1..] {
            assert!(c.abs() < 1e-10, "{:?}", cc.correlations);
        }
    }

    #[test]
    fn white_noise_correlations_are_zero() {
        let mut r = vec![0.0; 10];
        r[0] = 2.5;
        let cc = canonical_correlations_from_acov(&r, 5, 5).unwrap();
        for c in &cc.correlations {
            assert!(c.abs() < 1e-12);
        }
    }
}
