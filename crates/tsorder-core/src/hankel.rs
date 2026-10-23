//! McMillan degree by singular-value thresholding of the Hankel matrix of
//! sample autocovariances (M3 route 2).
//!
//! The rank of the autocovariance Hankel `H[i][j] = γ(i+j+1)` is the state
//! dimension. A raw SVD cut fails because the threshold must carry the scale
//! of the estimation error (the day-1 finding); following the calibration
//! lesson of Zheng, Jedra & Proutière (2025, arXiv:2506.08720), the cut is
//!
//! `ξ_n = c · ŝ · sqrt(K · log n / n)`,
//!
//! where `ŝ² = 2 Σ_{|h| ≤ 2K} γ̂(h)²` is a Bartlett-style plug-in bound on the
//! per-entry error variance `n·Var(γ̂(h)) ≈ Σ_k γ(k)² + γ(k+h)γ(k−h)`, and
//! `sqrt(K)` is the random-matrix scaling of the operator norm of an
//! entrywise-noise perturbation (the worst-case factor `K` proved hopelessly
//! conservative in calibration: it underestimated everywhere). Recovery
//! additionally needs *separation*: the smallest non-zero singular value of
//! the true Hankel must exceed `ξ_n` — e.g. MA(2) with θ = (0.6, 0.3) has
//! true Hankel singular values ≈ (0.88, 0.10), and 0.10 sits below any
//! honest threshold at practical `n`; that is the problem being hard, not
//! the threshold failing.

use crate::acov::autocovariance;
use crate::linalg::singular_values;
use crate::threshold::polylog_memory;
use crate::Result;

/// Configuration of [`hankel_svt_order`].
#[derive(Debug, Clone, Copy)]
pub struct HankelSvtConfig {
    /// Hankel dimension `K`; `None` → `⌈2 log n⌉` ([`polylog_memory`]).
    pub k: Option<usize>,
    /// Threshold constant `c`. Empirical: at the default 1.0 the calibration
    /// runs gave 100 % correct rank 0 on white noise, ~100 % on Bauer's
    /// Example I, 96–98 % on AR(1) φ=0.9, ~90 % on ARMA(1,1) — and MA(2)
    /// fails on separation, as expected.
    pub c: f64,
}

impl Default for HankelSvtConfig {
    fn default() -> Self {
        HankelSvtConfig { k: None, c: 1.0 }
    }
}

/// Output of [`hankel_svt_order`].
#[derive(Debug, Clone)]
pub struct HankelSvtResult {
    /// Estimated McMillan degree: number of singular values above `xi`.
    pub order: usize,
    /// Singular values of the autocovariance Hankel, descending.
    pub singular_values: Vec<f64>,
    /// The threshold `ξ_n` actually used.
    pub xi: f64,
    pub k: usize,
}

/// Singular-value-thresholding estimator of the McMillan degree.
pub fn hankel_svt_order(x: &[f64], config: HankelSvtConfig) -> Result<HankelSvtResult> {
    let n = x.len();
    let k = config.k.unwrap_or_else(|| polylog_memory(n, 2.0)).max(2);
    let r = autocovariance(x, 2 * k)?;
    let mut h = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            h[i * k + j] = r[i + j + 1];
        }
    }
    let sv = singular_values(&h, k, k);
    let s2 = 2.0 * r.iter().map(|g| g * g).sum::<f64>();
    let xi = config.c * s2.sqrt() * ((k as f64) * (n as f64).ln() / n as f64).sqrt();
    let order = sv.iter().take_while(|s| **s > xi).count();
    Ok(HankelSvtResult {
        order,
        singular_values: sv,
        xi,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ar1_hankel_has_rank_one_structure() {
        // Exact AR(1) autocovariances: Hankel is rank 1, so σ_2 = 0 exactly.
        let phi: f64 = 0.8;
        let n = 10;
        let r: Vec<f64> = (0..=2 * n)
            .map(|h| phi.powi(h as i32) / (1.0 - phi * phi))
            .collect();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] = r[i + j + 1];
            }
        }
        let sv = singular_values(&h, n, n);
        assert!(sv[0] > 1.0);
        assert!(sv[1].abs() < 1e-10, "{sv:?}");
    }
}
