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
use crate::criteria::{ic_order, Criterion};
use crate::linalg::{singular_values, SquareMatrix};
use crate::{check_input, Error, Result};

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

/// Canonical correlations from the *stacked-sample* covariances, Bauer's
/// (2001) convention: averages of `Y⁺_{t,f} (Z⁻_{t,p})'` etc. over
/// `t = p+1 … T−f` (his index range, divisor `1/T` — it cancels in the
/// correlations). Differs from [`canonical_correlations`] only by edge
/// effects `O((f+p)/T)`; the sample joint second-moment matrix is an average
/// of outer products, hence PSD, so correlations stay in `[0, 1]`. The series
/// is demeaned first (the paper's model is mean-zero; demeaning keeps the
/// estimator translation-invariant).
pub fn canonical_correlations_stacked(x: &[f64], f: usize, p: usize) -> Result<Cancorr> {
    if f < 1 || p < 1 {
        return Err(Error::Invalid("window sizes f and p must be >= 1"));
    }
    check_input(x, 2 * (f + p))?;
    let t_len = x.len();
    if t_len < f + p + f + p {
        return Err(Error::TooShort {
            n: t_len,
            needed: 2 * (f + p),
        });
    }
    let mean = x.iter().sum::<f64>() / t_len as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut gff = SquareMatrix::zeros(f);
    let mut gpp = SquareMatrix::zeros(p);
    let mut gfp = vec![0.0; f * p];
    // Bauer's range: t = p+1 ..= T−f (1-based) → 0-based t = p .. T−f.
    for t in p..t_len - f {
        for i in 0..f {
            let yi = xc[t + i];
            for j in 0..=i {
                let v = gff.get(i, j) + yi * xc[t + j];
                gff.set(i, j, v);
                gff.set(j, i, v);
            }
            for j in 0..p {
                gfp[i * p + j] += yi * xc[t - 1 - j];
            }
        }
        for i in 0..p {
            let zi = xc[t - 1 - i];
            for j in 0..=i {
                let v = gpp.get(i, j) + zi * xc[t - 1 - j];
                gpp.set(i, j, v);
                gpp.set(j, i, v);
            }
        }
    }
    whitened_correlations(gff, gpp, gfp, f, p)
}

/// Cholesky-whitened singular values of `Γ_ff^{-1/2} Γ_fp Γ_pp^{-1/2}`
/// (equal to those of the symmetric-square-root version Bauer analyses).
fn whitened_correlations(
    gff: SquareMatrix,
    gpp: SquareMatrix,
    gfp: Vec<f64>,
    f: usize,
    p: usize,
) -> Result<Cancorr> {
    let lf = gff.cholesky()?;
    let lp = gpp.cholesky()?;
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
    for c in correlations.iter_mut() {
        if *c > 1.0 && *c < 1.0 + 1e-10 {
            *c = 1.0;
        }
    }
    Ok(Cancorr { correlations, f, p })
}

/// Penalty `C(T)` of the SVC criterion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvcPenalty {
    /// `C(T) = log T` — Bauer's practical recommendation; not covered by his
    /// consistency theorem when `f, p` grow (keeps a flat few-percent
    /// overestimation rate, Table 1).
    LogT,
    /// `C(T) = f·p·log T` — satisfies Theorem 3 (consistent), but strongly
    /// underestimates in finite samples (Table 1).
    FpLogT,
}

/// Configuration of [`svc_order`].
#[derive(Debug, Clone, Copy)]
pub struct SvcConfig {
    /// Window `f = p`; `None` → `⌊window_factor · p̂_AIC⌋` with `p̂_AIC` the
    /// AIC order of a long autoregression (Bauer §5), clamped to `≥ 2`.
    pub window: Option<usize>,
    /// Bauer's `d`; he recommends `1 ≤ d ≤ 2`, and the benchmark's Q4 round
    /// found `d = 1` dominating every DGP tried, so it is the default.
    pub window_factor: f64,
    pub penalty: SvcPenalty,
}

impl Default for SvcConfig {
    fn default() -> Self {
        SvcConfig {
            window: None,
            window_factor: 1.0,
            penalty: SvcPenalty::LogT,
        }
    }
}

/// Output of [`svc_order`].
#[derive(Debug, Clone)]
pub struct SvcResult {
    /// Estimated McMillan degree (state dimension).
    pub order: usize,
    /// Estimated canonical correlations, descending.
    pub correlations: Vec<f64>,
    /// `SVC(n)` for `n = 0..=min(f, p)`.
    pub values: Vec<f64>,
    pub f: usize,
    pub p: usize,
}

/// Bauer's (2001, eq. 4) SVC estimator of the McMillan degree, output-only
/// scalar case with CCA weighting: `SVC(n) = σ̂²_{n+1} + C(T)·2n/T`
/// (`d(n) = 2n` parameters for a scalar state-space system), minimised over
/// `n = 0..=M`, `M = min(f, p)`, with the convention `σ̂_{M+1} = 0`.
/// Consistency (his Theorem 3) needs `C(T)/(fp log log T) → ∞`.
pub fn svc_order(x: &[f64], config: SvcConfig) -> Result<SvcResult> {
    let t_len = x.len();
    let w = match config.window {
        Some(w) => w.max(1),
        None => {
            let aic = ic_order(x, Criterion::Aic, None)?;
            ((config.window_factor * aic.order as f64).floor() as usize).max(2)
        }
    };
    let cc = canonical_correlations_stacked(x, w, w)?;
    let m = w;
    let t = t_len as f64;
    let c_t = match config.penalty {
        SvcPenalty::LogT => t.ln(),
        SvcPenalty::FpLogT => (w * w) as f64 * t.ln(),
    };
    let values: Vec<f64> = (0..=m)
        .map(|k| {
            let sig2 = if k < m {
                cc.correlations[k].powi(2)
            } else {
                0.0
            };
            sig2 + c_t * (2 * k) as f64 / t
        })
        .collect();
    let order = crate::criteria::argmin(&values);
    Ok(SvcResult {
        order,
        correlations: cc.correlations,
        values,
        f: w,
        p: w,
    })
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
