//! Hannan–Rissanen (1982) recursive ARMA order selection and initial
//! coefficient estimation (their stages 1–2; the efficient third stage is
//! left to the bindings/estimation layer).
//!
//! Stage 1 fits a long Yule–Walker autoregression (order chosen by AIC, their
//! practical recommendation) and extracts innovation estimates ε̂. Stage 2
//! regresses `x_t` on `p` lags of `x` and `q` lags of `ε̂` for every
//! candidate `(p, q)` on a common sample and minimises
//! `log σ̃²_{p,q} + (p + q) log T / T` (their BIC choice, eq. 1.11).
//!
//! Their 1983 Biometrika correction (70, p. 303) reports that Theorem 3's
//! proof is in error and the stage-2 selection as published can converge to
//! values *larger* than `(p₀, q₀)`, typically when the MA polynomial has
//! zeros near the unit circle; the fix they give — repeating stage 2 with the
//! residuals `ε̃(t)` of eq. (1.12) in place of ε̂ — restores convergence and
//! is applied here by default (`erratum_pass`). The common-factor degeneracy
//! makes the stage-2 design matrix singular exactly when `p > p₀` and
//! `q > q₀` — such candidates are skipped when the Cholesky fails, which is
//! itself the paper's monitoring advice.

use crate::criteria::{ic_order, Criterion};
use crate::levinson::levinson_from_series;
use crate::linalg::ols;
use crate::{check_input, Result};

/// Configuration of [`hannan_rissanen`].
#[derive(Debug, Clone, Copy)]
pub struct HrConfig {
    /// Candidate grid `0..=max_p × 0..=max_q` (defaults 5 × 5).
    pub max_p: usize,
    pub max_q: usize,
    /// Long-AR order of stage 1; `None` → AIC choice (clamped to ≥ 1).
    pub long_ar: Option<usize>,
    /// Repeat stage 2 with the eq. (1.12) residuals of the first-pass model
    /// (the 1983 erratum's consistency fix). Default true.
    pub erratum_pass: bool,
}

impl Default for HrConfig {
    fn default() -> Self {
        HrConfig {
            max_p: 5,
            max_q: 5,
            long_ar: None,
            erratum_pass: true,
        }
    }
}

/// Output of [`hannan_rissanen`].
#[derive(Debug, Clone)]
pub struct HrResult {
    pub p: usize,
    pub q: usize,
    /// AR coefficients in the `x_t = Σ φ_j x_{t−j} + ε_t + Σ θ_j ε_{t−j}` convention.
    pub phi: Vec<f64>,
    pub theta: Vec<f64>,
    /// Stage-2 residual variance at the selected order.
    pub sigma2: f64,
    /// Long-AR order used in stage 1.
    pub long_ar: usize,
}

/// Hannan–Rissanen stages 1–2: select `(p, q)` and return the (consistent,
/// not efficient) stage-2 coefficient estimates.
pub fn hannan_rissanen(x: &[f64], config: HrConfig) -> Result<HrResult> {
    let n = x.len();
    let h = match config.long_ar {
        Some(h) => h.max(1),
        None => ic_order(x, Criterion::Aic, None)?.order.max(1),
    };
    check_input(x, 2 * (h + config.max_p.max(config.max_q)) + 2)?;
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    // Stage 1: Yule–Walker long AR, innovations ε̂(t) for t = h..n.
    let l = levinson_from_series(&xc, h)?;
    let eps: Vec<f64> = (h..n)
        .map(|t| {
            let mut e = xc[t];
            for (j, a) in l.phi_last.iter().enumerate() {
                e -= a * xc[t - j - 1];
            }
            e
        })
        .collect();
    // Stage 2 on the common sample t = m..n with m = h + max(max_p, max_q):
    // eps index offset: ε̂(t) lives at eps[t - h].
    let m = h + config.max_p.max(config.max_q);
    let rows = n - m;
    if rows < 10 * (config.max_p + config.max_q + 1) {
        return Err(crate::Error::TooShort {
            n,
            needed: m + 10 * (config.max_p + config.max_q + 1),
        });
    }
    let (p, q, beta, sigma2) = stage2(&xc, &eps, h, m, config)?;
    let (p, q, beta, sigma2) = if config.erratum_pass {
        // 1983 erratum fix: recompute residuals through the eq. (1.12)
        // recursion of the first-pass model and repeat stage 2.
        let (phi, theta) = (&beta[..p], &beta[p..]);
        let start = p.max(q);
        let mut eps2 = vec![0.0; n];
        for t in start..n {
            let mut e = xc[t];
            for (j, ph) in phi.iter().enumerate() {
                e -= ph * xc[t - j - 1];
            }
            for (j, th) in theta.iter().enumerate() {
                e -= th * eps2[t - j - 1];
            }
            eps2[t] = e;
        }
        stage2(&xc, &eps2, 0, m, config)?
    } else {
        (p, q, beta, sigma2)
    };
    Ok(HrResult {
        p,
        q,
        phi: beta[..p].to_vec(),
        theta: beta[p..].to_vec(),
        sigma2,
        long_ar: h,
    })
}

/// The stage-2 grid: regress `xc[t]` on `p` lags of `xc` and `q` lags of the
/// innovation proxy (`eps[t - offset]` holds ε(t)), over the common sample
/// `t = m..n`, and minimise `log σ̃² + (p+q) log T / T`.
fn stage2(
    xc: &[f64],
    eps: &[f64],
    offset: usize,
    m: usize,
    config: HrConfig,
) -> Result<(usize, usize, Vec<f64>, f64)> {
    let n = xc.len();
    let rows = n - m;
    let tn = rows as f64;
    let mut best: Option<(f64, usize, usize, Vec<f64>, f64)> = None;
    for p in 0..=config.max_p {
        for q in 0..=config.max_q {
            let cols = p + q;
            let (beta, sigma2) = if cols == 0 {
                let s2 = (0..rows).map(|t| xc[m + t].powi(2)).sum::<f64>() / tn;
                (Vec::new(), s2)
            } else {
                let mut design = vec![0.0; rows * cols];
                let mut y = vec![0.0; rows];
                for t in 0..rows {
                    y[t] = xc[m + t];
                    for j in 0..p {
                        design[t * cols + j] = xc[m + t - j - 1];
                    }
                    for j in 0..q {
                        design[t * cols + p + j] = eps[m + t - j - 1 - offset];
                    }
                }
                match ols(&design, rows, cols, &y) {
                    Ok((beta, _s2, _se)) => {
                        // σ̃² with the paper's 1/T normalisation on the common sample.
                        let mut rss = 0.0;
                        for t in 0..rows {
                            let fit: f64 = (0..cols).map(|c| design[t * cols + c] * beta[c]).sum();
                            rss += (y[t] - fit).powi(2);
                        }
                        (beta, rss / tn)
                    }
                    // Common-factor singularity (p > p0 and q > q0): skip.
                    Err(_) => continue,
                }
            };
            let crit = sigma2.max(1e-300).ln() + (p + q) as f64 * tn.ln() / tn;
            if best.as_ref().map_or(true, |(b, ..)| crit < *b) {
                best = Some((crit, p, q, beta, sigma2));
            }
        }
    }
    let (_, p, q, beta, sigma2) = best.ok_or(crate::Error::Singular)?;
    Ok((p, q, beta, sigma2))
}
