//! Levinson–Durbin recursion: Yule–Walker AR fits of every order `1..=K` in
//! `O(K²)`, yielding the PACF and the innovation variance of each order.
//!
//! This single recursion feeds *both* estimator families: the information
//! criteria need `sigma2[k]` for every `k`; the Fine–Hwang estimators need the
//! PACF (`reflection[k]`) or the long-AR coefficients (`phi_last`).

use crate::{check_input, Error, Result};

/// Output of [`levinson`].
#[derive(Debug, Clone)]
pub struct Levinson {
    /// `reflection[k-1]` = partial autocorrelation at lag `k`, `k = 1..=K`.
    pub pacf: Vec<f64>,
    /// `sigma2[k]` = Yule–Walker innovation variance of the AR(`k`) fit, `k = 0..=K`.
    pub sigma2: Vec<f64>,
    /// Coefficients `(φ_1, …, φ_K)` of the AR(`K`) fit (the "long AR").
    pub phi_last: Vec<f64>,
}

/// Run Levinson–Durbin on autocovariances `r[0..=K]`.
pub fn levinson(r: &[f64]) -> Result<Levinson> {
    if r.len() < 2 {
        return Err(Error::TooShort {
            n: r.len(),
            needed: 2,
        });
    }
    let k_max = r.len() - 1;
    let mut phi: Vec<f64> = Vec::with_capacity(k_max);
    let mut pacf = Vec::with_capacity(k_max);
    let mut sigma2 = Vec::with_capacity(k_max + 1);
    let mut sig = r[0];
    if sig <= 0.0 {
        return Err(Error::Singular);
    }
    sigma2.push(sig);
    for k in 1..=k_max {
        let mut num = r[k];
        for j in 1..k {
            num -= phi[j - 1] * r[k - j];
        }
        let kk = num / sig;
        if !kk.is_finite() {
            return Err(Error::Singular);
        }
        let prev = phi.clone();
        for j in 1..k {
            phi[j - 1] = prev[j - 1] - kk * prev[k - j - 1];
        }
        phi.push(kk);
        sig *= 1.0 - kk * kk;
        if sig <= 0.0 {
            // Perfectly predictable at this order; clamp so later orders stay finite.
            sig = f64::EPSILON * r[0];
        }
        pacf.push(kk);
        sigma2.push(sig);
    }
    Ok(Levinson {
        pacf,
        sigma2,
        phi_last: phi,
    })
}

/// Convenience: autocovariance + Levinson in one call.
pub fn levinson_from_series(x: &[f64], k_max: usize) -> Result<Levinson> {
    check_input(x, k_max + 2)?;
    let r = crate::acov::autocovariance(x, k_max)?;
    levinson(&r)
}

/// Burg's method (Burg 1967; Kay & Marple 1981): each reflection coefficient is
/// chosen to minimise the sum of forward and backward prediction error power,
/// so the fitted AR is always stationary and behaves better than Yule–Walker
/// near the unit circle. Returns the same shape as [`levinson`]; `sigma2[0]` is
/// the biased sample variance and `sigma2[k] = sigma2[k-1] (1 - κ_k²)`.
pub fn burg_from_series(x: &[f64], k_max: usize) -> Result<Levinson> {
    check_input(x, k_max + 2)?;
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut sig = xc.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if sig <= 0.0 {
        return Err(Error::Singular);
    }
    let r0 = sig;
    let mut f = xc.clone();
    let mut b = xc;
    let mut phi: Vec<f64> = Vec::with_capacity(k_max);
    let mut pacf = Vec::with_capacity(k_max);
    let mut sigma2 = Vec::with_capacity(k_max + 1);
    sigma2.push(sig);
    for k in 1..=k_max {
        let mut num = 0.0;
        let mut den = 0.0;
        for t in k..n {
            num += f[t] * b[t - 1];
            den += f[t] * f[t] + b[t - 1] * b[t - 1];
        }
        if den <= 0.0 || !den.is_finite() {
            return Err(Error::Singular);
        }
        let kk = 2.0 * num / den;
        // Descending so f[t] and b[t-1] on the right-hand side are still the
        // order-(k-1) errors when they are read.
        for t in (k..n).rev() {
            let fo = f[t];
            let bo = b[t - 1];
            f[t] = fo - kk * bo;
            b[t] = bo - kk * fo;
        }
        let prev = phi.clone();
        for j in 1..k {
            phi[j - 1] = prev[j - 1] - kk * prev[k - j - 1];
        }
        phi.push(kk);
        sig *= 1.0 - kk * kk;
        if sig <= 0.0 {
            // Perfectly predictable at this order; clamp so later orders stay finite.
            sig = f64::EPSILON * r0;
        }
        pacf.push(kk);
        sigma2.push(sig);
    }
    Ok(Levinson {
        pacf,
        sigma2,
        phi_last: phi,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ar1_exact_autocovariance_recovers_phi() {
        // γ(k) = φ^k / (1-φ²) for AR(1) with unit innovations.
        let phi: f64 = 0.6;
        let r: Vec<f64> = (0..6).map(|k| phi.powi(k) / (1.0 - phi * phi)).collect();
        let l = levinson(&r).unwrap();
        assert!((l.pacf[0] - phi).abs() < 1e-12);
        for k in 1..5 {
            assert!(l.pacf[k].abs() < 1e-10, "pacf[{k}] = {}", l.pacf[k]);
        }
        assert!((l.sigma2[1] - 1.0).abs() < 1e-10);
    }
}
