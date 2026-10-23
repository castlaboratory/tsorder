//! Sample autocovariance / autocorrelation.

use crate::{check_input, Result};

/// Biased (divide-by-`n`) sample autocovariance for lags `0..=max_lag`,
/// after mean removal. The biased estimator keeps the Toeplitz matrix
/// positive semi-definite, which Levinson–Durbin requires.
///
/// Complexity `O(n · max_lag)`. (An FFT path is planned for `max_lag` large.)
pub fn autocovariance(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    check_input(x, max_lag + 2)?;
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut r = Vec::with_capacity(max_lag + 1);
    for k in 0..=max_lag {
        let mut s = 0.0;
        for t in 0..n - k {
            s += xc[t] * xc[t + k];
        }
        r.push(s / n as f64);
    }
    Ok(r)
}

/// Sample autocorrelation for lags `0..=max_lag` (`acf[0] == 1`).
pub fn acf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let r = autocovariance(x, max_lag)?;
    let r0 = r[0];
    if r0 <= 0.0 {
        return Err(crate::Error::Singular);
    }
    Ok(r.iter().map(|v| v / r0).collect())
}

/// Bartlett large-sample standard error of `acf[k]` under the hypothesis that
/// the process is MA(k−1) (i.e. true autocorrelations vanish from lag `k` on):
/// `sqrt((1 + 2 Σ_{j<k} ρ_j²) / n)`. Used to studentize the ACF so that the
/// Fine–Hwang threshold is scale-free for MA order selection.
pub fn bartlett_se(acf: &[f64], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(acf.len());
    let mut cum = 0.0;
    for (k, &rk) in acf.iter().enumerate() {
        if k == 0 {
            out.push(0.0);
        } else {
            out.push(((1.0 + 2.0 * cum) / n as f64).sqrt());
        }
        if k >= 1 {
            cum += rk * rk;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_noise_acf_small() {
        let x: Vec<f64> = (0..2000)
            .map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5)
            .collect();
        let r = acf(&x, 5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bartlett_lag1_is_one_over_sqrt_n() {
        let se = bartlett_se(&[1.0, 0.5, 0.2], 400);
        assert!((se[1] - 0.05).abs() < 1e-12);
        assert!(se[2] > se[1]);
    }
}
