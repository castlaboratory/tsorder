//! Order of integration `d` by sequential unit-root testing with a vanishing
//! level: difference until the test stops indicating a unit root, with
//! `α_n → 0` so the number of differences is estimated consistently.
//!
//! Two tests are offered. KPSS (default; null = stationarity): under a true
//! stationary level the rejection probability tends to `α_n → 0`, under a
//! unit root the statistic diverges, so `d̂ → d` either way. ADF (null =
//! unit root): the type-I error vanishes and the power tends to one, same
//! conclusion from the other side. Fractional `d` is out of scope.

use crate::criteria::{ic_order, Criterion};
use crate::dist::{df_tau_c_critical, kpss_critical};
use crate::linalg::ols;
use crate::{check_input, Error, Result};

/// Which unit-root test drives the sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnitRootTest {
    /// Kwiatkowski–Phillips–Schmidt–Shin level statistic with Bartlett/
    /// Newey–West long-run variance, bandwidth `⌊4 (T/100)^{1/4}⌋`.
    Kpss,
    /// Augmented Dickey–Fuller with constant; augmentation lag chosen by BIC
    /// on the differenced series (the "lag by tsorder" rule).
    Adf,
}

/// Configuration of [`integration_order`].
#[derive(Debug, Clone, Copy)]
pub struct IntegrationConfig {
    /// Largest `d` considered (default 2).
    pub max_d: usize,
    /// Level; `None` → `0.05·log 50 / log n` (the same vanishing-level rule
    /// as the rank tests; satisfies Hosoya's consistency condition).
    pub alpha: Option<f64>,
    pub test: UnitRootTest,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        IntegrationConfig {
            max_d: 2,
            alpha: None,
            test: UnitRootTest::Kpss,
        }
    }
}

/// Output of [`integration_order`].
#[derive(Debug, Clone)]
pub struct IntegrationResult {
    /// Estimated order of integration (`max_d` if every stage still indicates
    /// a unit root — inspect `stats` in that case).
    pub d: usize,
    /// Test statistic at each stage `0..=d` actually run.
    pub stats: Vec<f64>,
    /// Critical value used at each stage.
    pub crits: Vec<f64>,
    pub alpha: f64,
    pub test: UnitRootTest,
}

/// Estimate the order of integration by sequential testing with a vanishing
/// level. The ARMA order selectors in this crate assume stationarity; run
/// this first and difference `d` times.
pub fn integration_order(x: &[f64], config: IntegrationConfig) -> Result<IntegrationResult> {
    let n = x.len();
    check_input(x, 20 + 4 * config.max_d)?;
    let alpha = config
        .alpha
        .unwrap_or(0.05 * 50f64.ln() / (n as f64).ln().max(1.0));
    let mut series: Vec<f64> = x.to_vec();
    let mut stats = Vec::new();
    let mut crits = Vec::new();
    let mut d = config.max_d;
    for stage in 0..=config.max_d {
        let (stat, crit, unit_root) = match config.test {
            UnitRootTest::Kpss => {
                let s = kpss_stat(&series)?;
                let c = kpss_critical(alpha);
                (s, c, s > c) // rejecting stationarity = unit root indicated
            }
            UnitRootTest::Adf => {
                let s = adf_stat(&series)?;
                let c = df_tau_c_critical(alpha);
                (s, c, s >= c) // failing to reject the unit root
            }
        };
        stats.push(stat);
        crits.push(crit);
        if !unit_root {
            d = stage;
            break;
        }
        if stage < config.max_d {
            series = series.windows(2).map(|w| w[1] - w[0]).collect();
        }
    }
    Ok(IntegrationResult {
        d,
        stats,
        crits,
        alpha,
        test: config.test,
    })
}

/// KPSS level statistic: `T^{-2} Σ S_t² / σ̂²_lr`, Bartlett long-run variance.
fn kpss_stat(x: &[f64]) -> Result<f64> {
    let t = x.len();
    let mean = x.iter().sum::<f64>() / t as f64;
    let e: Vec<f64> = x.iter().map(|v| v - mean).collect();
    let mut s = 0.0;
    let mut sum_s2 = 0.0;
    for v in &e {
        s += v;
        sum_s2 += s * s;
    }
    let bandwidth = (4.0 * (t as f64 / 100.0).powf(0.25)).floor() as usize;
    let mut lr = e.iter().map(|v| v * v).sum::<f64>() / t as f64;
    for j in 1..=bandwidth.min(t - 1) {
        let w = 1.0 - j as f64 / (bandwidth as f64 + 1.0);
        let g = (0..t - j).map(|i| e[i] * e[i + j]).sum::<f64>() / t as f64;
        lr += 2.0 * w * g;
    }
    if lr <= 0.0 {
        return Err(Error::Singular);
    }
    Ok(sum_s2 / (t as f64 * t as f64 * lr))
}

/// ADF t-statistic (with constant): `Δx_t = α + ρ x_{t−1} + Σ δ_j Δx_{t−j}`,
/// `k` lags by BIC on the differenced series.
fn adf_stat(x: &[f64]) -> Result<f64> {
    let t = x.len();
    let dx: Vec<f64> = x.windows(2).map(|w| w[1] - w[0]).collect();
    let k = ic_order(&dx, Criterion::Bic, None)?.order;
    let start = k + 1; // Δx index offset
    let rows = dx.len() - start;
    let cols = 2 + k;
    if rows <= cols + 2 {
        return Err(Error::TooShort {
            n: t,
            needed: cols + start + 3,
        });
    }
    let mut design = vec![0.0; rows * cols];
    let mut y = vec![0.0; rows];
    for r in 0..rows {
        let ti = start + r; // index into dx; level index = ti (x[ti] is the lagged level)
        y[r] = dx[ti];
        design[r * cols] = 1.0;
        design[r * cols + 1] = x[ti];
        for j in 0..k {
            design[r * cols + 2 + j] = dx[ti - j - 1];
        }
    }
    let (beta, _s2, se) = ols(&design, rows, cols, &y)?;
    Ok(beta[1] / se[1])
}
