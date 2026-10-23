//! Fine–Hwang threshold estimators with pluggable statistics, plus an ensemble.
//!
//! Every statistic produces, for lags `1..=p_n`, a value and a null standard
//! error. The estimator clips `|stat_k| / se_k ≤ τ_n` to zero and returns the
//! last surviving lag ("last non-zero coefficient", Fine & Hwang 1979, Def. 3).

use crate::acov::{acf, autocovariance, bartlett_se};
use crate::levinson::levinson;
use crate::linalg::ols;
use crate::threshold::{default_memory, Threshold};
use crate::{check_input, Result};

/// Which partial description Σ̂ of the process to threshold.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistic {
    /// Partial autocorrelation (AR order). Self-studentized: `se = 1/sqrt(n)` for lags
    /// beyond the true order, whatever the process (Quenouille 1949). **Recommended for AR.**
    Pacf,
    /// Coefficients of the Yule–Walker long AR(`p_n`) with `se = 1/sqrt(n)` — the literal
    /// Fine–Hwang 1979 statistic. Scale-dependent; kept for benchmarking.
    LongArYw,
    /// Coefficients of the OLS long AR(`p_n`) with classical standard errors. Scale-free.
    LongArOls,
    /// Autocorrelation with `se = 1/sqrt(n)` (literal Fine–Hwang MA statistic, §V-D). Scale-dependent.
    Acf,
    /// Autocorrelation studentized by Bartlett's formula (MA order). **Recommended for MA.**
    AcfBartlett,
    /// Inverse PACF (MA order): PACF of the inverse process, computed from the
    /// inverse autocovariances of the long Yule–Walker AR(`p_n`) fit (the
    /// autoregressive estimator of Cleveland 1972). By spectral duality the
    /// inverse of an MA(q) is an AR(q), so the inverse PACF cuts off at `q`.
    /// `se = 1/sqrt(n)` (white-noise asymptotics; the general null theory is
    /// Bhansali 1983).
    InversePacf,
    /// Yule–Walker long-AR coefficients with the exact null studentization of
    /// Jirak (2012, Thm 2.2): `se_i = sqrt(γ̂*_ii · σ̂²(p_n) / n)` where `γ̂*_ii`
    /// is the diagonal of the inverse of the estimated autocovariance matrix
    /// `Γ̂_{p_n}` and `σ̂²(p_n)` the Yule–Walker innovation variance, so each
    /// `values[i]/se[i]` is asymptotically standard normal under the null.
    /// Paired with [`Threshold::Gumbel`] this is Jirak's practically
    /// recommended band (his Remark 2.6 calibrates the threshold by the max of
    /// `p_n` i.i.d. |N(0,1)|, which is exactly that rule); with `alpha_n → 0`
    /// it is his consistent estimator q̂⁽¹⁾. Upgrades [`Statistic::LongArYw`],
    /// whose `1/sqrt(n)` ignores the correlation structure.
    JirakBand,
}

impl Statistic {
    pub const ALL: [Statistic; 7] = [
        Statistic::Pacf,
        Statistic::LongArYw,
        Statistic::LongArOls,
        Statistic::Acf,
        Statistic::AcfBartlett,
        Statistic::InversePacf,
        Statistic::JirakBand,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Statistic::Pacf => "pacf",
            Statistic::LongArYw => "long_ar_yw",
            Statistic::LongArOls => "long_ar_ols",
            Statistic::Acf => "acf",
            Statistic::AcfBartlett => "acf_bartlett",
            Statistic::InversePacf => "inverse_pacf",
            Statistic::JirakBand => "jirak_band",
        }
    }

    /// Which model family's order this statistic targets.
    pub fn target(self) -> &'static str {
        match self {
            Statistic::Pacf | Statistic::LongArYw | Statistic::LongArOls | Statistic::JirakBand => {
                "AR"
            }
            Statistic::Acf | Statistic::AcfBartlett | Statistic::InversePacf => "MA",
        }
    }

    /// Compute `(values, se)` for lags `1..=p_n`.
    pub fn compute(self, x: &[f64], p_n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = x.len();
        let root_n_se = vec![1.0 / (n as f64).sqrt(); p_n];
        match self {
            Statistic::Pacf => {
                let r = autocovariance(x, p_n)?;
                let l = levinson(&r)?;
                Ok((l.pacf, root_n_se))
            }
            Statistic::LongArYw => {
                let r = autocovariance(x, p_n)?;
                let l = levinson(&r)?;
                Ok((l.phi_last, root_n_se))
            }
            Statistic::LongArOls => {
                // Demean first (D-15): the regression has no intercept, so without this
                // step a level shift (x + 50) leaks into every lag coefficient. The
                // autocovariance-based statistics demean inside `autocovariance`.
                let mean = x.iter().sum::<f64>() / n as f64;
                let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
                let rows = n - p_n;
                let mut design = vec![0.0; rows * p_n];
                let mut y = vec![0.0; rows];
                for t in 0..rows {
                    y[t] = xc[p_n + t];
                    for j in 0..p_n {
                        design[t * p_n + j] = xc[p_n + t - j - 1];
                    }
                }
                let (beta, _s2, se) = ols(&design, rows, p_n, &y)?;
                Ok((beta, se))
            }
            Statistic::Acf => {
                let r = acf(x, p_n)?;
                Ok((r[1..].to_vec(), root_n_se))
            }
            Statistic::AcfBartlett => {
                let r = acf(x, p_n)?;
                let se = bartlett_se(&r, n);
                Ok((r[1..].to_vec(), se[1..].to_vec()))
            }
            Statistic::InversePacf => {
                // Long AR fit, its inverse autocovariances (an MA(p_n) sequence,
                // so PSD and safe for Levinson), then the PACF of that sequence.
                let r = autocovariance(x, p_n)?;
                let l = levinson(&r)?;
                let vi = crate::acov::inverse_acov_from_ar(&l.phi_last);
                let li = levinson(&vi)?;
                Ok((li.pacf, root_n_se))
            }
            Statistic::JirakBand => {
                let r = autocovariance(x, p_n)?;
                let l = levinson(&r)?;
                // Diagonal of the inverse of the Toeplitz Γ̂_{p_n}.
                let mut g = crate::linalg::SquareMatrix::zeros(p_n);
                for i in 0..p_n {
                    for j in 0..p_n {
                        g.set(i, j, r[i.abs_diff(j)]);
                    }
                }
                let ch = g.cholesky()?;
                let diag = crate::linalg::SquareMatrix::cholesky_inverse_diag(&ch);
                let sigma2 = l.sigma2[p_n];
                let nf = n as f64;
                let se = diag.iter().map(|d| (d * sigma2 / nf).sqrt()).collect();
                Ok((l.phi_last, se))
            }
        }
    }
}

/// Configuration of a single Fine–Hwang estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhConfig {
    pub statistic: Statistic,
    pub threshold: Threshold,
    /// Growing memory `p_n`; `None` → [`default_memory`].
    pub memory: Option<usize>,
}

impl Default for FhConfig {
    fn default() -> Self {
        FhConfig {
            statistic: Statistic::Pacf,
            threshold: Threshold::default(),
            memory: None,
        }
    }
}

/// Output of [`fh_order`].
#[derive(Debug, Clone)]
pub struct FhResult {
    pub config: FhConfig,
    /// Selected order: last lag whose studentized statistic exceeds the threshold (0 if none).
    pub order: usize,
    /// Memory actually used.
    pub p_n: usize,
    /// Threshold actually used (studentized scale).
    pub tau: f64,
    /// Raw statistic per lag `1..=p_n`.
    pub values: Vec<f64>,
    /// Studentized statistic per lag `1..=p_n`.
    pub studentized: Vec<f64>,
}

/// Fine–Hwang order estimator.
pub fn fh_order(x: &[f64], config: FhConfig) -> Result<FhResult> {
    let n = x.len();
    config.threshold.validate()?;
    let p_n = config.memory.unwrap_or_else(|| default_memory(n));
    if p_n < 1 {
        return Err(crate::Error::Invalid("memory p_n must be >= 1"));
    }
    check_input(x, 2 * p_n + 2)?;
    let (values, se) = config.statistic.compute(x, p_n)?;
    let tau = config.threshold.value(n, p_n);
    if !tau.is_finite() || tau < 0.0 {
        return Err(crate::Error::Invalid(
            "threshold evaluated to a non-finite or negative value",
        ));
    }
    let studentized: Vec<f64> = values.iter().zip(&se).map(|(v, s)| v / s).collect();
    let order = studentized
        .iter()
        .rposition(|t| t.abs() > tau)
        .map(|i| i + 1)
        .unwrap_or(0);
    Ok(FhResult {
        config,
        order,
        p_n,
        tau,
        values,
        studentized,
    })
}

/// How to combine several estimators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnsembleRule {
    /// Most frequent order; ties broken towards the smaller order (parsimony).
    Majority,
    /// Median order (lower median on even counts).
    Median,
    /// Smallest order (most conservative).
    Min,
    /// Largest order.
    Max,
}

#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub rule: EnsembleRule,
    pub order: usize,
    pub members: Vec<FhResult>,
}

/// Run several Fine–Hwang configurations and combine them.
///
/// Design note: the ensemble is *not* expected to beat the best single
/// statistic in the well-specified case; its purpose is insurance under misspecification
/// (contamination, unknown family). The benchmark will decide whether it stays.
pub fn ensemble_order(
    x: &[f64],
    configs: &[FhConfig],
    rule: EnsembleRule,
) -> Result<EnsembleResult> {
    if configs.is_empty() {
        return Err(crate::Error::Invalid("ensemble needs at least one config"));
    }
    let members: Vec<FhResult> = configs
        .iter()
        .map(|&c| fh_order(x, c))
        .collect::<Result<_>>()?;
    let mut orders: Vec<usize> = members.iter().map(|m| m.order).collect();
    orders.sort_unstable();
    let order = match rule {
        EnsembleRule::Min => orders[0],
        EnsembleRule::Max => orders[orders.len() - 1],
        EnsembleRule::Median => orders[(orders.len() - 1) / 2],
        EnsembleRule::Majority => {
            let mut best = orders[0];
            let mut best_count = 0;
            let mut i = 0;
            while i < orders.len() {
                let mut j = i;
                while j < orders.len() && orders[j] == orders[i] {
                    j += 1;
                }
                if j - i > best_count {
                    best_count = j - i;
                    best = orders[i];
                }
                i = j;
            }
            best
        }
    };
    Ok(EnsembleResult {
        rule,
        order,
        members,
    })
}
