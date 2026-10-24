//! Fine–Hwang threshold estimators with pluggable statistics, plus an ensemble.
//!
//! Every statistic produces, for lags `1..=p_n`, a value and a null standard
//! error. The estimator clips `|stat_k| / se_k ≤ τ_n` to zero and returns the
//! last surviving lag ("last non-zero coefficient", Fine & Hwang 1979, Def. 3).

use crate::acov::{acf, autocovariance, bartlett_se};
use crate::levinson::levinson;
use crate::linalg::ols;
use crate::threshold::{polylog_memory, Memory, Threshold};
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
    /// autoregressive estimator of Cleveland 1972; the statistic itself is
    /// Bhansali 1983, *J. Multivariate Anal.* 13 — this is exactly his
    /// estimator). An MA(q) process has inverse PACF zero beyond `q` (his
    /// Theorem 3.1), and his Theorem 4.2 gives the exact published null:
    /// beyond `q` the estimates are asymptotically iid `N(0, 1/n)` and
    /// independent of the earlier lags, so `se = 1/sqrt(n)` is the exact
    /// studentisation (conditions: iid innovations with finite fourth moment,
    /// invertibility, and a long-AR window growing at least like a multiple
    /// of `log n` but with `p_n³/n → 0` — the defaults qualify).
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
    /// **Experimental.** PACF computed from the Q_n-based robust
    /// autocorrelations of Ma & Genton (2000) (the bounded ratio form, whose
    /// shared denominator cancels the additive-outlier bias), with the PSD
    /// shrinking repair as a safety net ([`crate::robust`]). 25 % breakdown
    /// point against additive outliers, where the ordinary PACF has none.
    ///
    /// Unlike the classical PACF this statistic is *not* self-studentized:
    /// its null dispersion grows with the serial dependence of the process
    /// (measured by simulation: ≈1.1/√n on white noise, ≈2.0/√n on a strongly
    /// correlated AR(2), ≈2.4/√n on an MA(2)). `se = 2.0/sqrt(n)` is a
    /// conservative empirical compromise pending exact theory, and the default
    /// memory is the short poly-log window (per-lag estimation errors are
    /// independent across lags, so long memories break positive definiteness).
    PacfRobust,
    /// **Experimental.** Robust autocorrelation (the Ma & Genton 2000 bounded
    /// Q_n-ratio form, [`crate::robust::robust_autocorrelation`]) studentized
    /// by Bartlett's formula computed from the robust ACF itself, inflated by
    /// 1.15. Monte Carlo calibration (white noise, MA(1), MA(2), AR(2);
    /// n = 500–8000; 400 reps) puts the null sd of
    /// `ρ̂_rob(k) / se_Bartlett(ρ̂_rob)` at 1.05–1.12 across all designs, so
    /// unlike [`Statistic::PacfRobust`] the plug-in pivot is nearly
    /// process-independent and 1.15 is a mild safety margin, pending exact
    /// theory (the asymptotic variance of the ratio estimator differs from
    /// the classical ACF's; Lévy-Leduc et al. 2011 give it for Gaussian
    /// long-memory processes). Targets MA order with a 25 % breakdown point
    /// against additive outliers, where [`Statistic::AcfBartlett`] has none.
    AcfRobust,
}

impl Statistic {
    pub const ALL: [Statistic; 9] = [
        Statistic::Pacf,
        Statistic::LongArYw,
        Statistic::LongArOls,
        Statistic::Acf,
        Statistic::AcfBartlett,
        Statistic::InversePacf,
        Statistic::JirakBand,
        Statistic::PacfRobust,
        Statistic::AcfRobust,
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
            Statistic::PacfRobust => "pacf_robust",
            Statistic::AcfRobust => "acf_robust",
        }
    }

    /// Default growing memory `p_n` used when [`FhConfig::memory`] is `None`:
    /// the poly-log window `⌈2 log n⌉` ([`polylog_memory`]), which dominated
    /// the power rules in the benchmark (it reaches a seasonal AR(12) from
    /// `n = 400`, where `⌊n^0.4⌋` cannot) and is the window covered by the
    /// uniform LIL bounds for autocovariances. [`Statistic::PacfRobust`] uses
    /// the shorter `⌈log n⌉` (its per-lag estimation errors are nearly
    /// independent across lags, so long memories drift out of positive
    /// definiteness and the deep PACF path turns into boundary noise).
    pub fn default_memory(self, n: usize) -> usize {
        match self {
            Statistic::PacfRobust => polylog_memory(n, 1.0),
            _ => polylog_memory(n, 2.0),
        }
    }

    /// Which model family's order this statistic targets.
    pub fn target(self) -> &'static str {
        match self {
            Statistic::Pacf
            | Statistic::LongArYw
            | Statistic::LongArOls
            | Statistic::JirakBand
            | Statistic::PacfRobust => "AR",
            Statistic::Acf
            | Statistic::AcfBartlett
            | Statistic::InversePacf
            | Statistic::AcfRobust => "MA",
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
                // Demean first: the regression has no intercept, so without this
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
            Statistic::PacfRobust => {
                let r = crate::robust::robust_autocorrelation(x, p_n)?;
                let r = crate::robust::shrink_to_psd(&r)?;
                let l = levinson(&r)?;
                // Monte Carlo null calibration pending exact theory (see the
                // enum documentation).
                let se = vec![2.0 / (n as f64).sqrt(); p_n];
                Ok((l.pacf, se))
            }
            Statistic::AcfRobust => {
                let r = crate::robust::robust_autocorrelation(x, p_n)?;
                // Bartlett plug-in from the robust ACF itself, with the
                // Monte Carlo safety margin (see the enum documentation).
                let se: Vec<f64> = bartlett_se(&r, n)[1..].iter().map(|s| 1.15 * s).collect();
                Ok((r[1..].to_vec(), se))
            }
        }
    }
}

/// Configuration of a single Fine–Hwang estimator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FhConfig {
    pub statistic: Statistic,
    pub threshold: Threshold,
    /// Growing memory rule; [`Memory::Default`] uses the statistic's own
    /// default ([`Statistic::default_memory`]).
    pub memory: Memory,
}

impl Default for FhConfig {
    fn default() -> Self {
        FhConfig {
            statistic: Statistic::Pacf,
            threshold: Threshold::default(),
            memory: Memory::Default,
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
    let p_n = config.memory.resolve(n, config.statistic.default_memory(n));
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

/// Configuration of [`jirak_order`] (Jirak 2012, q̂⁽⁵⁾).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct JirakConfig {
    /// Level of the Gumbel band. A fixed `alpha` controls the family-wise
    /// overestimation probability at ≈ `alpha`; `alpha_n → 0` slowly gives the
    /// consistent variant (Jirak 2012, Thm 2.4).
    pub alpha: f64,
    /// Growing memory rule for the maximal dimension `d_n`.
    pub memory: Memory,
}

impl Default for JirakConfig {
    fn default() -> Self {
        JirakConfig {
            alpha: 0.05,
            memory: Memory::Default,
        }
    }
}

/// Output of [`jirak_order`].
#[derive(Debug, Clone)]
pub struct JirakResult {
    pub config: JirakConfig,
    /// q̂⁽⁵⁾: the maximum of the per-dimension estimates.
    pub order: usize,
    /// Maximal fitting dimension `d_n` actually used.
    pub p_n: usize,
    /// Gumbel threshold actually used (studentized scale, normalized by `d_n`).
    pub tau: f64,
    /// Per-dimension estimate q̂(k) for `k = 1..=d_n`: the last lag of the
    /// Yule–Walker AR(`k`) fit whose studentized coefficient exceeds `tau`.
    pub per_dimension: Vec<usize>,
}

/// Jirak's (2012, *Ann. Statist.* 40(2), eq. 2.14) refitted order estimator
/// q̂⁽⁵⁾: fit Yule–Walker AR(`k`) for every dimension `k ≤ d_n`, studentize
/// each coefficient vector with its exact null standard error
/// `se_i = sqrt(γ̂*_ii(k) · σ̂²(k) / n)` (the [`Statistic::JirakBand`]
/// studentization at dimension `k`), apply the common Gumbel threshold
/// calibrated on the maximal dimension `d_n`, and return the **largest**
/// per-dimension last-crossing.
///
/// The refits recover power that the single full-dimension fit
/// ([`Statistic::JirakBand`] inside [`fh_order`]) can lose: a coefficient
/// that is small in the AR(`d_n`) representation may be large in a smaller
/// one, and under H₀ the extra comparisons are already paid for by the
/// `d_n`-normalized threshold. Costs `O(d_n⁴)` — negligible for the
/// poly-log default (`d_n = ⌈2 log n⌉`).
pub fn jirak_order(x: &[f64], config: JirakConfig) -> Result<JirakResult> {
    let n = x.len();
    if !(config.alpha.is_finite() && config.alpha > 0.0 && config.alpha < 1.0) {
        return Err(crate::Error::Invalid("alpha must be in (0, 1)"));
    }
    let p_n = config
        .memory
        .resolve(n, Statistic::JirakBand.default_memory(n));
    if p_n < 1 {
        return Err(crate::Error::Invalid("memory p_n must be >= 1"));
    }
    check_input(x, 2 * p_n + 2)?;
    let r = autocovariance(x, p_n)?;
    let tau = Threshold::Gumbel {
        alpha: config.alpha,
    }
    .value(n, p_n);
    let nf = n as f64;
    let mut per_dimension = Vec::with_capacity(p_n);
    for k in 1..=p_n {
        let l = levinson(&r[..=k])?;
        let mut g = crate::linalg::SquareMatrix::zeros(k);
        for i in 0..k {
            for j in 0..k {
                g.set(i, j, r[i.abs_diff(j)]);
            }
        }
        let ch = g.cholesky()?;
        let diag = crate::linalg::SquareMatrix::cholesky_inverse_diag(&ch);
        let sigma2 = l.sigma2[k];
        let q_k = l
            .phi_last
            .iter()
            .zip(&diag)
            .rposition(|(phi, d)| phi.abs() / (d * sigma2 / nf).sqrt() > tau)
            .map(|i| i + 1)
            .unwrap_or(0);
        per_dimension.push(q_k);
    }
    let order = per_dimension.iter().copied().max().unwrap_or(0);
    Ok(JirakResult {
        config,
        order,
        p_n,
        tau,
        per_dimension,
    })
}

#[cfg(feature = "ensemble")]
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

#[cfg(feature = "ensemble")]
#[derive(Debug, Clone)]
pub struct EnsembleResult {
    pub rule: EnsembleRule,
    pub order: usize,
    pub members: Vec<FhResult>,
}

#[cfg(feature = "ensemble")]
/// Run several Fine–Hwang configurations and combine them.
///
/// ⚠️ Removed from the default public API: neither this ensemble nor a
/// genuine mode-by-consensus beat their best member anywhere in the benchmark
/// (the mode can fall *below its worst voter* under a parsimonious
/// tie-break). Kept behind the `ensemble` feature so the benchmark figures
/// remain reproducible.
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
