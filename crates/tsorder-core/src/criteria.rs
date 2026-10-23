//! Classical information criteria over nested AR fits.
//!
//! Innovation variances come from the Yule–Walker/Levinson recursion
//! (same convention as R's `ar.yw`), so every order is costed in one `O(K²)`
//! pass. An OLS-conditional variant is planned so that the
//! comparison with the OLS-based Fine–Hwang statistic is apples-to-apples.

use crate::levinson::levinson_from_series;
use crate::Result;

/// Penalised-likelihood criteria (all "smaller is better").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Criterion {
    /// Akaike (1973): `n log σ̂² + 2k`. Not consistent; prediction-efficient (Shibata 1980).
    Aic,
    /// Hurvich & Tsai (1989) small-sample correction: `AIC + 2K(K+1)/(n-K-1)` with
    /// `K = k + 1` free parameters (the `k` AR coefficients **plus the innovation variance**),
    /// the convention of `forecast::Arima` without constant (D-14). The series is demeaned
    /// before fitting but the mean is not counted (same as R's `ar()`).
    Aicc,
    /// Schwarz (1978): `n log σ̂² + k log n`. Strongly consistent (Hannan 1980).
    Bic,
    /// Hannan & Quinn (1979): `n log σ̂² + 2 c k log log n`, `c = 1` here. Minimal consistent penalty.
    Hq,
    /// Akaike (1969) final prediction error: `σ̂² (n+k)/(n-k)`. Not consistent.
    Fpe,
}

impl Criterion {
    pub const ALL: [Criterion; 5] = [
        Criterion::Aic,
        Criterion::Aicc,
        Criterion::Bic,
        Criterion::Hq,
        Criterion::Fpe,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Criterion::Aic => "AIC",
            Criterion::Aicc => "AICc",
            Criterion::Bic => "BIC",
            Criterion::Hq => "HQ",
            Criterion::Fpe => "FPE",
        }
    }

    /// Number of free parameters counted by the penalty for an AR(`k`) fit: the `k`
    /// coefficients plus the innovation variance. For AIC/BIC/HQ the variance term is a
    /// constant shift that does not affect the argmin, so they use `k` directly as in the
    /// textbook formulas; AICc's correction is nonlinear in `K`, so there it matters (D-14).
    pub fn n_params(k: usize) -> usize {
        k + 1
    }

    /// Criterion value for an AR(`k`) fit with innovation variance `sigma2` on `n` observations.
    pub fn value(self, sigma2: f64, k: usize, n: usize) -> f64 {
        let nf = n as f64;
        let kf = k as f64;
        let ll = nf * sigma2.ln();
        match self {
            Criterion::Aic => ll + 2.0 * kf,
            Criterion::Aicc => {
                let kp = Self::n_params(k) as f64;
                ll + 2.0 * kp + 2.0 * kp * (kp + 1.0) / (nf - kp - 1.0).max(1.0)
            }
            Criterion::Bic => ll + kf * nf.ln(),
            Criterion::Hq => ll + 2.0 * kf * nf.ln().ln().max(0.0),
            Criterion::Fpe => sigma2 * (nf + kf) / (nf - kf).max(1.0),
        }
    }
}

/// Result of [`ic_order`].
#[derive(Debug, Clone)]
pub struct IcResult {
    pub criterion: Criterion,
    /// Selected order (argmin of `values`).
    pub order: usize,
    /// Criterion value for each order `0..=max_order`.
    pub values: Vec<f64>,
    /// Innovation variance for each order `0..=max_order`.
    pub sigma2: Vec<f64>,
}

/// Default maximum order used by R's `ar()`: `min(n-1, 10 log10 n)`.
pub fn default_max_order(n: usize) -> usize {
    let m = (10.0 * (n as f64).log10()).floor() as usize;
    m.min(n.saturating_sub(1)).max(1)
}

/// Select an AR order by minimising `criterion` over `0..=max_order`.
pub fn ic_order(x: &[f64], criterion: Criterion, max_order: Option<usize>) -> Result<IcResult> {
    let n = x.len();
    let k_max = max_order.unwrap_or_else(|| default_max_order(n));
    let l = levinson_from_series(x, k_max)?;
    let values: Vec<f64> = l
        .sigma2
        .iter()
        .enumerate()
        .map(|(k, &s2)| criterion.value(s2, k, n))
        .collect();
    let order = argmin(&values);
    Ok(IcResult {
        criterion,
        order,
        values,
        sigma2: l.sigma2,
    })
}

/// All five criteria in one pass (shares the Levinson recursion).
pub fn ic_order_all(x: &[f64], max_order: Option<usize>) -> Result<Vec<IcResult>> {
    let n = x.len();
    let k_max = max_order.unwrap_or_else(|| default_max_order(n));
    let l = levinson_from_series(x, k_max)?;
    Ok(Criterion::ALL
        .iter()
        .map(|&c| {
            let values: Vec<f64> = l
                .sigma2
                .iter()
                .enumerate()
                .map(|(k, &s2)| c.value(s2, k, n))
                .collect();
            IcResult {
                criterion: c,
                order: argmin(&values),
                values,
                sigma2: l.sigma2.clone(),
            }
        })
        .collect())
}

pub(crate) fn argmin(v: &[f64]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x < v[best] {
            best = i;
        }
    }
    best
}
