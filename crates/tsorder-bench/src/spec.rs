//! TOML design schema and estimator construction.

use serde::Deserialize;
use tsorder_core::{
    ensemble_order, fh_order, ic_order_with, polylog_memory, Criterion, EnsembleRule, Error,
    FhConfig, IcMethod, Statistic, Threshold,
};

#[derive(Deserialize)]
pub struct Design {
    pub run: Run,
    pub dgp: Vec<crate::dgp::Dgp>,
    pub estimator: Vec<Est>,
}

#[derive(Deserialize)]
pub struct Run {
    pub name: String,
    pub out: String,
    pub n: Vec<usize>,
    pub replications: usize,
    /// Replications for `n >= large_from` (default: `replications`).
    pub replications_large: Option<usize>,
    pub large_from: Option<usize>,
    /// Also record one-step-ahead prediction errors.
    pub predict: Option<bool>,
    /// Timing mode: median per-series wall time instead of selections.
    pub timing: Option<bool>,
    pub timing_series: Option<usize>,
}

/// One estimator. `kind` ∈ {"ic", "fh", "ensemble"}; the other fields apply
/// depending on the kind (serde keeps unknown-field detection per design file).
#[derive(Deserialize)]
pub struct Est {
    pub name: String,
    pub kind: String,
    // ic
    pub criterion: Option<String>,
    pub method: Option<String>,
    pub max_order: Option<usize>,
    // fh
    pub statistic: Option<String>,
    pub threshold: Option<String>,
    /// Threshold constant `c` (bic/lil/maxk/fixed) or level `alpha` (gumbel).
    pub c: Option<f64>,
    /// For gumbel: fixed level, or "inv_n" / "inv_sqrt_n" for α_n → 0.
    pub alpha: Option<toml::Value>,
    /// "n03" | "n04" | "sqrt2" | "log2" | "log4" | integer (as string) | absent (statistic default).
    pub memory: Option<String>,
    // ensemble
    pub statistics: Option<Vec<String>>,
    pub rule: Option<String>,
}

fn criterion(name: &str) -> Criterion {
    match name {
        "aic" => Criterion::Aic,
        "aicc" => Criterion::Aicc,
        "bic" => Criterion::Bic,
        "hq" => Criterion::Hq,
        "fpe" => Criterion::Fpe,
        "cat" => Criterion::Cat,
        "mdl" => Criterion::Mdl,
        other => panic!("unknown criterion {other}"),
    }
}

fn method(name: &str) -> IcMethod {
    match name {
        "yw" => IcMethod::YuleWalker,
        "ols" => IcMethod::Ols,
        "burg" => IcMethod::Burg,
        other => panic!("unknown method {other}"),
    }
}

fn statistic(name: &str) -> Statistic {
    match name {
        "pacf" => Statistic::Pacf,
        "long_ar_yw" => Statistic::LongArYw,
        "long_ar_ols" => Statistic::LongArOls,
        "acf" => Statistic::Acf,
        "acf_bartlett" => Statistic::AcfBartlett,
        "inverse_pacf" => Statistic::InversePacf,
        "jirak_band" => Statistic::JirakBand,
        "pacf_robust" => Statistic::PacfRobust,
        other => panic!("unknown statistic {other}"),
    }
}

impl Est {
    fn threshold(&self, n: usize) -> Threshold {
        let c = self.c.unwrap_or(1.5);
        match self.threshold.as_deref().unwrap_or("bic") {
            "bic" => Threshold::BicLike { c },
            "lil" => Threshold::Lil { c },
            "maxk" => Threshold::MaxOfK { c },
            "fixed" => Threshold::Fixed(c),
            "gumbel" => {
                let alpha = match &self.alpha {
                    Some(toml::Value::Float(a)) => *a,
                    Some(toml::Value::String(s)) if s == "inv_n" => 1.0 / n as f64,
                    Some(toml::Value::String(s)) if s == "inv_sqrt_n" => 1.0 / (n as f64).sqrt(),
                    None => 0.05,
                    other => panic!("bad alpha {other:?}"),
                };
                Threshold::Gumbel { alpha }
            }
            other => panic!("unknown threshold {other}"),
        }
    }

    fn memory(&self, n: usize, stat: Statistic) -> Option<usize> {
        let m = match self.memory.as_deref()? {
            "n03" => ((n as f64).powf(0.3).floor() as usize).max(2),
            "n04" => tsorder_core::power_memory(n),
            "sqrt2" => ((2.0 * (n as f64).sqrt()).floor() as usize).min(n / 4),
            "log2" => polylog_memory(n, 2.0),
            "log4" => polylog_memory(n, 4.0),
            "default" => stat.default_memory(n),
            s => s.parse().unwrap_or_else(|_| panic!("bad memory {s}")),
        };
        Some(m)
    }

    fn fh_config(&self, n: usize, stat: Statistic) -> FhConfig {
        FhConfig {
            statistic: stat,
            threshold: self.threshold(n),
            memory: self.memory(n, stat),
        }
    }

    /// Run the estimator on `x`; `n` is the nominal sample size used to
    /// resolve n-dependent settings (α_n, memory).
    pub fn select(&self, x: &[f64], n: usize) -> Result<usize, Error> {
        match self.kind.as_str() {
            "ic" => {
                let c = criterion(self.criterion.as_deref().expect("ic needs criterion"));
                let m = method(self.method.as_deref().unwrap_or("yw"));
                Ok(ic_order_with(x, c, m, self.max_order)?.order)
            }
            "fh" => {
                let stat = statistic(self.statistic.as_deref().expect("fh needs statistic"));
                Ok(fh_order(x, self.fh_config(n, stat))?.order)
            }
            "ensemble" => {
                let cfgs: Vec<FhConfig> = self
                    .statistics
                    .as_ref()
                    .expect("ensemble needs statistics")
                    .iter()
                    .map(|s| self.fh_config(n, statistic(s)))
                    .collect();
                let rule = match self.rule.as_deref().unwrap_or("majority") {
                    "majority" => EnsembleRule::Majority,
                    "median" => EnsembleRule::Median,
                    "min" => EnsembleRule::Min,
                    "max" => EnsembleRule::Max,
                    other => panic!("unknown rule {other}"),
                };
                Ok(ensemble_order(x, &cfgs, rule)?.order)
            }
            other => panic!("unknown estimator kind {other}"),
        }
    }
}
