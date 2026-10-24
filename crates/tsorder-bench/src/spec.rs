//! TOML design schema and estimator construction.

use serde::Deserialize;
use tsorder_core::{
    canonical_correlations_stacked, chi2_sf, eacf, ensemble_order, fh_order, hankel_rank_order,
    hankel_svt_order, hannan_rissanen, ic_order, ic_order_with, polylog_memory, Criterion,
    EacfConfig, EnsembleRule, Error, FhConfig, HankelRankConfig, HankelSvtConfig, HrConfig,
    IcMethod, RankAlpha, RankPvalue, Statistic, SvcConfig, SvcPenalty, Threshold,
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

/// One estimator. `kind` ∈ {"ic", "fh", "jirak5", "ensemble", "svc", "svt", "rank",
/// "cc_ic", "ba_seq", "mbc", "eacf", "hr"}; the other fields apply depending
/// on the kind. The ARMA identifiers ("eacf", "hr") return the encoded order
/// `10·p + q` (p, q ≤ 9); `eacf` with no clean vertex returns the failure
/// code −2 via an Err.
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
    // svc / cc_ic / ba_seq: window factor d (f = p = floor(d * p_AIC))
    pub window_factor: Option<f64>,
    // svc: "logt" | "fplogt"
    pub penalty: Option<String>,
    // hr: apply the 1983 erratum pass (default true)
    pub erratum: Option<bool>,
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
        "acf_robust" => Statistic::AcfRobust,
        other => panic!("unknown statistic {other}"),
    }
}

impl Est {
    /// Blank estimator with a name; used to build internal sub-estimators.
    fn named(name: &str) -> Est {
        Est {
            name: name.to_string(),
            kind: String::new(),
            criterion: None,
            method: None,
            max_order: None,
            statistic: None,
            threshold: None,
            c: None,
            alpha: None,
            memory: None,
            statistics: None,
            rule: None,
            window_factor: None,
            penalty: None,
            erratum: None,
        }
    }

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

    fn memory(&self, n: usize, stat: Statistic) -> tsorder_core::Memory {
        let Some(spec) = self.memory.as_deref() else {
            return tsorder_core::Memory::Default;
        };
        let m = match spec {
            "n03" => ((n as f64).powf(0.3).floor() as usize).max(2),
            "n04" => tsorder_core::power_memory(n),
            "sqrt2" => ((2.0 * (n as f64).sqrt()).floor() as usize).min(n / 4),
            "log2" => polylog_memory(n, 2.0),
            "log4" => polylog_memory(n, 4.0),
            "default" => stat.default_memory(n),
            s => s.parse().unwrap_or_else(|_| panic!("bad memory {s}")),
        };
        tsorder_core::Memory::Fixed(m)
    }

    fn fh_config(&self, n: usize, stat: Statistic) -> FhConfig {
        FhConfig {
            statistic: stat,
            threshold: self.threshold(n),
            memory: self.memory(n, stat),
        }
    }

    fn cc_window(&self, x: &[f64]) -> Result<usize, Error> {
        let d = self.window_factor.unwrap_or(2.0);
        let aic = ic_order(x, Criterion::Aic, None)?;
        Ok(((d * aic.order as f64).floor() as usize).max(2))
    }

    /// Run the estimator on `x`; `n` is the nominal sample size used to
    /// resolve n-dependent settings (α_n, memory).
    pub fn select(&self, x: &[f64], n: usize) -> Result<usize, Error> {
        match self.kind.as_str() {
            "svc" => {
                let penalty = match self.penalty.as_deref().unwrap_or("logt") {
                    "logt" => SvcPenalty::LogT,
                    "fplogt" => SvcPenalty::FpLogT,
                    other => panic!("unknown svc penalty {other}"),
                };
                Ok(tsorder_core::svc_order(
                    x,
                    SvcConfig {
                        window: None,
                        window_factor: self.window_factor.unwrap_or(2.0),
                        penalty,
                    },
                )?
                .order)
            }
            "svt" => Ok(hankel_svt_order(
                x,
                HankelSvtConfig {
                    k: None,
                    c: self.c.unwrap_or(1.0),
                },
            )?
            .order),
            "rank" => {
                let alpha = match &self.alpha {
                    Some(toml::Value::Float(a)) => RankAlpha::Fixed(*a),
                    None => RankAlpha::LogDecay,
                    other => panic!("bad rank alpha {other:?}"),
                };
                Ok(hankel_rank_order(
                    x,
                    HankelRankConfig {
                        k: None,
                        alpha,
                        pvalue: RankPvalue::Imhof,
                    },
                )?
                .order)
            }
            "cc_ic" => {
                // Akaike-1976-style criterion on the canonical correlations
                // (the AIC/BIC comparators of Camba-Méndez & Kapetanios):
                // minimise T log prod_{i<=r}(1 - rho_i^2)... equivalently
                // T sum_{i<=r} log(1 - rho_i^2) + C(T) * r(f + p - r).
                let w = self.cc_window(x)?;
                let cc = canonical_correlations_stacked(x, w, w)?;
                let t = n as f64;
                let c_t = match self.penalty.as_deref().unwrap_or("bic") {
                    "aic" => 2.0,
                    "bic" => t.ln(),
                    other => panic!("unknown cc_ic penalty {other}"),
                };
                let m = cc.correlations.len();
                let mut best = (f64::INFINITY, 0usize);
                let mut loglik = 0.0;
                for r in 0..=m {
                    let crit = t * loglik + c_t * (r * (2 * w - r)) as f64;
                    if crit < best.0 {
                        best = (crit, r);
                    }
                    if r < m {
                        let rho2 = cc.correlations[r].powi(2).min(1.0 - 1e-12);
                        loglik += (1.0 - rho2).ln();
                    }
                }
                Ok(best.1)
            }
            "ba_seq" => {
                // Bartlett's (1947) sequential canonical-correlation test as
                // used by CK: reject rank r if
                // [T - (f + p + 1)/2] * -sum_{i>r} log(1 - rho_i^2) exceeds
                // the chi2_{(f-r)(p-r)} quantile; level kappa / log n.
                let w = self.cc_window(x)?;
                let cc = canonical_correlations_stacked(x, w, w)?;
                let t = n as f64;
                let alpha = 0.05 * 50f64.ln() / t.ln().max(1.0);
                let m = cc.correlations.len();
                let mut tails: Vec<f64> = vec![0.0; m + 1];
                for r in (0..m).rev() {
                    let rho2 = cc.correlations[r].powi(2).min(1.0 - 1e-12);
                    tails[r] = tails[r + 1] - (1.0 - rho2).ln();
                }
                let scale = t - (2 * w + 1) as f64 / 2.0;
                for (r, tail) in tails.iter().take(m).enumerate() {
                    let stat = scale * tail;
                    let df = ((w - r) * (w - r)) as f64;
                    if chi2_sf(stat, df) >= alpha {
                        return Ok(r);
                    }
                }
                Ok(m)
            }
            "mbc" => {
                // Mode-by-consensus over the McMillan-degree estimators
                // (García-Hiernaux et al. 2012's MbC, with our voter set):
                // SVC(logT), cc-BIC, the rank test and the SVT cut.
                let votes = [
                    tsorder_core::svc_order(x, SvcConfig::default()).map(|r| r.order),
                    Est {
                        kind: "cc_ic".into(),
                        penalty: Some("bic".into()),
                        ..Est::named("mbc_cc")
                    }
                    .select(x, n),
                    hankel_rank_order(x, HankelRankConfig::default()).map(|r| r.order),
                    hankel_svt_order(x, HankelSvtConfig::default()).map(|r| r.order),
                ];
                let mut counts = std::collections::BTreeMap::new();
                for v in votes.into_iter().flatten() {
                    *counts.entry(v).or_insert(0usize) += 1;
                }
                counts
                    .into_iter()
                    .max_by_key(|&(order, c)| (c, std::cmp::Reverse(order)))
                    .map(|(order, _)| order)
                    .ok_or(Error::Singular)
            }
            "eacf" => {
                let r = eacf(x, EacfConfig::default())?;
                match r.order {
                    Some((p, q)) => Ok(10 * p + q),
                    None => Err(Error::Invalid("no clean EACF vertex")),
                }
            }
            "hr" => {
                let r = hannan_rissanen(
                    x,
                    HrConfig {
                        erratum_pass: self.erratum.unwrap_or(true),
                        ..Default::default()
                    },
                )?;
                Ok(10 * r.p + r.q)
            }
            "ic" => {
                let c = criterion(self.criterion.as_deref().expect("ic needs criterion"));
                let m = method(self.method.as_deref().unwrap_or("yw"));
                Ok(ic_order_with(x, c, m, self.max_order)?.order)
            }
            "fh" => {
                let stat = statistic(self.statistic.as_deref().expect("fh needs statistic"));
                Ok(fh_order(x, self.fh_config(n, stat))?.order)
            }
            "jirak5" => {
                let alpha = match &self.alpha {
                    Some(toml::Value::Float(a)) => *a,
                    None => 0.05,
                    other => panic!("bad jirak5 alpha {other:?}"),
                };
                Ok(tsorder_core::jirak_order(
                    x,
                    tsorder_core::JirakConfig {
                        alpha,
                        memory: self.memory(n, Statistic::JirakBand),
                    },
                )?
                .order)
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
