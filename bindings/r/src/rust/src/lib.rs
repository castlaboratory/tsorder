//! Thin extendr shell over `tsorder-core`: type conversion only, no
//! statistical logic. Every `rs_*` function maps a core call to an R list.

use extendr_api::prelude::*;
use extendr_api::Result;
use tsorder_core as core;

fn rerr<T>(r: std::result::Result<T, core::Error>) -> Result<T> {
    r.map_err(|e| Error::Other(e.to_string()))
}

fn criterion(name: &str) -> Result<core::Criterion> {
    Ok(match name {
        "aic" => core::Criterion::Aic,
        "aicc" => core::Criterion::Aicc,
        "bic" => core::Criterion::Bic,
        "hq" => core::Criterion::Hq,
        "fpe" => core::Criterion::Fpe,
        "cat" => core::Criterion::Cat,
        "mdl" => core::Criterion::Mdl,
        other => return Err(Error::Other(format!("unknown criterion '{other}'"))),
    })
}

fn statistic(name: &str) -> Result<core::Statistic> {
    Ok(match name {
        "pacf" => core::Statistic::Pacf,
        "long_ar_yw" => core::Statistic::LongArYw,
        "long_ar_ols" => core::Statistic::LongArOls,
        "acf" => core::Statistic::Acf,
        "acf_bartlett" => core::Statistic::AcfBartlett,
        "inverse_pacf" => core::Statistic::InversePacf,
        "jirak_band" => core::Statistic::JirakBand,
        "pacf_robust" => core::Statistic::PacfRobust,
        "acf_robust" => core::Statistic::AcfRobust,
        other => return Err(Error::Other(format!("unknown statistic '{other}'"))),
    })
}

fn opt_usize(v: Nullable<i32>) -> Option<usize> {
    match v {
        Nullable::NotNull(k) if k >= 0 => Some(k as usize),
        _ => None,
    }
}

/// Information-criterion order selection (Rust core).
/// @noRd
#[extendr]
fn rs_ic_order(
    x: Vec<f64>,
    criterion_name: &str,
    method_name: &str,
    max_order: Nullable<i32>,
) -> Result<List> {
    let method = match method_name {
        "yw" => core::IcMethod::YuleWalker,
        "ols" => core::IcMethod::Ols,
        "burg" => core::IcMethod::Burg,
        other => return Err(Error::Other(format!("unknown method '{other}'"))),
    };
    let r = rerr(core::ic_order_with(
        &x,
        criterion(criterion_name)?,
        method,
        opt_usize(max_order),
    ))?;
    Ok(list!(
        order = r.order as i32,
        values = r.values,
        sigma2 = r.sigma2,
        criterion = criterion_name,
        method = method_name
    ))
}

/// Fine–Hwang threshold order selection (Rust core).
/// @noRd
#[extendr]
fn rs_fh_order(
    x: Vec<f64>,
    statistic_name: &str,
    threshold_name: &str,
    c: f64,
    alpha: f64,
    memory: Nullable<i32>,
) -> Result<List> {
    let threshold = match threshold_name {
        "bic" => core::Threshold::BicLike { c },
        "lil" => core::Threshold::Lil { c },
        "maxk" => core::Threshold::MaxOfK { c },
        "gumbel" => core::Threshold::Gumbel { alpha },
        "fixed" => core::Threshold::Fixed(c),
        other => return Err(Error::Other(format!("unknown threshold '{other}'"))),
    };
    let cfg = core::FhConfig {
        statistic: statistic(statistic_name)?,
        threshold,
        memory: match opt_usize(memory) {
            Some(k) => core::Memory::Fixed(k),
            None => core::Memory::Default,
        },
    };
    let r = rerr(core::fh_order(&x, cfg))?;
    Ok(list!(
        order = r.order as i32,
        p_n = r.p_n as i32,
        tau = r.tau,
        values = r.values,
        studentized = r.studentized,
        statistic = statistic_name,
        threshold = threshold_name
    ))
}

/// Jirak's refitted order estimator q^(5) (Rust core).
/// @noRd
#[extendr]
fn rs_jirak_order(x: Vec<f64>, alpha: f64, memory: Nullable<i32>) -> Result<List> {
    let cfg = core::JirakConfig {
        alpha,
        memory: match opt_usize(memory) {
            Some(k) => core::Memory::Fixed(k),
            None => core::Memory::Default,
        },
    };
    let r = rerr(core::jirak_order(&x, cfg))?;
    Ok(list!(
        order = r.order as i32,
        p_n = r.p_n as i32,
        tau = r.tau,
        per_dimension = r.per_dimension.iter().map(|q| *q as i32).collect::<Vec<i32>>()
    ))
}

/// Bauer's SVC estimator of the McMillan degree (Rust core).
/// @noRd
#[extendr]
fn rs_svc_order(
    x: Vec<f64>,
    penalty_name: &str,
    window_factor: f64,
    window: Nullable<i32>,
) -> Result<List> {
    let penalty = match penalty_name {
        "logt" => core::SvcPenalty::LogT,
        "fplogt" => core::SvcPenalty::FpLogT,
        other => return Err(Error::Other(format!("unknown penalty '{other}'"))),
    };
    let r = rerr(core::svc_order(
        &x,
        core::SvcConfig {
            window: opt_usize(window),
            window_factor,
            penalty,
        },
    ))?;
    Ok(list!(
        order = r.order as i32,
        correlations = r.correlations,
        values = r.values,
        f = r.f as i32,
        p = r.p as i32
    ))
}

/// Hankel singular-value-threshold estimator (Rust core).
/// @noRd
#[extendr]
fn rs_hankel_svt_order(x: Vec<f64>, c: f64, k: Nullable<i32>) -> Result<List> {
    let r = rerr(core::hankel_svt_order(
        &x,
        core::HankelSvtConfig { k: opt_usize(k), c },
    ))?;
    Ok(list!(
        order = r.order as i32,
        singular_values = r.singular_values,
        xi = r.xi,
        k = r.k as i32
    ))
}

/// Sequential rank test on the autocovariance Hankel (Rust core).
/// @noRd
#[extendr]
fn rs_hankel_rank_order(x: Vec<f64>, alpha: Nullable<f64>, k: Nullable<i32>) -> Result<List> {
    let alpha_rule = match alpha {
        Nullable::NotNull(a) => core::RankAlpha::Fixed(a),
        Nullable::Null => core::RankAlpha::LogDecay,
    };
    let r = rerr(core::hankel_rank_order(
        &x,
        core::HankelRankConfig {
            k: opt_usize(k),
            alpha: alpha_rule,
            pvalue: core::RankPvalue::Imhof,
        },
    ))?;
    Ok(list!(
        order = r.order as i32,
        pvalues = r.pvalues,
        alpha = r.alpha,
        k = r.k as i32
    ))
}

/// Extended sample autocorrelation function (Rust core).
/// @noRd
#[extendr]
fn rs_eacf(x: Vec<f64>, max_ar: i32, max_ma: i32, band: f64) -> Result<List> {
    let r = rerr(core::eacf(
        &x,
        core::EacfConfig {
            max_ar: max_ar.max(0) as usize,
            max_ma: max_ma.max(0) as usize,
            band,
        },
    ))?;
    let rows = r.table.len();
    let cols = r.table[0].len();
    let table = RMatrix::new_matrix(rows, cols, |i, j| r.table[i][j]);
    // Logical matrices lack a typed-slice impl; 0/1 doubles convert cleanly in R.
    let signif = RMatrix::new_matrix(rows, cols, |i, j| {
        if r.significant[i][j] {
            1.0
        } else {
            0.0
        }
    });
    let (p, q, found) = match r.order {
        Some((p, q)) => (p as i32, q as i32, true),
        None => (-1, -1, false),
    };
    Ok(list!(
        p = p,
        q = q,
        found = found,
        table = table,
        significant = signif
    ))
}

/// Hannan–Rissanen order selection and initial estimates (Rust core).
/// @noRd
#[extendr]
fn rs_hannan_rissanen(x: Vec<f64>, max_p: i32, max_q: i32, erratum: bool) -> Result<List> {
    let r = rerr(core::hannan_rissanen(
        &x,
        core::HrConfig {
            max_p: max_p.max(0) as usize,
            max_q: max_q.max(0) as usize,
            long_ar: None,
            erratum_pass: erratum,
        },
    ))?;
    Ok(list!(
        p = r.p as i32,
        q = r.q as i32,
        phi = r.phi,
        theta = r.theta,
        sigma2 = r.sigma2,
        long_ar = r.long_ar as i32
    ))
}

/// Order of integration by sequential unit-root tests (Rust core).
/// @noRd
#[extendr]
fn rs_integration_order(
    x: Vec<f64>,
    max_d: i32,
    alpha: Nullable<f64>,
    test_name: &str,
) -> Result<List> {
    let test = match test_name {
        "kpss" => core::UnitRootTest::Kpss,
        "adf" => core::UnitRootTest::Adf,
        other => return Err(Error::Other(format!("unknown test '{other}'"))),
    };
    let alpha_opt = match alpha {
        Nullable::NotNull(a) => Some(a),
        Nullable::Null => None,
    };
    let r = rerr(core::integration_order(
        &x,
        core::IntegrationConfig {
            max_d: max_d.max(0) as usize,
            alpha: alpha_opt,
            test,
        },
    ))?;
    Ok(list!(
        d = r.d as i32,
        stats = r.stats,
        crits = r.crits,
        alpha = r.alpha,
        test = test_name
    ))
}

/// Q_n robust scale estimator (Rust core; robustbase conventions).
/// @noRd
#[extendr]
fn rs_qn(x: Vec<f64>) -> Result<f64> {
    rerr(core::qn(&x))
}

/// Q_n-based robust autocorrelations (Rust core).
/// @noRd
#[extendr]
fn rs_robust_acf(x: Vec<f64>, max_lag: i32) -> Result<Vec<f64>> {
    rerr(core::robust_autocorrelation(&x, max_lag.max(1) as usize))
}

/// Partial autocorrelations and innovation variances by Levinson (Rust core).
/// @noRd
#[extendr]
fn rs_levinson(x: Vec<f64>, k_max: i32) -> Result<List> {
    let l = rerr(core::levinson_from_series(&x, k_max.max(1) as usize))?;
    Ok(list!(pacf = l.pacf, sigma2 = l.sigma2, phi = l.phi_last))
}

extendr_module! {
    mod tsorder;
    fn rs_ic_order;
    fn rs_fh_order;
    fn rs_jirak_order;
    fn rs_svc_order;
    fn rs_hankel_svt_order;
    fn rs_hankel_rank_order;
    fn rs_eacf;
    fn rs_hannan_rissanen;
    fn rs_integration_order;
    fn rs_qn;
    fn rs_robust_acf;
    fn rs_levinson;
}
