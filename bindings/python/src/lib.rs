//! Thin PyO3 shell over `tsorder-core` (D-01): type conversion only, no
//! statistical logic. Functions accept any sequence of floats (lists, numpy
//! arrays, pandas Series) and return plain dicts.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use tsorder_core as core;

fn perr<T>(r: Result<T, core::Error>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

fn criterion(name: &str) -> PyResult<core::Criterion> {
    Ok(match name {
        "aic" => core::Criterion::Aic,
        "aicc" => core::Criterion::Aicc,
        "bic" => core::Criterion::Bic,
        "hq" => core::Criterion::Hq,
        "fpe" => core::Criterion::Fpe,
        "cat" => core::Criterion::Cat,
        "mdl" => core::Criterion::Mdl,
        other => return Err(PyValueError::new_err(format!("unknown criterion '{other}'"))),
    })
}

fn statistic(name: &str) -> PyResult<core::Statistic> {
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
        other => return Err(PyValueError::new_err(format!("unknown statistic '{other}'"))),
    })
}

/// Information-criterion order selection.
#[pyfunction]
#[pyo3(signature = (x, criterion="bic", method="yw", max_order=None))]
fn ic_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    criterion: &str,
    method: &str,
    max_order: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = match method {
        "yw" => core::IcMethod::YuleWalker,
        "ols" => core::IcMethod::Ols,
        "burg" => core::IcMethod::Burg,
        other => return Err(PyValueError::new_err(format!("unknown method '{other}'"))),
    };
    let r = perr(core::ic_order_with(&x, self::criterion(criterion)?, m, max_order))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("values", r.values)?;
    d.set_item("sigma2", r.sigma2)?;
    d.set_item("criterion", criterion)?;
    d.set_item("method", method)?;
    Ok(d)
}

/// Fine–Hwang threshold order selection.
#[pyfunction]
#[pyo3(signature = (x, statistic="pacf", threshold="bic", c=1.5, alpha=0.05, memory=None))]
fn fh_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    statistic: &str,
    threshold: &str,
    c: f64,
    alpha: f64,
    memory: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let thr = match threshold {
        "bic" => core::Threshold::BicLike { c },
        "lil" => core::Threshold::Lil { c },
        "maxk" => core::Threshold::MaxOfK { c },
        "gumbel" => core::Threshold::Gumbel { alpha },
        "fixed" => core::Threshold::Fixed(c),
        other => return Err(PyValueError::new_err(format!("unknown threshold '{other}'"))),
    };
    let cfg = core::FhConfig {
        statistic: self::statistic(statistic)?,
        threshold: thr,
        memory: match memory {
            Some(k) => core::Memory::Fixed(k),
            None => core::Memory::Default,
        },
    };
    let r = perr(core::fh_order(&x, cfg))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("p_n", r.p_n)?;
    d.set_item("tau", r.tau)?;
    d.set_item("values", r.values)?;
    d.set_item("studentized", r.studentized)?;
    d.set_item("statistic", statistic)?;
    d.set_item("threshold", threshold)?;
    Ok(d)
}

/// Jirak's refitted order estimator q^(5): max over per-dimension refits.
#[pyfunction]
#[pyo3(signature = (x, alpha=0.05, memory=None))]
fn jirak_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    alpha: f64,
    memory: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let cfg = core::JirakConfig {
        alpha,
        memory: match memory {
            Some(k) => core::Memory::Fixed(k),
            None => core::Memory::Default,
        },
    };
    let r = perr(core::jirak_order(&x, cfg))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("p_n", r.p_n)?;
    d.set_item("tau", r.tau)?;
    d.set_item("per_dimension", r.per_dimension)?;
    Ok(d)
}

/// Bauer's SVC estimator of the McMillan degree.
#[pyfunction]
#[pyo3(signature = (x, penalty="logt", window_factor=1.0, window=None))]
fn svc_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    penalty: &str,
    window_factor: f64,
    window: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let p = match penalty {
        "logt" => core::SvcPenalty::LogT,
        "fplogt" => core::SvcPenalty::FpLogT,
        other => return Err(PyValueError::new_err(format!("unknown penalty '{other}'"))),
    };
    let r = perr(core::svc_order(
        &x,
        core::SvcConfig {
            window,
            window_factor,
            penalty: p,
        },
    ))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("correlations", r.correlations)?;
    d.set_item("values", r.values)?;
    d.set_item("f", r.f)?;
    d.set_item("p", r.p)?;
    Ok(d)
}

/// Sequential rank test on the autocovariance Hankel.
#[pyfunction]
#[pyo3(signature = (x, alpha=None, k=None))]
fn hankel_rank_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    alpha: Option<f64>,
    k: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let rule = match alpha {
        Some(a) => core::RankAlpha::Fixed(a),
        None => core::RankAlpha::LogDecay,
    };
    let r = perr(core::hankel_rank_order(
        &x,
        core::HankelRankConfig {
            k,
            alpha: rule,
            pvalue: core::RankPvalue::Imhof,
        },
    ))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("pvalues", r.pvalues)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("k", r.k)?;
    Ok(d)
}

/// Hankel singular-value-threshold estimator of the McMillan degree.
#[pyfunction]
#[pyo3(signature = (x, c=1.0, k=None))]
fn hankel_svt_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    c: f64,
    k: Option<usize>,
) -> PyResult<Bound<'py, PyDict>> {
    let r = perr(core::hankel_svt_order(&x, core::HankelSvtConfig { k, c }))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("singular_values", r.singular_values)?;
    d.set_item("xi", r.xi)?;
    d.set_item("k", r.k)?;
    Ok(d)
}

/// Extended sample autocorrelation function with automated vertex reading.
#[pyfunction]
#[pyo3(signature = (x, max_ar=7, max_ma=13, band=2.0))]
fn eacf<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    max_ar: usize,
    max_ma: usize,
    band: f64,
) -> PyResult<Bound<'py, PyDict>> {
    let r = perr(core::eacf(
        &x,
        core::EacfConfig {
            max_ar,
            max_ma,
            band,
        },
    ))?;
    let d = PyDict::new(py);
    d.set_item("order", r.order)?;
    d.set_item("table", r.table)?;
    d.set_item("significant", r.significant)?;
    Ok(d)
}

/// Hannan–Rissanen ARMA order selection (stages 1–2, with the 1983 erratum pass).
#[pyfunction]
#[pyo3(signature = (x, max_p=5, max_q=5, erratum=true))]
fn hannan_rissanen<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    max_p: usize,
    max_q: usize,
    erratum: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let r = perr(core::hannan_rissanen(
        &x,
        core::HrConfig {
            max_p,
            max_q,
            long_ar: None,
            erratum_pass: erratum,
        },
    ))?;
    let d = PyDict::new(py);
    d.set_item("p", r.p)?;
    d.set_item("q", r.q)?;
    d.set_item("phi", r.phi)?;
    d.set_item("theta", r.theta)?;
    d.set_item("sigma2", r.sigma2)?;
    d.set_item("long_ar", r.long_ar)?;
    Ok(d)
}

/// Order of integration by sequential unit-root testing (vanishing level).
#[pyfunction]
#[pyo3(signature = (x, max_d=2, alpha=None, test="kpss"))]
fn integration_order<'py>(
    py: Python<'py>,
    x: Vec<f64>,
    max_d: usize,
    alpha: Option<f64>,
    test: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let t = match test {
        "kpss" => core::UnitRootTest::Kpss,
        "adf" => core::UnitRootTest::Adf,
        other => return Err(PyValueError::new_err(format!("unknown test '{other}'"))),
    };
    let r = perr(core::integration_order(
        &x,
        core::IntegrationConfig {
            max_d,
            alpha,
            test: t,
        },
    ))?;
    let d = PyDict::new(py);
    d.set_item("d", r.d)?;
    d.set_item("stats", r.stats)?;
    d.set_item("crits", r.crits)?;
    d.set_item("alpha", r.alpha)?;
    d.set_item("test", test)?;
    Ok(d)
}

/// Q_n robust scale estimator (robustbase conventions).
#[pyfunction]
fn qn(x: Vec<f64>) -> PyResult<f64> {
    perr(core::qn(&x))
}

/// Q_n-based robust autocorrelations (bounded ratio form).
#[pyfunction]
fn robust_acf(x: Vec<f64>, max_lag: usize) -> PyResult<Vec<f64>> {
    perr(core::robust_autocorrelation(&x, max_lag))
}

#[pymodule]
fn tsorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ic_order, m)?)?;
    m.add_function(wrap_pyfunction!(fh_order, m)?)?;
    m.add_function(wrap_pyfunction!(jirak_order, m)?)?;
    m.add_function(wrap_pyfunction!(svc_order, m)?)?;
    m.add_function(wrap_pyfunction!(hankel_rank_order, m)?)?;
    m.add_function(wrap_pyfunction!(hankel_svt_order, m)?)?;
    m.add_function(wrap_pyfunction!(eacf, m)?)?;
    m.add_function(wrap_pyfunction!(hannan_rissanen, m)?)?;
    m.add_function(wrap_pyfunction!(integration_order, m)?)?;
    m.add_function(wrap_pyfunction!(qn, m)?)?;
    m.add_function(wrap_pyfunction!(robust_acf, m)?)?;
    Ok(())
}
