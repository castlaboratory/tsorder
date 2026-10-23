//! Writes the cross-language golden fixture to stdout as JSON: a fixed
//! simulated series plus the outputs of every public selector on it. The
//! Python (pytest) and R (testthat) bindings replay the same series and
//! must reproduce these numbers exactly — the M5 "same results in both
//! languages" criterion. Regenerate with:
//!
//! ```text
//! cargo run --release --example golden > bindings/golden.json
//! cp bindings/golden.json bindings/python/tests/golden.json
//! cp bindings/golden.json bindings/r/tests/testthat/golden.json
//! ```

use tsorder_core::sim::arma;
use tsorder_core::*;

fn js(v: &[f64]) -> String {
    let items: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    format!("[{}]", items.join(","))
}

fn main() {
    let x = arma(&[1.2, -0.6], &[0.8], 500, 42);
    let mut s = 0.0;
    let rw: Vec<f64> = x
        .iter()
        .map(|v| {
            s += v;
            s
        })
        .collect();
    let short: Vec<f64> = x[..20].to_vec();

    let ic = ic_order(&x, Criterion::Bic, None).unwrap();
    let fh = fh_order(&x, FhConfig::default()).unwrap();
    let svc = svc_order(&x, SvcConfig::default()).unwrap();
    let rank = hankel_rank_order(&x, HankelRankConfig::default()).unwrap();
    let svt = hankel_svt_order(&x, HankelSvtConfig::default()).unwrap();
    let ea = eacf(&x, EacfConfig::default()).unwrap();
    let hr = hannan_rissanen(&x, HrConfig::default()).unwrap();
    let integ = integration_order(&rw, IntegrationConfig::default()).unwrap();
    let q = qn(&short).unwrap();
    let racf = robust_autocorrelation(&x, 5).unwrap();

    let (eap, eaq) = ea
        .order
        .map(|(p, q)| (p as i64, q as i64))
        .unwrap_or((-1, -1));
    println!("{{");
    println!("\"series\": {},", js(&x));
    println!("\"rw\": {},", js(&rw));
    println!("\"short\": {},", js(&short));
    println!(
        "\"ic_bic\": {{\"order\": {}, \"values\": {}}},",
        ic.order,
        js(&ic.values)
    );
    println!(
        "\"fh_pacf\": {{\"order\": {}, \"p_n\": {}, \"tau\": {:.17e}, \"studentized\": {}}},",
        fh.order,
        fh.p_n,
        fh.tau,
        js(&fh.studentized)
    );
    println!(
        "\"svc\": {{\"order\": {}, \"correlations\": {}}},",
        svc.order,
        js(&svc.correlations)
    );
    println!(
        "\"hankel_rank\": {{\"order\": {}, \"pvalues\": {}}},",
        rank.order,
        js(&rank.pvalues)
    );
    println!(
        "\"hankel_svt\": {{\"order\": {}, \"xi\": {:.17e}, \"singular_values\": {}}},",
        svt.order,
        svt.xi,
        js(&svt.singular_values)
    );
    println!(
        "\"eacf\": {{\"p\": {}, \"q\": {}, \"row0\": {}}},",
        eap,
        eaq,
        js(&ea.table[0])
    );
    println!(
        "\"hr\": {{\"p\": {}, \"q\": {}, \"phi\": {}, \"theta\": {}, \"sigma2\": {:.17e}}},",
        hr.p,
        hr.q,
        js(&hr.phi),
        js(&hr.theta),
        hr.sigma2
    );
    println!(
        "\"integration\": {{\"d\": {}, \"stats\": {}}},",
        integ.d,
        js(&integ.stats)
    );
    println!("\"qn\": {q:.17e},");
    println!("\"robust_acf\": {}", js(&racf));
    println!("}}");
}
