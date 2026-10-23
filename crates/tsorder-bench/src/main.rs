//! Benchmark runner: `tsorder-bench <design.toml>...`
//!
//! Each design file lists DGPs and estimators; the runner simulates every
//! (dgp, n, replication) series once — seeds are a pure function of
//! (dgp name, n, replication), so runs are reproducible and every estimator
//! sees the same series — evaluates all estimators on it, and writes
//! aggregated CSVs:
//!
//! * `<out>_orders.csv` — selected-order counts per (dgp, n, estimator, order)
//!   (a lossless aggregation of the selection distribution; order = -1 counts
//!   estimator errors);
//! * `<out>_pred.csv` — one-step-ahead squared-error sums (when `predict`);
//! * `<out>_time.csv` — median per-series wall time (when `timing`).

mod dgp;
mod spec;

use rayon::prelude::*;
use std::collections::BTreeMap;
use std::io::Write;
use std::time::Instant;

use dgp::Dgp;
use spec::Design;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("usage: tsorder-bench <design.toml>...");
        std::process::exit(2);
    }
    for path in &args {
        let text =
            std::fs::read_to_string(path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
        let design: Design =
            toml::from_str(&text).unwrap_or_else(|e| panic!("cannot parse {path}: {e}"));
        run(&design);
    }
}

fn seed_for(dgp: &str, n: usize, rep: usize) -> u64 {
    // FNV-1a over the identifying triple; never 0 (the sim remaps 0 anyway).
    let mut h: u64 = 0xcbf29ce484222325;
    for b in dgp.bytes().chain(n.to_le_bytes()).chain(rep.to_le_bytes()) {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

fn run(design: &Design) {
    let t0 = Instant::now();
    let run = &design.run;
    eprintln!("== run {} -> {}", run.name, run.out);
    let mut orders_rows: Vec<String> = Vec::new();
    let mut pred_rows: Vec<String> = Vec::new();
    let mut time_rows: Vec<String> = Vec::new();

    for d in &design.dgp {
        for &n in &run.n {
            if let Some(mx) = d.max_n {
                if n > mx {
                    continue;
                }
            }
            let reps = if n >= run.large_from.unwrap_or(3200) {
                run.replications_large.unwrap_or(run.replications)
            } else {
                run.replications
            };
            if run.timing.unwrap_or(false) {
                time_rows.extend(timing_rows(design, d, n));
                continue;
            }
            // counts[est][order+1] (index 0 = failure), plus prediction sums.
            let (counts, preds) = (0..reps)
                .into_par_iter()
                .map(|rep| evaluate_rep(design, d, n, rep))
                .reduce(
                    || {
                        (
                            vec![BTreeMap::<i64, u64>::new(); design.estimator.len()],
                            vec![(0.0f64, 0.0f64, 0u64); design.estimator.len()],
                        )
                    },
                    |mut a, b| {
                        for (ma, mb) in a.0.iter_mut().zip(b.0) {
                            for (k, v) in mb {
                                *ma.entry(k).or_insert(0) += v;
                            }
                        }
                        for (pa, pb) in a.1.iter_mut().zip(b.1) {
                            pa.0 += pb.0;
                            pa.1 += pb.1;
                            pa.2 += pb.2;
                        }
                        a
                    },
                );
            for (e, cmap) in design.estimator.iter().zip(&counts) {
                for (&order, &count) in cmap {
                    orders_rows.push(format!(
                        "{},{},{},{},{},{},{}",
                        run.name,
                        d.name,
                        n,
                        e.name,
                        order,
                        count,
                        d.true_order.map_or(String::new(), |t| t.to_string()),
                    ));
                }
            }
            if run.predict.unwrap_or(false) {
                for (e, &(s, s2, c)) in design.estimator.iter().zip(&preds) {
                    pred_rows.push(format!(
                        "{},{},{},{},{:.10e},{:.10e},{}",
                        run.name, d.name, n, e.name, s, s2, c
                    ));
                }
            }
            eprintln!(
                "   {} n={} ({} reps) done at {:.1}s",
                d.name,
                n,
                reps,
                t0.elapsed().as_secs_f64()
            );
        }
    }

    write_csv(
        &format!("{}_orders.csv", run.out),
        "run,dgp,n,estimator,order,count,true_order",
        &orders_rows,
    );
    if run.predict.unwrap_or(false) {
        write_csv(
            &format!("{}_pred.csv", run.out),
            "run,dgp,n,estimator,sum_sq_err,sum_sq_err_sq,count",
            &pred_rows,
        );
    }
    if run.timing.unwrap_or(false) {
        write_csv(
            &format!("{}_time.csv", run.out),
            "run,dgp,n,estimator,median_us",
            &time_rows,
        );
    }
    eprintln!(
        "== run {} finished in {:.1}s",
        run.name,
        t0.elapsed().as_secs_f64()
    );
}

/// One replication: simulate once, evaluate every estimator.
/// Returns per-estimator (order -> 1) and (err², err⁴, 1) for prediction.
#[allow(clippy::type_complexity)]
fn evaluate_rep(
    design: &Design,
    d: &Dgp,
    n: usize,
    rep: usize,
) -> (Vec<BTreeMap<i64, u64>>, Vec<(f64, f64, u64)>) {
    let predict = design.run.predict.unwrap_or(false);
    let m = n + usize::from(predict);
    let x = d.generate(m, seed_for(&d.name, n, rep));
    let (fit, next) = if predict {
        (&x[..n], Some(x[n]))
    } else {
        (&x[..], None)
    };
    let mut counts = vec![BTreeMap::new(); design.estimator.len()];
    let mut preds = vec![(0.0, 0.0, 0u64); design.estimator.len()];
    for (i, e) in design.estimator.iter().enumerate() {
        let order: i64 = match e.select(fit, n) {
            Ok(k) => k as i64,
            Err(_) => -1,
        };
        *counts[i].entry(order).or_insert(0) += 1;
        if let Some(x_next) = next {
            if order >= 0 {
                if let Some(err) = one_step_error(fit, order as usize, x_next) {
                    let e2 = err * err;
                    preds[i] = (e2, e2 * e2, 1);
                }
            }
        }
    }
    (counts, preds)
}

/// One-step-ahead forecast from the Yule–Walker AR(k) fit on `fit`.
fn one_step_error(fit: &[f64], k: usize, x_next: f64) -> Option<f64> {
    let mean = fit.iter().sum::<f64>() / fit.len() as f64;
    if k == 0 {
        return Some(x_next - mean);
    }
    let l = tsorder_core::levinson_from_series(fit, k).ok()?;
    let mut pred = mean;
    for (j, p) in l.phi_last.iter().enumerate() {
        pred += p * (fit[fit.len() - 1 - j] - mean);
    }
    Some(x_next - pred)
}

/// Median per-series wall time over `timing_series` series, per estimator.
fn timing_rows(design: &Design, d: &Dgp, n: usize) -> Vec<String> {
    let run = &design.run;
    let series: Vec<Vec<f64>> = (0..run.timing_series.unwrap_or(20))
        .map(|rep| d.generate(n, seed_for(&d.name, n, rep)))
        .collect();
    let mut rows = Vec::new();
    for e in &design.estimator {
        let mut times: Vec<f64> = series
            .iter()
            .map(|x| {
                let t = Instant::now();
                let _ = e.select(x, n);
                t.elapsed().as_secs_f64() * 1e6
            })
            .collect();
        times.sort_by(f64::total_cmp);
        rows.push(format!(
            "{},{},{},{},{:.2}",
            run.name,
            d.name,
            n,
            e.name,
            times[times.len() / 2]
        ));
    }
    rows
}

fn write_csv(path: &str, header: &str, rows: &[String]) {
    if rows.is_empty() {
        return;
    }
    if let Some(dir) = std::path::Path::new(path).parent() {
        std::fs::create_dir_all(dir).expect("cannot create output directory");
    }
    let mut f = std::fs::File::create(path).unwrap_or_else(|e| panic!("cannot write {path}: {e}"));
    writeln!(f, "{header}").unwrap();
    for r in rows {
        writeln!(f, "{r}").unwrap();
    }
    eprintln!("   wrote {path} ({} rows)", rows.len());
}
