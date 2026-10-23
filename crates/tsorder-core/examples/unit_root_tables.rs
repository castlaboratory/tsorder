//! Reproducibly regenerates the asymptotic quantile tables embedded in
//! `dist::{kpss_quantile, df_tau_c_quantile}`:
//!
//! ```text
//! cargo run --release --example unit_root_tables
//! ```
//!
//! KPSS level statistic: the integral of a squared Brownian bridge, simulated
//! as partial sums of demeaned iid normals; Dickey–Fuller τ_c: the t-statistic
//! of ρ in Δy_t = α + ρ y_{t−1} + e under a pure random walk. Discretisation
//! T = 5000, 400 000 replications, seeds 10^6 + r.

use tsorder_core::sim::Xorshift;

fn main() {
    let t = 5000usize;
    let reps = 400_000usize;
    let mut kpss: Vec<f64> = Vec::with_capacity(reps);
    let mut tau: Vec<f64> = Vec::with_capacity(reps);
    for r in 0..reps {
        let mut rng = Xorshift::new(1_000_000 + r as u64);
        let e: Vec<f64> = (0..t).map(|_| rng.normal()).collect();
        let mean = e.iter().sum::<f64>() / t as f64;
        let mut s = 0.0;
        let mut sum_s2 = 0.0;
        let mut var = 0.0;
        for v in &e {
            let d = v - mean;
            s += d;
            sum_s2 += s * s;
            var += d * d;
        }
        kpss.push(sum_s2 / (t as f64 * var));
        let mut y = 0.0f64;
        let n = t as f64;
        let (mut sy, mut syy, mut sd, mut syd) = (0.0, 0.0, 0.0, 0.0);
        let mut rows: Vec<(f64, f64)> = Vec::with_capacity(t);
        for v in &e {
            let (yl, d) = (y, *v);
            rows.push((yl, d));
            sy += yl;
            syy += yl * yl;
            sd += d;
            syd += yl * d;
            y += v;
        }
        let det = n * syy - sy * sy;
        let rho = (n * syd - sy * sd) / det;
        let alpha = (sd - rho * sy) / n;
        let rss: f64 = rows
            .iter()
            .map(|(yl, d)| (d - alpha - rho * yl).powi(2))
            .sum();
        let se = (rss / (n - 2.0) * n / det).sqrt();
        tau.push(rho / se);
    }
    kpss.sort_by(f64::total_cmp);
    tau.sort_by(f64::total_cmp);
    let probs = [
        0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50,
    ];
    print!("KPSS: ");
    for p in probs {
        print!(
            "({p}, {:.4}) ",
            kpss[(((1.0 - p) * reps as f64) as usize).min(reps - 1)]
        );
    }
    println!();
    print!("TAU_C: ");
    for p in probs {
        print!(
            "({p}, {:.4}) ",
            tau[((p * reps as f64) as usize).min(reps - 1)]
        );
    }
    println!();
}
