//! Prints, as JSON, the studentized PACF path and threshold used by the
//! landing page chart (site/index.html), so the figure is reproducible:
//!
//! ```text
//! cargo run --example pacf_path
//! ```
//!
//! Series: AR(2) with phi = (1.2, -0.6), n = 400, the same xorshift +
//! Box-Muller generator as the integration tests (seed 42).

use tsorder_core::*;

struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

fn sim_ar(phi: &[f64], n: usize, seed: u64) -> Vec<f64> {
    let burn = 500;
    let mut rng = Rng(seed);
    let total = n + burn;
    let mut x = vec![0.0; total];
    for t in phi.len()..total {
        let mut v = rng.normal();
        for (j, p) in phi.iter().enumerate() {
            v += p * x[t - j - 1];
        }
        x[t] = v;
    }
    x[burn..].to_vec()
}

fn main() {
    let x = sim_ar(&[1.2, -0.6], 400, 42);
    let fh = fh_order(&x, FhConfig::default()).unwrap();
    let bic = ic_order(&x, Criterion::Bic, None).unwrap();
    let path: Vec<String> = fh
        .studentized
        .iter()
        .map(|t| format!("{:.3}", t.abs()))
        .collect();
    println!(
        "{{\"n\": 400, \"p_n\": {}, \"tau\": {:.3}, \"order\": {}, \"bic_order\": {}, \"abs_studentized_pacf\": [{}]}}",
        fh.p_n,
        fh.tau,
        fh.order,
        bic.order,
        path.join(", ")
    );
}
