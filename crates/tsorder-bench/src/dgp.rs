//! Data-generating processes for the benchmark. Base ARMA paths come from
//! `tsorder_core::sim`; the bench adds Gaussian-magnitude additive outliers
//! (the design asks for N(0, sd²) outliers, while the core's `Contamination`
//! uses a fixed ±magnitude) and ARFIMA(0, d, 0) by MA(∞) truncation.

use serde::Deserialize;
use tsorder_core::sim::{ArmaSim, Innovations, Xorshift};

#[derive(Deserialize)]
pub struct Dgp {
    pub name: String,
    #[serde(default)]
    pub phi: Vec<f64>,
    #[serde(default)]
    pub theta: Vec<f64>,
    /// True order for accuracy metrics; omit when it is not defined (AR(∞)).
    pub true_order: Option<i64>,
    /// "gaussian" (default) or `t<df>` (e.g. "t3").
    pub innovations: Option<String>,
    /// Additive outliers: with probability `outlier_prob`, add N(0, outlier_sd²).
    pub outlier_prob: Option<f64>,
    pub outlier_sd: Option<f64>,
    /// Observation noise N(0, obs_noise_sd²) on every point.
    pub obs_noise_sd: Option<f64>,
    /// ARFIMA(0, d, 0) instead of ARMA (phi/theta must be empty).
    pub arfima_d: Option<f64>,
    /// Skip sample sizes above this (e.g. expensive ARFIMA generation).
    pub max_n: Option<usize>,
}

impl Dgp {
    pub fn generate(&self, n: usize, seed: u64) -> Vec<f64> {
        let mut x = if let Some(d) = self.arfima_d {
            arfima(d, n, seed)
        } else {
            let innovations = match self.innovations.as_deref() {
                None | Some("gaussian") => Innovations::Gaussian,
                Some(t) if t.starts_with('t') => Innovations::StudentT {
                    df: t[1..].parse().expect("bad innovations df"),
                },
                Some(other) => panic!("unknown innovations {other}"),
            };
            ArmaSim {
                phi: &self.phi,
                theta: &self.theta,
                innovations,
                obs_noise_sd: self.obs_noise_sd.unwrap_or(0.0),
                ..Default::default()
            }
            .simulate(n, seed)
            .expect("simulation failed")
        };
        if let (Some(p), Some(sd)) = (self.outlier_prob, self.outlier_sd) {
            // Separate stream so the base path is identical to the clean DGP.
            let mut rng = Xorshift::new(seed ^ 0xA5A5_A5A5_5A5A_5A5A);
            for v in x.iter_mut() {
                if rng.uniform() < p {
                    *v += sd * rng.normal();
                }
            }
        }
        x
    }
}

/// ARFIMA(0, d, 0): x_t = Σ_{j=0}^{J} ψ_j e_{t−j}, ψ_0 = 1,
/// ψ_j = ψ_{j−1} (j − 1 + d)/j, truncated at J = 1000 (ψ_j = O(j^{d−1})).
fn arfima(d: f64, n: usize, seed: u64) -> Vec<f64> {
    let j_max = 1000usize;
    let mut psi = Vec::with_capacity(j_max + 1);
    psi.push(1.0f64);
    for j in 1..=j_max {
        let prev = psi[j - 1];
        psi.push(prev * (j as f64 - 1.0 + d) / j as f64);
    }
    let mut rng = Xorshift::new(seed);
    let e: Vec<f64> = (0..n + j_max).map(|_| rng.normal()).collect();
    (0..n)
        .map(|t| {
            let base = t + j_max;
            psi.iter().enumerate().map(|(j, &p)| p * e[base - j]).sum()
        })
        .collect()
}
