//! Simulation utilities for tests and benchmarks (feature `sim`).
//!
//! A tiny xorshift64 + Box–Muller generator keeps the crate dependency-free
//! and the draws reproducible across platforms: the same seed gives the same
//! series, bit for bit. [`arma`] is the plain Gaussian ARMA generator the
//! integration tests use; [`ArmaSim`] adds Student-t innovations, additive
//! outliers and observation noise for benchmark scenarios.

use crate::{Error, Result};

/// Xorshift64 with Box–Muller normals. Deterministic and platform-independent.
#[derive(Debug, Clone)]
pub struct Xorshift(u64);

impl Xorshift {
    /// A zero seed would make xorshift stick at zero, so it is remapped.
    pub fn new(seed: u64) -> Self {
        Xorshift(if seed == 0 { 0x9E3779B97F4A7C15 } else { seed })
    }

    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform on `[0, 1)` with 53 random bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Standard normal via Box–Muller (one draw per call, cosine branch).
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-300);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f64::consts::PI * u2).cos()
    }
}

/// Innovation distribution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Innovations {
    /// Standard normal.
    Gaussian,
    /// Student t with `df` degrees of freedom (`df >= 1`), sampled as
    /// `Z / sqrt(χ²_df / df)`. Not rescaled to unit variance (which is
    /// `df/(df-2)` for `df > 2`); every statistic and criterion in the crate
    /// is scale-invariant, so only the tail weight matters.
    StudentT { df: u32 },
}

/// Additive-outlier contamination: with probability `prob`, add `±magnitude`
/// (random sign) to the observed value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contamination {
    pub prob: f64,
    pub magnitude: f64,
}

/// A simulation design: ARMA(φ, θ) plus the perturbations the benchmark needs.
#[derive(Debug, Clone, Copy)]
pub struct ArmaSim<'a> {
    pub phi: &'a [f64],
    pub theta: &'a [f64],
    pub innovations: Innovations,
    /// Applied to the observed series, after generation.
    pub additive_outliers: Option<Contamination>,
    /// Standard deviation of Gaussian measurement noise added to the observed
    /// series (0 = none).
    pub obs_noise_sd: f64,
    /// Transient discarded at the start.
    pub burn_in: usize,
}

impl Default for ArmaSim<'static> {
    fn default() -> Self {
        ArmaSim {
            phi: &[],
            theta: &[],
            innovations: Innovations::Gaussian,
            additive_outliers: None,
            obs_noise_sd: 0.0,
            burn_in: 500,
        }
    }
}

impl ArmaSim<'_> {
    /// Simulate `n` observations. Draw order is fixed (innovations first, then
    /// one uniform per observation for outliers plus a sign when one fires,
    /// then one normal per observation if `obs_noise_sd > 0`), so results are
    /// reproducible for a given seed whatever the configuration.
    pub fn simulate(&self, n: usize, seed: u64) -> Result<Vec<f64>> {
        if n == 0 {
            return Err(Error::Invalid("n must be >= 1"));
        }
        if let Innovations::StudentT { df } = self.innovations {
            if df == 0 {
                return Err(Error::Invalid("StudentT df must be >= 1"));
            }
        }
        if let Some(c) = self.additive_outliers {
            if !(0.0..=1.0).contains(&c.prob) || !c.magnitude.is_finite() {
                return Err(Error::Invalid("contamination prob must be in [0, 1]"));
            }
        }
        if !self.obs_noise_sd.is_finite() || self.obs_noise_sd < 0.0 {
            return Err(Error::Invalid("obs_noise_sd must be finite and >= 0"));
        }
        let mut rng = Xorshift::new(seed);
        let total = n + self.burn_in;
        let e: Vec<f64> = (0..total)
            .map(|_| draw(&mut rng, self.innovations))
            .collect();
        let start = self.phi.len().max(self.theta.len());
        let mut x = vec![0.0; total];
        for t in start..total {
            let mut v = e[t];
            for (j, p) in self.phi.iter().enumerate() {
                v += p * x[t - j - 1];
            }
            for (j, th) in self.theta.iter().enumerate() {
                v += th * e[t - j - 1];
            }
            x[t] = v;
        }
        let mut out = x[self.burn_in..].to_vec();
        if let Some(c) = self.additive_outliers {
            for v in out.iter_mut() {
                if rng.uniform() < c.prob {
                    let sign = if rng.uniform() < 0.5 { -1.0 } else { 1.0 };
                    *v += sign * c.magnitude;
                }
            }
        }
        if self.obs_noise_sd > 0.0 {
            for v in out.iter_mut() {
                *v += self.obs_noise_sd * rng.normal();
            }
        }
        Ok(out)
    }
}

fn draw(rng: &mut Xorshift, innovations: Innovations) -> f64 {
    match innovations {
        Innovations::Gaussian => rng.normal(),
        Innovations::StudentT { df } => {
            let z = rng.normal();
            let chi2: f64 = (0..df).map(|_| rng.normal().powi(2)).sum();
            z / (chi2 / df as f64).sqrt()
        }
    }
}

/// Gaussian ARMA with a 500-observation burn-in: the generator the integration
/// tests (and the landing-page example) use.
pub fn arma(phi: &[f64], theta: &[f64], n: usize, seed: u64) -> Vec<f64> {
    ArmaSim {
        phi,
        theta,
        ..Default::default()
    }
    .simulate(n, seed)
    .expect("default ArmaSim configuration is valid")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_series() {
        let a = arma(&[0.5], &[], 100, 42);
        let b = arma(&[0.5], &[], 100, 42);
        assert_eq!(a, b);
        let c = arma(&[0.5], &[], 100, 43);
        assert_ne!(a, c);
    }

    #[test]
    fn student_t_has_heavier_tails_than_gaussian() {
        let base = ArmaSim::default();
        let t3 = ArmaSim {
            innovations: Innovations::StudentT { df: 3 },
            ..Default::default()
        };
        let g = base.simulate(20000, 7).unwrap();
        let t = t3.simulate(20000, 7).unwrap();
        let kurt = |x: &[f64]| {
            let n = x.len() as f64;
            let m2 = x.iter().map(|v| v * v).sum::<f64>() / n;
            let m4 = x.iter().map(|v| v.powi(4)).sum::<f64>() / n;
            m4 / (m2 * m2)
        };
        assert!(kurt(&g) < 3.5, "gaussian kurtosis {}", kurt(&g));
        assert!(kurt(&t) > 4.5, "t3 kurtosis {}", kurt(&t));
    }

    #[test]
    fn invalid_configs_are_errors() {
        assert!(matches!(
            ArmaSim {
                innovations: Innovations::StudentT { df: 0 },
                ..Default::default()
            }
            .simulate(100, 1),
            Err(Error::Invalid(_))
        ));
        assert!(matches!(
            ArmaSim {
                additive_outliers: Some(Contamination {
                    prob: 1.5,
                    magnitude: 9.0
                }),
                ..Default::default()
            }
            .simulate(100, 1),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn outliers_and_noise_change_only_the_observed_series() {
        let clean = ArmaSim {
            phi: &[0.7],
            ..Default::default()
        };
        let dirty = ArmaSim {
            phi: &[0.7],
            additive_outliers: Some(Contamination {
                prob: 0.05,
                magnitude: 10.0,
            }),
            obs_noise_sd: 0.5,
            ..Default::default()
        };
        let a = clean.simulate(500, 11).unwrap();
        let b = dirty.simulate(500, 11).unwrap();
        // Same innovations: the two series differ only by the perturbations.
        let n_big = a
            .iter()
            .zip(&b)
            .filter(|(u, v)| (*u - *v).abs() > 5.0)
            .count();
        assert!(n_big > 5, "expected some outliers, got {n_big}");
    }
}
