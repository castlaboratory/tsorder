//! Vanishing thresholds `τ_n` and growing memory `p_n` for Fine–Hwang estimators.
//!
//! Thresholds are expressed on the *studentized* scale: lag `k` survives iff
//! `|stat_k / se_k| > τ_n`. On the raw scale of Fine & Hwang this is
//! `η_n = τ_n · se_k ≈ τ_n / sqrt(n)`.
//!
//! **Which theory justifies which threshold.** Hwang's (1978, Thm 5.3) sufficient
//! condition `p_n / (n η_n²) → 0` translates to `τ_n² / p_n → ∞`, i.e. the threshold
//! must grow *faster than `sqrt(p_n)`*. None of the `log n`-type thresholds below
//! satisfies it with `p_n = n^0.4` (the ratio `p_n/τ_n²` grows from 0.9 at n=100 to
//! 5.8 at n=10⁵). Hwang's condition comes from a Chebyshev bound and is far from
//! necessary; the `sqrt(c log n)` and Gumbel-type thresholds are instead justified by
//! maximal inequalities / extreme-value limits for the vector of long-AR estimates —
//! see Jirak (2012, *Ann. Statist.*), who proves a Gumbel limit for
//! `max_{i ≤ d_n} |θ̂_i − θ_i|` with `d_n = O(n^δ)` and derives consistent band-based
//! order estimators from it.

use crate::{Error, Result};

/// Threshold rule on the studentized scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Threshold {
    /// `sqrt(c · log n)`. With `c = 1` this equals the per-increment threshold implicit in BIC
    /// (adjacent-order BIC difference is negative iff `t_k² > log n` to first order) — this is
    /// *not* the same as BIC's global argmin. Requires `c > 0`.
    BicLike { c: f64 },
    /// `sqrt(2 c · log log n)`: Hannan–Quinn / law-of-the-iterated-logarithm rate. Requires `c > 0`;
    /// consistency arguments need `c > 1`.
    Lil { c: f64 },
    /// `sqrt(2 · log p_n) + c`: Bonferroni/extreme-value correction for the number of lags
    /// inspected — the natural companion of Jirak's (2012) Gumbel limit. Requires `c ≥ 0`.
    MaxOfK { c: f64 },
    /// Fixed value on the studentized scale, e.g. `1.96` reproduces the Box–Jenkins band
    /// (not consistent; for benchmarking). Requires a finite value `≥ 0`.
    Fixed(f64),
    /// Level-calibrated threshold: the `1−alpha` quantile of the Gumbel limit of
    /// `max_{k ≤ p_n} |Z_k|`, `Z_k` approximately standard normal, i.e.
    /// `b_p + a_p · (−log(−log(1−alpha)))` with `a_p = 1/sqrt(2 log p)` and
    /// `b_p = sqrt(2 log p) − (log log p + log π) / (2 sqrt(2 log p))` (absolute values ⇒ `2p` tails).
    /// With a fixed `alpha` this controls the family-wise overestimation probability at ≈ `alpha`
    /// (not consistent); with `alpha = alpha_n → 0` slowly it is the Jirak-type consistent rule
    /// (D-20). Jirak (2012) gives the exact normalising constants for Yule–Walker estimates;
    /// this generic form is a placeholder until `Statistic::JirakBand` lands (PLAN M1).
    /// Requires `0 < alpha < 1`.
    Gumbel { alpha: f64 },
}

impl Default for Threshold {
    /// `BicLike { c: 1.5 }` — matched BIC in the 2026-10-23 simulations. Chosen empirically;
    /// not derived from a theorem (D-06).
    fn default() -> Self {
        Threshold::BicLike { c: 1.5 }
    }
}

impl Threshold {
    /// Reject parameters that would make the rule meaningless (negative or NaN constants).
    pub fn validate(self) -> Result<()> {
        let ok = match self {
            Threshold::BicLike { c } | Threshold::Lil { c } => c.is_finite() && c > 0.0,
            Threshold::MaxOfK { c } => c.is_finite() && c >= 0.0,
            Threshold::Fixed(v) => v.is_finite() && v >= 0.0,
            Threshold::Gumbel { alpha } => alpha.is_finite() && alpha > 0.0 && alpha < 1.0,
        };
        if ok {
            Ok(())
        } else {
            Err(Error::Invalid(
                "threshold constant must be finite and positive (BicLike/Lil) or non-negative (MaxOfK/Fixed)",
            ))
        }
    }

    /// Numerical threshold for sample size `n` and memory `p_n`.
    pub fn value(self, n: usize, p_n: usize) -> f64 {
        let nf = (n.max(3)) as f64;
        match self {
            Threshold::BicLike { c } => (c * nf.ln()).sqrt(),
            Threshold::Lil { c } => (2.0 * c * nf.ln().ln().max(0.0)).sqrt(),
            Threshold::MaxOfK { c } => (2.0 * (p_n.max(2) as f64).ln()).sqrt() + c,
            Threshold::Fixed(v) => v,
            Threshold::Gumbel { alpha } => {
                // max of p |N(0,1)| ≈ max of 2p one-sided tails
                let m = (2 * p_n.max(2)) as f64;
                let l = (2.0 * m.ln()).sqrt();
                let a = 1.0 / l;
                let b = l - (m.ln().ln() + (4.0 * core::f64::consts::PI).ln()) / (2.0 * l);
                b + a * (-(-(1.0 - alpha).ln()).ln())
            }
        }
    }

    pub fn name(self) -> String {
        match self {
            Threshold::BicLike { c } => format!("bic(c={c})"),
            Threshold::Lil { c } => format!("lil(c={c})"),
            Threshold::MaxOfK { c } => format!("maxk(c={c})"),
            Threshold::Fixed(v) => format!("fixed({v})"),
            Threshold::Gumbel { alpha } => format!("gumbel(alpha={alpha})"),
        }
    }
}

/// Poly-logarithmic memory `p_n = ceil(c · log n)`, clamped to `[2, n/4]`. This is the window
/// size covered by the uniform law-of-the-iterated-logarithm bounds for sample autocovariances
/// (Hannan & Deistler 1988, Thm 5.3.2; Bauer 2001, Lemma 1), hence a candidate default
/// pending theory work. `c = 2` gives 9 at n=100, 14 at n=1000, 19 at n=10⁴.
pub fn polylog_memory(n: usize, c: f64) -> usize {
    let p = (c * (n.max(3) as f64).ln()).ceil() as usize;
    p.clamp(2, (n / 4).max(2))
}

/// Default growing memory `p_n = floor(n^0.4)`, clamped to `[2, n/4]`.
/// Satisfies `p_n²/n → 0` (Hwang) but not `p_n³/n → 0` (Berk 1974 / Lewis–Reinsel 1985
/// uniform results); `n^0.3` would satisfy both. Revisit after the benchmark (D-06).
pub fn default_memory(n: usize) -> usize {
    let p = (n as f64).powf(0.4).floor() as usize;
    p.clamp(2, (n / 4).max(2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn invalid_constants_rejected() {
        assert!(Threshold::BicLike { c: -1.0 }.validate().is_err());
        assert!(Threshold::BicLike { c: 0.0 }.validate().is_err());
        assert!(Threshold::Lil { c: f64::NAN }.validate().is_err());
        assert!(Threshold::Fixed(-5.0).validate().is_err());
        assert!(Threshold::Fixed(1.96).validate().is_ok());
        assert!(Threshold::MaxOfK { c: 0.0 }.validate().is_ok());
        assert!(Threshold::Gumbel { alpha: 0.0 }.validate().is_err());
        assert!(Threshold::Gumbel { alpha: 1.0 }.validate().is_err());
        assert!(Threshold::Gumbel { alpha: 0.05 }.validate().is_ok());
    }

    #[test]
    fn gumbel_threshold_is_sane() {
        // For p=10 lags at alpha=0.05 the Bonferroni value is Φ^{-1}(1-0.05/20) ≈ 2.81;
        // the Gumbel approximation should be in the same neighbourhood, and increase with p
        // and with smaller alpha.
        let t10 = Threshold::Gumbel { alpha: 0.05 }.value(1000, 10);
        assert!(t10 > 2.3 && t10 < 3.3, "{t10}");
        assert!(Threshold::Gumbel { alpha: 0.05 }.value(1000, 100) > t10);
        assert!(Threshold::Gumbel { alpha: 0.001 }.value(1000, 10) > t10);
    }

    #[test]
    fn polylog_memory_grows_slowly() {
        assert_eq!(polylog_memory(100, 2.0), 10);
        assert!(polylog_memory(10_000, 2.0) < default_memory(10_000));
        assert!(polylog_memory(10, 2.0) >= 2);
    }
}
