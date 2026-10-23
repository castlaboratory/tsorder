//! Distribution functions for the rank tests: chi-square survival and the
//! law of a weighted sum of independent χ²₁ variables (the null of the
//! Baram–Porat / Robin–Smith statistics), by Imhof's (1961) exact inversion
//! or Satterthwaite's two-moment approximation.
//!
//! Dependency-free: log-gamma by Lanczos, regularized incomplete gamma by
//! series/continued fraction (Numerical Recipes conventions), Imhof by
//! composite Simpson with an oscillation-aware step.

/// Natural log of the gamma function (Lanczos, g = 7, ~15 digits).
#[allow(clippy::excessive_precision)] // published Lanczos coefficients, kept verbatim
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        // Reflection.
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let mut a = G[0];
    let t = x + 7.5;
    for (i, &g) in G.iter().enumerate().skip(1) {
        a += g / (x + i as f64);
    }
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

/// Regularized upper incomplete gamma `Q(a, x) = Γ(a, x)/Γ(a)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x < 0.0 || a <= 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        // 1 - P(a, x) by series.
        let mut sum = 1.0 / a;
        let mut term = sum;
        let mut ap = a;
        for _ in 0..500 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        1.0 - sum * (-x + a * x.ln() - ln_gamma(a)).exp()
    } else {
        // Continued fraction (Lentz).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / 1e-300;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < 1e-300 {
                d = 1e-300;
            }
            c = b + an / c;
            if c.abs() < 1e-300 {
                c = 1e-300;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        (-x + a * x.ln() - ln_gamma(a)).exp() * h
    }
}

/// Chi-square survival function `P(χ²_df > q)` (df may be non-integer).
pub fn chi2_sf(q: f64, df: f64) -> f64 {
    gamma_q(df / 2.0, q / 2.0)
}

/// Satterthwaite two-moment approximation to `P(Σ λ_i χ²₁ > q)`:
/// match mean and variance with `g · χ²_ν`, `g = Σλ²/Σλ`, `ν = (Σλ)²/Σλ²`.
pub fn satterthwaite_sf(q: f64, weights: &[f64]) -> f64 {
    let s1: f64 = weights.iter().sum();
    let s2: f64 = weights.iter().map(|l| l * l).sum();
    if s1 <= 0.0 || s2 <= 0.0 {
        return if q > 0.0 { 0.0 } else { 1.0 };
    }
    chi2_sf(q * s1 / s2, s1 * s1 / s2)
}

/// Imhof's (1961) exact inversion for `P(Σ λ_i χ²₁ > q)` (central case):
/// `P = 1/2 + (1/π) ∫₀^∞ sin θ(u) / (u ρ(u)) du` with
/// `θ(u) = ½ Σ atan(λ_i u) − ½ q u`, `ρ(u) = Π (1 + λ_i² u²)^{1/4}`.
pub fn imhof_sf(q: f64, weights: &[f64]) -> f64 {
    let lam: Vec<f64> = weights.iter().copied().filter(|l| *l > 0.0).collect();
    if lam.is_empty() {
        return if q > 0.0 { 0.0 } else { 1.0 };
    }
    let sum_l: f64 = lam.iter().sum();
    let integrand = |u: f64| -> f64 {
        if u == 0.0 {
            // limit: θ(u)/u → (Σλ − q)/2, ρ → 1.
            return 0.5 * (sum_l - q);
        }
        let mut theta = -0.5 * q * u;
        let mut ln_rho = 0.0;
        for &l in &lam {
            theta += 0.5 * (l * u).atan();
            ln_rho += 0.25 * (1.0 + l * l * u * u).ln();
        }
        theta.sin() / (u * (ln_rho).exp())
    };
    // Oscillation-aware step: the phase moves at ≈ (q + Σλ)/2 per unit u.
    let rate = 0.5 * (q + sum_l) + 1.0;
    let h = (std::f64::consts::PI / rate / 20.0).min(0.05);
    let mut total = 0.0;
    let mut u = 0.0;
    let mut small_blocks = 0;
    for _ in 0..200_000 {
        // Simpson on [u, u + 2h].
        let block = h / 3.0 * (integrand(u) + 4.0 * integrand(u + h) + integrand(u + 2.0 * h));
        total += block;
        u += 2.0 * h;
        if block.abs() < 1e-13 {
            small_blocks += 1;
            if small_blocks > 50 {
                break;
            }
        } else {
            small_blocks = 0;
        }
    }
    (0.5 + total / std::f64::consts::PI).clamp(0.0, 1.0)
}

/// Upper-tail quantile table of the asymptotic KPSS level-statistic null
/// (integral of a squared Brownian bridge): pairs `(α, q)` with
/// `P(KPSS > q) = α`. Simulated by `examples/unit_root_tables.rs`
/// (T = 5000, 400 000 replications); matches the published Kwiatkowski et
/// al. (1992) points 0.347/0.463/0.739 at 10 %/5 %/1 % to the expected
/// discretisation accuracy.
const KPSS_TABLE: [(f64, f64); 12] = [
    (0.0005, 1.3136),
    (0.001, 1.1739),
    (0.0025, 1.0087),
    (0.005, 0.8769),
    (0.01, 0.7521),
    (0.025, 0.5837),
    (0.05, 0.4635),
    (0.10, 0.3477),
    (0.15, 0.2847),
    (0.20, 0.2421),
    (0.30, 0.1842),
    (0.50, 0.1189),
];

/// Lower-tail quantile table of the asymptotic Dickey–Fuller `τ_c` null
/// (t-statistic with constant under a pure random walk): pairs `(α, q)` with
/// `P(τ < q) = α`. Same generator; matches the standard asymptotic points
/// −2.57/−2.86/−3.43 at 10 %/5 %/1 %.
const DF_TAU_C_TABLE: [(f64, f64); 12] = [
    (0.0005, -4.2971),
    (0.001, -4.1176),
    (0.0025, -3.8743),
    (0.005, -3.6730),
    (0.01, -3.4508),
    (0.025, -3.1363),
    (0.05, -2.8713),
    (0.10, -2.5726),
    (0.15, -2.3749),
    (0.20, -2.2203),
    (0.30, -1.9725),
    (0.50, -1.5691),
];

fn interp_table(table: &[(f64, f64)], alpha: f64) -> f64 {
    let a = alpha.clamp(table[0].0, table[table.len() - 1].0);
    for w in table.windows(2) {
        let ((a0, q0), (a1, q1)) = (w[0], w[1]);
        if a <= a1 {
            // Log-linear in alpha (tail probabilities span decades).
            let t = (a.ln() - a0.ln()) / (a1.ln() - a0.ln());
            return q0 + t * (q1 - q0);
        }
    }
    table[table.len() - 1].1
}

/// Critical value `q(α)` with `P(KPSS > q) = α` for the level-stationarity
/// KPSS statistic (asymptotic null).
pub fn kpss_critical(alpha: f64) -> f64 {
    interp_table(&KPSS_TABLE, alpha)
}

/// Critical value `q(α)` with `P(τ_c < q) = α` for the Dickey–Fuller
/// t-statistic with constant (asymptotic null).
pub fn df_tau_c_critical(alpha: f64) -> f64 {
    interp_table(&DF_TAU_C_TABLE, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chi2_sf_matches_r_pchisq() {
        // R: pchisq(q, df, lower.tail = FALSE).
        let cases = [
            (3.84, 1.0, 0.050043521248705),
            (5.99, 2.0, 0.0500366270865863),
            (2.5, 1.7, 0.231036876696032),
            (10.0, 4.3, 0.0494046995154264),
            (0.5, 0.5, 0.256322055268539),
        ];
        for (q, df, want) in cases {
            let got = chi2_sf(q, df);
            assert!((got - want).abs() < 1e-12, "chi2_sf({q},{df}) = {got}");
        }
    }

    #[test]
    fn imhof_matches_exact_chi2_and_r_compquadform() {
        // Single unit weight: exactly chi-square(1).
        let got = imhof_sf(3.84, &[1.0]);
        assert!((got - chi2_sf(3.84, 1.0)).abs() < 1e-6, "{got}");
        // Equal weights 1,1: exactly chi-square(2).
        let got = imhof_sf(5.99, &[1.0, 1.0]);
        assert!((got - chi2_sf(5.99, 2.0)).abs() < 1e-6, "{got}");
        // R CompQuadForm::imhof references (their own integration error ~1e-4).
        let cases: [(f64, &[f64], f64); 4] = [
            (5.0, &[2.0, 1.0], 0.186424873796),
            (1.2, &[0.9, 0.5, 0.1], 0.449453666487),
            (10.0, &[3.0, 2.0, 1.0, 0.5], 0.193055225569),
            (0.3, &[0.2, 0.05], 0.279316877793),
        ];
        for (q, l, want) in cases {
            let got = imhof_sf(q, l);
            assert!((got - want).abs() < 5e-4, "imhof({q}) = {got} want {want}");
        }
    }

    #[test]
    fn satterthwaite_is_exact_for_equal_weights() {
        // Equal weights: the two-moment fit is exact (g = λ, ν = k).
        let got = satterthwaite_sf(5.99, &[1.0, 1.0]);
        assert!((got - chi2_sf(5.99, 2.0)).abs() < 1e-12);
        let got = satterthwaite_sf(3.0, &[0.5, 0.5, 0.5]);
        assert!((got - chi2_sf(6.0, 3.0)).abs() < 1e-12);
        // Unequal weights: close to Imhof but not equal.
        let a = satterthwaite_sf(5.0, &[2.0, 1.0]);
        let b = imhof_sf(5.0, &[2.0, 1.0]);
        assert!((a - b).abs() < 0.02, "satt {a} vs imhof {b}");
    }
}
