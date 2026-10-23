//! Robust scale and autocovariance estimation.
//!
//! * [`qn`] — the Q_n scale estimator of Rousseeuw & Croux (1993), computed by
//!   the O(n log n) algorithm of Croux & Rousseeuw (1992). Conventions
//!   (consistency constant and finite-sample corrections) follow R's
//!   `robustbase::Qn`, so the two agree to full double precision; note the
//!   published papers print the constant as 2.2219, a digit transposition of
//!   the correct 1/(sqrt(2) Φ⁻¹(5/8)) = 2.21914… (Ma & Genton 2000 print
//!   2.2191).
//! * [`robust_autocovariance`] — the Q_n-based autocovariance of Ma & Genton
//!   (2000, eq. 4): `γ̂_Q(h) = ¼ [Q²_{n−h}(u+v) − Q²_{n−h}(u−v)]` with
//!   `u = (X_1..X_{n−h})`, `v = (X_{h+1}..X_n)`. Breakdown point 25 %, the
//!   maximum possible for an autocovariance estimator.
//! * [`shrink_to_psd`] — γ̂_Q is not guaranteed positive semi-definite; this
//!   applies the shrinking repair (Rousseeuw & Molenberghs 1993): scale the
//!   off-diagonal lags by the largest `1−λ` that makes the Toeplitz matrix
//!   PSD (checked by Cholesky, found by bisection). λ = 0 on clean data.

use crate::linalg::SquareMatrix;
use crate::{check_input, Error, Result};

/// Consistency constant: 1/(sqrt(2) Φ⁻¹(5/8)), rounded as in `robustbase`.
const QN_CONSTANT: f64 = 2.21914;

/// Q_n scale estimator with Gaussian consistency constant and the
/// finite-sample corrections used by `robustbase::Qn` (Croux & Rousseeuw 1992,
/// updated constants). `Q_n = d_n · 2.21914 · {|x_i − x_j|; i<j}_(k)` with
/// `k = C(h,2)`, `h = ⌊n/2⌋+1`. Returns 0 for n < 2.
pub fn qn(x: &[f64]) -> Result<f64> {
    let n = x.len();
    if n < 2 {
        return Ok(0.0);
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(Error::NonFinite);
    }
    let raw = qn_raw(x);
    // Finite-sample corrections, robustbase conventions.
    let corr = match n {
        2 => 0.399356,
        3 => 0.99365,
        4 => 0.51321,
        5 => 0.84401,
        6 => 0.6122,
        7 => 0.85877,
        8 => 0.66993,
        9 => 0.87344,
        10 => 0.72014,
        11 => 0.88906,
        12 => 0.75743,
        _ => {
            let nf = n as f64;
            let c = if n % 2 == 1 {
                1.60188 + (-2.1284 - 5.172 / nf) / nf
            } else {
                3.67561 + (1.9654 + (6.987 - 77.0 / nf) / nf) / nf
            };
            1.0 / (c / nf + 1.0)
        }
    };
    Ok(corr * QN_CONSTANT * raw)
}

/// The raw k-th order statistic of the pairwise distances, `k = C(h,2)`,
/// computed in O(n log n) time and O(n) space (Croux & Rousseeuw 1992, after
/// Johnson & Mizoguchi 1978): select inside the implicit matrix
/// `U(i,j) = y_(i) − y_(n+1−j)` of the sorted values, narrowing per-row
/// candidate windows against weighted-median trial values.
/// Indices below are 1-based to mirror the published algorithm.
fn qn_raw(x: &[f64]) -> f64 {
    let n = x.len();
    let h = n / 2 + 1;
    let k = h * (h - 1) / 2;
    let mut y = vec![0.0]; // y[1..=n] = sorted x
    {
        let mut s = x.to_vec();
        s.sort_unstable_by(f64::total_cmp);
        y.extend_from_slice(&s);
    }
    let mut left = vec![0usize; n + 1];
    let mut right = vec![0usize; n + 1];
    for i in 1..=n {
        left[i] = n - i + 2; // row 1 starts empty (left = n+1)
        right[i] = n;
    }
    let nonpos = n * (n + 1) / 2; // entries on/above the anti-diagonal are <= 0
    let knew = k + nonpos;
    let mut nl = nonpos;
    let mut nr = n * n;
    let mut work: Vec<f64> = Vec::with_capacity(n);
    let mut weight: Vec<i64> = Vec::with_capacity(n);
    let mut p = vec![0usize; n + 1];
    let mut q = vec![0usize; n + 1];
    while nr - nl > n {
        // Trial value: weighted high median of the rows' middle candidates.
        work.clear();
        weight.clear();
        for i in 2..=n {
            if left[i] <= right[i] {
                let w = right[i] - left[i] + 1;
                let jh = left[i] + w / 2;
                work.push(y[i] - y[n + 1 - jh]);
                weight.push(w as i64);
            }
        }
        let trial = weighted_high_median(&mut work, &mut weight);
        // p[i] = #{j : U(i,j) < trial}; staircase merge over the sorted rows,
        // the j counter persisting across rows (that is what makes it O(n)).
        let mut j = 0usize;
        for i in (1..=n).rev() {
            while j < n && y[i] - y[n - j] < trial {
                j += 1;
            }
            p[i] = j;
        }
        // q[i] − 1 = #{j : U(i,j) <= trial}.
        let mut j = n + 1;
        for i in 1..=n {
            while j > 2 && y[i] - y[n + 2 - j] > trial {
                j -= 1;
            }
            q[i] = j;
        }
        let sum_p: usize = p[1..].iter().sum();
        let sum_q: usize = q[1..].iter().map(|v| v - 1).sum();
        if knew <= sum_p {
            right[1..=n].copy_from_slice(&p[1..=n]);
            nr = sum_p;
        } else if knew > sum_q {
            left[1..=n].copy_from_slice(&q[1..=n]);
            nl = sum_q;
        } else {
            return trial;
        }
    }
    // Few enough candidates left: gather and select directly.
    work.clear();
    for i in 2..=n {
        for jj in left[i]..=right[i] {
            work.push(y[i] - y[n - jj + 1]);
        }
    }
    select_kth(&mut work, knew - nl - 1)
}

/// k-th smallest (0-based) by sorting; candidate sets here are O(n).
fn select_kth(a: &mut [f64], k: usize) -> f64 {
    a.sort_unstable_by(f64::total_cmp);
    a[k]
}

/// Weighted high median: the smallest `a[j]` such that the total weight of
/// `{a[i] <= a[j]}` exceeds half the total weight. Consumes its inputs.
fn weighted_high_median(a: &mut Vec<f64>, w: &mut Vec<i64>) -> f64 {
    let wtotal: i64 = w.iter().sum();
    let mut wrest: i64 = 0;
    loop {
        let nn = a.len();
        let mut buf = a.clone();
        let trial = select_kth(&mut buf, nn / 2); // high median of candidates
        let mut wleft = 0i64;
        let mut wmid = 0i64;
        for (v, wi) in a.iter().zip(w.iter()) {
            if *v < trial {
                wleft += wi;
            } else if *v == trial {
                wmid += wi;
            }
        }
        if 2 * (wrest + wleft) > wtotal {
            // Answer lies strictly below the trial.
            let mut ka = 0;
            for i in 0..nn {
                if a[i] < trial {
                    a[ka] = a[i];
                    w[ka] = w[i];
                    ka += 1;
                }
            }
            a.truncate(ka);
            w.truncate(ka);
        } else if 2 * (wrest + wleft + wmid) > wtotal {
            return trial;
        } else {
            let mut ka = 0;
            for i in 0..nn {
                if a[i] > trial {
                    a[ka] = a[i];
                    w[ka] = w[i];
                    ka += 1;
                }
            }
            a.truncate(ka);
            w.truncate(ka);
            wrest += wleft + wmid;
        }
    }
}

/// Ma & Genton (2000, eq. 4) Q_n-based autocovariances for lags `0..=max_lag`:
/// `γ̂_Q(0) = Q²_n(x)` and `γ̂_Q(h) = ¼ [Q²_{n−h}(u+v) − Q²_{n−h}(u−v)]` with
/// `u = (X_1..X_{n−h})`, `v = (X_{h+1}..X_n)`. Location-invariant and scale-
/// equivariant; 25 % temporal breakdown point. **Not** guaranteed PSD — pass
/// the result through [`shrink_to_psd`] before Levinson/Yule–Walker use.
///
/// Caveat observed in simulation: an outlier in `x_t` contaminates the pair
/// vectors at every lag `h ≥ 1` but only one entry at lag 0, so under additive
/// outliers the *ratio* `γ̂_Q(h)/γ̂_Q(0)` is biased upward at every lag, which
/// easily breaks positive definiteness. For autocorrelations prefer
/// [`robust_autocorrelation`], whose shared denominator cancels that bias.
pub fn robust_autocovariance(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    check_input(x, max_lag + 2)?;
    let n = x.len();
    let mut r = Vec::with_capacity(max_lag + 1);
    let q0 = qn(x)?;
    r.push(q0 * q0);
    let mut s = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for h in 1..=max_lag {
        s.clear();
        d.clear();
        for i in 0..n - h {
            s.push(x[i] + x[i + h]);
            d.push(x[i] - x[i + h]);
        }
        let qs = qn(&s)?;
        let qd = qn(&d)?;
        r.push(0.25 * (qs * qs - qd * qd));
    }
    Ok(r)
}

/// Ma & Genton's (2000) robust autocorrelations for lags `0..=max_lag`:
/// `ρ̂_Q(h) = [Q²(u+v) − Q²(u−v)] / [Q²(u+v) + Q²(u−v)]`, bounded in [−1, 1]
/// by construction and independent of the consistency constant. Numerator and
/// denominator share the same contaminated pairs, so the additive-outlier bias
/// that affects `γ̂_Q(h)/γ̂_Q(0)` largely cancels (this is what
/// [`crate::Statistic::PacfRobust`] feeds into Levinson).
pub fn robust_autocorrelation(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    check_input(x, max_lag + 2)?;
    let n = x.len();
    let mut r = Vec::with_capacity(max_lag + 1);
    r.push(1.0);
    let mut s = Vec::with_capacity(n);
    let mut d = Vec::with_capacity(n);
    for h in 1..=max_lag {
        s.clear();
        d.clear();
        for i in 0..n - h {
            s.push(x[i] + x[i + h]);
            d.push(x[i] - x[i + h]);
        }
        let qs = qn(&s)?;
        let qd = qn(&d)?;
        let denom = qs * qs + qd * qd;
        if denom <= 0.0 {
            return Err(Error::Singular);
        }
        r.push((qs * qs - qd * qd) / denom);
    }
    Ok(r)
}

/// Shrinking repair for a possibly non-PSD autocovariance sequence
/// (Rousseeuw & Molenberghs 1993): return `r'` with `r'(0) = r(0)` and
/// `r'(h) = (1−λ) r(h)`, λ the smallest value in [0, 1] (up to bisection
/// tolerance) such that the Toeplitz matrix of `r'` stays positive definite
/// with a relative ridge margin of `1e-3 · r(0)` — the margin keeps estimated
/// sequences away from the PD boundary, where Levinson would report spurious
/// reflection coefficients near ±1. On a sequence that already satisfies the
/// margin this returns the input unchanged (λ = 0).
pub fn shrink_to_psd(r: &[f64]) -> Result<Vec<f64>> {
    if r.is_empty() || r[0] <= 0.0 {
        return Err(Error::Singular);
    }
    let ridge = 1e-3 * r[0];
    let is_pd = |lam: f64| -> bool {
        let m = r.len();
        let mut g = SquareMatrix::zeros(m);
        for i in 0..m {
            for j in 0..m {
                let v = if i == j {
                    r[0] - ridge
                } else {
                    (1.0 - lam) * r[i.abs_diff(j)]
                };
                g.set(i, j, v);
            }
        }
        g.cholesky().is_ok()
    };
    if is_pd(0.0) {
        return Ok(r.to_vec());
    }
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..40 {
        let mid = 0.5 * (lo + hi);
        if is_pd(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let lam = hi.min(1.0);
    let mut out = r.to_vec();
    for v in out.iter_mut().skip(1) {
        *v *= 1.0 - lam;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)] // literals carry R's full printed precision
    fn qn_matches_robustbase_fixtures() {
        // Reference values from R: robustbase::Qn (set.seed(1); rnorm(10)).
        let x = [
            -0.62645381074233242,
            0.18364332422208227,
            -0.83562861241004716,
            1.59528080213779155,
            0.32950777181536051,
            -0.82046838411801548,
            0.48742905242848528,
            0.73832470512921722,
            0.57578135165349209,
            -0.30538838715635602,
        ];
        let q = qn(&x).unwrap();
        assert!((q - 0.82314495445682179).abs() < 1e-14, "qn = {q}");
        // 1:8 — raw order statistic is exactly 2, uncorrected Qn = 2 * 2.21914.
        let y: Vec<f64> = (1..=8).map(|v| v as f64).collect();
        assert!((qn_raw(&y) - 2.0).abs() < 1e-15);
        let q8 = qn(&y).unwrap();
        assert!((q8 - 2.9733369204).abs() < 1e-9, "qn(1:8) = {q8}");
        // n = 200 with three gross outliers.
        let mut z: Vec<f64> = (1..=200).map(|v| v as f64).collect();
        z[9] = 1000.0;
        z[49] = -500.0;
        z[119] = 800.0;
        let q200 = qn(&z).unwrap();
        assert!((q200 - 63.190583283547305).abs() < 1e-10, "qn = {q200}");
    }

    #[test]
    fn qn_brute_force_agreement_and_equivariance() {
        // O(n log n) path vs direct order statistic on pseudo-random data.
        let mut s = 7u64;
        let x: Vec<f64> = (0..101)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64
            })
            .collect();
        let n = x.len();
        let h = n / 2 + 1;
        let k = h * (h - 1) / 2;
        let mut dists = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                dists.push((x[i] - x[j]).abs());
            }
        }
        dists.sort_unstable_by(f64::total_cmp);
        assert!((qn_raw(&x) - dists[k - 1]).abs() < 1e-15);
        // Scale equivariance and shift invariance.
        let y: Vec<f64> = x.iter().map(|v| 3.0 * v + 50.0).collect();
        let (qx, qy) = (qn(&x).unwrap(), qn(&y).unwrap());
        assert!((3.0 * qx - qy).abs() < 1e-12 * qy);
    }

    #[test]
    fn robust_acov_ignores_gross_outliers() {
        // AR(1)-ish deterministic check: on clean data γ̂_Q(1)/γ̂_Q(0) should be
        // positive and sizeable; adding gross outliers must not destroy it.
        let mut s = 3u64;
        let mut x = vec![0.0f64; 2000];
        for t in 1..2000 {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            x[t] = 0.7 * x[t - 1] + u;
        }
        let clean = robust_autocovariance(&x, 3).unwrap();
        let rho_clean = clean[1] / clean[0];
        let mut dirty = x.clone();
        for t in (0..2000).step_by(97) {
            dirty[t] += if t % 2 == 0 { 50.0 } else { -50.0 };
        }
        let rob = robust_autocovariance(&dirty, 3).unwrap();
        let rho_rob = rob[1] / rob[0];
        assert!(
            (rho_rob - rho_clean).abs() < 0.1,
            "{rho_clean} vs {rho_rob}"
        );
    }

    #[test]
    fn shrink_leaves_psd_untouched_and_repairs_non_psd() {
        let ok = [1.0, 0.5, 0.25];
        assert_eq!(shrink_to_psd(&ok).unwrap(), ok.to_vec());
        // r(1) = 0.9, r(2) = -0.9 is not a valid autocovariance sequence.
        let bad = [1.0, 0.9, -0.9];
        let fixed = shrink_to_psd(&bad).unwrap();
        assert!(fixed[1] < 0.9 && fixed[1] > 0.0);
        let l = crate::levinson::levinson(&fixed).unwrap();
        for k in &l.pacf {
            assert!(k.abs() < 1.0, "pacf {k}");
        }
    }
}
