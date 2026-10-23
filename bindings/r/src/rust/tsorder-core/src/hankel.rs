//! McMillan degree by singular-value thresholding of the Hankel matrix of
//! sample autocovariances (M3 route 2).
//!
//! The rank of the autocovariance Hankel `H[i][j] = γ(i+j+1)` is the state
//! dimension. A raw SVD cut fails because the threshold must carry the scale
//! of the estimation error (the day-1 finding); following the calibration
//! lesson of Zheng, Jedra & Proutière (2025, arXiv:2506.08720), the cut is
//!
//! `ξ_n = c · ŝ · sqrt(K · log n / n)`,
//!
//! where `ŝ² = 2 Σ_{|h| ≤ 2K} γ̂(h)²` is a Bartlett-style plug-in bound on the
//! per-entry error variance `n·Var(γ̂(h)) ≈ Σ_k γ(k)² + γ(k+h)γ(k−h)`, and
//! `sqrt(K)` is the random-matrix scaling of the operator norm of an
//! entrywise-noise perturbation (the worst-case factor `K` proved hopelessly
//! conservative in calibration: it underestimated everywhere). Recovery
//! additionally needs *separation*: the smallest non-zero singular value of
//! the true Hankel must exceed `ξ_n` — e.g. MA(2) with θ = (0.6, 0.3) has
//! true Hankel singular values ≈ (0.88, 0.10), and 0.10 sits below any
//! honest threshold at practical `n`; that is the problem being hard, not
//! the threshold failing.

use crate::acov::autocovariance;
use crate::dist::{imhof_sf, satterthwaite_sf};
use crate::linalg::{singular_values, SquareMatrix};
use crate::threshold::polylog_memory;
use crate::Result;

/// Configuration of [`hankel_svt_order`].
#[derive(Debug, Clone, Copy)]
pub struct HankelSvtConfig {
    /// Hankel dimension `K`; `None` → `⌈2 log n⌉` ([`polylog_memory`]).
    pub k: Option<usize>,
    /// Threshold constant `c`. Empirical: at the default 1.0 the calibration
    /// runs gave 100 % correct rank 0 on white noise, ~100 % on Bauer's
    /// Example I, 96–98 % on AR(1) φ=0.9, ~90 % on ARMA(1,1) — and MA(2)
    /// fails on separation, as expected.
    pub c: f64,
}

impl Default for HankelSvtConfig {
    fn default() -> Self {
        HankelSvtConfig { k: None, c: 1.0 }
    }
}

/// Output of [`hankel_svt_order`].
#[derive(Debug, Clone)]
pub struct HankelSvtResult {
    /// Estimated McMillan degree: number of singular values above `xi`.
    pub order: usize,
    /// Singular values of the autocovariance Hankel, descending.
    pub singular_values: Vec<f64>,
    /// The threshold `ξ_n` actually used.
    pub xi: f64,
    pub k: usize,
}

/// Singular-value-thresholding estimator of the McMillan degree.
pub fn hankel_svt_order(x: &[f64], config: HankelSvtConfig) -> Result<HankelSvtResult> {
    let n = x.len();
    let k = config.k.unwrap_or_else(|| polylog_memory(n, 2.0)).max(2);
    let r = autocovariance(x, 2 * k)?;
    let mut h = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            h[i * k + j] = r[i + j + 1];
        }
    }
    let sv = singular_values(&h, k, k);
    let s2 = 2.0 * r.iter().map(|g| g * g).sum::<f64>();
    let xi = config.c * s2.sqrt() * ((k as f64) * (n as f64).ln() / n as f64).sqrt();
    let order = sv.iter().take_while(|s| **s > xi).count();
    Ok(HankelSvtResult {
        order,
        singular_values: sv,
        xi,
        k,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_ar1_hankel_has_rank_one_structure() {
        // Exact AR(1) autocovariances: Hankel is rank 1, so σ_2 = 0 exactly.
        let phi: f64 = 0.8;
        let n = 10;
        let r: Vec<f64> = (0..=2 * n)
            .map(|h| phi.powi(h as i32) / (1.0 - phi * phi))
            .collect();
        let mut h = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                h[i * n + j] = r[i + j + 1];
            }
        }
        let sv = singular_values(&h, n, n);
        assert!(sv[0] > 1.0);
        assert!(sv[1].abs() < 1e-10, "{sv:?}");
    }
}

/// How the sequential rank test sets its level at sample size `n`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RankAlpha {
    /// Fixed level (the rank estimate then over-rejects with probability α forever).
    Fixed(f64),
    /// `α_n = κ / log n` with `κ = 0.05·log 50` (so `α_50 = 0.05`) — the
    /// Cragg–Donald rule used by Camba-Méndez & Kapetanios; satisfies the
    /// Hosoya consistency condition `α_n → 0`, `log(α_n)/n → 0`.
    LogDecay,
}

/// How the weighted-χ²₁ null p-value is computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankPvalue {
    /// Imhof's exact inversion (default).
    Imhof,
    /// Satterthwaite's two-moment approximation (faster, slightly rougher).
    Satterthwaite,
}

/// Configuration of [`hankel_rank_order`].
#[derive(Debug, Clone, Copy)]
pub struct HankelRankConfig {
    /// Hankel dimensions `k = p`; `None` → `max(3, ⌈log n⌉)`. Keep it small:
    /// the null-weight covariance grows with `k + p` and is poorly estimated
    /// when large (the lesson of Camba-Méndez & Kapetanios's `k = 4` runs).
    pub k: Option<usize>,
    pub alpha: RankAlpha,
    pub pvalue: RankPvalue,
}

impl Default for HankelRankConfig {
    fn default() -> Self {
        HankelRankConfig {
            k: None,
            alpha: RankAlpha::LogDecay,
            pvalue: RankPvalue::Imhof,
        }
    }
}

/// Output of [`hankel_rank_order`].
#[derive(Debug, Clone)]
pub struct HankelRankResult {
    /// Estimated McMillan degree: first `r` whose test does not reject.
    pub order: usize,
    /// P-value of H₀: rank ≤ r, for r = 0..=order (the sequence actually run).
    pub pvalues: Vec<f64>,
    /// The level `α_n` used.
    pub alpha: f64,
    pub k: usize,
}

/// Sequential rank test on the autocovariance Hankel: Robin & Smith's (2000)
/// characteristic-root statistic `CRT_r = n Σ_{i>r} σ̂_i²` (their Wald form,
/// with identity weighting, so the roots are the squared singular values of
/// the Hankel), with the null law `Σ λ_i χ²₁` of their Theorem 3.2. The
/// weights are the eigenvalues of `(D ⊗ C)' Ω (D ⊗ C)` with `C, D` the
/// null-space singular vectors; the Hankel's duplication structure reduces
/// this to the small matrix `(L' W)' (L' W)` where `Ξ = L L'` is the
/// Bartlett (Gaussian) covariance of the distinct sample autocovariances and
/// `W[ℓ, (a,b)] = Σ_{i+j+1=ℓ} C[i,a] D[j,b]`. Stops at the first
/// non-rejection (Camba-Méndez & Kapetanios's sequence, started at r = 0);
/// with `RankAlpha::LogDecay` the estimate is consistent (Hosoya 1989).
///
/// Combining Robin–Smith's statistic with the autocovariance Hankel is this
/// crate's assembly of published pieces (CK use Cragg–Donald's test instead);
/// the lineage is Baram & Porat (1988) → Robin & Smith (2000).
pub fn hankel_rank_order(x: &[f64], config: HankelRankConfig) -> Result<HankelRankResult> {
    let n = x.len();
    let k = config
        .k
        .unwrap_or_else(|| polylog_memory(n, 1.0).max(3))
        .max(2);
    let lags = 2 * k - 1; // distinct autocovariance lags 1..=2k-1
    let big = 2 * lags; // window for the Bartlett formula plug-in
    let r = autocovariance(x, big.min(n / 3))?;
    let gamma = |h: i64| -> f64 {
        let a = h.unsigned_abs() as usize;
        if a < r.len() {
            r[a]
        } else {
            0.0
        }
    };
    // Hankel H[i][j] = γ̂(i+j+1), k × k.
    let mut h = vec![0.0; k * k];
    for i in 0..k {
        for j in 0..k {
            h[i * k + j] = r[i + j + 1];
        }
    }
    // Bartlett covariance of the distinct lags: Ξ(g,h) =
    // Σ_u γ(u)γ(u+g−h) + γ(u+g)γ(u−h), truncated at the available window.
    let u_max = (r.len() - 1) as i64;
    let mut xi = SquareMatrix::zeros(lags);
    for a in 1..=lags {
        for b in a..=lags {
            let (ga, gb) = (a as i64, b as i64);
            let mut s = 0.0;
            for u in -u_max..=u_max {
                s += gamma(u) * gamma(u + gb - ga) + gamma(u + gb) * gamma(u - ga);
            }
            xi.set(a - 1, b - 1, s);
            xi.set(b - 1, a - 1, s);
        }
    }
    // Cholesky of Ξ, with a small ridge if the truncated plug-in is not PD.
    let lxi = match xi.cholesky() {
        Ok(l) => l,
        Err(_) => {
            let ridge = 1e-8 * (1..=lags).map(|i| xi.get(i - 1, i - 1)).sum::<f64>() / lags as f64;
            for i in 0..lags {
                let v = xi.get(i, i) + ridge;
                xi.set(i, i, v);
            }
            xi.cholesky()?
        }
    };
    // Full SVD data: we need singular values and both singular-vector sets.
    // One-sided Jacobi on H gives V (the rotations) and U = HV/σ; for these
    // small matrices recompute vectors by orthogonalising columns of H'H and
    // HH' via the same Jacobi pass is overkill — use the simple route:
    // eigenvectors of H'H (right) and HH' (left) via Jacobi eigen below.
    let (sv, u_vecs, v_vecs) = svd_small(&h, k);
    let alpha = match config.alpha {
        RankAlpha::Fixed(a) => a,
        RankAlpha::LogDecay => 0.05 * 50f64.ln() / (n as f64).ln().max(1.0),
    };
    let mut pvalues = Vec::new();
    let mut order = k; // if every stage rejects
    for rr in 0..k {
        let crt: f64 = (rr..k).map(|i| sv[i] * sv[i]).sum::<f64>() * n as f64;
        // W[ℓ, (a,b)] over null vectors a, b = rr..k-1.
        let m = k - rr;
        let mut w = vec![0.0; lags * m * m];
        for (ai, a) in (rr..k).enumerate() {
            for (bi, b) in (rr..k).enumerate() {
                let cidx = ai * m + bi;
                for i in 0..k {
                    for j in 0..k {
                        let l = i + j + 1;
                        w[(l - 1) * m * m + cidx] += u_vecs[i * k + a] * v_vecs[j * k + b];
                    }
                }
            }
        }
        // Weights = squared singular values of L' W (lags × m²).
        let mut lw = vec![0.0; lags * m * m];
        for row in 0..lags {
            for c in 0..m * m {
                let mut s = 0.0;
                for t in row..lags {
                    s += lxi.get(t, row) * w[t * m * m + c];
                }
                lw[row * m * m + c] = s;
            }
        }
        let wsv = singular_values(&lw, lags, m * m);
        let weights: Vec<f64> = wsv.iter().map(|s| s * s).filter(|v| *v > 1e-12).collect();
        let p = match config.pvalue {
            RankPvalue::Imhof => imhof_sf(crt, &weights),
            RankPvalue::Satterthwaite => satterthwaite_sf(crt, &weights),
        };
        pvalues.push(p);
        if p >= alpha {
            order = rr;
            break;
        }
    }
    Ok(HankelRankResult {
        order,
        pvalues,
        alpha,
        k,
    })
}

/// SVD of a small dense `k × k` matrix: singular values (descending) plus
/// left and right singular vectors (column-major: `u[i*k + a]` is component
/// `i` of the `a`-th left vector). Jacobi eigendecomposition of `HᵀH` gives
/// `V` and `σ²`; `U = H V / σ`, completed by Gram–Schmidt for null columns.
fn svd_small(h: &[f64], k: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // G = HᵀH.
    let mut g = SquareMatrix::zeros(k);
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0;
            for t in 0..k {
                s += h[t * k + i] * h[t * k + j];
            }
            g.set(i, j, s);
        }
    }
    // Cyclic Jacobi eigendecomposition of the symmetric G.
    let mut a = g;
    let mut v = SquareMatrix::zeros(k);
    for i in 0..k {
        v.set(i, i, 1.0);
    }
    for _ in 0..100 {
        let mut off = 0.0;
        for i in 0..k {
            for j in i + 1..k {
                off += a.get(i, j).abs();
            }
        }
        if off < 1e-13 {
            break;
        }
        for i in 0..k {
            for j in i + 1..k {
                let apq = a.get(i, j);
                if apq.abs() < 1e-300 {
                    continue;
                }
                let (app, aqq) = (a.get(i, i), a.get(j, j));
                let zeta = (aqq - app) / (2.0 * apq);
                let t = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;
                for l in 0..k {
                    let (ail, ajl) = (a.get(i, l), a.get(j, l));
                    a.set(i, l, c * ail - s * ajl);
                    a.set(j, l, s * ail + c * ajl);
                }
                for l in 0..k {
                    let (ali, alj) = (a.get(l, i), a.get(l, j));
                    a.set(l, i, c * ali - s * alj);
                    a.set(l, j, s * ali + c * alj);
                }
                for l in 0..k {
                    let (vli, vlj) = (v.get(l, i), v.get(l, j));
                    v.set(l, i, c * vli - s * vlj);
                    v.set(l, j, s * vli + c * vlj);
                }
            }
        }
    }
    // Sort by eigenvalue descending.
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&p, &q| a.get(q, q).total_cmp(&a.get(p, p)));
    let mut sv = Vec::with_capacity(k);
    let mut vv = vec![0.0; k * k];
    for (col, &p) in idx.iter().enumerate() {
        sv.push(a.get(p, p).max(0.0).sqrt());
        for i in 0..k {
            vv[i * k + col] = v.get(i, p);
        }
    }
    // U = H V / σ, Gram–Schmidt completion where σ ≈ 0.
    let mut uu = vec![0.0; k * k];
    for col in 0..k {
        if sv[col] > 1e-12 * sv[0].max(1e-300) {
            for i in 0..k {
                let mut s = 0.0;
                for j in 0..k {
                    s += h[i * k + j] * vv[j * k + col];
                }
                uu[i * k + col] = s / sv[col];
            }
        } else {
            // Any unit vector orthogonal to the previous columns.
            for seed in 0..k {
                for i in 0..k {
                    uu[i * k + col] = if i == seed { 1.0 } else { 0.0 };
                }
                for prev in 0..col {
                    let dot: f64 = (0..k).map(|i| uu[i * k + col] * uu[i * k + prev]).sum();
                    for i in 0..k {
                        uu[i * k + col] -= dot * uu[i * k + prev];
                    }
                }
                let norm: f64 = (0..k).map(|i| uu[i * k + col].powi(2)).sum::<f64>().sqrt();
                if norm > 1e-8 {
                    for i in 0..k {
                        uu[i * k + col] /= norm;
                    }
                    break;
                }
            }
        }
    }
    (sv, uu, vv)
}
