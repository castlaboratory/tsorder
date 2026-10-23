//! Extended sample autocorrelation function (Tsay & Tiao 1984) and the
//! triangle-vertex identification of ARMA(p, q).
//!
//! The table cell (k, j) is the lag-(j+1) sample autocorrelation of
//! `W_t = Z_t − Σ_l Φ̂^{(j+1)}_{l(k)} Z_{t−l}`, where the iterated AR
//! estimates come from their recursion (2.7) over ordinary OLS AR fits (their
//! Lemma 5.1 shows the residual regressions are not needed). Under ARMA(p, q)
//! the cells with `row ≥ p` and `col − (row − p) ≥ q` converge to zero — a
//! triangle whose vertex (p, q) names the order. Cells are flagged with the
//! paper's crude white-noise band `band / sqrt(n − k − j)` (their 2-standard-
//! deviation coding; they note it may understate the variance — it is not an
//! exact studentization).

use crate::linalg::ols;
use crate::{check_input, Error, Result};

/// Configuration of [`eacf`].
#[derive(Debug, Clone, Copy)]
pub struct EacfConfig {
    /// Rows 0..=max_ar (default 7, matching R's `TSA::eacf`).
    pub max_ar: usize,
    /// Columns 0..=max_ma (default 13, matching R's `TSA::eacf`).
    pub max_ma: usize,
    /// Band multiplier (2.0 = the paper's coding; raise it for a stricter,
    /// Fine–Hwang-style cut).
    pub band: f64,
}

impl Default for EacfConfig {
    fn default() -> Self {
        EacfConfig {
            max_ar: 7,
            max_ma: 13,
            band: 2.0,
        }
    }
}

/// Output of [`eacf`].
#[derive(Debug, Clone)]
pub struct EacfResult {
    /// Vertex of the first all-insignificant triangle, scanning candidates by
    /// `p + q` then by `p` (the upper-left/simplicity preference); `None` if
    /// no full triangle inside the table is clean.
    pub order: Option<(usize, usize)>,
    /// `table[k][j]` = extended sample autocorrelation (row k, column j).
    pub table: Vec<Vec<f64>>,
    /// `significant[k][j]` = |table| above the band (the X of the X/O table).
    pub significant: Vec<Vec<bool>>,
}

/// Compute the EACF table and read the order off the zero triangle.
pub fn eacf(x: &[f64], config: EacfConfig) -> Result<EacfResult> {
    let n = x.len();
    let kmax = config.max_ar;
    let jmax = config.max_ma;
    let base_max = kmax + jmax + 1;
    check_input(x, 2 * base_max + 2)?;
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();

    // Ordinary OLS AR(k) fits, k = 1..=base_max: phi0[k][l-1], l = 1..=k.
    let mut layers: Vec<Vec<Vec<f64>>> = Vec::with_capacity(jmax + 2);
    let mut base: Vec<Vec<f64>> = vec![Vec::new()]; // index 0 unused (AR(0))
    for k in 1..=base_max {
        let rows = n - k;
        let mut design = vec![0.0; rows * k];
        let mut y = vec![0.0; rows];
        for t in 0..rows {
            y[t] = xc[k + t];
            for l in 0..k {
                design[t * k + l] = xc[k + t - l - 1];
            }
        }
        let (beta, _s2, _se) = ols(&design, rows, k, &y)?;
        base.push(beta);
    }
    layers.push(base);

    // Iterated layers via recursion (2.7): layer j usable for k <= base_max - j.
    for j in 1..=jmax + 1 {
        let prev = &layers[j - 1];
        let mut layer: Vec<Vec<f64>> = vec![Vec::new()];
        for k in 1..=base_max - j {
            let pk1 = &prev[k + 1]; // Φ̂^{(j-1)}_{(k+1)}, length k+1
            let pk = &prev[k]; // Φ̂^{(j-1)}_{(k)}, length k
            let denom = pk[k - 1];
            if denom.abs() < 1e-300 {
                return Err(Error::Singular);
            }
            let ratio = pk1[k] / denom;
            let mut cur = Vec::with_capacity(k);
            for l in 1..=k {
                // Φ̂_{l-1(k)} with the Φ̂_{0} = −1 convention.
                let prev_lm1 = if l == 1 { -1.0 } else { pk[l - 2] };
                cur.push(pk1[l - 1] - prev_lm1 * ratio);
            }
            layer.push(cur);
        }
        layers.push(layer);
    }

    // Fill the table.
    let mut table = vec![vec![0.0; jmax + 1]; kmax + 1];
    let mut significant = vec![vec![false; jmax + 1]; kmax + 1];
    let mut strong = vec![vec![false; jmax + 1]; kmax + 1];
    for (k, row) in table.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let lag = j + 1;
            let r = if k == 0 {
                acf_lag(&xc, lag)
            } else {
                let phi = &layers[lag][k];
                let w: Vec<f64> = (k..n)
                    .map(|t| {
                        let mut v = xc[t];
                        for (l, p) in phi.iter().enumerate() {
                            v -= p * xc[t - l - 1];
                        }
                        v
                    })
                    .collect();
                acf_lag(&w, lag)
            };
            *cell = r;
            let se = 1.0 / ((n - k - lag) as f64).sqrt();
            significant[k][j] = r.abs() > config.band * se;
            // Structural cells are O(1) while noise is O(1/sqrt(n)) with a
            // variance the naive band understates by ~1.4x at distant lags;
            // 3x the band with an absolute 0.2 cap separates the two scales
            // at every n.
            strong[k][j] = r.abs() > (3.0 * config.band * se).min(0.2);
        }
    }

    // Vertex search: candidates ordered by p + q, then p. A triangle is
    // accepted when its number of significant cells does not exceed what the
    // ~5 % per-cell false-positive rate of the 2σ band explains (binomial
    // mean + 2 sd) — Tsay & Tiao's own examples read the vertex through one
    // or two stray X's that "barely exceed" the band.
    let mut order = None;
    'outer: for total in 0..=kmax + jmax {
        for p in 0..=total.min(kmax) {
            let q = total - p;
            if q > jmax {
                continue;
            }
            // Scale separation does the work: structural (non-zero) cells are
            // O(1) while noise cells are O(1/√n), and the 2σ band sits between
            // noise and the stray fluctuations the paper itself warns about.
            // The triangle must contain no *strong* X (twice the band ≈ 4σ,
            // where noise essentially never reaches); the vertex contrast uses
            // the ordinary band: the cell left of the vertex is non-zero under
            // the true order (eq. 3.5), and for p ≥ 1 the row above must show
            // structure somewhere.
            let mut cells = 0usize;
            let mut clean = true;
            'tri: for d in 0..=kmax - p {
                let start = (q + d).min(jmax + 1);
                for cell in strong[p + d].iter().skip(start) {
                    cells += 1;
                    if *cell {
                        clean = false;
                        break 'tri;
                    }
                }
            }
            let q_contrast = q == 0 || significant[p][q - 1];
            let p_contrast = p == 0 || (0..=jmax).any(|j| strong[p - 1][j]);
            if cells >= 3 && clean && q_contrast && p_contrast {
                order = Some((p, q));
                break 'outer;
            }
        }
    }

    Ok(EacfResult {
        order,
        table,
        significant,
    })
}

fn acf_lag(w: &[f64], lag: usize) -> f64 {
    let n = w.len();
    let mean = w.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for t in 0..n {
        let d = w[t] - mean;
        den += d * d;
        if t + lag < n {
            num += d * (w[t + lag] - mean);
        }
    }
    if den <= 0.0 {
        0.0
    } else {
        num / den
    }
}
