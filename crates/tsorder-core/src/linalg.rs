//! Minimal dense linear algebra (Cholesky) for the OLS long-AR path.
//! Kept tiny on purpose; the core has no external dependencies.

use crate::{Error, Result};

/// Dense column-major-agnostic square matrix stored row-major.
#[derive(Debug, Clone)]
pub struct SquareMatrix {
    pub n: usize,
    pub data: Vec<f64>,
}

impl SquareMatrix {
    pub fn zeros(n: usize) -> Self {
        Self {
            n,
            data: vec![0.0; n * n],
        }
    }
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    /// In-place lower Cholesky factor `L` with `A = L Lᵀ`.
    pub fn cholesky(&self) -> Result<SquareMatrix> {
        let n = self.n;
        let mut l = SquareMatrix::zeros(n);
        for i in 0..n {
            for j in 0..=i {
                let mut s = self.get(i, j);
                for k in 0..j {
                    s -= l.get(i, k) * l.get(j, k);
                }
                if i == j {
                    if s <= 0.0 || !s.is_finite() {
                        return Err(Error::Singular);
                    }
                    l.set(i, j, s.sqrt());
                } else {
                    l.set(i, j, s / l.get(j, j));
                }
            }
        }
        Ok(l)
    }

    /// Solve `A x = b` given the Cholesky factor `L` of `A`.
    pub fn cholesky_solve(l: &SquareMatrix, b: &[f64]) -> Vec<f64> {
        Self::cholesky_solve_leading(l, l.n, b)
    }

    /// Solve the leading `k × k` subsystem `A[..k, ..k] x = b` given the Cholesky
    /// factor `L` of the full matrix. Valid because the Cholesky factor of a
    /// leading principal submatrix is the leading submatrix of the factor, which
    /// lets one `O(n³)` factorisation serve every nested model.
    #[allow(clippy::needless_range_loop)] // index form mirrors the textbook recursion
    pub fn cholesky_solve_leading(l: &SquareMatrix, k: usize, b: &[f64]) -> Vec<f64> {
        debug_assert!(k <= l.n && b.len() >= k);
        let mut y = vec![0.0; k];
        for i in 0..k {
            let mut s = b[i];
            for j in 0..i {
                s -= l.get(i, j) * y[j];
            }
            y[i] = s / l.get(i, i);
        }
        let mut x = vec![0.0; k];
        for i in (0..k).rev() {
            let mut s = y[i];
            for j in i + 1..k {
                s -= l.get(j, i) * x[j];
            }
            x[i] = s / l.get(i, i);
        }
        x
    }

    /// Diagonal of `A⁻¹` from the Cholesky factor (solves `n` unit systems; `O(n³)`,
    /// fine for `n = p_n ≤ ~50`).
    pub fn cholesky_inverse_diag(l: &SquareMatrix) -> Vec<f64> {
        let n = l.n;
        let mut diag = vec![0.0; n];
        let mut e = vec![0.0; n];
        for j in 0..n {
            e.iter_mut().for_each(|v| *v = 0.0);
            e[j] = 1.0;
            let col = SquareMatrix::cholesky_solve(l, &e);
            diag[j] = col[j];
        }
        diag
    }
}

/// OLS fit of `y` on the columns of `X` (row-major `rows × cols`).
/// Returns `(beta, residual_variance, se)` with `se` the classical standard errors
/// `sqrt(σ̂² (XᵀX)⁻¹_jj)` and `σ̂² = RSS / (rows - cols)`.
pub fn ols(x: &[f64], rows: usize, cols: usize, y: &[f64]) -> Result<(Vec<f64>, f64, Vec<f64>)> {
    if rows <= cols {
        return Err(Error::TooShort {
            n: rows,
            needed: cols + 1,
        });
    }
    let mut xtx = SquareMatrix::zeros(cols);
    let mut xty = vec![0.0; cols];
    for t in 0..rows {
        let row = &x[t * cols..(t + 1) * cols];
        for i in 0..cols {
            xty[i] += row[i] * y[t];
            for j in 0..=i {
                let v = xtx.get(i, j) + row[i] * row[j];
                xtx.set(i, j, v);
                xtx.set(j, i, v);
            }
        }
    }
    let l = xtx.cholesky()?;
    let beta = SquareMatrix::cholesky_solve(&l, &xty);
    let mut rss = 0.0;
    for t in 0..rows {
        let row = &x[t * cols..(t + 1) * cols];
        let fit: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
        rss += (y[t] - fit).powi(2);
    }
    let s2 = rss / (rows - cols) as f64;
    let dinv = SquareMatrix::cholesky_inverse_diag(&l);
    let se = dinv.iter().map(|d| (s2 * d).sqrt()).collect();
    Ok((beta, s2, se))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_solve_2x2() {
        let mut a = SquareMatrix::zeros(2);
        a.set(0, 0, 4.0);
        a.set(0, 1, 2.0);
        a.set(1, 0, 2.0);
        a.set(1, 1, 3.0);
        let l = a.cholesky().unwrap();
        let x = SquareMatrix::cholesky_solve(&l, &[2.0, 1.0]);
        // 4x+2y=2, 2x+3y=1 -> x=0.5, y=0
        assert!((x[0] - 0.5).abs() < 1e-12 && x[1].abs() < 1e-12);
        let d = SquareMatrix::cholesky_inverse_diag(&l);
        // inv = [[3,-2],[-2,4]]/8
        assert!((d[0] - 3.0 / 8.0).abs() < 1e-12 && (d[1] - 0.5).abs() < 1e-12);
    }
}
