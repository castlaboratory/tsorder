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

/// Singular values of a dense `rows × cols` matrix (row-major), descending,
/// by one-sided Jacobi: pairwise orthogonalisation of the columns until every
/// pair is numerically orthogonal; the singular values are then the column
/// norms. `O(sweeps · cols² · rows)` with a handful of sweeps — fine for the
/// `K × K` matrices of the canonical-correlation step (`K ≤ ~50`).
pub fn singular_values(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), rows * cols);
    // Column-major working copy.
    let mut u: Vec<f64> = Vec::with_capacity(rows * cols);
    for j in 0..cols {
        for i in 0..rows {
            u.push(a[i * cols + j]);
        }
    }
    let tol = 1e-14;
    for _sweep in 0..60 {
        let mut rotated = false;
        for i in 0..cols {
            for j in i + 1..cols {
                let (mut aii, mut ajj, mut aij) = (0.0f64, 0.0f64, 0.0f64);
                for t in 0..rows {
                    let ui = u[i * rows + t];
                    let uj = u[j * rows + t];
                    aii += ui * ui;
                    ajj += uj * uj;
                    aij += ui * uj;
                }
                if aij.abs() <= tol * (aii * ajj).sqrt() || aii == 0.0 || ajj == 0.0 {
                    continue;
                }
                rotated = true;
                // Jacobi rotation that zeroes the off-diagonal of the 2×2 Gram block.
                let zeta = (ajj - aii) / (2.0 * aij);
                let t_ = zeta.signum() / (zeta.abs() + (1.0 + zeta * zeta).sqrt());
                let c = 1.0 / (1.0 + t_ * t_).sqrt();
                let s = c * t_;
                for t in 0..rows {
                    let ii = i * rows + t;
                    let jj = j * rows + t;
                    let (ui, uj) = (u[ii], u[jj]);
                    u[ii] = c * ui - s * uj;
                    u[jj] = s * ui + c * uj;
                }
            }
        }
        if !rotated {
            break;
        }
    }
    let mut sv: Vec<f64> = (0..cols)
        .map(|j| {
            (0..rows)
                .map(|t| u[j * rows + t].powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .collect();
    sv.sort_unstable_by(|a, b| b.total_cmp(a));
    sv
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
    fn singular_values_known_cases() {
        // Diagonal.
        let d = [3.0, 0.0, 0.0, 2.0];
        let sv = singular_values(&d, 2, 2);
        assert!((sv[0] - 3.0).abs() < 1e-12 && (sv[1] - 2.0).abs() < 1e-12);
        // 2x2 with known SVD: [[1,1],[0,1]] has sv sqrt((3±sqrt(5))/2).
        let a = [1.0, 1.0, 0.0, 1.0];
        let sv = singular_values(&a, 2, 2);
        let s1 = ((3.0 + 5.0f64.sqrt()) / 2.0).sqrt();
        let s2 = ((3.0 - 5.0f64.sqrt()) / 2.0).sqrt();
        assert!((sv[0] - s1).abs() < 1e-12, "{sv:?}");
        assert!((sv[1] - s2).abs() < 1e-12);
        // Rectangular 3x2: Frobenius norm and product (via Gram determinant) match.
        let b = [1.0, 2.0, -1.0, 0.5, 3.0, 1.0];
        let sv = singular_values(&b, 3, 2);
        let fro: f64 = b.iter().map(|v| v * v).sum();
        assert!((sv.iter().map(|v| v * v).sum::<f64>() - fro).abs() < 1e-10);
        // det(BᵀB) = (σ1 σ2)².
        let (g00, g01, g11) = (1.0f64 + 1.0 + 9.0, 2.0 - 0.5 + 3.0, 4.0f64 + 0.25 + 1.0);
        let det = g00 * g11 - g01 * g01;
        assert!(((sv[0] * sv[1]).powi(2) - det).abs() < 1e-9);
    }

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
