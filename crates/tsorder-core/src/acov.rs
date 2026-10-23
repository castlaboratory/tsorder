//! Sample autocovariance / autocorrelation.

use crate::{check_input, Result};

/// Biased (divide-by-`n`) sample autocovariance for lags `0..=max_lag`,
/// after mean removal. The biased estimator keeps the Toeplitz matrix
/// positive semi-definite, which Levinson–Durbin requires.
///
/// Direct `O(n · max_lag)` sums for small lags; above lag 64 the lags are
/// computed in one shot through a zero-padded FFT (`O(m log m)` with
/// `m ≥ n + max_lag` a power of two). Both paths give the same numbers up
/// to floating-point roundoff.
pub fn autocovariance(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    check_input(x, max_lag + 2)?;
    let n = x.len();
    let mean = x.iter().sum::<f64>() / n as f64;
    let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
    if max_lag > 64 {
        Ok(acov_fft(&xc, max_lag))
    } else {
        Ok(acov_direct(&xc, max_lag))
    }
}

fn acov_direct(xc: &[f64], max_lag: usize) -> Vec<f64> {
    let n = xc.len();
    let mut r = Vec::with_capacity(max_lag + 1);
    for k in 0..=max_lag {
        let mut s = 0.0;
        for t in 0..n - k {
            s += xc[t] * xc[t + k];
        }
        r.push(s / n as f64);
    }
    r
}

/// All lags at once: FFT of the zero-padded series, power spectrum, inverse
/// FFT. Padding to `m ≥ n + max_lag` makes the circular correlation linear.
fn acov_fft(xc: &[f64], max_lag: usize) -> Vec<f64> {
    let n = xc.len();
    let m = (n + max_lag + 1).next_power_of_two();
    let mut re = vec![0.0; m];
    let mut im = vec![0.0; m];
    re[..n].copy_from_slice(xc);
    fft(&mut re, &mut im, false);
    for i in 0..m {
        re[i] = re[i] * re[i] + im[i] * im[i];
        im[i] = 0.0;
    }
    fft(&mut re, &mut im, true);
    let scale = 1.0 / (m as f64 * n as f64); // 1/m undoes the unnormalised inverse FFT
    (0..=max_lag).map(|k| re[k] * scale).collect()
}

/// Iterative radix-2 Cooley–Tukey, unnormalised in both directions (the caller
/// divides by `m` after the inverse). Twiddles are computed by direct trig per
/// butterfly column, trading a few cycles for accuracy.
fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let base = sign * 2.0 * core::f64::consts::PI / len as f64;
        let mut i = 0;
        while i < n {
            for k in 0..half {
                let ang = base * k as f64;
                let (wi, wr) = ang.sin_cos();
                let (ur, ui) = (re[i + k], im[i + k]);
                let (xr, xi) = (re[i + k + half], im[i + k + half]);
                let vr = xr * wr - xi * wi;
                let vi = xr * wi + xi * wr;
                re[i + k] = ur + vr;
                im[i + k] = ui + vi;
                re[i + k + half] = ur - vr;
                im[i + k + half] = ui - vi;
            }
            i += len;
        }
        len <<= 1;
    }
}

/// Inverse autocovariances of a fitted AR(p), up to a `1/σ²` factor that
/// cancels in any correlation: for `x_t = Σ φ_j x_{t-j} + e_t`,
/// `vi(0) = 1 + Σ φ_i²` and `vi(k) = −φ_k + Σ_{i=1}^{p−k} φ_i φ_{i+k}`.
/// This is Cleveland's (1972) autoregressive estimator of the inverse
/// autocovariance function (his §4, in the α = −φ sign convention), and it is
/// a genuine MA(p) autocovariance sequence, hence positive semi-definite.
pub fn inverse_acov_from_ar(phi: &[f64]) -> Vec<f64> {
    let p = phi.len();
    let mut vi = Vec::with_capacity(p + 1);
    vi.push(1.0 + phi.iter().map(|v| v * v).sum::<f64>());
    for k in 1..=p {
        let mut s = -phi[k - 1];
        for i in 1..=p - k {
            s += phi[i - 1] * phi[i + k - 1];
        }
        vi.push(s);
    }
    vi
}

/// Sample autocorrelation for lags `0..=max_lag` (`acf[0] == 1`).
pub fn acf(x: &[f64], max_lag: usize) -> Result<Vec<f64>> {
    let r = autocovariance(x, max_lag)?;
    let r0 = r[0];
    if r0 <= 0.0 {
        return Err(crate::Error::Singular);
    }
    Ok(r.iter().map(|v| v / r0).collect())
}

/// Bartlett large-sample standard error of `acf[k]` under the hypothesis that
/// the process is MA(k−1) (i.e. true autocorrelations vanish from lag `k` on):
/// `sqrt((1 + 2 Σ_{j<k} ρ_j²) / n)`. Used to studentize the ACF so that the
/// Fine–Hwang threshold is scale-free for MA order selection.
pub fn bartlett_se(acf: &[f64], n: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(acf.len());
    let mut cum = 0.0;
    for (k, &rk) in acf.iter().enumerate() {
        if k == 0 {
            out.push(0.0);
        } else {
            out.push(((1.0 + 2.0 * cum) / n as f64).sqrt());
        }
        if k >= 1 {
            cum += rk * rk;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn white_noise_acf_small() {
        let x: Vec<f64> = (0..2000)
            .map(|i| ((i * 7919) % 1000) as f64 / 1000.0 - 0.5)
            .collect();
        let r = acf(&x, 5).unwrap();
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn fft_path_matches_direct_sums() {
        // Deterministic pseudo-random series; lag 120 forces the FFT path.
        let mut s = 1u64;
        let x: Vec<f64> = (0..500)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (s >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        let mean = x.iter().sum::<f64>() / x.len() as f64;
        let xc: Vec<f64> = x.iter().map(|v| v - mean).collect();
        let direct = acov_direct(&xc, 120);
        let fast = acov_fft(&xc, 120);
        for (k, (a, b)) in direct.iter().zip(&fast).enumerate() {
            assert!(
                (a - b).abs() < 1e-12 * direct[0].abs(),
                "lag {k}: {a} vs {b}"
            );
        }
        // And the public function agrees with the direct sums it replaced.
        let public = autocovariance(&x, 120).unwrap();
        assert!((public[120] - direct[120]).abs() < 1e-12 * direct[0].abs());
    }

    #[test]
    fn inverse_acov_matches_cleveland_example_1() {
        // Cleveland (1972), Example 1 / Display 1: X_t + .7 X_{t-1} + .4 X_{t-6} = R_t.
        // His alpha = -phi; reported inverse autocorrelations (x100): 42 at lag 1,
        // 17 at lag 5, 24 at lag 6, 0 elsewhere.
        let phi = [-0.7, 0.0, 0.0, 0.0, 0.0, -0.4];
        let vi = inverse_acov_from_ar(&phi);
        let ri: Vec<f64> = vi.iter().map(|v| v / vi[0]).collect();
        assert!((ri[1] - 0.7 / 1.65).abs() < 1e-12); // 0.4242 -> 42
        assert!((ri[5] - 0.28 / 1.65).abs() < 1e-12); // 0.1697 -> 17
        assert!((ri[6] - 0.4 / 1.65).abs() < 1e-12); // 0.2424 -> 24
        for k in [2, 3, 4] {
            assert!(ri[k].abs() < 1e-12, "ri[{k}] = {}", ri[k]);
        }
    }

    #[test]
    fn bartlett_lag1_is_one_over_sqrt_n() {
        let se = bartlett_se(&[1.0, 0.5, 0.2], 400);
        assert!((se[1] - 0.05).abs() < 1e-12);
        assert!(se[2] > se[1]);
    }
}
