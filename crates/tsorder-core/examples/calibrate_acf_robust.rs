//! Scratch calibration for the AcfRobust null studentization (not shipped
//! logic): measures the dispersion of the robust autocorrelation beyond the
//! true order, relative to the Bartlett plug-in computed from the robust ACF
//! itself, across processes and sample sizes.

use tsorder_core::acov::bartlett_se;
use tsorder_core::robust::robust_autocorrelation;
use tsorder_core::sim::arma as sim_arma;

fn main() {
    let designs: &[(&str, &[f64], &[f64], usize)] = &[
        ("wn", &[], &[], 0),
        ("ma2", &[], &[0.9, 0.5], 2),
        ("ar2", &[1.2, -0.6], &[], 30), // ACF never dies; look far out
        ("ma1neg", &[], &[-0.8], 1),
    ];
    for n in [500usize, 2000, 8000] {
        for (name, ar, ma, q) in designs {
            let p_n = ((2.0 * (n as f64).ln()).ceil() as usize).max(q + 4);
            let mut ratios: Vec<f64> = Vec::new();
            for seed in 0..400u64 {
                let x = sim_arma(ar, ma, n, 90_000 + seed);
                let r = match robust_autocorrelation(&x, p_n) {
                    Ok(r) => r,
                    Err(_) => continue,
                };
                let se = bartlett_se(&r, n);
                for k in (q + 1)..=p_n {
                    ratios.push(r[k] / se[k]);
                }
            }
            let m = ratios.iter().sum::<f64>() / ratios.len() as f64;
            let v =
                ratios.iter().map(|t| (t - m) * (t - m)).sum::<f64>() / (ratios.len() - 1) as f64;
            println!(
                "n={n:5} {name:7} null sd of (rho_rob/bartlett_rob) = {:.3}",
                v.sqrt()
            );
        }
    }
}
