//! McMillan-degree estimation: reproduction of Bauer (2001) Example I /
//! Table 1 as a regression test, plus basic sanity and invariance.

use tsorder_core::sim::Xorshift;
use tsorder_core::*;

/// Bauer (2001) Example I: x_{t+1} = A x_t + K e_t, y_t = C x_t + e_t with
/// A = [[0, 1], [-0.7, 0.5]], K = [1.3, 0.3]', C = [1, 0]. True state
/// dimension 2 (poles 0.25 ± 0.7984i).
fn bauer_example_i(t_len: usize, seed: u64) -> Vec<f64> {
    let burn = 500;
    let mut rng = Xorshift::new(seed);
    let (mut x1, mut x2) = (0.0f64, 0.0f64);
    let mut y = Vec::with_capacity(t_len);
    for t in 0..t_len + burn {
        let e = rng.normal();
        if t >= burn {
            y.push(x1 + e);
        }
        let nx1 = x2 + 1.3 * e;
        let nx2 = -0.7 * x1 + 0.5 * x2 + 0.3 * e;
        x1 = nx1;
        x2 = nx2;
    }
    y
}

#[test]
fn bauer_table1_svc_log_t_cca_t1000() {
    // Table 1, CCA row, SVC1 (C(T) = log T), f = p = p_hat_AIC, T = 1000:
    // P(n < 2) = 0.00, P(n = 2) = 0.93, P(n > 2) = 0.07 over 1000 reps.
    // 200 replications here: binomial sd ≈ 0.018 at p = 0.93.
    let (mut under, mut correct, mut over) = (0, 0, 0);
    let reps = 200;
    for seed in 0..reps {
        let y = bauer_example_i(1000, 7000 + seed);
        let r = svc_order(
            &y,
            SvcConfig {
                window_factor: 1.0,
                ..Default::default()
            },
        )
        .unwrap();
        match r.order.cmp(&2) {
            std::cmp::Ordering::Less => under += 1,
            std::cmp::Ordering::Equal => correct += 1,
            std::cmp::Ordering::Greater => over += 1,
        }
    }
    assert_eq!(under, 0, "SVC1/CCA never underestimates in Table 1");
    let pc = correct as f64 / reps as f64;
    assert!(pc > 0.85 && pc <= 1.0, "P(correct) = {pc} (Bauer: 0.93)");
    assert!(
        (over as f64 / reps as f64) < 0.15,
        "P(over) = {} (Bauer: 0.07)",
        over as f64 / reps as f64
    );
}

#[test]
fn bauer_table1_svc_fp_log_t_underestimates_at_t1000() {
    // Table 1, CCA row, SVC2 (C(T) = fp log T), T = 1000: P(n < 2) = 0.86 -
    // the theoretically consistent penalty is far too strong in finite
    // samples. Reproducing the *pattern* (heavy underestimation) is the test.
    let mut under = 0;
    let reps = 100;
    for seed in 0..reps {
        let y = bauer_example_i(1000, 9000 + seed);
        let r = svc_order(
            &y,
            SvcConfig {
                window_factor: 1.0,
                penalty: SvcPenalty::FpLogT,
                ..Default::default()
            },
        )
        .unwrap();
        if r.order < 2 {
            under += 1;
        }
    }
    let pu = under as f64 / reps as f64;
    assert!(pu > 0.6, "P(under) = {pu} (Bauer: 0.86)");
}

#[test]
fn svc_white_noise_gives_zero_and_arma11_gives_one() {
    for seed in 0..5 {
        let wn = sim::arma(&[], &[], 2000, 100 + seed);
        assert_eq!(svc_order(&wn, SvcConfig::default()).unwrap().order, 0);
        let arma11 = sim::arma(&[0.5], &[0.5], 4000, 200 + seed);
        assert_eq!(svc_order(&arma11, SvcConfig::default()).unwrap().order, 1);
    }
}

#[test]
fn hankel_svt_recovers_separated_systems() {
    let mut ok = [0usize; 3];
    for seed in 0..20 {
        let wn = sim::arma(&[], &[], 2000, 300 + seed);
        if hankel_svt_order(&wn, HankelSvtConfig::default())
            .unwrap()
            .order
            == 0
        {
            ok[0] += 1;
        }
        let ar1 = sim::arma(&[0.9], &[], 4000, 400 + seed);
        if hankel_svt_order(&ar1, HankelSvtConfig::default())
            .unwrap()
            .order
            == 1
        {
            ok[1] += 1;
        }
        let b = bauer_example_i(2000, 500 + seed);
        if hankel_svt_order(&b, HankelSvtConfig::default())
            .unwrap()
            .order
            == 2
        {
            ok[2] += 1;
        }
    }
    assert!(ok[0] >= 19, "white noise rank-0 rate {:?}", ok);
    assert!(ok[1] >= 17, "AR(1) rate {:?}", ok);
    assert!(ok[2] >= 19, "Bauer Example I rate {:?}", ok);
}

#[test]
fn hankel_svt_is_invariant_to_shift_and_scale() {
    let y = bauer_example_i(1500, 42);
    let shifted: Vec<f64> = y.iter().map(|v| 3.0 * v + 50.0).collect();
    let a = hankel_svt_order(&y, HankelSvtConfig::default()).unwrap();
    let b = hankel_svt_order(&shifted, HankelSvtConfig::default()).unwrap();
    // Singular values and xi both scale by 9; the order is unchanged.
    assert_eq!(a.order, b.order);
    assert!((9.0 * a.xi - b.xi).abs() < 1e-8 * b.xi);
}

#[test]
fn hankel_rank_test_recovers_and_is_calibrated() {
    // Level under H0 with fixed alpha: rejection approaches nominal from
    // below (the Robin-Smith finite-sample finding); recovery with the
    // decaying level is near-perfect on separated systems.
    let mut ok = [0usize; 3];
    for seed in 0..15 {
        let wn = sim::arma(&[], &[], 1500, 600 + seed);
        if hankel_rank_order(&wn, HankelRankConfig::default())
            .unwrap()
            .order
            == 0
        {
            ok[0] += 1;
        }
        let ar1 = sim::arma(&[0.9], &[], 1500, 700 + seed);
        if hankel_rank_order(&ar1, HankelRankConfig::default())
            .unwrap()
            .order
            == 1
        {
            ok[1] += 1;
        }
        let b = bauer_example_i(1500, 800 + seed);
        if hankel_rank_order(&b, HankelRankConfig::default())
            .unwrap()
            .order
            == 2
        {
            ok[2] += 1;
        }
    }
    assert!(ok[0] >= 14, "white noise {:?}", ok);
    assert!(ok[1] >= 13, "AR(1) {:?}", ok);
    assert!(ok[2] >= 13, "Bauer I {:?}", ok);
}

#[test]
fn eacf_identifies_arma_orders() {
    let cases: [(&[f64], &[f64], (usize, usize)); 4] = [
        (&[], &[], (0, 0)),
        (&[1.2, -0.6], &[], (2, 0)),
        (&[], &[0.6, 0.3], (0, 2)),
        (&[0.5], &[0.5], (1, 1)),
    ];
    for (phi, theta, want) in cases {
        let mut ok = 0;
        for seed in 0..5 {
            let x = sim::arma(phi, theta, 4000, 900 + seed);
            if eacf(&x, EacfConfig::default()).unwrap().order == Some(want) {
                ok += 1;
            }
        }
        assert!(ok >= 4, "eacf {want:?}: {ok}/5");
    }
}

#[test]
fn hannan_rissanen_selects_and_estimates_arma11() {
    let mut ok = 0;
    for seed in 0..5 {
        let x = sim::arma(&[0.5], &[0.8], 2000, 950 + seed);
        let h = hannan_rissanen(&x, HrConfig::default()).unwrap();
        if (h.p, h.q) == (1, 1) {
            ok += 1;
            assert!((h.phi[0] - 0.5).abs() < 0.15, "phi {:?}", h.phi);
            assert!((h.theta[0] - 0.8).abs() < 0.15, "theta {:?}", h.theta);
        }
    }
    assert!(ok >= 4, "HR correct {ok}/5");
}

#[test]
fn svc_is_invariant_to_shift_and_scale() {
    let y = bauer_example_i(1500, 42);
    let shifted: Vec<f64> = y.iter().map(|v| 3.0 * v + 50.0).collect();
    let a = svc_order(&y, SvcConfig::default()).unwrap();
    let b = svc_order(&shifted, SvcConfig::default()).unwrap();
    assert_eq!(a.order, b.order);
    for (u, v) in a.correlations.iter().zip(&b.correlations) {
        assert!((u - v).abs() < 1e-8);
    }
}
