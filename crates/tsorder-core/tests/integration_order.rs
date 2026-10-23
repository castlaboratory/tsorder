//! Order-of-integration estimation: recovery of d on simulated I(0), I(1),
//! I(2) series with both unit-root tests, and affine invariance.

use tsorder_core::sim::arma;
use tsorder_core::*;

fn cumsum(x: &[f64]) -> Vec<f64> {
    let mut s = 0.0;
    x.iter()
        .map(|v| {
            s += v;
            s
        })
        .collect()
}

#[test]
fn recovers_d_for_i0_i1_i2() {
    for test in [UnitRootTest::Kpss, UnitRootTest::Adf] {
        let cfg = IntegrationConfig {
            test,
            ..Default::default()
        };
        let mut ok = [0usize; 3];
        for seed in 0..10 {
            let base = arma(&[0.5], &[], 2000, 2000 + seed);
            if integration_order(&base, cfg).unwrap().d == 0 {
                ok[0] += 1;
            }
            let i1 = cumsum(&base);
            if integration_order(&i1, cfg).unwrap().d == 1 {
                ok[1] += 1;
            }
            let i2 = cumsum(&i1);
            if integration_order(&i2, cfg).unwrap().d == 2 {
                ok[2] += 1;
            }
        }
        assert!(ok[0] >= 9, "{test:?} I(0): {ok:?}");
        assert!(ok[1] >= 9, "{test:?} I(1): {ok:?}");
        assert!(ok[2] >= 9, "{test:?} I(2): {ok:?}");
    }
}

#[test]
fn near_unit_root_stationary_is_hard_but_level_is_respected() {
    // AR(1) with phi = 0.95 at n = 4000: still I(0); KPSS should mostly agree.
    let mut ok = 0;
    for seed in 0..10 {
        let x = arma(&[0.95], &[], 4000, 3000 + seed);
        if integration_order(&x, IntegrationConfig::default())
            .unwrap()
            .d
            == 0
        {
            ok += 1;
        }
    }
    assert!(ok >= 7, "near-unit-root I(0) rate {ok}/10");
}

#[test]
fn integration_order_is_affine_invariant() {
    let base = arma(&[0.5], &[], 1500, 77);
    let i1 = cumsum(&base);
    let shifted: Vec<f64> = i1.iter().map(|v| 3.0 * v + 50.0).collect();
    for test in [UnitRootTest::Kpss, UnitRootTest::Adf] {
        let cfg = IntegrationConfig {
            test,
            ..Default::default()
        };
        let a = integration_order(&i1, cfg).unwrap();
        let b = integration_order(&shifted, cfg).unwrap();
        assert_eq!(a.d, b.d);
        for (u, v) in a.stats.iter().zip(&b.stats) {
            assert!((u - v).abs() < 1e-8, "{test:?}: {u} vs {v}");
        }
    }
}

#[test]
fn critical_values_match_published_asymptotics() {
    assert!((kpss_critical(0.05) - 0.463).abs() < 0.005);
    assert!((kpss_critical(0.10) - 0.347).abs() < 0.005);
    assert!((df_tau_c_critical(0.05) - (-2.86)).abs() < 0.02);
    assert!((df_tau_c_critical(0.01) - (-3.43)).abs() < 0.03);
}
