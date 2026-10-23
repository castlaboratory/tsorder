//! Integration tests: every estimator must recover the order of a long,
//! well-separated simulated series. The generator is the crate's own `sim`
//! module (xorshift + Box–Muller), enabled here through the self
//! dev-dependency.

use tsorder_core::sim::arma as sim_arma;
use tsorder_core::*;

#[test]
fn ar2_recovered_by_all_ar_statistics_and_bic() {
    let x = sim_arma(&[1.2, -0.6], &[], 4000, 42);
    for stat in [Statistic::Pacf, Statistic::LongArOls] {
        let r = fh_order(
            &x,
            FhConfig {
                statistic: stat,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.order, 2, "{} gave {}", stat.name(), r.order);
    }
    let bic = ic_order(&x, Criterion::Bic, None).unwrap();
    assert_eq!(bic.order, 2);
    let hq = ic_order(&x, Criterion::Hq, None).unwrap();
    assert_eq!(hq.order, 2);
}

#[test]
fn subset_ar4_recovered_by_pacf() {
    let x = sim_arma(&[0.5, 0.0, 0.0, 0.3], &[], 4000, 7);
    let r = fh_order(&x, FhConfig::default()).unwrap();
    assert_eq!(r.order, 4);
}

#[test]
fn ma2_recovered_by_bartlett_acf() {
    let x = sim_arma(&[], &[0.6, 0.3], 4000, 99);
    let r = fh_order(
        &x,
        FhConfig {
            statistic: Statistic::AcfBartlett,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.order, 2);
}

#[test]
fn ma2_recovered_by_inverse_pacf() {
    let x = sim_arma(&[], &[0.6, 0.3], 4000, 99);
    let r = fh_order(
        &x,
        FhConfig {
            statistic: Statistic::InversePacf,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.order, 2, "inverse PACF gave {}", r.order);
}

#[test]
fn white_noise_gives_order_zero() {
    let x = sim_arma(&[], &[], 2000, 3);
    let r = fh_order(&x, FhConfig::default()).unwrap();
    assert_eq!(r.order, 0);
    let bic = ic_order(&x, Criterion::Bic, None).unwrap();
    assert_eq!(bic.order, 0);
}

#[test]
fn ensemble_majority_on_ar1() {
    let x = sim_arma(&[0.7], &[], 3000, 11);
    let cfgs = [
        FhConfig {
            statistic: Statistic::Pacf,
            ..Default::default()
        },
        FhConfig {
            statistic: Statistic::LongArOls,
            ..Default::default()
        },
        FhConfig {
            statistic: Statistic::LongArYw,
            ..Default::default()
        },
    ];
    let e = ensemble_order(&x, &cfgs, EnsembleRule::Majority).unwrap();
    assert_eq!(e.order, 1);
    assert_eq!(e.members.len(), 3);
}

#[test]
fn all_criteria_run() {
    let x = sim_arma(&[0.5], &[], 500, 5);
    for c in Criterion::ALL {
        let r = ic_order(&x, c, Some(10)).unwrap();
        assert_eq!(r.values.len(), 11, "{}", c.name());
    }
}

#[test]
fn ar2_recovered_by_all_ic_methods_and_wn_gives_zero() {
    let x = sim_arma(&[1.2, -0.6], &[], 4000, 42);
    let wn = sim_arma(&[], &[], 2000, 3);
    for m in IcMethod::ALL {
        let r = ic_order_with(&x, Criterion::Bic, m, None).unwrap();
        assert_eq!(r.order, 2, "{} gave {}", m.name(), r.order);
        let z = ic_order_with(&wn, Criterion::Bic, m, None).unwrap();
        assert_eq!(z.order, 0, "{} gave {} on white noise", m.name(), z.order);
    }
}

#[test]
fn burg_matches_yule_walker_on_long_series() {
    // Burg and YW are asymptotically equivalent away from the unit circle: on a
    // long AR(2) the PACF estimates should agree to a few decimal places.
    let x = sim_arma(&[1.2, -0.6], &[], 8000, 21);
    let yw = levinson_from_series(&x, 6).unwrap();
    let bg = burg_from_series(&x, 6).unwrap();
    for (a, b) in yw.pacf.iter().zip(&bg.pacf) {
        assert!((a - b).abs() < 0.02, "yw {a} vs burg {b}");
    }
}

#[test]
fn errors_are_reported() {
    assert!(matches!(
        fh_order(&[1.0, 2.0, 3.0], FhConfig::default()),
        Err(Error::TooShort { .. })
    ));
    assert!(matches!(
        ic_order(&[1.0, f64::NAN, 3.0, 4.0, 5.0], Criterion::Bic, Some(2)),
        Err(Error::NonFinite)
    ));
}

#[test]
fn all_statistics_invariant_to_shift_and_scale() {
    // Regression test for the review of 2026-10-23: LongArOls had no demeaning and
    // selected order ~10 on white noise + 50.
    let wn = sim_arma(&[], &[], 2000, 3);
    let ar2 = sim_arma(&[1.2, -0.6], &[], 2000, 21);
    for base in [&wn, &ar2] {
        let shifted: Vec<f64> = base.iter().map(|v| 3.0 * v + 50.0).collect();
        for stat in Statistic::ALL {
            let cfg = FhConfig {
                statistic: stat,
                ..Default::default()
            };
            let a = fh_order(base, cfg).unwrap();
            let b = fh_order(&shifted, cfg).unwrap();
            assert_eq!(
                a.order,
                b.order,
                "{} changed order under affine map",
                stat.name()
            );
            for (u, v) in a.studentized.iter().zip(&b.studentized) {
                assert!(
                    (u - v).abs() < 1e-8,
                    "{} studentized stats differ",
                    stat.name()
                );
            }
        }
        for c in Criterion::ALL {
            for m in IcMethod::ALL {
                let a = ic_order_with(base, c, m, Some(15)).unwrap();
                let b = ic_order_with(&shifted, c, m, Some(15)).unwrap();
                assert_eq!(
                    a.order,
                    b.order,
                    "{} ({}) changed order under affine map",
                    c.name(),
                    m.name()
                );
            }
        }
    }
}

#[test]
fn invalid_thresholds_are_errors_not_silent() {
    let x = sim_arma(&[0.5], &[], 500, 9);
    for thr in [
        Threshold::BicLike { c: -1.0 },
        Threshold::BicLike { c: 0.0 },
        Threshold::Lil { c: f64::NAN },
        Threshold::Fixed(-5.0),
        Threshold::Fixed(f64::INFINITY),
    ] {
        let r = fh_order(
            &x,
            FhConfig {
                threshold: thr,
                ..Default::default()
            },
        );
        assert!(
            matches!(r, Err(Error::Invalid(_))),
            "{} accepted",
            thr.name()
        );
    }
}

#[test]
fn sparse_ar_pacf_gives_order_but_long_ar_gives_support() {
    // AR(4) with φ2 = φ3 = 0. The PACF at lags 2 and 3 is NOT zero for this process, so
    // PACF-clip recovers the order 4 but says nothing about sparsity; the long-AR OLS
    // coefficients at lags 2 and 3 are clipped to zero and recover the support.
    let x = sim_arma(&[0.5, 0.0, 0.0, 0.3], &[], 8000, 7);
    let pacf = fh_order(&x, FhConfig::default()).unwrap();
    assert_eq!(pacf.order, 4);
    let ols = fh_order(
        &x,
        FhConfig {
            statistic: Statistic::LongArOls,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ols.order, 4);
    let support: Vec<bool> = ols.studentized.iter().map(|t| t.abs() > ols.tau).collect();
    assert_eq!(&support[..4], &[true, false, false, true]);
}

#[test]
fn aicc_counts_variance_and_matches_formula() {
    // k=2, n=50: K=3, AICc = n log s2 + 2K + 2K(K+1)/(n-K-1) = ... + 6 + 24/46
    let v = Criterion::Aicc.value(&[1.0, 1.0, 1.0], 2, 50);
    assert!((v - (6.0 + 24.0 / 46.0)).abs() < 1e-12);
    assert_eq!(Criterion::n_params(2), 3);
}

#[test]
fn cat_and_mdl_recover_ar2_and_mdl_matches_bic() {
    let x = sim_arma(&[1.2, -0.6], &[], 4000, 42);
    // CAT is asymptotically equivalent to AIC (efficient, not consistent): on this
    // series AIC, FPE and CAT all pick 3 while BIC/HQ/MDL pick the true 2.
    let cat = ic_order(&x, Criterion::Cat, None).unwrap();
    let aic = ic_order(&x, Criterion::Aic, None).unwrap();
    assert_eq!(cat.order, aic.order, "CAT gave {}", cat.order);
    assert!((2..=3).contains(&cat.order), "CAT gave {}", cat.order);
    let mdl = ic_order(&x, Criterion::Mdl, None).unwrap();
    let bic = ic_order(&x, Criterion::Bic, None).unwrap();
    assert_eq!(mdl.order, bic.order);
    for (m, b) in mdl.values.iter().zip(&bic.values) {
        assert!((2.0 * m - b).abs() < 1e-9, "MDL is not BIC/2");
    }
    // CAT(0) is Parzen's convention -(1 + 1/n).
    assert!((cat.values[0] + 1.0 + 1.0 / 4000.0).abs() < 1e-12);
}

#[test]
fn gumbel_level_is_controlled_on_white_noise() {
    // Family-wise overestimation on white noise at alpha = 0.05 should be ≈ 5 %
    // (loose check), both for the self-studentized PACF and for the Jirak band.
    for stat in [Statistic::Pacf, Statistic::JirakBand] {
        let mut over = 0;
        let reps = 400;
        for seed in 0..reps {
            let x = sim_arma(&[], &[], 1000, 1000 + seed as u64);
            let r = fh_order(
                &x,
                FhConfig {
                    statistic: stat,
                    threshold: Threshold::Gumbel { alpha: 0.05 },
                    memory: Some(polylog_memory(1000, 2.0)),
                },
            )
            .unwrap();
            if r.order > 0 {
                over += 1;
            }
        }
        let rate = over as f64 / reps as f64;
        assert!(rate < 0.12, "{}: overestimation rate {rate}", stat.name());
    }
}

#[test]
fn robust_pacf_survives_additive_outliers_where_plain_pacf_dies() {
    // AR(2) with 5 % additive outliers of magnitude 10 (~5 sd). The classical
    // PACF is destroyed (it never finds order 2 in 10 tries); the Q_n-based
    // robust PACF keeps working on most replications.
    let cfg = sim::ArmaSim {
        phi: &[1.2, -0.6],
        additive_outliers: Some(sim::Contamination {
            prob: 0.05,
            magnitude: 10.0,
        }),
        ..Default::default()
    };
    let mut plain_ok = 0;
    let mut robust_ok = 0;
    for seed in 0..10 {
        let x = cfg.simulate(3000, 5000 + seed).unwrap();
        if fh_order(&x, FhConfig::default()).unwrap().order == 2 {
            plain_ok += 1;
        }
        let r = fh_order(
            &x,
            FhConfig {
                statistic: Statistic::PacfRobust,
                ..Default::default()
            },
        )
        .unwrap();
        if r.order == 2 {
            robust_ok += 1;
        }
    }
    assert_eq!(
        plain_ok, 0,
        "plain PACF unexpectedly survived contamination"
    );
    assert!(robust_ok >= 8, "robust PACF correct only {robust_ok}/10");
    // And on clean data it still recovers the order.
    let clean = sim::ArmaSim {
        phi: &[1.2, -0.6],
        ..Default::default()
    }
    .simulate(3000, 42)
    .unwrap();
    let r = fh_order(
        &clean,
        FhConfig {
            statistic: Statistic::PacfRobust,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.order, 2);
}

#[test]
fn jirak_band_recovers_ar2_and_sparse_support() {
    let x = sim_arma(&[1.2, -0.6], &[], 4000, 42);
    let r = fh_order(
        &x,
        FhConfig {
            statistic: Statistic::JirakBand,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.order, 2, "JirakBand gave {}", r.order);
    // Jirak's selling point: on sparse AR the band reads the support directly.
    let s = sim_arma(&[0.5, 0.0, 0.0, 0.3], &[], 8000, 7);
    let rs = fh_order(
        &s,
        FhConfig {
            statistic: Statistic::JirakBand,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(rs.order, 4);
    let support: Vec<bool> = rs.studentized.iter().map(|t| t.abs() > rs.tau).collect();
    assert_eq!(&support[..4], &[true, false, false, true]);
}
