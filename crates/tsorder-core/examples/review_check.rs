use tsorder_core::*;
struct Rng(u64);
impl Rng {
    fn u(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 11) as f64 / (1u64 << 53) as f64
    }
    fn n(&mut self) -> f64 {
        let a = self.u().max(1e-300);
        let b = self.u();
        (-2.0 * a.ln()).sqrt() * (2.0 * std::f64::consts::PI * b).cos()
    }
}
fn main() {
    let mut r = Rng(3);
    let wn: Vec<f64> = (0..2000).map(|_| r.n()).collect();
    let shifted: Vec<f64> = wn.iter().map(|v| v + 50.0).collect();
    for (name, x) in [("ruido branco", &wn), ("ruido branco + 50", &shifted)] {
        for s in [
            Statistic::Pacf,
            Statistic::LongArOls,
            Statistic::LongArYw,
            Statistic::AcfBartlett,
        ] {
            let f = fh_order(
                x,
                FhConfig {
                    statistic: s,
                    ..Default::default()
                },
            )
            .unwrap();
            println!("{name:18} {:14} -> ordem {}", s.name(), f.order);
        }
    }
    let f = fh_order(
        &wn,
        FhConfig {
            threshold: Threshold::BicLike { c: -1.0 },
            ..Default::default()
        },
    )
    .unwrap();
    println!("c=-1: ordem {} tau {}", f.order, f.tau);
    let f = fh_order(
        &wn,
        FhConfig {
            threshold: Threshold::Fixed(-5.0),
            ..Default::default()
        },
    )
    .unwrap();
    println!("fixed(-5): ordem {} tau {}", f.order, f.tau);
    // Hwang condition p_n/(n eta^2) with eta = tau/sqrt(n)
    for n in [100usize, 1000, 10000, 100000] {
        let p = default_memory(n);
        let tau = Threshold::default().value(n, p);
        println!(
            "n={n} p_n={p} tau={tau:.2}  p_n/(n eta^2)=p_n/tau^2={:.2}",
            p as f64 / (tau * tau)
        );
    }
}
