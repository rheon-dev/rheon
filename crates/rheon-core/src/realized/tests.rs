use super::paper::{self, Dd};
use super::*;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A standard normal draw, Box-Muller over the LCG.
fn normal(state: &mut u64) -> f64 {
    let (u, v) = (lcg(state), lcg(state));
    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
}

/// One day of `n` returns of a stochastic-volatility diffusion, 1% a day:
/// an intraday U-shape times a log-OU factor of half-life `n/10` returns;
/// with `jumps`, five jumps at 10, 30, 50, 70 and 90% of the day of +8, −10,
/// +6, +12 and −7 local standard deviations (the realistic synthetic day the
/// jump study used).
pub(super) fn sv_day(n: usize, seed: u64, jumps: bool) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let base = 0.01 / (n as f64).sqrt();
    let kappa = std::f64::consts::LN_2 / (n as f64 / 10.0);
    let mut h = 0.0f64;
    let mut r: Vec<f64> = (0..n)
        .map(|j| {
            let u = (j as f64 + 0.5) / n as f64;
            let shape = 0.75 + 1.5 * (u - 0.5).powi(2) * 4.0;
            h = h * (-kappa).exp() + 0.3 * (2.0 * kappa).sqrt() * normal(&mut s);
            base * shape.sqrt() * h.exp() * normal(&mut s)
        })
        .collect();
    if jumps {
        for (at, size) in [
            (0.1, 8.0),
            (0.3, -10.0),
            (0.5, 6.0),
            (0.7, 12.0),
            (0.9, -7.0),
        ] {
            let j = (at * n as f64) as usize;
            let local = (r[j.saturating_sub(20)..(j + 20).min(n)]
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                / 40.0)
                .sqrt();
            r[j] += size * local;
        }
    }
    r
}

/// The distance of `a` from `b` in units in the last place of `b`.
fn ulps(a: f64, b: f64) -> f64 {
    let ulp = f64::from_bits(b.abs().to_bits() + 1) - b.abs();
    (a - b).abs() / ulp
}

/// `E|Z|^p` by its definition, `2 ∫_0^∞ x^p φ(x) dx`, by Simpson's rule
/// after `x = t³` (which makes the integrand smooth at 0 for these `p`):
/// a check of the constants that shares nothing with their closed form.
fn abs_moment(p: f64) -> f64 {
    let (top, n) = (12f64.cbrt() * 1.05, 40_000usize);
    let h = top / n as f64;
    let f = |t: f64| {
        let x = t * t * t;
        2.0 * 3.0 * t * t * x.powf(p) * (-0.5 * x * x).exp() / (std::f64::consts::TAU).sqrt()
    };
    let mut s = Dd::default();
    for i in 0..=n {
        let w = if i == 0 || i == n {
            1.0
        } else if i % 2 == 1 {
            4.0
        } else {
            2.0
        };
        s.add(w * f(i as f64 * h));
    }
    s.value() * h / 3.0
}

/// Each constant is the definition's `E|Z|^p` to quadrature's accuracy, and
/// the two tripower constants keep Γ's reflection, `μ_{4/3} μ_{2/3} = 2/3`,
/// to the rounding of the product.
#[test]
fn the_constants_are_the_moments_of_a_normal() {
    let mu1 = abs_moment(1.0);
    assert!((BV_SCALE * mu1 * mu1 - 1.0).abs() < 1e-12, "{mu1}");
    let mu43 = abs_moment(4.0 / 3.0);
    let mu23 = abs_moment(2.0 / 3.0);
    assert!((TQ_SCALE * mu43.powi(3) - 1.0).abs() < 1e-12, "{mu43}");
    assert!((TV_SCALE * mu23.powi(3) - 1.0).abs() < 1e-12, "{mu23}");
    // (μ_{4/3} μ_{2/3})^-3 = (3/2)^3 = 3.375, from the two constants.
    assert!(
        ulps(TQ_SCALE * TV_SCALE, 3.375) <= 2.0,
        "{}",
        TQ_SCALE * TV_SCALE
    );
    let pi = std::f64::consts::PI;
    // The sum cancels 5 to 0.61, so it is good to a rounding of 5, not of ϑ.
    assert!((BNS_THETA - (pi * pi / 4.0 + pi - 5.0)).abs() <= 4.0 * f64::EPSILON * 5.0);
}

/// The definitions on four returns, by hand: `RV = 30`; `BV = (π/2)(2 + 6
/// + 12)`; `TV` and `TQ` over the two triples; `Z` from them.
#[test]
fn the_definitions_on_four_returns_by_hand() {
    let m = paper::measures(&[1.0, -2.0, 3.0, -4.0]);
    assert_eq!(m.n, 4);
    assert_eq!(m.rv, 30.0);
    assert_eq!(m.bv, BV_SCALE * 20.0);
    let tv = TV_SCALE * (6f64.powf(2.0 / 3.0) + 24f64.powf(2.0 / 3.0));
    assert!(ulps(m.tv, tv) <= 2.0, "{} {tv}", m.tv);
    let tq = 4.0 * TQ_SCALE * (6f64.powf(4.0 / 3.0) + 24f64.powf(4.0 / 3.0));
    assert!(ulps(m.tq, tq) <= 4.0, "{} {tq}", m.tq);
    let z = 2.0 * (1.0 - m.bv / 30.0) / (BNS_THETA * (tq / (m.bv * m.bv)).max(1.0)).sqrt();
    assert!((m.z - z).abs() <= 1e-15 * z.abs().max(1.0), "{} {z}", m.z);
    // Too few returns for a product or a triple: the sums are empty.
    let one = paper::measures(&[0.5]);
    assert_eq!((one.rv, one.bv, one.tv, one.tq), (0.25, 0.0, 0.0, 0.0));
}

/// The double-double sum is the exact sum where it is representable, and
/// keeps what a plain sum rounds away.
#[test]
fn the_oracles_sum_keeps_what_a_plain_sum_drops() {
    assert_eq!(paper::sum([1.0, 1e-17, 1e-17, -1.0]), 2e-17);
    let terms: Vec<f64> = (1..=10_000).map(|i| 1.0 / i as f64).collect();
    let plain: f64 = terms.iter().sum();
    let dd = paper::sum(terms.iter().copied());
    let rev = paper::sum(terms.iter().rev().copied());
    assert_eq!(dd, rev, "the order of the terms does not show");
    assert!(ulps(plain, dd) >= 1.0, "the plain sum is off: {plain} {dd}");
}

/// Units: the variations scale with the square of the returns, the
/// quarticity with the fourth power, and the two test statistics not at
/// all (to the rounding a scale by a power of 2 leaves, none).
#[test]
fn the_statistics_are_free_of_the_returns_unit() {
    let r = sv_day(2_340, 3, true);
    let m = paper::measures(&r);
    let scaled: Vec<f64> = r.iter().map(|x| 8.0 * x).collect();
    let s = paper::measures(&scaled);
    assert_eq!(s.rv, 64.0 * m.rv);
    assert_eq!(s.bv, 64.0 * m.bv);
    assert!(ulps(s.tv, 64.0 * m.tv) <= 4.0);
    assert!(ulps(s.tq, 4096.0 * m.tq) <= 4.0);
    assert!((s.z - m.z).abs() <= 1e-13 * m.z.abs());
    let k = 100;
    let (l, ls) = (paper::lee_mykland(&r, k), paper::lee_mykland(&scaled, k));
    for (a, b) in l.iter().zip(&ls).skip(k - 1) {
        assert!((a - b).abs() <= 1e-14 * a.abs().max(1.0), "{a} {b}");
    }
}

/// On a day without jumps the three variations agree with one another, the
/// test does not reject and the adjustment's `max` takes `TQ/BV²` or 1 as
/// the day has it; with five jumps of 6 to 12 local deviations, `BV` and
/// `TV` stay near the jump-free day's variance while `RV` takes the jumps,
/// and `Z` rejects. Both sides of the test's threshold, 1.645 at 5%.
#[test]
fn the_test_rejects_a_day_with_jumps_and_not_one_without() {
    for seed in 0..3 {
        let clean = paper::measures(&sv_day(23_400, seed, false));
        assert!((clean.bv / clean.rv - 1.0).abs() < 0.03, "{clean:?}");
        assert!((clean.tv / clean.rv - 1.0).abs() < 0.03, "{clean:?}");
        assert!(clean.z.abs() < 1.645, "seed {seed}: {}", clean.z);
        let jumpy = paper::measures(&sv_day(2_340, seed, true));
        assert!(jumpy.rv > 1.08 * jumpy.bv, "{jumpy:?}");
        assert!(jumpy.z > 1.645, "seed {seed}: {}", jumpy.z);
    }
    // A day of constant volatility has TQ/BV² near 1 from either side:
    // both arms of the `max` are reached across seeds.
    let ratios: Vec<f64> = (0..20)
        .map(|seed| {
            let mut s = seed + 100;
            let r: Vec<f64> = (0..390).map(|_| 0.001 * normal(&mut s)).collect();
            let m = paper::measures(&r);
            m.tq / (m.bv * m.bv)
        })
        .collect();
    assert!(
        ratios.iter().any(|&q| q < 1.0) && ratios.iter().any(|&q| q > 1.0),
        "{ratios:?}"
    );
}

/// Lee and Mykland's statistic flags each planted jump, `c |L_i|` past the
/// paper's 1% critical value for one return (`2.576`), and is `NaN` until
/// `K − 1` returns precede the row; its volatility leaves the row's own
/// return out, so a jump does not shrink its own statistic.
#[test]
fn the_per_return_statistic_finds_the_planted_jumps() {
    let n = 2_340;
    let r = sv_day(n, 7, true);
    let k = 100;
    let l = paper::lee_mykland(&r, k);
    assert!(l[..k - 1].iter().all(|v| v.is_nan()));
    assert!(l[k - 1..].iter().all(|v| v.is_finite()));
    let c = (2.0 / std::f64::consts::PI).sqrt();
    for at in [0.1, 0.3, 0.5, 0.7, 0.9] {
        let j = (at * n as f64) as usize;
        assert!(c * l[j].abs() > 2.576, "{at}: {}", l[j]);
        assert_eq!(
            l[j].signum(),
            r[j].signum(),
            "the statistic keeps the return's sign"
        );
    }
    // Moving the row's own return changes no other row's volatility before
    // it: L_i scales with r_i alone.
    let mut moved = r.clone();
    moved[1_000] *= 3.0;
    let lm = paper::lee_mykland(&moved, k);
    assert!(ulps(lm[1_000], 3.0 * l[1_000]) <= 1.0);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&lm[..1_000]), bits(&l[..1_000]));
}
