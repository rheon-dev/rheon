use super::paper::{self, Dd};
use super::*;
use crate::oracle;

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

// --- the online form: one pass and decay (docs/PLAN.md task 499, part 2) ---

/// A row as the model takes it: clock step, returns, weight.
type Row = (f64, Vec<Option<f64>>, f64);

fn cfg(n: usize, half_life: f64) -> RealizedCfg {
    RealizedCfg {
        n_targets: n,
        decay: Decay::Halflife(half_life),
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window: None,
        lm_beta: DEFAULT_LM_BETA,
        max_state_bytes: None,
    }
}

fn run(m: &mut Realized, rows: &[Row]) -> Vec<Step> {
    rows.iter()
        .map(|(d, y, w)| m.step(&[], y, *d, *w))
        .collect()
}

fn unit_rows(r: &[f64]) -> Vec<Row> {
    r.iter().map(|&x| (1.0, vec![Some(x)], 1.0)).collect()
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// `|a − b|` in units of `cond · ε`, `cond = (RV + BV)/|RV − BV|`: the
/// test statistic is a difference of the two, so its rounding is theirs
/// amplified by that much.
fn z_gap(a: f64, b: f64, rv: f64, bv: f64) -> f64 {
    let cond = (rv + bv) / (rv - bv).abs();
    (a - b).abs() / (cond * f64::EPSILON * b.abs().max(1.0))
}

/// Without decay and at unit weights the online measures of every prefix
/// are the papers' on the returns before the row, `n − 1` products and `n
/// − 2` triples, to the rounding of the terms: on realistic days of 2,340
/// and 23,400 returns, with and without jumps. The largest gaps measured
/// (ulps): RV 0, BV 0, TV 10, TQ 20 (the cube roots against the papers'
/// powers), `Z` 0.3 in units of `cond · ε`, `L` 0.
#[test]
fn one_pass_is_the_papers_measures_on_every_prefix() {
    let mut worst = [0.0f64; 6];
    for (n, seed, jumps) in [(23_400, 1, true), (23_400, 2, false), (2_340, 3, true)] {
        let r = sv_day(n, seed, jumps);
        let mut m = Realized::new(cfg(1, f64::INFINITY)).unwrap();
        let steps = run(&mut m, &unit_rows(&r));
        for i in (3..n).step_by(97).chain([n - 1]) {
            let p = paper::measures(&r[..i]);
            let o = &steps[i].pred;
            let g = [
                ulps(o[0], p.rv),
                ulps(o[1], p.bv),
                ulps(o[2], p.tv),
                ulps(o[3], p.tq),
                z_gap(o[4], p.z, p.rv, p.bv),
            ];
            for (w, v) in worst.iter_mut().zip(g) {
                *w = w.max(v);
            }
            assert_eq!(steps[i].n_eff, i as f64);
            if i % 997 == 3 || i == n - 1 {
                let l = paper::lee_mykland(&r[..=i], i + 1)[i];
                worst[5] = worst[5].max(ulps(o[5], l));
            }
        }
        // The final state's own weight is every return's.
        let mut own = Vec::new();
        assert!(m.target_n_eff_into(&mut own));
        assert_eq!(own, vec![n as f64]);
    }
    let bound = [2.0, 4.0, 16.0, 32.0, 8.0, 8.0];
    for (k, (w, b)) in worst.iter().zip(bound).enumerate() {
        assert!(w <= &b, "output {k}: {w} ulps, bound {b} ({worst:?})");
    }
}

/// An irregular clock, steps of 1/64 to 4 units in 64ths so that their
/// sums are exact and the definition's ages carry no rounding of their
/// own, and weights in [0.5, 1.5].
fn irregular(r: &[f64], seed: u64) -> Vec<Row> {
    let mut s = seed;
    r.iter()
        .map(|&x| {
            let d = (1.0 + (lcg(&mut s) * 256.0).floor()) / 64.0;
            (d, vec![Some(x)], 0.5 + lcg(&mut s))
        })
        .collect()
}

/// Under a decay the online sums are the definition's weighted sums, each
/// term decayed by its own age taken in one power: half-lives of 50 and 500
/// clock units, an irregular clock and unequal weights. The products are
/// of consecutive returns, each at the weight of its later row; `n`, which
/// `TQ` and `Z` read, is Kish's effective count of the decayed weights
/// (`(Σ λw)² / Σ (λw)²`). The online form multiplies a factor a row, so its gap
/// grows with the rows a half-life holds.
///
/// Each output is held to a bound derived from both computations' rounding
/// ([`Bounds`]), not to a measured gap: the factors are the platform's
/// `exp2`, and Windows Server 2022's is up to 2 ulps off in 40% of calls
/// where Apple's and glibc's are within one, so a fixed bound measured on
/// one platform failed on that one (56 ulps against 32 on `RV`, 112
/// against 64 on `TQ`). Measured on aarch64 (macOS), the largest share of
/// its bound each output reached: RV 0.049, BV 0.018, TV 0.14, TQ 0.13,
/// `Z` 0.023, `L` 0.018, the weight 0.11. Server 2022's largest gaps in
/// ulps were 2.7 to 3.8 times the Mac's, on its own run.
#[test]
fn decay_is_the_weighted_definition() {
    let r = sv_day(5_000, 11, true);
    let rows = irregular(&r, 5);
    let mut worst = [0.0f64; 7];
    for h in [50.0, 500.0] {
        let mut m = Realized::new(cfg(1, h)).unwrap();
        let steps = run(&mut m, &rows);
        let bounds = Bounds::of(&rows, &steps, h);
        let mut t = 0.0;
        let learned: Vec<(f64, f64, f64)> = rows
            .iter()
            .map(|(d, y, w)| {
                t += d;
                (t, y[0].unwrap(), *w)
            })
            .collect();
        for i in (3..rows.len()).step_by(251) {
            let (p, _, s2) = paper::weighted(&learned[..i], h);
            let o = &steps[i].pred;
            let l = learned[i].1 / s2.sqrt();
            let rho = bounds.at(i, p.rv, p.bv, p.tq);
            let want = [
                p.rv,
                p.bv,
                p.tv,
                p.tq,
                p.z,
                l,
                sum_of_weights(&rows[..i], h),
            ];
            let got = [o[0], o[1], o[2], o[3], o[4], o[5], steps[i].n_eff];
            for k in 0..7 {
                if got[k] == want[k] {
                    continue;
                }
                let share = (got[k] - want[k]).abs() / (rho[k] * want[k].abs());
                assert!(
                    share <= 1.0,
                    "h {h}, row {i}, output {k}: {} against {}, past its bound {}",
                    got[k],
                    want[k],
                    rho[k]
                );
                worst[k] = worst[k].max(share);
            }
        }
        let (_, nw, _) = paper::weighted(&learned, h);
        let mut own = Vec::new();
        m.target_n_eff_into(&mut own);
        let rho = bounds.nw[rows.len()] / own[0] + (oracle::LIBM + 3.0) * oracle::U;
        assert!(
            (own[0] - nw).abs() <= rho * nw,
            "h {h}: {} {nw}, past {rho} of it",
            own[0]
        );
    }
    eprintln!("shares of the bounds: {worst:?}");
}

/// The bounds [`decay_is_the_weighted_definition`] holds each output to,
/// relative, from the rounding of both computations to first order in `u`
/// (`U`), each call to `exp2`, `powf` or `cbrt` within `LIBM` units of it.
///
/// - **The decay.** The online sums multiply a factor a row, `exp2(−d/h)`,
///   off the exact one by `LIBM` and by `ln 2 · d/h` from its argument's
///   rounding, and round the product (`κ = (LIBM + 1 + 2 ln 2 · d/h) u` a
///   row, [`oracle::decayed_error`]); the definition takes one power a
///   term, off by `LIBM` and by `ln 2 · age/h`, the sum of that term's
///   `ln 2 · d/h`, whence the 2. The weight `n_eff` rounds its sum too, and
///   `Σw²` decays by `f` twice, so twice the factor's errors.
/// - **The terms**, both sides' roundings in units of `u`: `RV` 2 and 3
///   and `BV` with its scale 3 and 4; `TV` `6 LIBM + 7` (three cube roots,
///   the triple, its square, the weight and the scale) and `3 LIBM + 6`
///   (three powers, five products and the scale); `TQ` `12 LIBM + 14` and
///   `3 LIBM + 7`, and the error of the weight it is multiplied by; the
///   weight's own 2, a product and an addition. Each read rounds once more
///   on each side.
/// - **The ratios.** `Z = √n (1 − BV/RV) / √(ϑ max(1, TQ/BV²))`: `n =
///   (Σw)²/Σw²` within `2ρ_W + ρ_{W2}`, `1 − BV/RV` within `(BV/RV)(ρ_BV +
///   ρ_RV)` of itself over `|1 − BV/RV|`, `TQ/BV²` within `ρ_TQ + 2ρ_BV`,
///   and the square roots halve their arguments' errors. `L = r / √(Σ
///   a a' / Σ w)` within half the errors of its two sums.
struct Bounds {
    rv: Vec<f64>,
    bv: Vec<f64>,
    tv: Vec<f64>,
    tq: Vec<f64>,
    nw: Vec<f64>,
    w: Vec<f64>,
    w2: Vec<f64>,
    /// The values the bounds are of, read off the run.
    vals: Vec<[f64; 7]>,
}

impl Bounds {
    fn of(rows: &[Row], steps: &[Step], h: f64) -> Self {
        use oracle::{LIBM, U, decayed_error};
        let dec = Decay::Halflife(h);
        let ln2 = std::f64::consts::LN_2;
        let kappa = |d: f64, roundings: f64| (LIBM + roundings + 2.0 * ln2 * d / h) * U;
        let finite = |v: f64| if v.is_finite() { v.abs() } else { 0.0 };
        // The sums' sizes as read at each row, and `Σw²` replayed.
        let mut w2 = 0.0;
        let vals: Vec<[f64; 7]> = steps
            .iter()
            .zip(rows)
            .map(|(s, (d, _, w))| {
                let n = finite(s.n_eff);
                let o = &s.pred;
                let tq = if n > 0.0 { finite(o[3]) / n } else { 0.0 };
                let v = [finite(o[0]), finite(o[1]), finite(o[2]), tq, n, n, w2];
                let f = dec.factor(*d);
                w2 = f * f * w2 + w * w;
                v
            })
            .collect();
        let run = |k: usize, roundings: f64, twice: bool| {
            decayed_error(rows.iter().zip(&vals).map(|((d, _, _), v)| {
                let f = dec.factor(*d);
                if twice {
                    (v[k], f * f, 2.0 * kappa(*d, roundings))
                } else {
                    (v[k], f, kappa(*d, roundings))
                }
            }))
        };
        Self {
            rv: run(0, 1.0, false),
            bv: run(1, 1.0, false),
            tv: run(2, 1.0, false),
            tq: run(3, 1.0, false),
            nw: run(4, 1.0, false),
            w: run(5, 2.0, false),
            w2: run(6, 1.0, true),
            vals,
        }
    }

    /// Each output's bound at row `i`, relative to the definition's value,
    /// `rv`, `bv` and `tq` the definition's measures there.
    fn at(&self, i: usize, rv: f64, bv: f64, tq: f64) -> [f64; 7] {
        use oracle::{LIBM, U, up};
        let v = &self.vals[i];
        // A sum's decay error over its size, its terms' roundings, the
        // definition's one power a term and each side's read.
        let rho = |b: &[f64], k: usize, terms: f64| {
            let decay = if v[k] > 0.0 { b[i] / v[k] } else { 0.0 };
            decay + (LIBM + terms + 2.0) * U
        };
        let r_nw = rho(&self.nw, 4, 1.0);
        let r_w2 = rho(&self.w2, 6, 2.0 * LIBM + 3.0);
        let r_rv = rho(&self.rv, 0, 5.0);
        let r_bv = rho(&self.bv, 1, 7.0);
        let r_tv = rho(&self.tv, 2, 9.0 * LIBM + 13.0);
        let r_tq = rho(&self.tq, 3, 15.0 * LIBM + 21.0) + r_nw + 2.0 * U;
        let r_w = rho(&self.w, 5, 2.0);
        let r_n = 2.0 * r_nw + r_w2 + 4.0 * U;
        let ratio = bv / rv;
        let r_diff = ratio * (r_bv + r_rv + U) / (1.0 - ratio).abs() + U;
        let r_tq_bv = if tq / (bv * bv) > 1.0 {
            r_tq + 2.0 * r_bv + 3.0 * U
        } else {
            0.0
        };
        let r_z = 0.5 * r_n + r_diff + 0.5 * r_tq_bv + 8.0 * U;
        // `L` against the products' sum and `Σ w`, which the weight's sum
        // bounds (its first row the one it has more).
        let r_l = 0.5 * (r_bv + r_nw) + 4.0 * U;
        [r_rv, r_bv, r_tv, r_tq, r_z, r_l, r_w].map(|r| r * up(16))
    }
}

/// Every row's weight, each decayed by its age from the last row.
fn sum_of_weights(rows: &[Row], h: f64) -> f64 {
    let mut t = 0.0;
    let ts: Vec<f64> = rows
        .iter()
        .map(|(d, _, _)| {
            t += d;
            t
        })
        .collect();
    paper::sum(
        rows.iter()
            .zip(&ts)
            .map(|((_, _, w), ti)| (-((t - ti) / h)).exp2() * w),
    )
}

/// Rows of weight 0, each carrying a return that is not a number, an
/// infinity, 1e300 or an ordinary value, learn nothing: without decay
/// the real rows report what they report without them, to the bit, `L`
/// included, since the products reach across them to the last return
/// learned. Under a decay a row of weight 0 handed to the model directly
/// ages the sums over its step, so splitting a step in two moves the
/// result only by the rounding of the two factors against the one: the same
/// rounding taken every third row and held for the rows a half-life of 100
/// keeps, within 256 ulps (the stream does not hand such rows over, which is
/// what keeps the bank's result to the bit). A stream that opens with such
/// rows leaves no NaN.
#[test]
fn a_row_of_weight_zero_learns_nothing() {
    let r = sv_day(2_340, 4, true);
    let junk = [f64::NAN, f64::INFINITY, 1e300, 0.37, -1e-300];
    for h in [f64::INFINITY, 100.0] {
        let clean = unit_rows(&r);
        let mut with = Vec::new();
        let mut at = Vec::new();
        for (i, row) in clean.iter().enumerate() {
            if i % 3 == 0 {
                with.push((0.5, vec![Some(junk[i % junk.len()])], 0.0));
                with.push((0.5, row.1.clone(), 1.0));
            } else {
                with.push(row.clone());
            }
            at.push(with.len() - 1);
        }
        let a = run(&mut Realized::new(cfg(1, h)).unwrap(), &clean);
        let b = run(&mut Realized::new(cfg(1, h)).unwrap(), &with);
        for (i, &j) in at.iter().enumerate() {
            // Under a decay the row of weight 0 just before a row has aged
            // the sums half a step early; compare the rows after a real row.
            if !h.is_infinite() && i % 3 == 0 {
                continue;
            }
            if h.is_infinite() {
                assert_eq!(bits(&a[i].pred), bits(&b[j].pred), "row {i}");
            } else {
                let (x, y) = (&a[i].pred, &b[j].pred);
                for k in [0, 1, 2, 3, 5] {
                    assert!(
                        (y[k].is_nan() && x[k].is_nan()) || ulps(y[k], x[k]) <= 256.0,
                        "row {i} output {k}: {x:?} {y:?}"
                    );
                }
                assert!(
                    (y[4].is_nan() && x[4].is_nan()) || z_gap(y[4], x[4], x[0], x[1]) <= 64.0,
                    "row {i}: {x:?} {y:?}"
                );
            }
        }
    }
    // A stream that opens with rows of weight 0: nothing learned, nothing
    // NaN in the state, and the first returns then learned as a fresh
    // stream's.
    let mut m = Realized::new(cfg(1, 50.0)).unwrap();
    for v in junk {
        m.step(&[], &[Some(v)], 1.0, 0.0);
    }
    let s = &m.series[0];
    assert!(s.hi.iter().chain(&s.lo).all(|v| *v == 0.0) && s.seen == 0);
    assert_eq!(m.n_eff(), 0.0);
    let first = m.step(&[], &[Some(0.01)], 1.0, 1.0);
    assert!(first.pred[..4].iter().all(|v| *v == 0.0));
    assert!(first.pred[4].is_nan() && first.pred[5].is_nan());
}

/// A target absent on a row has no return there: its products reach across
/// the gap to its last return, and the other target is untouched, to the
/// bit of the same series alone.
#[test]
fn an_absent_return_is_skipped_by_its_own_series_alone() {
    let (r0, r1) = (sv_day(2_340, 8, true), sv_day(2_340, 9, false));
    let rows: Vec<Row> = (0..r0.len())
        .map(|i| (1.0, vec![Some(r0[i]), (i % 3 != 1).then_some(r1[i])], 1.0))
        .collect();
    let mut both = Realized::new(cfg(2, f64::INFINITY)).unwrap();
    let steps = run(&mut both, &rows);
    let alone = run(
        &mut Realized::new(cfg(1, f64::INFINITY)).unwrap(),
        &unit_rows(&r0),
    );
    let present: Vec<f64> = (0..r1.len())
        .filter(|i| i % 3 != 1)
        .map(|i| r1[i])
        .collect();
    for i in 0..rows.len() {
        assert_eq!(
            bits(&steps[i].pred[..7]),
            bits(&alone[i].pred),
            "row {i} {:?} {:?}",
            &steps[i].pred[..7],
            alone[i].pred
        );
    }
    for i in (5..rows.len()).step_by(113) {
        let before = (0..i).filter(|j| j % 3 != 1).count();
        let p = paper::measures(&present[..before]);
        let o = &steps[i].pred[7..];
        assert!(
            ulps(o[1], p.bv) <= 4.0 && ulps(o[3], p.tq) <= 32.0,
            "row {i} {o:?} {p:?}"
        );
        assert_eq!(o[5].is_nan(), i % 3 == 1, "no return, no statistic");
    }
    let mut own = Vec::new();
    both.target_n_eff_into(&mut own);
    assert_eq!(own, vec![2_340.0, present.len() as f64]);
}

/// A save at any point, the first rows included, goes on to the bit, under
/// a decay, with two targets and an absent return.
#[test]
fn a_saved_state_goes_on_to_the_bit() {
    let (r0, r1) = (sv_day(600, 12, true), sv_day(600, 13, true));
    let mut s = 3;
    let rows: Vec<Row> = (0..r0.len())
        .map(|i| {
            let d = -lcg(&mut s).ln();
            (
                d,
                vec![Some(r0[i]), (i % 7 != 2).then_some(r1[i])],
                0.5 + lcg(&mut s),
            )
        })
        .collect();
    for cut in [0, 1, 2, 3, 4, 97, 450] {
        let mut m = Realized::new(cfg(2, 40.0)).unwrap();
        run(&mut m, &rows[..cut]);
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let mut back = Realized::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        assert_eq!(m, back);
        for (d, y, w) in &rows[cut..] {
            let a = m.step(&[], y, *d, *w);
            let b = back.step(&[], y, *d, *w);
            assert_eq!(bits(&a.pred), bits(&b.pred), "cut {cut}");
            assert_eq!(a.n_eff.to_bits(), b.n_eff.to_bits());
        }
        assert_eq!(m, back, "cut {cut}");
    }
}

/// Out of sample: a row's own return moves none of its outputs but its own
/// statistic `L`, which it scales; the next row's outputs move.
/// `predict_with` is the step without the step.
#[test]
fn a_row_reports_from_the_returns_before_it() {
    let r = sv_day(500, 14, false);
    let mut m = Realized::new(cfg(1, 200.0)).unwrap();
    run(&mut m, &unit_rows(&r[..400]));
    let ahead = m.predict_with(&[], &[Some(r[400])], 1.0);
    let mut twin = m.clone();
    let a = m.step(&[], &[Some(r[400])], 1.0, 1.0);
    let b = twin.step(&[], &[Some(3.0 * r[400])], 1.0, 1.0);
    assert_eq!(bits(&a.pred), bits(&ahead.pred));
    assert_eq!(a.n_eff, ahead.n_eff);
    assert_eq!(bits(&a.pred[..5]), bits(&b.pred[..5]));
    assert!(ulps(b.pred[5], 3.0 * a.pred[5]) <= 1.0);
    let (na, nb) = (
        m.step(&[], &[Some(r[401])], 1.0, 1.0),
        twin.step(&[], &[Some(r[401])], 1.0, 1.0),
    );
    assert!(nb.pred[0] > na.pred[0] && nb.pred[1] > na.pred[1]);
    // `predict` has no return, so no statistic, and the rest as the step.
    let p = m.predict(&[], 1.0);
    let s = m.clone().step(&[], &[Some(r[402])], 1.0, 1.0);
    assert_eq!(bits(&p.pred[..5]), bits(&s.pred[..5]));
    assert!(p.pred[5].is_nan());
}

/// `min_weight` gates each target on its own returns' weight, from both
/// sides: a target reports once its weight reaches the floor, not a row
/// before.
#[test]
fn min_weight_gates_each_series_on_its_own_weight() {
    let mut c = cfg(2, f64::INFINITY);
    c.min_weight = 3.0;
    let mut m = Realized::new(c).unwrap();
    let y = |i: usize| vec![Some(0.01 * (i as f64 + 1.0)), (i >= 2).then_some(-0.02)];
    let steps: Vec<Step> = (0..6).map(|i| m.step(&[], &y(i), 1.0, 1.0)).collect();
    for (i, s) in steps.iter().enumerate() {
        assert_eq!(s.pred[0].is_nan(), i < 3, "row {i}");
        assert_eq!(s.pred[7].is_nan(), i < 5, "row {i}");
    }
}

/// The state is `8 + 144 m` bytes whatever the stream's length, and the
/// settings refuse what they must, each naming the setting: no target, a
/// bad decay or floor, and a cap of 0, past 64 MiB, or below the state.
#[test]
fn the_state_is_a_few_numbers_a_series_and_the_settings_are_checked() {
    assert_eq!(Realized::state_bytes(1, None, None, false), 152);
    assert_eq!(Realized::state_bytes(4, None, None, false), 584);
    let mut m = Realized::new(cfg(3, 10.0)).unwrap();
    let r = sv_day(23_400, 15, true);
    let rows: Vec<Row> = r.iter().map(|&x| (1.0, vec![Some(x); 3], 1.0)).collect();
    run(&mut m, &rows);
    assert_eq!(m.series.len(), 3);
    let numbers = 1 + m.series.len() * (2 * SUMS + 4);
    assert_eq!(
        8 * numbers as u64,
        Realized::state_bytes(3, None, None, false)
    );
    let refuse = |c: RealizedCfg, says: &str| {
        let e = Realized::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    refuse(cfg(0, 10.0), "at least one target");
    refuse(cfg(1, 0.0), "half_life must be above 0");
    let mut c = cfg(1, 10.0);
    c.min_weight = -1.0;
    refuse(c, "min_weight must be at least 0");
    for (cap, says) in [
        (0, "must be above 0"),
        ((64 << 20) + 1, "at most"),
        (151, "would hold 152 bytes"),
    ] {
        let mut c = cfg(1, 10.0);
        c.max_state_bytes = Some(cap);
        refuse(c, says);
    }
    let mut c = cfg(1, 10.0);
    c.max_state_bytes = Some(152);
    assert!(Realized::new(c).is_ok(), "the cap itself holds");
}

/// A restored state of the wrong shape is refused as damaged.
#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut m = Realized::new(cfg(2, 10.0)).unwrap();
    m.series.pop();
    let e = Realized::restore(&m.state()).unwrap_err().to_string();
    assert!(e.contains("damaged"), "{e}");
    let mut m = Realized::new(cfg(1, 10.0)).unwrap();
    m.series[0].seen = 3;
    assert!(Realized::restore(&m.state()).is_err());
}

/// The count of returns `n` that `TQ`, `Z` and the flag read: without a
/// decay and at equal weights the count itself, exactly; under a decay
/// Kish's effective count `(Σ λw)² / Σ (λw)²`, the count the decayed
/// weights are worth (at unit weights near `(1 + λ)/(1 − λ)`, about twice
/// their sum); and unmoved by the scale of the weights, as a count is.
#[test]
fn n_is_the_count_of_returns_and_kish_under_a_decay() {
    let r = sv_day(3_000, 16, false);
    let mut m = Realized::new(cfg(1, f64::INFINITY)).unwrap();
    run(&mut m, &unit_rows(&r));
    assert_eq!(m.series[0].count(), 3_000.0);
    let h = 100.0;
    let mut m = Realized::new(cfg(1, h)).unwrap();
    run(&mut m, &unit_rows(&r));
    let (w, n) = (m.series[0].weight(), m.series[0].count());
    let lam = (-1.0 / h).exp2();
    // The geometric sums of 3,000 unit weights, by their closed forms.
    let big_n = lam.powi(3_000);
    let w_sum = (1.0 - big_n) / (1.0 - lam);
    let w2_sum = (1.0 - big_n * big_n) / (1.0 - lam * lam);
    let n_want = w_sum * w_sum / w2_sum;
    assert!((w - w_sum).abs() < 1e-11 * w_sum, "{w} {w_sum}");
    assert!((n - n_want).abs() < 1e-11 * n_want, "{n} {n_want}");
    assert!(
        n > w,
        "Kish's count of equal weights decayed is {n}, above their sum {w}"
    );
    let heavy: Vec<Row> = r.iter().map(|&x| (1.0, vec![Some(x)], 7.0)).collect();
    let mut m7 = Realized::new(cfg(1, h)).unwrap();
    run(&mut m7, &heavy);
    let n7 = m7.series[0].count();
    assert!((n7 - n).abs() < 1e-12 * n, "{n7} {n}");
}

/// The sd and the share past 2.326 (a one-sided 1% test) of `bns` at the
/// last of `n` rows of iid normal returns, over `reps` streams: `weight`
/// gives row `t`'s weight.
fn bns_null(c: &RealizedCfg, n: usize, reps: u64, weight: impl Fn(usize) -> f64) -> (f64, f64) {
    let mut zs = Vec::with_capacity(reps as usize);
    for rep in 0..reps {
        let mut s = 0x5EED_0000 + rep;
        let mut m = Realized::new(c.clone()).unwrap();
        let mut z = f64::NAN;
        for t in 0..n {
            let r = 0.001 * normal(&mut s);
            z = m.step(&[], &[Some(r)], 1.0, weight(t)).pred[4];
        }
        zs.push(z);
    }
    let k = zs.len() as f64;
    let mean = zs.iter().sum::<f64>() / k;
    let sd = (zs.iter().map(|z| (z - mean).powi(2)).sum::<f64>() / (k - 1.0)).sqrt();
    let size = zs.iter().filter(|z| **z > 2.326).count() as f64 / k;
    (sd, size)
}

/// `bns` is N(0, 1) without jumps under a decay, at a constant weight
/// below 1 and at unequal weights in a window, not only at the papers'
/// unit weights (review W13, B1 and B2). Its statistic's law is the
/// oracle: the simulated null over enough streams to resolve a 1% test's
/// size. `√n` is Kish's count, the variance of `1 − BV/RV` being `ϑ Σw² /
/// (Σw)²` at a constant volatility; `TQ/BV²` is then 1 there only with `TQ`
/// scaled by the weight `W`, the period its sum is of. With Kish's count in
/// `TQ` as well the sd was 1/√(1 + λ), 0.707 under any decay, and the size
/// 0.0009; at a weight of 0.25 the sd was 0.5; and under a window of
/// weights from 0.1 to 1.9 the count of returns in front put it near
/// √(1000/712) = 1.19, 712 being Kish's count of those weights.
/// The weights vary slowly from row to row, as a decay's do: `ϑ` is the
/// variance of the products and of their covariance with the next one,
/// which share a return, so weights that jump between neighbours (0.5 and
/// 1.5 in turn, measured: sd 1.23) are outside the paper's asymptotics.
/// The statistic is also unmoved by the weights' scale.
#[test]
fn bns_is_standard_normal_under_a_decay_and_weights() {
    check_bns_null(2_000, 0.06, (0.005, 0.02));
}

/// [`bns_is_standard_normal_under_a_decay_and_weights`] at 20,000 streams,
/// where the size is resolved to 0.0008 and the sd to 0.005.
#[test]
#[ignore = "extended: 20,000 simulated streams"]
fn bns_is_standard_normal_under_a_decay_and_weights_finely() {
    check_bns_null(20_000, 0.035, (0.008, 0.016));
}

/// A null stream's form: its name, settings, rows and each row's weight.
type NullCase<'a> = (&'a str, &'a RealizedCfg, usize, fn(usize) -> f64);

fn check_bns_null(reps: u64, sd_tol: f64, size: (f64, f64)) {
    let decayed = cfg(1, 100.0);
    let plain = cfg(1, f64::INFINITY);
    let mut windowed = cfg(1, f64::INFINITY);
    windowed.window_rows = Some(1_000);
    let cases: [NullCase; 3] = [
        ("half-life 100", &decayed, 1_500, |_| 1.0),
        ("weight 0.25", &plain, 1_000, |_| 0.25),
        ("window, weights 0.1 to 1.9", &windowed, 1_500, |t| {
            1.0 + 0.9 * (std::f64::consts::TAU * t as f64 / 400.0).sin()
        }),
    ];
    for (name, c, n, w) in cases {
        let (sd, share) = bns_null(c, n, reps, w);
        assert!((sd - 1.0).abs() < sd_tol, "{name}: sd {sd}");
        assert!(share > size.0 && share < size.1, "{name}: size {share}");
    }
    // the weights' scale moves no statistic
    let r = sv_day(2_000, 21, true);
    for h in [f64::INFINITY, 100.0] {
        let mut z = Vec::new();
        for w in [1.0, 0.25, 4.0] {
            let rows: Vec<Row> = r.iter().map(|&x| (1.0, vec![Some(x)], w)).collect();
            let mut m = Realized::new(cfg(1, h)).unwrap();
            z.push(run(&mut m, &rows).last().unwrap().pred[4]);
        }
        assert!(
            (z[1] - z[0]).abs() < 1e-12 && (z[2] - z[0]).abs() < 1e-12,
            "{h}: {z:?}"
        );
    }
}
