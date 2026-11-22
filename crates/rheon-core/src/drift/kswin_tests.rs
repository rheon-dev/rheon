use super::*;
use crate::oracle;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// The tail built from products of doubles against the exact sum in
/// integers rounded once (`crate::oracle::ks_equal_tail`), and so the
/// critical `h` against the exact one, at every `r` to 60 and the levels a
/// detector is run at.
#[test]
fn the_tail_and_its_critical_h_are_the_exact_ones() {
    let mut worst = 0.0f64;
    for r in 1..=60 {
        for h in 0..=r {
            let (got, want) = (ks_equal_tail(r, h), oracle::ks_equal_tail(r, h));
            let rel = (got - want).abs() / want;
            worst = worst.max(rel);
            assert!(rel < 1e-13, "r = {r}, h = {h}: {got} against {want}");
        }
        for alpha in [1e-4, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5] {
            let exact = (1..=r)
                .find(|&h| oracle::ks_equal_tail(r, h) <= alpha)
                .unwrap_or(r + 1);
            assert_eq!(ks_critical_h(r, alpha), exact, "r = {r}, alpha = {alpha}");
        }
    }
    assert!(
        worst > 0.0,
        "some rounding seen: the test reads the doubles"
    );
    // River's defaults: `p ≤ 0.005` at `r = 30` needs `h ≥ 14`.
    assert_eq!(ks_critical_h(30, 0.005), 14);
}

/// `h` against the definition: the largest gap between the two samples'
/// counts at or below any value either holds, ties included.
#[test]
fn h_is_the_largest_gap_between_the_counts() {
    let mut s = 7u64;
    for case in 0..400 {
        let r = 1 + case % 40;
        // One decimal, so ties within and across the samples.
        let mut a: Vec<f64> = (0..r)
            .map(|_| (lcg(&mut s) * 10.0).round() / 10.0)
            .collect();
        let mut b: Vec<f64> = (0..r)
            .map(|_| ((lcg(&mut s) + 0.2) * 10.0).round() / 10.0)
            .collect();
        let want = a
            .iter()
            .chain(&b)
            .map(|v| {
                let ca = a.iter().filter(|x| *x <= v).count();
                let cb = b.iter().filter(|x| *x <= v).count();
                ca.abs_diff(cb)
            })
            .max()
            .unwrap();
        a.sort_by(f64::total_cmp);
        b.sort_by(f64::total_cmp);
        assert_eq!(ks_h(&a, &b), want, "case {case}");
    }
}

fn full(cfg: KswinCfg, s: &mut u64) -> Kswin {
    let mut k = Kswin::new(cfg);
    for _ in 0..cfg.window_size - 1 {
        assert!(!k.update(lcg(s)), "no test before the ring is full");
    }
    k
}

/// Floyd's draw takes `r` distinct older values, each about equally often.
#[test]
fn the_draw_takes_distinct_older_values_evenly() {
    let cfg = KswinCfg {
        window_size: 40,
        stat_size: 10,
        ..KswinCfg::default()
    };
    let mut k = Kswin::new(cfg);
    // The ring's values are their own positions, so a sample names them.
    for i in 0..40 {
        k.window.push_back(f64::from(i));
    }
    let mut counts = [0u32; 30];
    let draws = 30_000;
    for _ in 0..draws {
        let old = k.draw().to_vec();
        let mut seen = old.clone();
        seen.sort_by(f64::total_cmp);
        seen.dedup();
        assert_eq!(seen.len(), 10, "distinct");
        for v in old {
            assert!(v < 30.0, "from the older 30");
            counts[v as usize] += 1;
        }
    }
    // Each is taken with probability 1/3: 10,000 of 30,000, sd about 82.
    for (i, c) in counts.iter().enumerate() {
        assert!((f64::from(*c) - 10_000.0).abs() < 400.0, "index {i}: {c}");
    }
}

/// A shift in the values' distribution is flagged soon after it; under
/// `restart` the ring then starts again empty, and under the paper's form
/// it keeps the newest `r` values.
#[test]
fn a_shift_is_flagged_and_the_ring_restarts_or_keeps_the_newest() {
    for restart in [true, false] {
        let cfg = KswinCfg {
            restart,
            ..KswinCfg::default()
        };
        let mut s = 3u64;
        let mut k = full(cfg, &mut s);
        let mut false_alarms = 0;
        for _ in 0..2000 {
            false_alarms += usize::from(k.update(lcg(&mut s)));
        }
        assert!(false_alarms <= 10, "{false_alarms} on 2,000 null rows");
        k = full(cfg, &mut s);
        let at = (0..200)
            .position(|_| k.update(lcg(&mut s) + 0.8))
            .expect("a shift of 0.8 on a unit range is found");
        assert!(at < 40, "{at} rows");
        let left = k.window().len();
        assert_eq!(left, if restart { 0 } else { 30 }, "restart {restart}");
    }
}

/// The two gates, each on both sides: `h ≥ h*` and `D > 0.1`. At `r = 20`
/// and `alpha` the tail at `h = 2`, `h*` is 2, but `D = 2/20 = 0.1` is not
/// past 0.1, so `h = 2` is no flag and `h = 3` is one. At a level just
/// under that tail `h*` is 3. Each row's flag is held to its own `h`, read
/// from its samples by the definition.
#[test]
fn both_gates_hold_on_both_sides() {
    let tail2 = ks_equal_tail(20, 2);
    assert_eq!(ks_critical_h(20, tail2), 2, "the level itself admits h = 2");
    assert_eq!(ks_critical_h(20, tail2 * (1.0 - 1e-12)), 3);
    let cfg = KswinCfg {
        alpha: tail2,
        window_size: 60,
        stat_size: 20,
        seed: 5,
        restart: false,
    };
    let mut s = 21u64;
    let mut k = full(cfg, &mut s);
    let mut seen = [0usize; 4];
    for i in 0..4000 {
        let x = (lcg(&mut s) * 4.0).round() + if i % 400 < 40 { 1.0 } else { 0.0 };
        let Some((old, new)) = k.peek(x) else {
            assert!(!k.update(x), "row {i}: no test while the ring fills");
            continue;
        };
        let h = old
            .iter()
            .chain(&new)
            .map(|v| {
                let ca = old.iter().filter(|x| *x <= v).count();
                let cb = new.iter().filter(|x| *x <= v).count();
                ca.abs_diff(cb)
            })
            .max()
            .unwrap();
        let flag = k.update(x);
        assert_eq!(flag, h >= 3, "row {i}: h = {h}");
        if h <= 3 {
            seen[h] += 1;
        }
    }
    assert!(seen[2] > 0 && seen[3] > 0, "both sides met: {seen:?}");
}

/// A detector saved mid-stream, its generator with it, goes on to the bit.
#[test]
fn a_saved_detector_goes_on_to_the_bit() {
    let cfg = KswinCfg {
        alpha: 0.05,
        ..KswinCfg::default()
    };
    let mut s = 9u64;
    let mut a = Kswin::new(cfg);
    for _ in 0..450 {
        a.update(lcg(&mut s));
    }
    let mut b: Kswin = rmp_serde::from_slice(&rmp_serde::to_vec_named(&a).unwrap()).unwrap();
    let mut flags = 0;
    for i in 0..3000 {
        let x = lcg(&mut s) + if i > 1500 { 0.3 } else { 0.0 };
        let (fa, fb) = (a.update(x), b.update(x));
        assert_eq!(fa, fb, "row {i}");
        flags += usize::from(fa);
    }
    assert!(flags > 0);
    assert_eq!(a, b);
}

/// The state's numbers: the ring holds `W` values once full and never
/// more, beside the generator and the settings, which the spec's cap counts
/// as `8 W + 48` bytes.
#[test]
fn the_state_is_its_ring() {
    let mut s = 1u64;
    let mut k = full(KswinCfg::default(), &mut s);
    for _ in 0..500 {
        k.update(lcg(&mut s));
        assert!(k.window().len() <= 100);
    }
    assert_eq!(Kswin::state_bytes(100), 848);
    assert_eq!(Kswin::state_bytes(1000), 8048);
}
