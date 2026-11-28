//! Tests that hold three lines of the garch family to numbers a mutation of
//! them would change (the survivors of W11's changed-lines pass): EGARCH's
//! variance bounds, `ln Γ` at its series' edge, and the asymmetric term of
//! `ln |μ|`.

use super::dist::ln_gamma;
use super::filter::Filter;
use super::guard::log_mu;
use super::math;
use super::*;

/// EGARCH's variance is held within a factor of 10⁶ of the 0.94-weighted mean
/// of `e²` below, and above it only logarithmically: `s` inside is itself,
/// below it is the lower bound with no slope, above it the upper bound plus
/// the log of the excess, with the excess's slope.
#[test]
fn the_egarch_variance_is_bounded_below_and_grows_only_logarithmically_above() {
    let f = Filter::new(GarchVol::Egarch, 4, 0, 1.0);
    let mut dl = [1.0; 4];
    let s = f.predict(GarchVol::Egarch, &[0.0, 0.1, 0.1, 0.1], &[], &mut dl);
    assert_eq!(s, 1.0);
    assert!(dl.iter().any(|d| *d != 0.0), "a slope inside");
    // Just inside each bound.
    let inside_hi = f.predict(GarchVol::Egarch, &[13.8, 0.0, 0.0, 0.0], &[], &mut dl);
    assert!(
        (inside_hi - math::exp(13.8)).abs() < 1e-6 * inside_hi,
        "{inside_hi}"
    );
    let inside_lo = f.predict(GarchVol::Egarch, &[-13.8, 0.0, 0.0, 0.0], &[], &mut dl);
    assert!(
        (inside_lo - math::exp(-13.8)).abs() < 1e-6 * inside_lo,
        "{inside_lo}"
    );
    // Below: the floor, flat.
    dl = [1.0; 4];
    let low = f.predict(GarchVol::Egarch, &[-20.0, 0.0, 0.0, 0.0], &[], &mut dl);
    assert_eq!(low, 1e-6);
    assert_eq!(dl, [0.0; 4]);
    // Above: the ceiling and the log of the excess over it, moving as `ln
    // s` does (`∂ ln s/∂ω` is 1 here, over the bounded `s`); flat past the
    // exponent's cap of 700.
    for t in [13.9, 20.0, 40.0] {
        dl = [1.0; 4];
        let s = f.predict(GarchVol::Egarch, &[t, 0.0, 0.0, 0.0], &[], &mut dl);
        let want = 1e6 + (t - 1e6f64.ln());
        assert!((s - want).abs() < 1e-9 * want, "{t}: {s} against {want}");
        assert_eq!(dl, [1.0 / s, 0.0, 0.0, 0.0]);
    }
    dl = [1.0; 4];
    let s = f.predict(GarchVol::Egarch, &[800.0, 0.0, 0.0, 0.0], &[], &mut dl);
    assert!((s - (1e6 + 700.0 - 1e6f64.ln())).abs() < 1e-6, "{s}");
    assert_eq!(dl, [0.0; 4]);
    // The bounds follow the mean of `e²`, not a constant.
    let wide = Filter::new(GarchVol::Egarch, 4, 0, 1e4);
    let s = wide.predict(GarchVol::Egarch, &[-20.0, 0.0, 0.0, 0.0], &[], &mut dl);
    assert_eq!(s, 1e4 / 1e6);
}

/// `ln |β − (α |z| + γ z) / 2|`, with both news terms in the sum.
#[test]
fn the_log_of_mu_adds_the_size_and_the_sign_terms() {
    let theta = [0.0, 0.3, 0.2, 0.9];
    for (z, mu) in [(1.5, 0.525), (-1.5, 0.9 - 0.5 * (0.45 - 0.3)), (0.0, 0.9)] {
        let got = log_mu(&theta, z);
        assert!((got - f64::ln(mu)).abs() < 1e-13, "z {z}: {got}");
    }
}

/// `ln Γ` at the edge where the series starts: 20 is read from its series
/// directly, to the bit, and agrees with `ln 19!` to rounding.
#[test]
fn ln_gamma_at_the_series_edge() {
    let at = ln_gamma(20.0);
    assert!((at - 39.339_884_187_199_495).abs() < 1e-13, "{at}");
    assert_eq!(at.to_bits(), 4_630_733_413_802_195_333);
}

/// `Garch::span_cap_check`, `Garch::discard_next`, and `step_into`'s per-series
/// retry buffer growth guard (`with <= cap && rows < bound`).
#[test]
fn span_cap_check_discard_next_and_retry_buffer_growth_bounds() {
    // 2-target GARCH with `start_span` and a small `max_state_bytes` where `bound(&cfg) == 5`.
    let base_cfg = GarchCfg {
        n_targets: 2,
        vol: GarchVol::Garch,
        dist: GarchDist::Normal,
        truncation: None,
        variance_targeting: false,
        start_rows: 1000,
        start_span: Some(100.0),
        horizons: Vec::new(),
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
        n_inputs: 0,
        adaptive_start: false,
    };
    let cap_for_5 = Garch::state_bytes(&base_cfg, 5);
    let cfg = GarchCfg {
        max_state_bytes: Some(cap_for_5),
        ..base_cfg
    };
    let b = series::bound(&cfg);
    assert_eq!(b, 5);

    let mut g = Garch::new(cfg.clone()).unwrap();
    // 1. `discard_next` clears pending handed stamp.
    g.stamp_next(Stamp::from(42.0));
    assert!(g.next.is_some());
    g.discard_next();
    assert!(g.next.is_none());

    // 2. `weight == 0.0` returns `None` (`weight > 0.0`, not `>=`).
    assert_eq!(g.span_cap_check(&[], &[Some(0.1), Some(0.2)], 0.0), None);

    // 3. `y[j].is_none() || series.started` (`||` not `&&`):
    //    - both `None`, neither started -> `None`
    assert_eq!(g.span_cap_check(&[], &[None, None], 1.0), None);
    //    - target 0 `Some`, started; target 1 `None`, not started -> `None`
    let mut g_started = g.clone();
    g_started.series[0].started = true;
    assert_eq!(g_started.span_cap_check(&[], &[Some(0.1), None], 1.0), None);

    // 4. Multi-target `else if total + delta <= cap && (rows as u64) < b { total += delta; }`:
    //    Target 0 has `tried = 1` (hits `else if`), target 1 has `tried = 0` (`checks = true`).
    let delta_one = Garch::series_bytes(&cfg, 1) - Garch::series_bytes(&cfg, 0);
    assert!(delta_one > 0);

    // (a) `total + delta <= cap` AND `rows < b`: adds BOTH target 0 and target 1 deltas.
    let mut g_both = g.clone();
    g_both.series[0].tried = 1;
    let base_bytes = g_both.bytes();
    assert_eq!(
        g_both.span_cap_check(&[], &[Some(0.1), Some(0.2)], 1.0),
        Some(base_bytes + 2 * delta_one)
    );

    // (b) `total + delta <= cap` AND `rows == b` (5 rows on target 0, 0 on target 1):
    //     target 0 does NOT grow (`< b`, not `<= b` or `||`), only target 1 adds `delta_one`.
    let mut g_at_b = g.clone();
    g_at_b.series[0].tried = 1;
    for _ in 0..b {
        g_at_b.series[0].buffer.push([0.1, 1.0]);
    }
    let bytes_at_b = g_at_b.bytes();
    assert!(bytes_at_b + delta_one <= cfg.cap());
    assert_eq!(
        g_at_b.span_cap_check(&[], &[Some(0.1), Some(0.2)], 1.0),
        Some(bytes_at_b + delta_one)
    );

    // (c) `total + delta > cap` AND `rows < b` (target 0 has 3 rows, target 1 has 7 rows so
    //     `bytes == cap`, `total + delta > cap` while `total - delta <= cap`):
    //     target 0 does NOT grow (`total + delta <= cap`, not `total - delta` or `||`).
    let mut g_over_cap = g.clone();
    g_over_cap.series[0].tried = 1;
    for _ in 0..3 {
        g_over_cap.series[0].buffer.push([0.1, 1.0]);
    }
    for _ in 0..7 {
        g_over_cap.series[1].buffer.push([0.1, 1.0]);
    }
    let bytes_at_cap = g_over_cap.bytes();
    assert_eq!(bytes_at_cap, cfg.cap());
    assert_eq!(
        g_over_cap.span_cap_check(&[], &[Some(0.1), Some(0.2)], 1.0),
        Some(bytes_at_cap + delta_one)
    );

    // 5. `step_into`: `let grows = with <= cap && (rows as u64) < bound(&self.cfg);`
    //    With target 0 at `rows == b (5)`, `tried = 1`, target 1 empty (`with <= cap`):
    //    stepping target 0 must NOT grow its buffer past `b` (`&&`, not `||`, and `<`, not `<=`).
    let mut g_step = g_at_b;
    g_step.step(&[], &[Some(0.2), None], 1.0, 1.0);
    assert_eq!(g_step.series[0].buffer.len() as u64, b);
}
