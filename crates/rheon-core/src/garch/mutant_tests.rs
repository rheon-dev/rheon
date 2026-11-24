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
/// the log of the excess.
#[test]
fn the_egarch_variance_is_bounded_below_and_grows_only_logarithmically_above() {
    let f = Filter::new(GarchVol::Egarch, 4, 0, 1.0);
    let mut dl = [1.0; 4];
    let s = f.predict(GarchVol::Egarch, &[0.0, 0.1, 0.1, 0.1], &mut dl);
    assert_eq!(s, 1.0);
    assert!(dl.iter().any(|d| *d != 0.0), "a slope inside");
    // Just inside each bound.
    let inside_hi = f.predict(GarchVol::Egarch, &[13.8, 0.0, 0.0, 0.0], &mut dl);
    assert!(
        (inside_hi - math::exp(13.8)).abs() < 1e-6 * inside_hi,
        "{inside_hi}"
    );
    let inside_lo = f.predict(GarchVol::Egarch, &[-13.8, 0.0, 0.0, 0.0], &mut dl);
    assert!(
        (inside_lo - math::exp(-13.8)).abs() < 1e-6 * inside_lo,
        "{inside_lo}"
    );
    // Below: the floor, flat.
    dl = [1.0; 4];
    let low = f.predict(GarchVol::Egarch, &[-20.0, 0.0, 0.0, 0.0], &mut dl);
    assert_eq!(low, 1e-6);
    assert_eq!(dl, [0.0; 4]);
    // Above: the ceiling and the log of the excess over it, flat.
    for t in [13.9, 20.0, 40.0] {
        dl = [1.0; 4];
        let s = f.predict(GarchVol::Egarch, &[t, 0.0, 0.0, 0.0], &mut dl);
        let want = 1e6 + (t - 1e6f64.ln());
        assert!((s - want).abs() < 1e-9 * want, "{t}: {s} against {want}");
        assert_eq!(dl, [0.0; 4]);
    }
    // The bounds follow the mean of `e²`, not a constant.
    let wide = Filter::new(GarchVol::Egarch, 4, 0, 1e4);
    let s = wide.predict(GarchVol::Egarch, &[-20.0, 0.0, 0.0, 0.0], &mut dl);
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
