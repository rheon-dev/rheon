//! The fit itself, as scikit-learn's `IsotonicRegression` computes it: the
//! distinct `x` pooled ([`pool_rows`], `_make_unique`), pool-adjacent-
//! violators over them ([`pava`], scipy's `optimize.isotonic_regression`),
//! the clip to `[y_min, y_max]`, and the read at a new `x` ([`interp`],
//! `numpy.interp` behind `interp1d`). Each is transcribed line for line, in
//! the source's own order of operations and without fused multiply-adds, so
//! that the fit is scipy's bits on a build that does not contract them
//! (docs/records/ISOTONIC-DESIGN.md).

use super::OutOfBounds;

/// Pool-adjacent-violators over `x` in place, non-decreasing, with the
/// weights `w` (each above 0): scipy's `pava` (Busing 2022, Algorithm 1,
/// with `>=` on both violation tests so that equal values share a block),
/// line for line. On return `x` holds the fit, and `w` and `r` are
/// scipy's scratch: `r` must hold `x.len() + 1` entries.
pub(crate) fn pava(x: &mut [f64], w: &mut [f64], r: &mut [usize]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    r[0] = 0;
    if n > 1 {
        r[1] = 1;
    }
    let mut b = 0usize;
    let mut xb_prev = x[0];
    let mut wb_prev = w[0];
    let mut i = 1usize;
    while i < n {
        b += 1;
        let mut xb = x[i];
        let mut wb = w[i];
        if xb_prev >= xb {
            b -= 1;
            let mut sb = wb_prev * xb_prev + wb * xb;
            wb += wb_prev;
            xb = sb / wb;
            while i < n - 1 && xb >= x[i + 1] {
                i += 1;
                sb += w[i] * x[i];
                wb += w[i];
                xb = sb / wb;
            }
            while b > 0 && x[b - 1] >= xb {
                b -= 1;
                sb += w[b] * x[b];
                wb += w[b];
                xb = sb / wb;
            }
        }
        x[b] = xb;
        xb_prev = xb;
        w[b] = wb;
        wb_prev = wb;
        r[b + 1] = i + 1;
        i += 1;
    }
    // The backward fill: each block's value over its rows, last block first.
    let mut f = n;
    for k in (0..=b).rev() {
        let t = r[k];
        let xk = x[k];
        x[t..f].fill(xk);
        f = t;
    }
}

/// The isotonic fit of `y` with the weights `w`, in place: non-decreasing,
/// or non-increasing where `increasing` is false (scipy runs the reversed
/// sequence and reverses it back), then clipped to `[lo, hi]` as `np.clip`
/// does. `w` and `r` are scratch.
pub(crate) fn isotonic(
    y: &mut [f64],
    w: &mut [f64],
    r: &mut Vec<usize>,
    increasing: bool,
    lo: f64,
    hi: f64,
) {
    r.clear();
    r.resize(y.len() + 1, 0);
    if increasing {
        pava(y, w, r);
    } else {
        y.reverse();
        w.reverse();
        pava(y, w, r);
        y.reverse();
    }
    if lo > f64::NEG_INFINITY || hi < f64::INFINITY {
        for v in y.iter_mut() {
            *v = v.max(lo).min(hi);
        }
    }
}

/// The distinct `x` of rows sorted by `(x, y)`, each with its pooled value
/// and weight, as `_make_unique` forms them: a run of equal `x` sums its
/// weights and its `y w` from 0 in the rows' order, and its value is the
/// sum over the weight. A row of weight 0 is no row (scikit-learn masks
/// it). Appends to `ux`, `uy` and `uw`.
pub(crate) fn pool_rows(
    x: &[f64],
    y: &[f64],
    w: &[f64],
    ux: &mut Vec<f64>,
    uy: &mut Vec<f64>,
    uw: &mut Vec<f64>,
) {
    let mut cur: Option<(f64, f64, f64)> = None;
    for ((&xi, &yi), &wi) in x.iter().zip(y).zip(w) {
        if wi <= 0.0 {
            continue;
        }
        match cur.as_mut() {
            Some((cx, cw, cs)) if xi == *cx => {
                *cw += wi;
                *cs += yi * wi;
            }
            _ => {
                if let Some((cx, cw, cs)) = cur {
                    ux.push(cx);
                    uy.push(cs / cw);
                    uw.push(cw);
                }
                cur = Some((xi, 0.0 + wi, 0.0 + yi * wi));
            }
        }
    }
    if let Some((cx, cw, cs)) = cur {
        ux.push(cx);
        uy.push(cs / cw);
        uw.push(cw);
    }
}

/// The fit read at `x`, as scikit-learn's `predict` reads it: one distinct
/// `x` is its value everywhere; outside `[x_0, x_last]` NaN under
/// [`OutOfBounds::Nan`], or `x` clamped to the range under
/// [`OutOfBounds::Clip`]; inside it `numpy.interp`'s line between the two
/// neighbours, its value exactly at an `x` it was fitted at. NaN before
/// any `x`.
pub(crate) fn interp(fx: &[f64], fy: &[f64], x: f64, oob: OutOfBounds) -> f64 {
    let n = fx.len();
    match n {
        0 => return f64::NAN,
        1 => return fy[0],
        _ => {}
    }
    let (first, last) = (fx[0], fx[n - 1]);
    let x = match oob {
        OutOfBounds::Nan if x < first || x > last => return f64::NAN,
        OutOfBounds::Nan => x,
        OutOfBounds::Clip => x.max(first).min(last),
    };
    if x == last {
        return fy[n - 1];
    }
    // The interval `fx[j] <= x < fx[j + 1]`: `j <= n - 2`, since `x` is
    // below the last.
    let j = fx.partition_point(|&v| v <= x) - 1;
    if fx[j] == x {
        // numpy's guard against a non-finite line at a point it holds.
        return fy[j];
    }
    let slope = (fy[j + 1] - fy[j]) / (fx[j + 1] - fx[j]);
    let mut r = slope * (x - fx[j]) + fy[j];
    if r.is_nan() {
        r = slope * (x - fx[j + 1]) + fy[j + 1];
        if r.is_nan() && fy[j] == fy[j + 1] {
            r = fy[j];
        }
    }
    r
}
