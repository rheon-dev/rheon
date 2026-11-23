//! The variance forecasts at the spec's horizons, and their sums.
//!
//! A horizon counts rows from the row being read: horizon `h` is the
//! variance of the return `h − 1` rows past this one, given the rows before
//! this one, so horizon 1 is the row's own variance, the same number
//! `variance` reports. Beside it, the sum of the variances over the `h`
//! rows, this one's included: the variance of the `h`-row return.
//!
//! ```text
//! garch, gjr   f_1 = s_t,  f_h = ω + P f_{h−1},  P = α + γ/2 + β   (γ = 0 for garch)
//! egarch       f_h = exp(L_h) Π_{j=1}^{h−1} M(β^{j−1}),     L_1 = ln s_t,  L_h = ω + β L_{h−1}
//!              M(c) = E e^{c g(z)} = e^{−cα√(2/π)} [e^{(A+B)²/2} Φ(A + B) + e^{(A−B)²/2} Φ(A − B)],
//!              A = cα,  B = cγ,  g(z) = α (|z| − √(2/π)) + γ z
//! figarch      f_h = ω/(1 − β) + Σ_{j=1}^{M} λ_j ê_{t+h−1−j},  ê a forecast at or after row t, the ring's e² before it
//! F_h = Σ_{i=1}^{h} f_i
//! ```
//!
//! Each later row's `e²` is read as its variance, and GJR's asymmetric term
//! as half of it (its innovations are symmetric), as `arch`'s analytic
//! forecast reads them, with the persistence `P` formed once, as the
//! forecast of one horizon was before the horizons (so a state of then goes
//! on to the bit). EGARCH's is Nelson's (1991) closed form for
//! normal innovations: its log-variance is linear in the news terms
//! `g(z)`, independent across rows, so the variance is the exponential of
//! the log-variance's mean times the product of their moment generating
//! function at the weights `β^{j−1}`. `arch` forecasts it only by
//! simulation. `exp` and `erfc` are the `libm` crate's (`garch::math`), the
//! same bits on every platform; the forecast is not held within the
//! variance's bounds the filter applies.
//!
//! **One pass.** The values are made in one pass from horizon 1 to the
//! largest, the sum carried beside them and read at each horizon asked
//! for, so no horizon is summed twice. The sum is compensated (TwoSum), so
//! it is the variances' sum to about a rounding, however many rows it
//! spans. The pass costs the largest horizon times one step's work
//! ([`step_cost`]), held under `max_horizon_work` when the spec is built.

use serde::{Deserialize, Deserializer};

use super::filter::{ABS_Z, Filter, figarch_weights};
use super::{GarchCfg, GarchVol, math};

/// The most horizons a spec may ask for.
pub const MAX_HORIZONS: usize = 64;
/// `max_horizon_work` by default: the multiply-adds a target's forecasts
/// may take a row, 2²⁰.
pub const HORIZON_WORK: u64 = 1 << 20;
/// The most `max_horizon_work` may be set to, 2³⁰.
pub const HORIZON_WORK_CEILING: u64 = 1 << 30;
/// What one call of `exp` or `erfc` counts for in a step's work, in
/// multiply-adds.
pub const LIBM_CALL_COST: u64 = 32;

/// The work of one step of the forecast's pass, in multiply-adds: GARCH's
/// and GJR's recursion; EGARCH's, with its four `exp`/`erfc` calls for
/// `M` and one for the variance; FIGARCH's dot product over its `M` lags.
pub fn step_cost(vol: GarchVol, truncation: usize) -> u64 {
    match vol {
        GarchVol::Garch | GarchVol::Gjr => 4,
        GarchVol::Egarch => 4 + 5 * LIBM_CALL_COST,
        GarchVol::Figarch => truncation as u64 + 2,
    }
}

/// The work a target's forecasts take a row: the largest horizon times one
/// step's.
pub fn work(cfg: &GarchCfg) -> u64 {
    let hmax = cfg.horizons.iter().copied().max().unwrap_or(0) as u64;
    hmax.saturating_mul(step_cost(cfg.vol, cfg.truncation()))
}

/// The horizons' checks, for [`GarchCfg::validate`].
pub fn check(cfg: &GarchCfg) -> Result<(), String> {
    let hs = &cfg.horizons;
    if hs.len() > MAX_HORIZONS {
        return Err(format!(
            "garch: horizons may hold at most {MAX_HORIZONS} horizons, but it holds {}. Each \
             adds two fields a target to every row. Ask for {MAX_HORIZONS} or fewer.",
            hs.len()
        ));
    }
    if hs.contains(&0) {
        return Err(
            "garch: each of horizons must be at least 1, but one is 0. A horizon counts the \
             rows from the row read, its own included: 1 is the row's own variance, 2 the next \
             row's. Give 1 or more."
                .into(),
        );
    }
    for (i, h) in hs.iter().enumerate() {
        if hs[..i].contains(h) {
            return Err(format!(
                "garch: horizons holds {h} twice. Each horizon's fields are named by it, so it \
                 may be asked for once. Leave out the second."
            ));
        }
    }
    let cap = cfg.max_horizon_work.unwrap_or(HORIZON_WORK);
    if cap == 0 {
        return Err(
            "garch: max_horizon_work must be above 0, but it is 0. Give the multiply-adds a \
             target's forecasts may take a row, or leave it out for 2^20 (1048576)."
                .into(),
        );
    }
    if cap > HORIZON_WORK_CEILING {
        return Err(format!(
            "garch: max_horizon_work may be at most {HORIZON_WORK_CEILING} (2^30), but it is \
             {cap}. Every row pays it for every target. Set max_horizon_work to 2^30 or less."
        ));
    }
    let w = work(cfg);
    if w > cap {
        let c = step_cost(cfg.vol, cfg.truncation());
        let hmax = hs.iter().copied().max().unwrap_or(0);
        return Err(format!(
            "garch: the forecasts to horizon {hmax} would take {w} multiply-adds a target a \
             row ({hmax} rows of {c} each{}), past max_horizon_work of {cap}. Lower the largest \
             horizon to {} or below, or raise max_horizon_work to {w}.",
            match cfg.vol {
                GarchVol::Egarch => format!(
                    ", four exp and erfc calls and an exp, each counted as {LIBM_CALL_COST}"
                ),
                GarchVol::Figarch => format!(", a dot product over {} lags", cfg.truncation()),
                _ => String::new(),
            },
            cap / c
        ));
    }
    Ok(())
}

/// Reads `horizons`, or a state's `horizon` of before it (a count of rows
/// ahead, 1 for none): `horizon = h` above 1 is `horizons = [h]`, the
/// same forecast, now with its sum beside it.
pub fn read<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<usize>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Either {
        One(usize),
        Many(Vec<usize>),
    }
    Ok(match Either::deserialize(d)? {
        Either::One(h) if h > 1 => vec![h],
        Either::One(_) => Vec::new(),
        Either::Many(v) => v,
    })
}

/// `a + b` as a pair: the rounded sum and what it leaves out (Knuth's
/// TwoSum, exact whatever the sizes).
#[inline]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

/// `Φ(x)`, the standard normal's distribution function.
#[inline]
fn norm_cdf(x: f64) -> f64 {
    0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
}

/// EGARCH's `M(c) = E e^{c g(z)}` for a standard normal `z`.
pub fn egarch_mgf(alpha: f64, gamma: f64, c: f64) -> f64 {
    let (a, b) = (c * alpha, c * gamma);
    let centre = c * alpha * ABS_Z;
    let (p, m) = (a + b, a - b);
    math::exp(0.5 * p * p - centre) * norm_cdf(p) + math::exp(0.5 * m * m - centre) * norm_cdf(m)
}

/// The forecasts of one target from a row whose variance is `s`, at the
/// parameters `theta` and the lags in `filter`: for each of `horizons`, in
/// its order, the variance at it and the sum to it, into `out` (two
/// slots a horizon). `scratch` holds FIGARCH's weights and the pass's
/// values.
pub fn forecasts(
    cfg: &GarchCfg,
    theta: &[f64],
    filter: &Filter,
    s: f64,
    out: &mut [f64],
    scratch: &mut Vec<f64>,
) {
    let hs = &cfg.horizons;
    // the horizons' slots in increasing order of horizon
    let mut order = [0usize; MAX_HORIZONS];
    let order = &mut order[..hs.len()];
    for (i, o) in order.iter_mut().enumerate() {
        *o = i;
    }
    order.sort_unstable_by_key(|&i| hs[i]);
    let hmax = hs[order[order.len() - 1]];
    let mut next = 0;
    let (mut hi, mut lo) = (0.0, 0.0);
    let mut put = |h: usize, f: f64, hi: f64, lo: f64, next: &mut usize| {
        while *next < order.len() && hs[order[*next]] == h {
            out[2 * order[*next]] = f;
            out[2 * order[*next] + 1] = hi + lo;
            *next += 1;
        }
    };
    let k = cfg.vol.k();
    match cfg.vol {
        GarchVol::Garch | GarchVol::Gjr => {
            // the persistence `P = α + γ/2 + β` (γ = 0 for garch), formed
            // once: each later row's `e²` read as its variance and GJR's
            // asymmetric term as half of it
            let persist = if k == 4 {
                theta[1] + 0.5 * theta[2] + theta[3]
            } else {
                theta[1] + theta[2]
            };
            let mut f = s;
            for h in 1..=hmax {
                if h > 1 {
                    f = theta[0] + persist * f;
                }
                let (a, e) = two_sum(hi, f);
                hi = a;
                lo += e;
                put(h, f, hi, lo, &mut next);
            }
        }
        GarchVol::Egarch => {
            let (omega, alpha, gamma, beta) = (theta[0], theta[1], theta[2], theta[3]);
            let mut ln_s = math::ln(s);
            let mut prod = 1.0;
            let mut c = 1.0;
            for h in 1..=hmax {
                let f = if h == 1 {
                    s
                } else {
                    prod *= egarch_mgf(alpha, gamma, c);
                    c *= beta;
                    ln_s = omega + beta * ln_s;
                    math::exp(ln_s) * prod
                };
                let (a, e) = two_sum(hi, f);
                hi = a;
                lo += e;
                put(h, f, hi, lo, &mut next);
            }
        }
        GarchVol::Figarch => {
            let m = filter.ring.len();
            scratch.clear();
            figarch_weights(theta, m, false, |_, lam, _, _, _| scratch.push(lam));
            let w = theta[0] / (1.0 - theta[3]);
            // scratch[m + i] is the variance of the row i after this one's
            scratch.push(s);
            for h in 1..=hmax {
                let f = if h == 1 {
                    s
                } else {
                    let i = h - 1;
                    let mut v = w;
                    for j in 1..=m {
                        // the row j back from row i: a forecast while it is
                        // this row or later, else the ring's (j − i) back
                        let x = if j <= i {
                            scratch[m + i - j]
                        } else {
                            filter.back(j - i)
                        };
                        v += scratch[j - 1] * x;
                    }
                    scratch.push(v);
                    v
                };
                let (a, e) = two_sum(hi, f);
                hi = a;
                lo += e;
                put(h, f, hi, lo, &mut next);
            }
        }
    }
}
