//! The models the frozen state fixtures (`tests/state_fixtures.rs`) and the
//! layout checks (`tests/state_repairs.rs`) build: at least one per
//! [`ModelState`] variant, each configured to keep what its variant can keep
//! -- a window, lags, bins, a warm-up buffer, far rows, held moments -- and
//! a stream for each, read in two parts: the rows before the save, and the
//! rows its continuation is checked on (docs/PLAN.md task 198).
//!
//! The streams come from an integer generator and arithmetic alone (no libm:
//! a stream made with one would differ by platform), and every number of a
//! continuation is frozen beside the state, so the generator could change
//! without moving a fixture.

#![allow(dead_code)]

use rheon_core::*;

#[path = "../garch_series/mod.rs"]
mod garch_series;

/// One row: features, targets, clock step, weight.
#[derive(Debug, Clone)]
pub struct Row {
    pub x: Vec<f64>,
    pub y: Vec<Option<f64>>,
    pub d: f64,
    pub w: f64,
}

/// What a row reports: the prediction slots, `n_eff`, and the numbers of
/// the step's extra (lasso's selected penalties, or `huber`'s emitted
/// scales), in that order.
#[derive(Debug, Clone, PartialEq)]
pub struct Out {
    pub pred: Vec<f64>,
    pub n_eff: f64,
    pub extra: Vec<f64>,
}

/// A model the harness runs, whatever its type.
pub trait Run {
    fn step(&mut self, r: &Row) -> Out;
    fn state(&self) -> State;
}

struct Model<M>(M);

impl<M: OnlineModel> Run for Model<M> {
    fn step(&mut self, r: &Row) -> Out {
        let s = self.0.step(&r.x, &r.y, r.d, r.w);
        let extra = match s.extra {
            Some(Extra::Lasso { lam_selected }) => lam_selected,
            Some(Extra::Huber { scale }) => scale,
            Some(other) => panic!("an extra the harness does not know: {other:?}"),
            None => Vec::new(),
        };
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra,
        }
    }

    fn state(&self) -> State {
        self.0.state()
    }
}

/// A `hawkes` over a window of clock units, each row handed its stamp in
/// integer nanoseconds as a stream hands it: so the state holds a stamp a
/// row and the last stamp, the forms an unstamped run does not write.
struct StampedHawkes {
    model: Hawkes,
    ns: i128,
}

impl Run for StampedHawkes {
    fn step(&mut self, r: &Row) -> Out {
        self.ns += ns_of_seconds(r.d).unwrap();
        self.model.stamp_next(Stamp::Ns(self.ns));
        let s = self.model.step(&r.x, &r.y, r.d, r.w);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.model.state()
    }
}

fn restore_stamped_hawkes(s: &State) -> Result<Box<dyn Run>, String> {
    let model = Hawkes::restore(s).map_err(|e| e.to_string())?;
    let ns = match model.last_stamp() {
        Some(Stamp::Ns(ns)) => ns,
        other => return Err(format!("expected a temporal stamp, found {other:?}")),
    };
    Ok(Box::new(StampedHawkes { model, ns }))
}

/// A `realized` over a window of clock units, each row handed its stamp in
/// integer nanoseconds as a stream hands it: its window holds a stamp a row.
struct StampedRealized {
    model: Realized,
    ns: i128,
}

impl Run for StampedRealized {
    fn step(&mut self, r: &Row) -> Out {
        self.ns += ns_of_seconds(r.d).unwrap();
        self.model.stamp_next(Stamp::Ns(self.ns));
        let s = self.model.step(&r.x, &r.y, r.d, r.w);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.model.state()
    }
}

/// The run goes on from the newest stamp its window holds, which is the
/// last row's where that row counted (the cases save after one that did).
fn restore_stamped_realized(s: &State) -> Result<Box<dyn Run>, String> {
    let model = Realized::restore(s).map_err(|e| e.to_string())?;
    let ns = match model.last_stamp() {
        Some(Stamp::Ns(ns)) => ns,
        other => return Err(format!("expected a temporal stamp, found {other:?}")),
    };
    Ok(Box::new(StampedRealized { model, ns }))
}

/// A `power_variation` over a window of clock units, each row handed its
/// stamp in integer nanoseconds as a stream hands it.
struct StampedPowerVariation {
    model: PowerVariation,
    ns: i128,
}

impl Run for StampedPowerVariation {
    fn step(&mut self, r: &Row) -> Out {
        self.ns += ns_of_seconds(r.d).unwrap();
        self.model.stamp_next(Stamp::Ns(self.ns));
        let s = self.model.step(&r.x, &r.y, r.d, r.w);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.model.state()
    }
}

/// The run goes on from the newest stamp its window holds.
fn restore_stamped_power_variation(s: &State) -> Result<Box<dyn Run>, String> {
    let model = PowerVariation::restore(s).map_err(|e| e.to_string())?;
    let ns = match model.last_stamp() {
        Some(Stamp::Ns(ns)) => ns,
        other => return Err(format!("expected a temporal stamp, found {other:?}")),
    };
    Ok(Box::new(StampedPowerVariation { model, ns }))
}

/// An `expectile` handed each row's stamp in integer nanoseconds, as a
/// stream hands it: its state holds the last stamp, the last counted row's,
/// and under a clock window a stamp a row.
struct StampedExpectile {
    model: Expectile,
    ns: i128,
}

impl Run for StampedExpectile {
    fn step(&mut self, r: &Row) -> Out {
        self.ns += ns_of_seconds(r.d).unwrap();
        self.model.stamp_next(Stamp::Ns(self.ns));
        let s = self.model.step(&r.x, &r.y, r.d, r.w);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.model.state()
    }
}

fn restore_stamped_expectile(s: &State) -> Result<Box<dyn Run>, String> {
    let model = Expectile::restore(s).map_err(|e| e.to_string())?;
    let ns = match model.last_stamp() {
        Some(Stamp::Ns(ns)) => ns,
        other => return Err(format!("expected a temporal stamp, found {other:?}")),
    };
    Ok(Box::new(StampedExpectile { model, ns }))
}

/// The bare accumulator, which no model's state is and which a variant
/// still names: stepped by its own `update`, its means and co-moments
/// reported as the slots, the co-moments all `K²` of them row by row. The
/// upper entries, `cov(i, j)` with `i < j`, are read through the lower
/// triangle's mirror, so a state of schema 61-63 continues within 1e-15
/// there, not to the bit (`state_fixtures.rs`, `CONTINUES_WITHIN`).
struct Accumulator(EwCov);

const ACC_DECAY: Decay = Decay::Halflife(8.0);

impl Run for Accumulator {
    fn step(&mut self, r: &Row) -> Out {
        let n_eff = self.0.n_eff();
        self.0.update(&r.x, ACC_DECAY.factor(r.d), r.w);
        let k = self.0.k();
        let mut pred: Vec<f64> = self.0.means().to_vec();
        pred.extend(
            (0..k)
                .flat_map(|i| (0..k).map(move |j| (i, j)))
                .map(|(i, j)| self.0.cov(i, j)),
        );
        Out {
            pred,
            n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::EwCov(Box::new(self.0.clone())))
    }
}

/// An `audit` fed each feature times `2^100`: values past `2^65`, whose
/// moments it keeps over a power of two (docs/PLAN.md task 223 (b)), so the
/// fixture writes a `scale` above 1. A power of two, so the stream is the
/// case's own, exactly, at another size.
struct ScaledAudit(Audit);

const AUDIT_SCALE: f64 = 1_267_650_600_228_229_401_496_703_205_376.0; // 2^100

impl Run for ScaledAudit {
    fn step(&mut self, r: &Row) -> Out {
        let x: Vec<f64> = r.x.iter().map(|v| v * AUDIT_SCALE).collect();
        let s = self.0.step(&x, &r.y, r.d, r.w);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.0.state()
    }
}

fn restore_scaled_audit(s: &State) -> Result<Box<dyn Run>, String> {
    Audit::restore(s)
        .map(|m| Box::new(ScaledAudit(m)) as Box<dyn Run>)
        .map_err(|e| e.to_string())
}

/// A model fed every weight times `2^-664`: weights whose squares are not
/// doubles, so the Kish sums keep a power of two of their own (review
/// 2026-11-23, REG-12; `sqsum`), and the fixture writes each exponent. A
/// power of two, so the stream is the case's own at another scale.
struct TinyWeights<M>(M);

/// `2^-664`.
const TINY_WEIGHT: f64 = 1.306_420_176_630_260_4e-200;

impl<M: OnlineModel> Run for TinyWeights<M> {
    fn step(&mut self, r: &Row) -> Out {
        let s = self.0.step(&r.x, &r.y, r.d, r.w * TINY_WEIGHT);
        Out {
            pred: s.pred,
            n_eff: s.n_eff,
            extra: Vec::new(),
        }
    }

    fn state(&self) -> State {
        self.0.state()
    }
}

fn restore_tiny<M: OnlineModel + 'static>(s: &State) -> Result<Box<dyn Run>, String> {
    M::restore(s)
        .map(|m| Box::new(TinyWeights(m)) as Box<dyn Run>)
        .map_err(|e| e.to_string())
}

fn restore_accumulator(s: &State) -> Result<Box<dyn Run>, String> {
    check_schema(s).map_err(|e| e.to_string())?;
    match &s.model {
        ModelState::EwCov(a) if a.has_shape(a.k()) => Ok(Box::new(Accumulator((**a).clone()))),
        ModelState::EwCov(_) => Err("the accumulator has the wrong shape".into()),
        other => Err(format!("expected the accumulator, found {}", other.kind())),
    }
}

fn restore<M: OnlineModel + 'static>(s: &State) -> Result<Box<dyn Run>, String> {
    M::restore(s)
        .map(|m| Box::new(Model(m)) as Box<dyn Run>)
        .map_err(|e| e.to_string())
}

/// What the targets of a case's stream are.
#[derive(Debug, Clone, Copy)]
pub enum Targets {
    /// `n` linear targets of the features, with noise.
    Linear(usize),
    /// The same, null on every row before the second number: a model that
    /// has seen its features and none of its targets.
    LinearFrom(usize, usize),
    /// The same, the first target on every row and the others null on
    /// every row before the second number: a target that joins late.
    JoiningFrom(usize, usize),
    /// `n` positive targets: each linear target squared, plus 0.1 -- a
    /// Gamma or Tweedie amount for `glm`.
    Positive(usize),
    /// `n` Gamma draws of shape 4 at a mean linear in the features
    /// (`(2 + 0.8 x0 - 0.4 x1) (1 + j)`, positive on this stream): the
    /// target `glm` is fitted to, drawn from its family. The draw is a sum
    /// of eight squared normals over 8 times the mean, from the generator
    /// alone.
    GammaDraw(usize),
    /// `n` series with a trend on the clock: `10 (1 + j) + (0.5 + 0.25 j) t`
    /// plus noise of about 0.3, `t` the clock the row stands at (a gap
    /// moves the line by its length), so a model with a trend has one to
    /// find.
    Trending(usize),
    /// `n` nodes of a discrete-time self-exciting process: node `j` has an
    /// event on a row with probability `0.1 + 0.25 R_j + 0.05 R_other`
    /// (at most 0.9), `R` each node's events before the row decayed by 0.6 a
    /// row, so an event makes the next few likelier. A value of 1 is an
    /// event and of 0 none.
    Excited(usize),
    /// One 0/1 target.
    Binary,
    /// One class label in `0..n`.
    Class(usize),
    /// None: the model reads only its features.
    None,
    /// `n` series of an asymmetric DCC process (`(a, b, g) = (0.04, 0.92,
    /// 0.03)`, every target correlation 0.5), each scaled by its own
    /// GARCH(1,1) volatility where the flag is set: a realistic stream for
    /// `dcc`, whose parameters it identifies. Arithmetic and `sqrt` alone,
    /// so it is the same bits on every platform. With the last flag unset,
    /// `(a, b, g) = 0`: one correlation, held, the null a start must not
    /// mistake for a recursion.
    Correlated(usize, bool, bool),
    /// `n` series of returns of a GARCH-family process at a truth `arch`
    /// is held to, each its own draw: a process a start of 2,000 rows
    /// identifies.
    Garch(usize, Process),
    /// None, and the first two features a correlated pair in place of the
    /// independent ones: correlation 0.3 before the row given and 0.85
    /// from it, each feature an Irwin-Hall draw (twelve uniforms less 6,
    /// near-Gaussian with arithmetic alone): a correlation to test, and a
    /// change in it (PLAN 550's realistic case for `corrchange`).
    CorrelatedPair(usize),
}

/// The process of a [`Targets::Garch`] stream.
#[derive(Debug, Clone, Copy)]
pub enum Process {
    /// GARCH(1,1) at `(0.05, 0.08, 0.9)` with these innovations.
    Garch(garch_series::Innovations),
    /// EGARCH at `(0, 0.15, −0.08, 0.97)`, normal innovations.
    Egarch,
    /// FIGARCH(1, d, 1) at `(0.03, 0.2, 0.4, 0.5)` cut at 20 lags, the
    /// truncation its fixture's model keeps.
    Figarch,
}

/// [`Targets::Correlated`]'s stream: `n` rows of `k` series after 200
/// discarded, innovations by Irwin and Hall's sum of twelve uniforms.
pub fn correlated(k: usize, n: usize, garch: bool, dynamic: bool, seed: u64) -> Vec<Vec<f64>> {
    let mut s = seed;
    let kk = k * k;
    let mut qbar = vec![0.5_f64; kk];
    for i in 0..k {
        qbar[i * k + i] = 1.0;
    }
    let (a, b, g) = if dynamic {
        (0.04, 0.92, 0.03)
    } else {
        (0.0, 0.0, 0.0)
    };
    let mut q = qbar.clone();
    let mut v = vec![1.0_f64; k];
    let mut e = vec![0.0_f64; k];
    let mut out = Vec::with_capacity(n);
    for t in 0..n + 200 {
        // R and its Cholesky factor
        let mut l = vec![0.0_f64; kk];
        for i in 0..k {
            for j in 0..=i {
                let r: f64 = q[i * k + j] / (q[i * k + i] * q[j * k + j]).sqrt();
                let mut acc: f64 = if i == j { 1.0 } else { r };
                for m in 0..j {
                    acc -= l[i * k + m] * l[j * k + m];
                }
                l[i * k + j] = if i == j {
                    acc.sqrt()
                } else {
                    acc / l[j * k + j]
                };
            }
        }
        let x: Vec<f64> = (0..k)
            .map(|_| (0..12).map(|_| 0.5 * (lcg(&mut s) + 1.0)).sum::<f64>() - 6.0)
            .collect();
        let z: Vec<f64> = (0..k)
            .map(|i| (0..=i).map(|j| l[i * k + j] * x[j]).sum())
            .collect();
        for i in 0..k {
            for j in 0..k {
                let ij = i * k + j;
                let (ni, nj) = (z[i].min(0.0), z[j].min(0.0));
                q[ij] = (1.0 - a - b) * qbar[ij] - g * 0.5 * qbar[ij]
                    + a * z[i] * z[j]
                    + g * ni * nj
                    + b * q[ij];
            }
        }
        for i in 0..k {
            v[i] = 0.05 + 0.08 * e[i] * e[i] + 0.9 * v[i];
            e[i] = if garch { v[i].sqrt() * z[i] } else { z[i] };
        }
        if t >= 200 {
            out.push(e.clone());
        }
    }
    out
}

/// One case: a model, configured, and its stream.
pub struct Case {
    /// The fixture's file name.
    pub name: &'static str,
    /// The [`ModelState`] variant its state is.
    pub variant: &'static str,
    pub build: fn() -> Box<dyn Run>,
    pub restore: fn(&State) -> Result<Box<dyn Run>, String>,
    pub n_features: usize,
    pub targets: Targets,
    /// Rows before the save; the continuation is [`CONTINUATION`] rows.
    pub before: usize,
}

/// Rows a continuation is checked on.
pub const CONTINUATION: usize = 16;

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
}

/// A standard normal to within the central limit theorem's reach: the sum
/// of six uniforms on [-1, 1] has variance 2, so it over the root of 2. Sums
/// and one correctly rounded square root, so the same bits on every
/// platform.
fn gauss(s: &mut u64) -> f64 {
    (0..6).map(|_| lcg(s)).sum::<f64>() / 2f64.sqrt()
}

impl Case {
    /// The case's whole stream: `before` rows, then the continuation. A clock
    /// gap, a null target, a zero weight and weights other than 1 among
    /// them, on both sides of the save; two regimes, so the detectors and
    /// clusterers have something to find.
    pub fn rows(&self) -> Vec<Row> {
        let mut s = 0x5eed_u64 ^ (self.name.len() as u64) << 32;
        for b in self.name.bytes() {
            s = s.wrapping_mul(31).wrapping_add(u64::from(b));
        }
        let n = self.before + CONTINUATION;
        let series = match self.targets {
            Targets::Correlated(k, garch, dynamic) => correlated(k, n, garch, dynamic, s),
            _ => Vec::new(),
        };
        let returns: Vec<Vec<f64>> = match self.targets {
            Targets::Garch(t, process) => (0..t)
                .map(|j| {
                    let seed = s ^ j as u64;
                    match process {
                        Process::Garch(z) => garch_series::gjr([0.05, 0.08, 0.0, 0.9], z, n, seed),
                        Process::Egarch => garch_series::egarch([0.0, 0.15, -0.08, 0.97], n, seed),
                        Process::Figarch => {
                            garch_series::figarch([0.03, 0.2, 0.4, 0.5], 20, n, seed)
                        }
                    }
                })
                .collect(),
            _ => Vec::new(),
        };
        let (mut clock, mut excitation) = (0.0, [0.0f64; 2]);
        (0..n)
            .map(|i| {
                let shift = if (i / 12) % 2 == 1 { 2.0 } else { 0.0 };
                let x: Vec<f64> = match self.targets {
                    Targets::CorrelatedPair(at) => {
                        let mut normal =
                            || (0..12).map(|_| lcg(&mut s) / 2.0 + 0.5).sum::<f64>() - 6.0;
                        let rho: f64 = if i < at { 0.3 } else { 0.85 };
                        let (z1, z2) = (normal(), normal());
                        let mut x = vec![z1, rho * z1 + (1.0 - rho * rho).sqrt() * z2];
                        x.extend((2..self.n_features).map(|_| normal()));
                        x
                    }
                    _ => (0..self.n_features).map(|_| shift + lcg(&mut s)).collect(),
                };
                let x0 = x.first().copied().unwrap_or(0.0);
                let x1 = x.get(1).copied().unwrap_or(0.0);
                let signal = 1.5 * x0 - 0.75 * x1 + 0.25;
                let noise = 0.1 * lcg(&mut s);
                let null = i % 11 == 7;
                let d = match i {
                    0 => 0.0,
                    _ if i % 23 == 13 => 9.0,
                    _ => 1.0,
                };
                clock += d;
                let y = match self.targets {
                    Targets::GammaDraw(t) => (0..t)
                        .map(|j| {
                            let mean = (2.0 + 0.8 * x0 - 0.4 * x1) * (1.0 + j as f64);
                            let chi: f64 = (0..8).map(|_| gauss(&mut s).powi(2)).sum();
                            (!null).then_some(mean * chi / 8.0)
                        })
                        .collect(),
                    Targets::Trending(t) => (0..t)
                        .map(|j| {
                            let level = 10.0 * (1.0 + j as f64);
                            let slope = 0.5 + 0.25 * j as f64;
                            (!null).then_some(level + slope * clock + 5.0 * noise)
                        })
                        .collect(),
                    Targets::Excited(t) => {
                        let before = excitation;
                        let events: Vec<bool> = (0..t)
                            .map(|j| {
                                let other = if t > 1 { before[(j + 1) % t] } else { 0.0 };
                                let p = (0.1 + 0.25 * before[j] + 0.05 * other).min(0.9);
                                (lcg(&mut s) + 1.0) / 2.0 < p && !null
                            })
                            .collect();
                        for (j, e) in events.iter().enumerate() {
                            excitation[j] = 0.6 * before[j] + f64::from(u8::from(*e));
                        }
                        (0..t)
                            .map(|j| (!null).then_some(f64::from(u8::from(events[j]))))
                            .collect()
                    }
                    Targets::Linear(t) => (0..t)
                        .map(|j| (!null).then_some(signal * (1.0 + j as f64) + noise - j as f64))
                        .collect(),
                    Targets::LinearFrom(t, first) => (0..t)
                        .map(|j| {
                            (!null && i >= first)
                                .then_some(signal * (1.0 + j as f64) + noise - j as f64)
                        })
                        .collect(),
                    Targets::JoiningFrom(t, first) => (0..t)
                        .map(|j| {
                            (!null && (j == 0 || i >= first))
                                .then_some(signal * (1.0 + j as f64) + noise - j as f64)
                        })
                        .collect(),
                    Targets::Positive(t) => (0..t)
                        .map(|j| {
                            let v = signal * (1.0 + j as f64) + noise - j as f64;
                            (!null).then_some(v * v + 0.1)
                        })
                        .collect(),
                    Targets::Binary => {
                        vec![(!null).then_some(f64::from(u8::from(signal + noise > 1.0)))]
                    }
                    Targets::Class(c) => {
                        let label = ((signal + noise + 2.0).max(0.0) as usize).min(c - 1);
                        vec![(!null).then_some(label as f64)]
                    }
                    Targets::None | Targets::CorrelatedPair(_) => Vec::new(),
                    Targets::Correlated(..) => {
                        series[i].iter().map(|v| (!null).then_some(*v)).collect()
                    }
                    Targets::Garch(..) => returns.iter().map(|r| (!null).then_some(r[i])).collect(),
                };
                let w = match i % 9 {
                    4 => 0.0,
                    2 => 2.5,
                    6 => 0.5,
                    _ => 1.0,
                };
                Row { x, y, d, w }
            })
            .collect()
    }
}

const K: usize = 2;
const H: f64 = 10.0;

/// The `solve_every` of the `*_first_period` cases: the clock the stream
/// reaches on row 42, two rows into the continuation (rows 1-42 a unit
/// apart, rows 13 and 36 nine).
pub const FIRST_PERIOD_EVERY: f64 = 58.0;

fn decay() -> Decay {
    Decay::Halflife(H)
}

/// `hmm` with full covariance over `d` features, seeded from its first ten
/// learned rows: the `hmm_full` cases.
fn hmm_full_cfg(d: usize) -> HmmCfg {
    HmmCfg {
        n_features: d,
        k: 2,
        decay: decay(),
        covariance: Covariance::Full,
        precision_prior: 1e-2,
        min_weight: 0.0,
        learn: true,
        transition_prior: 1.0,
        transition: None,
        means: None,
        covs: None,
        warm_rows: 10,
        seed_rule: SeedRule::First,
        seed: 1,
        tvtp: None,
        prior_rows: None,
        warm_clock: None,
    }
}

fn ewridge_cfg() -> EwRidgeCfg {
    EwRidgeCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        ridge: vec![1e-6, 0.1],
        feature_sets: vec![],
        standardize: true,
        ridge_scale: false,
        session_shrink: None,
        long_half_life: None,
        coef_prior: None,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        gram_block_rows: 0,
        target_gaps: TargetGaps::Pairwise,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn lasso_cfg() -> LassoCfg {
    LassoCfg {
        min_weight_holds_learning: true,
        n_features: K,
        n_targets: 1,
        fit_intercept: true,
        decay: decay(),
        lasso_path: vec![0.1, 0.01, 0.0],
        l1_ratio: 1.0,
        select_half_life: Some(5.0),
        min_weight: 3.0,
        target_min_weight: Vec::new(),
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
        max_iter: 100,
        tol: 1e-10,
        target_gaps: TargetGaps::OwnRows,
    }
}

fn kalman_cfg() -> KalmanCfg {
    KalmanCfg {
        horizons: Vec::new(),
        min_weight_holds_learning: true,
        noise_units: false,
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        half_life: vec![100.0],
        q: None,
        obs_var: None,
        p0: 1.0,
        share_p: false,
        min_weight: 3.0,
        revert_half_life: vec![f64::INFINITY, 25.0, 6.0],
        standardize: true,
    }
}

fn robust_cfg(loss: RobustLoss) -> RobustCfg {
    RobustCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        loss,
        ridge: 1e-6,
        standardize: true,
        min_weight: 3.0,
        solve_every: 0.0,
        max_rows_between_solves: 1,
        solve_share: None,
        quantile_eps: 1e-3,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        feature_sets: Vec::new(),
        ridge_scale: false,
        emit_scale: false,
        refit_rows: 0,
        mad_anchor: false,
        // The cases a state written before schema 64 converts to: the gate
        // held learning back then (docs/PLAN.md task 278, D10). The form
        // since is the refit cases' and `quantile_mad`'s.
        min_weight_holds_learning: true,
        refit_clock: None,
        max_state_bytes: None,
    }
}

fn ew_cov_cfg() -> EwCovCfg {
    EwCovCfg {
        bias: false,
        n_features: 3,
        decay: decay(),
        stats: vec![
            EwCovStat::Mean,
            EwCovStat::Var,
            EwCovStat::Corr,
            EwCovStat::Mahal,
            EwCovStat::LagCorr,
        ],
        min_weight: 3.0,
        precision_prior: Some(0.01),
        mahal_quantiles: vec![0.5, 0.9],
        pca: 1,
        pca_every: 4.0,
        max_rows_between_pca: 8,
        lags: vec![1, 3],
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn ew_class_cfg() -> EwClassCfg {
    EwClassCfg {
        n_features: K,
        n_classes: 3,
        decay: decay(),
        min_weight: 3.0,
        covariance: Covariance::Full,
        precision_prior: 0.1,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn marginal_cfg() -> MarginalCfg {
    MarginalCfg {
        n_features: K,
        n_targets: 2,
        decay: decay(),
        min_weight: vec![3.0; 2],
        lags: vec![],
        serial_rule: None,
        cross_lags: None,
        bins: None,
        feature_moments: FeatureMomentLayout::PerTarget,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

fn kmeans_cfg() -> KMeansCfg {
    KMeansCfg {
        n_features: K,
        k: 3,
        decay: decay(),
        min_weight: 3.0,
        warm_rows: 10,
        seed_rule: SeedRule::Lloyd,
        seed: 0,
        update_every_rows: 1,
        split_merge: 0.5,
        split_merge_every_rows: 20,
        dead_frac: 0.05,
        standardize: true,
        scale_floor: 0.1,
        warm_clock: None,
        update_every: None,
        split_merge_every: None,
    }
}

fn sgd_cfg(loss: SgdLoss, n_targets: usize) -> SgdCfg {
    SgdCfg {
        n_features: K,
        n_targets,
        fit_intercept: true,
        decay: decay(),
        loss,
        learning_rate: 0.05,
        schedule: LearningRate::AdaGrad,
        l2: 0.01,
        clip_gradient: 1e3,
        constraint: None,
        standardize: true,
        min_weight: 3.0,
        strict_binary: false,
        huber_scale: rheon_core::HuberScale::Rms,
        huber_scale_decay: None,
        refit_rows: 0,
        // The cases a state written before schema 64 converts to: the gate
        // held learning back then (docs/PLAN.md task 278, D10). The form
        // since is the refit cases' and `quantile_mad`'s.
        min_weight_holds_learning: true,
        target_spread: false,
        mad_anchor: false,
        refit_clock: None,
        max_state_bytes: None,
    }
}

/// A Gamma `glm` of two targets whose refit fills on its `refit_rows`-th
/// learned row.
fn glm_cfg(refit_rows: u32) -> GlmCfg {
    GlmCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        family: GlmFamily::Gamma,
        link: GlmLink::Log,
        var_power: 1.5,
        ridge: 1.0,
        min_weight: 3.0,
        refit_rows,
        refit_clock: None,
        step_cap: f64::INFINITY,
        max_state_bytes: None,
    }
}

fn pa_cfg() -> PaCfg {
    PaCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        mode: PaMode::Pa1,
        c: 1.0,
        eps: 0.1,
        min_weight: 3.0,
        constraint: None,
        standardize: true,
    }
}

fn bocpd_cfg() -> BocpdCfg {
    BocpdCfg {
        n_features: K,
        hazard: 20.0,
        hazard_from_row: false,
        emission: BocpdEmission::Diag,
        prior_mean: None,
        prior_kappa: 1.0,
        prior_nu: Some(4.0),
        prior_scale: Some(vec![1.0]),
        robust_beta: 0.0,
        prune_below: 1e-6,
        max_run: 60,
        min_weight: 0.0,
        warm_rows: None,
        hazard_on_clock: true,
        warm_clock: None,
    }
}

fn corrchange_cfg(kind: CorrChangeKind) -> CorrChangeCfg {
    CorrChangeCfg {
        n_features: K,
        kind,
        span_rows: 10,
        alpha: 0.05,
        alpha_adjust: "bonferroni".into(),
        bandwidth: None,
        scalar: false,
        decay: decay(),
        crit: None,
        n_perm: 20,
        permute_every_rows: 10,
        perm_block: 1,
        norm: ChangeNorm::L1,
        seed: 5,
        reset: false,
        monitor_rows: 15,
        boundary_gamma: 0.0,
        permute_every: None,
        span_clock: None,
        monitor_clock: None,
        max_state_bytes: None,
    }
}

/// The window every windowed case runs: twelve clock units, a snapshot
/// every three.
const WINDOW: (Option<f64>, Option<f64>, Option<usize>) = (Some(12.0), Some(3.0), None);

/// A fresh `m` as a build of schema 64 saved it, restored by this build:
/// each anchored sketch awaiting its first value (an anchor of 0, where
/// this build fixes the grid at 1), each refit buffer holding no `|e|`, and
/// no deviations held for a histogram's refit -- the forms that build
/// wrote, every other field the same bytes under either build -- and the
/// state relabelled 64. Its restore marks each anchored sketch to keep
/// anchoring at its first value, and the model goes on as that build did
/// (schema 65): the forms a 64 state takes in a state this build saves.
pub fn restored_from_64<M: OnlineModel>(m: M) -> M {
    fn old(v: &mut rmpv::Value) {
        match v {
            rmpv::Value::Map(m) => {
                for (k, v) in m.iter_mut() {
                    match k.as_str() {
                        Some("anchor") if v.as_f64() == Some(1.0) => *v = rmpv::Value::from(0.0),
                        Some("resid") => *v = rmpv::Value::Nil,
                        // A histogram's held deviations; the warm-up's
                        // held rows are a list, and stay.
                        Some("held") if v.is_map() => *v = rmpv::Value::Nil,
                        _ => old(v),
                    }
                }
            }
            rmpv::Value::Array(a) => a.iter_mut().for_each(old),
            _ => {}
        }
    }
    let mut s = m.state();
    s.schema_version = 64;
    let bytes = rmp_serde::to_vec_named(&s).unwrap();
    let mut v = rmpv::decode::read_value(&mut &bytes[..]).unwrap();
    old(&mut v);
    let mut out = Vec::new();
    rmpv::encode::write_value(&mut out, &v).unwrap();
    let s: State = rmp_serde::from_slice(&out).unwrap();
    M::restore(&s).expect("a fresh state as 64 wrote it restores")
}

macro_rules! case {
    ($name:literal, $variant:literal, $ty:ty, $build:expr, $k:expr, $targets:expr, $before:expr) => {
        Case {
            name: $name,
            variant: $variant,
            build: || Box::new(Model::<$ty>($build)),
            restore: restore::<$ty>,
            n_features: $k,
            targets: $targets,
            before: $before,
        }
    };
}

/// Every case, in file order.
pub fn all() -> Vec<Case> {
    vec![
        Case {
            name: "ew_cov_accumulator",
            variant: "EwCov",
            build: || Box::new(Accumulator(EwCov::new(K))),
            restore: restore_accumulator,
            n_features: K,
            targets: Targets::None,
            before: 40,
        },
        case!(
            "ewridge",
            "EwRidge",
            EwRidge,
            EwRidge::new(ewridge_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "ewridge_window",
            "EwRidge",
            EwRidge,
            EwRidge::new(EwRidgeCfg {
                target_gaps: TargetGaps::OwnRows,
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ewridge_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Keeping each solve's system for the per-row leverage
        // (`emit_error_inflation`): the `systems` of schema 38, empty in
        // every other ewridge state.
        case!(
            "ewridge_leverage",
            "EwRidge",
            EwRidge,
            {
                let mut m = EwRidge::new(ewridge_cfg()).unwrap();
                m.set_keep_factor(true);
                m
            },
            K,
            Targets::Linear(2),
            40
        ),
        // `min_weight` withholding the output alone (schema 64, docs/PLAN.md
        // task 278): the residual variance folds the prediction at the
        // model's own floor, a row per unknown, on the rows the gate
        // withholds, and the first solve falls on the row the weight
        // reaches that floor. The clock gap on row 36 takes the weight back
        // under the gate, which withholds the continuation's first rows
        // while the variance folds them. Every other `ewridge` case is the
        // form a state saved before it goes on in.
        case!(
            "ewridge_output_only",
            "EwRidge",
            EwRidge,
            {
                let mut m = EwRidge::new(EwRidgeCfg {
                    min_weight: 12.0,
                    ..ewridge_cfg()
                })
                .unwrap();
                m.set_target_min_weight(vec![12.0, 12.0]).unwrap();
                m.set_min_weight_output_only(true);
                m
            },
            K,
            Targets::Linear(2),
            40
        ),
        // An explicit cadence's first period on the default weight-share
        // schedule (schema 64, task 278, "Default cadence first"): saved
        // inside it, with the mark set and the cadence's clock measured
        // from the model's start, and the continuation's row 42, 58 clock
        // units from it, ends the period and clears the mark. Every other
        // `ewridge` case is the form a state saved before it goes on in.
        case!(
            "ewridge_first_period",
            "EwRidge",
            EwRidge,
            {
                let mut m = EwRidge::new(EwRidgeCfg {
                    solve_every: FIRST_PERIOD_EVERY,
                    max_rows_between_solves: u32::MAX,
                    ..ewridge_cfg()
                })
                .unwrap();
                m.set_min_weight_output_only(true);
                m.set_first_period_by_share(true);
                m
            },
            K,
            Targets::Linear(2),
            40
        ),
        // A cadence that never comes due (`solve_every = inf`, task 278): the
        // spec sets the first period's mark and the model does not take it,
        // so there is no first period; it solves only at its own floor.
        case!(
            "ewridge_never",
            "EwRidge",
            EwRidge,
            {
                let mut m = EwRidge::new(EwRidgeCfg {
                    solve_every: f64::INFINITY,
                    max_rows_between_solves: u32::MAX,
                    ..ewridge_cfg()
                })
                .unwrap();
                m.set_min_weight_output_only(true);
                m.set_first_period_by_share(true);
                m
            },
            K,
            Targets::Linear(2),
            40
        ),
        // A blocked Gram saved mid-block (schema 62, docs/PLAN.md task
        // 257): its held rows, and each target's values on them waiting for
        // the block's merge, `acc.held`, empty in every other state.
        case!(
            "ewridge_blocked",
            "EwRidge",
            EwRidge,
            EwRidge::new(EwRidgeCfg {
                gram_block_rows: 8,
                solve_every: 5.0,
                max_rows_between_solves: 10,
                ..ewridge_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            43
        ),
        // Weights whose squares are not doubles (schema 72, review
        // 2026-11-23, REG-12): each Kish sum's exponent, written.
        Case {
            name: "ewridge_tiny_weights",
            variant: "EwRidge",
            build: || {
                Box::new(TinyWeights(
                    EwRidge::new(EwRidgeCfg {
                        min_weight: 0.0,
                        ..ewridge_cfg()
                    })
                    .unwrap(),
                ))
            },
            restore: restore_tiny::<EwRidge>,
            n_features: K,
            targets: Targets::Linear(2),
            before: 40,
        },
        Case {
            name: "rls_tiny_weights",
            variant: "Rls",
            build: || {
                Box::new(TinyWeights(
                    Rls::new(RlsCfg {
                        n_features: K,
                        n_targets: 2,
                        fit_intercept: true,
                        decay: decay(),
                        delta: 1e-200,
                        coef_prior: None,
                        min_weight: 0.0,
                    })
                    .unwrap(),
                ))
            },
            restore: restore_tiny::<Rls>,
            n_features: K,
            targets: Targets::Linear(2),
            before: 40,
        },
        Case {
            name: "sgd_tiny_weights",
            variant: "Sgd",
            build: || {
                Box::new(TinyWeights(
                    Sgd::new(SgdCfg {
                        target_spread: true,
                        min_weight_holds_learning: false,
                        min_weight: 0.0,
                        ..sgd_cfg(SgdLoss::Squared, 2)
                    })
                    .unwrap(),
                ))
            },
            restore: restore_tiny::<Sgd>,
            n_features: K,
            targets: Targets::Linear(2),
            before: 40,
        },
        case!(
            "rls",
            "Rls",
            Rls,
            Rls::new(RlsCfg {
                n_features: K,
                n_targets: 2,
                fit_intercept: true,
                decay: decay(),
                delta: 1.0,
                coef_prior: None,
                min_weight: 3.0,
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "lasso",
            "Lasso",
            Lasso,
            Lasso::new(lasso_cfg()).unwrap(),
            K,
            Targets::Linear(1),
            40
        ),
        case!(
            "lasso_window",
            "Lasso",
            Lasso,
            Lasso::new(LassoCfg {
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..lasso_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(1),
            40
        ),
        // `min_weight` withholding the output alone (schema 64, docs/PLAN.md
        // task 278, D10): the selection counts from the floor, a row per
        // coefficient, and the gate, which the clock gap on row 36 takes the
        // weight back under, withholds the continuation's first rows while
        // the selection scores them. `lasso` and `lasso_window` are the form
        // a state saved before it goes on in.
        case!(
            "lasso_floor",
            "Lasso",
            Lasso,
            Lasso::new(LassoCfg {
                min_weight_holds_learning: false,
                min_weight: 10.0,
                ..lasso_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(1),
            40
        ),
        // The first period's form (`ewridge_first_period`'s comment).
        case!(
            "lasso_first_period",
            "Lasso",
            Lasso,
            {
                let mut m = Lasso::new(LassoCfg {
                    min_weight_holds_learning: false,
                    solve_every: FIRST_PERIOD_EVERY,
                    max_rows_between_solves: u32::MAX,
                    ..lasso_cfg()
                })
                .unwrap();
                m.set_first_period_by_share(true);
                m
            },
            K,
            Targets::Linear(1),
            40
        ),
        case!(
            "kalman",
            "Kalman",
            Kalman,
            Kalman::new(kalman_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Saved after row 40, whose targets are null: each covariance holds
        // the clock since its last observation (docs/PLAN.md task 211).
        case!(
            "kalman_gap",
            "Kalman",
            Kalman,
            Kalman::new(kalman_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            41
        ),
        // `P` in units of the noise (schema 64, docs/PLAN.md task 278), saved
        // after three rows whose targets are null: each covariance's
        // intercept still diffuse, its row and column 0, the moments and the
        // clock since an observation moved; the continuation's first row
        // that observes the targets takes the diffuse step. `kalman` and
        // `kalman_gap` are the form a state saved before it goes on in.
        case!(
            "kalman_noise_units",
            "Kalman",
            Kalman,
            Kalman::new(KalmanCfg {
                noise_units: true,
                ..kalman_cfg()
            })
            .unwrap(),
            K,
            Targets::LinearFrom(2, 3),
            3
        ),
        // The form a spec builds (schema 64, docs/PLAN.md task 278, D10):
        // `P` in units of the noise and `min_weight` withholding the output
        // alone, so the residual variance learns from the rows the gate
        // withholds -- the continuation's first, after the clock gap on row
        // 36 took the weight back under it -- and the explicit `q`, read
        // over it, carries that into the predictions. `kalman`, `kalman_gap`
        // and `kalman_noise_units` hold learning back, the form a state
        // saved before goes on in.
        case!(
            "kalman_output_only",
            "Kalman",
            Kalman,
            Kalman::new(KalmanCfg {
                noise_units: true,
                min_weight_holds_learning: false,
                min_weight: 10.0,
                q: Some(vec![1e-3, 1e-3, 1e-3]),
                ..kalman_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The square root's form (schema 68, docs/PLAN.md task 281 (3)): the
        // spec's form, `P` in units of the noise with its intercepts
        // diffuse until a row observes the targets, held as its factor
        // `L = Cᵀ`, at a vague prior, saved past the diffuse step and the
        // drift, with the reversion's coupling, so every write to the
        // factor has acted. `kalman_output_only` is the same filter held as
        // `P`, the covariance form, which a state saved before 68 is.
        case!(
            "kalman_square_root",
            "Kalman",
            Kalman,
            Kalman::new(KalmanCfg {
                noise_units: true,
                min_weight_holds_learning: false,
                min_weight: 10.0,
                p0: 1e12,
                ..kalman_cfg()
            })
            .unwrap()
            .with_covariance_form(rheon_core::CovarianceForm::SquareRoot)
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "huber",
            "Robust",
            Robust,
            Robust::new(robust_cfg(RobustLoss::Huber { delta: 1.5 })).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The first period's form (`ewridge_first_period`'s comment).
        case!(
            "huber_first_period",
            "Robust",
            Robust,
            {
                let mut m = Robust::new(RobustCfg {
                    min_weight_holds_learning: false,
                    solve_every: FIRST_PERIOD_EVERY,
                    max_rows_between_solves: u32::MAX,
                    ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
                })
                .unwrap();
                m.set_first_period_by_share(true);
                m
            },
            K,
            Targets::Linear(2),
            40
        ),
        // A target short of the model's floor, a row per coefficient (three
        // here): the second target joins on row 20 and the state is saved
        // after it, its count at 1 of 3 and the first's at 3 (review round
        // 10, S2-3). The continuation's next rows with it each solve the fit
        // for its sake until it reaches the floor (`Robust::own_solve`).
        case!(
            "huber_joining",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                min_weight_holds_learning: false,
                solve_every: f64::INFINITY,
                max_rows_between_solves: u32::MAX,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::JoiningFrom(2, 20),
            21
        ),
        // The MAD's sketch of `|e|` per target and the scale's own memory
        // (schema 63, docs/PLAN.md task 271); `huber` is the RMS's form.
        case!(
            "huber_mad",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                huber_scale_decay: Some(Decay::Halflife(8.0)),
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The MAD's sketches anchored, in units of each target's first
        // `|e|` (schema 64, task 278, the user's 2026-10-26): saved after
        // the anchor is set, and saved after two rows, before any
        // prediction and so before any residual, the anchors awaiting
        // their first value, which the continuation sets. `huber_mad` is
        // the form a state saved before it goes on in.
        case!(
            "huber_mad_anchored",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                mad_anchor: true,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "huber_mad_awaiting",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                mad_anchor: true,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            2
        ),
        // A state of schema 64 saved while its anchors were awaited, read
        // by this build (schema 65): each sketch marked to keep anchoring
        // at its first residual, as the build that saved it did, where a
        // sketch this build makes keeps the grid fixed at 1. Saved before
        // any residual, so the continuation sets the anchors.
        case!(
            "huber_mad_awaiting_first_value",
            "Robust",
            Robust,
            restored_from_64(
                Robust::new(RobustCfg {
                    huber_scale: rheon_core::HuberScale::Mad,
                    mad_anchor: true,
                    ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
                })
                .unwrap()
            ),
            K,
            Targets::Linear(2),
            2
        ),
        // Feature sets from each target's one set of sums, the rows weighed
        // by the largest set's residual (schema 63, docs/PLAN.md task 272
        // (a)).
        case!(
            "huber_sets",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                feature_sets: vec![("one".into(), vec![1]), ("all".into(), (0..K).collect())],
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // RLS's sum-scale prior on each target's raw moments (schema 63,
        // docs/PLAN.md task 272 (a)), its prior scale aged in the sums, and
        // the emitted scale switched on.
        case!(
            "huber_sum",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                ridge: 0.5,
                standardize: false,
                ridge_scale: true,
                emit_scale: true,
                refit_rows: 0,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The refit of each target's first rows (schema 64, docs/PLAN.md
        // task 277): saved with its buffers part full, the refit falling in
        // the continuation, and saved after it, the buffers freed and the
        // scale rebuilt from the converged residuals (the MAD's sketch in
        // the first, the RMS in the second).
        // Task 538: the refit over a span of the clock, saved inside it.
        case!(
            "huber_refit_span",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_clock: Some(60.0),
                max_state_bytes: Some(200_000),
                min_weight_holds_learning: false,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "quantile_refit_span",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_clock: Some(60.0),
                max_state_bytes: Some(200_000),
                min_weight_holds_learning: false,
                mad_anchor: true,
                ..robust_cfg(RobustLoss::Quantile { tau: 0.75 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "sgd_huber_refit_span",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_clock: Some(60.0),
                max_state_bytes: Some(200_000),
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "huber_refit",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_rows: 40,
                min_weight_holds_learning: false,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // A state of schema 64 saved mid-buffer, read by this build: its
        // buffers hold no `|e|`, so its scale is read from the sketch it
        // feeds, anchored at its first residual, until the refit, which
        // falls in the continuation (task 323).
        case!(
            "huber_refit_first_value",
            "Robust",
            Robust,
            restored_from_64(
                Robust::new(RobustCfg {
                    huber_scale: rheon_core::HuberScale::Mad,
                    mad_anchor: true,
                    refit_rows: 40,
                    min_weight_holds_learning: false,
                    ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
                })
                .unwrap()
            ),
            K,
            Targets::Linear(2),
            30
        ),
        case!(
            "huber_refit_done",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                refit_rows: 20,
                min_weight_holds_learning: false,
                ..robust_cfg(RobustLoss::Huber { delta: 1.5 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "quantile",
            "Robust",
            Robust,
            Robust::new(robust_cfg(RobustLoss::Quantile { tau: 0.75 })).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The quantile band drawn in the MAD, each target's sketch of `|e|`
        // in the state, and `min_weight` withholding the output alone
        // (schema 64, docs/PLAN.md task 278, D-Q and D10); `quantile` is the
        // RMS's form and the gate's old one. Its sketches are anchored, the
        // one form a spec writes (task 278, the anchored sketch): a quantile
        // band in the MAD is new in schema 64, so no state has the fixed
        // buckets' form.
        case!(
            "quantile_mad",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                min_weight_holds_learning: false,
                mad_anchor: true,
                ..robust_cfg(RobustLoss::Quantile { tau: 0.75 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The quantile refit of each target's first rows (schema 64,
        // docs/PLAN.md task 278, D-Q2): saved with its buffers part full,
        // the refit falling in the continuation, and saved after it, the
        // sums rebuilt at the converged fit and the scale from its
        // residuals; both in the form a spec writes.
        case!(
            "quantile_refit",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                min_weight_holds_learning: false,
                mad_anchor: true,
                refit_rows: 40,
                ..robust_cfg(RobustLoss::Quantile { tau: 0.75 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "quantile_refit_done",
            "Robust",
            Robust,
            Robust::new(RobustCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                min_weight_holds_learning: false,
                mad_anchor: true,
                refit_rows: 20,
                ..robust_cfg(RobustLoss::Quantile { tau: 0.75 })
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "ftrl",
            "Ftrl",
            Ftrl,
            Ftrl::new(FtrlCfg {
                n_features: K,
                n_targets: 1,
                fit_intercept: true,
                decay: decay(),
                alpha: 0.1,
                beta: 1.0,
                l1: 0.01,
                l2: 1.0,
                min_weight: 3.0,
                strict_binary: false,
                loss: FtrlLoss::Logistic,
                standardize: false,
            })
            .unwrap(),
            K,
            Targets::Binary,
            40
        ),
        // Under `standardize` (PLAN 569, at schema 72): the scaler and its
        // warm-up count, past the count's 22 (40 rows) and inside it (20
        // rows, the continuation crossing it), and the cfg's flag.
        case!(
            "ftrl_standardized",
            "Ftrl",
            Ftrl,
            Ftrl::new(FtrlCfg {
                n_features: K,
                n_targets: 2,
                fit_intercept: true,
                decay: decay(),
                alpha: 0.1,
                beta: 1.0,
                l1: 0.01,
                l2: 1.0,
                min_weight: 3.0,
                strict_binary: false,
                loss: FtrlLoss::Squared,
                standardize: true,
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "ftrl_standardized_warming",
            "Ftrl",
            Ftrl,
            Ftrl::new(FtrlCfg {
                n_features: K,
                n_targets: 1,
                fit_intercept: false,
                decay: decay(),
                alpha: 0.1,
                beta: 1.0,
                l1: 0.01,
                l2: 1.0,
                min_weight: 3.0,
                strict_binary: false,
                loss: FtrlLoss::Logistic,
                standardize: true,
            })
            .unwrap(),
            K,
            Targets::Binary,
            20
        ),
        case!(
            "ew_cov",
            "EwCovModel",
            EwCovModel,
            EwCovModel::new(ew_cov_cfg()).unwrap(),
            3,
            Targets::None,
            40
        ),
        case!(
            "ew_cov_window",
            "EwCovModel",
            EwCovModel,
            EwCovModel::new(EwCovCfg {
                bias: true,
                stats: vec![EwCovStat::Mean, EwCovStat::Var, EwCovStat::Corr],
                mahal_quantiles: Vec::new(),
                pca: 0,
                pca_every: 0.0,
                max_rows_between_pca: u32::MAX,
                lags: Vec::new(),
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ew_cov_cfg()
            })
            .unwrap(),
            3,
            Targets::None,
            40
        ),
        case!(
            "sgd",
            "Sgd",
            Sgd,
            Sgd::new(sgd_cfg(SgdLoss::Squared, 2)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Saved inside the scaler's warm-up (schema 47, docs/PLAN.md task
        // 206): after 20 rows Kish's count of the weights is 14.8, and the
        // continuation crosses the switch at 22. The 40-row cases are past
        // it (29.1).
        case!(
            "sgd_warming",
            "Sgd",
            Sgd,
            Sgd::new(sgd_cfg(SgdLoss::Squared, 2)).unwrap(),
            K,
            Targets::Linear(2),
            20
        ),
        // The per-loss state: the residual scale `sig2`/`wsig` under the
        // Huber loss (schema 40), the target's own spread under the
        // epsilon-insensitive one (44), and the logistic link.
        case!(
            "sgd_huber",
            "Sgd",
            Sgd,
            Sgd::new(sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The MAD's sketch under the Huber loss, on the model's clock
        // (schema 63, docs/PLAN.md task 271).
        case!(
            "sgd_huber_mad",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                ..sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The MAD's sketches anchored under the Huber loss (schema 64, task
        // 278); `sgd_huber_mad` is the form a state saved before it goes
        // on in.
        case!(
            "sgd_huber_mad_anchored",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                mad_anchor: true,
                ..sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // `sgd`'s refit (schema 64, docs/PLAN.md task 277): saved with its
        // buffers part full, and saved after a refit that fell while the
        // scaler warmed up, so its fit was set in the moments' coordinates.
        case!(
            "sgd_huber_refit",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_rows: 40,
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "sgd_huber_refit_done",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                huber_scale: rheon_core::HuberScale::Mad,
                refit_rows: 10,
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Huber { delta: 1.5 }, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // The target's own units (schema 64, task 278: D1, D2, D8, D-S):
        // the target's spread under the squared loss and the step gains
        // under a shared rate, at a stable rate and at one past its bound,
        // saved with the gate withholding, and under AdaGrad, which keeps
        // the spread for its epsilon and no gains.
        case!(
            "sgd_spread",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                schedule: LearningRate::InvScaling { power: 0.5 },
                target_spread: true,
                // The form since schema 64 (docs/PLAN.md task 278, D10).
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Squared, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "sgd_unstable",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                schedule: LearningRate::Constant,
                learning_rate: 2.0,
                target_spread: true,
                // The form since schema 64 (docs/PLAN.md task 278, D10).
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Squared, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // A calm gate, resting (task 314): a stable rate, saved after its
        // sums hold two half-lives of rows, so its continuation reads the
        // flag from the file and takes no gains until a wide residual.
        case!(
            "sgd_resting",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                schedule: LearningRate::Constant,
                learning_rate: 0.01,
                target_spread: true,
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Squared, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            300
        ),
        case!(
            "sgd_spread_adagrad",
            "Sgd",
            Sgd,
            Sgd::new(SgdCfg {
                target_spread: true,
                // The form since schema 64 (docs/PLAN.md task 278, D10).
                min_weight_holds_learning: false,
                ..sgd_cfg(SgdLoss::Squared, 2)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "sgd_eps",
            "Sgd",
            Sgd,
            Sgd::new(sgd_cfg(SgdLoss::EpsilonInsensitive { eps: 0.5 }, 2)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "sgd_logistic",
            "Sgd",
            Sgd,
            Sgd::new(sgd_cfg(SgdLoss::Logistic, 1)).unwrap(),
            K,
            Targets::Binary,
            40
        ),
        case!(
            "pa",
            "Pa",
            Pa,
            Pa::new(pa_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Inside the scaler's warm-up, as `sgd_warming`.
        case!(
            "pa_warming",
            "Pa",
            Pa,
            Pa::new(pa_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            20
        ),
        // Saved with each target's refit buffer half full.
        case!(
            "glm_refit",
            "Glm",
            Glm,
            Glm::new(glm_cfg(60)).unwrap(),
            K,
            Targets::Positive(2),
            40
        ),
        // Saved after each target's refit ran, its buffer freed.
        case!(
            "glm_refit_done",
            "Glm",
            Glm,
            Glm::new(glm_cfg(15)).unwrap(),
            K,
            Targets::Positive(2),
            40
        ),
        // Gamma amounts drawn from the family at a log-linear mean, saved
        // after each target's refit ran: a fit with a dispersion to learn.
        case!(
            "glm_gamma_draws",
            "Glm",
            Glm,
            Glm::new(glm_cfg(15)).unwrap(),
            K,
            Targets::GammaDraw(2),
            120
        ),
        // No refit: a Tweedie fit at p = 1.5 held to the step cap.
        case!(
            "glm_cap",
            "Glm",
            Glm,
            Glm::new(GlmCfg {
                family: GlmFamily::Tweedie,
                refit_rows: 0,
                refit_clock: None,
                step_cap: 1.0,
                ..glm_cfg(0)
            })
            .unwrap(),
            K,
            Targets::Positive(2),
            40
        ),
        case!(
            "holt",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                horizons: Vec::new(),
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                trend: true,
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Without decay, the weighted line fit alone (schema 64, docs/PLAN.md
        // task 278, D-H): its weight, mean age and spread per target.
        // A series with a trend, as `holt` is for: the level and the slope
        // the clock carries, a gap moving the line by its length.
        case!(
            "holt_trending",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                horizons: Vec::new(),
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                trend: true,
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Trending(2),
            80
        ),
        case!(
            "holt_inf",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                horizons: Vec::new(),
                n_targets: 2,
                level_half_life: f64::INFINITY,
                trend_half_life: f64::INFINITY,
                min_weight: 3.0,
                trend: true,
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // The fixed forms of seasonal smoothing (schema 72, task 488). An
        // additive season under the known start, 4 values over 8 clock
        // units, so the clock of 1 puts two rows in a bucket and its gap of
        // 9 leaves a bucket empty.
        case!(
            "holt_season_add",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                season: HoltSeason::Add,
                season_periods: 4,
                season_length: 8.0,
                seasonal_half_life: 3.0 * H,
                start: HoltStart::Known,
                initial_level: vec![0.25, -0.75],
                initial_trend: vec![0.0, 0.01],
                initial_seasonal: vec![vec![0.1, -0.1, 0.05, -0.05], vec![0.0, 0.2, -0.2, 0.0]],
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // A multiplicative season and a damped trend, known.
        case!(
            "holt_season_mul_damped",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                season: HoltSeason::Mul,
                season_periods: 3,
                seasonal_half_life: 3.0 * H,
                damping_half_life: 2.0 * H,
                start: HoltStart::Known,
                initial_level: vec![1.0, 2.0],
                initial_trend: vec![0.0, -0.01],
                initial_seasonal: vec![vec![1.1, 0.9, 1.0], vec![0.95, 1.0, 1.05]],
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Positive(2),
            40
        ),
        // The heuristic start, fitted (12 rows at m = 2) and learning on.
        case!(
            "holt_heuristic",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                season: HoltSeason::Add,
                season_periods: 2,
                seasonal_half_life: 3.0 * H,
                start: HoltStart::Heuristic,
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // The heuristic start saved while it holds its rows: 20 wanted at
        // m = 4, fewer held after 14 rows.
        case!(
            "holt_heuristic_holding",
            "Holt",
            Holt,
            Holt::new(HoltCfg {
                n_targets: 2,
                level_half_life: H,
                trend_half_life: 4.0 * H,
                min_weight: 3.0,
                season: HoltSeason::Mul,
                season_periods: 4,
                seasonal_half_life: 3.0 * H,
                damping_half_life: 2.0 * H,
                start: HoltStart::Heuristic,
                ..Default::default()
            })
            .unwrap(),
            0,
            Targets::Positive(2),
            14
        ),
        // Saved with its warm-up buffer still filling: 10 rows wanted, 6
        // held.
        case!(
            "kmeans_warming",
            "KMeans",
            KMeans,
            KMeans::new(kmeans_cfg()).unwrap(),
            K,
            Targets::None,
            6
        ),
        case!(
            "kmeans",
            "KMeans",
            KMeans,
            KMeans::new(kmeans_cfg()).unwrap(),
            K,
            Targets::None,
            40
        ),
        // Task 538: a warm-up over a span of the clock and both cadences
        // on the clock, saved inside the warm-up and after the seeding.
        case!(
            "kmeans_spans_warming",
            "KMeans",
            KMeans,
            KMeans::new(KMeansCfg {
                warm_rows: 3,
                warm_clock: Some(12.0),
                update_every_rows: u32::MAX,
                update_every: Some(3.0),
                split_merge_every_rows: u32::MAX,
                split_merge_every: Some(15.0),
                ..kmeans_cfg()
            })
            .unwrap(),
            K,
            Targets::None,
            6
        ),
        case!(
            "kmeans_spans",
            "KMeans",
            KMeans,
            KMeans::new(KMeansCfg {
                warm_rows: 3,
                warm_clock: Some(12.0),
                update_every_rows: u32::MAX,
                update_every: Some(7.5),
                split_merge_every_rows: u32::MAX,
                split_merge_every: Some(15.0),
                ..kmeans_cfg()
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "micro",
            "Micro",
            Micro,
            Micro::new(MicroCfg {
                n_features: K,
                decay: decay(),
                min_weight: 3.0,
                eps: 0.6,
                beta_mu: 2.0,
                max_clusters: 50,
                prune_every: 6.0,
                max_rows_between_prunes: 10,
                macro_link: None,
                standardize: true,
                scale_floor: 0.1,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "ew_class",
            "EwClass",
            EwClass,
            EwClass::new(ew_class_cfg()).unwrap(),
            K,
            Targets::Class(3),
            40
        ),
        // The diagonal shape, which keeps each class's variances alone
        // (schema 66, task 327); `ew_class_window` is its windowed form.
        case!(
            "ew_class_diag",
            "EwClass",
            EwClass,
            EwClass::new(EwClassCfg {
                covariance: Covariance::Diagonal,
                ..ew_class_cfg()
            })
            .unwrap(),
            K,
            Targets::Class(3),
            40
        ),
        case!(
            "ew_class_window",
            "EwClass",
            EwClass,
            EwClass::new(EwClassCfg {
                covariance: Covariance::Diagonal,
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..ew_class_cfg()
            })
            .unwrap(),
            K,
            Targets::Class(3),
            40
        ),
        case!(
            "seqtest",
            "SeqTest",
            SeqTest,
            SeqTest::new(SeqTestCfg {
                n_targets: 2,
                min_weight: 3.0,
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        case!(
            "marginal_lags",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                lags: vec![1, 2],
                cross_lags: Some(vec![2]),
                feature_moments: FeatureMomentLayout::Shared,
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Saved while the bins hold their warm-up rows: 30 wanted.
        case!(
            "marginal_bins",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 30,
                    stat: rheon_core::BinStat::Mean,
                    refit_rows: 0,
                    budget_mib: None,
                    mad_anchor: false,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            24
        ),
        // Huber's location per bin (63, docs/PLAN.md task 272 (c)): saved
        // past the warm-up, with every cell's robust weight and each
        // target's location and scale.
        case!(
            "marginal_bins_huber",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 12,
                    stat: rheon_core::BinStat::Huber { delta: 1.0 },
                    refit_rows: 0,
                    budget_mib: None,
                    mad_anchor: false,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            30
        ),
        // The bins' Huber refit (64, docs/PLAN.md task 278): saved with
        // given edges while it holds its rows (30 wanted, 21 held; the
        // continuation reaches the count), and saved after
        // it ran past a learned warm-up of 12 rows, at 20, with each
        // target's passes.
        case!(
            "marginal_bins_refit",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: Some(vec![vec![-0.5, 0.0, 0.5]; K]),
                    rule: BinRule::Quantile,
                    warm_rows: 12,
                    stat: rheon_core::BinStat::Huber { delta: 1.0 },
                    refit_rows: 30,
                    budget_mib: None,
                    mad_anchor: true,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            24
        ),
        case!(
            "marginal_bins_refit_done",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 12,
                    stat: rheon_core::BinStat::Huber { delta: 1.0 },
                    refit_rows: 20,
                    budget_mib: None,
                    mad_anchor: true,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            30
        ),
        // A state of schema 64 saved inside a learned warm-up, read by this
        // build: the histogram its continuation builds at the freeze (row
        // 12) anchors each sketch at its first deviation and feeds it every
        // deviation until the refit (row 20), as that build did, where one
        // this build makes holds the deviations (task 323).
        case!(
            "marginal_bins_refit_first_value",
            "Marginal",
            Marginal,
            restored_from_64(
                Marginal::new(MarginalCfg {
                    bins: Some(Box::new(BinCfg {
                        n_bins: 4,
                        edges: None,
                        rule: BinRule::Quantile,
                        warm_rows: 12,
                        stat: rheon_core::BinStat::Huber { delta: 1.0 },
                        refit_rows: 20,
                        budget_mib: None,
                        mad_anchor: true,
                    })),
                    ..marginal_cfg()
                })
                .unwrap()
            ),
            K,
            Targets::Linear(2),
            8
        ),
        // The same state read by this build and saved past its freeze (row
        // 12), before its refit (row 20): the histogram was built by the
        // schema-64 rule above, and the mark that built it so has acted
        // and is cleared (task 340, schema 67; before it the mark was
        // written on every later save), so the saved form is the current
        // one. The 66 build's file is kept under `v66/`, marked: it loads,
        // goes on to the bit and converts to this one's bytes.
        case!(
            "marginal_bins_refit_first_value_frozen",
            "Marginal",
            Marginal,
            restored_from_64(
                Marginal::new(MarginalCfg {
                    bins: Some(Box::new(BinCfg {
                        n_bins: 4,
                        edges: None,
                        rule: BinRule::Quantile,
                        warm_rows: 12,
                        stat: rheon_core::BinStat::Huber { delta: 1.0 },
                        refit_rows: 20,
                        budget_mib: None,
                        mad_anchor: true,
                    })),
                    ..marginal_cfg()
                })
                .unwrap()
            ),
            K,
            Targets::Linear(2),
            16
        ),
        // Each target's MAD sketch anchored (schema 64, task 278);
        // `marginal_bins_huber` is the form a state saved before it goes
        // on in.
        case!(
            "marginal_bins_huber_anchored",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                bins: Some(Box::new(BinCfg {
                    n_bins: 4,
                    edges: None,
                    rule: BinRule::Quantile,
                    warm_rows: 12,
                    stat: rheon_core::BinStat::Huber { delta: 1.0 },
                    refit_rows: 0,
                    budget_mib: None,
                    mad_anchor: true,
                })),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            30
        ),
        case!(
            "marginal_window",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Lags under a window take their moments about a fixed origin, with
        // the lagged sides' sums beside them (task 280, phase 3).
        case!(
            "marginal_window_lags",
            "Marginal",
            Marginal,
            Marginal::new(MarginalCfg {
                window: WINDOW.0,
                window_every: WINDOW.1,
                max_rows_between_snapshots: WINDOW.2,
                lags: vec![1, 2],
                cross_lags: Some(vec![2]),
                ..marginal_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        case!(
            "deco",
            "Deco",
            Deco,
            Deco::new(DecoCfg {
                n_features: 3,
                decay: decay(),
                dynamics: DecoDynamics::Ew,
                alpha: None,
                beta: None,
                blocks: Vec::new(),
                min_weight: 3.0,
            })
            .unwrap(),
            3,
            Targets::None,
            40
        ),
        // The linear form over two blocks: `N`, `D` and each pair's own
        // second denominator.
        case!(
            "deco_linear",
            "Deco",
            Deco,
            Deco::new(DecoCfg {
                n_features: 4,
                decay: decay(),
                dynamics: DecoDynamics::Linear,
                alpha: Some(0.05),
                beta: Some(0.9),
                blocks: vec![vec![0, 1], vec![2, 3]],
                min_weight: 0.0,
            })
            .unwrap(),
            4,
            Targets::None,
            40
        ),
        // Saved inside the standardiser's warm-up: no level yet, the share
        // above a fifth, and the continuation crosses it.
        case!(
            "deco_warming",
            "Deco",
            Deco,
            Deco::new(DecoCfg {
                n_features: 3,
                decay: decay(),
                dynamics: DecoDynamics::Ew,
                alpha: None,
                beta: None,
                blocks: Vec::new(),
                min_weight: 0.0,
            })
            .unwrap(),
            3,
            Targets::None,
            4
        ),
        case!(
            "rcov",
            "Rcov",
            Rcov,
            Rcov::new(RcovCfg {
                n_features: K,
                kind: RcovKind::Kernel,
                kernel: "parzen".into(),
                bandwidth: Some(3),
                jitter: 2,
                theta: 1.0,
                psd: true,
                block_rows: Some(8),
                max_bandwidth: None,
                preavg_rows: None,
                noise_stride: 1,
                iv_stride: 20,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "hmm",
            "Hmm",
            Hmm,
            Hmm::new(HmmCfg {
                n_features: K,
                k: 2,
                decay: decay(),
                // The pooled shape, so the fixture freezes its word, "tied"
                // (task 244, schema 61); `ew_class_window`'s is "diag".
                covariance: Covariance::Shared,
                precision_prior: 1e-2,
                min_weight: 0.0,
                learn: true,
                transition_prior: 1.0,
                transition: None,
                means: None,
                covs: None,
                warm_rows: 10,
                seed_rule: SeedRule::First,
                seed: 1,
                tvtp: None,
                prior_rows: None,
                warm_clock: None,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        // Task 538: a warm-up over a span of the clock, saved inside it.
        case!(
            "hmm_warm_span",
            "Hmm",
            Hmm,
            Hmm::new(HmmCfg {
                n_features: K,
                k: 2,
                decay: decay(),
                covariance: Covariance::Diagonal,
                precision_prior: 1e-2,
                min_weight: 0.0,
                learn: true,
                transition_prior: 1.0,
                transition: None,
                means: None,
                covs: None,
                warm_rows: 2,
                seed_rule: SeedRule::Lloyd,
                seed: 1,
                tvtp: None,
                prior_rows: None,
                warm_clock: Some(15.0),
            })
            .unwrap(),
            K,
            Targets::None,
            8
        ),
        // The diagonal shape, which keeps each state's variances alone
        // (schema 66, task 327).
        case!(
            "hmm_diag",
            "Hmm",
            Hmm,
            Hmm::new(HmmCfg {
                n_features: K,
                k: 2,
                decay: decay(),
                covariance: Covariance::Diagonal,
                precision_prior: 1e-2,
                min_weight: 0.0,
                learn: true,
                transition_prior: 1.0,
                transition: None,
                means: None,
                covs: None,
                warm_rows: 10,
                seed_rule: SeedRule::First,
                seed: 1,
                tvtp: None,
                prior_rows: None,
                warm_clock: None,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        // The full shape on each side of the width from which each state
        // keeps its covariance as a Cholesky factor (schema 69, task 479):
        // 13 features, saved some 25 learned rows after the factor was
        // taken up, so the factor has moved by its rotations and its scale;
        // and 12, which keeps the co-moments.
        case!(
            "hmm_full",
            "Hmm",
            Hmm,
            Hmm::new(hmm_full_cfg(13)).unwrap(),
            13,
            Targets::None,
            40
        ),
        // The full shape under a prior of its own, saved inside its
        // warm-up (schema 70): it holds `ŝ` at 0 and no factor, takes them up
        // at its seeding five rows after the save, and goes on under the
        // prior.
        case!(
            "hmm_full_warming",
            "Hmm",
            Hmm,
            Hmm::new(HmmCfg {
                warm_rows: 45,
                prior_rows: Some(6.5),
                ..hmm_full_cfg(13)
            })
            .unwrap(),
            13,
            Targets::None,
            40
        ),
        case!(
            "hmm_full_narrow",
            "Hmm",
            Hmm,
            Hmm::new(hmm_full_cfg(12)).unwrap(),
            12,
            Targets::None,
            40
        ),
        case!(
            "corrchange_monitor",
            "CorrChange",
            CorrChange,
            CorrChange::new(corrchange_cfg(CorrChangeKind::Monitor)).unwrap(),
            K,
            Targets::None,
            40
        ),
        // Two windows of 16 rows over two columns, four blocks of four
        // rows: their sums merged, the ring past its first row, so the
        // count of rows it has taken (schema 66) is more than it holds.
        case!(
            "corrchange_window",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 16,
                ..corrchange_cfg(CorrChangeKind::Window)
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        // Task 538: the window's redraw on the clock, saved between draws.
        case!(
            "corrchange_permute_every",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 16,
                permute_every_rows: usize::MAX,
                permute_every: Some(6.0),
                ..corrchange_cfg(CorrChangeKind::Window)
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "corrchange_sequential",
            "CorrChange",
            CorrChange,
            CorrChange::new(corrchange_cfg(CorrChangeKind::Sequential)).unwrap(),
            K,
            Targets::None,
            40
        ),
        // PLAN 550 and 553: the monitor and the sequential detector on a
        // correlated pair whose correlation moves from 0.3 to 0.85 at row
        // 150, saved inside a span and inside a monitoring period.
        case!(
            "corrchange_monitor_realistic",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 60,
                ..corrchange_cfg(CorrChangeKind::Monitor)
            })
            .unwrap(),
            K,
            Targets::CorrelatedPair(150),
            200
        ),
        case!(
            "corrchange_sequential_realistic",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 50,
                monitor_rows: 60,
                boundary_gamma: 0.25,
                ..corrchange_cfg(CorrChangeKind::Sequential)
            })
            .unwrap(),
            K,
            Targets::CorrelatedPair(150),
            200
        ),
        // Task 553: each kind's span on the clock (`window_size`), saved
        // inside it: the clock, the span's first stamp, a window's keys and
        // the counts its held value was drawn over, a cycle's own history
        // and period, and a cap of the state's bytes.
        case!(
            "corrchange_monitor_clock",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 0,
                span_clock: Some(7.0),
                max_state_bytes: Some(1 << 20),
                ..corrchange_cfg(CorrChangeKind::Monitor)
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "corrchange_sequential_clock",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 0,
                monitor_rows: 0,
                span_clock: Some(9.0),
                monitor_clock: Some(14.0),
                ..corrchange_cfg(CorrChangeKind::Sequential)
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        case!(
            "corrchange_window_clock",
            "CorrChange",
            CorrChange,
            CorrChange::new(CorrChangeCfg {
                span_rows: 0,
                span_clock: Some(8.0),
                permute_every_rows: usize::MAX,
                permute_every: Some(5.0),
                ..corrchange_cfg(CorrChangeKind::Window)
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        // Task 538: a warm-up over a span of the clock, saved inside it.
        case!(
            "bocpd_warm_span",
            "Bocpd",
            Bocpd,
            Bocpd::new(BocpdCfg {
                warm_rows: None,
                warm_clock: Some(15.0),
                ..bocpd_cfg()
            })
            .unwrap(),
            K,
            Targets::None,
            6
        ),
        // Saved with its warm-up buffer still filling: 10 learned rows
        // wanted, 5 held (schema 40's `WarmRow`), as `kmeans_warming` is.
        case!(
            "bocpd_warming",
            "Bocpd",
            Bocpd,
            Bocpd::new(BocpdCfg {
                warm_rows: Some(10),
                ..bocpd_cfg()
            })
            .unwrap(),
            K,
            Targets::None,
            6
        ),
        case!(
            "bocpd",
            "Bocpd",
            Bocpd,
            Bocpd::new(bocpd_cfg()).unwrap(),
            K,
            Targets::None,
            40
        ),
        // Saved past its cap and past a digest compression (256 values), so
        // the counters are a summary and the digest holds centroids beside
        // its buffer; with pairs, and gaps at `gap_cap` (docs/PLAN.md task
        // 223 (b)).
        case!(
            "audit",
            "Audit",
            Audit,
            Audit::new(AuditCfg {
                n_columns: K,
                pairs: true,
                distinct_cap: 8,
                gap_cap: Some(5.0),
                has_clock: true,
            })
            .unwrap(),
            K,
            Targets::None,
            300
        ),
        // The same audit over values past 2^65: its moments kept over a
        // power of two.
        Case {
            name: "audit_scaled",
            variant: "Audit",
            build: || {
                Box::new(ScaledAudit(
                    Audit::new(AuditCfg {
                        n_columns: K,
                        pairs: true,
                        distinct_cap: 8,
                        gap_cap: Some(5.0),
                        has_clock: true,
                    })
                    .unwrap(),
                ))
            },
            restore: restore_scaled_audit,
            n_features: K,
            targets: Targets::None,
            before: 40,
        },
        // Saved with its counts exact: fewer distinct values than its cap.
        case!(
            "audit_exact",
            "Audit",
            Audit,
            Audit::new(AuditCfg {
                n_columns: K,
                pairs: false,
                distinct_cap: 512,
                gap_cap: None,
                has_clock: false,
            })
            .unwrap(),
            K,
            Targets::None,
            40
        ),
        // Two nodes and two kernels, decayed: `R`, `b`, `Q` and `G₀₀`.
        case!(
            "hawkes",
            "Hawkes",
            Hawkes,
            Hawkes::new(hawkes_cfg(None)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Events that excite each other, as the process is for: sparse
        // rows, runs of events after an event, both nodes.
        case!(
            "hawkes_excited",
            "Hawkes",
            Hawkes,
            Hawkes::new(hawkes_cfg(None)).unwrap(),
            K,
            Targets::Excited(2),
            120
        ),
        // In a window of six rows with an event, saved with rows in both
        // blocks: the older block's offsets, marks and forward sums, the
        // newer's clocks, marks and sums.
        case!(
            "hawkes_window",
            "Hawkes",
            Hawkes,
            Hawkes::new(hawkes_cfg(Some(6))).unwrap(),
            K,
            Targets::Linear(2),
            45
        ),
        // Over a span of the clock, rows stamped: the older block's rows with
        // their stamps, the clock the model tracks and the last stamp.
        Case {
            name: "hawkes_clock_window",
            variant: "Hawkes",
            build: || {
                Box::new(StampedHawkes {
                    model: Hawkes::new(HawkesCfg {
                        decay: Decay::Halflife(f64::INFINITY),
                        window_clock: Some(6.0),
                        ..hawkes_cfg(None)
                    })
                    .unwrap(),
                    ns: 0,
                })
            },
            restore: restore_stamped_hawkes,
            n_features: K,
            targets: Targets::Linear(2),
            before: 45,
        },
        // The one pass, saved on a row of weight 0 (row 40): the sums, and
        // the clock that row stepped over, held for the next row that counts.
        case!(
            "expectile",
            "Expectile",
            Expectile,
            Expectile::new(expectile_cfg(0)).unwrap(),
            K,
            Targets::Linear(2),
            41
        ),
        // Saved with each target's refit buffer part full.
        case!(
            "expectile_refit",
            "Expectile",
            Expectile,
            Expectile::new(expectile_cfg(60)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Saved after each target's refit ran, its buffer freed.
        case!(
            "expectile_refit_done",
            "Expectile",
            Expectile,
            Expectile::new(expectile_cfg(15)).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // A full window of twelve rows, the fit its exact refit.
        case!(
            "expectile_window",
            "Expectile",
            Expectile,
            Expectile::new(ExpectileCfg {
                decay: Decay::Halflife(f64::INFINITY),
                window_rows: Some(12),
                ..expectile_cfg(0)
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Over a span of the clock, rows stamped, saved on a row of weight
        // 0: a stamp a row, the last counted row's and the last row's.
        Case {
            name: "expectile_clock_window",
            variant: "Expectile",
            build: || {
                Box::new(StampedExpectile {
                    model: Expectile::new(ExpectileCfg {
                        decay: Decay::Halflife(f64::INFINITY),
                        window_clock: Some(8.0),
                        closed: WindowClosed::Both,
                        ..expectile_cfg(0)
                    })
                    .unwrap(),
                    ns: 0,
                })
            },
            restore: restore_stamped_expectile,
            n_features: K,
            targets: Targets::Linear(2),
            before: 41,
        },
        // Saved within its start: two series' buffers.
        case!(
            "garch",
            "Garch",
            Garch,
            Garch::new(garch_cfg(GarchVol::Garch, GarchDist::Normal)).unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Saved after the start: the parameters, the information matrix,
        // the lags and their sensitivity, the running mean square.
        case!(
            "garch_started",
            "Garch",
            Garch,
            Garch::new(garch_cfg(GarchVol::Garch, GarchDist::Normal)).unwrap(),
            0,
            Targets::Linear(2),
            300
        ),
        // Variance targeting: the Huber score's centring and slope.
        case!(
            "garch_targeted",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 0,
                variance_targeting: true,
                ..garch_cfg(GarchVol::Garch, GarchDist::Normal)
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            300
        ),
        // A GARCH with a skewed t: the shape beside the process.
        case!(
            "garch_skewt",
            "Garch",
            Garch,
            Garch::new(garch_cfg(GarchVol::Garch, GarchDist::Skewt)).unwrap(),
            0,
            Targets::Linear(2),
            300
        ),
        // An EGARCH: its `z` and `ln s` lags, the variance bounds' mean,
        // and the running invertibility.
        case!(
            "garch_egarch",
            "Garch",
            Garch,
            Garch::new(garch_cfg(GarchVol::Egarch, GarchDist::Normal)).unwrap(),
            0,
            Targets::Linear(2),
            300
        ),
        // A FIGARCH of 20 lags: its ring of squared residuals.
        case!(
            "garch_figarch",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 0,
                truncation: Some(20),
                start_rows: 2000,
                // too small a cap to double its start: the fixture stays short
                max_state_bytes: Some(90_000),
                ..garch_cfg(GarchVol::Figarch, GarchDist::Normal)
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            2700
        ),
        // `realized` under a decay, with Lee and Mykland's own window of
        // ten returns, full (task 499): the sums, the last returns and the
        // ring of products.
        case!(
            "realized",
            "Realized",
            Realized,
            Realized::new(realized_cfg()).unwrap(),
            0,
            Targets::JoiningFrom(2, 20),
            40
        ),
        // A window of twelve rows, full and turning over: each target's
        // terms and how many of its products were taken, the rows' weights.
        case!(
            "realized_window",
            "Realized",
            Realized,
            Realized::new(RealizedCfg {
                decay: Decay::Halflife(f64::INFINITY),
                window_rows: Some(12),
                lm_window: None,
                ..realized_cfg()
            })
            .unwrap(),
            0,
            Targets::JoiningFrom(2, 20),
            40
        ),
        // A window of clock units over stamped rows, both edges' `closed`
        // the `both` one: a stamp a row and the clock summed.
        Case {
            name: "realized_clock_window",
            variant: "Realized",
            build: || {
                Box::new(StampedRealized {
                    model: Realized::new(RealizedCfg {
                        decay: Decay::Halflife(f64::INFINITY),
                        window_clock: Some(20.0),
                        closed: WindowClosed::Both,
                        ..realized_cfg()
                    })
                    .unwrap(),
                    ns: 0,
                })
            },
            restore: restore_stamped_realized,
            n_features: 0,
            targets: Targets::Linear(2),
            before: 45,
        },
        // A clock window whose rows passed its cap: warned past half of it,
        // and the row that would pass it recorded; nothing learned since.
        // Its rows are handed no stamps, so it reads the clock summed from
        // the steps.
        case!(
            "realized_clock_capped",
            "Realized",
            Realized,
            Realized::new(RealizedCfg {
                decay: Decay::Halflife(f64::INFINITY),
                window_clock: Some(20.0),
                lm_window: None,
                max_state_bytes: Some(1_500),
                ..realized_cfg()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            45
        ),
        // Principal components by the incremental SVD, three columns, two
        // emitted and three kept: saved with every direction started.
        case!(
            "pca",
            "StreamPca",
            StreamPca,
            StreamPca::new(pca_cfg(PcaMethod::Svd)).unwrap(),
            3,
            Targets::None,
            40
        ),
        // Saved on the third row: the directions not yet started, all
        // zeros beside the started one, and a cap of the spec's own.
        case!(
            "pca_early",
            "StreamPca",
            StreamPca,
            StreamPca::new(StreamPcaCfg {
                max_state_bytes: Some(4096),
                ..pca_cfg(PcaMethod::Svd)
            })
            .unwrap(),
            3,
            Targets::None,
            3
        ),
        // CCIPCA: each direction its variance times its unit vector.
        case!(
            "pca_ccipca",
            "StreamPca",
            StreamPca,
            StreamPca::new(pca_cfg(PcaMethod::Ccipca)).unwrap(),
            3,
            Targets::None,
            40
        ),
        // Oja's rule with a gain of its own.
        case!(
            "pca_oja",
            "StreamPca",
            StreamPca,
            StreamPca::new(StreamPcaCfg {
                learning_rate: Some(1.5),
                ..pca_cfg(PcaMethod::Oja)
            })
            .unwrap(),
            3,
            Targets::None,
            40
        ),
        // `power_variation` (task 555) under a decay, a term of each form:
        // the sums, the last raw return and both scales' rings.
        case!(
            "power_variation",
            "PowerVariation",
            PowerVariation,
            PowerVariation::new(power_variation_cfg()).unwrap(),
            0,
            Targets::JoiningFrom(2, 20),
            40
        ),
        // A window of twelve rows, full and turning over: each return's
        // credits, the rows' weights.
        case!(
            "power_variation_window",
            "PowerVariation",
            PowerVariation,
            PowerVariation::new(PowerVariationCfg {
                decay: Decay::Halflife(f64::INFINITY),
                window_rows: Some(12),
                ..power_variation_cfg()
            })
            .unwrap(),
            0,
            Targets::JoiningFrom(2, 20),
            40
        ),
        // A window of clock units over stamped rows, `closed` the `both`
        // one: a stamp a row and the clock summed.
        Case {
            name: "power_variation_clock_window",
            variant: "PowerVariation",
            build: || {
                Box::new(StampedPowerVariation {
                    model: PowerVariation::new(PowerVariationCfg {
                        decay: Decay::Halflife(f64::INFINITY),
                        window_clock: Some(20.0),
                        closed: WindowClosed::Both,
                        ..power_variation_cfg()
                    })
                    .unwrap(),
                    ns: 0,
                })
            },
            restore: restore_stamped_power_variation,
            n_features: 0,
            targets: Targets::Linear(2),
            before: 45,
        },
        // A clock window whose rows passed its cap (552 bytes fixed, 368 a
        // row, so six rows under 3,000): warned past half of it, the row
        // that would pass it recorded, nothing learned since.
        case!(
            "power_variation_clock_capped",
            "PowerVariation",
            PowerVariation,
            PowerVariation::new(PowerVariationCfg {
                decay: Decay::Halflife(f64::INFINITY),
                window_clock: Some(20.0),
                max_state_bytes: Some(3_000),
                ..power_variation_cfg()
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            45
        ),
        // A refit over a span of the clock (task 538), saved within the
        // span: each target's first stamp, on the clock summed from the
        // steps, which is state.
        case!(
            "glm_refit_clock",
            "Glm",
            Glm,
            Glm::new(GlmCfg {
                refit_rows: 0,
                refit_clock: Some(60.0),
                max_state_bytes: Some(200_000),
                ..glm_cfg(0)
            })
            .unwrap(),
            K,
            Targets::Positive(2),
            40
        ),
        // The same in `expectile`, each row stamped in nanoseconds.
        Case {
            name: "expectile_refit_clock",
            variant: "Expectile",
            build: || {
                Box::new(StampedExpectile {
                    model: Expectile::new(ExpectileCfg {
                        refit_clock: Some(60.0),
                        ..expectile_cfg(0)
                    })
                    .unwrap(),
                    ns: 0,
                })
            },
            restore: restore_stamped_expectile,
            n_features: K,
            targets: Targets::Linear(2),
            before: 40,
        },
        // Realistic (PLAN 550): each form on a series of its process, its
        // 2,000-row start fitted (about row 2,480, the rows of weight 0 and
        // the nulls not counted) and stepped about 260 learned rows past
        // it: the parameters, `H`, the lags and their sensitivity, the
        // targeted Huber score, a skewed t's shape, EGARCH's bins and
        // FIGARCH's ring on a process the start identifies. The cases above
        // are degenerate, their start's rows a linear target with no
        // conditional variance, and kept as edge tests: those saved past
        // their start now wait.
        case!(
            "garch_realistic",
            "Garch",
            Garch,
            Garch::new(realistic(GarchVol::Garch, GarchDist::Normal)).unwrap(),
            0,
            Targets::Garch(2, Process::Garch(garch_series::Innovations::Normal)),
            2800
        ),
        case!(
            "garch_t_realistic",
            "Garch",
            Garch,
            Garch::new(realistic(GarchVol::Garch, GarchDist::T)).unwrap(),
            0,
            Targets::Garch(2, Process::Garch(garch_series::Innovations::T6)),
            2800
        ),
        case!(
            "garch_targeted_realistic",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 0,
                variance_targeting: true,
                ..realistic(GarchVol::Garch, GarchDist::Normal)
            })
            .unwrap(),
            0,
            Targets::Garch(2, Process::Garch(garch_series::Innovations::Normal)),
            2800
        ),
        case!(
            "garch_skewt_realistic",
            "Garch",
            Garch,
            Garch::new(realistic(GarchVol::Garch, GarchDist::Skewt)).unwrap(),
            0,
            Targets::Garch(2, Process::Garch(garch_series::Innovations::SkewT6(-0.2))),
            2800
        ),
        case!(
            "garch_egarch_realistic",
            "Garch",
            Garch,
            Garch::new(realistic(GarchVol::Egarch, GarchDist::Normal)).unwrap(),
            0,
            Targets::Garch(2, Process::Egarch),
            2800
        ),
        case!(
            "garch_figarch_realistic",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 0,
                truncation: Some(20),
                // a cap of its own on the forecasts' work: 22 a step to 10
                max_horizon_work: Some(220),
                ..realistic(GarchVol::Figarch, GarchDist::Normal)
            })
            .unwrap(),
            0,
            Targets::Garch(2, Process::Figarch),
            2800
        ),
        // A start withheld at its bound: a cap that holds 80 of each
        // target's rows, so after its tries at 50 and 100 rows the buffer is
        // a ring of the latest 80, its oldest row at `head`, the rows it
        // let go counted, and the rows of its last try and since it.
        case!(
            "garch_bound",
            "Garch",
            Garch,
            Garch::new({
                let c = garch_cfg(GarchVol::Garch, GarchDist::Normal);
                GarchCfg {
                    max_state_bytes: Some(Garch::state_bytes(&c, 80)),
                    ..c
                }
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            333
        ),
        // A start of a clock span, saved within it: the first row's stamp
        // (the targets join at the fifth row, so it is not the clock's 0).
        case!(
            "garch_span",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 0,
                start_span: Some(80.0),
                ..garch_cfg(GarchVol::Garch, GarchDist::T)
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // `dcc` (task 493) on a realistic ADCC stream, its parameters
        // estimated: saved within the start, its buffer of residuals.
        case!(
            "dcc",
            "Dcc",
            Dcc,
            Dcc::new(dcc_cfg(3, true, None)).unwrap(),
            0,
            Targets::Correlated(3, false, true),
            40
        ),
        // Saved after the start: the targets Q̄ and N̄, Q, its sensitivities,
        // the information and the estimate, under a decay; its start of 400
        // rows identified its recursion (the likelihood-ratio statistic,
        // kept).
        case!(
            "dcc_started",
            "Dcc",
            Dcc,
            Dcc::new(DccCfg {
                decay: Decay::Halflife(4000.0),
                start_rows: 400,
                ..dcc_cfg(3, true, None)
            })
            .unwrap(),
            0,
            Targets::Correlated(3, false, true),
            1500
        ),
        // A start withheld: two series of one correlation, no recursion to
        // find, waiting for twice its rows with the statistic it fell
        // short with.
        case!(
            "dcc_withheld",
            "Dcc",
            Dcc,
            Dcc::new(dcc_cfg(2, false, None)).unwrap(),
            0,
            Targets::Correlated(2, false, false),
            120
        ),
        // A start withheld at its bound: a cap that holds 80 of its rows,
        // so after its tries at 50 and 80 rows the buffer is a ring of the
        // latest 80, its oldest row at `head`, the rows it let go counted.
        case!(
            "dcc_bound",
            "Dcc",
            Dcc,
            Dcc::new({
                let mut c = dcc_cfg(2, false, None);
                c.max_state_bytes = Some(c.stage_bytes() + 32 + 80 * c.row_bytes());
                c
            })
            .unwrap(),
            0,
            Targets::Correlated(2, false, false),
            333
        ),
        // Given parameters, the symmetric model: no sensitivities.
        case!(
            "dcc_given",
            "Dcc",
            Dcc,
            Dcc::new(dcc_cfg(2, false, Some(vec![0.05, 0.9]))).unwrap(),
            0,
            Targets::Correlated(2, false, true),
            120
        ),
        // GARCH margins on returns: the margins started, the stage's own
        // start buffering their residuals.
        case!(
            "dcc_margins",
            "Dcc",
            Dcc,
            Dcc::new(DccCfg {
                margins: Some(GarchCfg {
                    horizons: Vec::new(),
                    start_rows: 400,
                    ..garch_cfg(GarchVol::Garch, GarchDist::Normal)
                }),
                start_rows: 2000,
                ..dcc_cfg(2, false, None)
            })
            .unwrap(),
            0,
            Targets::Correlated(2, true, true),
            // the margins' start waits for twice its rows while its fit
            // does not identify a variance process, so it may take 800 rows
            // of each series; the correlation's long start is still
            // buffering
            CASE_DCC_MARGINS_BEFORE
        ),
        // A start over a span of the clock, saved within it: its first
        // stamp and the summed clock.
        case!(
            "dcc_span",
            "Dcc",
            Dcc,
            Dcc::new(DccCfg {
                start_span: Some(80.0),
                ..dcc_cfg(2, true, None)
            })
            .unwrap(),
            0,
            Targets::Correlated(2, false, true),
            40
        ),
        // A t-digest a target (PLAN 544), saved before its first merge:
        // each digest's buffer, its extremes and the weights, no centroid.
        case!(
            "tdigest",
            "TDigest",
            TDigestModel,
            TDigestModel::new(tdigest_cfg()).unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Saved past its merges, its buffer part-filled again: the
        // centroids beside the buffer.
        case!(
            "tdigest_merged",
            "TDigest",
            TDigestModel,
            TDigestModel::new(tdigest_cfg()).unwrap(),
            0,
            Targets::Linear(2),
            300
        ),
        // A DDSketch a target (PLAN 544, part 7), never forgetting: both
        // stores, the zero count stays empty on a linear target.
        case!(
            "ddsketch",
            "DdSketch",
            DdSketchModel,
            DdSketchModel::new(ddsketch_cfg(None, None)).unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // A row window, full and turning over: the ring's codes and
        // weights, and the stores' row counts.
        case!(
            "ddsketch_rows",
            "DdSketch",
            DdSketchModel,
            DdSketchModel::new(ddsketch_cfg(Some(25), None)).unwrap(),
            0,
            Targets::Linear(2),
            60
        ),
        // The decayed form: the decay in the cfg, the weights decayed.
        case!(
            "ddsketch_decayed",
            "DdSketch",
            DdSketchModel,
            DdSketchModel::new(DdSketchModelCfg {
                decay: Decay::Halflife(15.0),
                ..ddsketch_cfg(None, None)
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            60
        ),
        // A clock window: the ring's stamps and the clock kept for them.
        case!(
            "ddsketch_clock",
            "DdSketch",
            DdSketchModel,
            DdSketchModel::new(ddsketch_cfg(None, Some(12.0))).unwrap(),
            0,
            Targets::Linear(2),
            60
        ),
        // Every kind of term to order 6, and the unbiased forms to order 4:
        // the central moments of orders 2 to 6 and all three of `r_2`, `r_3`
        // and `r_4`.
        case!(
            "ewm_moments",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_cfg(true)).unwrap(),
            1,
            Targets::None,
            40
        ),
        // The plug-in terms alone: Kish's `r_2` and no other.
        case!(
            "ewm_moments_plugin",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_cfg(false)).unwrap(),
            1,
            Targets::None,
            40
        ),
        // Row lags of two columns, an autocorrelation and lagged cross
        // terms: a summary a set of legs, and one buffer a lagged column.
        case!(
            "ewm_moments_lags",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_lag_cfg(false, None, None)).unwrap(),
            2,
            Targets::None,
            40
        ),
        // The same lags on the clock, as-of within a tolerance: each
        // buffer's stamps, and the clock's place.
        case!(
            "ewm_moments_clock_lags",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_lag_cfg(true, None, None)).unwrap(),
            2,
            Targets::None,
            40
        ),
        // A window of 12 rows: the ring of rows, no summary.
        case!(
            "ewm_moments_window",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_lag_cfg(false, Some(12), None)).unwrap(),
            2,
            Targets::None,
            40
        ),
        // A window of 7 clock units: the ring's stamps.
        case!(
            "ewm_moments_clock_window",
            "EwmMoments",
            EwmMoments,
            EwmMoments::new(ewm_moments_lag_cfg(true, None, Some(7.0))).unwrap(),
            2,
            Targets::None,
            40
        ),
        // An AR of lags 1 and 3 over two targets with the features as
        // exogenous regressors: each target's full ring of lags and its fit.
        case!(
            "ar",
            "Ar",
            Ar,
            Ar::new(ar_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Its order chosen by BIC after every learned row: the chosen
        // coefficients beside the fit at the largest order.
        case!(
            "ar_select",
            "Ar",
            Ar,
            Ar::new(ArCfg {
                lags: vec![1, 2, 3],
                ridge: 0.0,
                select: Some(ArSelect::Bic),
                ..ar_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Under a hard window of 15 clock units: each fit's snapshots.
        case!(
            "ar_window",
            "Ar",
            Ar,
            Ar::new(ArCfg {
                window: Some(15.0),
                ..ar_cfg()
            })
            .unwrap(),
            K,
            Targets::Linear(2),
            40
        ),
        // Two series' transforms at three frequencies.
        case!(
            "spectral",
            "Spectral",
            Spectral,
            Spectral::new(SpectralCfg {
                n_targets: 2,
                frequencies: vec![0.0, 0.2, 0.5],
                spacing: 1.0,
                decay: decay(),
                scaling: SpectralScaling::Density,
                detrend: true,
                min_weight: 0.0,
                max_state_bytes: None,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Two series' Lomb-Scargle sums on an uneven clock: the weights'
        // transform at twice each frequency and the squares beside the
        // transform, one frequency past the even clock's Nyquist.
        case!(
            "spectral_ls",
            "Spectral",
            Spectral,
            Spectral::new(SpectralCfg {
                n_targets: 2,
                frequencies: vec![0.2, 0.7],
                spacing: 1.0,
                decay: decay(),
                scaling: SpectralScaling::LombScargle,
                detrend: true,
                min_weight: 0.0,
                max_state_bytes: None,
                window: None,
                window_every: None,
                max_rows_between_snapshots: None,
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // Two series' transforms over a window of 15 clock units: its
        // blocks, each at its own origin, and the closed blocks' total.
        case!(
            "spectral_window",
            "Spectral",
            Spectral,
            Spectral::new(SpectralCfg {
                n_targets: 2,
                frequencies: vec![0.2, 0.5],
                spacing: 1.0,
                decay: decay(),
                scaling: SpectralScaling::LombScargle,
                detrend: true,
                min_weight: 0.0,
                max_state_bytes: None,
                window: Some(15.0),
                window_every: None,
                max_rows_between_snapshots: Some(3),
            })
            .unwrap(),
            0,
            Targets::Linear(2),
            40
        ),
        // The pooled form, decreasing and clipped: two targets' distinct
        // `x`, each with its weight and sum.
        case!(
            "isotonic",
            "Isotonic",
            Isotonic,
            Isotonic::new(IsotonicCfg {
                increasing: false,
                y_min: Some(-1.0),
                ..isotonic_cfg()
            })
            .unwrap(),
            1,
            Targets::Linear(2),
            40
        ),
        // Under a decay, past an origin's move: the clock, its origin and
        // the scaled weights.
        case!(
            "isotonic_decayed",
            "Isotonic",
            Isotonic,
            Isotonic::new(IsotonicCfg {
                decay: Decay::Halflife(0.4),
                ..isotonic_cfg()
            })
            .unwrap(),
            1,
            Targets::Linear(2),
            60
        ),
        // A row window, full and turning over: the ring and each target's
        // rows sorted by `(x, y)`.
        case!(
            "isotonic_rows",
            "Isotonic",
            Isotonic,
            Isotonic::new(IsotonicCfg {
                window_rows: Some(25),
                ..isotonic_cfg()
            })
            .unwrap(),
            1,
            Targets::Linear(2),
            60
        ),
        // A clock window under a decay: the ring's stamps.
        case!(
            "isotonic_clock",
            "Isotonic",
            Isotonic,
            Isotonic::new(IsotonicCfg {
                window_clock: Some(12.0),
                decay: Decay::Halflife(20.0),
                out_of_bounds: IsotonicOutOfBounds::Clip,
                ..isotonic_cfg()
            })
            .unwrap(),
            1,
            Targets::Linear(2),
            60
        ),
        // Saved two rows in: rings still filling, no fit yet.
        case!(
            "ar_filling",
            "Ar",
            Ar,
            Ar::new(ar_cfg()).unwrap(),
            K,
            Targets::Linear(2),
            2
        ),
        // GARCH-X (task 559, part 4): EGARCH's with two inputs, its start
        // fitted and stepped (the inputs' `π`, their running mean `x̄`, `H`
        // and the sensitivity over them); and GJR's saved within its start,
        // holding the start's inputs beside its rows, the rows whose input
        // falls below 0 learned as rows of weight 0.
        case!(
            "garch_x_realistic",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 2,
                ..realistic(GarchVol::Egarch, GarchDist::Normal)
            })
            .unwrap(),
            2,
            Targets::Garch(2, Process::Egarch),
            2800
        ),
        case!(
            "garch_x_start",
            "Garch",
            Garch,
            Garch::new(GarchCfg {
                n_inputs: 2,
                ..realistic(GarchVol::Gjr, GarchDist::Normal)
            })
            .unwrap(),
            2,
            Targets::Garch(2, Process::Garch(garch_series::Innovations::Normal)),
            300
        ),
    ]
}

/// Rows before `dcc_margins`' save: past its margins' start, within the
/// stage's.
const CASE_DCC_MARGINS_BEFORE: usize = 1500;

/// `dcc` of the case's series, the start 50 rows, no margins.
fn dcc_cfg(k: usize, asymmetric: bool, params: Option<Vec<f64>>) -> DccCfg {
    DccCfg {
        n_targets: k,
        asymmetric,
        margins: None,
        params,
        decay: Decay::Halflife(f64::INFINITY),
        start_rows: 50,
        start_span: None,
        min_weight: 0.0,
        max_state_bytes: None,
    }
}

/// A `ddsketch` of two targets at a coarse accuracy, read at three levels,
/// plain or under a window.
/// `isotonic` of two targets on the one feature, scikit-learn's defaults.
fn isotonic_cfg() -> IsotonicCfg {
    IsotonicCfg {
        n_targets: 2,
        increasing: true,
        y_min: None,
        y_max: None,
        out_of_bounds: IsotonicOutOfBounds::Nan,
        min_weight: 0.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    }
}

fn ddsketch_cfg(window_rows: Option<usize>, window_clock: Option<f64>) -> DdSketchModelCfg {
    DdSketchModelCfg {
        n_targets: 2,
        alpha: 0.02,
        levels: vec![0.1, 0.4, 0.9],
        window_rows,
        window_clock,
        closed: WindowClosed::Right,
        decay: Decay::Halflife(f64::INFINITY),
        max_state_bytes: None,
    }
}

/// A `tdigest` of two targets at the smallest compression, whose buffer
/// merges every 80 rows, read at three levels.
fn tdigest_cfg() -> TDigestModelCfg {
    TDigestModelCfg {
        n_targets: 2,
        compression: 10.0,
        levels: vec![0.1, 0.4, 0.9],
        max_state_bytes: None,
    }
}

/// `ewm_moments` of the first two features with lags: the first's
/// autocorrelation at 2, a cross term of the first with the second a step
/// back, the second's lagged skew and the first's unbiased fourth
/// cumulant, each standard error; on the clock with `clocked`, under a
/// window of `rows` or of `clock`.
fn ewm_moments_lag_cfg(clocked: bool, rows: Option<usize>, clock: Option<f64>) -> EwmMomentsCfg {
    let term = |name: &str, powers: Vec<(MomentLeg, u32)>, kind, bias| MomentTerm {
        name: name.to_owned(),
        powers,
        kind,
        bias,
    };
    let terms = vec![
        term(
            "acor2",
            vec![(MomentLeg::of(0), 1), (MomentLeg::lagged(0, 2.0), 1)],
            MomentKind::Standardized,
            true,
        ),
        term(
            "x2y1",
            vec![(MomentLeg::of(0), 2), (MomentLeg::lagged(1, 1.0), 1)],
            MomentKind::Central,
            true,
        ),
        term(
            "sk_y1",
            vec![(MomentLeg::lagged(1, 1.0), 3)],
            MomentKind::Standardized,
            true,
        ),
        term(
            "k4_u",
            vec![(MomentLeg::of(0), 4)],
            MomentKind::Cumulant,
            false,
        ),
    ];
    EwmMomentsCfg {
        min_weight: 2.0,
        clocked,
        lag_tolerance: clocked.then_some(1.0),
        se: true,
        window_rows: rows,
        window_clock: clock,
        ..EwmMomentsCfg::new(decay(), terms)
    }
}

/// `ewm_moments` of the first feature: plug-in central, standardized,
/// cumulant and raw terms to order 6, and with `unbiased` the unbiased
/// central moments, cumulants and skew and kurtosis to order 4.
fn ewm_moments_cfg(unbiased: bool) -> EwmMomentsCfg {
    let term = MomentTerm::of_column;
    let mut terms = vec![
        term("var", 2, MomentKind::Central, true),
        term("skew", 3, MomentKind::Standardized, true),
        term("kurt", 4, MomentKind::Standardized, true),
        term("m6", 6, MomentKind::Central, true),
        term("k5", 5, MomentKind::Cumulant, true),
        term("raw3", 3, MomentKind::Raw, true),
    ];
    if unbiased {
        terms.extend([
            term("var_u", 2, MomentKind::Central, false),
            term("skew_u", 3, MomentKind::Standardized, false),
            term("kurt_u", 4, MomentKind::Standardized, false),
            term("k4_u", 4, MomentKind::Cumulant, false),
        ]);
    }
    EwmMomentsCfg {
        min_weight: 2.0,
        ..EwmMomentsCfg::new(decay(), terms)
    }
}

/// An AR of lags 1 and 3 under decay, its two targets' fits with a small
/// ridge.
fn ar_cfg() -> ArCfg {
    ArCfg {
        n_targets: 2,
        n_features: K,
        lags: vec![1, 3],
        fit_intercept: true,
        decay: decay(),
        ridge: 0.01,
        min_weight: 2.0,
        max_state_bytes: None,
        select: None,
        window: None,
        window_every: None,
        max_rows_between_snapshots: None,
    }
}

/// An upper `expectile` of two targets under decay, its first rows' refit
/// filling on its `refit_rows`-th learned row (0: off).
fn expectile_cfg(refit_rows: u32) -> ExpectileCfg {
    ExpectileCfg {
        n_features: K,
        n_targets: 2,
        fit_intercept: true,
        decay: decay(),
        expectile: 0.8,
        ridge: 1e-6,
        min_weight: 2.0,
        refit_rows,
        refit_clock: None,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

fn garch_cfg(vol: GarchVol, dist: GarchDist) -> GarchCfg {
    GarchCfg {
        n_inputs: 0,
        n_targets: 2,
        vol,
        dist,
        truncation: None,
        variance_targeting: false,
        start_rows: 50,
        start_span: None,
        // EGARCH's states of schema 71 have no horizon; the others one,
        // at 3 rows (task 559 read them as these horizons)
        horizons: if vol == GarchVol::Egarch {
            Vec::new()
        } else {
            vec![3]
        },
        max_horizon_work: None,
        min_weight: 0.0,
        max_state_bytes: None,
    }
}

/// [`garch_cfg`] at the default start of 2,000 rows, for a realistic case,
/// its forecasts at three horizons out of order (task 559).
fn realistic(vol: GarchVol, dist: GarchDist) -> GarchCfg {
    GarchCfg {
        n_inputs: 0,
        start_rows: 2000,
        horizons: vec![10, 1, 3],
        ..garch_cfg(vol, dist)
    }
}

/// Two return series, a term of each form: a sum of squares, a normalized
/// bipower term, a centered Parzen sum and a flat-top Bartlett sum of
/// two-step returns.
fn power_variation_cfg() -> PowerVariationCfg {
    let mp = |powers: &[f64], normalize: bool| PvTerm {
        kind: PvTermKind::Multipower {
            powers: powers.to_vec(),
            normalize,
        },
        scale: 1,
    };
    let ker = |kernel, bandwidth, flat_top, center, scale| PvTerm {
        kind: PvTermKind::Kernel {
            kernel,
            bandwidth,
            flat_top,
            center,
        },
        scale,
    };
    PowerVariationCfg {
        n_targets: 2,
        terms: vec![
            mp(&[2.0], false),
            mp(&[1.0, 1.0], true),
            ker(PvKernel::Parzen, 3, false, true, 1),
            ker(PvKernel::Bartlett, 2, true, false, 2),
        ],
        decay: decay(),
        min_weight: 2.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

fn realized_cfg() -> RealizedCfg {
    RealizedCfg {
        n_targets: 2,
        decay: decay(),
        min_weight: 2.0,
        window_rows: None,
        window_clock: None,
        closed: WindowClosed::Right,
        lm_window: Some(10),
        lm_beta: 0.95,
        max_state_bytes: None,
    }
}

fn hawkes_cfg(window_events: Option<usize>) -> HawkesCfg {
    HawkesCfg {
        n_targets: 2,
        kernel_decays: vec![0.8, 3.0],
        decay: if window_events.is_some() {
            Decay::Halflife(f64::INFINITY)
        } else {
            decay()
        },
        min_weight: 2.0,
        window_events,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// The case's model after its first part, and its rows.
pub fn saved(case: &Case) -> (Box<dyn Run>, Vec<Row>) {
    let rows = case.rows();
    let mut m = (case.build)();
    for r in &rows[..case.before] {
        m.step(r);
    }
    (m, rows)
}

/// A state as the bank writes one: named msgpack.
pub fn named(s: &State) -> Vec<u8> {
    rmp_serde::to_vec_named(s).expect("a state encodes")
}

/// `bytes` decoded and restored as `case`'s model, or why not.
pub fn load(case: &Case, bytes: &[u8]) -> Result<Box<dyn Run>, String> {
    let s: State = rmp_serde::from_slice(bytes).map_err(|e| format!("decoding: {e}"))?;
    (case.restore)(&s)
}

/// A frozen fixture, as `tests/state_fixtures/<name>.rs` holds it (written
/// by `PRINT_STATE_FIXTURES=1`): the case's state after its first part, as
/// named msgpack in hex, the schema and the platform it was written at, and
/// its continuation -- every input of each row and every number it reported
/// -- as the bits of each double, so nothing is read back through a decimal.
pub struct Fixture {
    pub name: &'static str,
    pub variant: &'static str,
    pub schema: u32,
    /// `<os>-<arch>` of the build that wrote it ([`writer`]).
    pub writer: &'static str,
    pub state: &'static str,
    /// Per row, `n_features` doubles.
    pub x: &'static [u64],
    /// Per row, one entry a target.
    pub y: &'static [Option<u64>],
    pub d: &'static [u64],
    pub w: &'static [u64],
    /// Per row, `n_pred` doubles.
    pub pred: &'static [u64],
    pub n_eff: &'static [u64],
    /// Per row, `n_extra` doubles.
    pub extra: &'static [u64],
    pub n_features: usize,
    pub n_targets: usize,
    pub n_pred: usize,
    pub n_extra: usize,
}

impl Fixture {
    pub fn bytes(&self) -> Vec<u8> {
        unhex(self.state)
    }

    /// The continuation's rows, as the fixture holds them.
    pub fn rows(&self) -> Vec<Row> {
        let f = f64::from_bits;
        (0..self.d.len())
            .map(|i| Row {
                x: self.x[i * self.n_features..(i + 1) * self.n_features]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
                y: self.y[i * self.n_targets..(i + 1) * self.n_targets]
                    .iter()
                    .map(|b| b.map(f))
                    .collect(),
                d: f(self.d[i]),
                w: f(self.w[i]),
            })
            .collect()
    }

    /// What each continuation row reported when the fixture was written.
    pub fn outs(&self) -> Vec<Out> {
        let f = f64::from_bits;
        (0..self.d.len())
            .map(|i| Out {
                pred: self.pred[i * self.n_pred..(i + 1) * self.n_pred]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
                n_eff: f(self.n_eff[i]),
                extra: self.extra[i * self.n_extra..(i + 1) * self.n_extra]
                    .iter()
                    .map(|&b| f(b))
                    .collect(),
            })
            .collect()
    }
}

/// The platform a build runs on, as a fixture records its writer.
pub fn writer() -> String {
    format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH)
}

pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn unhex(s: &str) -> Vec<u8> {
    let s: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace()).collect();
    s.chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

/// Whether two numbers a row reported agree: both NaN -- whatever their
/// bits, which an x86 and an ARM build make differently -- or the same
/// bits, and otherwise, at `tol > 0`, within `tol` relative to
/// `1 + |want|`. An infinity agrees only with itself: two equal
/// infinities differ by NaN, which no tolerance admits, and against an
/// infinite `want` the relative bound is infinite and would admit any
/// number.
pub fn agree(got: f64, want: f64, tol: f64) -> bool {
    if got.is_nan() || want.is_nan() {
        return got.is_nan() && want.is_nan();
    }
    if got.to_bits() == want.to_bits() {
        return true;
    }
    tol > 0.0
        && got.is_finite()
        && want.is_finite()
        && (got - want).abs() <= tol * (1.0 + want.abs())
}

fn pca_cfg(method: PcaMethod) -> StreamPcaCfg {
    StreamPcaCfg {
        n_features: 3,
        n_components: 2,
        method,
        decay: decay(),
        min_weight: 0.0,
        learning_rate: None,
        max_state_bytes: None,
    }
}
