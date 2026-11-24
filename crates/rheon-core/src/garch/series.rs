//! One target's series: its start's buffer, then its parameters, their
//! information matrix and the filter's lags, stepped a row at a time.

use serde::{Deserialize, Serialize};

use super::dist;
use super::filter::Filter;
use super::fit::OMEGA_FLOOR_FRAC;
use super::fit::{ab_set, backcast, clip_shape, fit, identified, set_of};
use super::guard;
use super::linalg::{HalfSpaces, backward, chol_solve, cholesky, forward, tri, unpack};
use super::math;
use super::{GarchCfg, GarchDist, GarchVol};
use crate::Stamp;

/// The Huber cut of the variance-targeted score, in standard deviations:
/// `z²` is clipped at `c² = 4`.
const HUBER_C2: f64 = 4.0;
/// The ridge on the step's information matrix, relative to each diagonal
/// entry: it scales with each parameter's units.
const STEP_RIDGE: f64 = 1e-12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(super) struct Series {
    /// The start's rows, `[e, weight]` in order, until it is fitted. At its
    /// bound, a ring whose oldest row is at `head`.
    pub buffer: Vec<[f64; 2]>,
    /// The ring's oldest row once the buffer is at its bound; 0 before, and
    /// after each try, which puts the rows back in order.
    #[serde(default)]
    pub head: usize,
    /// The stamp of the first row buffered, for a start of a clock span.
    pub first: Option<Stamp>,
    /// Whether the start has been fitted.
    pub started: bool,
    /// The tries of the start that did not start: a fit its rows did not
    /// identify, one short of its optimum, or none with a finite
    /// likelihood.
    pub refits: u32,
    /// The rows the buffer held at the last try; 0 before one.
    #[serde(default)]
    pub tried: u64,
    /// The rows buffered since the last try; 0 before one.
    #[serde(default)]
    pub since: u64,
    /// The rows the buffer let go at its bound, oldest first.
    #[serde(default)]
    pub dropped: u64,
    /// Whether the fit the series stepped from was at its optimum: always,
    /// once started, since a fit short of it is never stepped from.
    pub sound: bool,
    /// The weight this target has learned, the start's rows the buffer
    /// holds included.
    pub w_sum: f64,
    /// The process's parameters, then the shape (`ν`, then `λ`). Before
    /// the start, the last fit withheld because its rows did not tell its
    /// process from a constant variance ([`Series::no_effect`]), or not a
    /// number while no fit has been.
    pub theta: Vec<f64>,
    /// The information matrix of the stepped coordinates, packed.
    pub h: Vec<f64>,
    pub filter: Filter,
    /// The weighted running mean of `e²`.
    pub mean_e2: f64,
    /// EGARCH's standardized residuals, weighted counts in the guard's bins
    /// ([`guard::BINS`]); empty for the other processes.
    pub zbins: Vec<f64>,
    /// The Huber score's running centring `E[min(z², c²)]` and slope
    /// `E[z² 1[z² < c²]]`, under variance targeting.
    pub kappa: f64,
    pub slope: f64,
}

impl Series {
    pub fn new(cfg: &GarchCfg) -> Self {
        let nf = cfg.n_free();
        Self {
            buffer: Vec::new(),
            head: 0,
            first: None,
            started: false,
            refits: 0,
            tried: 0,
            since: 0,
            dropped: 0,
            sound: false,
            w_sum: 0.0,
            theta: vec![f64::NAN; cfg.n_params()],
            h: vec![0.0; nf * (nf + 1) / 2],
            filter: Filter::empty(),
            mean_e2: 0.0,
            zbins: if cfg.vol == GarchVol::Egarch {
                vec![0.0; guard::BINS]
            } else {
                Vec::new()
            },
            kappa: 0.0,
            slope: 0.0,
        }
    }

    /// Whether the state has the shape `cfg` gives it.
    pub fn has_shape(&self, cfg: &GarchCfg) -> bool {
        let nf = cfg.n_free();
        let k = cfg.vol.k();
        let lags = if !self.started {
            self.filter.sens.is_empty() && self.filter.ring.is_empty()
        } else if cfg.vol == GarchVol::Figarch {
            self.filter.ring.len() == cfg.truncation() && self.filter.head < cfg.truncation()
        } else {
            self.filter.sens.len() == k && self.filter.ring.is_empty()
        };
        let bins = if cfg.vol == GarchVol::Egarch {
            guard::BINS
        } else {
            0
        };
        self.theta.len() == cfg.n_params()
            && self.h.len() == nf * (nf + 1) / 2
            && self.zbins.len() == bins
            && lags
            && (!self.started || (self.buffer.is_empty() && self.tried == 0 && self.head == 0))
            && (self.head == 0 || self.head < self.buffer.len())
            && if self.tried == 0 {
                // before a try; or a state of schema 71 that tried and kept
                // no count of it, under its rule of a buffer under four
                // times the start's rows (its next row tries again)
                let most = if self.refits > 0 {
                    cfg.start_rows << 2
                } else {
                    cfg.start_rows
                };
                self.since == 0 && (cfg.start_span.is_some() || self.buffer.len() < most)
            } else {
                self.since < self.tried
                    && (cfg.start_span.is_some() || self.buffer.len() as u64 <= bound(cfg))
            }
    }

    /// Whether the state's values are in their ranges, beside the shape
    /// [`Series::has_shape`] reads: every weight and running mean a number,
    /// the weights at 0 or above and each buffered row's above 0; and once
    /// started, the parameters in the process's set (to `10⁻⁹` of each
    /// bound's size: a projection meets them to `10⁻¹²`), a t's `ν` in
    /// `[2.05, 300]` and a skewed t's `|λ| ≤ 0.99`.
    pub fn has_values(&self, cfg: &GarchCfg) -> bool {
        let finite = |v: &[f64]| v.iter().all(|x| x.is_finite());
        let k = cfg.vol.k();
        let means = self.w_sum >= 0.0
            && self.w_sum.is_finite()
            && self.mean_e2 >= 0.0
            && self.mean_e2.is_finite()
            && self.kappa.is_finite()
            && self.slope.is_finite()
            && finite(&self.h)
            && self.zbins.iter().all(|c| *c >= 0.0 && c.is_finite())
            && self
                .buffer
                .iter()
                .all(|r| r[0].is_finite() && r[1] > 0.0 && r[1].is_finite())
            && finite(&self.filter.lag)
            && finite(&self.filter.sens)
            && self.filter.ring.iter().all(|x| *x >= 0.0 && x.is_finite());
        if !means {
            return false;
        }
        if !self.started {
            return true;
        }
        let th = &self.theta;
        let shape = match cfg.dist {
            GarchDist::Normal => true,
            _ => {
                (dist::NU_LO..=dist::NU_HI).contains(&th[k])
                    && th.get(k + 1).is_none_or(|l| l.abs() <= dist::LAM_LIM)
            }
        };
        finite(th) && shape && set_of(cfg.vol, 0.0).holds(&th[..k], 1e-9)
    }

    /// The variance of the row about to be read, and its sensitivity into
    /// `dl`; not a number before the start.
    pub fn read(&self, cfg: &GarchCfg, dl: &mut [f64]) -> f64 {
        if !self.started {
            return f64::NAN;
        }
        self.filter.predict(cfg.vol, &self.theta[..cfg.vol.k()], dl)
    }

    /// The variance of the row about to be read, without the sensitivity.
    pub fn variance(&self, cfg: &GarchCfg) -> f64 {
        if !self.started {
            return f64::NAN;
        }
        let k = cfg.vol.k();
        self.filter.variance(cfg.vol, &self.theta[..k], k)
    }

    /// Whether the series waits because its last start's fit did not tell
    /// its process from a constant variance: not started, and holding that
    /// withheld fit.
    pub fn no_effect(&self) -> bool {
        !self.started && self.theta.iter().all(|v| v.is_finite())
    }

    /// Fit the buffered rows. A fit whose likelihood-ratio statistic
    /// against constant variance is under the form's cut
    /// ([`super::fit::identified`]) is withheld, and kept in `theta` to say
    /// why the series waits; a fit not at its optimum by its Newton
    /// decrement inside the process's set, or none with a finite
    /// likelihood, is not stepped from either, and says nothing. Either
    /// keeps its buffer, and the next try comes once as many new rows have
    /// come as this one held ([`Series::buffer_row`]). Otherwise the buffer
    /// is replayed at the fit: the filter from the backcast, the
    /// information matrix and the running means built as equal-weight
    /// means over it, the parameters held.
    pub fn start(&mut self, cfg: &GarchCfg) {
        // a ring's rows back in order, oldest first
        self.buffer.rotate_left(self.head);
        self.head = 0;
        let bc = backcast(&self.buffer);
        let wsum: f64 = self.buffer.iter().map(|r| r[1]).sum();
        let v = self.buffer.iter().map(|r| r[1] * r[0] * r[0]).sum::<f64>() / wsum;
        // EGARCH's first `ln s` lag is the backcast's logarithm, so a
        // backcast of 0 (the first 75 rows all 0) is read as the rows' mean
        // square, which no `ln` takes to `−∞` (review 2026-11-23, GARCH-5).
        let bc = if cfg.vol == GarchVol::Egarch && (bc.is_nan() || bc <= 0.0) {
            v
        } else {
            bc
        };
        let got = fit(
            cfg.vol,
            cfg.dist,
            cfg.variance_targeting,
            cfg.truncation(),
            &self.buffer,
            bc,
            v,
        )
        .filter(|f| f.theta.iter().all(|v| v.is_finite()));
        // a try that does not start: the buffer kept, the reason this
        // try's alone
        let wait = |series: &mut Self, withheld: Option<Vec<f64>>| {
            series.theta = withheld.unwrap_or_else(|| vec![f64::NAN; cfg.n_params()]);
            series.refits = series.refits.saturating_add(1);
            series.tried = series.buffer.len() as u64;
            series.since = 0;
        };
        let got = match got {
            // Rows all 0: a constant variance of 0 says all they tell, and
            // no process, so the wait is `not_identified`, the withheld
            // "fit" that constant (review 2026-11-23, GARCH-11).
            None if v == 0.0 => return wait(self, Some(vec![0.0; cfg.n_params()])),
            // No fit with a finite likelihood.
            None => return wait(self, None),
            // No evidence of a variance process: withheld.
            Some(f) if !identified(cfg.vol, f.lr) => return wait(self, Some(f.theta)),
            // Not at its optimum: never stepped from.
            Some(f) if !f.sound => return wait(self, None),
            Some(f) => f,
        };
        let rows = std::mem::take(&mut self.buffer);
        let k = cfg.vol.k();
        self.theta = got.theta;
        if !cfg.variance_targeting {
            exact(cfg.vol, &mut self.theta[..k], OMEGA_FLOOR_FRAC * v);
        }
        self.sound = true;
        self.started = true;
        self.tried = 0;
        self.since = 0;
        self.w_sum = 0.0;
        self.mean_e2 = 0.0;
        self.zbins.fill(0.0);
        self.kappa = 0.0;
        self.slope = 0.0;
        self.h.fill(0.0);
        self.filter = Filter::new(cfg.vol, k, cfg.truncation(), bc);
        let mut dl = [0.0; 4];
        for r in &rows {
            let s = self.read(cfg, &mut dl[..k]);
            self.learn(cfg, r[0], r[1], s, &dl[..k], true);
        }
        self.first = None;
    }

    /// Buffer a row of residual `e` at weight `w > 0` for the start, `now`
    /// its stamp under a start of a span, and say whether a try is due. The
    /// first try comes at the start's rows or span; each later one once as
    /// many rows have come as the try before held. After a try the buffer
    /// grows while `grows` (the cap holds a row more, [`bound`] and the
    /// state's bytes), and from then on keeps the latest rows: the oldest
    /// leaves as each new one comes, its weight with it, and the wait goes
    /// on by name. Nothing is cleared, which could not cure a shortage of
    /// evidence, and no count of tries ends the wait.
    pub fn buffer_row(
        &mut self,
        cfg: &GarchCfg,
        e: f64,
        w: f64,
        now: Option<Stamp>,
        grows: bool,
    ) -> bool {
        if self.buffer.is_empty() {
            self.first = now;
        }
        if self.tried > 0 && !grows {
            let old = std::mem::replace(&mut self.buffer[self.head], [e, w]);
            self.head = (self.head + 1) % self.buffer.len();
            self.dropped += 1;
            self.w_sum += w - old[1];
        } else {
            self.buffer.push([e, w]);
            self.w_sum += w;
        }
        if self.tried > 0 {
            self.since += 1;
            return self.since >= self.tried;
        }
        let rows = self.buffer.len();
        match (cfg.start_span, now, self.first) {
            (Some(span), Some(now), Some(first)) => {
                rows >= super::MIN_START_ROWS && crate::clock_span::span_reached(now, first, span)
            }
            _ => rows >= cfg.start_rows,
        }
    }

    /// Learn a row of residual `e` at weight `w > 0`, whose variance was `s`
    /// and sensitivity `dl`: the running means, the information matrix and,
    /// unless `replay`, the step; then the filter's lags.
    pub fn learn(&mut self, cfg: &GarchCfg, e: f64, w: f64, s: f64, dl: &[f64], replay: bool) {
        let (vol, k, nf) = (cfg.vol, cfg.vol.k(), cfg.n_free());
        self.w_sum += w;
        let a = w / self.w_sum;
        let e2 = e * e;
        self.mean_e2 += a * (e2 - self.mean_e2);
        if vol == GarchVol::Egarch {
            self.zbins[guard::bin(e / s.sqrt())] += w;
        }
        let mut sc = [0.0; 6];
        if cfg.dist == GarchDist::Normal {
            let mut df = [0.0; 4];
            let z2 = e2 / s;
            let resid = if cfg.variance_targeting {
                // the free (α, β) through ω = v̄ (1 − α − β), and the
                // Huber-clipped score centred on its own running mean,
                // whose level the target fixes
                let v = self.mean_e2;
                df[0] = dl[1] - v * dl[0];
                df[1] = dl[2] - v * dl[0];
                let clipped = z2.min(HUBER_C2);
                self.kappa += a * (clipped - self.kappa);
                self.slope += a * (if z2 < HUBER_C2 { z2 } else { 0.0 } - self.slope);
                if self.slope > 0.0 {
                    (clipped - self.kappa) / self.slope
                } else {
                    clipped - self.kappa
                }
            } else {
                df[..k].copy_from_slice(dl);
                z2 - 1.0
            };
            for i in 0..nf {
                sc[i] = 0.5 * resid * df[i];
                for j in 0..=i {
                    let t = tri(i, j);
                    self.h[t] += a * (0.5 * df[i] * df[j] - self.h[t]);
                }
            }
        } else {
            let (nu, lam) = (self.theta[k], self.theta.get(k + 1).copied().unwrap_or(0.0));
            let skew = cfg.dist == GarchDist::Skewt;
            let (wt, s_nu, s_lam) = dist::scores(e / s.sqrt(), nu, lam, skew);
            for i in 0..k {
                sc[i] = 0.5 * wt * dl[i];
            }
            sc[k] = (nu - 2.0) * s_nu;
            if skew {
                sc[k + 1] = s_lam;
            }
            for i in 0..nf {
                for j in 0..=i {
                    let t = tri(i, j);
                    self.h[t] += a * (sc[i] * sc[j] - self.h[t]);
                }
            }
        }
        if !replay {
            self.step(cfg, a, &sc[..nf]);
        }
        self.filter.push(vol, e, s, dl);
    }

    /// The step `θ ← Proj(θ + a H⁻¹ g)` at gain `a`, projected in `H`'s
    /// metric onto the form's set (EGARCH's box, [`super::fit::egarch_set`],
    /// then cut back to keep its filter contracting, [`guard`]); none where
    /// `H` is not positive definite
    /// or the step would leave a parameter that is not a number. No step is
    /// refused on a running mean: such a mean holds each past row at the
    /// parameters of its time, so once it crosses its bound it refuses every
    /// step one way and takes every step the other, a ratchet driven by the
    /// score's noise.
    fn step(&mut self, cfg: &GarchCfg, a: f64, sc: &[f64]) {
        let (vol, k, nf) = (cfg.vol, cfg.vol.k(), cfg.n_free());
        let mut hm = unpack(&self.h, nf);
        for i in 0..nf {
            hm[i * nf + i] += STEP_RIDGE * hm[i * nf + i].abs();
        }
        let Some(dir) = chol_solve(&hm, nf, sc) else {
            return;
        };
        let mv: Vec<f64> = dir.iter().map(|d| a * d).collect();
        let th = &self.theta;
        let new = if cfg.variance_targeting {
            let prop = [th[1] + mv[0], th[2] + mv[1]];
            let (mut y, _) = project(&ab_set(), &prop, &hm, 2);
            y[0] = y[0].max(0.0);
            y[1] = y[1].max(0.0);
            vec![self.mean_e2 * (1.0 - y[0] - y[1]), y[0], y[1]]
        } else {
            let floor = OMEGA_FLOOR_FRAC * self.mean_e2;
            let mut prop: Vec<f64> = th.iter().zip(&mv).map(|(t, m)| t + m).collect();
            let (mut y, shift) = project(&set_of(vol, floor), &prop, &hm, k);
            exact(vol, &mut y, floor);
            if nf > k {
                // a t: the shape's step, ν − 2 ← (ν − 2) e^{Δs} (and λ), with
                // the move the projection gave it on top
                prop[k] = 2.0 + (th[k] - 2.0) * math::exp((mv[k] + shift[0]).min(700.0));
                if nf > k + 1 {
                    prop[k + 1] = th[k + 1] + mv[k + 1] + shift[1];
                }
                clip_shape(&mut prop[k..]);
            }
            prop[..k].copy_from_slice(&y);
            prop
        };
        if !new.iter().all(|v| v.is_finite()) {
            return;
        }
        // EGARCH: the step cut back to keep its filter contracting
        let new = if vol == GarchVol::Egarch {
            guard::cut(&self.theta, &new, &self.zbins)
        } else {
            new
        };
        self.theta = new;
    }

    /// The bytes this series holds: [`super::Garch::series_bytes`] at its
    /// buffer's length.
    pub fn bytes(&self, cfg: &GarchCfg) -> u64 {
        super::Garch::series_bytes(cfg, self.buffer.len())
    }

    /// The variance `horizon` rows ahead of a row whose variance is `s`.
    pub fn forecast(&self, cfg: &GarchCfg, s: f64, scratch: &mut Vec<f64>) -> f64 {
        let h = cfg.horizon;
        if h <= 1 || !s.is_finite() {
            return s;
        }
        let th = &self.theta;
        match cfg.vol {
            GarchVol::Figarch => self.filter.figarch_forecast(th, s, h, scratch),
            _ => {
                // a later row's e² is read as its variance, and GJR's
                // asymmetric term as half of it (its innovations are
                // symmetric: a skewed t takes no GJR), as `arch` reads them
                let persist = match cfg.vol {
                    GarchVol::Gjr => th[1] + 0.5 * th[2] + th[3],
                    _ => th[1] + th[2],
                };
                let mut f = s;
                for _ in 1..h {
                    f = th[0] + persist * f;
                }
                f
            }
        }
    }
}

/// The first `kc` coordinates of `prop` onto `set` in the metric of the
/// step's `H` (`nf × nf`, row-major), and the move the rest -- a t's shape,
/// which no constraint binds -- takes from it. With the shape eliminated
/// the metric on the constrained part is the Schur complement `S = H₁₁ −
/// H₁₂ H₂₂⁻¹ H₂₁`, and the shape moves by `−H₂₂⁻¹ H₂₁ (y − prop)`. A point
/// inside is its own projection; one the metric leaves a hair outside by
/// rounding is pushed back in the Euclidean metric.
fn project(set: &HalfSpaces, prop: &[f64], hm: &[f64], kc: usize) -> (Vec<f64>, Vec<f64>) {
    let nf = (hm.len() as f64).sqrt().round() as usize;
    let m = nf - kc;
    let head = &prop[..kc];
    if set.holds(head, 1e-15) {
        return (head.to_vec(), vec![0.0; m]);
    }
    let fallback = || (set.project(head), vec![0.0; m]);
    // W = H₂₂⁻¹ H₂₁ (m × kc), column by column, and S
    let mut wm = vec![0.0; m * kc];
    if m > 0 {
        let h22: Vec<f64> = (0..m * m)
            .map(|q| hm[(kc + q / m) * nf + kc + q % m])
            .collect();
        let Some(l22) = cholesky(&h22, m) else {
            return fallback();
        };
        for c in 0..kc {
            let col: Vec<f64> = (0..m).map(|i| hm[(kc + i) * nf + c]).collect();
            let x = backward(&l22, m, &forward(&l22, m, &col));
            for i in 0..m {
                wm[i * kc + c] = x[i];
            }
        }
    }
    let mut schur = vec![0.0; kc * kc];
    for i in 0..kc {
        for j in 0..kc {
            let cross: f64 = (0..m).map(|q| hm[i * nf + kc + q] * wm[q * kc + j]).sum();
            schur[i * kc + j] = hm[i * nf + j] - cross;
        }
    }
    let Some(mut y) = set.project_metric(head, &schur) else {
        return fallback();
    };
    if !set.holds(&y, 1e-12) {
        y = set.project(&y);
    }
    let shift = (0..m)
        .map(|i| {
            -(0..kc)
                .map(|c| wm[i * kc + c] * (y[c] - head[c]))
                .sum::<f64>()
        })
        .collect();
    (y, shift)
}

/// The bounds that keep every shock's weight in the variance at 0 or above,
/// made exact on a projected `θ`: the projection meets them only to its
/// tolerance (`10⁻¹²` of each constraint's size), and a weight a hair below
/// 0 times a squared return as large as the inputs allow (`10²⁰⁰`) takes
/// the variance below 0 (review 2026-11-23, GARCH-12). `ω` at least its
/// floor, and GARCH's `α, β ≥ 0`; GJR's `α ≥ 0`, `α + γ ≥ 0` (as `γ ≥ −α`,
/// which rounds to an exact 0 sum) and `β ≥ 0`; FIGARCH's `φ ≥ 0`, `0 ≤ d ≤
/// 1 − 10⁻⁴`, `2φ + d ≤ 1` and `0 ≤ β ≤ φ + d`, whose weights' own
/// rounding the filter takes care of ([`super::filter`]). EGARCH's
/// variance is an exponential; its box, `|γ| ≤ α ≤ 1` and `0 ≤ β ≤ 1 −
/// 10⁻⁴`, is made exact too, as the guard reads it.
pub(super) fn exact(vol: GarchVol, y: &mut [f64], floor: f64) {
    if !y.iter().all(|v| v.is_finite()) {
        return;
    }
    match vol {
        GarchVol::Garch => {
            y[0] = y[0].max(floor);
            y[1] = y[1].max(0.0);
            y[2] = y[2].max(0.0);
        }
        GarchVol::Gjr => {
            y[0] = y[0].max(floor);
            y[1] = y[1].max(0.0);
            y[2] = y[2].max(-y[1]);
            y[3] = y[3].max(0.0);
        }
        GarchVol::Figarch => {
            y[0] = y[0].max(floor);
            y[2] = y[2].clamp(0.0, super::fit::D_MAX);
            y[1] = y[1].clamp(0.0, 0.5 * (1.0 - y[2]));
            y[3] = y[3].clamp(0.0, y[1] + y[2]);
        }
        GarchVol::Egarch => {
            // the box alone, which the guard reads: `|γ| ≤ α ≤ 1`, `0 ≤ β ≤
            // 1 − 10⁻⁴`
            y[1] = y[1].clamp(0.0, 1.0);
            y[2] = y[2].clamp(-y[1], y[1]);
            y[3] = y[3].clamp(0.0, 1.0 - super::fit::MARGIN);
        }
    }
}

/// The rows a target's start may buffer: its share of what
/// `max_state_bytes` holds, the most at which every target's series and the
/// model's own bytes fit under it ([`super::Garch::state_bytes`]).
pub(super) fn bound(cfg: &GarchCfg) -> u64 {
    let share = cfg.cap().saturating_sub(16) / cfg.n_targets as u64;
    share.saturating_sub(super::Garch::series_bytes(cfg, 0) - 8 * cfg.truncation() as u64) / 16
}
