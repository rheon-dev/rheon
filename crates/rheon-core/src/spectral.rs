//! `Spectral`: an exponentially weighted discrete Fourier transform of each
//! target at chosen frequencies, read as scipy's periodogram.
//!
//! # The transform
//!
//! For a target `x`, frequencies `f_k` in cycles per clock unit (each the
//! inverse of a period), and each learned row `s` at clock `τ_s` with
//! weight `w_s`, the row's amplitude is `a_s = w_s λ^{age_s}`, its age on
//! the clock. The transform and the sums beside it are
//!
//! ```text
//! X_k = Σ_s a_s x_s e^{2πi f_k (t − τ_s)}      U_k = Σ_s a_s e^{2πi f_k (t − τ_s)}
//! A   = Σ_s a_s      B = Σ_s a_s x_s           Q = Σ_s a_s²
//! ```
//!
//! referred to the current clock `t`, which moves no modulus. Per row, with
//! clock step `Δ`, `λ = decay.factor(Δ)` and the rotation
//! `r_k = e^{2πi f_k Δ}`:
//!
//! ```text
//! X_k ← λ r_k X_k + w x      U_k ← λ r_k U_k + w
//! A   ← λ A + w              B   ← λ B + w x        Q ← λ² Q + w²
//! ```
//!
//! # The read
//!
//! With `detrend`, the weighted mean `m = B / A` is removed, `X_k − m U_k`;
//! that is the transform of `x − m` under the window `a`. The power is
//! scipy's `periodogram` of the rows with `window = a`, one-sided:
//!
//! ```text
//! density:   P_k = c_k |X_k − m U_k|² · spacing / Q
//! spectrum:  P_k = c_k |X_k − m U_k|² / A²
//! ```
//!
//! `c_k` is 2, and 1 at `f = 0` and at the Nyquist frequency
//! `f = 1 / (2 spacing)`; `spacing` is the clock units between rows, `1/fs`
//! in scipy's terms. On an even clock of that spacing, with the frequencies
//! on scipy's grid, the read is `periodogram(x − m, fs, window=a,
//! detrend=False, scaling=…)` there.
//!
//! # Lomb-Scargle
//!
//! Under `LombScargle` the read is the generalised Lomb-Scargle power
//! (Zechmeister and Kürster 2009), which asks no even clock: per frequency,
//! the weighted least squares fit of `x` on `cos θ_s` and `sin θ_s`, `θ_s =
//! 2π f (t − τ_s)`, with a constant under `detrend`, and the share of the
//! weighted variance it explains. Its sums are the transform's and two
//! more, the weights' transform at twice the frequency and `Σ a x²`:
//!
//! ```text
//! V_k ← λ r_k² V_k + w      S ← λ S + w x²
//! C = Re U   Sn = Im U   YC = Re X   YS = Im X
//! CC = (A + Re V)/2   SS = (A − Re V)/2   CS = Im V / 2   YY = S
//! ```
//!
//! With `detrend` each is centred, `CC −= C²/A`, `CS −= C·Sn/A`, `SS −=
//! Sn²/A`, `YC −= B·C/A`, `YS −= B·Sn/A`, `YY −= B²/A`, and the power is
//!
//! ```text
//! P_k = (SS·YC² − 2 CS·YC·YS + CC·YS²) / ((CC·SS − CS²) · YY)
//! ```
//!
//! the fit's explained sum of squares over the total, in `[0, 1]`: scipy's
//! `lombscargle(t, x, 2πf, normalize=True, weights=a,
//! floating_mean=detrend)`. Where the fit's two columns are not spread
//! (`CC·SS − CS²` at or below `1e-10` of `(CC + SS)²/4`: the phases have not
//! covered enough of a cycle) or the target has no spread, the power is
//! withheld. The state is `8 (6F + 4)` bytes a target.
//!
//! # A window
//!
//! Under `window` the read is of the rows less than `window` clock units
//! old alone, a hard cutoff: with no decay, scipy's periodogram of the
//! window's rows. The window is a subtraction, as every windowed model's
//! is ([`crate::Snapshots`]): the live sums less the sums a snapshot held
//! before the window's first row, decayed forward. The sums are referred
//! to an origin fixed on the clock, so a snapshot's sums need turning only
//! once, to the live origin, before they are subtracted: a row at `σ`
//! clock units past the origin adds `w x e^{-2πi f σ}`, the origin moves
//! to the row (the sums turned by `e^{2πi f σ}`) each time a snapshot is
//! taken, and the turns are kept as a running product a frequency, from
//! which the boundary's own turn is read ([`SpectralWindow`]). A row costs
//! one `sin` and `cos` a frequency for its phase, and another where a
//! snapshot moves the origin; the view is `O(F)` a target. Each snapshot
//! holds the state, `8 (4F + 3)` bytes a target (`8 (6F + 4)` under
//! Lomb-Scargle), and the turns, `16 F`; the ring is bounded by the
//! window's budget. A break empties the boundary's sums too.
//!
//! The cost is `O(F)` a row for `F` frequencies, one `sin` and one `cos`
//! each (none where the clock's step is the last one's), from the `libm`
//! crate so the state is the same bits on every OS ([`rotation`]), and the state
//! `8 (4F + 3)` bytes a target, bounded by the frequencies, never by the
//! stream.
//!
//! **A break empties the transform.** The phase between rows either side of
//! a session change or a capped gap is not known, as a lag across one is
//! not, so [`OnlineModel::clear_lags`] zeroes each target's sums; the
//! shared weight is kept.

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};

/// `[cos θ, sin θ]`, the one place the transform takes its trigonometry
/// from: the `libm` crate's, the same bits on every OS, as `garch` takes
/// its `ln` and `exp`. A platform's library may differ in the last bit, and
/// a rotation applied on every row compounds that bit: one ulp in it moved
/// the power by up to 140,089 ulps over 20,000 rows with no decay (1,260 at
/// a half-life of 300), measured. On x86-64 the crate's `sin` and `cos`
/// cost no more than glibc's within the run's noise: 187 to 257 ns a row at
/// 16 frequencies on an uneven clock against 212 to 230, 651 to 681 at 64
/// against 666 to 747. An even clock takes them once, not once a row.
#[inline]
pub(crate) fn rotation(theta: f64) -> [f64; 2] {
    [libm::cos(theta), libm::sin(theta)]
}

/// What a [`Spectral`] reports at each frequency: scipy's `periodogram`
/// `scaling`, or the generalised Lomb-Scargle power.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectralScaling {
    /// Power spectral density, `|X|² spacing / Σa²`: scipy's default.
    #[default]
    Density,
    /// Power spectrum, `|X|² / (Σa)²`.
    Spectrum,
    /// The generalised Lomb-Scargle power (Zechmeister and Kürster 2009):
    /// the share of the target's weighted variance that a sinusoid at the
    /// frequency explains, between 0 and 1, from the weighted least squares
    /// fit of `a cos + b sin` (plus a constant under `detrend`). scipy's
    /// `lombscargle(normalize=True, weights=a, floating_mean=detrend)`.
    /// It needs no even clock, and has no Nyquist limit.
    LombScargle,
}

/// The settings of a [`Spectral`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralCfg {
    pub n_targets: usize,
    /// The frequencies, in cycles per clock unit, each at least 0 and at
    /// most the Nyquist frequency `1 / (2 spacing)`; under Lomb-Scargle,
    /// each above 0 and no limit above.
    pub frequencies: Vec<f64>,
    /// Clock units between rows (`1/fs`): the scale of the density and the
    /// place of the Nyquist frequency.
    pub spacing: f64,
    pub decay: Decay,
    pub scaling: SpectralScaling,
    /// Remove the weighted mean before the transform.
    pub detrend: bool,
    /// A target's power is withheld while its weight is below this.
    pub min_weight: f64,
    /// The bytes a group's state may hold; `None` is
    /// [`crate::DEFAULT_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
    /// A hard window of this many clock units: the transform of the rows
    /// inside it alone ([`crate::Snapshots`]); `None` for none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window: Option<f64>,
    /// Clock units between the window's snapshots; `None` with no row cap
    /// either is a snapshot every row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub window_every: Option<f64>,
    /// At most this many rows between the window's snapshots.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rows_between_snapshots: Option<usize>,
}

/// The most frequencies a spectrum keeps.
pub const MAX_FREQUENCIES: usize = 4096;

impl SpectralCfg {
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
    }

    /// Whether the read is the Lomb-Scargle power, which keeps `V` and
    /// `Σ a x²` beside the transform.
    pub fn lomb_scargle(&self) -> bool {
        self.scaling == SpectralScaling::LombScargle
    }

    /// The window's snapshot cadence.
    pub fn cadence(&self) -> crate::Cadence {
        crate::Cadence::of(self.window_every, self.max_rows_between_snapshots)
    }

    /// The Nyquist frequency, `1 / (2 spacing)`.
    pub fn nyquist(&self) -> f64 {
        0.5 / self.spacing
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("spectral: {e}")));
        if self.n_targets == 0 {
            no.refuse(
                "spectral: the model needs at least one target, the series it transforms, but \
                 it has none. Name at least one column in targets.",
            );
        }
        if !(self.spacing.is_finite() && self.spacing > 0.0) {
            no.refuse(format!(
                "spectral: spacing must be finite and above 0, but it is {}. Set spacing to \
                 the clock units between rows.",
                self.spacing
            ));
        }
        if self.frequencies.is_empty() {
            no.refuse(
                "spectral: periods must list at least one period, but it lists none. Give the \
                 periods to transform at.",
            );
        }
        if self.frequencies.len() > MAX_FREQUENCIES {
            no.refuse(format!(
                "spectral: periods may list at most {MAX_FREQUENCIES}, but it lists {}. List \
                 fewer periods.",
                self.frequencies.len()
            ));
        }
        no.stop_here()?;
        let nyq = self.nyquist();
        if self.scaling == SpectralScaling::LombScargle {
            if let Some(f) = self
                .frequencies
                .iter()
                .find(|f| !(f.is_finite() && **f > 0.0))
            {
                no.refuse(format!(
                    "spectral: under scaling \"lomb_scargle\" each period must be finite, \
                     but one is {}: a sinusoid of no frequency is the constant the fit \
                     already holds. Give finite periods.",
                    1.0 / f
                ));
            }
        } else if let Some(f) = self
            .frequencies
            .iter()
            .find(|f| !(f.is_finite() && **f >= 0.0 && **f <= nyq))
        {
            no.refuse(format!(
                "spectral: each period must be at least twice the spacing, the Nyquist \
                 limit, but one is {}. Give periods of at least {}.",
                1.0 / f,
                2.0 * self.spacing
            ));
        }
        no.take(crate::model::check_min_weight("spectral", self.min_weight));
        if let Some(w) = self.window {
            no.take(
                crate::Snapshots::<SpectralSnap>::with_cadence(w, self.cadence())
                    .map(|_| ())
                    .map_err(|e| format!("spectral: {e}")),
            );
        }
        no.take(crate::clock_span::check_max_state_bytes(
            "spectral",
            self.max_state_bytes,
        ));
        no.stop_here()?;
        let bytes = Spectral::state_bytes(self);
        if bytes > self.cap() {
            no.refuse(format!(
                "spectral: the state of {} periods over {} targets would hold {bytes} bytes, \
                 past max_state_bytes of {}. List fewer periods, or raise max_state_bytes.",
                self.frequencies.len(),
                self.n_targets,
                self.cap()
            ));
        }
        no.finish()
    }
}

/// One target's sums.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Sums {
    /// `X_k` and `U_k`, `[re, im]` a frequency each.
    x: Vec<[f64; 2]>,
    u: Vec<[f64; 2]>,
    /// Under Lomb-Scargle, `V_k`, the weights' transform at twice each
    /// frequency, `[re, im]`; empty otherwise.
    v: Vec<[f64; 2]>,
    a: f64,
    b: f64,
    q: f64,
    /// Under Lomb-Scargle, `Σ a x²`; 0 otherwise.
    s: f64,
}

impl Sums {
    /// The numbers a snapshot keeps of it, in order: `X`, `U`, `V` as
    /// `[re, im]` pairs, then `A`, `B`, `Q`, `S`.
    fn write(&self, out: &mut Vec<f64>) {
        for z in self.x.iter().chain(&self.u).chain(&self.v) {
            out.extend_from_slice(z);
        }
        out.extend_from_slice(&[self.a, self.b, self.q, self.s]);
    }

    /// How many numbers [`Self::write`] writes.
    fn len(&self) -> usize {
        2 * (self.x.len() + self.u.len() + self.v.len()) + 4
    }

    /// Every sum times `lam`, `Q` times `lam2`: the decay to the next row.
    fn decay(&mut self, lam: f64, lam2: f64) {
        for z in self.x.iter_mut().chain(&mut self.u).chain(&mut self.v) {
            z[0] *= lam;
            z[1] *= lam;
        }
        self.a *= lam;
        self.b *= lam;
        self.s *= lam;
        self.q *= lam2;
    }

    /// The sums referred to an origin `rot` later: `X` and `U` turned by
    /// each frequency's `rot`, `V` by its square.
    fn turn(&mut self, rot: &[[f64; 2]]) {
        let mul = |z: &mut [f64; 2], c: f64, s: f64| {
            let [r, i] = *z;
            *z = [c * r - s * i, c * i + s * r];
        };
        for (k, &[c, sn]) in rot.iter().enumerate() {
            mul(&mut self.x[k], c, sn);
            mul(&mut self.u[k], c, sn);
            if let Some(v) = self.v.get_mut(k) {
                mul(v, c * c - sn * sn, 2.0 * c * sn);
            }
        }
    }

    fn zero(&mut self) {
        for z in self.x.iter_mut().chain(&mut self.u).chain(&mut self.v) {
            *z = [0.0; 2];
        }
        self.a = 0.0;
        self.b = 0.0;
        self.q = 0.0;
        self.s = 0.0;
    }

    fn new(f: usize, ls: bool) -> Self {
        Self {
            x: vec![[0.0; 2]; f],
            u: vec![[0.0; 2]; f],
            v: vec![[0.0; 2]; if ls { f } else { 0 }],
            a: 0.0,
            b: 0.0,
            q: 0.0,
            s: 0.0,
        }
    }
}

/// An exponentially weighted DFT per target; see the module docs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Spectral {
    cfg: SpectralCfg,
    sums: Vec<Sums>,
    /// The weight of every row before the next (hard rule 8).
    w_sum: f64,
    /// The last clock step's rotations, kept so an even clock takes no
    /// `sin` or `cos` a row: not state (recomputed to the same bits).
    #[serde(skip)]
    rot: Option<(u64, Vec<[f64; 2]>)>,
    /// Under a window, its ring and the sums' clock origin.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    win: Option<SpectralWindow>,
    /// The window's view, kept so a row allocates nothing: not state.
    #[serde(skip)]
    view: ViewKept,
}

/// A [`Spectral`]'s window. Under it the sums are referred to a fixed
/// origin on the clock, not to the current row: a row at `since` clock
/// units past it adds at the phase `2π f since`, and nothing turns on a
/// row but the row's own term. The origin moves to the row each time a
/// snapshot is taken, the sums turned once to it, so `since` spans one
/// block between snapshots and its phase holds its digits however long the
/// stream. The turns are multiplied into a running product a frequency,
/// as the decay's factors are ([`crate::Snapshots`]); each snapshot keeps
/// the product it was taken at, and the window's view turns the boundary's
/// sums by the ratio of the live product to it, the turns the live sums
/// went through since, before it subtracts them. Read from the summed
/// clock instead, the turn parted from the sums' by the clock's own
/// rounding: at ten million rows with no decay, 8e-6 of the power, against
/// 1.5e-8 from the product (the subtraction's own, which grows with the
/// live sums).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SpectralWindow {
    /// The model's clock, summed from the steps: what keys the snapshots.
    clock: f64,
    /// Clock units from the live sums' origin to the last learned row.
    since: f64,
    /// The product of every turn of the origin, `[re, im]` a frequency.
    turned: Vec<[f64; 2]>,
    snaps: crate::Snapshots<SpectralSnap>,
}

/// The state before a row, decayed to it, at its origin: the turns'
/// product there (`[re, im]` a frequency, flat), the shared weight, and
/// each target's sums ([`Sums::write`]) one after another.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralSnap {
    turned: Vec<f64>,
    w: f64,
    sums: Vec<f64>,
}

crate::pack_fields!(SpectralSnap { turned, w, sums });

impl crate::Footprint for SpectralSnap {
    fn footprint(&self) -> usize {
        8 * (1 + self.turned.len() + self.sums.len())
    }
}

/// The window's view and the turns of its boundary, kept between rows.
#[derive(Debug, Default)]
struct ViewKept {
    sums: Vec<Sums>,
    w: f64,
    /// Whether `sums` and `w` hold the view; otherwise the live sums are
    /// read (no window, or nothing has left it).
    on: bool,
    /// The boundary's turn to the live origin, a frequency each.
    turn: Vec<[f64; 2]>,
    /// The origin's turn on a row that moves it, a frequency each.
    rebase: Vec<[f64; 2]>,
}

impl Clone for ViewKept {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for ViewKept {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Spectral {
    pub fn new(cfg: SpectralCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (f, ls) = (cfg.frequencies.len(), cfg.lomb_scargle());
        let win = match cfg.window {
            Some(w) => Some(SpectralWindow {
                clock: 0.0,
                since: 0.0,
                turned: vec![[1.0, 0.0]; cfg.frequencies.len()],
                snaps: crate::Snapshots::with_cadence(w, cfg.cadence())?,
            }),
            None => None,
        };
        Ok(Self {
            sums: (0..cfg.n_targets).map(|_| Sums::new(f, ls)).collect(),
            w_sum: 0.0,
            rot: None,
            win,
            view: ViewKept::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &SpectralCfg {
        &self.cfg
    }

    /// The weight before the next row: inside the window under one.
    pub fn n_eff(&self) -> f64 {
        let mut view = ViewKept::default();
        self.view_into(&mut view);
        self.read_w(&view)
    }

    /// The bytes the state holds: `8 (4F + 3)` a target, `8 (6F + 4)`
    /// under Lomb-Scargle, and 8 shared.
    pub fn state_bytes(cfg: &SpectralCfg) -> u64 {
        let f = cfg.frequencies.len() as u64;
        let each = if cfg.lomb_scargle() {
            8 * (6 * f + 4)
        } else {
            8 * (4 * f + 3)
        };
        each.saturating_mul(cfg.n_targets as u64) + 8
    }

    /// The smallest determinant, as a share of `(CC + SS)² / 4`, at which
    /// the Lomb-Scargle fit's two columns count as spread: below it the
    /// rows' phases at the frequency have not covered a cycle enough to
    /// tell a cosine from a sine, and the power is withheld. A determinant
    /// that rounding alone leaves is about `1e-16` of it.
    pub const SPREAD: f64 = 1e-10;

    /// The Lomb-Scargle power of target `s` at frequency `k`; `None` where
    /// the fit is not determined (see [`Self::SPREAD`]) or the target has
    /// no spread of its own.
    fn ls_power(&self, s: &Sums, k: usize) -> Option<f64> {
        let a = s.a;
        let [c, sn] = s.u[k];
        let [yc, ys] = s.x[k];
        let [vr, vi] = s.v[k];
        let (mut cc, mut ss, mut cs) = (0.5 * (a + vr), 0.5 * (a - vr), 0.5 * vi);
        let (mut yc, mut ys, mut yy) = (yc, ys, s.s);
        if self.cfg.detrend {
            cc -= c * c / a;
            ss -= sn * sn / a;
            cs -= c * sn / a;
            yc -= s.b * c / a;
            ys -= s.b * sn / a;
            yy -= s.b * s.b / a;
        }
        let det = cc * ss - cs * cs;
        let scale = 0.25 * (cc + ss) * (cc + ss);
        if !(det > Self::SPREAD * scale && yy > 0.0) {
            return None;
        }
        let explained = (ss * yc * yc - 2.0 * cs * yc * ys + cc * ys * ys) / det;
        Some(explained / yy)
    }

    /// The rotation `e^{2πi f Δ}` of each frequency for a clock step `d`.
    fn rotations(&self, d: f64) -> Vec<[f64; 2]> {
        self.cfg
            .frequencies
            .iter()
            .map(|f| rotation(std::f64::consts::TAU * f * d))
            .collect()
    }

    /// Target `j`'s sums as they are read: the window's view under one,
    /// the live sums otherwise.
    fn read_sums<'a>(&'a self, view: &'a ViewKept, j: usize) -> &'a Sums {
        if view.on {
            &view.sums[j]
        } else {
            &self.sums[j]
        }
    }

    /// The weight read with them (hard rule 8: inside the window under one).
    fn read_w(&self, view: &ViewKept) -> f64 {
        if view.on { view.w } else { self.w_sum }
    }

    /// The power at every frequency of the sums `s`, written into `out`.
    fn power_into(&self, s: &Sums, out: &mut [f64]) {
        let open = s.q > 0.0 && s.a > 0.0;
        let m = if self.cfg.detrend && open {
            s.b / s.a
        } else {
            0.0
        };
        let scale = match self.cfg.scaling {
            SpectralScaling::Density => self.cfg.spacing / s.q,
            SpectralScaling::Spectrum => 1.0 / (s.a * s.a),
            SpectralScaling::LombScargle => 1.0,
        };
        let nyq = self.cfg.nyquist();
        for (k, f) in self.cfg.frequencies.iter().enumerate() {
            if !open {
                out[k] = f64::NAN;
                continue;
            }
            if self.cfg.lomb_scargle() {
                out[k] = self.ls_power(s, k).unwrap_or(f64::NAN);
                continue;
            }
            let re = s.x[k][0] - m * s.u[k][0];
            let im = s.x[k][1] - m * s.u[k][1];
            let c = if *f == 0.0 || *f == nyq { 1.0 } else { 2.0 };
            out[k] = c * (re * re + im * im) * scale;
        }
    }

    fn read(&self, view: &ViewKept, out: &mut Step) {
        let f = self.cfg.frequencies.len();
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets * f, f64::NAN);
        out.n_eff = self.read_w(view);
        out.extra = None;
        for j in 0..self.cfg.n_targets {
            let s = self.read_sums(view, j);
            if s.a >= self.cfg.min_weight {
                self.power_into(s, &mut out.pred[j * f..(j + 1) * f]);
            }
        }
    }

    /// The window's view written into `kept`: each live sum less the
    /// boundary snapshot's, decayed forward by the products the sums went
    /// through and turned once from its origin to the live one. `kept.on`
    /// is false, and the live sums are read, without a window or where
    /// nothing has left it yet.
    fn view_into(&self, kept: &mut ViewKept) {
        kept.on = false;
        let Some(win) = self.win.as_ref() else {
            return;
        };
        let Some((f, old)) = win.snaps.cut(win.clock, self.cfg.decay, 0) else {
            return;
        };
        if old.w == 0.0 {
            // Nothing has left the window: the live sums, to the bit.
            return;
        }
        let f2 = win
            .snaps
            .cut(win.clock, self.cfg.decay, 1)
            .map_or(f * f, |(f2, _)| f2);
        let ViewKept {
            sums, w, on, turn, ..
        } = kept;
        // The turns since the boundary: the live product over its.
        turn.clear();
        turn.extend(win.turned.iter().enumerate().map(|(k, &[pr, pi])| {
            let (or, oi) = (old.turned[2 * k], old.turned[2 * k + 1]);
            let m = or * or + oi * oi;
            [(pr * or + pi * oi) / m, (pi * or - pr * oi) / m]
        }));
        let rot = &*turn;
        let (nf, ls) = (self.cfg.frequencies.len(), self.cfg.lomb_scargle());
        if sums.len() != self.sums.len() {
            *sums = (0..self.sums.len()).map(|_| Sums::new(nf, ls)).collect();
        }
        let per = self.sums.first().map_or(0, Sums::len);
        for (j, (live, out)) in self.sums.iter().zip(sums.iter_mut()).enumerate() {
            let o = &old.sums[j * per..(j + 1) * per];
            let nv = live.v.len();
            let pair = |at: usize| [o[2 * at], o[2 * at + 1]];
            for (k, &[c, sn]) in rot.iter().enumerate().take(nf) {
                let sub = |l: [f64; 2], z: [f64; 2], c: f64, sn: f64| {
                    let t = [c * z[0] - sn * z[1], c * z[1] + sn * z[0]];
                    [l[0] - f * t[0], l[1] - f * t[1]]
                };
                out.x[k] = sub(live.x[k], pair(k), c, sn);
                out.u[k] = sub(live.u[k], pair(nf + k), c, sn);
                if k < nv {
                    out.v[k] = sub(live.v[k], pair(2 * nf + k), c * c - sn * sn, 2.0 * c * sn);
                }
            }
            let t = 2 * (2 * nf + nv);
            out.a = live.a - f * o[t];
            out.b = live.b - f * o[t + 1];
            out.q = live.q - f2 * o[t + 2];
            out.s = live.s - f * o[t + 3];
            // A target with no weight left inside the window, its
            // remainder a rounding of the subtraction: nothing to read.
            if out.a.is_nan() || out.a <= crate::window::EMPTY_FRACTION * live.a {
                out.zero();
            }
        }
        *w = self.w_sum - f * old.w;
        if w.is_nan() || *w <= crate::window::EMPTY_FRACTION * self.w_sum {
            *w = 0.0;
        }
        *on = true;
    }

    /// The snapshot of the state as it stands, its turns' product `turned`.
    fn snapshot(&self, turned: &[[f64; 2]]) -> SpectralSnap {
        let mut sums = Vec::with_capacity(self.sums.iter().map(Sums::len).sum());
        for s in &self.sums {
            s.write(&mut sums);
        }
        SpectralSnap {
            turned: turned.iter().flatten().copied().collect(),
            w: self.w_sum,
            sums,
        }
    }

    /// One row learned under a window: the sums decayed to it; the origin
    /// moved to it where a snapshot is due, the sums turned once; the
    /// snapshot offered; then the row's term at its phase from the origin.
    fn learn_windowed(&mut self, y: &[Option<f64>], d_clock: f64, lam: f64, weight: f64) {
        let lam2 = lam * lam;
        self.w_sum *= lam;
        for s in &mut self.sums {
            s.decay(lam, lam2);
        }
        let win = self.win.as_mut().expect("a window");
        let mut since = win.since + d_clock;
        if win.snaps.takes_next(win.clock, d_clock) && since != 0.0 {
            let rot = &mut self.view.rebase;
            rot.clear();
            rot.extend(
                self.cfg
                    .frequencies
                    .iter()
                    .map(|f| rotation(std::f64::consts::TAU * f * since)),
            );
            for s in &mut self.sums {
                s.turn(rot);
            }
            for (p, &[c, sn]) in win.turned.iter_mut().zip(rot.iter()) {
                let [pr, pi] = *p;
                *p = [c * pr - sn * pi, c * pi + sn * pr];
            }
            since = 0.0;
        }
        win.since = since;
        let (sums, w_sum, turned) = (&self.sums, self.w_sum, &win.turned);
        win.snaps
            .learn_decayed(&mut win.clock, d_clock, &[lam, lam2], || {
                let mut flat = Vec::with_capacity(sums.iter().map(Sums::len).sum());
                for s in sums {
                    s.write(&mut flat);
                }
                SpectralSnap {
                    turned: turned.iter().flatten().copied().collect(),
                    w: w_sum,
                    sums: flat,
                }
            });
        self.w_sum += weight;
        for (j, s) in self.sums.iter_mut().enumerate() {
            let Some(x) = y[j].filter(|_| weight > 0.0) else {
                continue;
            };
            let (wx, w) = (weight * x, weight);
            for (k, f) in self.cfg.frequencies.iter().enumerate() {
                let [c, sn] = if since == 0.0 {
                    [1.0, 0.0]
                } else {
                    rotation(std::f64::consts::TAU * f * since)
                };
                // The row at `since` past the origin: `e^{-2πi f since}`.
                s.x[k][0] += wx * c;
                s.x[k][1] -= wx * sn;
                s.u[k][0] += w * c;
                s.u[k][1] -= w * sn;
                if let Some(v) = s.v.get_mut(k) {
                    v[0] += w * (c * c - sn * sn);
                    v[1] -= w * 2.0 * c * sn;
                }
            }
            s.a += w;
            s.b += wx;
            s.q += w * w;
            if !s.v.is_empty() {
                s.s += wx * x;
            }
        }
    }

    /// One row learned under decay `lam`, its weight `weight`, with each
    /// frequency's rotation `rot` for the row's clock step.
    pub(crate) fn learn(&mut self, y: &[Option<f64>], lam: f64, weight: f64, rot: &[[f64; 2]]) {
        self.w_sum = lam * self.w_sum + weight;
        for (j, s) in self.sums.iter_mut().enumerate() {
            let xj = y[j].filter(|_| weight > 0.0);
            let (wx, w) = xj.map_or((0.0, 0.0), |v| (weight * v, weight));
            for ((&[c, si], x), u) in rot.iter().zip(s.x.iter_mut()).zip(s.u.iter_mut()) {
                let [xr, xi] = *x;
                let [ur, ui] = *u;
                *x = [lam * (c * xr - si * xi) + wx, lam * (c * xi + si * xr)];
                *u = [lam * (c * ur - si * ui) + w, lam * (c * ui + si * ur)];
            }
            // Twice the frequency: the rotation squared.
            for (&[c, si], v) in rot.iter().zip(s.v.iter_mut()) {
                let (c2, s2) = (c * c - si * si, 2.0 * c * si);
                let [vr, vi] = *v;
                *v = [lam * (c2 * vr - s2 * vi) + w, lam * (c2 * vi + s2 * vr)];
            }
            if !s.v.is_empty() {
                s.s = lam * s.s + wx * xj.unwrap_or(0.0);
            }
            s.a = lam * s.a + w;
            s.b = lam * s.b + wx;
            s.q = lam * lam * s.q + w * w;
        }
    }

    /// Each target's power at every frequency after the last row, `F` each.
    pub fn powers(&self) -> Vec<Vec<f64>> {
        let f = self.cfg.frequencies.len();
        let mut view = ViewKept::default();
        self.view_into(&mut view);
        (0..self.cfg.n_targets)
            .map(|j| {
                let mut v = vec![f64::NAN; f];
                self.power_into(self.read_sums(&view, j), &mut v);
                v
            })
            .collect()
    }
}

impl OnlineModel for Spectral {
    fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        if let Some(refused) = crate::model::refused_step(self, x, y, d_clock, weight) {
            *out = refused;
            return;
        }
        // The power before the row (out of sample, hard rule 2).
        let mut view = std::mem::take(&mut self.view);
        self.view_into(&mut view);
        self.read(&view, out);
        self.view = view;
        let lam = self.cfg.decay.factor(d_clock);
        if self.win.is_some() {
            self.learn_windowed(y, d_clock, lam, weight);
            return;
        }
        let key = d_clock.to_bits();
        if self.rot.as_ref().is_none_or(|(k, _)| *k != key) {
            self.rot = Some((key, self.rotations(d_clock)));
        }
        let rot = std::mem::take(&mut self.rot.as_mut().expect("set above").1);
        self.learn(y, lam, weight, &rot);
        self.rot.as_mut().expect("set above").1 = rot;
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        let mut view = ViewKept::default();
        self.view_into(&mut view);
        self.read(&view, &mut out);
        out
    }

    /// A break: the phase across it is unknown, so each target's sums go;
    /// the shared weight stays.
    fn clear_lags(&mut self) {
        for s in &mut self.sums {
            s.zero();
        }
        // The boundary's sums too: what leaves the window afterwards is
        // the rows after the break, the shared weight kept.
        if let Some(win) = self.win.as_mut() {
            for snap in win.snaps.edit().iter_mut() {
                snap.sums.iter_mut().for_each(|v| *v = 0.0);
            }
        }
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        let mut view = ViewKept::default();
        self.view_into(&mut view);
        out.clear();
        out.extend((0..self.sums.len()).map(|j| self.read_sums(&view, j).a));
        true
    }

    fn stamp_next(&mut self, stamp: crate::Stamp) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.stamp_next(stamp);
        }
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_budget(budget);
        }
    }

    fn set_window_closed(&mut self, closed: crate::WindowClosed) {
        if let Some(win) = self.win.as_mut() {
            win.snaps.set_closed(closed);
        }
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        self.win.as_ref()?.snaps.over_budget()
    }

    fn window_shadow(&self) -> Option<crate::WindowShadow> {
        let win = self.win.as_ref()?;
        Some(crate::WindowShadow::new(win.clock, &win.snaps, || {
            self.snapshot(&win.turned)
        }))
    }

    fn window_could_refuse(&self, rows: usize) -> bool {
        self.win
            .as_ref()
            .is_some_and(|win| win.snaps.could_refuse(rows, || self.snapshot(&win.turned)))
    }

    fn state(&self) -> State {
        State::new(ModelState::Spectral(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Spectral(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("spectral", m.cfg.validate())?;
                let f = m.cfg.frequencies.len();
                let fv = if m.cfg.lomb_scargle() { f } else { 0 };
                let per = 4 * f + 2 * fv + 4;
                let shaped = m.sums.len() == m.cfg.n_targets
                    && m.sums
                        .iter()
                        .all(|s| s.x.len() == f && s.u.len() == f && s.v.len() == fv)
                    && m.win.is_some() == m.cfg.window.is_some()
                    && m.win.as_ref().is_none_or(|w| {
                        w.turned.len() == f
                            && w.snaps.iter().all(|s| {
                                s.sums.len() == per * m.cfg.n_targets && s.turned.len() == 2 * f
                            })
                    });
                if !shaped {
                    return Err(crate::model::damaged(
                        "spectral",
                        "its sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "spectral",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * self.cfg.frequencies.len()
    }
}

#[cfg(test)]
mod tests;
