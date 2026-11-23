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
//! window's rows. The rows are kept in blocks, each block's sums referred
//! to its own origin, the clock of its first row: a row adds
//! `w x e^{-2πi f (τ − o)}`, its phase from `libm` on its own clock less
//! its block's origin. A block is closed on the window's cadence (every
//! row by default) and added to a running total of the closed blocks,
//! turned once to a reference origin by `e^{-2πi f (o − R)}`, from `libm`
//! on that difference; it leaves whole, taken out of the total the same
//! way, once its first row is past the window, so the window never keeps
//! an older row. The reference moves to the newest block once it is two
//! windows old and the total is written afresh there, so what the
//! subtractions leave goes with it: no sum is turned row after row, and
//! the clock's size enters only through differences of nearby clocks.
//! Measured against the direct sum over the window's rows, with no decay,
//! from a thousand to ten million rows and with the clock starting near 0
//! and near `2^40`: within `1e-12` of the power. A row costs one `sin` and
//! `cos` a frequency for its phase and `O(F)` for the read; a block closing
//! or leaving another one each, and the reference moving `O(B F)`. The
//! window holds its blocks, `8 (4 + n (4F + 3))` bytes each for `n`
//! targets (`8 (4 + n (6F + 4))` under Lomb-Scargle), at most one a row
//! inside it, and the total, the size of the state's sums; a budget thins
//! them by merging neighbors, the younger turned once to the older's
//! origin, or refuses the run. A break empties every block's sums.
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
                crate::Snapshots::<crate::Bytes>::with_cadence(w, self.cadence())
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
    /// How many numbers it holds: `X`, `U`, `V` as pairs, `A`, `B`, `Q`, `S`.
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

/// A [`Spectral`]'s window: its rows in blocks, each block's sums
/// referred to its own origin on the clock, the clock of its first row
/// (the module docs). A closed block is turned once to the reference
/// origin, `e^{−2πi f (o − R)}` from `libm` on that difference, and added
/// to the total; no sum is turned row after row, so nothing compounds with
/// the stream's length. A block leaves whole once its first row is past the
/// window, so the window never keeps an older row; a block is closed and
/// the next begun on the cadence of `window_every` and
/// `max_rows_between_snapshots`, every row by default, which makes the edge
/// exact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct SpectralWindow {
    /// The clock of the last learned row, summed from the steps.
    clock: f64,
    /// The reference origin the blocks are turned to: a block's origin,
    /// moved to the newest block's once it is two windows old, so the
    /// turns' differences stay within two windows.
    reference: f64,
    /// Whether a row exactly one window old is held (`closed = "both"`).
    closed_both: bool,
    /// The closed blocks' sums, turned to the reference and decayed to the
    /// clock, a running total written afresh whenever the reference moves.
    total: Vec<Sums>,
    total_w: f64,
    /// The cadence a block is closed on, doubled by any thinning.
    spacing: f64,
    rows: usize,
    /// The budget, and the bytes a refusing one saw the blocks reach.
    budget: Option<crate::WindowBudget>,
    over: Option<usize>,
    /// The blocks, oldest first; the last is open.
    blocks: std::collections::VecDeque<Block>,
}

/// A run of rows of a window: its sums at its own origin, decayed to its
/// last row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Block {
    /// The clock of its first row, its origin, and of its last.
    origin: f64,
    last: f64,
    rows: usize,
    /// The shared weight of its rows, and each target's sums.
    w: f64,
    sums: Vec<Sums>,
}

impl Block {
    fn bytes(&self) -> usize {
        8 * (4 + self.sums.iter().map(Sums::len).sum::<usize>())
    }
}

/// The window's view, kept so a row allocates nothing: not state.
#[derive(Debug, Default)]
struct ViewKept {
    sums: Vec<Sums>,
    w: f64,
    /// Whether `sums` and `w` hold the view; otherwise the live sums are
    /// read (no window).
    on: bool,
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
            Some(_) => {
                let c = cfg.cadence();
                Some(SpectralWindow {
                    clock: 0.0,
                    reference: 0.0,
                    closed_both: false,
                    total: Vec::new(),
                    total_w: 0.0,
                    spacing: c.spacing,
                    rows: c.rows,
                    budget: None,
                    over: None,
                    blocks: std::collections::VecDeque::new(),
                })
            }
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

    /// The window's view written into `kept`: the closed blocks' running
    /// total at the reference origin, and the open block turned once to
    /// it. `kept.on` is false without a window.
    fn view_into(&self, kept: &mut ViewKept) {
        kept.on = false;
        let Some(win) = self.win.as_ref() else {
            return;
        };
        let (nf, ls) = (self.cfg.frequencies.len(), self.cfg.lomb_scargle());
        let ViewKept { sums, w, on } = kept;
        if sums.len() != self.sums.len() {
            *sums = (0..self.sums.len()).map(|_| Sums::new(nf, ls)).collect();
        }
        for s in sums.iter_mut() {
            s.zero();
        }
        for (out, t) in sums.iter_mut().zip(&win.total) {
            out.clone_from(t);
        }
        *w = win.total_w;
        if let Some(open) = win.blocks.back() {
            add_turned(sums, open, &self.cfg, open.origin - win.reference, 1.0, 1.0);
            *w += open.w;
        }
        *on = true;
    }

    /// The closed blocks' total written afresh: each turned once to the
    /// reference and decayed once from its last row to the clock.
    fn retotal(win: &mut SpectralWindow, cfg: &SpectralCfg) {
        for t in &mut win.total {
            t.zero();
        }
        win.total_w = 0.0;
        let n = win.blocks.len().saturating_sub(1);
        for b in win.blocks.iter().take(n) {
            let g = cfg.decay.factor(win.clock - b.last);
            add_turned(&mut win.total, b, cfg, b.origin - win.reference, g, 1.0);
            win.total_w += g * b.w;
        }
    }

    /// One row learned under a window: the closed blocks' total decayed to
    /// it; the open block closed into the total and a new one begun on the
    /// cadence, or the open block decayed to the row; the row's term at its
    /// phase from its block's origin; the blocks past the window taken out.
    fn learn_windowed(&mut self, y: &[Option<f64>], d_clock: f64, lam: f64, weight: f64) {
        let (nf, ls) = (self.cfg.frequencies.len(), self.cfg.lomb_scargle());
        let width = self.cfg.window.expect("a window");
        self.w_sum = lam * self.w_sum + weight;
        let cfg = &self.cfg;
        let win = self.win.as_mut().expect("a window");
        let tau = win.clock + d_clock;
        win.clock = tau;
        if win.total.len() != cfg.n_targets {
            win.total = (0..cfg.n_targets).map(|_| Sums::new(nf, ls)).collect();
        }
        for t in &mut win.total {
            t.decay(lam, lam * lam);
        }
        win.total_w *= lam;
        let begin = match win.blocks.back() {
            None => true,
            Some(b) => win.over.is_none() && (b.rows >= win.rows || tau - b.origin >= win.spacing),
        };
        if begin {
            if let Some(closing) = win.blocks.back() {
                // Closed into the total: decayed by the row's factor to the
                // row, as the total was, and turned once to the reference.
                let d = closing.origin - win.reference;
                add_turned(&mut win.total, closing, cfg, d, lam, 1.0);
                win.total_w += lam * closing.w;
            } else {
                win.reference = tau;
            }
            win.blocks.push_back(Block {
                origin: tau,
                last: tau,
                rows: 0,
                w: 0.0,
                sums: (0..cfg.n_targets).map(|_| Sums::new(nf, ls)).collect(),
            });
        }
        let open = win.blocks.back_mut().expect("a block");
        if !begin {
            open.w *= lam;
            for s in &mut open.sums {
                s.decay(lam, lam * lam);
            }
        }
        open.last = tau;
        open.rows += 1;
        open.w += weight;
        let since = tau - open.origin;
        for (j, s) in open.sums.iter_mut().enumerate() {
            let Some(x) = y[j].filter(|_| weight > 0.0) else {
                continue;
            };
            let (wx, w) = (weight * x, weight);
            for (k, f) in cfg.frequencies.iter().enumerate() {
                let [c, sn] = if since == 0.0 {
                    [1.0, 0.0]
                } else {
                    rotation(std::f64::consts::TAU * f * since)
                };
                // The row at `since` past its block's origin: `e^{−2πi f since}`.
                s.x[k][0] += wx * c;
                s.x[k][1] -= wx * sn;
                s.u[k][0] += w * c;
                s.u[k][1] -= w * sn;
                if let Some(v) = s.v.get_mut(k) {
                    let [c2, s2] = if since == 0.0 {
                        [1.0, 0.0]
                    } else {
                        rotation(2.0 * std::f64::consts::TAU * f * since)
                    };
                    v[0] += w * c2;
                    v[1] -= w * s2;
                }
            }
            s.a += w;
            s.b += wx;
            s.q += w * w;
            if !s.v.is_empty() {
                s.s += wx * x;
            }
        }
        // A block leaves whole once its first row has left: taken out of
        // the total, decayed once and turned once as it went in.
        while win.blocks.len() > 1
            && win
                .blocks
                .front()
                .is_some_and(|b| left(win.closed_both, tau - b.origin, width))
        {
            let b = win.blocks.pop_front().expect("a block");
            let g = cfg.decay.factor(tau - b.last);
            add_turned(&mut win.total, &b, cfg, b.origin - win.reference, g, -1.0);
            win.total_w -= g * b.w;
        }
        if win.blocks.len() == 1 {
            // No closed block: the total is nothing, exactly.
            for t in &mut win.total {
                t.zero();
            }
            win.total_w = 0.0;
        }
        // The reference moves to the open block once it is two windows
        // old, and the total is written afresh at it: what the subtractions
        // left in it goes, so nothing grows with the stream.
        if tau - win.reference > 2.0 * width {
            win.reference = win.blocks.back().expect("a block").origin;
            Self::retotal(win, cfg);
        }
        if Self::enforce(win, cfg) {
            Self::retotal(win, cfg);
        }
    }

    /// Hold the blocks to the budget: past a thinning one, adjacent closed
    /// blocks merge in pairs, the younger turned once to the older's
    /// origin, and the cadence doubles; past a refusing one, the overrun
    /// is recorded and the open block takes every row after it. Whether
    /// any merged.
    fn enforce(win: &mut SpectralWindow, cfg: &SpectralCfg) -> bool {
        let Some(budget) = win.budget else {
            return false;
        };
        let limit = budget.mib() * 1024.0 * 1024.0;
        let bytes = |w: &SpectralWindow| w.blocks.iter().map(Block::bytes).sum::<usize>();
        let held = bytes(win);
        if held as f64 <= limit {
            return false;
        }
        let mut merged_any = false;
        match budget {
            crate::WindowBudget::Refuse(_) => {
                win.over.get_or_insert(held);
            }
            crate::WindowBudget::Thin(_) => {
                while bytes(win) as f64 > limit && win.blocks.len() > 2 {
                    let open = win.blocks.pop_back().expect("a block");
                    let mut merged = std::collections::VecDeque::new();
                    while let Some(mut a) = win.blocks.pop_front() {
                        if let Some(b) = win.blocks.pop_front() {
                            merge(&mut a, &b, cfg);
                        }
                        merged.push_back(a);
                    }
                    merged.push_back(open);
                    win.blocks = merged;
                    win.rows = win.rows.saturating_mul(2);
                    win.spacing *= 2.0;
                    merged_any = true;
                }
            }
        }
        merged_any
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

/// Whether a row `age` clock units old has left a window of `width`: a
/// row exactly `width` old has under `closed = "right"`, and not under
/// `"both"`.
fn left(both: bool, age: f64, width: f64) -> bool {
    if both { age > width } else { age >= width }
}

/// `src`'s sums turned by `e^{−iω d}` (`V` by `e^{−2iω d}`), each from
/// `libm` on `d` itself, times `g`, added to `dst` with `sign`.
fn add_turned(dst: &mut [Sums], src: &Block, cfg: &SpectralCfg, d: f64, g: f64, sign: f64) {
    let ls = cfg.lomb_scargle();
    for (k, f) in cfg.frequencies.iter().enumerate() {
        let [c, sn] = rotation(std::f64::consts::TAU * f * d);
        let [c2, s2] = if ls {
            rotation(2.0 * std::f64::consts::TAU * f * d)
        } else {
            [1.0, 0.0]
        };
        let t = |z: [f64; 2], c: f64, sn: f64| {
            let m = sign * g;
            [m * (c * z[0] + sn * z[1]), m * (c * z[1] - sn * z[0])]
        };
        for (out, bs) in dst.iter_mut().zip(&src.sums) {
            let (x, u) = (t(bs.x[k], c, sn), t(bs.u[k], c, sn));
            out.x[k] = [out.x[k][0] + x[0], out.x[k][1] + x[1]];
            out.u[k] = [out.u[k][0] + u[0], out.u[k][1] + u[1]];
            if let (Some(o), Some(&v)) = (out.v.get_mut(k), bs.v.get(k)) {
                let v = t(v, c2, s2);
                *o = [o[0] + v[0], o[1] + v[1]];
            }
        }
    }
    let m = sign * g;
    for (out, bs) in dst.iter_mut().zip(&src.sums) {
        out.a += m * bs.a;
        out.b += m * bs.b;
        out.q += m * g * bs.q;
        out.s += m * bs.s;
    }
}

/// `b`, the younger, merged into `a`: `a` decayed once to `b`'s last row,
/// `b` turned once to `a`'s origin, `e^{−iω(o_b − o_a)}` from `libm` on the
/// difference.
fn merge(a: &mut Block, b: &Block, cfg: &SpectralCfg) {
    let g = cfg.decay.factor(b.last - a.last);
    let d = b.origin - a.origin;
    a.w = g * a.w + b.w;
    for (sa, sb) in a.sums.iter_mut().zip(&b.sums) {
        sa.decay(g, g * g);
        for (k, f) in cfg.frequencies.iter().enumerate() {
            let [c, sn] = rotation(std::f64::consts::TAU * f * d);
            let t = |z: [f64; 2], c: f64, sn: f64| [c * z[0] + sn * z[1], c * z[1] - sn * z[0]];
            let (x, u) = (t(sb.x[k], c, sn), t(sb.u[k], c, sn));
            sa.x[k] = [sa.x[k][0] + x[0], sa.x[k][1] + x[1]];
            sa.u[k] = [sa.u[k][0] + u[0], sa.u[k][1] + u[1]];
            if let (Some(v), Some(&vb)) = (sa.v.get_mut(k), sb.v.get(k)) {
                let [c2, s2] = rotation(2.0 * std::f64::consts::TAU * f * d);
                let z = t(vb, c2, s2);
                *v = [v[0] + z[0], v[1] + z[1]];
            }
        }
        sa.a += sb.a;
        sa.b += sb.b;
        sa.q += sb.q;
        sa.s += sb.s;
    }
    a.last = b.last;
    a.rows += b.rows;
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
        // Every block's sums under a window: what is read afterwards is
        // the rows after the break, the shared weight kept.
        if let Some(win) = self.win.as_mut() {
            for b in &mut win.blocks {
                b.sums.iter_mut().for_each(Sums::zero);
            }
            win.total.iter_mut().for_each(Sums::zero);
        }
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        let mut view = ViewKept::default();
        self.view_into(&mut view);
        out.clear();
        out.extend((0..self.sums.len()).map(|j| self.read_sums(&view, j).a));
        true
    }

    fn set_window_budget(&mut self, budget: Option<crate::WindowBudget>) {
        if let Some(win) = self.win.as_mut() {
            win.budget = budget;
            win.over = None;
            if Self::enforce(win, &self.cfg) {
                Self::retotal(win, &self.cfg);
            }
        }
    }

    fn set_window_closed(&mut self, closed: crate::WindowClosed) {
        if let Some(win) = self.win.as_mut() {
            win.closed_both = closed == crate::WindowClosed::Both;
        }
    }

    fn window_over_budget(&self) -> Option<(usize, crate::Cadence)> {
        let win = self.win.as_ref()?;
        win.over.map(|bytes| {
            (
                bytes,
                crate::Cadence {
                    spacing: win.spacing,
                    rows: win.rows,
                },
            )
        })
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
                        (w.total.is_empty() || w.total.len() == m.cfg.n_targets)
                            && w.total.iter().all(|s| s.len() == per)
                            && w.blocks.iter().all(|b| {
                                b.sums.len() == m.cfg.n_targets
                                    && b.sums.iter().all(|s| s.len() == per)
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
