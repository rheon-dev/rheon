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
//! The cost is `O(F)` a row for `F` frequencies, one `sin` and one `cos`
//! each (none where the clock's step is the last one's), and the state
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

/// What a [`Spectral`] reports at each frequency: scipy's `scaling`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpectralScaling {
    /// Power spectral density, `|X|² spacing / Σa²`: scipy's default.
    #[default]
    Density,
    /// Power spectrum, `|X|² / (Σa)²`.
    Spectrum,
}

/// The settings of a [`Spectral`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpectralCfg {
    pub n_targets: usize,
    /// The frequencies, in cycles per clock unit, each at least 0 and at
    /// most the Nyquist frequency `1 / (2 spacing)`.
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
}

/// The most frequencies a spectrum keeps.
pub const MAX_FREQUENCIES: usize = 4096;

impl SpectralCfg {
    pub fn cap(&self) -> u64 {
        self.max_state_bytes
            .unwrap_or(crate::clock_span::DEFAULT_STATE_BYTES)
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
        if let Some(f) = self
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
    a: f64,
    b: f64,
    q: f64,
}

impl Sums {
    fn new(f: usize) -> Self {
        Self {
            x: vec![[0.0; 2]; f],
            u: vec![[0.0; 2]; f],
            a: 0.0,
            b: 0.0,
            q: 0.0,
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
}

impl Spectral {
    pub fn new(cfg: SpectralCfg) -> Result<Self, String> {
        cfg.validate()?;
        let f = cfg.frequencies.len();
        Ok(Self {
            sums: (0..cfg.n_targets).map(|_| Sums::new(f)).collect(),
            w_sum: 0.0,
            rot: None,
            cfg,
        })
    }

    pub fn cfg(&self) -> &SpectralCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The bytes the state holds: `8 (4F + 3)` a target and 8 shared.
    pub fn state_bytes(cfg: &SpectralCfg) -> u64 {
        let f = cfg.frequencies.len() as u64;
        (8 * (4 * f + 3)).saturating_mul(cfg.n_targets as u64) + 8
    }

    /// The rotation `e^{2πi f Δ}` of each frequency for a clock step `d`,
    /// from the platform's `sin` and `cos`.
    fn rotations(&self, d: f64) -> Vec<[f64; 2]> {
        self.cfg
            .frequencies
            .iter()
            .map(|f| {
                let th = std::f64::consts::TAU * f * d;
                [th.cos(), th.sin()]
            })
            .collect()
    }

    /// The power at every frequency of target `j`, written into `out`.
    fn power_into(&self, j: usize, out: &mut [f64]) {
        let s = &self.sums[j];
        let open = s.q > 0.0 && s.a > 0.0;
        let m = if self.cfg.detrend && open {
            s.b / s.a
        } else {
            0.0
        };
        let scale = match self.cfg.scaling {
            SpectralScaling::Density => self.cfg.spacing / s.q,
            SpectralScaling::Spectrum => 1.0 / (s.a * s.a),
        };
        let nyq = self.cfg.nyquist();
        for (k, f) in self.cfg.frequencies.iter().enumerate() {
            if !open {
                out[k] = f64::NAN;
                continue;
            }
            let re = s.x[k][0] - m * s.u[k][0];
            let im = s.x[k][1] - m * s.u[k][1];
            let c = if *f == 0.0 || *f == nyq { 1.0 } else { 2.0 };
            out[k] = c * (re * re + im * im) * scale;
        }
    }

    fn read(&self, out: &mut Step) {
        let f = self.cfg.frequencies.len();
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets * f, f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        for j in 0..self.cfg.n_targets {
            if self.sums[j].a >= self.cfg.min_weight {
                self.power_into(j, &mut out.pred[j * f..(j + 1) * f]);
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
            for k in 0..rot.len() {
                let [c, si] = rot[k];
                let [xr, xi] = s.x[k];
                let [ur, ui] = s.u[k];
                s.x[k] = [lam * (c * xr - si * xi) + wx, lam * (c * xi + si * xr)];
                s.u[k] = [lam * (c * ur - si * ui) + w, lam * (c * ui + si * ur)];
            }
            s.a = lam * s.a + w;
            s.b = lam * s.b + wx;
            s.q = lam * lam * s.q + w * w;
        }
    }

    /// Each target's power at every frequency after the last row, `F` each.
    pub fn powers(&self) -> Vec<Vec<f64>> {
        let f = self.cfg.frequencies.len();
        (0..self.cfg.n_targets)
            .map(|j| {
                let mut v = vec![f64::NAN; f];
                self.power_into(j, &mut v);
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
        self.read(out);
        let lam = self.cfg.decay.factor(d_clock);
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
        self.read(&mut out);
        out
    }

    /// A break: the phase across it is unknown, so each target's sums go;
    /// the shared weight stays.
    fn clear_lags(&mut self) {
        let f = self.cfg.frequencies.len();
        for s in &mut self.sums {
            *s = Sums::new(f);
        }
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend(self.sums.iter().map(|s| s.a));
        true
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
                let shaped = m.sums.len() == m.cfg.n_targets
                    && m.sums.iter().all(|s| s.x.len() == f && s.u.len() == f);
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
#[path = "spectral_tests.rs"]
mod tests;
