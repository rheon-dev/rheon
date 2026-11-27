//! Online kernel recursive least squares (`krls`).
//!
//! Fits nonlinear functions of the features in a reproducing kernel Hilbert
//! space (RKHS) in one pass, with bounded per-group state, in two exact modes:
//!
//! 1. **Approximate linear dependence (ALD-KRLS)** (Engel, Mannor & Meir 2004,
//!    with ridge regularization `r > 0` and optional exponential decay):
//!    keeps a bounded dictionary `D = {d_1, ..., d_m}` (`m <= max_atoms`) and
//!    the exact lower-triangular Cholesky factor `L_D` of its kernel matrix
//!    `K_D = L_D L_D^T`. For each row `x`, `z = L_D^{-1} k(D, x)` gives both
//!    the ALD residual `delta = k(x, x) - ||z||^2` and the whitened feature
//!    vector `psi(x)`:
//!
//!    ```text
//!    kv = k(D, x),   L_D z = kv,   delta = k(x, x) - ||z||^2
//!    pred = b + z^T theta                                    (out-of-sample)
//!    admit when m < max_atoms and k(x, x) > 0 and delta > max(nu, 1e-12) * k(x, x):
//!        L_D <- [[L_D, 0], [z^T, sqrt(delta)]],   psi = [z; sqrt(delta)]
//!    otherwise:
//!        psi = z
//!    S_t = lam S_{t-1} + w psi psi^T,   c_t = lam c_{t-1} + w psi y^T
//!    (S_t + r I) theta_t = c_t                               (centered when fit_intercept)
//!    ```
//!
//!    Without decay (`lam = 1`), the factor `L_s` of `S + r I` and rotated RHS
//!    `u = L_s^{-1} c` are updated in `O(m^2)` per row via orthogonal Givens
//!    rotations (for both `fit_intercept = false` and `fit_intercept = true`,
//!    using the exact weighted Welford positive rank-1 update). With decay
//!    (`lam < 1`), `S` and `c` decay multiplicatively and `(S + r I) theta = c`
//!    is solved by packed Cholesky (`O(m^3 / 3)`). Whenever no row is learned
//!    by projection before the last admission, this equals scikit-learn's
//!    `Nystroem(landmarks=D)` followed by `Ridge(alpha=ridge, sample_weight=u)`
//!    to rounding.
//!
//! 2. **Sliding-window KRLS** (Van Vaerenbergh, Via & Santamaria 2006, in
//!    Cholesky form): under `window_rows` (last `W` positive-weight rows) or
//!    `window_size` (last span of the clock under `closed`), every row in the
//!    window is an atom and `L_W L_W^T = K_W + r diag(1 / w)` is maintained in
//!    `O(W^2)` per row by orthogonal Givens rank-1 Cholesky updates on drop and
//!    exact Cholesky extension on add. Without an intercept, this is
//!    `sklearn.kernel_ridge.KernelRidge(alpha=ridge, sample_weight=w)` on the
//!    window's rows to rounding; with `fit_intercept = true`, it eliminates the
//!    unpenalized intercept via `b = (1^T v_y) / (1^T v_1)` and `alpha = v_y -
//!    v_1 b^T`, matching `Nystroem + Ridge(alpha=ridge, fit_intercept=True)`.
//!
//! **Multi-target null policy:** like [`crate::Rls`], the dictionary and factor
//! are shared across targets, so a row with any null target is predict-only for
//! all targets.
//!
//! **Weight scale:** `ridge` is on the sum scale (`S + r I`, `K_W + r W^{-1}`),
//! matching scikit-learn's `KernelRidge(alpha=ridge, sample_weight=w)`.
//!
//! **Bounded state:** ALD mode and `window_rows` fix their maximum bytes at
//! creation and refuse configurations past `max_state_bytes` (default 256 KB,
//! ceiling 64 MiB). `window_size` checks post-trim bytes atomically via
//! [`HeldCap`] before mutating state.

use serde::{Deserialize, Serialize};

use crate::clock_span::{DEFAULT_STATE_BYTES, HeldCap, check_max_state_bytes};
use crate::model::{Extra, ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::{Decay, Stamp, WindowClosed};

mod ald;
mod kernel;
mod window;

pub use ald::ALD_PIVOT_FLOOR;
pub use kernel::{DEFAULT_POLY_COEF0, DEFAULT_POLY_DEGREE, KernelKind};

use ald::{AldScratch, AldState, tri_len};
use kernel::KernelEval;
use window::{WinScratch, WindowState};

/// Default dictionary budget in ALD mode: fits comfortably under the 256 KB
/// per-group state cap (~88 KB at 5 features, 1 target, with intercept).
pub const DEFAULT_MAX_ATOMS: usize = 100;

/// Default relative ALD threshold `nu`: admits a row when
/// `delta > nu * k(x, x)`.
pub const DEFAULT_ALD_THRESHOLD: f64 = 1e-3;

/// Default ridge penalty `r` on the sum scale, matching scikit-learn's
/// `KernelRidge(alpha=1.0)`.
pub const DEFAULT_RIDGE: f64 = 1.0;

/// Configuration for [`Krls`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KrlsCfg {
    pub n_features: usize,
    pub n_targets: usize,
    /// Fit an unpenalized intercept `b` beside the RKHS expansion.
    pub fit_intercept: bool,
    /// Kernel family: `"rbf"` (default), `"laplacian"`, `"poly"`, or `"linear"`.
    pub kernel: KernelKind,
    /// Kernel scale `gamma`; `None` resolves to `1 / n_features` for non-linear
    /// kernels. Refused when `kernel == KernelKind::Linear`.
    pub gamma: Option<f64>,
    /// Polynomial degree (default `3`); refused unless `kernel == KernelKind::Poly`.
    pub degree: Option<u32>,
    /// Polynomial offset `coef0` (default `1.0`, `>= 0`); refused unless
    /// `kernel == KernelKind::Poly`.
    pub coef0: Option<f64>,
    /// Sum-scale ridge penalty `r > 0`, matching `KernelRidge(alpha=ridge)`.
    pub ridge: f64,
    /// Maximum dictionary atoms in ALD mode (`None` -> `100`); refused beside
    /// `window_rows` or `window_size`.
    pub max_atoms: Option<usize>,
    /// Relative ALD admission threshold in `[0, 1)` (`None` -> `1e-3`);
    /// refused beside `window_rows` or `window_size`.
    pub ald_threshold: Option<f64>,
    /// Sliding window of the last this many positive-weight rows; refused
    /// beside `window_clock` or a finite decay.
    pub window_rows: Option<usize>,
    /// Sliding window of the last this many clock units; refused beside
    /// `window_rows` or a finite decay.
    pub window_clock: Option<f64>,
    /// Which edge of `window_clock` holds a row exactly that old.
    pub closed: WindowClosed,
    /// Exponential decay on the clock (`Halflife(inf)` by default).
    pub decay: Decay,
    /// Prediction warm-up weight floor (default `0.0`).
    pub min_weight: f64,
    /// Emit `n_atoms` (`Int64`, pre-row dictionary or window size) in
    /// [`Extra::Krls`].
    pub emit_n_atoms: bool,
    /// Per-group state cap in bytes (`None` -> 256 KB, ceiling 64 MiB).
    pub max_state_bytes: Option<u64>,
}

impl KrlsCfg {
    /// Per-group state byte cap.
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    /// True if configured in sliding-window mode (`window_rows` or `window_clock`).
    pub fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// True if `decay` forgets past rows (finite half-life or `lam < 1`).
    pub fn decays(&self) -> bool {
        match self.decay {
            Decay::Halflife(h) => h.is_finite(),
            Decay::Lam(l) => l < 1.0,
        }
    }

    /// Resolved dictionary cap in ALD mode.
    pub fn max_atoms_or_default(&self) -> usize {
        self.max_atoms.unwrap_or(DEFAULT_MAX_ATOMS)
    }

    /// Resolved relative ALD threshold in ALD mode.
    pub fn ald_threshold_or_default(&self) -> f64 {
        self.ald_threshold.unwrap_or(DEFAULT_ALD_THRESHOLD)
    }

    /// Resolved kernel evaluator.
    fn kernel_eval(&self) -> KernelEval {
        KernelEval {
            kind: self.kernel,
            gamma: self.gamma.unwrap_or(1.0 / self.n_features.max(1) as f64),
            degree: self.degree.unwrap_or(DEFAULT_POLY_DEGREE),
            coef0: self.coef0.unwrap_or(DEFAULT_POLY_COEF0),
        }
    }

    /// Fixed struct overhead in bytes.
    pub fn header_bytes(&self) -> u64 {
        std::mem::size_of::<Krls>() as u64
    }

    /// State bytes in ALD mode holding `m` atoms (including live solution cache
    /// `theta`, `intercept`, and compensated means when `fit_intercept`).
    pub fn ald_bytes(&self, m: usize) -> u64 {
        let m = m as u64;
        let k = self.n_features as u64;
        let t = self.n_targets as u64;
        let int = u64::from(self.fit_intercept);
        // atoms (m*k) + l_d (m(m+1)/2) + mat (m(m+1)/2) + rhs (m*t) + theta (m*t)
        // + psi_bar & psi_bar_lo (2*m*int) + w_target (t) + y_bar/lo/intercept (3*t*int)
        let floats = m
            .saturating_mul(m.saturating_add(1))
            .saturating_add(m.saturating_mul(k))
            .saturating_add(2u64.saturating_mul(m).saturating_mul(t))
            .saturating_add(2u64.saturating_mul(m).saturating_mul(int))
            .saturating_add(t.saturating_mul(1 + 3 * int));
        self.header_bytes().saturating_add(floats.saturating_mul(8))
    }

    /// State bytes in sliding-window mode holding `w` rows (including packed
    /// Cholesky `L_W`, cached dual `alpha` & `intercept`, and 32-byte `Stamp`
    /// per row under `window_clock`).
    pub fn window_bytes(&self, w: usize) -> u64 {
        let w = w as u64;
        let k = self.n_features as u64;
        let t = self.n_targets as u64;
        let stamp = if self.window_clock.is_some() {
            std::mem::size_of::<Stamp>() as u64
        } else {
            0
        };
        // L_W: 8 * w*(w+1)/2 = 4*w*(w+1)
        // per row: x (8*k) + y (8*t) + alpha (8*t) + weight (8) + stamp
        // per target: w_target (8) + intercept (8)
        let tri_bytes = 4u64.saturating_mul(w).saturating_mul(w.saturating_add(1));
        let per_row = 8u64
            .saturating_mul(k)
            .saturating_add(16u64.saturating_mul(t))
            .saturating_add(8)
            .saturating_add(stamp);
        self.header_bytes()
            .saturating_add(tri_bytes)
            .saturating_add(w.saturating_mul(per_row))
            .saturating_add(16u64.saturating_mul(t))
    }

    /// Maximum state bytes fixed by the settings (ALD mode at `max_atoms`,
    /// `window_rows` at full window, or header-only under `window_clock`).
    pub fn fixed_bytes(&self) -> u64 {
        if let Some(w) = self.window_rows {
            self.window_bytes(w)
        } else if self.window_clock.is_some() {
            self.header_bytes()
        } else {
            self.ald_bytes(self.max_atoms_or_default())
        }
    }

    /// Largest `max_atoms` (ALD) or `window_rows` (window) that fits within
    /// `cap`, for actionable refusal messages.
    fn largest_fitting(&self, cap: u64) -> usize {
        let eval = |n: usize| -> u64 {
            if self.window_rows.is_some() {
                self.window_bytes(n)
            } else {
                self.ald_bytes(n)
            }
        };
        if eval(1) > cap {
            return 0;
        }
        let mut lo = 1usize;
        let mut hi = 1usize;
        while hi < 65_536 && eval(hi) <= cap {
            lo = hi;
            hi *= 2;
        }
        while lo + 1 < hi {
            let mid = lo + (hi - lo) / 2;
            if eval(mid) <= cap {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("krls: {e}")));
        no.take(crate::model::check_dims(
            "krls",
            self.n_features,
            self.n_targets,
        ));
        no.take(crate::model::check_min_weight("krls", self.min_weight));

        if !(self.ridge.is_finite() && self.ridge > 0.0) {
            no.refuse(format!(
                "krls: ridge must be finite and above 0, but it is {}. Set ridge to a positive \
                 value such as the default 1.0: kernel RLS requires a positive penalty to keep \
                 its system invertible.",
                self.ridge
            ));
        }

        // Kernel-specific parameters.
        match self.kernel {
            KernelKind::Linear => {
                if self.gamma.is_some() {
                    no.refuse(
                        "krls: gamma does not apply to kernel = \"linear\", which takes the plain \
                         inner product x'y. Leave gamma out, or choose \"rbf\", \"laplacian\" or \
                         \"poly\".",
                    );
                }
                if self.degree.is_some() {
                    no.refuse(
                        "krls: degree only applies to kernel = \"poly\", not \"linear\". Leave \
                         degree out.",
                    );
                }
                if self.coef0.is_some() {
                    no.refuse(
                        "krls: coef0 only applies to kernel = \"poly\", not \"linear\". Leave \
                         coef0 out.",
                    );
                }
            }
            KernelKind::Rbf | KernelKind::Laplacian => {
                if let Some(g) = self.gamma
                    && !(g.is_finite() && g > 0.0)
                {
                    no.refuse(format!(
                        "krls: gamma must be finite and above 0, but it is {g}. Set gamma to a \
                         positive inverse length scale, or leave it out for 1 / len(features)."
                    ));
                }
                if self.degree.is_some() {
                    no.refuse(format!(
                        "krls: degree only applies to kernel = \"poly\", not {:?}. Leave degree \
                         out.",
                        self.kernel.as_str()
                    ));
                }
                if self.coef0.is_some() {
                    no.refuse(format!(
                        "krls: coef0 only applies to kernel = \"poly\", not {:?}. Leave coef0 out.",
                        self.kernel.as_str()
                    ));
                }
            }
            KernelKind::Poly => {
                if let Some(g) = self.gamma
                    && !(g.is_finite() && g > 0.0)
                {
                    no.refuse(format!(
                        "krls: gamma must be finite and above 0, but it is {g}. Set gamma to a \
                         positive scale, or leave it out for 1 / len(features)."
                    ));
                }
                if let Some(d) = self.degree
                    && d == 0
                {
                    no.refuse(
                        "krls: degree must be at least 1, but it is 0. Set degree to a positive \
                         integer such as the default 3.",
                    );
                }
                if let Some(c0) = self.coef0
                    && !(c0.is_finite() && c0 >= 0.0)
                {
                    no.refuse(format!(
                        "krls: coef0 must be finite and at least 0, but it is {c0}: a negative \
                         offset breaks positive definiteness of the polynomial kernel. Set coef0 \
                         to 0 or above, such as the default 1.0."
                    ));
                }
            }
        }

        // Window and ALD checks.
        if self.window_rows.is_some() && self.window_clock.is_some() {
            no.refuse(
                "krls: window_rows and window_size are each a window, and only one can decide \
                 which rows are fitted. Give window_rows or window_size, not both.",
            );
        }
        if self.closed != WindowClosed::Right && self.window_clock.is_none() {
            no.refuse(
                "krls: closed = \"both\" names the edge of window_size, which is not given. Give \
                 window_size, or leave closed out.",
            );
        }
        if self.window_rows == Some(0) {
            no.refuse(
                "krls: window_rows must be at least 1, but it is 0. Give the number of rows the \
                 window keeps, or leave it out for ALD mode.",
            );
        }
        if let Some(d) = self.window_clock
            && !(d.is_finite() && d > 0.0)
        {
            no.refuse(format!(
                "krls: window_size must be finite and above 0, but it is {d}. Give the span of \
                 the clock the window keeps, or leave it out for ALD mode."
            ));
        }

        if self.windowed() {
            if self.decay.check().is_ok() && self.decays() {
                no.refuse(
                    "krls: a window forgets by dropping its oldest rows, so it takes no decay, \
                     but half_life (or lam) decays too. Set half_life to inf (or leave it out), \
                     or leave out the window.",
                );
            }
            if self.max_atoms.is_some() {
                no.refuse(
                    "krls: max_atoms does not apply beside a window, which keeps every learned \
                     row inside it as an atom. Leave max_atoms out, or leave out the window.",
                );
            }
            if self.ald_threshold.is_some() {
                no.refuse(
                    "krls: ald_threshold does not apply beside a window, which keeps every \
                     learned row inside it as an atom. Leave ald_threshold out, or leave out the \
                     window.",
                );
            }
        } else {
            if self.max_atoms == Some(0) {
                no.refuse(
                    "krls: max_atoms must be at least 1, but it is 0. Give the maximum number of \
                     dictionary atoms, such as the default 100.",
                );
            }
            if let Some(nu) = self.ald_threshold
                && !(nu.is_finite() && (0.0..1.0).contains(&nu))
            {
                no.refuse(format!(
                    "krls: ald_threshold must be finite, at least 0 and below 1, but it is {nu}. \
                     Set ald_threshold to a small relative tolerance such as the default 1e-3."
                ));
            }
        }

        no.take(check_max_state_bytes("krls", self.max_state_bytes));
        if self.n_features > 0
            && self.n_targets > 0
            && check_max_state_bytes("krls", self.max_state_bytes).is_ok()
        {
            let cap = self.cap();
            let most = self.fixed_bytes();
            if most > cap {
                if self.window_clock.is_some() {
                    no.refuse(format!(
                        "krls: its fixed state holds {most} bytes a group, past max_state_bytes \
                         {cap}. Raise max_state_bytes."
                    ));
                } else {
                    let param = if self.window_rows.is_some() {
                        "window_rows"
                    } else {
                        "max_atoms"
                    };
                    let fits = self.largest_fitting(cap);
                    if fits > 0 {
                        no.refuse(format!(
                            "krls: a group's state may hold {most} bytes, past max_state_bytes \
                             {cap}. Lower {param} to at most {fits} (fits in {cap} bytes), or \
                             raise max_state_bytes."
                        ));
                    } else {
                        no.refuse(format!(
                            "krls: a group's state may hold {most} bytes, past max_state_bytes \
                             {cap}. Raise max_state_bytes, or fit fewer features or targets in \
                             this spec."
                        ));
                    }
                }
            }
        }

        no.finish()
    }
}

/// Clock bookkeeping for sliding clock windows (`window_clock`), matching
/// [`crate::Isotonic`]'s exact stamp rules (and preserving `on_rows` on
/// `Clock::Rows`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Clock {
    own: f64,
    counted: u64,
    read: Option<Stamp>,
    on_rows: bool,
}

/// Internal mode state: ALD dictionary or sliding window.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum Mode {
    Ald(AldState),
    Window(WindowState),
}

/// Reusable scratch buffers (never serialized).
#[derive(Debug, Clone, Default)]
struct Scratch {
    ald: AldScratch,
    win: WinScratch,
}

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Online kernel recursive least squares (module docs state the equations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Krls {
    cfg: KrlsCfg,
    mode: Mode,
    /// Accumulated weight before the current row's update and decay (`n_eff`).
    w_sum: f64,
    /// Per-target weight of rows the fit learned from (all targets move
    /// together because the kernel design is shared, as in [`crate::Rls`]).
    w_target: Vec<f64>,
    clock: Clock,
    /// Bytes past `max_state_bytes` that triggered refusal in `window_clock`.
    over: Option<u64>,
    /// Whether the half-cap warning has already been emitted for this group.
    warned: bool,
    /// Next row's stamp and whether it came from `Clock::Rows`. Never state.
    #[serde(skip)]
    next: Option<(Stamp, bool)>,
    /// Pending half-cap warning waiting for the caller to drain. Never state.
    #[serde(skip)]
    warning: Option<u64>,
    #[serde(skip)]
    scratch: Scratch,
}

impl Krls {
    pub fn new(cfg: KrlsCfg) -> Result<Self, String> {
        cfg.validate()?;
        let n_targets = cfg.n_targets;
        let mode = if cfg.windowed() {
            Mode::Window(WindowState::new(n_targets))
        } else {
            Mode::Ald(AldState::new(&cfg))
        };
        Ok(Self {
            cfg,
            mode,
            w_sum: 0.0,
            w_target: vec![0.0; n_targets],
            clock: Clock::default(),
            over: None,
            warned: false,
            next: None,
            warning: None,
            scratch: Scratch::default(),
        })
    }

    pub fn cfg(&self) -> &KrlsCfg {
        &self.cfg
    }

    /// Accumulated weight before the next row's update (`n_eff`), or window
    /// weight sum under a window.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Per-target weight of rows learned (all equal because the dictionary is
    /// shared across targets).
    pub fn target_weights(&self) -> &[f64] {
        &self.w_target
    }

    /// Current number of dictionary atoms (ALD) or rows held (window).
    pub fn n_atoms(&self) -> usize {
        match &self.mode {
            Mode::Ald(a) => a.n_atoms(self.cfg.n_features),
            Mode::Window(w) => w.len(),
        }
    }

    /// True when the dictionary / window is empty so no prediction is defined.
    pub fn not_identified(&self) -> bool {
        self.n_atoms() == 0
    }

    /// Current state size in bytes.
    pub fn bytes(&self) -> u64 {
        match &self.mode {
            Mode::Ald(a) => self.cfg.ald_bytes(a.n_atoms(self.cfg.n_features)),
            Mode::Window(w) => self.cfg.window_bytes(w.len()),
        }
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    fn window_key(&self, now: Option<(Stamp, bool)>, d_clock: f64, counting: bool) -> Stamp {
        match now {
            Some((_, true)) => Stamp::from((self.clock.counted + u64::from(counting)) as f64),
            Some((s, false)) => s,
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    fn key_ahead(&self, d_clock: f64) -> Stamp {
        match self.clock.read {
            _ if self.clock.on_rows => Stamp::from(self.clock.counted as f64 + 1.0),
            Some(last) => crate::since::back_by(last, -d_clock),
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    fn note_clock(&mut self, handed: Option<(Stamp, bool)>, counting: bool) {
        if self.cfg.window_clock.is_none() {
            return;
        }
        self.clock.counted += u64::from(counting);
        if let Some((s, rows)) = handed {
            self.clock.read = Some(s);
            self.clock.on_rows = rows;
        }
    }

    #[inline]
    fn write_extra(&self, out: &mut Step) {
        out.extra = if self.cfg.emit_n_atoms {
            Some(Extra::Krls {
                n_atoms: self.n_atoms() as f64,
            })
        } else {
            None
        };
    }

    /// Read prediction from current state into `out` (allocates nothing when
    /// scratch is supplied).
    fn read_current_into(&self, x: &[f64], kv: &mut Vec<f64>, z: &mut Vec<f64>, out: &mut Step) {
        let kern = self.cfg.kernel_eval();
        let k = self.cfg.n_features;
        let t = self.cfg.n_targets;
        match &self.mode {
            Mode::Ald(a) => {
                let m = a.n_atoms(k);
                kern.eval_slice(&a.atoms, m, k, x, kv);
                if kv.iter().all(|v| v.is_finite()) {
                    ald::forward_solve_lower(&a.l_d, m, kv, z);
                } else {
                    z.clear();
                    z.resize(m, f64::NAN);
                }
                a.predict_from_z(z, &self.w_target, self.cfg.min_weight, t, &mut out.pred);
            }
            Mode::Window(w) => {
                w.predict_into(
                    kern,
                    x,
                    &self.w_target,
                    self.cfg.min_weight,
                    k,
                    t,
                    kv,
                    &mut out.pred,
                );
            }
        }
        out.n_eff = self.w_sum;
        self.write_extra(out);
    }

    /// Read prediction at stamp `key` without mutating `self` (trims expired
    /// clock-window rows on a clone only when at least one row has left).
    fn read_at_key_into(&self, key: Stamp, x: &[f64], out: &mut Step) {
        if let Mode::Window(w) = &self.mode {
            let expired = w.expired_count(&self.cfg, key);
            if expired > 0 {
                let mut at = self.clone();
                if let Mode::Window(aw) = &mut at.mode {
                    let mut sc = WinScratch::default();
                    for _ in 0..expired {
                        aw.drop_oldest(at.cfg.n_features, at.cfg.n_targets, &mut sc.drop_v);
                    }
                    aw.solve(&at.cfg, &mut sc);
                    at.w_sum = aw.weight_sum();
                    at.w_target.fill(at.w_sum);
                    at.read_current_into(x, &mut sc.kv, &mut sc.z, out);
                }
                return;
            }
        }
        let mut kv = Vec::new();
        let mut z = Vec::new();
        self.read_current_into(x, &mut kv, &mut z, out);
    }

    /// [`OnlineModel::predict`] for a row whose exact clock `stamp` is known
    /// (zero-weight rows and `predict` on a bank under `window_size`).
    /// Preserves `on_rows` semantics on `Clock::Rows`.
    pub fn predict_at(&self, x: &[f64], d_clock: f64, stamp: Stamp) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        if self.cfg.window_clock.is_some() {
            let key = if self.clock.on_rows {
                Stamp::from(self.clock.counted as f64 + 1.0)
            } else {
                stamp
            };
            self.read_at_key_into(key, x, &mut out);
        } else {
            let mut kv = Vec::new();
            let mut z = Vec::new();
            self.read_current_into(x, &mut kv, &mut z, &mut out);
        }
        out
    }

    /// [`OnlineModel::predict_with_floor`] at an exact `stamp` (for diagnostics
    /// reading predictions below `min_weight` under a `window_size` window).
    pub fn predict_with_floor_at(
        &mut self,
        x: &[f64],
        _y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
        stamp: Stamp,
    ) -> Step {
        let prev = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = self.predict_at(x, d_clock, stamp);
        self.cfg.min_weight = prev;
        out
    }

    fn has_shape(&self) -> bool {
        let t = self.cfg.n_targets;
        let weights_ok = self.w_sum.is_finite()
            && self.w_sum >= 0.0
            && self.w_target.len() == t
            && self.w_target.iter().all(|w| w.is_finite() && *w >= 0.0);
        if !weights_ok || !self.clock.own.is_finite() {
            return false;
        }
        let mode_ok = match (&self.mode, self.cfg.windowed()) {
            (Mode::Ald(a), false) => a.has_shape(&self.cfg),
            (Mode::Window(w), true) => w.has_shape(&self.cfg),
            _ => false,
        };
        mode_ok && self.bytes() <= self.cfg.cap()
    }
}

impl OnlineModel for Krls {
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
        let handed = self.next.take();
        let kern = self.cfg.kernel_eval();
        let k = self.cfg.n_features;
        let t_cnt = self.cfg.n_targets;

        // Zero-weight row (or subnormal weight below normal resolution):
        // counts in no clock and learns nothing. Directly handed to ALD with
        // decay, decays weights and accumulators without 0/0 or NaN.
        if weight < f64::MIN_POSITIVE {
            if self.cfg.window_clock.is_some() {
                let key = self.window_key(handed, d_clock, false);
                self.read_at_key_into(key, x, out);
                return;
            }
            let mut scratch = std::mem::take(&mut self.scratch);
            self.read_current_into(x, &mut scratch.ald.kv, &mut scratch.ald.z, out);
            if let Mode::Ald(a) = &mut self.mode
                && self.cfg.decays()
            {
                let lam = self.cfg.decay.factor(d_clock);
                self.w_sum *= lam;
                for wt in &mut self.w_target {
                    *wt *= lam;
                }
                a.decay_only(&self.cfg, lam);
            }
            self.scratch = scratch;
            return;
        }

        let learns = y.iter().take(t_cnt).all(Option::is_some);

        match &self.mode {
            Mode::Ald(_) => {
                let lam = self.cfg.decay.factor(d_clock);
                let mut scratch = std::mem::take(&mut self.scratch);

                // Predict before update (populates `scratch.ald.z = L_D^{-1} kv`
                // so `learn` reuses it with zero duplicate work!).
                self.read_current_into(x, &mut scratch.ald.kv, &mut scratch.ald.z, out);

                let w_old_learned = self.w_target.first().copied().unwrap_or(0.0);
                self.w_sum = lam * self.w_sum + weight;
                if learns {
                    let ok = if let Mode::Ald(a) = &mut self.mode {
                        a.learn(
                            &self.cfg,
                            kern,
                            x,
                            y,
                            weight,
                            lam,
                            w_old_learned,
                            &mut scratch.ald,
                        )
                    } else {
                        unreachable!()
                    };
                    if ok {
                        crate::model::age_target_weights(&mut self.w_target, |_| true, lam, weight);
                    } else {
                        // Overflow guard tripped: decay cleanly without poisoning state.
                        crate::model::age_target_weights(&mut self.w_target, |_| false, lam, 0.0);
                        if let Mode::Ald(a) = &mut self.mode {
                            a.decay_only(&self.cfg, lam);
                        }
                    }
                } else {
                    crate::model::age_target_weights(&mut self.w_target, |_| false, lam, 0.0);
                    if let Mode::Ald(a) = &mut self.mode {
                        a.decay_only(&self.cfg, lam);
                    }
                }
                self.scratch = scratch;
            }
            Mode::Window(w_ro) => {
                let key = self.window_key(handed, d_clock, true);
                let expired = w_ro.expired_count(&self.cfg, key);
                let cap = self.cfg.cap();
                let post_len = w_ro.len() - expired + usize::from(learns);
                let bytes_after = self.cfg.window_bytes(post_len);

                // Atomic HeldCap check BEFORE any mutation of clock, ring, or Cholesky.
                if self.over.is_some()
                    || (learns && self.cfg.window_clock.is_some() && bytes_after > cap)
                {
                    self.read_at_key_into(key, x, out);
                    if self.over.is_none() {
                        let _ = self.held_cap().admit(bytes_after, cap);
                    }
                    return;
                }

                // Happy path: mutate in place without cloning.
                self.clock.own += d_clock;
                self.note_clock(handed, true);
                let mut scratch = std::mem::take(&mut self.scratch);
                if expired > 0
                    && let Mode::Window(w) = &mut self.mode
                {
                    for _ in 0..expired {
                        w.drop_oldest(k, t_cnt, &mut scratch.win.drop_v);
                    }
                    w.solve(&self.cfg, &mut scratch.win);
                    self.w_sum = w.weight_sum();
                    self.w_target.fill(self.w_sum);
                }

                // Out-of-sample prediction before learning this row.
                self.read_current_into(x, &mut scratch.win.kv, &mut scratch.win.z, out);

                if learns && kern.eval_self(x).is_finite() && (self.cfg.ridge / weight).is_finite()
                {
                    let mut pushed = false;
                    if let Mode::Window(w) = &mut self.mode {
                        if self
                            .cfg
                            .window_rows
                            .is_some_and(|cap_rows| w.len() >= cap_rows)
                        {
                            w.drop_oldest(k, t_cnt, &mut scratch.win.drop_v);
                        }
                        let st = self.cfg.window_clock.map(|_| key);
                        pushed = w.push_row(&self.cfg, kern, x, y, weight, st, &mut scratch.win);
                        w.solve(&self.cfg, &mut scratch.win);
                        self.w_sum = w.weight_sum();
                        self.w_target.fill(self.w_sum);
                    }
                    if pushed && self.cfg.window_clock.is_some() {
                        let now_bytes = self.bytes();
                        self.held_cap().note(now_bytes, cap);
                    }
                }
                self.scratch = scratch;
            }
        }
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        if self.cfg.window_clock.is_some() {
            let key = self.key_ahead(d_clock);
            self.read_at_key_into(key, x, &mut out);
        } else {
            let mut kv = Vec::new();
            let mut z = Vec::new();
            self.read_current_into(x, &mut kv, &mut z, &mut out);
        }
        out
    }

    fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> Step {
        let prev = std::mem::replace(&mut self.cfg.min_weight, floor);
        let out = OnlineModel::predict_with(self, x, y, d_clock);
        self.cfg.min_weight = prev;
        out
    }

    fn stamp_next(&mut self, stamp: Stamp) {
        self.next = Some((stamp, false));
    }

    fn stamp_next_on_rows(&mut self, stamp: Stamp) {
        self.next = Some((stamp, true));
    }

    fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        self.warning.take().map(|bytes| (bytes, self.cfg.cap()))
    }

    fn state_over_cap(&self) -> Option<(u64, u64)> {
        self.over.map(|bytes| (bytes, self.cfg.cap()))
    }

    fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        out.clear();
        out.extend_from_slice(&self.w_target);
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::Krls(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Krls(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("krls", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "krls",
                        "its running sums have the wrong shape",
                    ));
                }
                match &mut m.mode {
                    Mode::Ald(a) => {
                        let n = a.n_atoms(m.cfg.n_features);
                        let mut buf = Vec::with_capacity(tri_len(n));
                        let mut tmp = Vec::with_capacity(n);
                        a.solve(&m.cfg, &mut buf, &mut tmp);
                    }
                    Mode::Window(w) => {
                        let mut sc = WinScratch::default();
                        w.solve(&m.cfg, &mut sc);
                    }
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "krls",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }
}

#[cfg(test)]
#[path = "krls/tests.rs"]
mod tests;
