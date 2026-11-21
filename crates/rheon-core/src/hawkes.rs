//! A self-exciting point process (Hawkes) fitted online by least squares,
//! exactly, one row at a time.
//!
//! Each target is a node `i = 1..D`, and a row is an instant on the clock: a
//! target's value `m > 0` on the row is an event of that node, of mark `m`
//! (1 for an event, a count for that many at once, or a size); a value of 0
//! or below, or none, is no event. The intensity of node `i` is
//!
//! ```text
//! λ_i(t) = μ_i + Σ_j Σ_u α_iju β_u Σ_{events t_k of j before t} m_k e^(−β_u (t − t_k))
//! ```
//!
//! with the kernel decays `β_u`, `u = 1..U`, given (`kernel_decays`), so the
//! kernel `α β e^(−β t)` integrates to `α`. With the excitation per source
//! node and decay, `R_ju(t) = Σ_{t_k of j < t} β_u m_k e^(−β_u (t − t_k))`, and
//! `φ(t) = (1, R_11, .., R_DU)` (`p = 1 + k` terms, `k = D U`), the intensity is
//! `λ_i = θ_iᵀ φ` with `θ_i = (μ_i, α_i11, .., α_iDU)`: linear in the
//! parameters. The fit minimises the least-squares contrast, each instant
//! weighted `w(t) = e^(−γ (T − t))` (`γ` the forgetting rate per clock unit,
//! `ln 2 / half_life`; 0 forgets nothing):
//!
//! ```text
//! L(θ) = Σ_i [ ∫ w λ_i² dt − 2 ∫ w λ_i dN_i ] = Σ_i [ θ_iᵀ G θ_i − 2 θ_iᵀ b_i ]
//! G   = ∫ w φ φᵀ dt          b_i = Σ_{events of i} w(t_k) m_k φ(t_k−)
//! θ_i = G⁻¹ b_i
//! ```
//!
//! `φ(t_k−)` is the left limit: an event is not in its own intensity, nor
//! in that of an event on the same row. An event on an earlier row is, at
//! the same clock or not.
//!
//! **The recursion.** On a row `d` clock units after the last, with `f =
//! e^(−γ d)`:
//!
//! ```text
//! R_ju ← e^(−β_u d) R_ju        b ← f b        Q ← f Q
//! G₀₀ ← f G₀₀ + (1 − e^(−γ d)) / γ   (d where γ = 0)
//! ```
//!
//! then, for each node `j` with an event of mark `m_j`, from `R` as it
//! stands before any of the row's events:
//!
//! ```text
//! b_j ← b_j + m_j φ(t−)          Q_jl ← Q_jl + m_j m_l          R_ju ← R_ju + m_j β_u
//! ```
//!
//! `G` is not kept. Between events `R_a R_b` decays at `β_a + β_b`, and at an
//! row with events `m` it jumps by `m_j(a) β_a R_b + m_j(b) β_b R_a + m_j(a)
//! m_j(b) β_a β_b`; summed over the rows, `Σ m_j R_b(t−)` is `b_j`'s entry
//! for `b`, and `Σ m_j m_l` is `Q_jl` (`l = j` but where two nodes have an
//! event on one row). Integrating `w d(R_a R_b)` and
//! `w dR_a` by parts then gives `G` from the state, at any read:
//!
//! ```text
//! G₀ₐ = (β_a b_{j(a)}[0] − R_a) / (β_a − γ)
//! Gₐᵦ = (β_a b_{j(a)}[b] + β_b b_{j(b)}[a] + β_a β_b Q_{j(a) j(b)} − R_a R_b)
//!       / (β_a + β_b − γ)
//! ```
//!
//! So the state is `R` (`k`), `b` (`D p`), `Q` (symmetric, `D (D + 1) / 2`),
//! `G₀₀` and the weight: `8 (k + D (1 + k) + D (D + 1) / 2 + 2)` bytes, whatever the stream's length, and a row
//! costs `U` exponentials and `O(D k)` additions to learn. A read assembles `G`
//! in `O(k²)` and solves it by Cholesky in `O(p³)`, once a row, for the
//! intensities the row reports. `γ` within 0.1% of a decay `β_u` or of a sum
//! `β_u + β_v` is refused: those divisions lose about `log10(β / |β − γ|)`
//! digits, and the exact formula at equality is another one.
//!
//! **What a row reports.** Each node's intensity `λ_i(t−)` before the row's
//! events, from the fit on everything before them (hard rule 2): `G` read to
//! the row's clock, which uses only that no event came since the last row,
//! and `b` without the row's events. `NaN` while `weight_sum` is under
//! `min_weight`, or while `G` is not positive definite (before every node
//! has had an event, or before any clock has passed), or is so near
//! singular that the events cannot tell two of its terms apart: a pivot of
//! `G` scaled to a unit diagonal below `1e-10`, a condition number above
//! `1e10`, as when two nodes' events fall on the same rows. `coefficients` reads
//! `θ_i` the same way. The fit is unconstrained: a parameter, and an
//! intensity, can be negative where the data say so.
//!
//! **The weight.** A row's weight decides only whether its events are
//! learned: 0 advances the clock and learns nothing (hard rule 9), and any
//! weight above 0 learns the row's events at their marks. An event's
//! excitation is a fact of the process, not evidence to scale, and `G`'s
//! event part is read from `b` only because the same marks build both.
//! `weight_sum` is the decayed sum of the rows' weights, every row's, as in
//! every model (hard rule 8).
//!
//! **The window** (`window_events = W`). The fit is that of the last `W`
//! rows that carried an event, alone, observed from the clock of the row
//! dropped last (from the first row until one is): excitation by an event
//! before the window is dropped too. Dropping the oldest row `e` takes back
//! exactly what it added:
//!
//! ```text
//! b_j[0] −= m_e,j      Q_jl −= m_e,j m_e,l      R_ju −= m_e,j β_u e^(−β_u (T − t_e))
//! b_i[(j, u)] −= β_u m_e,j F_e[i, u],    F_e[i, u] = Σ_{later rows l} m_l,i e^(−β_u (t_l − t_e))
//! G₀₀ = T − t_e
//! ```
//!
//! `F_e` cannot be kept as one running sum: rescaled as `t_e` moves, it would
//! multiply its rounding by up to `e^(β W)`. The window keeps its rows in two
//! blocks. The older holds, for each row, `F` over the later rows of its own
//! block, formed once by the backward recursion `F_r = e^(−β (t_{r+1} − t_r))
//! (m_{r+1} + F_{r+1})`; the newer keeps `B[i, u] = Σ m_l,i e^(−β_u (t_l −
//! t_m))` from its own start `t_m`, so `F_e = F_e^block + e^(−β (t_m − t_e)) B`,
//! every factor at most 1. When the older block runs out, the newer becomes
//! it, and `R`, `b` and `Q` are formed afresh from its rows, so their
//! rounding is that of one window, not of every row the stream ever dropped.
//! A row costs `O(D k)` amortised; the window holds `8 (1 + D)` bytes a row
//! in the newer block and `8 (1 + D + k)` in the older, at most `8 W (1 + D +
//! k)` beside the state. A window forgets by dropping, so it takes no decay.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::solve::{KEPT_ORDER, Solves, solve_definite_in};

/// The least pivot of `G` scaled to a unit diagonal that a fit is read
/// at: below it `G`'s condition number is above `1e10`, and the fit has
/// lost all but a few of its digits to two terms the events cannot tell
/// apart -- two nodes whose events fall on the same rows, or a decay the
/// data cannot separate from another.
const PIVOT_FLOOR: f64 = 1e-10;

/// How near the forgetting rate may come to a kernel decay, or to a sum of
/// two, relative to it: the read divides by `β − γ` and `β_u + β_v − γ`.
const RATE_MARGIN: f64 = 1e-3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HawkesCfg {
    /// The nodes, one per target.
    pub n_targets: usize,
    /// The kernels' decays `β_u`, per clock unit: each node excites every
    /// node through one kernel `α β e^(−β t)` per decay.
    pub kernel_decays: Vec<f64>,
    /// The forgetting, per clock unit.
    pub decay: Decay,
    /// `weight_sum` before a row reports.
    pub min_weight: f64,
    /// Fit the last this many rows that carried an event, alone.
    pub window_events: Option<usize>,
}

/// `γ`, the forgetting rate per clock unit that `decay` gives.
fn rate(decay: Decay) -> f64 {
    match decay {
        Decay::Halflife(h) if h.is_infinite() => 0.0,
        Decay::Halflife(h) => std::f64::consts::LN_2 / h,
        Decay::Lam(1.0) => 0.0,
        Decay::Lam(l) => -l.ln(),
    }
}

impl HawkesCfg {
    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        if self.n_targets == 0 {
            no.refuse(
                "hawkes: the model needs at least one target, one per node, but it has none. \
                 Name at least one column in targets.",
            );
        }
        let betas = &self.kernel_decays;
        if betas.is_empty() {
            no.refuse(
                "hawkes: kernel_decays must list at least one decay, but it lists none. Give \
                 the decay of each exponential kernel, per clock unit.",
            );
        }
        let bad: Vec<String> = betas
            .iter()
            .filter(|b| !(b.is_finite() && **b > 0.0))
            .map(f64::to_string)
            .collect();
        if !bad.is_empty() {
            no.refuse(format!(
                "hawkes: each of kernel_decays must be a finite number above 0, but {} {}. \
                 Give each kernel's decay as a positive number per clock unit.",
                if bad.len() == 1 {
                    "one is"
                } else {
                    "these are not:"
                },
                crate::model::listed(&bad),
            ));
        }
        let mut sorted: Vec<f64> = betas.clone();
        sorted.sort_by(f64::total_cmp);
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            no.refuse(format!(
                "hawkes: kernel_decays must be distinct, but {} lists one twice. Two equal \
                 decays are one kernel, and their parameters cannot be told apart; list each \
                 decay once.",
                crate::model::listed(&betas.iter().map(f64::to_string).collect::<Vec<_>>()),
            ));
        }
        if let Err(e) = self.decay.check() {
            no.refuse(format!("hawkes: {e}"));
        } else if bad.is_empty() {
            let gamma = rate(self.decay);
            let near = |s: f64| gamma > 0.0 && (s - gamma).abs() <= RATE_MARGIN * s;
            let mut close: Vec<String> = Vec::new();
            for (u, &bu) in betas.iter().enumerate() {
                if near(bu) {
                    close.push(format!("{bu}"));
                }
                for &bv in &betas[u..] {
                    if near(bu + bv) {
                        close.push(format!("{bu} + {bv}"));
                    }
                }
            }
            if !close.is_empty() {
                no.refuse(format!(
                    "hawkes: the forgetting rate, {gamma} per clock unit (ln 2 / half_life), is \
                     within 0.1% of {}, where the fit divides by the difference and loses its \
                     digits. Set half_life well above the kernels' memory (1 / the smallest \
                     decay), or to inf to forget nothing.",
                    crate::model::listed(&close),
                ));
            }
        }
        no.take(crate::model::check_min_weight("hawkes", self.min_weight));
        if let Some(w) = self.window_events {
            if w == 0 {
                no.refuse(
                    "hawkes: window_events must be at least 1, but it is 0. Give the number \
                     of rows with an event the fit keeps, or leave it out for no window.",
                );
            }
            if self.decay.check().is_ok() && rate(self.decay) > 0.0 {
                no.refuse(
                    "hawkes: window_events forgets by dropping the oldest events, so it takes \
                     no decay, but half_life (or lam) decays too. Set half_life to inf, or \
                     leave out window_events.",
                );
            }
        }
        no.finish()
    }

    fn dims(&self) -> (usize, usize, usize) {
        let k = self.n_targets * self.kernel_decays.len();
        (self.n_targets, self.kernel_decays.len(), k)
    }
}

/// The rows a window keeps, in two blocks (see the module docs). Times are
/// held relative to the newer block's start `t_m`, never as the clock's sum.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct EventWindow {
    /// The clock since `t_m`, now.
    since: f64,
    /// The older block, oldest first: each row's `t_m − t_r`, its `D` marks
    /// and its `k` forward sums over the later rows of the block.
    old_off: VecDeque<f64>,
    old_marks: VecDeque<f64>,
    old_fwd: VecDeque<f64>,
    /// The newer block, oldest first: each row's `t_r − t_m` and its marks.
    new_at: Vec<f64>,
    new_marks: Vec<f64>,
    /// `B[i, u] = Σ_newer m_l,i e^(−β_u (t_l − t_m))`, `k`.
    new_sum: Vec<f64>,
}

impl EventWindow {
    fn new(k: usize) -> Self {
        Self {
            since: 0.0,
            old_off: VecDeque::new(),
            old_marks: VecDeque::new(),
            old_fwd: VecDeque::new(),
            new_at: Vec::new(),
            new_marks: Vec::new(),
            new_sum: vec![0.0; k],
        }
    }

    fn rows(&self) -> usize {
        self.old_off.len() + self.new_at.len()
    }

    fn has_shape(&self, n: usize, k: usize, cap: usize) -> bool {
        let old = self.old_off.len();
        let new = self.new_at.len();
        self.old_marks.len() == old * n
            && self.old_fwd.len() == old * k
            && self.new_marks.len() == new * n
            && self.new_sum.len() == k
            && old + new <= cap
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Hawkes {
    cfg: HawkesCfg,
    /// `R`, node-major: `a = j U + u`.
    r: Vec<f64>,
    /// `b`, one row of `p = 1 + k` per node.
    b: Vec<f64>,
    /// `Q_jl = Σ w m_j m_l` over the rows, its lower triangle packed by
    /// rows ([`tri`]).
    q: Vec<f64>,
    /// `G₀₀ = ∫ w dt`.
    g0: f64,
    w_sum: f64,
    window: Option<EventWindow>,
}

/// The buffers a read assembles and solves its system in, one set a thread,
/// kept between reads and written over; nothing in it is read between calls.
#[derive(Default)]
struct ReadScratch {
    solves: Solves,
    g: Vec<f64>,
    x: Vec<f64>,
    r: Vec<f64>,
    b: Vec<f64>,
    q: Vec<f64>,
}

thread_local! {
    /// Lent to a read and set back after it, never borrowed: see
    /// [`crate::scratch`].
    static READ: std::cell::Cell<Option<Box<ReadScratch>>> = const { std::cell::Cell::new(None) };
}

/// `f` in this thread's [`ReadScratch`], which a read of order `p` past
/// [`KEPT_ORDER`] leaves empty.
fn with_scratch<T>(p: usize, f: impl FnOnce(&mut ReadScratch) -> T) -> T {
    let mut sc = READ.take().unwrap_or_default();
    let out = f(&mut sc);
    if p > KEPT_ORDER {
        *sc = ReadScratch::default();
    }
    READ.set(Some(sc));
    out
}

/// The place of `Q_jl` in its packed lower triangle.
#[inline]
fn tri(j: usize, l: usize) -> usize {
    let (hi, lo) = if j >= l { (j, l) } else { (l, j) };
    hi * (hi + 1) / 2 + lo
}

/// The system's `G`, row-major `p × p`, from a state (see the module docs).
#[allow(clippy::too_many_arguments)]
fn gram_into(
    betas: &[f64],
    n: usize,
    gamma: f64,
    r: &[f64],
    b: &[f64],
    q: &[f64],
    g0: f64,
    g: &mut Vec<f64>,
) {
    let u_n = betas.len();
    let k = n * u_n;
    let p = 1 + k;
    g.clear();
    g.resize(p * p, 0.0);
    g[0] = g0;
    for a in 0..k {
        let (ja, ba) = (a / u_n, betas[a % u_n]);
        let v = (ba * b[ja * p] - r[a]) / (ba - gamma);
        g[a + 1] = v;
        g[(a + 1) * p] = v;
        for c in 0..=a {
            let (jc, bc) = (c / u_n, betas[c % u_n]);
            let mut jump = ba * b[ja * p + 1 + c] + bc * b[jc * p + 1 + a];
            jump += ba * bc * q[tri(ja, jc)];
            let v = (jump - r[a] * r[c]) / (ba + bc - gamma);
            g[(a + 1) * p + c + 1] = v;
            g[(c + 1) * p + a + 1] = v;
        }
    }
}

/// `θ` for every node, column-major `p × D` in `x`, from a state: false
/// where `G` is not positive definite as it stands (a factorization that
/// needed a jitter is refused, so a singular `G` reports nothing rather than
/// a fit of the jitter), or is so near singular that the fit is not the
/// data's ([`PIVOT_FLOOR`]).
#[allow(clippy::too_many_arguments)]
fn solve_into(
    betas: &[f64],
    n: usize,
    gamma: f64,
    r: &[f64],
    b: &[f64],
    q: &[f64],
    g0: f64,
    solves: &mut Solves,
    g: &mut Vec<f64>,
    x: &mut Vec<f64>,
) -> bool {
    let p = 1 + n * betas.len();
    gram_into(betas, n, gamma, r, b, q, g0, g);
    solve_definite_in(solves, g, b, p, n, PIVOT_FLOOR, x)
}

impl Hawkes {
    pub fn new(cfg: HawkesCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (n, _, k) = cfg.dims();
        Ok(Self {
            r: vec![0.0; k],
            b: vec![0.0; n * (1 + k)],
            q: vec![0.0; n * (n + 1) / 2],
            g0: 0.0,
            w_sum: 0.0,
            window: cfg.window_events.map(|_| EventWindow::new(k)),
            cfg,
        })
    }

    pub fn cfg(&self) -> &HawkesCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// The rows with an event the window holds; 0 without a window.
    pub fn window_rows(&self) -> usize {
        self.window.as_ref().map_or(0, EventWindow::rows)
    }

    fn gamma(&self) -> f64 {
        rate(self.cfg.decay)
    }

    /// The least-squares system as the state stands: `G` (row-major `p ×
    /// p`) and the right-hand sides `b` (one row of `p` per node), whose
    /// solutions are the fit.
    pub fn system(&self) -> (Vec<f64>, Vec<f64>) {
        let mut g = Vec::new();
        gram_into(
            &self.cfg.kernel_decays,
            self.cfg.n_targets,
            self.gamma(),
            &self.r,
            &self.b,
            &self.q,
            self.g0,
            &mut g,
        );
        (g, self.b.clone())
    }

    /// The fitted parameters as the state stands, one list per node: `μ_i`,
    /// then `α_iju` for each source node `j` and decay `u` (`j` outer). `NaN`
    /// throughout where the system is not positive definite.
    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        let (n, _, k) = self.cfg.dims();
        let p = 1 + k;
        with_scratch(p, |sc| {
            let ok = solve_into(
                &self.cfg.kernel_decays,
                n,
                self.gamma(),
                &self.r,
                &self.b,
                &self.q,
                self.g0,
                &mut sc.solves,
                &mut sc.g,
                &mut sc.x,
            );
            (0..n)
                .map(|i| {
                    if ok {
                        sc.x[i * p..(i + 1) * p].to_vec()
                    } else {
                        vec![f64::NAN; p]
                    }
                })
                .collect()
        })
    }

    /// The intensities from a state advanced to the row, into `pred`:
    /// `NaN` for each node where `n_eff` is under `min_weight` or the system
    /// is not solvable.
    #[allow(clippy::too_many_arguments)]
    fn intensities_into(
        cfg: &HawkesCfg,
        n_eff: f64,
        r: &[f64],
        b: &[f64],
        q: &[f64],
        g0: f64,
        sc: &mut ReadScratch,
        pred: &mut Vec<f64>,
    ) {
        let (n, _, k) = cfg.dims();
        let p = 1 + k;
        pred.clear();
        let ReadScratch { solves, g, x, .. } = sc;
        if !(n_eff >= cfg.min_weight
            && solve_into(
                &cfg.kernel_decays,
                n,
                rate(cfg.decay),
                r,
                b,
                q,
                g0,
                solves,
                g,
                x,
            ))
        {
            pred.resize(n, f64::NAN);
            return;
        }
        for i in 0..n {
            let theta = &x[i * p..(i + 1) * p];
            let mut lam = theta[0];
            for (t, ra) in theta[1..].iter().zip(r) {
                lam += t * ra;
            }
            pred.push(lam);
        }
    }

    /// `G₀₀`'s step over `d`: `(1 − e^(−γ d)) / γ`, `d` where `γ = 0`.
    fn g0_step(gamma: f64, d: f64) -> f64 {
        if gamma == 0.0 {
            d
        } else {
            -(-gamma * d).exp_m1() / gamma
        }
    }

    /// Learn one row's events into `r`, `b`, `q`: each node's mark `m > 0`
    /// adds `m φ(t−)` to its `b` and `m m_l` to its `Q`, from `R` before any of
    /// the row's events, then raises its `R` by `m β`.
    fn learn_marks(betas: &[f64], marks: &[f64], r: &mut [f64], b: &mut [f64], q: &mut [f64]) {
        let u_n = betas.len();
        let k = r.len();
        let p = 1 + k;
        for (j, &m) in marks.iter().enumerate() {
            if m > 0.0 {
                let row = &mut b[j * p..(j + 1) * p];
                row[0] += m;
                for (bj, ra) in row[1..].iter_mut().zip(r.iter()) {
                    *bj += m * ra;
                }
                for (l, &ml) in marks.iter().enumerate().take(j + 1) {
                    if ml > 0.0 {
                        q[tri(j, l)] += m * ml;
                    }
                }
            }
        }
        for (j, &m) in marks.iter().enumerate() {
            if m > 0.0 {
                for (ra, beta) in r[j * u_n..(j + 1) * u_n].iter_mut().zip(betas) {
                    *ra += m * beta;
                }
            }
        }
    }

    /// The row's marks: a target's value above 0, else 0.
    fn mark(y: Option<f64>) -> f64 {
        match y {
            Some(v) if v > 0.0 => v,
            _ => 0.0,
        }
    }

    /// Add the row's events to the newer block, and drop the oldest row
    /// while the window holds more than its rows.
    fn window_learn(&mut self, y: &[Option<f64>]) {
        let (n, u_n, k) = self.cfg.dims();
        let betas = &self.cfg.kernel_decays;
        let Some(w) = self.window.as_mut() else {
            return;
        };
        w.new_at.push(w.since);
        for j in 0..n {
            w.new_marks.push(Self::mark(y.get(j).copied().flatten()));
        }
        let at = w.new_marks.len() - n;
        for (u, beta) in betas.iter().enumerate() {
            let e = (-beta * w.since).exp();
            for i in 0..n {
                let m = w.new_marks[at + i];
                if m > 0.0 {
                    w.new_sum[i * u_n + u] += m * e;
                }
            }
        }
        debug_assert_eq!(w.new_sum.len(), k);
        let cap = self.cfg.window_events.unwrap_or(usize::MAX);
        if w.rows() > cap {
            if w.old_off.is_empty() {
                self.form_old_block();
            } else {
                self.drop_oldest();
            }
        }
    }

    /// The oldest row out of the older block, taking back what it added.
    fn drop_oldest(&mut self) {
        let (n, u_n, k) = self.cfg.dims();
        let p = 1 + k;
        let betas = &self.cfg.kernel_decays;
        let w = self.window.as_mut().expect("windowed");
        let off = w.old_off.pop_front().expect("an older row");
        let marks: Vec<f64> = w.old_marks.drain(..n).collect();
        let fwd: Vec<f64> = w.old_fwd.drain(..k).collect();
        for (u, &beta) in betas.iter().enumerate() {
            let to_block = (-beta * off).exp();
            let to_now = (-beta * (w.since + off)).exp();
            for (j, &m) in marks.iter().enumerate() {
                if m <= 0.0 {
                    continue;
                }
                let a = j * u_n + u;
                self.r[a] -= m * beta * to_now;
                for i in 0..n {
                    let f = fwd[i * u_n + u] + to_block * w.new_sum[i * u_n + u];
                    self.b[i * p + 1 + a] -= beta * m * f;
                }
            }
        }
        for (j, &m) in marks.iter().enumerate() {
            if m > 0.0 {
                self.b[j * p] -= m;
                for (l, &ml) in marks.iter().enumerate().take(j + 1) {
                    if ml > 0.0 {
                        self.q[tri(j, l)] -= m * ml;
                    }
                }
            }
        }
        self.g0 = w.since + off;
    }

    /// The older block has run out: the newer becomes it, less its oldest
    /// row, which is the one to drop. Its forward sums by the backward
    /// recursion, and `R`, `b` and `q` formed afresh from its rows.
    fn form_old_block(&mut self) {
        let (n, u_n, k) = self.cfg.dims();
        let betas = &self.cfg.kernel_decays;
        let w = self.window.as_mut().expect("windowed");
        let at = std::mem::take(&mut w.new_at);
        let marks = std::mem::take(&mut w.new_marks);
        let rows = at.len();
        let now = w.since;
        // The kept rows are 1..rows; `fwd[t]` is row `t + 1`'s.
        let kept = rows - 1;
        let mut fwd = vec![0.0; kept * k];
        for t in (0..kept.saturating_sub(1)).rev() {
            let gap = at[t + 2] - at[t + 1];
            for (u, beta) in betas.iter().enumerate() {
                let e = (-beta * gap).exp();
                for i in 0..n {
                    let a = i * u_n + u;
                    fwd[t * k + a] = e * (marks[(t + 2) * n + i] + fwd[(t + 1) * k + a]);
                }
            }
        }
        self.r.fill(0.0);
        self.b.fill(0.0);
        self.q.fill(0.0);
        for rr in 1..rows {
            if rr > 1 {
                let gap = at[rr] - at[rr - 1];
                for (u, beta) in betas.iter().enumerate() {
                    let e = (-beta * gap).exp();
                    for j in 0..n {
                        self.r[j * u_n + u] *= e;
                    }
                }
            }
            Self::learn_marks(
                betas,
                &marks[rr * n..(rr + 1) * n],
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        }
        let gap = now - at[rows - 1];
        for (u, beta) in betas.iter().enumerate() {
            let e = (-beta * gap).exp();
            for j in 0..n {
                self.r[j * u_n + u] *= e;
            }
        }
        self.g0 = now - at[0];
        w.old_off = at[1..].iter().map(|t| now - t).collect();
        w.old_marks = marks[n..].iter().copied().collect();
        w.old_fwd = fwd.into();
        w.new_at = Vec::new();
        w.new_marks = Vec::new();
        w.new_sum.fill(0.0);
        w.since = 0.0;
    }
}

impl OnlineModel for Hawkes {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A target or a weight that is not usable, by the rule every model
        // keeps (`OnlineModel`): such a target is no event, and such a
        // weight learns nothing. The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let (n, u_n, _) = self.cfg.dims();
        let gamma = self.gamma();
        let f = self.cfg.decay.factor(d_clock);
        // The state to the row's clock: what `predict` reads on copies.
        for (u, beta) in self.cfg.kernel_decays.iter().enumerate() {
            let e = (-beta * d_clock).exp();
            for j in 0..n {
                self.r[j * u_n + u] *= e;
            }
        }
        for v in self.b.iter_mut().chain(self.q.iter_mut()) {
            *v *= f;
        }
        self.g0 = f * self.g0 + Self::g0_step(gamma, d_clock);
        if let Some(w) = self.window.as_mut() {
            w.since += d_clock;
        }
        let n_eff = self.w_sum;
        let p = self.b.len() / n;
        with_scratch(p, |sc| {
            Self::intensities_into(
                &self.cfg,
                n_eff,
                &self.r,
                &self.b,
                &self.q,
                self.g0,
                sc,
                &mut out.pred,
            )
        });
        out.n_eff = n_eff;
        out.extra = None;
        self.w_sum = f * self.w_sum + weight;
        // A weight is usable here, so not above 0 is 0.
        if weight <= 0.0 || !y.iter().take(n).any(|v| Self::mark(*v) > 0.0) {
            return;
        }
        // Each event from `R` before the row's events (the left limit).
        let k = self.r.len();
        let mut marks = [0.0f64; 8];
        if n <= marks.len() {
            for (j, m) in marks.iter_mut().enumerate().take(n) {
                *m = Self::mark(y.get(j).copied().flatten());
            }
            Self::learn_marks(
                &self.cfg.kernel_decays,
                &marks[..n],
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        } else {
            let marks: Vec<f64> = (0..n)
                .map(|j| Self::mark(y.get(j).copied().flatten()))
                .collect();
            Self::learn_marks(
                &self.cfg.kernel_decays,
                &marks,
                &mut self.r,
                &mut self.b,
                &mut self.q,
            );
        }
        debug_assert_eq!(self.r.len(), k);
        if self.window.is_some() {
            self.window_learn(y);
        }
    }

    fn predict(&self, _x: &[f64], d_clock: f64) -> Step {
        let (n, u_n, _) = self.cfg.dims();
        let gamma = self.gamma();
        let f = self.cfg.decay.factor(d_clock);
        let g0 = f * self.g0 + Self::g0_step(gamma, d_clock);
        let mut pred = Vec::new();
        let p = self.b.len() / n;
        with_scratch(p, |sc| {
            let mut r = std::mem::take(&mut sc.r);
            let mut b = std::mem::take(&mut sc.b);
            let mut q = std::mem::take(&mut sc.q);
            r.clear();
            r.extend_from_slice(&self.r);
            for (u, beta) in self.cfg.kernel_decays.iter().enumerate() {
                let e = (-beta * d_clock).exp();
                for j in 0..n {
                    r[j * u_n + u] *= e;
                }
            }
            b.clear();
            b.extend(self.b.iter().map(|v| v * f));
            q.clear();
            q.extend(self.q.iter().map(|v| v * f));
            Self::intensities_into(&self.cfg, self.w_sum, &r, &b, &q, g0, sc, &mut pred);
            sc.r = r;
            sc.b = b;
            sc.q = q;
        });
        Step {
            pred,
            n_eff: self.w_sum,
            extra: None,
        }
    }

    fn state(&self) -> State {
        State::new(ModelState::Hawkes(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::Hawkes(m) => {
                let m = (**m).clone();
                crate::model::check_cfg("hawkes", m.cfg.validate())?;
                let (n, _, k) = m.cfg.dims();
                let shaped = m.r.len() == k
                    && m.b.len() == n * (1 + k)
                    && m.q.len() == n * (n + 1) / 2
                    && match (&m.window, m.cfg.window_events) {
                        (None, None) => true,
                        (Some(w), Some(cap)) => w.has_shape(n, k, cap),
                        _ => false,
                    };
                if !shaped {
                    return Err(crate::model::damaged(
                        "hawkes",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "hawkes",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The nodes are the targets; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets
    }
}

#[cfg(test)]
mod tests;
