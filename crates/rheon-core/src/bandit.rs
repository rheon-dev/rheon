//! Contextual linear bandits in square-root information (QR) form.
//!
//! Each arm `a in 0..K` keeps the upper-triangular Cholesky factor `R_a` of its
//! decayed ridge information matrix `A_a = R_a' R_a` (`A_0 = l2 I`) and rotated
//! right-hand side `u_a = R_a^-T b_a` (`b_0 = 0`), with cached coefficients
//! `beta_a = R_a^-1 u_a`:
//!
//! ```text
//! learn (logged arm a, reward y, weight w > 0):
//!     every arm a':  R_a' <- sqrt(lam) R_a',   u_a' <- sqrt(lam) u_a'
//!     arm a:         Givens rotations of [R_a  u_a; sqrt(w) z'  sqrt(w) y]
//!                    beta_a <- R_a^-1 u_a
//!                    n <- n + 1
//! ```
//!
//! where `z = [1, x]` when `fit_intercept` is true and `z = x` otherwise.
//! Because uniform scaling of `(R_a', u_a')` leaves `beta_a' = R_a'^-1 u_a'`
//! invariant, only the single arm that learned on a row runs Givens rotations
//! and back-substitution.
//!
//! Before the row is learned (out of sample), each arm is scored according to
//! the configured [`BanditPolicy`]:
//!
//! - [`BanditPolicy::Linucb`] (Li et al. 2010):
//!   `s_a = z' beta_a + alpha * ||R_a^-T z||` (deterministic; ties keep lowest arm).
//! - [`BanditPolicy::Lints`] (Agrawal and Goyal 2013):
//!   `s_a = z' (beta_a + alpha * R_a^-1 z_draw) = z' beta_a + alpha * (R_a^-T z)' z_draw`
//!   with `z_draw ~ N(0, I)` from Philox4x64-10 lane 1 at counter `n`.
//! - [`BanditPolicy::EpsGreedy`]:
//!   `s_a = z' beta_a`; with probability `epsilon` (Philox lane 0) choose an arm
//!   uniformly, else `argmax_a s_a`. Emits exact choice probability `prob`
//!   (`1 - epsilon + epsilon / K` for the greedy arm, `epsilon / K` otherwise).
//!
//! ## Targets layout
//!
//! `OnlineModel::step_into` receives `y = [Some(reward), Some(arm_index)]`
//! (`n_targets() == 2`). If either `reward` or `arm_index` is missing or not
//! usable, the row is scored out-of-sample and teaches no arm (leaving the
//! Philox learned-row counter `n` unchanged).

use serde::{Deserialize, Serialize};

use crate::Decay;
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::philox::{DrawStream, LANE_POLICY_NORMAL, LANE_POLICY_UNIFORM, below};

/// Default per-group state cap for [`Bandit`]: 256 KiB.
pub const BANDIT_MAX_STATE_BYTES: u64 = crate::clock_span::DEFAULT_STATE_BYTES;

/// Exploration policy for [`Bandit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BanditPolicy {
    /// Disjoint LinUCB (Li et al. 2010): `z' beta_a + alpha * sqrt(z' A_a^-1 z)`.
    #[default]
    Linucb,
    /// Disjoint Linear Thompson Sampling (Agrawal and Goyal 2013).
    Lints,
    /// Linear epsilon-greedy over per-arm ridge expectations.
    EpsGreedy,
}

impl BanditPolicy {
    pub fn name(self) -> &'static str {
        match self {
            BanditPolicy::Linucb => "linucb",
            BanditPolicy::Lints => "lints",
            BanditPolicy::EpsGreedy => "eps_greedy",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "linucb" => Ok(BanditPolicy::Linucb),
            "lints" => Ok(BanditPolicy::Lints),
            "eps_greedy" => Ok(BanditPolicy::EpsGreedy),
            other => Err(format!(
                "policy must be \"linucb\", \"lints\" or \"eps_greedy\", but it is {other:?}. \
                 Use one of those words."
            )),
        }
    }
}

/// Configuration for [`Bandit`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BanditCfg {
    pub n_features: usize,
    pub n_arms: usize,
    pub fit_intercept: bool,
    pub decay: Decay,
    pub policy: BanditPolicy,
    /// Exploration width multiplier for `Linucb` and `Lints` (`>= 0`).
    pub alpha: f64,
    /// Uniform exploration probability for `EpsGreedy` (`[0, 1]`).
    pub epsilon: f64,
    /// Prior strength `A_0 = l2 * I` (`> 0`).
    pub l2: f64,
    pub min_weight: f64,
    /// Philox4x64-10 seed (default `0`).
    pub seed: u64,
    /// Canonical hash of the group key (`0` when ungrouped).
    #[serde(default)]
    pub group_hash: u64,
    /// Hash of the spec name (`0` in standalone core use).
    #[serde(default)]
    pub slot: u64,
    /// Per-group byte cap (`None` uses [`BANDIT_MAX_STATE_BYTES`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_state_bytes: Option<u64>,
}

impl BanditCfg {
    #[inline]
    pub fn k_total(&self) -> usize {
        self.n_features + usize::from(self.fit_intercept)
    }

    /// Number of output slots written into [`Step::pred`]:
    /// `[chosen_arm, chosen_score, score_0, ..., score_{K-1}]`, plus `[prob]`
    /// for [`BanditPolicy::EpsGreedy`].
    #[inline]
    pub fn n_outputs(&self) -> usize {
        2 + self.n_arms + usize::from(self.policy == BanditPolicy::EpsGreedy)
    }

    /// Fixed per-group state bytes (`R` upper triangle + `u` per arm, plus `n`).
    pub fn state_bytes(&self) -> u64 {
        let k = self.k_total() as u64;
        let tri = k.saturating_mul(k.saturating_add(1)) / 2;
        let per_arm = tri.saturating_add(k).saturating_mul(8);
        (self.n_arms as u64)
            .saturating_mul(per_arm)
            .saturating_add(8)
    }

    pub fn cap_bytes(&self) -> u64 {
        self.max_state_bytes.unwrap_or(BANDIT_MAX_STATE_BYTES)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("bandit: {e}")));
        if self.n_features == 0 {
            no.refuse(
                "bandit: the model needs at least one feature, but it has none. Name at least \
                 one column in features.",
            );
        }
        if self.n_arms < 2 {
            no.refuse(format!(
                "bandit: the model needs at least 2 arms, but it has {}. List 2 or more arms \
                 in arms.",
                self.n_arms
            ));
        }
        if !(self.l2.is_finite() && self.l2 > 0.0) {
            no.refuse(format!(
                "bandit: l2 must be finite and above 0, but it is {}. Set l2 to a positive \
                 prior precision, such as the default 1.",
                self.l2
            ));
        }
        match self.policy {
            BanditPolicy::Linucb | BanditPolicy::Lints => {
                if !(self.alpha.is_finite() && self.alpha >= 0.0) {
                    no.refuse(format!(
                        "bandit: alpha must be finite and at least 0, but it is {}. Set alpha \
                         to a non-negative exploration scale, such as the default 1.",
                        self.alpha
                    ));
                }
            }
            BanditPolicy::EpsGreedy => {
                if !(self.epsilon.is_finite() && (0.0..=1.0).contains(&self.epsilon)) {
                    no.refuse(format!(
                        "bandit: epsilon must be between 0 and 1, but it is {}. Set epsilon to \
                         the exploration probability in [0, 1], such as 0.1.",
                        self.epsilon
                    ));
                }
            }
        }
        no.take(crate::model::check_min_weight("bandit", self.min_weight));
        no.take(crate::clock_span::check_max_state_bytes(
            "bandit",
            self.max_state_bytes,
        ));
        if self.n_features > 0 && self.n_arms >= 2 {
            let cap = self.cap_bytes();
            let bytes = self.state_bytes();
            if cap > 0 && cap <= crate::clock_span::STATE_BYTES_CEILING && bytes > cap {
                no.refuse(format!(
                    "bandit: state holds {bytes} bytes for {} arms and {} terms, past \
                     max_state_bytes of {cap}. Lower the number of arms or features, or raise \
                     max_state_bytes.",
                    self.n_arms,
                    self.k_total()
                ));
            }
        }
        no.finish()
    }
}

#[inline(always)]
const fn base(k: usize, i: usize) -> usize {
    i * (2 * k - i - 1) / 2
}

#[inline]
const fn packed_len(k: usize) -> usize {
    k * (k + 1) / 2
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct ArmState {
    /// Packed upper-triangular Cholesky factor `R` (`A = R' R`), diagonal >= 0.
    r: Vec<f64>,
    /// Rotated RHS `u = R^-T b`.
    u: Vec<f64>,
    /// Cached coefficients `beta = R^-1 u`.
    beta: Vec<f64>,
}

impl ArmState {
    fn new(k: usize, l2: f64) -> Self {
        let root = l2.sqrt();
        let mut r = vec![0.0; packed_len(k)];
        for i in 0..k {
            r[base(k, i) + i] = root;
        }
        Self {
            r,
            u: vec![0.0; k],
            beta: vec![0.0; k],
        }
    }

    fn scale(&mut self, root: f64) {
        let mut any_nonzero = false;
        for v in &mut self.r {
            *v *= root;
            any_nonzero |= *v != 0.0;
        }
        for v in &mut self.u {
            *v *= root;
        }
        if !any_nonzero {
            self.beta.fill(0.0);
        }
    }

    /// Givens QR update of `[R u; sw*z' sw*y]` (after scaling by `root` when
    /// `DECAY` is handled outside). Diagonal of `R` stays non-negative.
    fn fold(&mut self, k: usize, zrow: &mut [f64], mut ry: f64) {
        for i in 0..k {
            let b = zrow[i];
            if b == 0.0 {
                continue;
            }
            let row = &mut self.r[base(k, i) + i..base(k, i) + k];
            let a = row[0];
            let rho = a.hypot(b);
            if rho == 0.0 {
                continue;
            }
            let c = a / rho;
            let s = b / rho;
            row[0] = rho;
            zrow[i] = 0.0;
            for (rij, zj) in row[1..].iter_mut().zip(zrow[i + 1..].iter_mut()) {
                let (aa, bb) = (*rij, *zj);
                *rij = c * aa + s * bb;
                *zj = c * bb - s * aa;
            }
            let ui = self.u[i];
            self.u[i] = c * ui + s * ry;
            ry = c * ry - s * ui;
        }
    }

    /// Back-substitution `beta = R^-1 u`. Zero pivots yield 0 (no 0/0).
    fn solve_beta(&mut self, k: usize) {
        for i in (0..k).rev() {
            let row = &self.r[base(k, i) + i..base(k, i) + k];
            let mut acc = self.u[i];
            for (rij, &bj) in row[1..].iter().zip(&self.beta[i + 1..]) {
                acc -= *rij * bj;
            }
            let d = row[0];
            self.beta[i] = if d > 0.0 {
                let v = acc / d;
                if v.is_finite() { v } else { 0.0 }
            } else {
                0.0
            };
        }
    }

    /// Forward substitution solving `R' v = z` into `work`, returning `true` if
    /// all pivots encountered were positive and finite.
    fn forward_sub_t(&self, k: usize, z: &[f64], work: &mut [f64]) {
        for i in 0..k {
            let mut acc = z[i];
            let mut at = i;
            for (m, &vm) in work.iter().enumerate().take(i) {
                acc -= self.r[at] * vm;
                at += k - m - 1;
            }
            let d = self.r[at];
            work[i] = if d > 0.0 {
                let v = acc / d;
                if v.is_finite() { v } else { 0.0 }
            } else {
                0.0
            };
        }
    }

    #[inline]
    fn width(&self, k: usize, z: &[f64], work: &mut [f64]) -> f64 {
        self.forward_sub_t(k, z, work);
        let scale = work.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        if scale == 0.0 || !scale.is_finite() {
            return scale;
        }
        scale
            * work
                .iter()
                .map(|&x| {
                    let s = x / scale;
                    s * s
                })
                .sum::<f64>()
                .sqrt()
    }
}

/// Reusable per-row scratch buffers (never serialized).
#[derive(Clone, Default)]
struct BanditWork {
    z: Vec<f64>,
    v: Vec<f64>,
}

impl std::fmt::Debug for BanditWork {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("BanditWork")
    }
}

impl PartialEq for BanditWork {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Contextual linear bandit (LinUCB, LinTS, or linear epsilon-greedy).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bandit {
    cfg: BanditCfg,
    arms: Vec<ArmState>,
    w_sum: f64,
    /// Count of positive-weight rows that learned a `(arm, reward)` pair.
    n_learned: u64,
    #[serde(skip)]
    work: BanditWork,
    /// Test-only switch for draw-for-draw oracle checks against MABWiser's
    /// `chol(alpha^2 * A^-1)` Thompson sampling root. Never serialized or
    /// exposed in the public API.
    #[cfg(test)]
    #[serde(skip)]
    ts_compatible_root: bool,
}

impl Bandit {
    pub fn new(cfg: BanditCfg) -> Result<Self, String> {
        cfg.validate()?;
        let k = cfg.k_total();
        let arms = (0..cfg.n_arms).map(|_| ArmState::new(k, cfg.l2)).collect();
        Ok(Self {
            arms,
            w_sum: 0.0,
            n_learned: 0,
            work: BanditWork {
                z: vec![0.0; k],
                v: vec![0.0; k],
            },
            #[cfg(test)]
            ts_compatible_root: false,
            cfg,
        })
    }

    pub fn cfg(&self) -> &BanditCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    pub fn n_learned(&self) -> u64 {
        self.n_learned
    }

    pub fn coefficients(&self) -> Vec<Vec<f64>> {
        self.arms.iter().map(|a| a.beta.clone()).collect()
    }

    #[inline]
    fn draw_stream(&self) -> DrawStream {
        DrawStream::new(self.cfg.seed, self.cfg.group_hash, self.cfg.slot)
    }

    #[inline]
    fn fill_z(&self, x: &[f64], z: &mut [f64]) {
        if self.cfg.fit_intercept {
            z[0] = 1.0;
            z[1..].copy_from_slice(x);
        } else {
            z.copy_from_slice(x);
        }
    }

    #[inline]
    fn arm_index(&self, y: &[Option<f64>]) -> Option<usize> {
        let v = y.get(1).copied().flatten()?;
        if v.is_finite() && v >= 0.0 && v.fract() == 0.0 && v < self.cfg.n_arms as f64 {
            Some(v as usize)
        } else {
            None
        }
    }

    #[inline]
    fn dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| x * y).sum()
    }

    /// Score all arms into `out` without mutating state.
    fn score_with(&self, x: &[f64], z: &mut [f64], v: &mut [f64], out: &mut Step) {
        let n_out = self.cfg.n_outputs();
        out.pred.clear();
        out.pred.resize(n_out, f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;

        if self.w_sum < self.cfg.min_weight {
            return;
        }

        let k = self.cfg.k_total();
        self.fill_z(x, z);
        let stream = self.draw_stream();

        match self.cfg.policy {
            BanditPolicy::Linucb => {
                let mut best_arm = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for (a, st) in self.arms.iter().enumerate() {
                    let mean = Self::dot(&st.beta, z);
                    let bonus = if self.cfg.alpha > 0.0 {
                        self.cfg.alpha * st.width(k, z, v)
                    } else {
                        0.0
                    };
                    let s = mean + bonus;
                    if !s.is_finite() {
                        out.pred.fill(f64::NAN);
                        return;
                    }
                    out.pred[2 + a] = s;
                    if s > best_score {
                        best_score = s;
                        best_arm = a;
                    }
                }
                out.pred[0] = best_arm as f64;
                out.pred[1] = best_score;
            }
            BanditPolicy::Lints => {
                let mut normals = stream.reader(LANE_POLICY_NORMAL, self.n_learned, 0);
                let mut best_arm = 0usize;
                let mut best_score = f64::NEG_INFINITY;
                for (a, st) in self.arms.iter().enumerate() {
                    let mean = Self::dot(&st.beta, z);
                    #[cfg(test)]
                    let bonus = if self.ts_compatible_root {
                        self.compatible_ts_bonus(st, z, &mut normals)
                    } else {
                        self.fast_ts_bonus(st, k, z, v, &mut normals)
                    };
                    #[cfg(not(test))]
                    let bonus = self.fast_ts_bonus(st, k, z, v, &mut normals);
                    let s = mean + bonus;
                    if !s.is_finite() {
                        out.pred.fill(f64::NAN);
                        return;
                    }
                    out.pred[2 + a] = s;
                    if s > best_score {
                        best_score = s;
                        best_arm = a;
                    }
                }
                out.pred[0] = best_arm as f64;
                out.pred[1] = best_score;
            }
            BanditPolicy::EpsGreedy => {
                let mut greedy_arm = 0usize;
                let mut greedy_score = f64::NEG_INFINITY;
                for (a, st) in self.arms.iter().enumerate() {
                    let s = Self::dot(&st.beta, z);
                    if !s.is_finite() {
                        out.pred.fill(f64::NAN);
                        return;
                    }
                    out.pred[2 + a] = s;
                    if s > greedy_score {
                        greedy_score = s;
                        greedy_arm = a;
                    }
                }
                let mut uniforms = stream.reader(LANE_POLICY_UNIFORM, self.n_learned, 0);
                let coin = uniforms.next_u01();
                let explore_word = uniforms.next_word();
                let chosen = if coin < self.cfg.epsilon {
                    below(explore_word, self.cfg.n_arms)
                } else {
                    greedy_arm
                };
                let eps = self.cfg.epsilon;
                let k_arms = self.cfg.n_arms as f64;
                let prob = if chosen == greedy_arm {
                    (1.0 - eps) + eps / k_arms
                } else {
                    eps / k_arms
                };
                out.pred[0] = chosen as f64;
                out.pred[1] = out.pred[2 + chosen];
                out.pred[2 + self.cfg.n_arms] = prob;
            }
        }
    }

    /// Fast `O(d^2)` Thompson sampling term `alpha * (R^-T z)' z_draw`, which
    /// equals `z' (alpha * R^-1 z_draw)` and has exact covariance `alpha^2 A^-1`.
    #[inline]
    fn fast_ts_bonus(
        &self,
        st: &ArmState,
        k: usize,
        z: &[f64],
        v: &mut [f64],
        normals: &mut crate::philox::LaneReader,
    ) -> f64 {
        st.forward_sub_t(k, z, v);
        let mut dot = 0.0;
        for &vi in v.iter() {
            let draw = normals.next_normal();
            dot += vi * draw;
        }
        self.cfg.alpha * dot
    }

    /// Test-only `O(d^3)` MABWiser-compatible square root `chol(alpha^2 A^-1)`.
    #[cfg(test)]
    fn compatible_ts_bonus(
        &self,
        st: &ArmState,
        z: &[f64],
        normals: &mut crate::philox::LaneReader,
    ) -> f64 {
        let k = self.cfg.k_total();
        let mut r_inv = vec![0.0; k * k];
        // Columns of R^-1 by back-substitution on e_col.
        for col in 0..k {
            for i in (0..=col).rev() {
                let row = &st.r[base(k, i) + i..base(k, i) + k];
                let mut acc = if i == col { 1.0 } else { 0.0 };
                for j in (i + 1)..=col {
                    acc -= row[j - i] * r_inv[j * k + col];
                }
                let d = row[0];
                r_inv[i * k + col] = if d > 0.0 { acc / d } else { 0.0 };
            }
        }
        let a2 = self.cfg.alpha * self.cfg.alpha;
        let mut cov = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                let mut s = 0.0;
                for m in i.max(j)..k {
                    s += r_inv[i * k + m] * r_inv[j * k + m];
                }
                let val = a2 * s;
                cov[i * k + j] = val;
                cov[j * k + i] = val;
            }
        }
        // Lower-triangular Cholesky L of cov.
        let mut l = vec![0.0; k * k];
        for i in 0..k {
            for j in 0..=i {
                let mut s = cov[i * k + j];
                for m in 0..j {
                    s -= l[i * k + m] * l[j * k + m];
                }
                l[i * k + j] = if i == j {
                    s.max(0.0).sqrt()
                } else if l[j * k + j] > 0.0 {
                    s / l[j * k + j]
                } else {
                    0.0
                };
            }
        }
        let mut z_draw = vec![0.0; k];
        for zi in &mut z_draw {
            *zi = normals.next_normal();
        }
        let mut delta = 0.0;
        for i in 0..k {
            let mut lzi = 0.0;
            for j in 0..=i {
                lzi += l[i * k + j] * z_draw[j];
            }
            delta += z[i] * lzi;
        }
        delta
    }
}

impl OnlineModel for Bandit {
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
        let k = self.cfg.k_total();
        if self.work.z.len() != k {
            self.work.z.resize(k, 0.0);
        }
        if self.work.v.len() != k {
            self.work.v.resize(k, 0.0);
        }
        let mut work = std::mem::take(&mut self.work);
        self.score_with(x, &mut work.z, &mut work.v, out);

        let lam = self.cfg.decay.factor(d_clock);
        if lam != 1.0 {
            let root = lam.sqrt();
            for arm in &mut self.arms {
                arm.scale(root);
            }
        }
        if weight > 0.0 {
            self.w_sum = lam * self.w_sum + weight;
            let reward = y.first().copied().flatten().filter(|&v| crate::usable(v));
            let arm_idx = self.arm_index(y);
            if let (Some(r), Some(a)) = (reward, arm_idx) {
                let sw = weight.sqrt();
                self.fill_z(x, &mut work.z);
                for zi in &mut work.z {
                    *zi *= sw;
                }
                self.arms[a].fold(k, &mut work.z, sw * r);
                self.arms[a].solve_beta(k);
                self.n_learned += 1;
            }
        } else {
            self.w_sum *= lam;
        }
        self.work = work;
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let k = self.cfg.k_total();
        let mut z = vec![0.0; k];
        let mut v = vec![0.0; k];
        let mut out = Step::default();
        self.score_with(x, &mut z, &mut v, &mut out);
        out
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_targets(&self) -> usize {
        2
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_outputs()
    }

    fn state(&self) -> State {
        State::new(ModelState::Bandit(Box::new(self.clone())))
    }

    fn restore(state: &State) -> Result<Self, StateError> {
        check_schema(state)?;
        let m = match &state.model {
            ModelState::Bandit(b) => (**b).clone(),
            other => {
                return Err(StateError::WrongModel {
                    expected: "bandit",
                    found: other.kind(),
                });
            }
        };
        crate::model::check_cfg("bandit", m.cfg.validate())?;
        let k = m.cfg.k_total();
        let tri = packed_len(k);
        if m.arms.len() != m.cfg.n_arms {
            return Err(crate::model::damaged(
                "bandit",
                "its arm states have the wrong shape",
            ));
        }
        if !(m.w_sum.is_finite() && m.w_sum >= 0.0) {
            return Err(crate::model::damaged(
                "bandit",
                "its accumulated weight is not a non-negative finite number",
            ));
        }
        for arm in &m.arms {
            if arm.r.len() != tri || arm.u.len() != k || arm.beta.len() != k {
                return Err(crate::model::damaged(
                    "bandit",
                    "its running sums have the wrong shape",
                ));
            }
            if !arm.r.iter().all(|v| v.is_finite())
                || !arm.u.iter().all(|v| v.is_finite())
                || !arm.beta.iter().all(|v| v.is_finite())
            {
                return Err(crate::model::damaged(
                    "bandit",
                    "its running sums contain a non-finite number",
                ));
            }
            for i in 0..k {
                if arm.r[base(k, i) + i] < 0.0 {
                    return Err(crate::model::damaged(
                        "bandit",
                        "its Cholesky factor has a negative diagonal",
                    ));
                }
            }
        }
        let mut m = m;
        m.work = BanditWork {
            z: vec![0.0; k],
            v: vec![0.0; k],
        };
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle;

    fn cfg(policy: BanditPolicy) -> BanditCfg {
        BanditCfg {
            n_features: 3,
            n_arms: 4,
            fit_intercept: true,
            decay: Decay::Halflife(f64::INFINITY),
            policy,
            alpha: 1.0,
            epsilon: 0.1,
            l2: 1.0,
            min_weight: 0.0,
            seed: 7,
            group_hash: 0,
            slot: 0,
            max_state_bytes: None,
        }
    }

    #[test]
    fn linucb_matches_explicit_inverse_oracle() {
        let mut b = Bandit::new(cfg(BanditPolicy::Linucb)).unwrap();
        let k = 4usize; // 1 + 3
        let mut gram = vec![vec![0.0; k * k]; 4];
        let mut rhs = vec![vec![0.0; k]; 4];
        for g in &mut gram {
            for i in 0..k {
                g[i * k + i] = 1.0;
            }
        }
        for t in 0..80 {
            let x = [
                ((t * 3 + 1) % 11) as f64 * 0.2 - 1.0,
                ((t * 7 + 2) % 13) as f64 * 0.15 - 0.8,
                ((t * 5 + 3) % 17) as f64 * 0.1 - 0.5,
            ];
            let z = [1.0, x[0], x[1], x[2]];
            let arm = t % 4;
            let y = 0.5 * (arm as f64 + 1.0) + (arm as f64 - 1.5) * x[0] + 0.3 * x[1] - 0.2 * x[2];
            let w = if t % 5 == 0 { 0.5 } else { 1.0 };

            let step = b.step(&x, &[Some(y), Some(arm as f64)], 1.0, w);

            // Exact oracle via faer LU in crate::oracle.
            let mut best_a = 0;
            let mut best_s = f64::NEG_INFINITY;
            for a in 0..4 {
                let beta = oracle::solve(&gram[a], &rhs[a]);
                let a_inv_z = oracle::solve(&gram[a], &z);
                let mean: f64 = z.iter().zip(&beta).map(|(p, q)| p * q).sum();
                let var: f64 = z.iter().zip(&a_inv_z).map(|(p, q)| p * q).sum();
                let s = mean + var.sqrt();
                assert!(
                    (step.pred[2 + a] - s).abs() < 1e-12 * s.abs().max(1.0),
                    "t={t} a={a}: got {} want {s}",
                    step.pred[2 + a]
                );
                if s > best_s {
                    best_s = s;
                    best_a = a;
                }
            }
            assert_eq!(step.pred[0], best_a as f64);
            assert!((step.pred[1] - best_s).abs() < 1e-12 * best_s.abs().max(1.0));

            for i in 0..k {
                rhs[arm][i] += w * y * z[i];
                for j in 0..k {
                    gram[arm][i * k + j] += w * z[i] * z[j];
                }
            }
        }
    }

    #[test]
    fn lints_fast_root_has_exact_posterior_covariance() {
        let mut b = Bandit::new(cfg(BanditPolicy::Lints)).unwrap();
        let k = b.cfg.k_total();
        for t in 0..60 {
            let x = [
                (t as f64 * 0.37).sin(),
                (t as f64 * 0.71).cos(),
                (t as f64 * 0.19).sin(),
            ];
            let arm = t % 4;
            let y = 1.0 + x[0] * (arm as f64) - 0.5 * x[1];
            b.step(&x, &[Some(y), Some(arm as f64)], 1.0, 1.0);
        }
        // Check (alpha * R^-1)(alpha * R^-1)' == alpha^2 * (R' R)^-1 for each arm.
        for a in 0..4 {
            let st = &b.arms[a];
            let mut ata = vec![0.0; k * k];
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0;
                    for m in 0..=i.min(j) {
                        s += st.r[base(k, m) + i] * st.r[base(k, m) + j];
                    }
                    ata[i * k + j] = s;
                }
            }
            // M = R^-1 by solving R' v = e_i -> v' v_j = (R^-1 R^-T)_{ij}.
            let mut cols = vec![vec![0.0; k]; k];
            for i in 0..k {
                let mut e = vec![0.0; k];
                e[i] = 1.0;
                st.forward_sub_t(k, &e, &mut cols[i]);
            }
            for i in 0..k {
                let mut e = vec![0.0; k];
                e[i] = 1.0;
                let a_inv_col = oracle::solve(&ata, &e);
                for j in 0..k {
                    let fast_cov: f64 = cols[i].iter().zip(&cols[j]).map(|(p, q)| p * q).sum();
                    assert!(
                        (fast_cov - a_inv_col[j]).abs() < 1e-12,
                        "arm {a} ({i},{j}): fast={fast_cov} exact={}",
                        a_inv_col[j]
                    );
                }
            }
        }
        // Compatible root smoke test: finite, matches predict.
        b.ts_compatible_root = true;
        let p1 = b.predict(&[0.5, -0.3, 0.2], 1.0);
        let p2 = b.predict(&[0.5, -0.3, 0.2], 1.0);
        assert_eq!(p1, p2);
        assert!(p1.pred.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn eps_greedy_emits_exact_probabilities_and_zero_weight_leaves_n_unchanged() {
        let mut b = Bandit::new(cfg(BanditPolicy::EpsGreedy)).unwrap();
        // Train arm 2 to clearly dominate.
        for t in 0..30 {
            let x = [1.0, 0.5, -0.2];
            let arm = t % 4;
            let y = if arm == 2 { 10.0 } else { 0.0 };
            b.step(&x, &[Some(y), Some(arm as f64)], 1.0, 1.0);
        }
        let before = b.state();
        let s0 = b.step(&[1.0, 0.5, -0.2], &[Some(99.0), Some(0.0)], 1.0, 0.0);
        assert_eq!(b.state(), before);
        assert_eq!(b.n_learned(), 30);
        let p0 = b.predict(&[1.0, 0.5, -0.2], 1.0);
        assert_eq!(s0, p0);
        let prob = s0.pred[2 + 4];
        let expected_greedy = 0.9 + 0.1 / 4.0;
        let expected_explore = 0.1 / 4.0;
        assert!(
            (prob - expected_greedy).abs() < 1e-15 || (prob - expected_explore).abs() < 1e-15,
            "unexpected prob {prob}"
        );
    }

    #[test]
    fn a_state_of_the_wrong_shape_is_refused() {
        let mut b = Bandit::new(cfg(BanditPolicy::Linucb)).unwrap();
        b.step(&[1.0, 2.0, 3.0], &[Some(1.0), Some(0.0)], 1.0, 1.0);
        let mut bad = b.clone();
        bad.arms.pop();
        assert!(Bandit::restore(&bad.state()).is_err());

        let mut bad2 = b.clone();
        bad2.arms[0].r.pop();
        assert!(Bandit::restore(&bad2.state()).is_err());

        let mut bad3 = b.clone();
        bad3.arms[0].r[0] = -1.0;
        assert!(Bandit::restore(&bad3.state()).is_err());
    }
}
