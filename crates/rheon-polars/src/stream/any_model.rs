//! Enum dispatch over every model kind the bank can run (`AnyModel`).
//! Split out of `stream.rs` to keep every source file under the repository cap.

use super::*;

/// Enum dispatch over the models the bank can run (serde-friendly).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AnyModel {
    EwRidge(Box<EwRidge>),
    Rls(Box<Rls>),
    Lasso(Box<Lasso>),
    Kalman(Box<Kalman>),
    Robust(Box<Robust>),
    Ftrl(Box<Ftrl>),
    EwCov(Box<EwCovModel>),
    Sgd(Box<Sgd>),
    Pa(Box<Pa>),
    Holt(Box<Holt>),
    KMeans(Box<KMeans>),
    Micro(Box<Micro>),
    EwClass(Box<EwClass>),
    SeqTest(Box<SeqTest>),
    Marginal(Box<Marginal>),
    Deco(Box<Deco>),
    Rcov(Box<Rcov>),
    Hmm(Box<Hmm>),
    CorrChange(Box<CorrChange>),
    Bocpd(Box<Bocpd>),
    Audit(Box<rheon_core::Audit>),
    Glm(Box<rheon_core::Glm>),
    Expectile(Box<rheon_core::Expectile>),
    Hawkes(Box<rheon_core::Hawkes>),
    Garch(Box<rheon_core::Garch>),
    Dcc(Box<rheon_core::Dcc>),
    Realized(Box<rheon_core::Realized>),
    Pca(Box<rheon_core::StreamPca>),
    PowerVariation(Box<rheon_core::PowerVariation>),
    TDigest(Box<rheon_core::TDigestModel>),
    DdSketch(Box<rheon_core::DdSketchModel>),
    EwmMoments(Box<rheon_core::EwmMoments>),
    Ar(Box<rheon_core::Ar>),
    VectorAr(Box<rheon_core::VectorAr>),
    Spectral(Box<rheon_core::Spectral>),
    Isotonic(Box<rheon_core::Isotonic>),
    Krls(Box<rheon_core::Krls>),
    Dependence(Box<rheon_core::Dependence>),
    HoeffdingTreeClassifier(Box<rheon_core::HoeffdingTreeClassifier>),
    HoeffdingTreeRegressor(Box<rheon_core::HoeffdingTreeRegressor>),
    Bandit(Box<rheon_core::Bandit>),
    HalfSpaceTrees(Box<rheon_core::HalfSpaceTrees>),
    RandomCutForest(Box<rheon_core::RandomCutForest>),
}

/// A flush's shards, run on whichever pool the caller is in: the bank's
/// own (pool.rs), since every learning path runs inside it.
fn on_the_pool(count: usize) -> rheon_core::Shards<'static> {
    fn run(shards: &mut [rheon_core::MarginalShard<'_>]) {
        use rayon::prelude::*;
        shards
            .par_iter_mut()
            .for_each(rheon_core::MarginalShard::run);
    }
    rheon_core::Shards { count, run: &run }
}

/// Bind the boxed model of whichever variant `$self` is, then run `$body`.
///
/// Only for the methods that are the *same* call on every variant. Three of
/// `AnyModel`'s six are not — `solve_failures` groups the models that never
/// factorize, `coefficients` reshapes per model, and `restore` matches on
/// `ModelState` rather than on `Self` — and those stay written out, because a
/// macro that needed a per-variant escape hatch would be harder to read than
/// the match it replaced (docs/SIMPLIFICATION.md S3).
macro_rules! dispatch {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            AnyModel::EwRidge($m) => $body,
            AnyModel::Rls($m) => $body,
            AnyModel::Lasso($m) => $body,
            AnyModel::Kalman($m) => $body,
            AnyModel::Robust($m) => $body,
            AnyModel::Ftrl($m) => $body,
            AnyModel::EwCov($m) => $body,
            AnyModel::Sgd($m) => $body,
            AnyModel::Pa($m) => $body,
            AnyModel::Holt($m) => $body,
            AnyModel::KMeans($m) => $body,
            AnyModel::Micro($m) => $body,
            AnyModel::EwClass($m) => $body,
            AnyModel::SeqTest($m) => $body,
            AnyModel::Marginal($m) => $body,
            AnyModel::Deco($m) => $body,
            AnyModel::Rcov($m) => $body,
            AnyModel::Hmm($m) => $body,
            AnyModel::CorrChange($m) => $body,
            AnyModel::Bocpd($m) => $body,
            AnyModel::Audit($m) => $body,
            AnyModel::Glm($m) => $body,
            AnyModel::Expectile($m) => $body,
            AnyModel::Hawkes($m) => $body,
            AnyModel::Garch($m) => $body,
            AnyModel::Dcc($m) => $body,
            AnyModel::Realized($m) => $body,
            AnyModel::Pca($m) => $body,
            AnyModel::PowerVariation($m) => $body,
            AnyModel::TDigest($m) => $body,
            AnyModel::DdSketch($m) => $body,
            AnyModel::EwmMoments($m) => $body,
            AnyModel::Ar($m) => $body,
            AnyModel::VectorAr($m) => $body,
            AnyModel::Spectral($m) => $body,
            AnyModel::Isotonic($m) => $body,
            AnyModel::Krls($m) => $body,
            AnyModel::Dependence($m) => $body,
            AnyModel::HoeffdingTreeClassifier($m) => $body,
            AnyModel::HoeffdingTreeRegressor($m) => $body,
            AnyModel::Bandit($m) => $body,
            AnyModel::HalfSpaceTrees($m) => $body,
            AnyModel::RandomCutForest($m) => $body,
        }
    };
}

impl AnyModel {
    /// Mix the main accumulators toward a long-run twin, where the model has
    /// one (`session_shrink`). A no-op elsewhere.
    pub fn blend_toward_long_run(&mut self) {
        if let AnyModel::EwRidge(m) = self {
            m.blend_toward_long_run();
        }
    }

    pub fn step(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
    ) -> rheon_core::Step {
        dispatch!(self, m => m.step(x, y, d_clock, weight))
    }

    /// [`Self::step`], into `out` ([`OnlineModel::step_into`]): the same
    /// numbers, with `out`'s buffers reused where the model writes in place
    /// (docs/PLAN.md task 270, M-4).
    pub fn step_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut rheon_core::Step,
    ) {
        dispatch!(self, m => m.step_into(x, y, d_clock, weight, out))
    }

    /// [`Self::step_into`], with a `marginal`'s pair work split into
    /// `shards` ranges of features and run on the pool a batch of rows at a
    /// time ([`rheon_core::Marginal::step_sharded`], docs/PLAN.md task
    /// 126). Every other model, and one shard, steps as `step_into` does.
    /// The rows a sharded step holds are learned by [`Self::flush`], which
    /// every run of rows ends with; the numbers are `step`'s to the bit.
    pub fn step_sharded_into(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        shards: usize,
        out: &mut rheon_core::Step,
    ) {
        match self {
            AnyModel::Marginal(m) if shards > 1 => {
                *out = m.step_sharded(x, y, d_clock, weight, &on_the_pool(shards));
            }
            _ => self.step_into(x, y, d_clock, weight, out),
        }
    }

    /// End a run of rows: learn every row a sharded step holds
    /// ([`Self::step_sharded_into`]).
    pub fn flush(&mut self, shards: usize) {
        if let AnyModel::Marginal(m) = self {
            m.flush(&on_the_pool(shards));
        }
    }

    /// Take the readiness shares of every solve an `ewridge`, `huber` or
    /// `quantile` left unread, dropping the factors they wait on
    /// ([`rheon_core::EwRidge::settle_readiness`], docs/PLAN.md tasks 140
    /// and 116). Once a stream's chunk is done, not once a run: a share
    /// costs a solve against the identity, about `k³`, and a chunk of a
    /// wide spec is many cache-sized runs (task 265).
    pub fn settle_readiness(&mut self) {
        match self {
            AnyModel::EwRidge(m) => m.settle_readiness(),
            AnyModel::Robust(m) => m.settle_readiness(),
            _ => {}
        }
    }

    /// The bytes the model's Huber refit buffers can hold at most
    /// (`refit_reservation`, docs/PLAN.md task 277): 0 for a model without
    /// the refit.
    pub fn refit_reservation(&self) -> usize {
        match self {
            AnyModel::Robust(m) => m.refit_reservation(),
            AnyModel::Sgd(m) => m.refit_reservation(),
            AnyModel::Marginal(m) => m.refit_reservation(),
            AnyModel::Glm(m) => m.refit_reservation(),
            AnyModel::Expectile(m) => m.refit_reservation(),
            _ => 0,
        }
    }

    /// Give the model's refits up, past its spec's `refit_budget`; a no-op
    /// for a model without the refit.
    pub fn refit_give_up(&mut self) {
        match self {
            AnyModel::Robust(m) => m.refit_give_up(),
            AnyModel::Sgd(m) => m.refit_give_up(),
            AnyModel::Marginal(m) => m.refit_give_up(),
            AnyModel::Glm(m) => m.refit_give_up(),
            AnyModel::Expectile(m) => m.refit_give_up(),
            _ => {}
        }
    }

    /// How each target's refit stands; empty for a model without it.
    pub fn refit_status(&self) -> Vec<rheon_core::RefitStatus> {
        match self {
            AnyModel::Robust(m) => m.refit_status(),
            AnyModel::Sgd(m) => m.refit_status(),
            AnyModel::Marginal(m) => m.refit_status(),
            AnyModel::Glm(m) => m.refit_status(),
            AnyModel::Expectile(m) => m.refit_status(),
            _ => Vec::new(),
        }
    }

    /// The step's answer without the step
    /// ([`OnlineModel::predict_with`]). `y` is passed because two models
    /// read a number out of the targets slot -- `bocpd`'s hazard column and
    /// `hmm`'s exogenous column -- and their answer depends on it; every
    /// other model ignores it, which is what makes `predict` out of sample
    /// (docs/REVIEW-E54-E64.md C1).
    pub fn predict(&self, x: &[f64], y: &[Option<f64>], d_clock: f64) -> rheon_core::Step {
        dispatch!(self, m => m.predict_with(x, y, d_clock))
    }

    /// [`Self::predict_into`] for a row whose stamp the stream knows: a
    /// model whose read decides a window's edge on the clock
    /// ([`ModelKind::reads_at_stamp`]) reads it at `stamp`, as a step would
    /// be handed it; every other model, and a row with no stamp, as
    /// `predict_into`.
    pub fn predict_at_into(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        stamp: Option<Stamp>,
        out: &mut rheon_core::Step,
    ) {
        match (self, stamp) {
            (AnyModel::EwmMoments(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Hawkes(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Expectile(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Isotonic(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::DdSketch(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::CorrChange(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Audit(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Krls(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            (AnyModel::Dependence(m), Some(s)) => *out = m.predict_at(x, d_clock, s),
            _ => self.predict_into(x, y, d_clock, out),
        }
    }

    /// [`Self::predict`] into the caller's step, which it overwrites in
    /// full ([`OnlineModel::predict_with_into`]).
    pub fn predict_into(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        out: &mut rheon_core::Step,
    ) {
        dispatch!(self, m => m.predict_with_into(x, y, d_clock, out))
    }

    /// [`Self::predict`] at another `min_weight` floor, set for the read and
    /// put back ([`OnlineModel::predict_with_floor`]): the prediction a
    /// floor withholds from the output, which the residual spread folds.
    pub fn predict_with_floor(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        floor: f64,
    ) -> rheon_core::Step {
        dispatch!(self, m => m.predict_with_floor(x, y, d_clock, floor))
    }

    /// [`Self::predict_with_floor`] at `stamp` for models whose read decides
    /// a window's edge on the clock ([`ModelKind::reads_at_stamp`]).
    pub fn predict_with_floor_at(
        &mut self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        stamp: Option<Stamp>,
        floor: f64,
    ) -> rheon_core::Step {
        let Some(s) = stamp else {
            return self.predict_with_floor(x, y, d_clock, floor);
        };
        match self {
            AnyModel::EwmMoments(m) => m.predict_with_floor_at(x, d_clock, s, floor),
            AnyModel::Hawkes(m) => m.predict_with_floor_at(x, d_clock, s, floor),
            AnyModel::Expectile(m) => m.predict_with_floor_at(x, d_clock, s, floor),
            AnyModel::Isotonic(m) => m.predict_with_floor_at(x, d_clock, s, floor),
            AnyModel::DdSketch(m) => m.predict_at(x, d_clock, s),
            AnyModel::CorrChange(m) => m.predict_at(x, d_clock, s),
            AnyModel::Audit(m) => m.predict_at(x, d_clock, s),
            AnyModel::Krls(m) => m.predict_with_floor_at(x, y, d_clock, floor, s),
            AnyModel::Dependence(m) => m.predict_at(x, d_clock, s),
            _ => self.predict_with_floor(x, y, d_clock, floor),
        }
    }

    /// [`Self::predict_into`] for a row at raw clock `clock`
    /// ([`OnlineModel::predict_at_into`]), which a seasonal `holt` reads
    /// its phase from.
    pub fn predict_at_clock_into(
        &self,
        x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        clock: Option<ClockValue>,
        out: &mut rheon_core::Step,
    ) {
        dispatch!(self, m => m.predict_at_into(x, y, d_clock, clock, out))
    }

    /// Hand the model the row's raw clock ([`OnlineModel::clock_next`]); a
    /// no-op for every model but a seasonal `holt`.
    pub fn clock_next(&mut self, clock: ClockValue) {
        dispatch!(self, m => m.clock_next(clock))
    }

    /// Drop the row-lagged state ([`OnlineModel::clear_lags`]); a no-op for
    /// a model that keeps none.
    pub fn clear_lags(&mut self) {
        dispatch!(self, m => m.clear_lags())
    }

    /// Hand the model the next learned row's stamp, its decayed clock held
    /// exactly ([`OnlineModel::stamp_next`], docs/PLAN.md task 175); a no-op
    /// for a model without a window.
    pub fn stamp_next(&mut self, stamp: Stamp) {
        dispatch!(self, m => m.stamp_next(stamp))
    }

    /// [`Self::stamp_next`] on a stream with no clock column
    /// ([`OnlineModel::stamp_next_on_rows`], task 278, D13).
    pub fn stamp_next_on_rows(&mut self, stamp: Stamp) {
        dispatch!(self, m => m.stamp_next_on_rows(stamp))
    }

    /// The next stepped row's clock label for a fixed-lag smoother
    /// ([`OnlineModel::label_next`]).
    pub fn label_next(&mut self, label: ClockValue) {
        dispatch!(self, m => m.label_next(label))
    }

    /// The gate withheld `target` on the row just stepped
    /// ([`OnlineModel::hold_smoothed`]).
    pub fn hold_smoothed(&mut self, target: usize) {
        dispatch!(self, m => m.hold_smoothed(target))
    }

    /// What the row's fixed-lag smoother emitted
    /// ([`OnlineModel::smoothed_into`]).
    pub fn smoothed_into(&self, out: &mut rheon_core::Smoothed) -> bool {
        dispatch!(self, m => m.smoothed_into(out))
    }

    /// Bound the model's window ([`OnlineModel::set_window_budget`]).
    pub fn set_window_budget(&mut self, budget: Option<rheon_core::WindowBudget>) {
        dispatch!(self, m => m.set_window_budget(budget))
    }

    /// Run the model's Gram work on `threads` threads
    /// ([`OnlineModel::set_gram_threads`]).
    pub fn set_gram_threads(&mut self, threads: usize) {
        dispatch!(self, m => m.set_gram_threads(threads))
    }

    /// Set the model's window edge ([`OnlineModel::set_window_closed`]).
    pub fn set_window_closed(&mut self, closed: rheon_core::WindowClosed) {
        dispatch!(self, m => m.set_window_closed(closed))
    }

    /// The model's window ring as a shadow ([`OnlineModel::window_shadow`]).
    pub fn window_shadow(&self) -> Option<WindowShadow> {
        dispatch!(self, m => m.window_shadow())
    }

    /// [`OnlineModel::window_could_refuse`].
    pub fn window_could_refuse(&self, rows: usize) -> bool {
        dispatch!(self, m => m.window_could_refuse(rows))
    }

    /// The default schedule in an explicit cadence's first period
    /// ([`OnlineModel::set_first_period_by_share`]).
    pub fn set_first_period_by_share(&mut self, on: bool) {
        dispatch!(self, m => m.set_first_period_by_share(on))
    }

    /// The weight-share cadence ([`OnlineModel::set_solve_share`]).
    pub fn set_solve_share(&mut self, share: Option<f64>) {
        dispatch!(self, m => m.set_solve_share(share))
    }

    /// The weight-share cadence the model runs by ([`OnlineModel::solve_share`]).
    pub fn solve_share(&self) -> Option<f64> {
        dispatch!(self, m => m.solve_share())
    }

    /// A refusing budget's overrun ([`OnlineModel::window_over_budget`]).
    pub fn window_over_budget(&self) -> Option<(usize, rheon_core::Cadence)> {
        dispatch!(self, m => m.window_over_budget())
    }

    /// The state's bytes past half of the cap, taken
    /// ([`OnlineModel::take_state_warning`]).
    pub fn take_state_warning(&mut self) -> Option<(u64, u64)> {
        dispatch!(self, m => m.take_state_warning())
    }

    /// The bytes a row would have taken the state to, past the cap
    /// ([`OnlineModel::state_over_cap`]).
    pub fn state_over_cap(&self) -> Option<(u64, u64)> {
        dispatch!(self, m => m.state_over_cap())
    }

    /// Each target's own weight, for the per-target warmup
    /// ([`OnlineModel::target_n_eff_into`]).
    pub fn target_n_eff_into(&self, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.target_n_eff_into(out))
    }

    /// The noise gate's statistic per slot
    /// ([`OnlineModel::error_inflation_into`]); `false` for a model
    /// without one.
    pub fn error_inflation_into(&self, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.error_inflation_into(out))
    }

    /// The same for the noise gate at `limit` on the row `x`, `d_clock`
    /// after the last, where a bound may stand in below it
    /// ([`OnlineModel::error_inflation_gate_into`]).
    pub fn error_inflation_gate_into(
        &self,
        x: &[f64],
        d_clock: f64,
        out: &mut Vec<f64>,
        limit: f64,
    ) -> bool {
        dispatch!(self, m => m.error_inflation_gate_into(x, d_clock, out, limit))
    }

    /// The same for one row's features
    /// ([`OnlineModel::row_error_inflation_into`]).
    pub fn row_error_inflation_into(&self, x: &[f64], d_clock: f64, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.row_error_inflation_into(x, d_clock, out))
    }

    /// The coefficients' sampling variances
    /// ([`OnlineModel::coef_variance`]).
    pub fn coef_variance(&self) -> Option<rheon_core::CoefVariance> {
        dispatch!(self, m => m.coef_variance())
    }

    /// Each coefficient's data share ([`OnlineModel::support_coef`]), laid
    /// out like [`Self::coefficients`].
    pub fn support_coef(&self) -> Option<Vec<Vec<f64>>> {
        dispatch!(self, m => m.support_coef())
    }

    /// Each coefficient's data share as one list, slot after slot, written
    /// over `out` ([`OnlineModel::support_coef_into`]); `false` where there
    /// is none.
    pub fn support_coef_into(&self, out: &mut Vec<f64>) -> bool {
        dispatch!(self, m => m.support_coef_into(out))
    }

    /// Whether the model is still in its own warm-up: a standardizing `sgd`
    /// or `pa` whose scaler has not switched the fit to the caller's units
    /// ([`rheon_core::Warmup`]). Every other model has none.
    pub fn scaler_warming(&self) -> bool {
        match self {
            AnyModel::Sgd(m) => m.scaler_warming(),
            AnyModel::Pa(m) => m.scaler_warming(),
            _ => false,
        }
    }

    /// The model's own wait, if it is in one, and so reports nothing:
    /// [`OwnWait::Seeding`] while it holds its first rows to seed itself
    /// from, its `warm_rows` -- `kmeans`' centres, `hmm`'s states, `bocpd`'s
    /// prior, `kalman`'s unsized priors, the rows `garch`'s start
    /// buffers for a target, and the rows before `deco`'s standardiser is
    /// warm enough for its level to learn; [`OwnWait::NotIdentified`] while a fit is
    /// withheld because its rows do not identify the model -- `garch`'s
    /// start, whose likelihood did not tell the variance process from a
    /// constant variance, and `dcc`'s, whose did not tell the correlation's
    /// recursion from constant correlation, so they buffer more rows. One
    /// dispatch a row answers both.
    /// Whether the model is one of the kinds [`Self::own_wait`] can answer
    /// for, so a row of every other kind is spared the dispatch. Keep it in
    /// step with that match: a kind missing here never reports its wait.
    pub fn waits(&self) -> bool {
        matches!(
            self,
            AnyModel::KMeans(_)
                | AnyModel::Hmm(_)
                | AnyModel::Bocpd(_)
                | AnyModel::Kalman(_)
                | AnyModel::Deco(_)
                | AnyModel::Garch(_)
                | AnyModel::Dcc(_)
                | AnyModel::VectorAr(_)
                | AnyModel::Dependence(_)
                | AnyModel::HalfSpaceTrees(_)
                | AnyModel::RandomCutForest(_)
        ) || matches!(self, AnyModel::Holt(m) if m.heuristic_start())
    }

    #[inline]
    pub fn own_wait(&self) -> OwnWait {
        match self {
            AnyModel::KMeans(m) if !m.seeded() => OwnWait::Seeding,
            AnyModel::Hmm(m) if !m.seeded() => OwnWait::Seeding,
            AnyModel::Bocpd(m) if m.warming() => OwnWait::Seeding,
            AnyModel::Kalman(m) if m.sizing() => OwnWait::Seeding,
            AnyModel::Deco(m) if m.warming() => OwnWait::Seeding,
            // A `holt` heuristic start holding a target's first rows.
            AnyModel::Holt(m) if m.starting() => OwnWait::Seeding,
            // `garch`'s start and `dcc`'s, its margins' and its own, wait two
            // ways: buffering, or buffering for want of an effect.
            AnyModel::Garch(m) => Self::start_wait(m.starting(), || m.no_effect()),
            AnyModel::VectorAr(m) if m.not_identified() => OwnWait::NotIdentified,
            AnyModel::Dependence(m) if m.warming() => OwnWait::Seeding,
            AnyModel::Dcc(m) => Self::start_wait(m.starting(), || m.no_effect()),
            AnyModel::HalfSpaceTrees(m) if m.warming() => OwnWait::Seeding,
            AnyModel::RandomCutForest(m) if m.warming() => OwnWait::Seeding,
            _ => OwnWait::None,
        }
    }

    #[inline]
    fn start_wait(starting: bool, no_effect: impl FnOnce() -> bool) -> OwnWait {
        if starting {
            OwnWait::Seeding
        } else if no_effect() {
            OwnWait::NotIdentified
        } else {
            OwnWait::None
        }
    }

    /// Whether the row's own report withheld part of what it reports, read
    /// from the step's predictions: `corrchange`'s flag, withheld where
    /// every pair over the critical value is one whose two columns' scale
    /// ratio moved inside the span (`rheon_core::CorrChange::flag_withheld`).
    /// Every other model withholds no part of a report by the row itself.
    pub fn withheld_by_its_row(&self, pred: &[f64]) -> bool {
        match self {
            AnyModel::CorrChange(_) => rheon_core::CorrChange::flag_withheld(pred),
            _ => false,
        }
    }

    /// Whether the model refuses the row's inputs `xs`, and so reports
    /// nothing for it: `garch`'s, an input below 0 under GARCH or GJR
    /// ([`rheon_core::GarchCfg::refuses_inputs`]). Every other model reads
    /// any finite feature.
    pub fn refuses_inputs(&self, xs: &[f64]) -> bool {
        match self {
            AnyModel::Garch(m) => m.cfg().refuses_inputs(xs),
            _ => false,
        }
    }

    /// Whether the row's report was withheld because its events do not
    /// identify the fit, read from the step's own report: `hawkes`'
    /// intensities, not numbers with its weight at `min_weight`, where its
    /// least-squares system was singular or too near it -- a node with no
    /// excitation over the observation, as before its first event or in a
    /// window that holds none of its events, or two terms the events cannot
    /// tell apart (`rheon_core::Hawkes::not_identified`). Every other model
    /// that waits so names its wait before the step ([`Self::own_wait`]).
    pub fn not_identified_by_its_row(&self, step: &rheon_core::Step) -> bool {
        match self {
            AnyModel::Hawkes(m) => m.not_identified(step),
            _ => false,
        }
    }

    /// Whether a slot's fit is withheld because the system it was solved
    /// from is past the condition at which a solve keeps a digit:
    /// `ewridge`'s, whose normal equations can factorize past it
    /// (`rheon_core::SpdFactor::carries_no_digit`). Every other model
    /// solves nothing of the kind.
    pub fn ill_conditioned(&self) -> bool {
        match self {
            AnyModel::EwRidge(m) => m.ill_conditioned(),
            _ => false,
        }
    }

    /// Per target, whether its predictions are withheld because its recent
    /// steps say the learning rate is past its stability bound: `sgd`'s
    /// gate (`rheon_core::Sgd::unstable_targets`), read from the state, so
    /// it names the prediction the next row makes; empty for every other
    /// model, which takes no gradient step of a rate the caller sets (task
    /// 278, D8).
    pub fn unstable_targets(&self) -> Vec<bool> {
        match self {
            AnyModel::Sgd(m) if m.unstable() => m.unstable_targets(),
            _ => Vec::new(),
        }
    }

    /// Whether some level a `tdigest` reads is withheld because the rows
    /// before it do not identify it: past the plotting position of the
    /// smallest or largest row (`rheon_core::TDigestModel::beyond_sample`).
    /// Read from the state, so it names the levels the next row reads. A
    /// bank row asks only of the two kinds here (`run_instance`'s
    /// `has_levels`, decided once an instance): keep that in step with
    /// this match.
    pub fn beyond_sample(&self) -> bool {
        match self {
            AnyModel::TDigest(m) => m.beyond_sample(),
            AnyModel::DdSketch(m) => m.beyond_sample(),
            _ => false,
        }
    }

    /// What went wrong in a model's solves, counted (docs/PLAN.md §7):
    /// `ewridge` and `robust` count their jittered or failed factorizations,
    /// `lasso` its coordinate descents that ran out of sweeps, `ew_class` the
    /// rows whose scoring met a failed factorization, `hmm` the rows its
    /// filter could not evaluate, `bocpd` the rows whose predictive could
    /// not be, `kmeans` and `hmm` the seeding's Lloyd runs that stopped at
    /// their cap short of a fixed point (task 278, D7), and `robust`'s Huber
    /// and `sgd(loss="huber")` the targets whose refit of their first rows
    /// took every IRLS pass (task 278). Every other model reports 0 because
    /// it counts nothing, not because nothing can fail: `kalman` inverts an
    /// innovation variance, and `ew_cov` solves for `mahal` and
    /// `partial_corr` (review 2026-09-12, S8).
    pub fn solve_failures(&self) -> u64 {
        match self {
            AnyModel::EwRidge(m) => m.solve_failures,
            AnyModel::Lasso(m) => m.solve_failures,
            AnyModel::Robust(m) => m.solve_failures,
            AnyModel::EwClass(m) => m.solve_failures,
            AnyModel::Hmm(m) => m.solve_failures,
            AnyModel::Bocpd(m) => m.solve_failures,
            AnyModel::Glm(m) => m.solve_failures(),
            AnyModel::Expectile(m) => m.solve_failures(),
            AnyModel::KMeans(m) => m.solve_failures,
            AnyModel::Sgd(m) => m.solve_failures(),
            AnyModel::Marginal(m) => m.solve_failures(),
            AnyModel::Rls(_)
            | AnyModel::Kalman(_)
            | AnyModel::Ftrl(_)
            | AnyModel::EwCov(_)
            | AnyModel::Pa(_)
            | AnyModel::Holt(_)
            | AnyModel::Micro(_)
            | AnyModel::SeqTest(_)
            | AnyModel::Deco(_)
            | AnyModel::Rcov(_)
            | AnyModel::CorrChange(_)
            | AnyModel::Audit(_)
            // Every read it withholds is a row's, not a failure.
            | AnyModel::Hawkes(_)
            | AnyModel::Garch(_)
            | AnyModel::Dcc(_)
            | AnyModel::Realized(_)
            | AnyModel::PowerVariation(_)
            | AnyModel::TDigest(_)
            | AnyModel::DdSketch(_)
            | AnyModel::Isotonic(_)
            | AnyModel::Krls(_)
            | AnyModel::Dependence(_)
            // A term it withholds is withheld by its condition, no failure.
            | AnyModel::EwmMoments(_) => 0,
            // The components are withheld, never failed: the rotation has no
            // factorization to fail.
            | AnyModel::Pca(_) => 0,
            // Each fit counts its own; none withholds a row as a failure.
            | AnyModel::Ar(_)
            | AnyModel::VectorAr(_)
            | AnyModel::Spectral(_)
            | AnyModel::HoeffdingTreeClassifier(_)
            | AnyModel::HoeffdingTreeRegressor(_)
            | AnyModel::Bandit(_)
            | AnyModel::HalfSpaceTrees(_)
            | AnyModel::RandomCutForest(_) => 0,
        }
    }

    pub fn n_outputs(&self) -> usize {
        dispatch!(self, m => m.n_outputs())
    }

    pub fn n_features(&self) -> usize {
        dispatch!(self, m => m.n_features())
    }

    pub fn n_targets(&self) -> usize {
        dispatch!(self, m => m.n_targets())
    }

    /// The accumulated weight behind the model as it stands: what the next
    /// row's `n_eff` field reports (before that row's update).
    pub fn n_eff(&self) -> f64 {
        dispatch!(self, m => m.n_eff())
    }

    pub fn coefficients(&self) -> Option<Vec<Vec<f64>>> {
        match self {
            AnyModel::EwRidge(m) => m.coefficients().map(|b| b.to_vec()),
            AnyModel::Rls(m) => Some(m.coefficients().to_vec()),
            // Flattened to (target x path point) rows, matching the pred slots.
            AnyModel::Lasso(m) => m
                .coefficients()
                .map(|b| b.iter().flat_map(|per_t| per_t.iter().cloned()).collect()),
            AnyModel::Kalman(m) => Some(m.coefficients()),
            AnyModel::Robust(m) => m.coefficients().map(|b| b.to_vec()),
            AnyModel::Ftrl(m) => Some(m.coefficients()),
            // ew_cov has no coefficients: its outputs are the statistics.
            AnyModel::EwCov(_) => None,
            AnyModel::Sgd(m) => Some(m.coefficients()),
            AnyModel::Pa(m) => Some(m.coefficients().to_vec()),
            AnyModel::Glm(m) => Some(m.coefficients()),
            // Absent until a target has a fit.
            AnyModel::Expectile(m) => m.has_fit().then(|| m.coefficients()),
            AnyModel::Holt(m) => Some(m.coefficients()),
            // The centres, k rows of p; absent until seeded.
            AnyModel::KMeans(m) => m.coefficients(),
            // micro: one row per potential summary -- ragged, and absent
            // until there is one.
            AnyModel::Micro(m) => m.coefficients(),
            // The class means, one row per class (NaN for a class not seen).
            AnyModel::EwClass(m) => Some(m.coefficients()),
            // seqtest has no coefficients: its outputs are the e-values and
            // the counts they are staked on.
            AnyModel::SeqTest(_) => None,
            // marginal has no coefficients and no outputs: its pairs are
            // read from the state by `Bank::marginal`.
            AnyModel::Marginal(_) => None,
            // The levels themselves, one row: `coef_fields` names one slot
            // per correlation value, term `rho`.
            AnyModel::Deco(m) => Some(vec![m.rho().to_vec()]),
            // rcov has no coefficients: its value is the block it emits at
            // the group's close.
            AnyModel::Rcov(_) => None,
            // The state means, one row per state, as `ew_class` reports its
            // class means.
            AnyModel::Hmm(m) => Some((0..m.cfg().k).map(|s| m.state_means(s).to_vec()).collect()),
            // corrchange has no coefficients: its outputs are the test.
            AnyModel::CorrChange(_) => None,
            // bocpd has none either: its value is the run-length posterior.
            AnyModel::Bocpd(_) => None,
            // audit has none: its value is what it counted, read by
            // `Bank::audit`.
            AnyModel::Audit(_) => None,
            // `μ` and the `α`s, one row per node: `coef_fields` names them.
            AnyModel::Hawkes(m) => Some(m.coefficients()),
            // Each target's process parameters, then its shape.
            AnyModel::Garch(m) => Some(m.coefficients()),
            // `a`, `b` and the asymmetric model's `g`, one row; not a number
            // before a learned model's start.
            AnyModel::Dcc(m) => Some(vec![m.coefficients()]),
            // realized has none: its outputs are the estimators.
            AnyModel::Realized(_) => None,
            // pca has none: its components are its output fields.
            AnyModel::Pca(_) => None,
            // power_variation has none: its outputs are the terms.
            AnyModel::PowerVariation(_) => None,
            // tdigest has none: its value is the levels it reads.
            AnyModel::TDigest(_) | AnyModel::DdSketch(_) => None,
            // ewm_moments has none: its outputs are the terms.
            AnyModel::EwmMoments(_) => None,
            // The constant, the lags, then the features, one row per
            // target; absent until a target has a fit.
            AnyModel::Ar(m) => m.coefficients(),
            AnyModel::VectorAr(m) => m.coefficients(),
            // Its value is the power it reports per row.
            AnyModel::Spectral(_) => None,
            // Its fit is a curve, read at each row's feature.
            AnyModel::Isotonic(_) => None,
            // Its dual weights grow with the dictionary and have no fixed term layout.
            AnyModel::Krls(_) => None,
            AnyModel::Dependence(_) => None,
            // Trees have no linear coefficients.
            AnyModel::HoeffdingTreeClassifier(_) | AnyModel::HoeffdingTreeRegressor(_) => None,
            AnyModel::Bandit(m) => Some(m.coefficients()),
            AnyModel::HalfSpaceTrees(_) | AnyModel::RandomCutForest(_) => None,
        }
    }

    pub fn state(&self) -> State {
        dispatch!(self, m => m.state())
    }

    /// The model's configuration as built, as JSON: what
    /// [`crate::resolved_defaults`] reads every model default from. Refused
    /// where the JSON would not read back as the configuration, so an
    /// infinity written as `null` cannot be pinned as one (`faithful_json`).
    pub fn cfg_json(&self) -> Result<serde_json::Value, String> {
        let mut v = dispatch!(self, m => crate::defaults::faithful_json(m.cfg()))?;
        // `ftrl`'s state writes `standardize` only when on, so that a state
        // without it keeps its bytes (PLAN 569); the resolved settings name
        // the default all the same.
        if let (AnyModel::Ftrl(m), Some(o)) = (self, v.as_object_mut()) {
            o.entry("standardize")
                .or_insert(serde_json::Value::Bool(m.cfg().standardize));
        }
        Ok(v)
    }
}
