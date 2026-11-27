//! Methods on [`ModelKind`]: registry of model names, intercept/coef flags,
//! windowing configuration, and target-prediction classification. Split out
//! of `spec.rs` to keep every source file well under the repository size cap.

use super::{
    Compare, CountOrAuto, GramThreads, ModelClosed, ModelKind, REFIT_BUDGET_MIB, SpectralSpec,
    VectorArSpec, WindowBudgetSpec, model_edge, num_label,
};
use crate::span::Span;

impl ModelKind {
    /// A `hawkes` spec's kernels, each as its decay `β` in the clock's unit
    /// and the term's name in `coef` (`beta2`, or `h5m` for a half-life):
    /// from `kernel_decays`, or `β = ln 2 / half-life` from
    /// `kernel_half_lives`. Empty for another kind and while neither is
    /// given.
    pub fn hawkes_kernels(&self) -> Vec<(f64, String)> {
        let ModelKind::Hawkes {
            kernel_decays,
            kernel_half_lives,
            ..
        } = self
        else {
            return Vec::new();
        };
        match (kernel_decays, kernel_half_lives) {
            (Some(b), _) => b
                .iter()
                .map(|b| (*b, format!("beta{}", num_label(*b))))
                .collect(),
            (None, Some(h)) => h
                .spans()
                .iter()
                .map(|h| {
                    (
                        std::f64::consts::LN_2 / h.value(),
                        format!("h{}", h.label()),
                    )
                })
                .collect(),
            (None, None) => Vec::new(),
        }
    }

    /// Every model the bank can build, by the `type` a spec names it with,
    /// in declaration order. The registry the Python side and the tests
    /// check themselves against (docs/EXTENDING.md); `kinds_tests` holds it
    /// to the enum, so a new variant fails a test until it is listed here.
    pub const KINDS: &'static [&'static str] = &[
        "ewridge",
        "lasso",
        "kalman",
        "huber",
        "quantile",
        "ftrl",
        "ew_cov",
        "sgd",
        "pa",
        "holt",
        "rls",
        "kmeans",
        "micro",
        "ew_class",
        "seqtest",
        "marginal",
        "deco",
        "rcov",
        "hmm",
        "corrchange",
        "bocpd",
        "audit",
        "glm",
        "hawkes",
        "expectile",
        "garch",
        "dcc",
        "realized",
        "pca",
        "power_variation",
        "tdigest",
        "ddsketch",
        "ewm_moments",
        "ar",
        "spectral",
        "isotonic",
        "vector_ar",
        "bandit",
        "half_space_trees",
        "random_cut_forest",
    ];

    pub fn kind_name(&self) -> &'static str {
        match self {
            ModelKind::EwRidge { .. } => "ewridge",
            ModelKind::Rls { .. } => "rls",
            ModelKind::Lasso { .. } => "lasso",
            ModelKind::Kalman { .. } => "kalman",
            ModelKind::Huber { .. } => "huber",
            ModelKind::Quantile { .. } => "quantile",
            ModelKind::Ftrl { .. } => "ftrl",
            ModelKind::EwCov { .. } => "ew_cov",
            ModelKind::Sgd { .. } => "sgd",
            ModelKind::Pa { .. } => "pa",
            ModelKind::Holt { .. } => "holt",
            ModelKind::KMeans { .. } => "kmeans",
            ModelKind::Micro { .. } => "micro",
            ModelKind::EwClass { .. } => "ew_class",
            ModelKind::SeqTest { .. } => "seqtest",
            ModelKind::Marginal { .. } => "marginal",
            ModelKind::Deco { .. } => "deco",
            ModelKind::Rcov { .. } => "rcov",
            ModelKind::Hmm { .. } => "hmm",
            ModelKind::CorrChange { .. } => "corrchange",
            ModelKind::Bocpd { .. } => "bocpd",
            ModelKind::Audit { .. } => "audit",
            ModelKind::Glm { .. } => "glm",
            ModelKind::Hawkes { .. } => "hawkes",
            ModelKind::Expectile { .. } => "expectile",
            ModelKind::Garch { .. } => "garch",
            ModelKind::Dcc { .. } => "dcc",
            ModelKind::Realized { .. } => "realized",
            ModelKind::Pca { .. } => "pca",
            ModelKind::PowerVariation { .. } => "power_variation",
            ModelKind::TDigest { .. } => "tdigest",
            ModelKind::DdSketch { .. } => "ddsketch",
            ModelKind::EwmMoments { .. } => "ewm_moments",
            ModelKind::Ar { .. } => "ar",
            ModelKind::Spectral(_) => "spectral",
            ModelKind::Isotonic { .. } => "isotonic",
            ModelKind::VectorAr(_) => "vector_ar",
            ModelKind::Bandit(_) => "bandit",
            ModelKind::HalfSpaceTrees(_) => "half_space_trees",
            ModelKind::RandomCutForest(_) => "random_cut_forest",
        }
    }

    /// True for the model that reads every row whatever its columns hold:
    /// `audit`, which counts what the others skip (docs/PLAN.md task 223
    /// (b)). Every other model's row is skipped at a feature that is not
    /// usable.
    pub fn reads_every_row(&self) -> bool {
        matches!(self, ModelKind::Audit { .. })
    }

    /// The column a model with no target reads from the targets slot, when
    /// the spec names one: `bocpd`'s `hazard_col` and `hmm`'s `exog_tvtp`.
    /// It fills `targets` ([`Spec::fill_defaults`]) so the column is read
    /// from the targets slot (review 2026-09-12, C20, S24).
    pub fn targets_slot_column(&self) -> Option<&str> {
        match self {
            ModelKind::Bocpd { hazard_col, .. } => hazard_col.as_deref(),
            ModelKind::Hmm { exog_tvtp, .. } => exog_tvtp.as_deref(),
            _ => None,
        }
    }

    /// True for the models whose fit has a constant term of its own, named
    /// `intercept` in `coef`'s terms (and in the Gram's columns, where the
    /// model keeps one) when `fit_intercept` is on. An allowlist: a list of
    /// the models without one refused a feature named `intercept` on six
    /// kinds that have no such term (review 2026-11-23, SPEC-6), and
    /// `tests/test_model_registry.py` holds it to the terms `coef_index`
    /// gives.
    pub fn names_its_intercept(&self) -> bool {
        matches!(
            self,
            ModelKind::EwRidge { .. }
                | ModelKind::Rls { .. }
                | ModelKind::Lasso { .. }
                | ModelKind::Kalman { .. }
                | ModelKind::Huber { .. }
                | ModelKind::Quantile { .. }
                | ModelKind::Ftrl { .. }
                | ModelKind::Sgd { .. }
                | ModelKind::Pa { .. }
                | ModelKind::Glm { .. }
                | ModelKind::Expectile { .. }
                | ModelKind::Ar { .. }
                | ModelKind::VectorAr(_)
                | ModelKind::Bandit(_)
        )
    }

    /// False for the models that report no coefficients -- no `coef` field
    /// in their output: `ew_cov`, `seqtest`, `marginal`, `rcov`, `corrchange`,
    /// `bocpd`, `audit`, `realized`, `tdigest`, `ewm_moments`, `spectral`,
    /// `isotonic`, `half_space_trees` and `random_cut_forest`.
    /// `tests/test_model_registry.py` holds this to the field names, which are
    /// rendered elsewhere.
    pub fn has_coef(&self) -> bool {
        !matches!(
            self,
            ModelKind::EwCov { .. }
                | ModelKind::SeqTest { .. }
                | ModelKind::Marginal { .. }
                | ModelKind::Rcov { .. }
                | ModelKind::CorrChange { .. }
                | ModelKind::Bocpd { .. }
                | ModelKind::Audit { .. }
                | ModelKind::Realized { .. }
                | ModelKind::Pca { .. }
                | ModelKind::PowerVariation { .. }
                | ModelKind::TDigest { .. }
                | ModelKind::DdSketch { .. }
                | ModelKind::EwmMoments { .. }
                | ModelKind::Spectral(_)
                | ModelKind::Isotonic { .. }
                | ModelKind::HalfSpaceTrees(_)
                | ModelKind::RandomCutForest(_)
        )
    }

    /// `gram_threads` as the spec gives it, for the two kinds that take it
    /// (docs/PLAN.md task 225); `None` for every other kind.
    pub fn gram_threads_given(&self) -> Option<GramThreads> {
        match self {
            ModelKind::EwRidge { gram_threads, .. } | ModelKind::EwCov { gram_threads, .. } => {
                *gram_threads
            }
            _ => None,
        }
    }

    /// The threads a model's Gram update may run on, as the model takes
    /// it ([`rheon_core::OnlineModel::set_gram_threads`]), before the bank
    /// shares its pool out: the spec's count, or [`usize::MAX`], every
    /// thread of the pool, for `"auto"` and for none given (task 253: the
    /// default is `"auto"`). The model caps it at the pool it runs on, and
    /// the bank sets [`Self::gram_threads_sharing`] on each chunk.
    /// Configuration, not state: the stream sets it on every model it
    /// builds or restores.
    pub fn gram_threads(&self) -> usize {
        self.gram_threads_sharing(usize::MAX)
    }

    /// [`Self::gram_threads`] when the bank's pool is shared: `share` is
    /// [`crate::pool::gram_share`]`(T, P)` for the chunk's `T` parallel
    /// tasks on a pool of `P`. `"auto"`, and none given, take the share, so
    /// a model splits its rows only where the pool has idle threads; a count
    /// is the caller's, capped at the pool by the model as before.
    pub fn gram_threads_sharing(&self, share: usize) -> usize {
        match self.gram_threads_given() {
            Some(CountOrAuto::Count(n)) => n,
            Some(CountOrAuto::Auto) | None => share,
        }
    }

    /// Whether a stream of this kind keeps a window's ring: a model's, or a
    /// spread's cut from it, both of which need a `window_size`. Without one
    /// the window pre-pass has nothing to replay (task 270, G-1).
    pub fn keeps_window_ring(&self) -> bool {
        self.window_parts()
            .is_some_and(|(window, _)| window.is_some())
    }

    /// A windowed kind's `window_size` and `window_budget`; `None` for a kind with
    /// no window to bound.
    pub fn window_parts(&self) -> Option<(Option<f64>, Option<WindowBudgetSpec>)> {
        match self {
            ModelKind::EwRidge {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Lasso {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwCov {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::EwClass {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Marginal {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Ar {
                window_size: window,
                window_budget,
                ..
            }
            | ModelKind::Spectral(SpectralSpec {
                window_size: window,
                window_budget,
                ..
            })
            | ModelKind::VectorAr(VectorArSpec {
                window_size: window,
                window_budget,
                ..
            }) => Some((window.as_ref().map(Span::value), *window_budget)),
            _ => None,
        }
    }

    /// The MiB a spec's groups' Huber refit buffers may reserve in a bank
    /// (docs/PLAN.md task 277): `refit_budget`, [`REFIT_BUDGET_MIB`] by default, `None` for
    /// no bound (`inf`) or a kind without the refit.
    pub fn refit_budget_mib(&self) -> Option<f64> {
        let given = match self {
            ModelKind::Huber { refit_budget, .. }
            | ModelKind::Quantile { refit_budget, .. }
            | ModelKind::Glm { refit_budget, .. }
            | ModelKind::Expectile { refit_budget, .. } => refit_budget,
            ModelKind::Sgd {
                refit_budget, loss, ..
            } if loss.as_deref() == Some("huber") => refit_budget,
            ModelKind::Marginal {
                refit_budget,
                bin_stat,
                ..
            } if bin_stat.as_deref() == Some("huber") => refit_budget,
            _ => return None,
        };
        let mib = given.map_or(REFIT_BUDGET_MIB, |n| n.0);
        mib.is_finite().then_some(mib)
    }

    /// The budget a model's window runs under: the spec's, or refuse past
    /// 256 MiB where it names none (`rheon_core::WindowBudget::DEFAULT`).
    /// `None` without a window.
    pub fn window_budget(&self) -> Option<rheon_core::WindowBudget> {
        let (window, budget) = self.window_parts()?;
        window.map(|_| budget.map_or(rheon_core::WindowBudget::DEFAULT, WindowBudgetSpec::to_core))
    }

    /// A windowed kind's `snapshot_every` and `max_rows_between_snapshots`, as
    /// the spec gives them; `None` for a kind with no window (docs/PLAN.md
    /// task 162).
    pub fn window_cadence(&self) -> Option<(Option<&Span>, Option<u32>)> {
        match self {
            ModelKind::EwRidge {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Lasso {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwCov {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::EwClass {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Marginal {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Ar {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }
            | ModelKind::Spectral(SpectralSpec {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            })
            | ModelKind::VectorAr(VectorArSpec {
                snapshot_every,
                max_rows_between_snapshots,
                ..
            }) => Some((snapshot_every.as_ref(), *max_rows_between_snapshots)),
            _ => None,
        }
    }

    /// A windowed kind's `closed`, as the spec gives it; `None` for a kind
    /// with no window (docs/PLAN.md task 196, N17).
    pub fn window_closed(&self) -> Option<ModelClosed> {
        match self {
            ModelKind::EwRidge { closed, .. }
            | ModelKind::Lasso { closed, .. }
            | ModelKind::EwCov { closed, .. }
            | ModelKind::EwClass { closed, .. }
            | ModelKind::Marginal { closed, .. }
            | ModelKind::Ar { closed, .. }
            | ModelKind::Spectral(SpectralSpec { closed, .. })
            | ModelKind::VectorAr(VectorArSpec { closed, .. }) => Some(*closed),
            _ => None,
        }
    }

    /// The edge a windowed model's ring keeps ([`rheon_core::WindowClosed`]):
    /// the spec's `closed`, `Right` by default and for a kind with no window,
    /// which ignores it. A `closed` a model refuses never gets here: it is
    /// refused as the spec is read ([`ModelClosed`]).
    pub fn window_edge(&self) -> rheon_core::WindowClosed {
        self.window_closed().map(model_edge).unwrap_or_default()
    }

    /// Every `max_rows_between_*` the model takes, as given, beside the
    /// clock parameter it caps: what [`Spec::validate`] holds to `>= 1`
    /// (docs/PLAN.md task 196, U7). The spec's own `max_rows_between_coefs`
    /// is checked there.
    pub fn row_caps(&self) -> Vec<(&'static str, Option<u32>, &'static str)> {
        let mut caps = Vec::new();
        match self {
            ModelKind::EwRidge {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Lasso {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Huber {
                max_rows_between_solves,
                ..
            }
            | ModelKind::Quantile {
                max_rows_between_solves,
                ..
            }
            | ModelKind::VectorAr(VectorArSpec {
                max_rows_between_solves,
                ..
            }) => caps.push((
                "max_rows_between_solves",
                *max_rows_between_solves,
                "solve_every",
            )),
            ModelKind::EwCov {
                max_rows_between_pcas,
                ..
            } => caps.push(("max_rows_between_pcas", *max_rows_between_pcas, "pca_every")),
            ModelKind::Micro {
                max_rows_between_prunes,
                ..
            } => caps.push((
                "max_rows_between_prunes",
                *max_rows_between_prunes,
                "prune_every",
            )),
            _ => {}
        }
        if let Some((_, rows)) = self.window_cadence() {
            caps.push(("max_rows_between_snapshots", rows, "snapshot_every"));
        }
        caps
    }

    /// The cadence a windowed model's ring takes its snapshots on, as the
    /// model maps its configuration ([`rheon_core::Cadence::of`]): every
    /// row unless given (docs/PLAN.md task 162).
    pub fn snapshot_cadence(&self) -> Option<rheon_core::Cadence> {
        let (every, rows) = self.window_cadence()?;
        Some(rheon_core::Cadence::of(
            every.map(Span::value),
            rows.map(|r| r as usize),
        ))
    }

    /// The `window_size` and snapshot cadence of a windowed model that
    /// predicts a target: what the stream cuts its residual spread with
    /// (review 2026-09-12, S1), the model's own cadence, so the spread's
    /// boundary is the fit's. The other windowed models predict none, so
    /// the stream keeps no spread for them.
    pub fn window_and_cadence(&self) -> Option<(f64, rheon_core::Cadence)> {
        match self {
            ModelKind::EwRidge {
                window_size: Some(w),
                ..
            }
            | ModelKind::Lasso {
                window_size: Some(w),
                ..
            }
            | ModelKind::Ar {
                window_size: Some(w),
                ..
            }
            | ModelKind::Spectral(SpectralSpec {
                window_size: Some(w),
                ..
            })
            | ModelKind::VectorAr(VectorArSpec {
                window_size: Some(w),
                ..
            }) => Some((w.value(), self.snapshot_cadence()?)),
            _ => None,
        }
    }

    /// True for the models that predict no target as a number: the
    /// unsupervised ones, `ew_class`, whose target is a label it
    /// classifies, `bandit`, which recommends an arm and scores each arm,
    /// `seqtest`, whose targets are the signs it tests, and
    /// `marginal`, whose targets are the columns it correlates the features
    /// with. Their outputs are statistics, assignments, posteriors or
    /// e-values read from the state *before* each row (or, for `marginal`,
    /// nothing at all), and nothing residual-based (`sigma`, `zscore`,
    /// metrics, quantiles, conformal, autocorrelation, drift, selection,
    /// averaging) applies to them; their slots are whatever rides in
    /// `pred`, not targets × combos.
    pub fn predicts_no_target(&self) -> bool {
        self.is_unsupervised()
            || matches!(
                self,
                ModelKind::EwClass { .. }
                    | ModelKind::SeqTest { .. }
                    | ModelKind::Marginal { .. }
                    | ModelKind::Hawkes { .. }
                    | ModelKind::Garch { .. }
                    | ModelKind::Dcc { .. }
                    | ModelKind::Realized { .. }
                    | ModelKind::PowerVariation { .. }
                    | ModelKind::TDigest { .. }
                    | ModelKind::DdSketch { .. }
                    | ModelKind::Spectral(_)
                    | ModelKind::Bandit(_)
            )
    }

    /// The two specs a `seqtest` compares, when it compares rather than
    /// tests columns. Such a spec's targets name residual fields, read from
    /// the two specs' own output in the bank, not columns of the frame; the
    /// bank runs it after them.
    pub fn compares(&self) -> Option<Compare<'_>> {
        match self {
            ModelKind::SeqTest {
                a: Some(a),
                b: Some(b),
                a_suffix,
                b_suffix,
            } => Some(Compare {
                a,
                b,
                a_suffix: a_suffix.as_deref().unwrap_or(""),
                b_suffix: b_suffix.as_deref().unwrap_or(""),
            }),
            _ => None,
        }
    }
}
