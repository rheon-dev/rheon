//! `StreamPca`: principal components of a vector stream in `O(q·k)` state,
//! by one of three recursions, with the components and the row's scores
//! emitted on every row.
//!
//! Every method keeps the decayed, weighted mean and the trace of the
//! decayed, weighted covariance exactly, in the mean form every moment
//! model here uses ([`crate::EwDiag`]), and `q = min(n_components + 1, k)`
//! directions with the variance along each:
//!
//! ```text
//! W'  = lam W + w,      a = lam W / W',      b = w / W'     (a + b = 1)
//! d   = x - m           (the row's deviation from the mean before it)
//! m'  = m + b d
//! C'  = a C + a b d d'  (the covariance the components are those of)
//! tr' = a tr + a b |d|^2
//! ```
//!
//! `C` is never formed. The three methods estimate its top eigenvectors
//! from the rows:
//!
//! **`svd`, the incremental SVD** (Brand 2002; Ross et al. 2008, the update
//! scikit-learn's `IncrementalPCA` runs, here one row at a time and in the
//! mean form). Keep `C ~ V' diag(s^2) V`, `V` the `q` directions as unit
//! rows. Stack
//!
//! ```text
//! S = [ sqrt(a) diag(s) V ]     q rows
//!     [ sqrt(a b) d'      ]     1 row
//! ```
//!
//! so that `S'S` is exactly the `C'` above with `C` replaced by
//! `V' diag(s^2) V`. A one-sided Jacobi rotation of `S`'s rows until they
//! are orthogonal gives `S`'s right singular vectors and values: each row
//! is `s'_i v'_i`. The `q` longest are kept, longest first. It is exact
//! while the data's rank is at most `q` -- `n_components = k`, or a stream
//! of lower rank -- and otherwise drops, on every row, the variance outside
//! the `q` kept directions, and never gets it back.
//!
//! **`ccipca`, candid covariance-free incremental PCA** (Weng, Zhang and
//! Hwang 2003), in the mean form, its forgetting the model's decay. Each
//! direction is kept as `v_i = l_i e_i`, its variance times its unit vector:
//!
//! ```text
//! u_0  = d
//! v_i' = a v_i + a b (u_i . e_i) u_i                      (e_i = v_i / |v_i|)
//! u_{i+1} = u_i - (u_i . e_i') e_i'                       (deflation)
//! ```
//!
//! A direction not yet started takes the first deviation left over by the
//! ones before it, as the update would leave it were it already along it:
//! `v_i' = a b |u_i| u_i`.
//!
//! **`oja`, Oja's rule** for the top `q` directions (the stochastic
//! orthogonal iteration of Oja 1982 and Oja and Karhunen 1985), its rate
//! the row's share of the weight over the total variance, times a gain:
//!
//! ```text
//! eta  = learning_rate * b / tr'
//! V'   = GramSchmidt(V + eta (V d) d')          (rows in order)
//! l_i' = a l_i + a b (e_i . d)^2                (the variance along e_i)
//! ```
//!
//! A direction not yet started takes the deviation's residual against the
//! started ones, as `ccipca`'s does.
//!
//! **Which components are emitted, and when.** Component `j` is emitted
//! while the data tells it apart from its neighbours: two components whose
//! eigenvalues are equal are not identified, any rotation of the two being
//! as good, and two whose sample eigenvalues differ by no more than their
//! sampling error may be equal. The test is the likelihood ratio for the
//! equality of two eigenvalues of a normal covariance (Anderson 1963), with
//! `N` Kish's effective number of rows, `W^2 / sum w^2` under the weights
//! and the decay:
//!
//! ```text
//! (N - 1) ln( ((l + m) / 2)^2 / (l m) )  >  9.2103   (chi-squared, 2 df, at 1%)
//! ```
//!
//! for `j` and the component above it, and for `j` and the one below. A
//! neighbour of no variance (below the trace's rounding) is told apart once
//! `N` exceeds the rows that could leave it at none, `N > i + 1` for the
//! neighbour at index `i`, since `N` rows span at most `N - 1` dimensions.
//! `j` itself needs a variance above the trace's rounding. One direction
//! more than `n_components` is kept so the last emitted component has a
//! neighbour to be told apart from. No count of rows enters: the first rows
//! are withheld because `N` is small, and a stream whose spectrum has a tie
//! is withheld at the tie for as long as the test cannot tell it apart.
//! Under `svd` truncated below the data's rank the kept neighbour's
//! variance is an underestimate, so the gap reads wider than it is.
//!
//! **Order and sign.** `svd` emits its components by variance, largest
//! first; `ccipca` and `oja` in their deflation order, which is that order
//! once they have converged. A component's sign is arbitrary, so each is
//! signed for continuity with the same component on the row before (`e' . e
//! >= 0`), and a direction with no predecessor makes its largest-magnitude
//! entry positive, the first on a tie: the rule `ew_cov`'s components keep.
//!
//! **Out of sample.** A row's scores, `e_j . (x - m)`, read the directions
//! and the mean before the row, as every prediction here does.
//!
//! **No window.** None of the three can remove a row it has folded in. The
//! exact windowed components are `ew_cov`'s `pca` over a `window`.

use serde::{Deserialize, Serialize};

use crate::{EwDiag, ModelState, OnlineModel, State, StateError, Step};

/// `max_state_bytes` where a spec sets none: 256 KB a group.
pub const PCA_MAX_STATE_BYTES: u64 = crate::clock_span::DEFAULT_STATE_BYTES;

/// The sweeps of the Jacobi rotation a row may take. The rows it rotates
/// are orthogonal but for the new one, so a row takes two to four; the cap
/// is never met on a finite row, and a row that met it would keep the
/// rotations it made.
const MAX_SWEEPS: usize = 64;

/// The recursion the components are estimated by (the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PcaMethod {
    /// The incremental SVD: exact while the data's rank is at most the kept
    /// directions.
    #[default]
    Svd,
    /// Candid covariance-free incremental PCA.
    Ccipca,
    /// Oja's rule, with a gain on its rate.
    Oja,
}

impl PcaMethod {
    pub fn name(self) -> &'static str {
        match self {
            PcaMethod::Svd => "svd",
            PcaMethod::Ccipca => "ccipca",
            PcaMethod::Oja => "oja",
        }
    }
}

/// `oja`'s gain where a spec sets none.
pub const OJA_DEFAULT_LEARNING_RATE: f64 = 2.0;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamPcaCfg {
    pub n_features: usize,
    /// Components emitted, `1..=n_features`; one more direction is kept
    /// where there is room, as the last one's neighbour (the module docs).
    pub n_components: usize,
    #[serde(default)]
    pub method: PcaMethod,
    pub decay: crate::Decay,
    /// The weight below which nothing is emitted. The components' own gate
    /// is the eigengap rule of the module docs; this floor is on top of it.
    #[serde(default)]
    pub min_weight: f64,
    /// `oja`'s gain on its rate, `eta = learning_rate * b / tr'`: above 0
    /// and finite. Refused under the other methods, which take no rate.
    #[serde(default)]
    pub learning_rate: Option<f64>,
    /// The most bytes a group's state may hold; the state's size is fixed
    /// by `n_features` and `n_components`, so it is checked as the model is
    /// built. `None` is [`PCA_MAX_STATE_BYTES`].
    #[serde(default)]
    pub max_state_bytes: Option<u64>,
}

impl StreamPcaCfg {
    /// The directions kept: one more than emitted, where there is room.
    pub fn kept(&self) -> usize {
        (self.n_components + 1).min(self.n_features)
    }

    /// Slots a row emits: per component its variance, its share of the
    /// total variance, its `k` loadings and the row's score.
    pub fn n_outputs(&self) -> usize {
        self.n_components * (self.n_features + 3)
    }

    /// The state's bytes as numbers: the weight, its sum of squares, the
    /// mean and its low parts, the variances, the directions and their
    /// variances: `8 (2 + 3k + q k + q)`.
    pub fn state_bytes(&self) -> u64 {
        let k = self.n_features as u64;
        let q = self.kept() as u64;
        8 * (2 + 3 * k + q * k + q)
    }

    fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(PCA_MAX_STATE_BYTES)
    }

    pub fn learning_rate(&self) -> f64 {
        self.learning_rate.unwrap_or(OJA_DEFAULT_LEARNING_RATE)
    }

    pub fn validate(&self) -> Result<(), String> {
        let mut no = crate::Refusals::new();
        no.take(self.decay.check().map_err(|e| format!("pca: {e}")));
        if self.n_features == 0 {
            no.refuse(
                "pca: the model needs at least one feature, but it has none. Name at least one \
                 column in features.",
            );
        }
        if self.n_components == 0 {
            no.refuse(
                "pca: n_components must be at least 1, but it is 0. Set n_components to the \
                 number of components to emit.",
            );
        } else if self.n_features > 0 && self.n_components > self.n_features {
            no.refuse(format!(
                "pca: n_components must be at most the number of features ({}), but it is {}. \
                 Set n_components to {} or fewer.",
                self.n_features, self.n_components, self.n_features
            ));
        }
        if self.min_weight.is_nan() || self.min_weight < 0.0 {
            no.refuse(format!(
                "pca: min_weight must be at least 0, but it is {}. Set min_weight to 0, or to a \
                 positive weight.",
                self.min_weight
            ));
        }
        match (self.method, self.learning_rate) {
            (PcaMethod::Oja, Some(r)) if !(r.is_finite() && r > 0.0) => no.refuse(format!(
                "pca: learning_rate must be above 0 and finite, but it is {r}. Set it to a \
                 positive gain, or leave it out for {OJA_DEFAULT_LEARNING_RATE}."
            )),
            (PcaMethod::Oja, _) | (_, None) => {}
            (m, Some(_)) => no.refuse(format!(
                "pca: learning_rate is the rate of method \"oja\", but the method is \"{}\", \
                 which takes no rate. Leave out learning_rate, or set method to \"oja\".",
                m.name()
            )),
        }
        no.take(crate::clock_span::check_max_state_bytes(
            "pca",
            self.max_state_bytes,
        ));
        no.stop_here()?;
        let (bytes, cap) = (self.state_bytes(), self.cap());
        if bytes > cap {
            no.refuse(format!(
                "pca: the state of {} features and {} components holds {bytes} bytes a group, \
                 past max_state_bytes of {cap}. Lower n_components, or raise max_state_bytes.",
                self.n_features, self.n_components
            ));
        }
        no.finish()
    }
}

/// Working space kept between rows so a row allocates nothing: the row's
/// deviation, what is left of it, the directions before the row, the stack
/// the rotation turns, and its rows' lengths and order. Not state, and
/// equal whatever it holds.
#[derive(Debug, Clone, Default)]
struct Scratch {
    d: Vec<f64>,
    u: Vec<f64>,
    v: Vec<f64>,
    old: Vec<f64>,
    stack: Vec<f64>,
    norms: Vec<f64>,
    order: Vec<usize>,
}

impl PartialEq for Scratch {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// Streaming principal components (the module docs).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StreamPca {
    cfg: StreamPcaCfg,
    /// The weight, the mean (as its pairs) and the variances, whose sum is
    /// the total variance the shares are of.
    moments: EwDiag,
    /// `sum w^2` under the decay: `lam^2 Q + w^2`. Kish's `N` is `W^2 / Q`.
    q_sum: f64,
    /// The `q` directions, row-major `q * k`, each a unit vector or all
    /// zeros for a direction not yet started.
    rows: Vec<f64>,
    /// The variance along each direction, in the mean form.
    vals: Vec<f64>,
    #[serde(skip)]
    scratch: Scratch,
}

impl StreamPca {
    pub fn new(cfg: StreamPcaCfg) -> Result<Self, String> {
        cfg.validate()?;
        let (k, q) = (cfg.n_features, cfg.kept());
        Ok(Self {
            moments: EwDiag::new(k),
            q_sum: 0.0,
            rows: vec![0.0; q * k],
            vals: vec![0.0; q],
            scratch: Scratch::default(),
            cfg,
        })
    }

    pub fn cfg(&self) -> &StreamPcaCfg {
        &self.cfg
    }

    pub fn n_eff(&self) -> f64 {
        self.moments.n_eff()
    }

    /// Kish's effective number of rows, `W^2 / sum w^2`; 0 before a row of
    /// weight.
    pub fn n_kish(&self) -> f64 {
        let w = self.moments.n_eff();
        if self.q_sum > 0.0 {
            w * w / self.q_sum
        } else {
            0.0
        }
    }

    /// The total variance, the sum of the variances (mean form).
    pub fn trace(&self) -> f64 {
        (0..self.cfg.n_features)
            .map(|i| self.moments.var(i))
            .sum::<f64>()
    }

    /// The kept directions, row-major `q * k`, and the variance along each
    /// (mean form): the state as it stands.
    pub fn directions(&self) -> (&[f64], &[f64]) {
        (&self.rows, &self.vals)
    }

    /// Whether component `j` is told apart from its neighbours (the module
    /// docs' rule), from the state as it stands.
    pub fn separated(&self, j: usize) -> bool {
        separated(
            &self.vals,
            j,
            self.n_kish(),
            self.trace(),
            self.cfg.n_features,
        )
    }

    /// The row's slots into `out` (the module docs), from the state before
    /// the row.
    fn read_into(&self, x: &[f64], out: &mut Vec<f64>) {
        let k = self.cfg.n_features;
        let r = self.cfg.n_components;
        out.clear();
        out.resize(self.cfg.n_outputs(), f64::NAN);
        let w = self.moments.n_eff();
        if !(w >= self.cfg.min_weight) {
            return;
        }
        let tr = self.trace();
        let n = self.n_kish();
        // The unbiased factor under unequal weights, `W^2 / (W^2 - sum w^2)`
        // (scikit-learn's `n - 1` at unit weights).
        let w2 = w * w;
        let unbias = w2 / (w2 - self.q_sum);
        for j in 0..r {
            if !separated(&self.vals, j, n, tr, k) {
                continue;
            }
            let e = &self.rows[j * k..(j + 1) * k];
            let slot = &mut out[j * (k + 3)..(j + 1) * (k + 3)];
            slot[0] = unbias * self.vals[j];
            slot[1] = self.vals[j] / tr;
            slot[2..2 + k].copy_from_slice(e);
            let mut score = 0.0;
            for (i, ei) in e.iter().enumerate() {
                score += ei * self.moments.deviation(i, x[i]);
            }
            slot[2 + k] = score;
        }
    }

    /// The directions after one row of deviation `scratch.d`, weight share
    /// `b` and carry `a` (the module docs).
    fn learn(&mut self, a: f64, b: f64, tr_new: f64) {
        let ab = a * b;
        if a <= 0.0 {
            // The whole history forgotten (the first row, or a decay that
            // underflowed): no direction survives it, and one row of no
            // spread starts none.
            self.rows.fill(0.0);
            self.vals.fill(0.0);
            return;
        }
        if ab <= 0.0 {
            // Nothing added: a carry of exactly 1 leaves every direction.
            return;
        }
        let mut old = std::mem::take(&mut self.scratch.old);
        old.clear();
        old.extend_from_slice(&self.rows);
        match self.cfg.method {
            PcaMethod::Svd => self.learn_svd(a, ab),
            PcaMethod::Ccipca => self.learn_ccipca(a, ab),
            PcaMethod::Oja => self.learn_oja(a, ab, b, tr_new),
        }
        sign_for_continuity(&mut self.rows, &old, self.cfg.n_features);
        self.scratch.old = old;
    }

    fn learn_svd(&mut self, a: f64, ab: f64) {
        let k = self.cfg.n_features;
        let q = self.vals.len();
        let sc = &mut self.scratch;
        let s = &mut sc.stack;
        s.clear();
        for j in 0..q {
            let f = (a * self.vals[j]).sqrt();
            s.extend(self.rows[j * k..(j + 1) * k].iter().map(|v| f * v));
        }
        let f = ab.sqrt();
        s.extend(sc.d.iter().map(|v| f * v));
        orthogonalize_rows(s, q + 1, k);
        // Longest first, the earlier row on a tie, so the order is the
        // arithmetic's alone.
        sc.norms.clear();
        sc.norms
            .extend((0..=q).map(|i| dot(row(s, i, k), row(s, i, k))));
        sc.order.clear();
        sc.order.extend(0..=q);
        let norms = &sc.norms;
        sc.order
            .sort_by(|&i, &j| norms[j].total_cmp(&norms[i]).then(i.cmp(&j)));
        for (slot, &i) in sc.order.iter().take(q).enumerate() {
            let n2 = norms[i];
            let dst = &mut self.rows[slot * k..(slot + 1) * k];
            if n2 > 0.0 && n2.is_finite() {
                let n = n2.sqrt();
                for (o, v) in dst.iter_mut().zip(row(s, i, k)) {
                    *o = v / n;
                }
                self.vals[slot] = n2;
            } else {
                dst.fill(0.0);
                self.vals[slot] = 0.0;
            }
        }
    }

    fn learn_ccipca(&mut self, a: f64, ab: f64) {
        let k = self.cfg.n_features;
        let q = self.vals.len();
        let sc = &mut self.scratch;
        sc.u.clear();
        sc.u.extend_from_slice(&sc.d);
        let d2 = dot(&sc.d, &sc.d);
        for j in 0..q {
            let e = &mut self.rows[j * k..(j + 1) * k];
            let l = self.vals[j];
            let u = &mut sc.u;
            let v = &mut sc.v;
            v.clear();
            if l > 0.0 {
                let p = dot(u, e);
                v.extend(
                    e.iter()
                        .zip(u.iter())
                        .map(|(ei, ui)| a * l * ei + ab * p * ui),
                );
            } else {
                // Not started: the residual left by the directions before,
                // where it is more than their rounding.
                let u2 = dot(u, u);
                if !(u2 > started_floor(d2, k)) {
                    continue;
                }
                let nu = u2.sqrt();
                v.extend(u.iter().map(|ui| ab * nu * ui));
            }
            let n2 = dot(v, v);
            if !(n2 > 0.0 && n2.is_finite()) {
                e.fill(0.0);
                self.vals[j] = 0.0;
                continue;
            }
            let n = n2.sqrt();
            for (ei, vi) in e.iter_mut().zip(v.iter()) {
                *ei = vi / n;
            }
            self.vals[j] = n;
            let p = dot(u, e);
            for (ui, ei) in u.iter_mut().zip(e.iter()) {
                *ui -= p * ei;
            }
        }
    }

    fn learn_oja(&mut self, a: f64, ab: f64, b: f64, tr_new: f64) {
        let k = self.cfg.n_features;
        let q = self.vals.len();
        let eta = self.cfg.learning_rate() * b / tr_new;
        if !(eta.is_finite() && tr_new > 0.0) {
            // No spread yet to learn a direction from: the variances age.
            for l in &mut self.vals {
                *l *= a;
            }
            return;
        }
        let sc = &mut self.scratch;
        let d = &sc.d;
        let d2 = dot(d, d);
        for j in 0..q {
            let e = &mut self.rows[j * k..(j + 1) * k];
            if e.iter().all(|v| *v == 0.0) {
                continue;
            }
            let p = dot(e, d);
            for (ei, di) in e.iter_mut().zip(d) {
                *ei += eta * p * di;
            }
            self.vals[j] = a * self.vals[j] + ab * p * p;
        }
        // The started directions, orthonormal in order (modified
        // Gram-Schmidt); then at most one direction started from what is
        // left of the row.
        let mut started = 0;
        for j in 0..q {
            let (done, rest) = self.rows.split_at_mut(j * k);
            let e = &mut rest[..k];
            if e.iter().all(|v| *v == 0.0) {
                continue;
            }
            for i in 0..j {
                let prev = &done[i * k..(i + 1) * k];
                if prev.iter().all(|v| *v == 0.0) {
                    continue;
                }
                let p = dot(e, prev);
                for (ei, pi) in e.iter_mut().zip(prev) {
                    *ei -= p * pi;
                }
            }
            let n2 = dot(e, e);
            if n2 > 0.0 && n2.is_finite() {
                let n = n2.sqrt();
                for ei in e.iter_mut() {
                    *ei /= n;
                }
                started = j + 1;
            } else {
                e.fill(0.0);
                self.vals[j] = 0.0;
            }
        }
        if started < q
            && self.rows[started * k..(started + 1) * k]
                .iter()
                .all(|v| *v == 0.0)
        {
            let u = &mut sc.u;
            u.clear();
            u.extend_from_slice(d);
            for i in 0..started {
                let e = &self.rows[i * k..(i + 1) * k];
                let p = dot(u, e);
                for (ui, ei) in u.iter_mut().zip(e) {
                    *ui -= p * ei;
                }
            }
            let u2 = dot(u, u);
            if u2 > started_floor(d2, k) {
                let n = u2.sqrt();
                let e = &mut self.rows[started * k..(started + 1) * k];
                for (ei, ui) in e.iter_mut().zip(u.iter()) {
                    *ei = ui / n;
                }
                self.vals[started] = ab * u2;
            }
        }
    }
}

/// The 99th percentile of the chi-squared distribution with 2 degrees of
/// freedom, `-2 ln 0.01`: the critical value of the test that two
/// eigenvalues are equal, at the 1% level.
pub const PCA_GAP_CRITICAL: f64 = 9.210_340_371_976_184;

/// Whether component `j` of the variances `vals` is told apart from its
/// neighbours at Kish's `n` and the trace `tr` (the module docs' rule).
fn separated(vals: &[f64], j: usize, n: f64, tr: f64, k: usize) -> bool {
    let lj = vals[j];
    // Above the trace's rounding: a direction of no variance, or of
    // rounding alone, is no direction.
    let floor = tr * f64::EPSILON * k as f64;
    if !(n > 1.0 && lj.is_finite() && floor.is_finite() && lj > floor) {
        return false;
    }
    // Two eigenvalues told apart (the module docs): a neighbour of no
    // variance is told apart once the rows could have given it some, past
    // the dimensions the rows so far span; two of some variance by the test
    // of their equality, `(n - 1) ln(((l + m) / 2)^2 / (l m))` past the
    // critical value. `libm`'s `log`, so the decision is the same on every
    // platform for the same state.
    let apart = |l: f64, m: f64, below: usize| {
        let (hi, lo) = if l >= m { (l, m) } else { (m, l) };
        if lo <= floor {
            return n > (below + 1) as f64;
        }
        let am = 0.5 * (hi + lo);
        (n - 1.0) * libm::log(am / hi * (am / lo)) > PCA_GAP_CRITICAL
    };
    let above = j == 0 || apart(vals[j - 1], lj, j - 1);
    let below = j + 1 >= vals.len() || apart(lj, vals[j + 1], j + 1);
    above && below
}

/// Each direction of `rows` signed for continuity with the same one in
/// `old`; one with no predecessor, or exactly orthogonal to it, makes its
/// largest-magnitude entry positive, the first on a tie.
fn sign_for_continuity(rows: &mut [f64], old: &[f64], k: usize) {
    for (e, o) in rows.chunks_exact_mut(k).zip(old.chunks_exact(k)) {
        let along = dot(e, o);
        let flip = if along != 0.0 {
            along < 0.0
        } else {
            let mut lead = 0;
            for (i, v) in e.iter().enumerate() {
                if v.abs() > e[lead].abs() {
                    lead = i;
                }
            }
            e[lead] < 0.0
        };
        if flip {
            for v in e.iter_mut() {
                *v = -*v;
            }
        }
    }
}

/// The residual below which a deviation starts no direction: the rounding
/// of the deflation that left it, `k` roundings of `|d|`.
fn started_floor(d2: f64, k: usize) -> f64 {
    let r = f64::EPSILON * k as f64;
    d2 * r * r
}

fn row(s: &[f64], i: usize, k: usize) -> &[f64] {
    &s[i * k..(i + 1) * k]
}

/// `a . b`, summed in order.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        s += x * y;
    }
    s
}

/// The `n` rows of `s` (each `k` long) rotated in pairs until every pair is
/// orthogonal to within `k` roundings of their lengths (Hestenes' one-sided
/// Jacobi): the rows are then `S`'s singular values times its right
/// singular vectors. Only `+`, `*`, `/` and `sqrt`, each correctly rounded,
/// in a fixed order, so the bits are the same on every platform.
fn orthogonalize_rows(s: &mut [f64], n: usize, k: usize) {
    let tol = f64::EPSILON * k as f64;
    for _ in 0..MAX_SWEEPS {
        let mut rotated = false;
        for p in 0..n {
            for r in (p + 1)..n {
                let (head, tail) = s.split_at_mut(r * k);
                let rp = &mut head[p * k..(p + 1) * k];
                let rr = &mut tail[..k];
                let alpha = dot(rp, rp);
                let beta = dot(rr, rr);
                let gamma = dot(rp, rr);
                if gamma == 0.0 || gamma.abs() <= tol * (alpha.sqrt() * beta.sqrt()) {
                    continue;
                }
                let zeta = (beta - alpha) / (2.0 * gamma);
                let t = if zeta.abs() > 1e150 {
                    0.5 / zeta
                } else {
                    let sgn = if zeta >= 0.0 { 1.0 } else { -1.0 };
                    sgn / (zeta.abs() + (1.0 + zeta * zeta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let sn = c * t;
                for (x, y) in rp.iter_mut().zip(rr.iter_mut()) {
                    let (xp, yr) = (*x, *y);
                    *x = c * xp - sn * yr;
                    *y = sn * xp + c * yr;
                }
                rotated = true;
            }
        }
        if !rotated {
            break;
        }
    }
}

impl OnlineModel for StreamPca {
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
        self.read_into(x, &mut out.pred);
        out.n_eff = self.moments.n_eff();
        out.extra = None;
        // A weight below the smallest normal double is a weight of 0: the
        // row advances the clock and learns nothing (hard rule 9).
        let weight = if weight < f64::MIN_POSITIVE {
            0.0
        } else {
            weight
        };
        let lam = self.cfg.decay.factor(d_clock);
        let k = self.cfg.n_features;
        let w = self.moments.n_eff();
        let w_new = lam * w + weight;
        let mut d = std::mem::take(&mut self.scratch.d);
        d.clear();
        d.extend((0..k).map(|i| self.moments.deviation(i, x[i])));
        self.scratch.d = d;
        self.moments.update(x, lam, weight);
        self.q_sum = lam * lam * self.q_sum + weight * weight;
        if !(w_new > 0.0) {
            // Nothing carried and nothing added: no share to take, and
            // nothing moves but the weight, which the update decayed.
            return;
        }
        let (a, b) = (lam * w / w_new, weight / w_new);
        let tr_new = self.trace();
        self.learn(a, b, tr_new);
    }

    fn predict(&self, x: &[f64], d_clock: f64) -> Step {
        if let Some(refused) = crate::model::refused_predict(self, x, d_clock) {
            return refused;
        }
        let mut out = Step::default();
        self.read_into(x, &mut out.pred);
        out.n_eff = self.moments.n_eff();
        out
    }

    fn state(&self) -> State {
        State::new(ModelState::StreamPca(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        Self::restore_owned(s.clone())
    }

    fn restore_owned(s: State) -> Result<Self, StateError> {
        crate::check_schema(&s)?;
        match s.model {
            ModelState::StreamPca(m) => {
                let m = *m;
                crate::model::check_cfg("pca", m.cfg.validate())?;
                let (k, q) = (m.cfg.n_features, m.cfg.kept());
                if m.moments.k() != k || m.rows.len() != q * k || m.vals.len() != q {
                    return Err(crate::model::damaged(
                        "pca",
                        "its running sums have the wrong shape",
                    ));
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "pca",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        0
    }

    fn n_features(&self) -> usize {
        self.cfg.n_features
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_outputs()
    }
}

#[cfg(test)]
mod tests;
