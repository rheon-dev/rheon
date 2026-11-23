//! `DdSketchModel`: one DDSketch ([`crate::DdSketch`]) per target column,
//! read at a list of quantile levels before each row is added, each with the
//! mean of its tail. The bank's `ddsketch` kind.
//!
//! For target `t` with sketch `D_t`, level `p` and the row's value `y`:
//!
//! ```text
//! q_p,t  = D_t.quantile(p)        read before the row (out of sample)
//! es_p,t = D_t.tail_mean(p)       the lower tail below 0.5, the upper above
//! D_t   <- D_t + (y, w)           a row of weight w > 0 with y present
//! W_t   <- W_t + w,   S_t <- S_t + w^2
//! ```
//!
//! The quantile at `p` is the representative of the first bucket, in value
//! order, whose cumulative weight reaches `p W_t`: the exact quantile's
//! bucket (numpy's `inverted_cdf` over the rows and their weights), so
//! within `alpha` of it, relatively. The tail's mean is the
//! partial-mass rule over the buckets ([`crate::DdSketch::reads`]): each
//! bucket wholly in the tail at its representative, the bucket the tail
//! ends in at only the weight the tail still needs, so the mean is within
//! `alpha` times the mean magnitude of the tail of the exact tail mean.
//!
//! **A window.** `window_rows = N` keeps the last `N` rows of positive
//! weight; `window_clock = D` the rows of the last `D` clock units, decided
//! on the rows' exact stamps ([`crate::Stamp::cmp_span`]) under `closed`:
//! `Right`, the default, drops a row exactly `D` old, `Both` keeps it. A
//! ring keeps each row's weight and, per target, its bucket (32 bits: the
//! key and a tag, [`crate::ddsketch::Slot::code`]); a row leaving takes its
//! weight out of its bucket, and a bucket whose rows have all left is
//! exactly 0, so the sketch is the one built from the rows left. A clock
//! window's rows leave at the first row at or after the one they age out
//! on, before that row's read. Under a window `n_eff`, `W_t` and `S_t` are
//! the window's, the rows the row reads (hard rule 8), and a target's sums are exactly 0 once its
//! rows have all left. No decay: a decayed sketch's weights are rounded by
//! every decay step, so it is refused for the windowed form, which takes a
//! row out exactly.
//!
//! **When a level is withheld.** As `tdigest`'s
//! ([`crate::tdigest_model::level_identified`]): with Kish's effective size
//! `n_t = W_t^2 / S_t`, a level is reported where `p (n_t + 1) >= 1` and
//! `(1 - p) (n_t + 1) >= 1`, else it and its tail mean are NaN and the
//! stream names the reason.
//!
//! **A row of weight 0** learns nothing and counts in no clock (hard rule
//! 9): it is read as any row is, from the window as it would stand at the
//! row, and changes nothing.
//!
//! **State.** Per target its live buckets in two stores, a 32-bit key and
//! an `f64` weight each (and a `u32` row count under a window), 12 or 16
//! bytes, and three numbers; a row window `N (8 + 4 m)` bytes for `m`
//! targets, checked at the call; a clock window a stamp more a row. The
//! buckets are the data's distinct ones, so the bytes are
//! checked as the rows go in: the first row to take the state past half of
//! `max_state_bytes` is said, and a row that would take it past the cap is
//! not learned, nor any after it.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::Stamp;
use crate::WindowClosed;
use crate::clock_span::{DEFAULT_STATE_BYTES, HeldCap, check_max_state_bytes};
use crate::ddsketch::{DdSketch, Mapping, Slot};
use crate::model::{ModelState, OnlineModel, State, StateError, Step, check_schema};
use crate::tdigest_model::level_identified;

/// The settings of a [`DdSketchModel`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DdSketchModelCfg {
    /// One sketch per target column.
    pub n_targets: usize,
    /// The relative accuracy, in [`crate::ddsketch::DDSKETCH_MIN_ALPHA`], 1).
    pub alpha: f64,
    /// The levels read, each in (0, 1), no two alike; per target, one
    /// quantile a level, then one tail mean a level, in this order.
    pub levels: Vec<f64>,
    /// Keep the last this many rows of positive weight alone.
    pub window_rows: Option<usize>,
    /// Keep the rows of the last this many clock units alone.
    pub window_clock: Option<f64>,
    /// Which edge of `window_clock` holds a row exactly that old.
    pub closed: WindowClosed,
    /// The bytes a group's state may hold; `None` is 256 KB.
    pub max_state_bytes: Option<u64>,
}

impl DdSketchModelCfg {
    pub fn cap(&self) -> u64 {
        self.max_state_bytes.unwrap_or(DEFAULT_STATE_BYTES)
    }

    fn windowed(&self) -> bool {
        self.window_rows.is_some() || self.window_clock.is_some()
    }

    /// The bytes a ring row takes: its weight, a 32-bit slot a target, and
    /// a stamp under a clock window.
    fn ring_row_bytes(&self) -> u64 {
        let stamp = if self.window_clock.is_some() {
            std::mem::size_of::<Stamp>() as u64
        } else {
            0
        };
        8 + 4 * self.n_targets as u64 + stamp
    }

    /// The bytes the settings fix: the header, three numbers a target, the
    /// levels, and a row window's ring at its fullest.
    pub fn fixed_bytes(&self) -> u64 {
        std::mem::size_of::<DdSketchModel>() as u64
            + self.n_targets as u64 * (std::mem::size_of::<DdSketch>() as u64 + 32)
            + 8 * self.levels.len() as u64
            + self
                .window_rows
                .map_or(0, |n| n as u64 * self.ring_row_bytes())
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.n_targets == 0 {
            return Err(
                "ddsketch: it needs at least one target, a column whose quantiles it \
                 reads. Name the column in targets."
                    .into(),
            );
        }
        Mapping::new(self.alpha).map_err(|e| e.replace("DdSketch:", "ddsketch:"))?;
        if self.levels.is_empty() {
            return Err(
                "ddsketch: quantiles must list at least one level, but it is empty. \
                 Give the levels to read, such as [0.05, 0.5, 0.95]."
                    .into(),
            );
        }
        if let Some(p) = self.levels.iter().find(|p| !(**p > 0.0 && **p < 1.0)) {
            return Err(format!(
                "ddsketch: each of quantiles must be above 0 and below 1, but one is {p}. A \
                 level of 0 or 1 is the smallest or largest row, which no sample identifies \
                 as a quantile. Give levels strictly between 0 and 1."
            ));
        }
        if self.levels.contains(&0.5) {
            return Err(
                "ddsketch: quantiles lists 0.5, whose tail is neither the lower nor the upper, \
                 so its expected shortfall has no side. Give a level below or above 0.5, such \
                 as 0.49 or 0.51; the median itself is the level a tail's mean is not taken at."
                    .into(),
            );
        }
        for (i, p) in self.levels.iter().enumerate() {
            if self.levels[..i].iter().any(|q| q.to_bits() == p.to_bits()) {
                return Err(format!(
                    "ddsketch: quantiles lists {p} twice, and each level names an output. \
                     List each level once."
                ));
            }
        }
        if self.window_rows.is_some() && self.window_clock.is_some() {
            return Err(
                "ddsketch: window_rows and window_size are each a window, and only one can \
                 decide which rows are read. Give window_rows or window_size, not both."
                    .into(),
            );
        }
        if self.closed == WindowClosed::Both && self.window_clock.is_none() {
            return Err(
                "ddsketch: closed = \"both\" names the edge of window_size, which is not \
                 given. Give window_size, or leave closed out."
                    .into(),
            );
        }
        if self.window_rows == Some(0) {
            return Err(
                "ddsketch: window_rows must be at least 1, but it is 0. Give the number of \
                 rows the window keeps."
                    .into(),
            );
        }
        if let Some(d) = self.window_clock
            && !(d > 0.0 && d.is_finite())
        {
            return Err(format!(
                "ddsketch: window_size must be above 0 and finite, but it is {d}. Give the \
                 span of the clock the window keeps."
            ));
        }
        check_max_state_bytes("ddsketch", self.max_state_bytes)?;
        let (most, cap) = (self.fixed_bytes(), self.cap());
        if most > cap {
            return Err(format!(
                "ddsketch: a group's {} sketches and their window may hold {most} bytes before \
                 any bucket, past max_state_bytes {cap}. Shorten window_rows, read fewer \
                 targets in this spec, or raise max_state_bytes.",
                self.n_targets
            ));
        }
        Ok(())
    }
}

/// A window's rows, oldest first: each row's weight, its slot code a target
/// ([`Slot::ABSENT`] where the target was absent), and under a clock window
/// its stamp.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Ring {
    w: VecDeque<f64>,
    codes: VecDeque<i32>,
    stamp: VecDeque<Stamp>,
}

/// Where the clock stands, for a clock window's keys.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct Clock {
    /// The clock summed from every step: the key for a caller that hands
    /// no stamps.
    own: f64,
    /// The rows that counted: the key on the row clock.
    counted: u64,
    /// The stamp of the last row stepped, whatever its weight.
    read: Option<Stamp>,
    /// The last stamp handed was the row clock's.
    on_rows: bool,
}

/// A DDSketch a target, read at the configured levels (the module's
/// equations).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DdSketchModel {
    cfg: DdSketchModelCfg,
    map: Mapping,
    sketches: Vec<DdSketch>,
    /// Every learned row's weight (the window's under a window): `n_eff`.
    w_sum: f64,
    /// Per target, the weight `W_t`, the squared weights `S_t` and (under a
    /// window) the rows of the rows its sketch holds.
    target_w: Vec<f64>,
    target_w2: Vec<f64>,
    target_n: Vec<u64>,
    ring: Option<Ring>,
    clock: Clock,
    /// The bytes a row would have taken the state to, past the cap: from
    /// then on no row is learned.
    over: Option<u64>,
    /// The half-of-the-cap warning has been raised (once a group).
    warned: bool,
    /// The next row's stamp, and whether it is the row clock's; and a
    /// warning raised and not yet taken. Never state.
    #[serde(skip)]
    next: Option<(Stamp, bool)>,
    #[serde(skip)]
    warning: Option<u64>,
}

impl DdSketchModel {
    /// Refuse a decay: `given` names the field a spec set (`half_life`,
    /// `lam`), `None` when it set none. A decayed sketch multiplies every
    /// bucket by the decay at each step, so its weights are rounded on every
    /// row and a row can never be taken out of them exactly; the windowed
    /// form takes each row out of its bucket exactly. Run once at spec
    /// build, before any row.
    pub fn check_no_decay(given: Option<&str>) -> Result<(), String> {
        match given {
            None => Ok(()),
            Some(field) => Err(format!(
                "DdSketch: {field} is not allowed. A decayed DDSketch multiplies every bucket \
                 by the decay at each row, so its weights are rounded on every row and no row \
                 can be taken out of them exactly. For a quantile that forgets, use the \
                 windowed DDSketch: give window_rows (the last rows of positive weight) or \
                 window_size (the rows of the last span of the clock), whose ring takes each \
                 row out of its bucket exactly as it leaves."
            )),
        }
    }

    pub fn new(cfg: DdSketchModelCfg) -> Result<Self, String> {
        cfg.validate()?;
        let map = Mapping::new(cfg.alpha)?;
        let n = cfg.n_targets;
        Ok(Self {
            map,
            sketches: vec![DdSketch::default(); n],
            w_sum: 0.0,
            target_w: vec![0.0; n],
            target_w2: vec![0.0; n],
            target_n: vec![0; n],
            ring: cfg.windowed().then(Ring::default),
            clock: Clock::default(),
            over: None,
            warned: false,
            next: None,
            warning: None,
            cfg,
        })
    }

    pub fn cfg(&self) -> &DdSketchModelCfg {
        &self.cfg
    }

    pub fn mapping(&self) -> &Mapping {
        &self.map
    }

    /// Every learned row's weight (the window's under a window): `n_eff`.
    pub fn n_eff(&self) -> f64 {
        self.w_sum
    }

    /// Target `j`'s sketch.
    pub fn sketch(&self, j: usize) -> &DdSketch {
        &self.sketches[j]
    }

    /// The rows a window holds now; 0 without one.
    pub fn window_len(&self) -> usize {
        self.ring.as_ref().map_or(0, |r| r.w.len())
    }

    /// Kish's effective size of target `j`'s rows, `W^2 / S`; 0 before any.
    pub fn effective_rows(&self, j: usize) -> f64 {
        let (w, s) = (self.target_w[j], self.target_w2[j]);
        if s > 0.0 { w / s * w } else { 0.0 }
    }

    /// Whether the next row reads some level of some target as not
    /// identified by its rows, so withheld.
    pub fn beyond_sample(&self) -> bool {
        (0..self.cfg.n_targets).any(|j| {
            let n = self.effective_rows(j);
            self.cfg.levels.iter().any(|&p| !level_identified(p, n))
        })
    }

    /// The bytes the state holds now.
    pub fn bytes(&self) -> u64 {
        let counted = self.cfg.windowed();
        let ring = match (&self.ring, self.cfg.window_clock) {
            (Some(r), Some(_)) => r.w.len() as u64 * self.cfg.ring_row_bytes(),
            _ => 0,
        };
        self.cfg.fixed_bytes()
            + self
                .sketches
                .iter()
                .map(|s| s.heap_bytes(counted))
                .sum::<u64>()
            + ring
    }

    /// The bytes the state would hold with a row of `slots` added.
    fn bytes_with(&self, slots: &[Option<Slot>]) -> u64 {
        let per = crate::ddsketch::Store::bucket_bytes(self.cfg.windowed());
        let grow: u64 = self
            .sketches
            .iter()
            .zip(slots)
            .filter_map(|(s, slot)| slot.map(|k| (s.len_with(k) - s.buckets_len()) as u64))
            .sum();
        let ring = if self.cfg.window_clock.is_some() {
            self.cfg.ring_row_bytes()
        } else {
            0
        };
        self.bytes() + per * grow + ring
    }

    fn held_cap(&mut self) -> HeldCap<'_> {
        HeldCap {
            over: &mut self.over,
            warned: &mut self.warned,
            warning: &mut self.warning,
        }
    }

    /// The levels and their tail means, target by target (the quantiles at
    /// `2 nl j + l`, the tail means at `2 nl j + nl + l`), NaN where a level
    /// is withheld.
    fn read_into(&self, out: &mut Step) {
        let nl = self.cfg.levels.len();
        out.pred.clear();
        out.pred.resize(self.cfg.n_targets * 2 * nl, f64::NAN);
        out.n_eff = self.w_sum;
        out.extra = None;
        for (j, s) in self.sketches.iter().enumerate() {
            let n = self.effective_rows(j);
            if !self.cfg.levels.iter().any(|&p| level_identified(p, n)) {
                continue;
            }
            let reads = s.reads(&self.cfg.levels);
            for (li, (&p, r)) in self.cfg.levels.iter().zip(reads).enumerate() {
                if let Some((q, es)) = r.filter(|_| level_identified(p, n)) {
                    out.pred[2 * nl * j + li] = q;
                    out.pred[2 * nl * j + nl + li] = es;
                }
            }
        }
    }

    /// The window's key for a row: on the row clock the count of the rows
    /// that counted (this one too, where `counting`), else the stamp
    /// handed, else the clock summed from the steps.
    fn window_key(&self, now: Option<(Stamp, bool)>, d_clock: f64, counting: bool) -> Stamp {
        match now {
            Some((_, true)) => Stamp::from((self.clock.counted + u64::from(counting)) as f64),
            Some((s, false)) => s,
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// A read ahead's key, for a row `d_clock` after the last stepped.
    fn key_ahead(&self, d_clock: f64) -> Stamp {
        match self.clock.read {
            _ if self.clock.on_rows => Stamp::from(self.clock.counted as f64 + 1.0),
            Some(last) => crate::since::back_by(last, -d_clock),
            None => Stamp::from(self.clock.own + d_clock),
        }
    }

    /// Where a row left the clock: kept for a clock window's keys alone, so
    /// without one a row of weight 0 changes nothing at all.
    fn note_clock(&mut self, d_clock: f64, handed: Option<(Stamp, bool)>, counting: bool) {
        if self.cfg.window_clock.is_none() {
            return;
        }
        self.clock.own += d_clock;
        self.clock.counted += u64::from(counting);
        if let Some((s, rows)) = handed {
            self.clock.read = Some(s);
            self.clock.on_rows = rows;
        }
    }

    /// Whether the oldest row of a clock window has left at `now`.
    fn oldest_has_left(&self, now: Stamp) -> bool {
        let Some(span) = self.cfg.window_clock else {
            return false;
        };
        self.ring
            .as_ref()
            .and_then(|r| r.stamp.front())
            .is_some_and(|&s| self.cfg.closed.has_left(now, s, span))
    }

    /// Take the oldest row out of the window and the sketches.
    fn pop_oldest(&mut self) {
        let m = self.cfg.n_targets;
        let ring = self.ring.as_mut().expect("windowed");
        let Some(w) = ring.w.pop_front() else {
            return;
        };
        ring.stamp.pop_front();
        for j in 0..m {
            let code = ring.codes.pop_front().expect("a code a target");
            let Some(slot) = Slot::decode(code) else {
                continue;
            };
            self.sketches[j].remove(slot, w);
            self.target_n[j] -= 1;
            if self.target_n[j] == 0 {
                self.target_w[j] = 0.0;
                self.target_w2[j] = 0.0;
            } else {
                self.target_w[j] = (self.target_w[j] - w).max(0.0);
                self.target_w2[j] = (self.target_w2[j] - w * w).max(0.0);
            }
        }
        self.w_sum = if ring.w.is_empty() {
            0.0
        } else {
            (self.w_sum - w).max(0.0)
        };
    }

    /// Drop a clock window's rows that have left at `now`.
    fn trim(&mut self, now: Stamp) {
        while self.oldest_has_left(now) {
            self.pop_oldest();
        }
    }

    /// The read for a row at `key` that learns nothing: from the window as
    /// it would stand at the row.
    fn read_ahead(&self, key: Stamp, out: &mut Step) {
        if self.oldest_has_left(key) {
            let mut at = self.clone();
            at.trim(key);
            at.read_into(out);
            return;
        }
        self.read_into(out);
    }

    /// Whether a restored state has the shape [`Self::new`] gives it.
    fn has_shape(&self) -> bool {
        let n = self.cfg.n_targets;
        let counted = self.cfg.windowed();
        let ring = match &self.ring {
            None => !counted,
            Some(r) => {
                counted
                    && r.codes.len() == n * r.w.len()
                    && r.stamp.len()
                        == if self.cfg.window_clock.is_some() {
                            r.w.len()
                        } else {
                            0
                        }
                    && self.cfg.window_rows.is_none_or(|cap| r.w.len() <= cap)
                    && r.w.iter().all(|w| *w > 0.0 && w.is_finite())
                    && r.codes.iter().all(|&c| {
                        Slot::decode(c).is_none() || Slot::decode(c).map(Slot::code) == Some(c)
                    })
            }
        };
        self.map.alpha().to_bits() == self.cfg.alpha.to_bits()
            && Mapping::new(self.cfg.alpha).is_ok_and(|m| m == self.map)
            && self.sketches.len() == n
            && self.target_w.len() == n
            && self.target_w2.len() == n
            && self.target_n.len() == n
            && ring
            && self.sketches.iter().all(|s| s.has_shape(counted))
            && self
                .target_w
                .iter()
                .chain(&self.target_w2)
                .chain(std::iter::once(&self.w_sum))
                .all(|w| w.is_finite() && *w >= 0.0)
    }
}

impl OnlineModel for DdSketchModel {
    fn step_into(
        &mut self,
        _x: &[f64],
        y: &[Option<f64>],
        d_clock: f64,
        weight: f64,
        out: &mut Step,
    ) {
        // A value or a weight that is not usable, by the rule every model
        // keeps: such a value is absent, and such a weight learns nothing.
        // The model reads no features.
        if let Some(refused) = crate::model::refused_step(self, &[], y, d_clock, weight) {
            *out = refused;
            return;
        }
        let handed = self.next.take();
        if weight <= 0.0 {
            // Counts in no clock (hard rule 9): read from the window as it
            // would stand at the row, and nothing kept but where it stood.
            let key = self.window_key(handed, d_clock, false);
            self.read_ahead(key, out);
            self.note_clock(d_clock, handed, false);
            return;
        }
        let key = self.window_key(handed, d_clock, true);
        self.note_clock(d_clock, handed, true);
        // The rows out of a clock window by now leave before the row is
        // read, and `n_eff` is the weight of the rows it reads (hard rule 8).
        self.trim(key);
        self.read_into(out);
        if self.over.is_some() {
            return;
        }
        let slots: Vec<Option<Slot>> = (0..self.cfg.n_targets)
            .map(|j| y.get(j).copied().flatten().map(|v| Slot::of(&self.map, v)))
            .collect();
        let (bytes, cap) = (self.bytes_with(&slots), self.cfg.cap());
        if !self.held_cap().admit(bytes, cap) {
            return;
        }
        let counted = self.cfg.windowed();
        for (j, slot) in slots.iter().enumerate() {
            if let Some(slot) = *slot {
                self.sketches[j].add(&self.map, slot, weight, counted);
                self.target_w[j] += weight;
                self.target_w2[j] += weight * weight;
                if counted {
                    self.target_n[j] += 1;
                }
            }
        }
        self.w_sum += weight;
        if let Some(ring) = self.ring.as_mut() {
            ring.w.push_back(weight);
            ring.codes
                .extend(slots.iter().map(|s| s.map_or(Slot::ABSENT, Slot::code)));
            if self.cfg.window_clock.is_some() {
                ring.stamp.push_back(key);
            }
            if self.cfg.window_rows.is_some_and(|cap| ring.w.len() > cap) {
                self.pop_oldest();
            }
        }
        let bytes = self.bytes();
        self.held_cap().note(bytes, cap);
    }

    fn predict(&self, _x: &[f64], d_clock: f64) -> Step {
        let mut out = Step::default();
        let key = self.key_ahead(d_clock);
        self.read_ahead(key, &mut out);
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
        out.extend_from_slice(&self.target_w);
        true
    }

    fn state(&self) -> State {
        State::new(ModelState::DdSketch(Box::new(self.clone())))
    }

    fn restore(s: &State) -> Result<Self, StateError> {
        check_schema(s)?;
        match &s.model {
            ModelState::DdSketch(m) => {
                let mut m = (**m).clone();
                crate::model::check_cfg("ddsketch", m.cfg.validate())?;
                if !m.has_shape() {
                    return Err(crate::model::damaged(
                        "ddsketch",
                        "its sketches have the wrong shape",
                    ));
                }
                for s in &mut m.sketches {
                    s.refill(&m.map);
                }
                Ok(m)
            }
            other => Err(StateError::WrongModel {
                expected: "ddsketch",
                found: other.kind(),
            }),
        }
    }

    fn n_targets(&self) -> usize {
        self.cfg.n_targets
    }

    /// The sketches are the targets'; the model reads no features.
    fn n_features(&self) -> usize {
        0
    }

    fn n_outputs(&self) -> usize {
        self.cfg.n_targets * 2 * self.cfg.levels.len()
    }
}

#[cfg(test)]
#[path = "ddsketch/model_tests.rs"]
mod tests;
