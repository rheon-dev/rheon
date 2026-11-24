//! One target's series in a [`super::Sarima`]: its ring of past values for
//! the differencing, and the Kalman filter of its differenced value.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use super::poly::{Ssm, difference, pack, unpack};

/// One target: the ring the differencing reads, and the filter's predicted
/// state and its covariance for the next slot.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Series {
    /// The last `depth = d + sD` slots' values, oldest first: the value
    /// observed, or for a slot not observed, its prediction (NaN where
    /// there was none).
    pub(crate) ys: VecDeque<f64>,
    /// Whether each slot of `ys` was observed.
    pub(crate) seen: VecDeque<bool>,
    /// `a_{t|t-1}`: the state's mean for the next slot, `r` long.
    pub(crate) a: Vec<f64>,
    /// `P_{t|t-1}`: its covariance, symmetric after every step, so kept as
    /// its upper triangle row after row, `r (r + 1) / 2` long. Its row 0 is
    /// its first `r` entries.
    pub(crate) p: Vec<f64>,
    /// The weight of the rows that observed this target, before the next
    /// row: what its `min_weight` is checked against.
    pub(crate) weight: f64,
}

/// What one slot gives a target: the differenced value's prediction `ŵ_t`
/// and the value's `ŷ_t`, each NaN where it cannot be formed.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Slot {
    pub(crate) w_hat: f64,
    pub(crate) y_hat: f64,
}

/// The working space of a slot, shared by every target of a model: not
/// state, so a clone starts empty and two are always equal.
#[derive(Debug, Default)]
pub(crate) struct Work {
    win: Vec<f64>,
    stages: Vec<f64>,
    /// The covariance written out in full, `r x r`, while it steps.
    full: Vec<f64>,
    row0: Vec<f64>,
    gain: Vec<f64>,
}

impl Clone for Work {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl PartialEq for Work {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

/// The differencing orders and the ring's depth.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Diff {
    pub(crate) d: usize,
    pub(crate) big_d: usize,
    pub(crate) s: usize,
}

impl Diff {
    pub(crate) fn depth(&self) -> usize {
        self.d + self.big_d * self.s
    }

    /// The window `ring ++ [last]` differenced (see [`difference`]).
    pub(crate) fn of(&self, ring: &VecDeque<f64>, last: f64, work: &mut Work) -> f64 {
        work.win.clear();
        work.win.extend(ring.iter().copied());
        work.win.push(last);
        difference(&work.win, self.d, self.big_d, self.s, &mut work.stages)
    }

    /// [`Self::of`] with each slot not `seen` as NaN: the differenced value
    /// statsmodels forms from a series with its missing values as `NaN`,
    /// NaN exactly where the differencing reads one of them. A slot the
    /// differencing does not read (with `d = 1, D = 1`, every lag but `0`,
    /// `1`, `s` and `s + 1`) leaves it observed.
    pub(crate) fn of_seen(
        &self,
        ring: &VecDeque<f64>,
        seen: &VecDeque<bool>,
        last: f64,
        work: &mut Work,
    ) -> f64 {
        work.win.clear();
        work.win.extend(
            ring.iter()
                .zip(seen)
                .map(|(&v, &s)| if s { v } else { f64::NAN }),
        );
        work.win.push(last);
        difference(&work.win, self.d, self.big_d, self.s, &mut work.stages)
    }
}

impl Series {
    /// A series before its first slot: the state at its stationary
    /// distribution, mean 0 and covariance `p0` (`r x r`, in full).
    pub(crate) fn new(depth: usize, r: usize, p0: &[f64]) -> Self {
        let mut p = Vec::with_capacity(r * (r + 1) / 2);
        pack(p0, r, &mut p);
        Self {
            ys: VecDeque::with_capacity(depth),
            seen: VecDeque::with_capacity(depth),
            a: vec![0.0; r],
            p,
            weight: 0.0,
        }
    }

    /// Back to the start of a series, the weight kept: the ring emptied and
    /// the state's mean at 0, its covariance at `p0` (in full) where one is
    /// given.
    pub(crate) fn restart(&mut self, p0: Option<&[f64]>) {
        self.ys.clear();
        self.seen.clear();
        self.a.fill(0.0);
        if let Some(p0) = p0 {
            pack(p0, self.a.len(), &mut self.p);
        }
    }

    /// One slot: the prediction from the state before it, then the update
    /// with `y` where the slot's differenced value is observed, then the
    /// state carried to the next slot. `y` is `None` for a slot not
    /// observed (a null target, a row of weight 0, a slot the clock
    /// skipped); `d_t` is the exogenous part of the differenced value,
    /// `x_t' beta` (0 without regressors, NaN for a skipped slot's). A slot
    /// not observed keeps its prediction in the ring where `fill`, NaN
    /// otherwise. With `cov` false only the mean is carried: what a
    /// prediction reads.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn slot(
        &mut self,
        ssm: &Ssm,
        diff: Diff,
        y: Option<f64>,
        d_t: f64,
        fill: bool,
        cov: bool,
        work: &mut Work,
    ) -> Slot {
        let depth = diff.depth();
        if self.ys.len() < depth {
            // No differenced value yet: the series' first `d + sD` slots,
            // which statsmodels' simple differencing drops.
            self.push(y.unwrap_or(f64::NAN), y.is_some(), depth);
            return Slot {
                w_hat: f64::NAN,
                y_hat: f64::NAN,
            };
        }
        let r = self.a.len();
        let w_hat = d_t + self.a[0];
        let y_hat = w_hat - diff.of(&self.ys, 0.0, work);
        // `F = P_00`, and `P_.0 = P_0.`, the packed matrix's first `r`.
        let f = self.p[0];
        let innovation = match y {
            Some(v) if f > 0.0 => {
                let w = diff.of_seen(&self.ys, &self.seen, v, work);
                (!w.is_nan()).then_some(w - w_hat)
            }
            _ => None,
        };
        if cov {
            unpack(&self.p, r, &mut work.full);
        }
        if let Some(v) = innovation {
            let finv = 1.0 / f;
            let g = v * finv;
            for i in 0..r {
                self.a[i] += self.p[i] * g;
            }
            if cov {
                // `P <- P - P_.0 (P_0. / F)`.
                work.gain.clear();
                work.gain.extend(self.p[..r].iter().map(|v| finv * v));
                for i in 0..r {
                    let ci = self.p[i];
                    for j in 0..r {
                        work.full[i * r + j] -= ci * work.gain[j];
                    }
                }
            }
        }
        ssm.predict_mean(&mut self.a);
        if cov {
            ssm.predict_cov(&mut work.full, &mut work.row0);
            pack(&work.full, r, &mut self.p);
        }
        let kept = if fill { y_hat } else { f64::NAN };
        self.push(y.unwrap_or(kept), y.is_some(), depth);
        Slot { w_hat, y_hat }
    }

    pub(crate) fn push(&mut self, v: f64, seen: bool, depth: usize) {
        if depth == 0 {
            return;
        }
        if self.ys.len() == depth {
            self.ys.pop_front();
            self.seen.pop_front();
        }
        self.ys.push_back(v);
        self.seen.push_back(seen);
    }
}
