//! STL and MSTL online, behind an embargo: a ring of the last `W` learned
//! rows of a series, and on each row the decomposition of the ring read at
//! the row `lag` learned rows back.
//!
//! **The ring.** A row is learned when its value is present; a missing row
//! pushes nothing and counts in no lag. Once the ring holds `W` rows it is
//! the window `x_{t-W+1}, ..., x_t` of the last `W` learned rows, oldest
//! first, and every emission decomposes exactly that window, as statsmodels
//! would given those `W` values.
//!
//! **What a row emits.** With `closed = Left` the ring as it stood before
//! the row, read at position `W - lag` (0-based, `1 <= lag <= W`), so the
//! emitted value reads no row at or after the current one; with `closed =
//! Right` the ring after the row is pushed, read at position `W - 1 - lag`
//! (`0 <= lag <= W - 1`), which reads the row's own value. Nothing is
//! emitted until the ring holds `W` rows. A missing row reads the ring as it
//! stands, so under `Right` it repeats the row before's output.
//!
//! **The embargo.** The seasonal smoother of a season of period `p` and span
//! `s` reaches `p (s - 1) / 2` rows either side, so the default lag is
//! `h = max_i p_i (s_i - 1) / 2` (plus 1 under `Left`) and the default window
//! `W = 2h + 1`, which puts the emitted row at the window's centre, where a
//! window's decomposition is closest to the whole history's.
//!
//! **State.** `W` values, the head, the count and the last learned row's
//! clock: `8 W + 24` bytes a group, bounded by the window, never the stream.

use crate::stl::{Mstl, Stl};

/// Which ring an emission reads: before or after the row is pushed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Closed {
    /// Before: no row at or after the current one is read.
    Left,
    /// After: the row's own value is read.
    Right,
}

/// The decomposition an operator runs on its window.
#[derive(Debug, Clone, PartialEq)]
pub enum Decomposition {
    Stl(Stl),
    Mstl(Mstl),
}

impl Decomposition {
    /// Each season as `(period, seasonal span)`, shortest period first.
    pub fn seasons(&self) -> Vec<(usize, usize)> {
        match self {
            Self::Stl(s) => vec![(s.period(), s.seasonal_span())],
            Self::Mstl(m) => m
                .stls()
                .iter()
                .map(|s| (s.period(), s.seasonal_span()))
                .collect(),
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Stl(_) => "stl",
            Self::Mstl(_) => "mstl",
        }
    }
}

/// The components at one position of a window: one seasonal per season
/// (shortest period first), the trend and the remainder.
#[derive(Debug, Clone, PartialEq)]
pub struct Components {
    pub seasonal: Vec<f64>,
    pub trend: f64,
    pub resid: f64,
}

/// The seasonal smoother's reach, `max_i p_i (s_i - 1) / 2`: the default lag
/// under `Right`, and half the default window less one.
pub fn seasonal_reach(seasons: &[(usize, usize)]) -> usize {
    seasons
        .iter()
        .map(|&(p, s)| p * (s - 1) / 2)
        .max()
        .unwrap_or(0)
}

/// The default lag: the seasonal reach, plus 1 under `Left`.
pub fn default_lag(seasons: &[(usize, usize)], closed: Closed) -> usize {
    seasonal_reach(seasons) + usize::from(closed == Closed::Left)
}

/// The default window, `2 h + 1` with `h` the seasonal reach.
pub fn default_window(seasons: &[(usize, usize)]) -> usize {
    2 * seasonal_reach(seasons) + 1
}

/// A group's state in bytes for a window of `w` rows: `8 w + 24`.
pub fn state_bytes(w: usize) -> u64 {
    8 * w as u64 + 24
}

/// Refuses a window whose ring would hold more than `cap` bytes a group,
/// checked when a run is built, since `w` is known then.
pub fn check_state_bytes(operator: &str, w: usize, cap: u64) -> Result<(), String> {
    let bytes = state_bytes(w);
    if bytes > cap {
        return Err(format!(
            "{operator}: a window of {w} rows holds {bytes} bytes a group (8 W + 24), past \
             max_state_bytes of {cap}. Shorten the window, or raise max_state_bytes."
        ));
    }
    Ok(())
}

/// The last `W` learned rows of one group's series.
#[derive(Debug, Clone, PartialEq)]
pub struct StlRing {
    values: Vec<f64>,
    head: usize,
    count: usize,
    last_clock: f64,
}

impl StlRing {
    pub fn new(w: usize) -> Self {
        Self {
            values: vec![0.0; w],
            head: 0,
            count: 0,
            last_clock: f64::NAN,
        }
    }

    /// Learns a row: its value at the head, and its clock as the last.
    pub fn push(&mut self, x: f64, clock: f64) {
        let w = self.values.len();
        self.values[self.head] = x;
        self.head = (self.head + 1) % w;
        self.count = (self.count + 1).min(w);
        self.last_clock = clock;
    }

    /// Forgets every row: a session change, a reset, a break in the step.
    pub fn clear(&mut self) {
        self.head = 0;
        self.count = 0;
        self.last_clock = f64::NAN;
    }

    pub fn is_full(&self) -> bool {
        self.count == self.values.len()
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// The last learned row's clock; NaN when the ring is empty.
    pub fn last_clock(&self) -> f64 {
        self.last_clock
    }

    /// The rows held, oldest first.
    pub fn window(&self) -> Vec<f64> {
        let w = self.values.len();
        let start = (self.head + w - self.count) % w;
        (0..self.count)
            .map(|i| self.values[(start + i) % w])
            .collect()
    }
}

/// An operator's settings, checked: what it decomposes, its window, its lag
/// and which ring it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct StlOperator {
    decomposition: Decomposition,
    window: usize,
    lag: usize,
    closed: Closed,
}

impl StlOperator {
    /// `window` and `lag` default to the embargo above.
    pub fn new(
        decomposition: Decomposition,
        window: Option<usize>,
        lag: Option<usize>,
        closed: Closed,
    ) -> Result<Self, String> {
        let name = decomposition.name();
        let seasons = decomposition.seasons();
        let w = window.unwrap_or_else(|| default_window(&seasons));
        let lag = lag.unwrap_or_else(|| default_lag(&seasons, closed));
        match &decomposition {
            Decomposition::Stl(s) if w < 2 * s.period() => {
                return Err(format!(
                    "{name}: the window of {w} rows holds fewer than two periods of {}. Give a \
                     window of at least {} rows.",
                    s.period(),
                    2 * s.period()
                ));
            }
            Decomposition::Mstl(m) => {
                if let Some(p) = m.stls().iter().map(Stl::period).find(|&p| 2 * p >= w) {
                    return Err(format!(
                        "{name}: the period {p} is not below half the window of {w} rows, which \
                         statsmodels would drop. Give a window above {} rows.",
                        2 * p
                    ));
                }
            }
            _ => {}
        }
        let (lo, hi) = match closed {
            Closed::Left => (1, w),
            Closed::Right => (0, w - 1),
        };
        if lag < lo || lag > hi {
            return Err(format!(
                "{name}: lag {lag} is outside the window: with closed={closed:?} and a window of \
                 {w} rows it must be from {lo} to {hi}. Give a lag in that range."
            ));
        }
        Ok(Self {
            decomposition,
            window: w,
            lag,
            closed,
        })
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn lag(&self) -> usize {
        self.lag
    }

    pub fn closed(&self) -> Closed {
        self.closed
    }

    /// A ring for one group.
    pub fn ring(&self) -> StlRing {
        StlRing::new(self.window)
    }

    /// The position in the window that an emission reads.
    pub fn position(&self) -> usize {
        match self.closed {
            Closed::Left => self.window - self.lag,
            Closed::Right => self.window - 1 - self.lag,
        }
    }

    /// One row of a group: learns `x` when present, and returns the
    /// components at the emitted position once the ring is full.
    pub fn step(&self, ring: &mut StlRing, x: Option<f64>, clock: f64) -> Option<Components> {
        let out = match self.closed {
            Closed::Left => self.emit(ring),
            Closed::Right => None,
        };
        if let Some(x) = x {
            ring.push(x, clock);
        }
        match self.closed {
            Closed::Left => out,
            Closed::Right => self.emit(ring),
        }
    }

    /// The exact decomposition of a full ring at the emitted position.
    fn emit(&self, ring: &StlRing) -> Option<Components> {
        if !ring.is_full() {
            return None;
        }
        let y = ring.window();
        let at = self.position();
        Some(match &self.decomposition {
            Decomposition::Stl(s) => {
                let f = s.fit(&y);
                Components {
                    seasonal: vec![f.seasonal[at]],
                    trend: f.trend[at],
                    resid: f.resid[at],
                }
            }
            Decomposition::Mstl(m) => {
                let f = m.fit(&y);
                Components {
                    seasonal: f.seasonal.iter().map(|s| s[at]).collect(),
                    trend: f.trend[at],
                    resid: f.resid[at],
                }
            }
        })
    }
}

#[cfg(test)]
#[path = "stl_ring_tests.rs"]
mod tests;
