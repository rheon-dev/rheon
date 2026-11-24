//! STL and MSTL: the seasonal-trend decomposition by loess of Cleveland,
//! Cleveland, McRae and Terpenning (1990), and its composition over several
//! seasons of Bandara, Hyndman and Bergmeir (2021), as statsmodels computes
//! them (`statsmodels.tsa.stl._stl.STL`, a translation of the authors'
//! Fortran, and `statsmodels.tsa.seasonal.MSTL`), so that a fit is bit-equal
//! to statsmodels' on the same platform.
//!
//! **The loess** at position `x_s` over rows `[l, r]` of a series of `n`
//! rows (positions `j = 1..n`), span `q`:
//!
//! ```text
//! h   = max(x_s - l, r - x_s),  plus floor((q - n) / 2) when q > n   (the Fortran's rule)
//! v_j = 1 if |j - x_s| <= 0.001 h,  0 if |j - x_s| > 0.999 h,
//!       else (1 - (|j - x_s| / h)^3)^3,  times the robustness weight rho_j if used
//! a   = sum v_j   (no value when a <= 0);   v_j /= a
//! degree 1, h > 0: abar = sum v_j j,  c = sum v_j (j - abar)^2,
//!       if sqrt(c) > 0.001 (n - 1):  v_j *= (x_s - abar) / c * (j - abar) + 1
//! value = sum v_j y_j
//! ```
//!
//! Each cube is `powf(3.0)`, the platform's `pow`, as statsmodels' compiled
//! code has it; each square is `x * x`. Sums run in row order.
//!
//! **One inner pass**, the trend `T` (0 at first), `p` the period:
//!
//! ```text
//! D = Y - T
//! C = each cycle-subseries of D smoothed by the loess (span `seasonal`),
//!     extended by one value before its first row and one after its last:
//!     n + 2p rows
//! L = loess (span `low_pass`) of the moving averages of p, p and 3 rows of C
//! S = C[p .. p + n] - L
//! T = loess (span `trend`) of Y - S
//! ```
//!
//! The fit runs `inner` passes, `outer + 1` times; between runs the
//! robustness weights are `rho = B(|R| / (6 median |R|))`, `R = Y - T - S`,
//! `B(u) = (1 - u^2)^2` (1 below 0.001, 0 above 0.999), the median of an even
//! count the mean of its two middle values.
//!
//! **MSTL**: seasons in ascending period; `iterate` times (once with one
//! season), for each season `i`: add `S_i` back, fit STL of period `p_i` and
//! seasonal span `s_i`, take its seasonal as `S_i` and remove it. The trend is
//! the last STL's; the remainder is the deseasonalised series less it.
//!
//! The loops index by row as the Fortran's do, so each line can be read
//! against statsmodels' `_stl.pyx`.
#![allow(clippy::needless_range_loop)]

/// The settings of an STL fit, statsmodels' names and defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct StlCfg {
    pub period: usize,
    /// The seasonal smoother's span: odd, at least 3. statsmodels' default 7.
    pub seasonal: usize,
    /// The trend's span: odd, at least 3, above `period`. `None`: the
    /// smallest odd integer at or above `1.5 p / (1 - 1.5 / seasonal)`.
    pub trend: Option<usize>,
    /// The low-pass filter's span: odd, at least 3, above `period`. `None`:
    /// the smallest odd integer above `period`.
    pub low_pass: Option<usize>,
    pub seasonal_deg: u8,
    pub trend_deg: u8,
    pub low_pass_deg: u8,
    pub seasonal_jump: usize,
    pub trend_jump: usize,
    pub low_pass_jump: usize,
    pub robust: bool,
    /// Inner passes; `None`: 2 when robust, else 5.
    pub inner_iter: Option<usize>,
    /// Outer passes; `None`: 15 when robust, else 0.
    pub outer_iter: Option<usize>,
}

impl StlCfg {
    /// statsmodels' defaults for a period and a seasonal span.
    pub fn new(period: usize, seasonal: usize) -> Self {
        Self {
            period,
            seasonal,
            trend: None,
            low_pass: None,
            seasonal_deg: 1,
            trend_deg: 1,
            low_pass_deg: 1,
            seasonal_jump: 1,
            trend_jump: 1,
            low_pass_jump: 1,
            robust: false,
            inner_iter: None,
            outer_iter: None,
        }
    }
}

/// A fit's components, each as long as the series.
#[derive(Debug, Clone, PartialEq)]
pub struct StlFit {
    pub seasonal: Vec<f64>,
    pub trend: Vec<f64>,
    pub resid: Vec<f64>,
    /// The robustness weights the last inner passes used (1 without `robust`).
    pub weights: Vec<f64>,
}

/// An STL with its settings checked and its defaults resolved.
#[derive(Debug, Clone, PartialEq)]
pub struct Stl {
    period: usize,
    seasonal: usize,
    trend: usize,
    low_pass: usize,
    seasonal_deg: u8,
    trend_deg: u8,
    low_pass_deg: u8,
    seasonal_jump: usize,
    trend_jump: usize,
    low_pass_jump: usize,
    inner: usize,
    outer: usize,
}

fn odd_at_least_3(name: &str, v: usize) -> Result<(), String> {
    if v < 3 || v.is_multiple_of(2) {
        return Err(format!(
            "{name} must be an odd integer of at least 3, but it is {v}. Give an odd span such as 7."
        ));
    }
    Ok(())
}

impl Stl {
    pub fn new(cfg: &StlCfg) -> Result<Self, String> {
        let p = cfg.period;
        if p < 2 {
            return Err(format!(
                "period must be at least 2, but it is {p}. Give the season's length in rows."
            ));
        }
        odd_at_least_3("seasonal", cfg.seasonal)?;
        let trend = cfg.trend.unwrap_or_else(|| {
            let t = (1.5 * p as f64 / (1.0 - 1.5 / cfg.seasonal as f64)).ceil() as usize;
            t + usize::from(t.is_multiple_of(2))
        });
        odd_at_least_3("trend", trend)?;
        if trend <= p {
            return Err(format!(
                "trend must be above the period {p}, but it is {trend}. Give a wider odd span, or leave it to its default."
            ));
        }
        let low_pass = cfg
            .low_pass
            .unwrap_or(p + 1 + usize::from((p + 1).is_multiple_of(2)));
        odd_at_least_3("low_pass", low_pass)?;
        if low_pass <= p {
            return Err(format!(
                "low_pass must be above the period {p}, but it is {low_pass}. Give a wider odd span, or leave it to its default."
            ));
        }
        for (name, d) in [
            ("seasonal_deg", cfg.seasonal_deg),
            ("trend_deg", cfg.trend_deg),
            ("low_pass_deg", cfg.low_pass_deg),
        ] {
            if d > 1 {
                return Err(format!(
                    "{name} must be 0 or 1, but it is {d}. Give 0 for a local mean or 1 for a local line."
                ));
            }
        }
        for (name, j) in [
            ("seasonal_jump", cfg.seasonal_jump),
            ("trend_jump", cfg.trend_jump),
            ("low_pass_jump", cfg.low_pass_jump),
        ] {
            if j == 0 {
                return Err(format!(
                    "{name} must be at least 1, but it is 0. Give 1 to fit every row."
                ));
            }
        }
        Ok(Self {
            period: p,
            seasonal: cfg.seasonal,
            trend,
            low_pass,
            seasonal_deg: cfg.seasonal_deg,
            trend_deg: cfg.trend_deg,
            low_pass_deg: cfg.low_pass_deg,
            seasonal_jump: cfg.seasonal_jump,
            trend_jump: cfg.trend_jump,
            low_pass_jump: cfg.low_pass_jump,
            inner: cfg.inner_iter.unwrap_or(if cfg.robust { 2 } else { 5 }),
            outer: cfg.outer_iter.unwrap_or(if cfg.robust { 15 } else { 0 }),
        })
    }

    pub fn period(&self) -> usize {
        self.period
    }

    pub fn seasonal_span(&self) -> usize {
        self.seasonal
    }

    pub fn trend_span(&self) -> usize {
        self.trend
    }

    pub fn low_pass_span(&self) -> usize {
        self.low_pass
    }

    /// Whether the fit is linear in the series: no outer pass, so no
    /// robustness weights.
    pub fn is_linear(&self) -> bool {
        self.outer == 0
    }

    /// The fit of `y`, as statsmodels' `STL(y, ...).fit()`.
    pub fn fit(&self, y: &[f64]) -> StlFit {
        let n = y.len();
        let np = self.period;
        let m = n + 2 * np;
        let mut work: [Vec<f64>; 5] = std::array::from_fn(|_| vec![0.0; m]);
        let mut season = vec![0.0; n];
        let mut trend = vec![0.0; n];
        let mut rw = vec![1.0; n];
        let mut use_rw = false;
        let mut k = 0;
        loop {
            self.onestp(y, &mut work, &mut season, &mut trend, &rw, use_rw);
            k += 1;
            if k > self.outer {
                break;
            }
            for i in 0..n {
                work[0][i] = trend[i] + season[i];
            }
            rwts(y, &work[0][..n], &mut rw);
            use_rw = true;
        }
        let resid = (0..n).map(|i| y[i] - season[i] - trend[i]).collect();
        StlFit {
            seasonal: season,
            trend,
            resid,
            weights: rw,
        }
    }

    fn onestp(
        &self,
        y: &[f64],
        work: &mut [Vec<f64>; 5],
        season: &mut [f64],
        trend: &mut [f64],
        rw: &[f64],
        use_rw: bool,
    ) {
        let n = y.len();
        let np = self.period;
        for _ in 0..self.inner {
            for i in 0..n {
                work[0][i] = y[i] - trend[i];
            }
            self.ss(work, season, rw, use_rw, n);
            // fts: moving averages of p, p and 3 rows of C (work[1]).
            let [w0, w1, w2, _, w4] = work;
            ma(w1, n + 2 * np, np, w2);
            ma(w2, n + np + 1, np, w0);
            ma(w0, n + 2, 3, w2);
            ess(
                &w2[..n],
                n,
                self.low_pass,
                self.low_pass_deg,
                self.low_pass_jump,
                None,
                w0,
                w4,
            );
            for i in 0..n {
                season[i] = w1[np + i] - w0[i];
                w0[i] = y[i] - season[i];
            }
            ess(
                &w0[..n],
                n,
                self.trend,
                self.trend_deg,
                self.trend_jump,
                use_rw.then_some(rw),
                trend,
                w2,
            );
        }
    }

    /// The cycle-subseries smoothing: each phase's subseries of `work[0]`,
    /// smoothed and extended at both ends, into `work[1]`. `season` is
    /// scratch for the loess weights, as in statsmodels.
    fn ss(&self, work: &mut [Vec<f64>; 5], season: &mut [f64], rw: &[f64], use_rw: bool, n: usize) {
        let np = self.period;
        let ns = self.seasonal;
        let [w0, w1, w2, w3, w4] = work;
        for j in 0..np {
            let k = (n - (j + 1)) / np + 1;
            for i in 0..k {
                w2[i] = w0[i * np + j];
            }
            if use_rw {
                for i in 0..k {
                    w4[i] = rw[i * np + j];
                }
            }
            let sub_rw = use_rw.then_some(&w4[..k]);
            let (head, rest) = w3.split_at_mut(1);
            ess(
                &w2[..k],
                k,
                ns,
                self.seasonal_deg,
                self.seasonal_jump,
                sub_rw,
                rest,
                season,
            );
            let first = est(
                &w2[..k],
                k,
                ns,
                self.seasonal_deg,
                0,
                1,
                ns.min(k),
                season,
                sub_rw,
            );
            head[0] = if first.is_nan() { rest[0] } else { first };
            let nleft = if k + 1 > ns { k + 1 - ns } else { 1 };
            let last = est(
                &w2[..k],
                k,
                ns,
                self.seasonal_deg,
                k + 1,
                nleft,
                k,
                season,
                sub_rw,
            );
            w3[k + 1] = if last.is_nan() { w3[k] } else { last };
            for mm in 0..k + 2 {
                w1[mm * np + j] = w3[mm];
            }
        }
    }
}

/// The loess at position `xs` (1-based) over rows `nleft..=nright` (1-based),
/// or NaN when every weight vanishes. `w` is scratch, indexed by row.
#[allow(clippy::too_many_arguments)]
fn est(
    y: &[f64],
    n: usize,
    len: usize,
    ideg: u8,
    xs: usize,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
    rw: Option<&[f64]>,
) -> f64 {
    let rng = n as f64 - 1.0;
    // The Fortran's signed max: at an end point (xs = 0 or n + 1) one side is -1.
    let mut h = (xs as i64 - nleft as i64).max(nright as i64 - xs as i64) as f64;
    if len > n {
        h += ((len - n) / 2) as f64;
    }
    let h9 = 0.999 * h;
    let h1 = 0.001 * h;
    let mut a = 0.0;
    for j in nleft - 1..nright {
        w[j] = 0.0;
        let r = (j + 1).abs_diff(xs) as f64;
        if r <= h9 {
            w[j] = if r <= h1 {
                1.0
            } else {
                (1.0 - (r / h).powf(3.0)).powf(3.0)
            };
            if let Some(rw) = rw {
                w[j] *= rw[j];
            }
            a += w[j];
        }
    }
    if a <= 0.0 {
        return f64::NAN;
    }
    for wj in &mut w[nleft - 1..nright] {
        *wj /= a;
    }
    if h > 0.0 && ideg > 0 {
        let mut a = 0.0;
        for j in nleft - 1..nright {
            a += w[j] * (j + 1) as f64;
        }
        let mut b = xs as f64 - a;
        let mut c = 0.0;
        for j in nleft - 1..nright {
            let d = (j + 1) as f64 - a;
            c += w[j] * (d * d);
        }
        if c.sqrt() > 0.001 * rng {
            b /= c;
            for j in nleft - 1..nright {
                w[j] *= b * ((j + 1) as f64 - a) + 1.0;
            }
        }
    }
    let mut ys = 0.0;
    for j in nleft - 1..nright {
        ys += w[j] * y[j];
    }
    ys
}

/// The loess of `y` at every row, or every `njump`-th row with linear
/// interpolation between, into `ys`; a row whose weights all vanish takes
/// its own value. `res` is scratch.
#[allow(clippy::too_many_arguments)]
fn ess(
    y: &[f64],
    n: usize,
    len: usize,
    ideg: u8,
    njump: usize,
    rw: Option<&[f64]>,
    ys: &mut [f64],
    res: &mut [f64],
) {
    if n < 2 {
        ys[0] = y[0];
        return;
    }
    let newnj = njump.min(n - 1);
    let mut nleft = 1;
    let mut nright = n;
    let fill = |v: f64, i: usize| if v.is_nan() { y[i] } else { v };
    if len >= n {
        let mut i = 0;
        while i < n {
            ys[i] = fill(est(y, n, len, ideg, i + 1, nleft, nright, res, rw), i);
            i += newnj;
        }
    } else if newnj == 1 {
        let nsh = (len + 2) / 2;
        nright = len;
        for i in 0..n {
            if i + 1 > nsh && nright != n {
                nleft += 1;
                nright += 1;
            }
            ys[i] = fill(est(y, n, len, ideg, i + 1, nleft, nright, res, rw), i);
        }
    } else {
        let nsh = len.div_ceil(2);
        let mut i = 0;
        while i < n {
            if i + 1 < nsh {
                nleft = 1;
                nright = len;
            } else if i >= n - nsh {
                nleft = n - len + 1;
                nright = n;
            } else {
                nleft = i + 1 - nsh + 1;
                nright = len + i + 1 - nsh;
            }
            ys[i] = fill(est(y, n, len, ideg, i + 1, nleft, nright, res, rw), i);
            i += newnj;
        }
    }
    if newnj == 1 {
        return;
    }
    let mut i = 0;
    while i < n - newnj {
        let delta = (ys[i + newnj] - ys[i]) / newnj as f64;
        for j in i..i + newnj {
            ys[j] = ys[i] + delta * ((j + 1) - (i + 1)) as f64;
        }
        i += newnj;
    }
    let k = ((n - 1) / newnj) * newnj + 1;
    if k != n {
        ys[n - 1] = fill(est(y, n, len, ideg, n, nleft, nright, res, rw), n - 1);
        if k != n - 1 {
            let delta = (ys[n - 1] - ys[k - 1]) / (n - k) as f64;
            for j in k..n {
                ys[j] = ys[k - 1] + delta * ((j + 1) - k) as f64;
            }
        }
    }
}

/// The running mean of `len` rows of the first `n` of `x`, into `ave`:
/// `n - len + 1` values, a running sum updated by `x[k] - x[m]`.
fn ma(x: &[f64], n: usize, len: usize, ave: &mut [f64]) {
    let newn = n - len + 1;
    let flen = len as f64;
    let mut v = 0.0;
    for &xi in &x[..len] {
        v += xi;
    }
    ave[0] = v / flen;
    let (mut k, mut m) = (len, 0);
    for slot in &mut ave[1..newn] {
        v += x[k] - x[m];
        *slot = v / flen;
        k += 1;
        m += 1;
    }
}

/// The robustness weights from the residuals `|y - fit|`.
fn rwts(y: &[f64], fit: &[f64], rw: &mut [f64]) {
    let n = y.len();
    for i in 0..n {
        rw[i] = (y[i] - fit[i]).abs();
    }
    let mut sorted = rw.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid0 = n / 2;
    let mid1 = n - mid0 - 1;
    let cmad = 3.0 * (sorted[mid0] + sorted[mid1]);
    if cmad == 0.0 {
        rw.fill(1.0);
        return;
    }
    let c9 = 0.999 * cmad;
    let c1 = 0.001 * cmad;
    for r in rw.iter_mut() {
        *r = if *r <= c1 {
            1.0
        } else if *r <= c9 {
            let u = *r / cmad;
            let v = 1.0 - u * u;
            v * v
        } else {
            0.0
        };
    }
}

/// MSTL's components: one seasonal per period, in the order of `periods`
/// sorted ascending, the trend and the remainder.
#[derive(Debug, Clone, PartialEq)]
pub struct MstlFit {
    pub periods: Vec<usize>,
    pub seasonal: Vec<Vec<f64>>,
    pub trend: Vec<f64>,
    pub resid: Vec<f64>,
    pub weights: Vec<f64>,
}

/// MSTL with checked settings: one STL per season, shortest period first.
#[derive(Debug, Clone, PartialEq)]
pub struct Mstl {
    stls: Vec<Stl>,
    iterate: usize,
}

impl Mstl {
    /// `seasons` as `(period, seasonal span)`; `base` gives every other
    /// setting each STL takes (its `period` and `seasonal` are ignored).
    pub fn new(seasons: &[(usize, usize)], base: &StlCfg, iterate: usize) -> Result<Self, String> {
        if seasons.is_empty() {
            return Err(
                "MSTL needs at least one season, but it has none. Give one or more periods."
                    .to_string(),
            );
        }
        if iterate == 0 {
            return Err(
                "iterate must be at least 1, but it is 0. Give 1 or more passes.".to_string(),
            );
        }
        let mut sorted = seasons.to_vec();
        sorted.sort_by_key(|&(p, _)| p);
        let stls = sorted
            .iter()
            .map(|&(period, seasonal)| {
                Stl::new(&StlCfg {
                    period,
                    seasonal,
                    ..base.clone()
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { stls, iterate })
    }

    pub fn stls(&self) -> &[Stl] {
        &self.stls
    }

    pub fn fit(&self, y: &[f64]) -> MstlFit {
        let n = y.len();
        let mut seasonal = vec![vec![0.0; n]; self.stls.len()];
        let mut deseas = y.to_vec();
        let passes = if self.stls.len() == 1 {
            1
        } else {
            self.iterate
        };
        let mut last = None;
        for _ in 0..passes {
            for (i, stl) in self.stls.iter().enumerate() {
                for (d, s) in deseas.iter_mut().zip(&seasonal[i]) {
                    *d += s;
                }
                let fit = stl.fit(&deseas);
                seasonal[i].copy_from_slice(&fit.seasonal);
                for (d, s) in deseas.iter_mut().zip(&seasonal[i]) {
                    *d -= s;
                }
                last = Some(fit);
            }
        }
        let last = last.expect("at least one season and one pass");
        let resid = deseas.iter().zip(&last.trend).map(|(d, t)| d - t).collect();
        MstlFit {
            periods: self.stls.iter().map(Stl::period).collect(),
            seasonal,
            trend: last.trend,
            resid,
            weights: last.weights,
        }
    }
}

#[cfg(test)]
#[path = "stl_tests.rs"]
mod tests;
