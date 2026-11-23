//! `Tails`: the exponentially weighted skewness, kurtosis and Jarque and
//! Bera's (1980) statistic of a slot's recursive residuals, with a memory of
//! their own (docs/PLAN.md task 221 (e)).
//!
//! The recursive residual `v = resid / error_inflation` (see
//! [`crate::Breaks`]) is the residual with the fit's estimation error taken
//! out; skewness and kurtosis are ratios of its moments, so its scale does
//! not enter them. With `ω` each scored row's weight times the decay since
//! it, `W = Σω`, the weighted mean `μ` and the central sums
//! `M_j = Σ ω (v − μ)^j` give the central moments `m_j = M_j / W` and
//!
//! ```text
//! skew = m_3 / m_2^{3/2}       kurtosis = m_4 / m_2² − 3   (excess, as Polars')
//! JB   = n/6 · (skew² + kurtosis²/4)                       ~ χ²(2)
//! ```
//!
//! The sums are kept central by Pébay's weighted one-pass updates (Pébay,
//! Terriberry, Kolla and Bennett 2016): a decay scales `W` and each `M_j`
//! and leaves `μ`, and a row `v` at weight `w` merges the set so far with a
//! set of one. With `δ = v − μ`, `W' = W + w`, `q = W/W'` and `r = w/W'`,
//!
//! ```text
//! M_4' = M_4 + δ⁴ W r (q² − q r + r²) + 6 δ² r² M_2 − 4 δ r M_3
//! M_3' = M_3 + δ³ W r (q − r) − 3 δ r M_2
//! M_2' = M_2 + δ² W r                    μ' = μ + δ r
//! ```
//!
//! Raw power sums about a fixed origin, the first residual, cancelled
//! catastrophically once that origin sat far from where the residuals
//! settled: after a first residual of `+1e5`, a memory of 50 rows read a
//! kurtosis of −129,065 where two passes read 0.99 (review round 6, B-5).
//!
//! `n` Kish's size. Run once with unit weights these are the biased
//! moments of `scipy.stats.skew` and `kurtosis` and `statsmodels`'
//! `jarque_bera`.

use serde::{Deserialize, Serialize};

/// The tails of one slot's recursive residuals. See the module docs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Tails {
    /// `W = Σω`, and `Σω²` for Kish's size.
    w: f64,
    w2: f64,
    /// The weighted mean `μ`.
    mean: f64,
    /// The central sums `M_2`, `M_3`, `M_4`.
    m: [f64; 3],
}

impl Tails {
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether a restored accumulator can be read and folded.
    pub fn has_shape(&self) -> bool {
        [self.w, self.w2, self.mean]
            .iter()
            .chain(&self.m)
            .all(|v| v.is_finite())
            && self.w >= 0.0
            && self.w2 >= 0.0
            && self.m[0] >= 0.0
    }

    /// The clock moves on by a step whose decay is `lam`, with nothing
    /// scored.
    pub fn age(&mut self, lam: f64) {
        self.w *= lam;
        self.w2 *= lam * lam;
        self.m.iter_mut().for_each(|v| *v *= lam);
    }

    /// One recursive residual `v` at weight `w`, after the step's decay
    /// `lam`. A `v` that is not finite, or a weight of 0, only ages
    /// (CLAUDE.md hard rule 9).
    pub fn update(&mut self, v: f64, lam: f64, w: f64) {
        self.age(lam);
        if !(v.is_finite() && w > 0.0) {
            return;
        }
        // The merge of the set so far, weight `a`, with the row alone. No
        // term divides by `w`, so a tiny weight cannot overflow one; at
        // `a = 0` (the first row, or a memory decayed to nothing) every
        // term but the mean's is 0 and the mean is `v`.
        let a = self.w;
        let total = a + w;
        let (q, r) = (a / total, w / total);
        let d = v - self.mean;
        let (d2, ar) = (d * d, a * r);
        let [m2, m3, m4] = self.m;
        let next = [
            m2 + d2 * ar,
            m3 + d2 * d * ar * (q - r) - 3.0 * d * r * m2,
            m4 + d2 * d2 * ar * (q * q - q * r + r * r) + 6.0 * d2 * r * r * m2 - 4.0 * d * r * m3,
        ];
        // A deviation whose fourth power passes the largest double (`|δ|`
        // past about 1.2e77, inside the input bound) is a row the sums
        // cannot hold: it only ages, and the state stays one a load takes.
        if !next.iter().all(|v| v.is_finite()) {
            return;
        }
        self.m = next;
        self.mean += d * r;
        self.w = total;
        self.w2 += w * w;
    }

    /// The central moments `(m_2, m_3, m_4)`; `None` before a spread.
    fn moments(&self) -> Option<(f64, f64, f64)> {
        let w = self.w;
        if w <= 0.0 {
            return None;
        }
        let [m2, m3, m4] = self.m.map(|v| v / w);
        if m2.is_nan() || m2 <= 0.0 {
            return None;
        }
        Some((m2, m3, m4))
    }

    /// The skewness, `m_3 / m_2^{3/2}`, with `m_2^{3/2}` as `m_2·√m_2`: two
    /// correctly rounded operations, so a target scaled by `2^k` scales
    /// `m_2` by `2^{2k}`, `m_3` and the denominator by `2^{3k}`, and leaves
    /// the skewness to the bit. `m_2.powf(1.5)` is the platform's `pow`,
    /// which glibc and `ucrtbase.dll` round to within an ulp but not
    /// correctly: `pow(c²·m, 1.5) ≠ c³·pow(m, 1.5)` for 236 of 400,000
    /// arguments under glibc 2.39 at `c = 2^±20`, and one row a run of
    /// `kalman`, `lasso` or `quantile_median` then left its units by an ulp.
    ///
    /// `None` where `m_2^{3/2}` is below the normal doubles: residuals near
    /// `1e-100`, whose `m_3` underflows with it and read NaN.
    pub fn skew(&self) -> Option<f64> {
        let (m2, m3, _) = self.moments()?;
        let den = m2 * m2.sqrt();
        den.is_normal().then(|| m3 / den)
    }

    /// The excess kurtosis, `m_4 / m_2² − 3`. `None` where `m_2²` or `m_4`
    /// is below the normal doubles: residuals near `1e-90`, where `m_4`
    /// underflows to 0 and the ratio read NaN or `−3`.
    pub fn kurtosis(&self) -> Option<f64> {
        let (m2, _, m4) = self.moments()?;
        let den = m2 * m2;
        (den.is_normal() && m4.is_normal()).then(|| m4 / den - 3.0)
    }

    /// Jarque and Bera's statistic at Kish's size: `None` before a spread,
    /// and where `Σω²` is below the normal doubles, as a gap of about 538
    /// half-lives leaves it while `W` is still above 0: no Kish size is read
    /// there, where it read infinite.
    pub fn jarque_bera(&self) -> Option<f64> {
        let (s, k) = (self.skew()?, self.kurtosis()?);
        if !self.w2.is_normal() {
            return None;
        }
        let n = self.w * self.w / self.w2;
        let jb = n / 6.0 * (s * s + k * k / 4.0);
        jb.is_finite().then_some(jb)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }

    /// Skewness, kurtosis and Jarque and Bera by their definitions, two
    /// passes over every row at its present weight -- the mean, then the
    /// deviations' powers -- under a decay, uneven weights and zeros, on a
    /// skewed stream at a level.
    #[test]
    fn the_tails_are_their_definitions() {
        let mut t = Tails::new();
        let mut st = 17u64;
        let mut rows: Vec<(f64, f64)> = Vec::new();
        let lam = 0.995;
        for i in 0..500 {
            let u = lcg(&mut st) + 0.5;
            let v = 3.0 + u * u * u * 4.0;
            let w = if i % 9 == 1 {
                0.0
            } else {
                0.5 + lcg(&mut st) + 0.5
            };
            t.update(v, lam, w);
            rows.iter_mut().for_each(|r| r.1 *= lam);
            rows.push((v, w));
            if i < 10 || i % 41 != 0 {
                continue;
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let sw2: f64 = rows.iter().map(|r| r.1 * r.1).sum();
            let mu = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
            let m = |j: i32| rows.iter().map(|r| r.1 * (r.0 - mu).powi(j)).sum::<f64>() / sw;
            let skew = m(3) / m(2).powf(1.5);
            let kurt = m(4) / m(2).powi(2) - 3.0;
            let jb = sw * sw / sw2 / 6.0 * (skew * skew + kurt * kurt / 4.0);
            assert!((t.skew().unwrap() - skew).abs() < 1e-9, "row {i}");
            assert!((t.kurtosis().unwrap() - kurt).abs() < 1e-8, "row {i}");
            assert!(
                (t.jarque_bera().unwrap() - jb).abs() < 1e-7 * jb.max(1.0),
                "row {i}"
            );
        }
        assert!(t.skew().unwrap() > 0.5, "a skewed stream");
        assert!(t.has_shape());
    }

    /// A target scaled by a power of two leaves the skewness and the
    /// kurtosis to the bit: the moments scale exactly, and `m_2^{3/2}` is
    /// `m_2·√m_2`. As `m_2.powf(1.5)`, the platform's `pow`, about one row
    /// in 1,700 came out an ulp off under glibc 2.39
    /// (`test_convergence_equivariance.py`).
    #[test]
    fn the_shape_keeps_its_bits_under_a_power_of_two_scale() {
        let mut st = 5u64;
        for c in [2f64.powi(-20), 2f64.powi(20), 0.125, 8.0] {
            let (mut a, mut b) = (Tails::new(), Tails::new());
            for i in 0..20_000 {
                let u = lcg(&mut st) + 0.5;
                let v = 3.0 + if i % 3 == 0 { -1.0 } else { 4.0 } * u * u * u;
                let w = 0.5 + lcg(&mut st) + 0.5;
                a.update(v, 0.999, w);
                b.update(v * c, 0.999, w);
                if let (Some(sa), Some(sb)) = (a.skew(), b.skew()) {
                    assert_eq!(sa.to_bits(), sb.to_bits(), "skew, row {i}, scale {c}");
                    let (ka, kb) = (a.kurtosis().unwrap(), b.kurtosis().unwrap());
                    assert_eq!(ka.to_bits(), kb.to_bits(), "kurtosis, row {i}, scale {c}");
                }
            }
        }
    }

    /// A first residual far from the rest -- a bad first print -- leaves the
    /// moments as a two-pass reading gives them, whether the memory keeps it
    /// or has all but forgotten it (review round 6, B-5). Raw power sums
    /// about that first residual read a kurtosis of −129,065 at `+1e5`
    /// where the two passes read 0.99.
    #[test]
    fn a_far_first_residual_does_not_cancel_the_moments() {
        for (first, lam) in [
            (1e3, 0.986),
            (1e4, 0.986),
            (1e5, 0.986),
            (1e5, 1.0),
            (1e7, 1.0),
        ] {
            let mut t = Tails::new();
            let mut st = 23u64;
            let mut rows: Vec<(f64, f64)> = Vec::new();
            for i in 0..3_000 {
                let v = if i == 0 {
                    first
                } else {
                    lcg(&mut st) + lcg(&mut st) + lcg(&mut st)
                };
                let w = 0.5 + lcg(&mut st) + 0.5;
                t.update(v, lam, w);
                rows.iter_mut().for_each(|r| r.1 *= lam);
                rows.push((v, w));
            }
            let sw: f64 = rows.iter().map(|r| r.1).sum();
            let mu = rows.iter().map(|r| r.1 * r.0).sum::<f64>() / sw;
            let m = |j: i32| rows.iter().map(|r| r.1 * (r.0 - mu).powi(j)).sum::<f64>() / sw;
            let skew = m(3) / m(2).powf(1.5);
            let kurt = m(4) / m(2).powi(2) - 3.0;
            let (gs, gk) = (t.skew().unwrap(), t.kurtosis().unwrap());
            assert!(
                (gs - skew).abs() < 1e-6 * skew.abs().max(1.0),
                "first {first}, lam {lam}: skew {gs} vs {skew}"
            );
            assert!(
                (gk - kurt).abs() < 1e-6 * kurt.abs().max(1.0),
                "first {first}, lam {lam}: kurtosis {gk} vs {kurt}"
            );
        }
    }

    #[test]
    fn zero_weights_and_missing_values_only_age() {
        let mut t = Tails::new();
        t.update(1.0, 0.9, 0.0);
        t.update(f64::NAN, 0.9, 1.0);
        assert_eq!(t, Tails::new());
        t.update(1.0, 0.9, 1.0);
        assert!(t.skew().is_none(), "one value has no spread");
        // A weight too small to square, beside rows of weight 1, leaves
        // every sum finite: no term divides by it.
        t.update(1e5, 0.9, 1e-300);
        t.update(-1.0, 0.9, 1.0);
        t.update(0.5, 0.9, 1.0);
        assert!(t.has_shape() && t.kurtosis().is_some_and(f64::is_finite));
    }

    /// A restored accumulator is refused unless every sum and the mean are
    /// finite and the weights and the second central sum are not negative.
    #[test]
    fn a_restored_accumulator_must_be_finite_and_not_negative() {
        let mut good = Tails::new();
        for v in [1.0, -0.5, 2.0, 0.3] {
            good.update(v, 0.9, 1.0);
        }
        assert!(good.has_shape());
        let bad = |edit: fn(&mut Tails)| {
            let mut t = good.clone();
            edit(&mut t);
            !t.has_shape()
        };
        assert!(bad(|t| t.mean = f64::NAN));
        assert!(bad(|t| t.m[2] = f64::INFINITY));
        assert!(bad(|t| t.w = -1.0));
        assert!(bad(|t| t.w2 = -1.0));
        assert!(bad(|t| t.m[0] = -1.0));
    }

    /// A memory decayed past where `Σω²` can be held (a gap of about 538
    /// half-lives underflows it to 0 while `W` is still above 0) has no
    /// Kish size: Jarque and Bera's statistic is `None` there, never
    /// infinite or NaN, and the next row reads again (review 2026-11-23,
    /// DIAG-4 (1)).
    #[test]
    fn a_memory_decayed_past_its_kish_size_reads_no_statistic() {
        let mut t = Tails::new();
        let mut st = 5u64;
        for _ in 0..200 {
            t.update(lcg(&mut st), 0.98, 1.0);
        }
        assert!(t.jarque_bera().unwrap().is_finite());
        for half_lives in [530, 538, 540, 600, 1100] {
            let mut g = t.clone();
            g.age(0.5f64.powi(half_lives));
            let jb = g.jarque_bera();
            assert!(jb.is_none_or(f64::is_finite), "{half_lives}: {jb:?}");
            if g.w2 == 0.0 {
                assert_eq!(jb, None, "{half_lives}");
            }
            g.update(0.3, 1.0, 1.0);
            g.update(-0.2, 1.0, 1.0);
            g.update(0.7, 1.0, 1.0);
            assert!(g.jarque_bera().unwrap().is_finite(), "{half_lives}");
        }
    }

    /// A residual inside the input bound whose fourth power passes the
    /// largest double leaves the state finite, so a saved state still
    /// loads (review 2026-11-23, DIAG-4 (2): `|d| > 1.16e77`).
    #[test]
    fn a_residual_whose_fourth_power_overflows_leaves_the_state_loadable() {
        for big in [2e77, 1e100, -1e100, 1.3e154, 1e200] {
            let mut t = Tails::new();
            let mut st = 9u64;
            for _ in 0..50 {
                t.update(lcg(&mut st), 0.99, 1.0);
            }
            t.update(big, 0.99, 1.0);
            assert!(t.has_shape(), "{big}: {t:?}");
            for _ in 0..50 {
                t.update(lcg(&mut st), 0.99, 1.0);
            }
            assert!(t.has_shape(), "{big}");
            assert!(t.kurtosis().unwrap().is_finite(), "{big}");
        }
    }

    /// Residuals near `1e-90` read infinite or NaN kurtosis: `m_2²` and
    /// `m_4` underflow there (review 2026-11-23, DIAG-4 (3)). A ratio whose
    /// denominator the doubles cannot hold reads `None`; where they hold it,
    /// a scale by a power of two leaves the shape to the bit; and no scale
    /// reads a number that is not finite.
    #[test]
    fn tiny_residuals_read_their_shape_or_none() {
        let mut st = 13u64;
        let rows: Vec<f64> = (0..300)
            .map(|_| {
                let u = lcg(&mut st) + 0.5;
                u * u * u
            })
            .collect();
        let fed = |scale: f64| {
            let mut t = Tails::new();
            for &v in &rows {
                t.update(v * scale, 0.99, 1.0);
            }
            t
        };
        let one = fed(1.0);
        for k in [-150, -240, 150, 240] {
            let t = fed(2f64.powi(k));
            assert_eq!(t.skew(), one.skew(), "2^{k}");
            assert_eq!(t.kurtosis(), one.kurtosis(), "2^{k}");
            assert_eq!(t.jarque_bera(), one.jarque_bera(), "2^{k}");
        }
        for k in [-280, -300, -330, -400, -500, 280, 300, 400] {
            let t = fed(2f64.powi(k));
            assert!(t.has_shape(), "2^{k}");
            for (what, got, want) in [
                ("skew", t.skew(), one.skew()),
                ("kurtosis", t.kurtosis(), one.kurtosis()),
                ("jarque_bera", t.jarque_bera(), one.jarque_bera()),
            ] {
                assert!(got.is_none() || got == want, "2^{k}: {what} {got:?} against {want:?}");
            }
        }
        // At 2^-300 the fourth moment is below the doubles: no kurtosis.
        assert_eq!(fed(2f64.powi(-300)).kurtosis(), None);
    }
}
