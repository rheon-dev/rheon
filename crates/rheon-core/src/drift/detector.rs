//! The drift detectors a stream can keep on each slot's errors, behind one
//! type, and what each reads of a row.
//!
//! | kind | reads | flags |
//! |---|---|---|
//! | Page-Hinkley | `\|r\| / σ`, over the clock | its mean's excess past `threshold` |
//! | CUSUM | `r / σ`, over the clock | `C⁺` or `C⁻` past `h` |
//! | ADWIN | `\|r\|` | a cut of its window |
//! | KSWIN | `\|r\|` | its two samples' KS test |
//! | DDM, EDDM | `\|r\| > error_above`, an error or not | the error rate's rise |
//!
//! `r` is the row's out-of-sample residual and `σ` the slot's error scale
//! before the row. The two read on the clock need the scale, and a row
//! without one ages Page-Hinkley's mean and is not read; ADWIN and KSWIN read
//! the error's size as it is. KSWIN's test compares ranks, so its flags are
//! the same in any unit; ADWIN's bound takes the values' range `R`, river's
//! 1 by default, so its flags are not, unless `R` is the window's own range
//! ([`crate::AdwinRange::Window`]); DDM and EDDM count errors, and what counts as
//! one is the caller's `error_above`, in the target's units. ADWIN, KSWIN, DDM
//! and EDDM count rows, not clock: their statistics are defined on a number of
//! observations (ADWIN's bound on its parts' rows, KSWIN's exact null on two
//! samples of `r`, DDM's binomial deviation), which a span of clock does not
//! give.
//!
//! **The file's form.** A Page-Hinkley detector is written as it always was,
//! its fields alone; every other kind under its name, `{"Adwin": {...}}`, so
//! a file written before the other kinds existed loads as it did, and its
//! bytes are written again as they were.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::{Adwin, Cusum, Ddm, Eddm, Kswin, PageHinkley};

/// A detector that counts errors, and what counts as one: a residual whose
/// size is past `error_above`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorRate<T> {
    pub error_above: f64,
    pub detector: T,
}

/// One slot's drift detector, of any kind.
#[derive(Debug, Clone, PartialEq)]
pub enum DriftDetector {
    PageHinkley(PageHinkley),
    Adwin(Adwin),
    Ddm(ErrorRate<Ddm>),
    Eddm(ErrorRate<Eddm>),
    Kswin(Kswin),
    Cusum(Cusum),
}

/// `error_above` as a spec takes it: at least 0 and not NaN. A spec that
/// leaves it out of a kind that does not count errors stores infinity.
fn error_above_sound(v: f64) -> bool {
    v >= 0.0
}

impl DriftDetector {
    /// Feed one row's residual `r`, the slot's scale before it, the row's
    /// clock step `d` and the decay `lam` over it. Returns whether the row
    /// flags a drift. The caller passes only a row that counts, of positive
    /// weight with a finite residual; [`Self::age`] the others.
    #[inline]
    pub fn update(&mut self, r: f64, scale: f64, d: f64, lam: f64) -> bool {
        let scaled = scale.is_finite() && scale > 0.0;
        // A finite residual whose ratio to a positive scale overflows is the
        // largest double of scales out, not a row unseen: so read, it flags.
        let ratio = |r: f64| (r / scale).clamp(-f64::MAX, f64::MAX);
        match self {
            Self::PageHinkley(p) => {
                if scaled {
                    p.update(ratio(r.abs()), d, lam)
                } else {
                    p.age(lam);
                    false
                }
            }
            Self::Cusum(c) => scaled && c.update(ratio(r), d),
            Self::Adwin(a) => a.update(r.abs()),
            Self::Kswin(k) => k.update(r.abs()),
            Self::Ddm(e) => e.detector.update(r.abs() > e.error_above),
            Self::Eddm(e) => e.detector.update(r.abs() > e.error_above),
        }
    }

    /// A step of the clock with nothing to read: Page-Hinkley's mean ages by
    /// `lam`; no other kind keeps a clock.
    pub fn age(&mut self, lam: f64) {
        if let Self::PageHinkley(p) = self {
            p.age(lam);
        }
    }

    /// Whether a restored detector is of `fresh`'s kind and keeps the shape
    /// its rows rely on: a file that says otherwise is damaged, and is
    /// refused rather than run.
    pub fn restores_as(&self, fresh: &Self) -> bool {
        std::mem::discriminant(self) == std::mem::discriminant(fresh)
            && match self {
                Self::PageHinkley(p) => p.sound(),
                Self::Adwin(a) => a.sound(),
                Self::Kswin(k) => k.sound(),
                Self::Cusum(c) => c.sound(),
                Self::Ddm(e) => error_above_sound(e.error_above) && e.detector.sound(),
                Self::Eddm(e) => error_above_sound(e.error_above) && e.detector.sound(),
            }
    }

    /// Start again from nothing seen.
    pub fn reset(&mut self) {
        match self {
            Self::PageHinkley(p) => p.reset(),
            Self::Cusum(c) => c.reset(),
            Self::Adwin(a) => a.reset(),
            Self::Kswin(k) => k.reset(),
            Self::Ddm(e) => e.detector.reset(),
            Self::Eddm(e) => e.detector.reset(),
        }
    }
}

#[derive(Serialize)]
enum TaggedRef<'a> {
    Adwin(&'a Adwin),
    Ddm(&'a ErrorRate<Ddm>),
    Eddm(&'a ErrorRate<Eddm>),
    Kswin(&'a Kswin),
    Cusum(&'a Cusum),
}

#[derive(Deserialize)]
enum Tagged {
    Adwin(Adwin),
    Ddm(ErrorRate<Ddm>),
    Eddm(ErrorRate<Eddm>),
    Kswin(Kswin),
    Cusum(Cusum),
}

/// A tagged kind first: a Page-Hinkley detector's fields are a map of seven
/// entries, which no tag of one entry matches.
#[derive(Deserialize)]
#[serde(untagged)]
enum Wire {
    Tagged(Tagged),
    PageHinkley(PageHinkley),
}

impl Serialize for DriftDetector {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::PageHinkley(p) => p.serialize(s),
            Self::Adwin(a) => TaggedRef::Adwin(a).serialize(s),
            Self::Ddm(e) => TaggedRef::Ddm(e).serialize(s),
            Self::Eddm(e) => TaggedRef::Eddm(e).serialize(s),
            Self::Kswin(k) => TaggedRef::Kswin(k).serialize(s),
            Self::Cusum(c) => TaggedRef::Cusum(c).serialize(s),
        }
    }
}

impl<'de> Deserialize<'de> for DriftDetector {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        Ok(match Wire::deserialize(d)? {
            Wire::PageHinkley(p) => Self::PageHinkley(p),
            Wire::Tagged(Tagged::Adwin(a)) => Self::Adwin(a),
            Wire::Tagged(Tagged::Ddm(e)) => Self::Ddm(e),
            Wire::Tagged(Tagged::Eddm(e)) => Self::Eddm(e),
            Wire::Tagged(Tagged::Kswin(k)) => Self::Kswin(k),
            Wire::Tagged(Tagged::Cusum(c)) => Self::Cusum(c),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AdwinCfg, KswinCfg};

    fn every_kind() -> Vec<DriftDetector> {
        vec![
            DriftDetector::PageHinkley(PageHinkley::new(0.5, 20.0)),
            DriftDetector::Adwin(Adwin::new(AdwinCfg::default())),
            DriftDetector::Ddm(ErrorRate {
                error_above: 1.0,
                detector: Ddm::new(30, 3.0),
            }),
            DriftDetector::Eddm(ErrorRate {
                error_above: 1.0,
                detector: Eddm::new(30, 0.9),
            }),
            DriftDetector::Kswin(Kswin::new(KswinCfg::default())),
            DriftDetector::Cusum(Cusum::new(0.5, 5.0)),
        ]
    }

    fn lcg(state: &mut u64) -> f64 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }

    /// Each kind saved mid-stream loads as itself and goes on to the bit; a
    /// Page-Hinkley detector's bytes are its fields' alone, as before.
    #[test]
    fn every_kind_is_saved_and_goes_on_to_the_bit() {
        for mut a in every_kind() {
            let mut s = 3u64;
            for _ in 0..400 {
                a.update(lcg(&mut s), 1.0, 1.0, 1.0);
            }
            let bytes = rmp_serde::to_vec_named(&a).unwrap();
            let mut b: DriftDetector = rmp_serde::from_slice(&bytes).unwrap();
            assert_eq!(a, b);
            let mut flags = 0;
            for i in 0..3000 {
                let r = lcg(&mut s) * if i > 1500 { 3.0 } else { 1.0 } + 0.4 * f64::from(i > 1500);
                let (fa, fb) = (a.update(r, 1.0, 1.0, 1.0), b.update(r, 1.0, 1.0, 1.0));
                assert_eq!(fa, fb);
                flags += usize::from(fa);
            }
            assert!(flags > 0, "{a:?} flagged the change");
            assert_eq!(a, b);
        }
        let ph = PageHinkley::new(0.5, 20.0);
        assert_eq!(
            rmp_serde::to_vec_named(&DriftDetector::PageHinkley(ph.clone())).unwrap(),
            rmp_serde::to_vec_named(&ph).unwrap(),
            "the file's form of a Page-Hinkley detector is unchanged"
        );
    }

    /// A restored detector of another kind, or of a shape its rows cannot
    /// run on, is refused; one saved mid-stream restores.
    #[test]
    fn a_restored_detector_is_held_to_its_kind_and_shape() {
        let kinds = every_kind();
        for (i, a) in kinds.iter().enumerate() {
            for (j, b) in kinds.iter().enumerate() {
                assert_eq!(a.restores_as(b), i == j, "{a:?} as {b:?}");
            }
        }
        let fresh = DriftDetector::Adwin(Adwin::new(AdwinCfg::default()));
        let mut a = fresh.clone();
        let mut s = 5u64;
        for _ in 0..500 {
            a.update(lcg(&mut s), 1.0, 1.0, 1.0);
        }
        assert!(a.restores_as(&fresh));
        let damaged = |edit: &dyn Fn(&mut serde_json::Value)| {
            let mut v = serde_json::to_value(&a).unwrap();
            edit(&mut v);
            serde_json::from_value::<DriftDetector>(v).unwrap()
        };
        for edit in [
            &(|v: &mut serde_json::Value| v["Adwin"]["levels"] = serde_json::json!([]))
                as &dyn Fn(&mut serde_json::Value),
            &|v| v["Adwin"]["width"] = serde_json::json!(3.0),
            &|v| v["Adwin"]["clock"] = serde_json::json!(0),
            &|v| v["Adwin"]["max_buckets"] = serde_json::json!(1),
        ] {
            assert!(!damaged(edit).restores_as(&fresh));
        }
        let fresh = DriftDetector::Kswin(Kswin::new(KswinCfg::default()));
        let mut k = fresh.clone();
        for _ in 0..150 {
            k.update(lcg(&mut s), 1.0, 1.0, 1.0);
        }
        assert!(k.restores_as(&fresh));
        let mut v = serde_json::to_value(&k).unwrap();
        v["Kswin"]["stat_size"] = serde_json::json!(60);
        assert!(
            !serde_json::from_value::<DriftDetector>(v)
                .unwrap()
                .restores_as(&fresh)
        );
        let mut v = serde_json::to_value(&k).unwrap();
        v["Kswin"]["window_size"] = serde_json::json!(50);
        assert!(
            !serde_json::from_value::<DriftDetector>(v)
                .unwrap()
                .restores_as(&fresh)
        );
    }

    /// Every kind is held to a shape its rows run on, each bound on its own
    /// (review 2026-11-23, DRIFT-5): an ADWIN whose top level is empty (its
    /// next drop, of the oldest bucket, finds none), wider than its cap, with
    /// no shortest part or a sum or spread that is not a finite number;
    /// an EDDM whose last error is after its last row (`n - last_error`
    /// wraps); a DDM, a Page-Hinkley or a CUSUM whose numbers are out of
    /// their range. Each kind saved mid-stream restores. (A number that is
    /// not finite, which JSON cannot write, is held in each kind's own
    /// tests.)
    #[test]
    fn every_kind_is_held_to_its_shape() {
        let mut s = 21u64;
        for fresh in every_kind() {
            let mut a = fresh.clone();
            for i in 0..400 {
                let r = lcg(&mut s) * 2.0 + if i > 300 { 1.5 } else { 0.0 };
                a.update(r, 1.0, 1.0, 0.99);
            }
            assert!(a.restores_as(&fresh), "{a:?}");
        }
        // Each kind after 100 rows of steady noise: every number finite
        // (a DDM's minimum is infinite before its first).
        let damaged = |kind: usize, edit: &dyn Fn(&mut serde_json::Value)| {
            let fresh = every_kind().swap_remove(kind);
            let mut fed = fresh.clone();
            let mut s = 23u64;
            for _ in 0..100 {
                fed.update(lcg(&mut s), 1.0, 1.0, 0.99);
            }
            assert!(fed.restores_as(&fresh));
            let mut v = serde_json::to_value(&fed).unwrap();
            edit(&mut v);
            let d = serde_json::from_value::<DriftDetector>(v).unwrap();
            d.restores_as(&fresh)
        };
        // Page-Hinkley, untagged.
        for edit in [
            &(|v: &mut serde_json::Value| v["threshold"] = serde_json::json!(0.0))
                as &dyn Fn(&mut serde_json::Value),
            &|v| v["delta"] = serde_json::json!(-1.0),
            &|v| v["n"] = serde_json::json!(-1.0),
            &|v| v["w"] = serde_json::json!(-1.0),
            &|v| v["min_cum"] = serde_json::json!(1.0),
            &|v| {
                v["cum"] = serde_json::json!(-2.0);
                v["min_cum"] = serde_json::json!(-1.0);
            },
        ] {
            assert!(!damaged(0, edit), "page_hinkley");
        }
        // ADWIN.
        for edit in [
            &(|v: &mut serde_json::Value| {
                v["Adwin"]["levels"] = serde_json::json!([[[1.0, 0.0]], []]);
                v["Adwin"]["width"] = serde_json::json!(1.0);
                v["Adwin"]["total"] = serde_json::json!(1.0);
            }) as &dyn Fn(&mut serde_json::Value),
            &|v| {
                v["Adwin"]["levels"] = serde_json::json!([[[1.0, 0.0], [1.0, 0.0]]]);
                v["Adwin"]["width"] = serde_json::json!(2.0);
                v["Adwin"]["max_width"] = serde_json::json!(1);
            },
            &|v| v["Adwin"]["min_window_length"] = serde_json::json!(0),
            &|v| v["Adwin"]["var"] = serde_json::json!(-1.0),
            &|v| {
                v["Adwin"]["levels"] = serde_json::json!([[[1.0, -1.0]]]);
                v["Adwin"]["width"] = serde_json::json!(1.0);
            },
        ] {
            assert!(!damaged(1, edit), "adwin");
        }
        // DDM.
        for edit in [
            &(|v: &mut serde_json::Value| v["Ddm"]["detector"]["p"] = serde_json::json!(1.5))
                as &dyn Fn(&mut serde_json::Value),
            &|v| v["Ddm"]["detector"]["n"] = serde_json::json!(-1.0),
            &|v| v["Ddm"]["detector"]["drift_threshold"] = serde_json::json!(0.0),
            &|v| v["Ddm"]["detector"]["s_min"] = serde_json::json!(-1.0),
            &|v| v["Ddm"]["error_above"] = serde_json::json!(-1.0),
        ] {
            assert!(!damaged(2, edit), "ddm");
        }
        // EDDM.
        for edit in [
            &(|v: &mut serde_json::Value| {
                v["Eddm"]["detector"]["last_error"] = serde_json::json!(101)
            }) as &dyn Fn(&mut serde_json::Value),
            &|v| v["Eddm"]["detector"]["errors"] = serde_json::json!(101),
            &|v| v["Eddm"]["detector"]["k"] = serde_json::json!(3.0),
            &|v| v["Eddm"]["detector"]["beta"] = serde_json::json!(0.0),
            &|v| v["Eddm"]["detector"]["sq"] = serde_json::json!(-1.0),
        ] {
            assert!(!damaged(3, edit), "eddm");
        }
        // CUSUM.
        for edit in [
            &(|v: &mut serde_json::Value| v["Cusum"]["h"] = serde_json::json!(0.0))
                as &dyn Fn(&mut serde_json::Value),
            &|v| v["Cusum"]["k"] = serde_json::json!(-1.0),
            &|v| v["Cusum"]["up"] = serde_json::json!(-1.0),
            &|v| v["Cusum"]["down"] = serde_json::json!(-0.5),
        ] {
            assert!(!damaged(5, edit), "cusum");
        }
    }

    /// A damaged ADWIN whose top level is empty is refused; were it run, the
    /// drop of its oldest bucket would find none and leave the window past
    /// its cap.
    #[test]
    fn an_empty_top_level_is_no_window() {
        let mut a = Adwin::new(AdwinCfg {
            max_width: 1,
            ..AdwinCfg::default()
        });
        a.update(1.0);
        let mut v = serde_json::to_value(&a).unwrap();
        v["levels"] = serde_json::json!([[[1.0, 0.0]], []]);
        let d: Adwin = serde_json::from_value(v).unwrap();
        assert!(!d.sound());
    }

    /// A residual so many scales out that `r / σ` overflows is a drift, not
    /// a row unseen; a clock step of 0 with such a residual wipes nothing;
    /// and an excess that overflows below leaves Page-Hinkley reading on,
    /// not NaN forever (review 2026-11-23, DRIFT-6).
    #[test]
    fn overflowing_rows_flag_or_pass_and_leave_the_detectors_reading() {
        // (2) `|r| / σ = inf`: Page-Hinkley and CUSUM flag.
        for k in [
            DriftDetector::PageHinkley(PageHinkley::new(0.5, 20.0)),
            DriftDetector::Cusum(Cusum::new(0.5, 5.0)),
        ] {
            let mut k = k;
            for _ in 0..50 {
                k.update(0.3, 1.0, 1.0, 0.99);
            }
            assert!(k.update(1e100, 1e-250, 1.0, 0.99), "{k:?}");
        }
        // (3) A step of 0 with `r / σ = ±inf` keeps CUSUM's sums.
        let mut c = Cusum::new(0.5, 50.0);
        for _ in 0..20 {
            c.update(2.0, 1.0);
        }
        let before = c.sums();
        let mut k = DriftDetector::Cusum(c);
        k.update(1e100, 1e-250, 0.0, 1.0);
        k.update(-1e100, 1e-250, 0.0, 1.0);
        let DriftDetector::Cusum(c) = k else {
            unreachable!()
        };
        assert_eq!(c.sums(), before);
        // (1) A mean of `1e305`, then a long step with an error of 0: the
        // excess overflows below, and the statistic stays a number that
        // the next rows of a drift move past the threshold.
        let mut ph = PageHinkley::new(0.5, 20.0);
        ph.update(1e305, 1.0, 1.0);
        ph.update(0.0, 86400.0, 1.0);
        assert!(!ph.statistic().is_nan(), "{ph:?}");
        let flagged = (0..1000).any(|_| ph.update(1e306, 1.0, 1.0));
        assert!(flagged, "{ph:?}");
    }

    /// The tags a file carries for each kind other than Page-Hinkley are
    /// frozen: a rename would break every file that holds one.
    #[test]
    fn the_tags_are_frozen() {
        let tags: Vec<String> = every_kind()
            .iter()
            .skip(1)
            .map(|k| {
                let v: serde_json::Value = serde_json::to_value(k).unwrap();
                let map = v.as_object().unwrap();
                assert_eq!(map.len(), 1, "{v}");
                map.keys().next().unwrap().clone()
            })
            .collect();
        assert_eq!(tags, ["Adwin", "Ddm", "Eddm", "Kswin", "Cusum"]);
        let ph = serde_json::to_value(&every_kind()[0]).unwrap();
        assert!(
            ph.get("threshold").is_some(),
            "Page-Hinkley's fields, untagged: {ph}"
        );
    }

    /// The scale gates the two kinds read on the clock and no other: a row
    /// with no scale ages Page-Hinkley and is no row for CUSUM.
    #[test]
    fn only_the_clocked_kinds_need_a_scale() {
        for scale in [0.0, f64::NAN, f64::INFINITY] {
            for mut k in every_kind() {
                let before = k.clone();
                let mut s = 7u64;
                for _ in 0..200 {
                    k.update(lcg(&mut s) * 5.0, scale, 1.0, 1.0);
                }
                let moved = k != before;
                let clocked = matches!(k, DriftDetector::PageHinkley(_) | DriftDetector::Cusum(_));
                assert_eq!(moved, !clocked, "{k:?} at scale {scale}");
            }
        }
        // With a scale, z = 3 / 2 = 1.5 and C+ = 1, past h = 0.9.
        let mut c = DriftDetector::Cusum(Cusum::new(0.5, 0.9));
        assert!(c.update(3.0, 2.0, 1.0, 1.0));
    }

    /// A row is read through its scale: Page-Hinkley takes `|r| / σ` and
    /// CUSUM `r / σ`, to the bit, and a scale that is not a positive finite
    /// number is no row for CUSUM, even for a residual that would flag it.
    #[test]
    fn a_row_is_read_in_units_of_its_scale() {
        let ph = PageHinkley::new(0.5, 20.0);
        let (mut a, mut direct) = (DriftDetector::PageHinkley(ph.clone()), ph);
        for (r, scale, d) in [(3.0, 2.0, 1.5), (-1.0, 0.5, 0.7), (4.0, 3.0, 2.0)] {
            assert_eq!(
                a.update(r, scale, d, 0.9),
                direct.update(f64::abs(r) / scale, d, 0.9)
            );
            assert_eq!(a, DriftDetector::PageHinkley(direct.clone()));
        }
        assert!(direct.n() == 3.0);
        let cusum = Cusum::new(0.5, 4.0);
        let (mut a, mut direct) = (DriftDetector::Cusum(cusum.clone()), cusum);
        for (r, scale, d) in [(3.0, 2.0, 1.5), (-5.0, 0.5, 0.7), (4.0, 3.0, 2.0)] {
            assert_eq!(a.update(r, scale, d, 1.0), direct.update(r / scale, d));
            assert_eq!(a, DriftDetector::Cusum(direct.clone()));
        }
        // No scale: no row, whatever the residual.
        for scale in [
            0.0,
            -1.0,
            -1e-300,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ] {
            let mut c = DriftDetector::Cusum(Cusum::new(0.5, 1.0));
            assert!(!c.update(100.0, scale, 1.0, 1.0), "scale {scale}");
            assert!(!c.update(-100.0, scale, 1.0, 1.0), "scale {scale}");
            assert_eq!(c, DriftDetector::Cusum(Cusum::new(0.5, 1.0)));
        }
    }

    /// A row with nothing to read ages Page-Hinkley's mean and no other kind.
    #[test]
    fn age_moves_page_hinkleys_mean_and_nothing_else() {
        let mut ph = PageHinkley::new(0.5, 1e9);
        for e in [1.0, 2.0, 3.0] {
            ph.update(e, 1.0, 0.9);
        }
        let mut aged = ph.clone();
        aged.age(0.5);
        assert_ne!(aged, ph);
        let mut a = DriftDetector::PageHinkley(ph.clone());
        a.age(0.5);
        assert_eq!(a, DriftDetector::PageHinkley(aged));
        for mut k in every_kind().into_iter().skip(1) {
            let before = k.clone();
            k.age(0.5);
            assert_eq!(k, before);
        }
    }

    /// `reset` starts each kind again from nothing seen: equal to a new one.
    #[test]
    fn reset_is_a_new_detector_of_every_kind() {
        for (mut a, fresh) in every_kind().into_iter().zip(every_kind()) {
            let mut s = 13u64;
            for _ in 0..300 {
                a.update(lcg(&mut s) * 2.0 + 1.0, 1.0, 1.0, 0.95);
            }
            assert_ne!(a, fresh, "{a:?} has seen rows");
            a.reset();
            assert_eq!(a, fresh);
        }
    }

    /// What each kind reads: CUSUM the signed scaled residual (a residual of
    /// `−3σ` is a downward shift), ADWIN and KSWIN its size, DDM and EDDM
    /// whether its size passes `error_above`, strictly.
    #[test]
    fn each_kind_reads_its_own_signal() {
        let mut c = Cusum::new(0.5, 1.0);
        let mut k = DriftDetector::Cusum(c.clone());
        assert!(k.update(-4.0, 2.0, 1.0, 1.0));
        assert!(c.update(-2.0, 1.0));
        let mut a = DriftDetector::Adwin(Adwin::new(AdwinCfg::default()));
        a.update(-3.0, 1.0, 1.0, 1.0);
        let DriftDetector::Adwin(a) = a else {
            unreachable!()
        };
        assert_eq!(a.total(), 3.0);
        let mut kw = DriftDetector::Kswin(Kswin::new(KswinCfg::default()));
        kw.update(-2.5, 1.0, 1.0, 1.0);
        let DriftDetector::Kswin(kw) = kw else {
            unreachable!()
        };
        assert_eq!(kw.window()[0], 2.5);
        let mut d = DriftDetector::Ddm(ErrorRate {
            error_above: 1.0,
            detector: Ddm::new(30, 3.0),
        });
        d.update(-1.0, 1.0, 1.0, 1.0);
        d.update(1.0 + 1e-12, 1.0, 1.0, 1.0);
        let DriftDetector::Ddm(d) = d else {
            unreachable!()
        };
        assert_eq!(d.detector.p(), 0.5, "at error_above no error, past it one");
        let mut e = DriftDetector::Eddm(ErrorRate {
            error_above: 1.0,
            detector: Eddm::new(30, 0.9),
        });
        e.update(1.0, 1.0, 1.0, 1.0);
        e.update(-2.0, 1.0, 1.0, 1.0);
        let DriftDetector::Eddm(e) = e else {
            unreachable!()
        };
        assert_eq!(e.detector.distances().0, 2.0, "the first error is row 2");
    }
}
