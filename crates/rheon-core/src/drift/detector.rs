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
//! without one ages Page-Hinkley's mean and is not read; ADWIN and KSWIN test
//! a location or a distribution with no unit in their settings, so they read
//! the error's size as it is; DDM and EDDM count errors, and what counts as
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

impl DriftDetector {
    /// Feed one row's residual `r`, the slot's scale before it, the row's
    /// clock step `d` and the decay `lam` over it. Returns whether the row
    /// flags a drift. The caller passes only a row that counts, of positive
    /// weight with a finite residual; [`Self::age`] the others.
    #[inline]
    pub fn update(&mut self, r: f64, scale: f64, d: f64, lam: f64) -> bool {
        let scaled = scale.is_finite() && scale > 0.0;
        match self {
            Self::PageHinkley(p) => {
                if scaled {
                    p.update(r.abs() / scale, d, lam)
                } else {
                    p.age(lam);
                    false
                }
            }
            Self::Cusum(c) => scaled && c.update(r / scale, d),
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
