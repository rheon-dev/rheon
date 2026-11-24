//! A `vector_ar` spec's reads, as frames: `ModelBank.vector_ar` in Python.
//! Each is computed from the fit as its moments stand
//! ([`rheon_core::VectorAr::fit_now`]); nothing is kept for them.

use polars::prelude::*;
use rheon_core::{VectorArFit, VectorArTable};

use super::{Bank, GroupKey};
use crate::spec::ModelKind;
use crate::stream::AnyModel;

/// What a read asks beside the table.
#[derive(Debug, Clone, Default)]
pub struct VectorArRead {
    /// Periods of the impulse responses and the decomposition.
    pub periods: usize,
    /// The impulse responses' standard errors beside them.
    pub stderr: bool,
    /// A test's caused series, its causing ones (absent: every other) and
    /// its kind (absent: the spec's).
    pub caused: Vec<String>,
    pub causing: Option<Vec<String>>,
    pub kind: Option<String>,
}

/// NaN, the core's "undefined", as null.
fn opt(v: f64) -> Option<f64> {
    (!v.is_nan()).then_some(v)
}

/// The rows of one table, column by column, before the leading `spec`,
/// `group` and `instance`.
#[derive(Default)]
struct Rows {
    lead: Vec<(Option<String>, String)>,
    ints: Vec<(&'static str, Vec<i64>)>,
    strs: Vec<(&'static str, Vec<String>)>,
    nums: Vec<(&'static str, Vec<Option<f64>>)>,
    flags: Vec<(&'static str, Vec<bool>)>,
}

impl Rows {
    fn push_int(&mut self, name: &'static str, v: i64) {
        match self.ints.iter_mut().find(|(n, _)| *n == name) {
            Some((_, c)) => c.push(v),
            None => self.ints.push((name, vec![v])),
        }
    }
    fn push_str(&mut self, name: &'static str, v: &str) {
        match self.strs.iter_mut().find(|(n, _)| *n == name) {
            Some((_, c)) => c.push(v.to_string()),
            None => self.strs.push((name, vec![v.to_string()])),
        }
    }
    fn push_num(&mut self, name: &'static str, v: f64) {
        match self.nums.iter_mut().find(|(n, _)| *n == name) {
            Some((_, c)) => c.push(opt(v)),
            None => self.nums.push((name, vec![opt(v)])),
        }
    }
    fn push_flag(&mut self, name: &'static str, v: bool) {
        match self.flags.iter_mut().find(|(n, _)| *n == name) {
            Some((_, c)) => c.push(v),
            None => self.flags.push((name, vec![v])),
        }
    }
}

impl Bank {
    /// A `vector_ar` spec's fit read as a frame ([`VectorArTable`]), led by
    /// `spec`, `group` and `instance`, per group in key order and instance:
    ///
    /// - `Irf` and `OrthIrf`: `period` (0 to `periods`), `response`,
    ///   `impulse`, `value` and, with `stderr`, `stderr`;
    /// - `Fevd`: `horizon` (1 to `periods`), `response`, `shock`, `share`;
    /// - `Stability`: `root` (largest modulus first), `real`, `imag`,
    ///   `modulus`, `stable`;
    /// - `Mean`: `target`, `mean`;
    /// - `Order`: `order` (0 to `lags`), `aic`, `bic`, `hqic`, `fpe`,
    ///   `in_force`;
    /// - `Causality`: `statistic`, `p_value`, `df_num`, `df_denom` (null for
    ///   a χ² law), `order`.
    ///
    /// # Errors
    ///
    /// `spec` out of range or not a `vector_ar`; a group whose fit is not
    /// identified yet, named with the reason; a test naming a series that
    /// is not a target; `Mean` of a spec with features.
    pub fn vector_ar(
        &self,
        spec: usize,
        group: Option<&[GroupKey]>,
        table: VectorArTable,
        read: &VectorArRead,
    ) -> Result<DataFrame, String> {
        let keys = self.sorted_keys(spec, group)?;
        let (s, states) = (&self.specs[spec], &self.states[spec]);
        let ModelKind::VectorAr(v) = &s.model else {
            return Err(format!(
                "spec {:?}: its model is {:?}, not \"vector_ar\", and only a vector_ar keeps \
                 these reads. Name a vector_ar spec of the bank.",
                s.name,
                s.model.kind_name()
            ));
        };
        let targets: Vec<&str> = s.targets.iter().map(String::as_str).collect();
        if table == VectorArTable::Mean && !s.features.is_empty() {
            return Err(format!(
                "spec {:?}: its long-run mean depends on its features' values, which a VAR with \
                 features does not keep. Read the mean of a spec without features.",
                s.name
            ));
        }
        let test = if table == VectorArTable::Causality {
            if read.caused.is_empty() {
                return Err(format!(
                    "spec {:?}: table=\"causality\" needs caused, the series whose equations \
                     the test reads. Name at least one target in caused.",
                    s.name
                ));
            }
            let t = crate::spec::test_of(
                s,
                &crate::spec::CausalitySpec {
                    caused: read.caused.clone(),
                    causing: read.causing.clone(),
                },
            )
            .map_err(|e| format!("spec {:?}: {}", s.name, e.trim_start_matches("vector_ar: ")))?;
            let kind = crate::spec::causality_kind_of(
                read.kind.as_deref().or(v.causality_kind.as_deref()),
            )
            .map_err(|e| e.replace("vector_ar: causality_kind", "ModelBank.vector_ar: kind"))?;
            let mut no = String::new();
            if t.causing.is_empty() || t.caused.iter().any(|j| t.causing.contains(j)) {
                no = "the caused and causing series must be two sets, neither empty, with no \
                      series in both."
                    .into();
            }
            if kind.single_equation() && t.caused.len() != 1 {
                no = format!("kind {kind:?} tests one caused equation, but caused names more.");
            }
            if !no.is_empty() {
                return Err(format!("spec {:?}: {no} Name the two sets again.", s.name));
            }
            Some((t, kind))
        } else {
            None
        };
        let mut rows = Rows::default();
        for key in keys {
            for (label, model) in &states[key].models {
                let AnyModel::VectorAr(m) = model else {
                    unreachable!("a vector_ar spec builds vector_ar models");
                };
                let fit = m.fit_now().map_err(|e| {
                    format!(
                        "spec {:?}, group {:?}: its fit cannot be read, since {e}",
                        s.name,
                        key.as_str()
                    )
                })?;
                let lead = (key.as_str().map(str::to_string), label.clone());
                fill(&mut rows, &lead, &fit, table, read, &targets, test.as_ref())?;
            }
        }
        frame(&s.name, rows)
    }
}

fn fill(
    rows: &mut Rows,
    lead: &(Option<String>, String),
    fit: &VectorArFit,
    table: VectorArTable,
    read: &VectorArRead,
    targets: &[&str],
    test: Option<&(rheon_core::VectorArTest, rheon_core::CausalityKind)>,
) -> Result<(), String> {
    let k = fit.k;
    let h = read.periods;
    match table {
        VectorArTable::Irf | VectorArTable::OrthIrf => {
            let orth = table == VectorArTable::OrthIrf;
            let vals = if orth {
                fit.orth_ma_rep(h)
            } else {
                fit.ma_rep(h)
            };
            let se = read.stderr.then(|| fit.irf_stderr(h, orth));
            for i in 0..=h {
                for r in 0..k {
                    for c in 0..k {
                        rows.lead.push(lead.clone());
                        rows.push_int("period", i as i64);
                        rows.push_str("response", targets[r]);
                        rows.push_str("impulse", targets[c]);
                        let at = (i * k + r) * k + c;
                        rows.push_num("value", vals[at]);
                        if let Some(se) = &se {
                            rows.push_num("stderr", se[at]);
                        }
                    }
                }
            }
        }
        VectorArTable::Fevd => {
            let d = fit.fevd(h);
            for e in 0..k {
                for i in 0..h {
                    for j in 0..k {
                        rows.lead.push(lead.clone());
                        rows.push_int("horizon", i as i64 + 1);
                        rows.push_str("response", targets[e]);
                        rows.push_str("shock", targets[j]);
                        rows.push_num("share", d[(e * h + i) * k + j]);
                    }
                }
            }
        }
        VectorArTable::Stability => {
            let ev = fit.eigenvalues()?;
            let stable = ev.iter().all(|(r, i)| r.hypot(*i) < 1.0);
            for (n, (re, im)) in ev.iter().enumerate() {
                rows.lead.push(lead.clone());
                rows.push_int("root", n as i64);
                rows.push_num("real", *re);
                rows.push_num("imag", *im);
                rows.push_num("modulus", re.hypot(*im));
                rows.push_flag("stable", stable);
            }
        }
        VectorArTable::Mean => {
            let mean = fit.mean();
            for (j, t) in targets.iter().enumerate() {
                rows.lead.push(lead.clone());
                rows.push_str("target", t);
                rows.push_num("mean", mean.as_ref().map_or(f64::NAN, |m| m[j]));
            }
        }
        VectorArTable::Order => {
            for (q, ics) in fit.info_criteria().iter().enumerate() {
                rows.lead.push(lead.clone());
                rows.push_int("order", q as i64);
                for (name, v) in ["aic", "bic", "hqic", "fpe"].into_iter().zip(ics) {
                    rows.push_num(name, *v);
                }
                rows.push_flag("in_force", q == fit.order);
            }
        }
        VectorArTable::Causality => {
            let (t, kind) = test.expect("a causality read carries its test");
            let r = fit.test(&t.caused, &t.causing, *kind);
            rows.lead.push(lead.clone());
            for (name, v) in ["statistic", "p_value", "df_num", "df_denom"]
                .into_iter()
                .zip(r)
            {
                rows.push_num(name, v);
            }
            rows.push_int("order", fit.order as i64);
        }
    }
    Ok(())
}

fn frame(spec: &str, rows: Rows) -> Result<DataFrame, String> {
    let n = rows.lead.len();
    let mut cols = vec![
        Column::new("spec".into(), vec![spec; n]),
        Column::new(
            "group".into(),
            rows.lead.iter().map(|(g, _)| g.clone()).collect::<Vec<_>>(),
        ),
        Column::new(
            "instance".into(),
            rows.lead.iter().map(|(_, i)| i.clone()).collect::<Vec<_>>(),
        ),
    ];
    // In the order the table's columns are documented: the integers lead,
    // then the names, the numbers and the flags.
    for (name, v) in rows.ints {
        cols.push(Column::new(name.into(), v));
    }
    for (name, v) in rows.strs {
        cols.push(Column::new(name.into(), v));
    }
    for (name, v) in rows.nums {
        cols.push(Column::new(name.into(), v));
    }
    for (name, v) in rows.flags {
        cols.push(Column::new(name.into(), v));
    }
    DataFrame::new(n, cols).map_err(|e| e.to_string())
}
