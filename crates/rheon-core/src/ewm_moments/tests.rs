//! `ewm_moments`' state (PLAN 554, part 3): held to the Python reference
//! (`tests/reference_ewm_moments.py`, whose numbers on these streams are
//! `reference_values.rs`, written by `scripts/ewm_moments_literals_554.py`)
//! and to the definition longhand; order 2 to `EwDiag` to the bit; chunks
//! and a saved state to the bit; and each withholding condition from both
//! sides.

use super::*;
use crate::EwDiag;

include!("reference_values.rs");

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 11) as f64 / (1u64 << 53) as f64
}

/// A row: value, weight, clock step.
type Row = (f64, f64, f64);

/// The stream `scripts/ewm_moments_literals_554.py` feeds the reference:
/// `x = 3 + 4u³ + v/2`, `w = 1/4 + q` but 0 on `zero_rows`, the clock
/// stepping 1, 1.5, 2 under `steps` and 1 a row otherwise. The Rust step is
/// the clock's difference, which is exact: every clock is a sum of halves.
fn stream(seed: u64, zero_rows: &[usize], steps: bool) -> Vec<Row> {
    let mut s = seed;
    (0..60)
        .map(|i| {
            let (u, v, q) = (lcg(&mut s), lcg(&mut s), lcg(&mut s));
            let x = 3.0 + 4.0 * u * u * u + 0.5 * v;
            let w = if zero_rows.contains(&i) {
                0.0
            } else {
                0.25 + q
            };
            let d = if steps {
                1.0 + (i % 3) as f64 * 0.5
            } else {
                1.0
            };
            (x, w, d)
        })
        .collect()
}

fn term(name: &str, order: u32, kind: MomentKind, bias: bool) -> MomentTerm {
    MomentTerm::of_column(name, order, kind, bias)
}

fn cfg(decay: Decay, terms: Vec<MomentTerm>) -> EwmMomentsCfg {
    EwmMomentsCfg {
        decay,
        terms,
        min_weight: 0.0,
        max_state_bytes: None,
        clocked: false,
        lag_tolerance: None,
        se: false,
        window_rows: None,
        window_clock: None,
    }
}

/// The reference's terms: plug-in central and cumulant to 6, standardized
/// 3 to 6 and raw 1 to 6; then, suffixed `u`, the unbiased central and
/// cumulant to 4 and standardized 3 and 4.
fn reference_terms() -> Vec<MomentTerm> {
    use MomentKind::*;
    let mut t = Vec::new();
    for p in 2..=6 {
        t.push(term(&format!("c{p}"), p, Central, true));
        t.push(term(&format!("k{p}"), p, Cumulant, true));
    }
    for p in 3..=6 {
        t.push(term(&format!("s{p}"), p, Standardized, true));
    }
    for p in 1..=6 {
        t.push(term(&format!("r{p}"), p, Raw, true));
    }
    for p in 2..=4 {
        t.push(term(&format!("c{p}u"), p, Central, false));
        t.push(term(&format!("k{p}u"), p, Cumulant, false));
    }
    for p in 3..=4 {
        t.push(term(&format!("s{p}u"), p, Standardized, false));
    }
    t
}

/// `|got − want| / max(|want|, 1)`, 0 where both are NaN, infinite where
/// one is.
fn gap(got: f64, want: f64) -> f64 {
    match (got.is_nan(), want.is_nan()) {
        (true, true) => 0.0,
        (false, false) => (got - want).abs() / want.abs().max(1.0),
        _ => f64::INFINITY,
    }
}

/// Runs `rows` and returns, at each of the reference's rows, every number
/// by the reference's name.
fn run_rust(decay: Decay, rows: &[Row]) -> Vec<(usize, String, f64)> {
    let mut m = EwmMoments::new(cfg(decay, reference_terms())).unwrap();
    let mut out = Vec::new();
    for (i, &(x, w, d)) in rows.iter().enumerate() {
        m.update(x, m.cfg.decay.factor(d), w);
        if [2, 31, 59].contains(&i) {
            out.push((i, "W".to_owned(), m.n_eff()));
            out.push((i, "kish_r2".to_owned(), m.kish_ratio()));
            for (t, v) in m.cfg.terms.iter().zip(m.read()) {
                out.push((i, t.name.clone(), v.unwrap_or(f64::NAN)));
            }
        }
    }
    out
}

/// The Rust against the reference, number by number: the largest gap, and
/// the numbers that are the reference's to the bit.
fn against(reference: &[(usize, &str, f64)], got: &[(usize, String, f64)]) -> (f64, Vec<String>) {
    let mut worst = 0.0f64;
    let mut exact = Vec::new();
    assert_eq!(reference.len(), got.len(), "every reference number is read");
    for &(row, name, want) in reference {
        let &(_, _, v) = got
            .iter()
            .find(|(r, n, _)| *r == row && n == name)
            .unwrap_or_else(|| panic!("row {row}: no {name}"));
        let g = gap(v, want);
        assert!(g.is_finite(), "row {row} {name}: {v} against {want}");
        worst = worst.max(g);
        if v.to_bits() == want.to_bits() || (v.is_nan() && want.is_nan()) {
            exact.push(format!("{row}:{name}"));
        }
    }
    (worst, exact)
}

/// The reference on stream A (`lam = 0.9`, one clock unit a row, a
/// zero-weight first row) and stream B (a half-life of 7 on an uneven
/// clock, zero-weight rows inside): every term, plug-in and unbiased, within
/// 5e-14 of the reference's own number relative to its size (or to 1 below
/// it), the reference withholding exactly where the Rust does. Measured
/// 2026-11-23: 2.6e-15 on A and 3.5e-15 on B. The two part by rounding
/// alone: the Rust keeps the mean as a pair (`crate::comp`) where the
/// reference keeps it whole, forms order 2 by `EwDiag`'s expression, `r_3`
/// and `r_4` by products where the reference calls `pow`, and on B decays a
/// zero-weight row's clock step on its own where the reference folds it
/// into the next row's.
///
/// To the bit where the operations are the same: on A the weight `W` and
/// Kish's `r_2`, at every row checked (no libm on a unit step), and at row
/// 2 every central moment from order 3, whose rows' mean held no rounding
/// yet.
#[test]
fn matches_the_python_reference() {
    let a = run_rust(Decay::Lam(0.9), &stream(554, &[0], false));
    let (worst_a, exact_a) = against(REF_A, &a);
    let b = run_rust(Decay::Halflife(7.0), &stream(5541, &[0, 17, 18, 40], true));
    let (worst_b, _) = against(REF_B, &b);
    eprintln!("ewm_moments against the reference: A {worst_a:e}, B {worst_b:e}");
    assert!(worst_a < 5e-14, "stream A: {worst_a:e}");
    assert!(worst_b < 5e-14, "stream B: {worst_b:e}");
    for row in [2, 31, 59] {
        for name in ["W", "kish_r2"] {
            assert!(
                exact_a.contains(&format!("{row}:{name}")),
                "A row {row}: {name}"
            );
        }
    }
    for p in 3..=6 {
        assert!(exact_a.contains(&format!("2:c{p}")), "A row 2: c{p}");
    }
    // The reference withholds the unbiased third and fourth moments at row
    // 2 (two rows of weight); the comparison above held the Rust to the
    // same, and the reasons are these.
    let mut m = EwmMoments::new(cfg(Decay::Lam(0.9), reference_terms())).unwrap();
    for &(x, w, d) in &stream(554, &[0], false)[..3] {
        m.update(x, m.cfg.decay.factor(d), w);
    }
    let why = |name: &str| {
        let i = m.cfg.terms.iter().position(|t| t.name == name).unwrap();
        m.value(i)
    };
    assert_eq!(why("c3u"), Err(Withheld::TooFewRowsThird));
    assert_eq!(why("c4u"), Err(Withheld::TooFewRowsFourth));
    assert!(why("c2u").is_ok());
}

/// The weighted definition, two-pass, from each row's final weight
/// `ω_t = w_t Π_{s>t} λ_s`: every central moment to order 6 and the raw
/// ones, on stream B, within 1e-13.
#[test]
fn matches_the_definition_longhand() {
    let rows = stream(5541, &[0, 17, 18, 40], true);
    let decay = Decay::Halflife(7.0);
    let mut m = EwmMoments::new(cfg(decay, reference_terms())).unwrap();
    for &(x, w, d) in &rows {
        m.update(x, decay.factor(d), w);
    }
    let omega: Vec<f64> = (0..rows.len())
        .map(|t| {
            rows[t + 1..]
                .iter()
                .fold(rows[t].1, |o, later| o * decay.factor(later.2))
        })
        .collect();
    let w: f64 = omega.iter().sum();
    let mean = rows.iter().zip(&omega).map(|(r, o)| o * r.0).sum::<f64>() / w;
    for p in 2..=6 {
        let want = rows
            .iter()
            .zip(&omega)
            .map(|(r, o)| o * (r.0 - mean).powi(p))
            .sum::<f64>()
            / w;
        assert!(gap(m.central(p as u32), want) < 1e-13, "c{p}");
        let raw = rows
            .iter()
            .zip(&omega)
            .map(|(r, o)| o * r.0.powi(p))
            .sum::<f64>()
            / w;
        let i = m
            .cfg
            .terms
            .iter()
            .position(|t| t.name == format!("r{p}"))
            .unwrap();
        assert!(gap(m.value(i).unwrap(), raw) < 1e-13, "r{p}");
    }
    assert!(gap(m.n_eff(), w) < 1e-13);
    let kish = omega.iter().map(|o| o * o).sum::<f64>() / (w * w);
    assert!(gap(m.kish_ratio(), kish) < 1e-13);
}

/// Order 2 is `EwDiag`'s update: the weight, the mean's pair and the
/// centred second moment to the bit on every row, through zero-weight rows
/// (the first among them), a long gap, and a level of 1e8 held long enough
/// for the pair's low part to carry the mean.
#[test]
fn order_two_is_ewdiag_to_the_bit() {
    let decay = Decay::Halflife(5.0);
    let terms = (2..=6)
        .map(|p| term(&format!("c{p}"), p, MomentKind::Central, true))
        .collect();
    let mut m = EwmMoments::new(cfg(decay, terms)).unwrap();
    let mut e = EwDiag::new(1);
    let mut s = 99u64;
    for i in 0..1500 {
        let x = if i >= 300 {
            1e8 + 0.37
        } else {
            1e8 + lcg(&mut s) - 0.5
        };
        let w = if i == 0 || i % 13 == 4 {
            0.0
        } else {
            0.5 + lcg(&mut s)
        };
        let d = if i == 700 { 300.0 } else { 1.0 };
        let lam = decay.factor(d);
        m.update(x, lam, w);
        e.update(&[x], lam, w);
        let (mean, lo) = e.mean_pair(0);
        assert_eq!(m.n_eff().to_bits(), e.n_eff().to_bits(), "row {i}");
        assert_eq!(
            (m.sums[0].m[0].to_bits(), m.sums[0].m_lo[0].to_bits()),
            (mean.to_bits(), lo.to_bits()),
            "row {i}"
        );
        assert_eq!(m.central(2).to_bits(), e.var(0).to_bits(), "row {i}");
    }
    assert_eq!(m.mean(), 1e8 + 0.37);
}

/// The rows fed whole and in chunks of 1 to 7, the state saved and loaded
/// at each chunk's end in both encodings, read the same bits on every row
/// and end in the same state.
#[test]
fn chunks_and_a_saved_state_go_on_to_the_bit() {
    let rows = stream(5541, &[0, 17, 18, 40], true);
    let decay = Decay::Halflife(7.0);
    let fresh = || EwmMoments::new(cfg(decay, reference_terms())).unwrap();
    let bits = |m: &EwmMoments| -> Vec<u64> {
        m.read()
            .iter()
            .map(|v| v.map_or(u64::MAX, f64::to_bits))
            .collect()
    };
    let mut whole = fresh();
    let mut want = Vec::new();
    for &(x, w, d) in &rows {
        whole.update(x, decay.factor(d), w);
        want.push(bits(&whole));
    }
    for named in [false, true] {
        let mut s = 7u64;
        let mut m = fresh();
        let mut i = 0;
        while i < rows.len() {
            let n = 1 + (lcg(&mut s) * 7.0) as usize;
            for &(x, w, d) in &rows[i..(i + n).min(rows.len())] {
                m.update(x, decay.factor(d), w);
                assert_eq!(bits(&m), want[i], "row {i}, named {named}");
                i += 1;
            }
            let bytes = if named {
                rmp_serde::to_vec_named(&m).unwrap()
            } else {
                rmp_serde::to_vec(&m).unwrap()
            };
            m = rmp_serde::from_slice::<EwmMoments>(&bytes)
                .unwrap()
                .checked()
                .unwrap();
        }
        assert_eq!(
            rmp_serde::to_vec(&m).unwrap(),
            rmp_serde::to_vec(&whole).unwrap(),
            "named {named}"
        );
    }
}

/// A zero-weight first row -- of any value, NaN among them -- leaves the
/// state empty, every term withheld as "no weight yet" and no NaN in the
/// state; the rows after it read what they read without it, to the bit.
#[test]
fn a_zero_weight_first_row_changes_nothing() {
    let decay = Decay::Lam(0.9);
    let rows = stream(554, &[], false);
    for first in [f64::NAN, 1e300, 0.0, 3.5] {
        let mut m = EwmMoments::new(cfg(decay, reference_terms())).unwrap();
        m.update(first, decay.factor(1.0), 0.0);
        m.update(first, decay.factor(1.0), 0.0);
        assert!(
            m.read().iter().all(|v| *v == Err(Withheld::NoWeight)),
            "{first}"
        );
        assert_eq!(m, EwmMoments::new(cfg(decay, reference_terms())).unwrap());
        let mut plain = EwmMoments::new(cfg(decay, reference_terms())).unwrap();
        for &(x, w, d) in &rows {
            m.update(x, decay.factor(d), w);
            plain.update(x, decay.factor(d), w);
            assert_eq!(
                rmp_serde::to_vec(&m).unwrap(),
                rmp_serde::to_vec(&plain).unwrap()
            );
        }
        assert!(
            m.sums[0]
                .c
                .iter()
                .chain(&m.sums[0].r)
                .all(|v| v.is_finite())
        );
    }
}

/// A row of weight 0 inside the stream decays the weight and moves nothing
/// else, whatever its value.
#[test]
fn a_zero_weight_row_moves_no_moment() {
    let decay = Decay::Lam(0.9);
    let mut m = EwmMoments::new(cfg(decay, reference_terms())).unwrap();
    for &(x, w, d) in &stream(554, &[], false)[..10] {
        m.update(x, decay.factor(d), w);
    }
    for x in [f64::NAN, f64::INFINITY, 1e300, m.mean()] {
        let mut z = m.clone();
        z.update(x, 0.9, 0.0);
        assert_eq!(z.n_eff(), 0.9 * m.n_eff());
        let (a, b) = (&z.sums[0], &m.sums[0]);
        assert_eq!(
            (&a.m, &a.m_lo, &a.c, &a.r),
            (&b.m, &b.m_lo, &b.c, &b.r),
            "{x}"
        );
    }
}

/// A column held at one value has no spread: its standardized moments are
/// withheld with that reason, and its central moments and cumulants are 0,
/// with no NaN in the state. One row off the value gives the column a
/// spread, and they report.
#[test]
fn a_constant_column_withholds_its_standardized_moments() {
    use MomentKind::*;
    let terms = vec![
        term("skew", 3, Standardized, true),
        term("kurt", 4, Standardized, true),
        term("skew_u", 3, Standardized, false),
        term("c3", 3, Central, true),
        term("var", 2, Central, false),
        term("k4", 4, Cumulant, true),
    ];
    let decay = Decay::Halflife(10.0);
    let mut m = EwmMoments::new(cfg(decay, terms)).unwrap();
    for _ in 0..50 {
        m.update(1e6 + 0.1, decay.factor(1.0), 1.0);
    }
    let read = m.read();
    for v in &read[..3] {
        assert_eq!(*v, Err(Withheld::ZeroSpread));
    }
    assert_eq!(
        Withheld::ZeroSpread.reason(),
        "the spread of the column is zero"
    );
    assert_eq!(&read[3..], &[Ok(0.0), Ok(0.0), Ok(0.0)]);
    assert!(
        m.sums[0]
            .c
            .iter()
            .chain(&m.sums[0].r)
            .all(|v| v.is_finite())
    );
    m.update(1e6 + 0.2, decay.factor(1.0), 1.0);
    assert!(m.read().iter().all(Result::is_ok), "{:?}", m.read());
}

/// The unbiased forms are withheld where their denominators keep no digit,
/// read from the weights: order 2 on one row (`r_2 = 1`), order 3 on two
/// equal rows and order 4 on three, each reported one row later. At equal
/// weights and no decay they are Fisher's k-statistics (`scipy.stats.kstat`)
/// and `skew(bias=False)`, here against their closed forms.
#[test]
fn the_unbiased_forms_withhold_where_the_weights_identify_nothing() {
    use MomentKind::*;
    let terms = vec![
        term("k2", 2, Cumulant, false),
        term("k3", 3, Cumulant, false),
        term("k4", 4, Cumulant, false),
        term("g1", 3, Standardized, false),
    ];
    let mut m = EwmMoments::new(cfg(Decay::Lam(1.0), terms)).unwrap();
    let xs = [1.0, 2.5, 4.0, 2.0, 7.5, 3.25];
    let want_at = |n: usize| -> [Result<(), Withheld>; 4] {
        let ok = Ok(());
        // One row has no spread, which a standardized term reads first.
        match n {
            1 => [
                Err(Withheld::OneRow),
                Err(Withheld::TooFewRowsThird),
                Err(Withheld::TooFewRowsFourth),
                Err(Withheld::ZeroSpread),
            ],
            2 => [
                ok,
                Err(Withheld::TooFewRowsThird),
                Err(Withheld::TooFewRowsFourth),
                Err(Withheld::TooFewRowsThird),
            ],
            3 => [ok, ok, Err(Withheld::TooFewRowsFourth), ok],
            _ => [ok, ok, ok, ok],
        }
    };
    for (n, &x) in xs.iter().enumerate().map(|(i, x)| (i + 1, x)) {
        m.update(x, 1.0, 1.0);
        let got: Vec<Result<(), Withheld>> = m.read().iter().map(|v| v.map(drop)).collect();
        assert_eq!(got, want_at(n), "{n} rows");
        if n >= 4 {
            // Fisher's k-statistics from the power sums about the mean.
            let xs = &xs[..n];
            let nf = n as f64;
            let mean = xs.iter().sum::<f64>() / nf;
            let s = |p: i32| xs.iter().map(|x| (x - mean).powi(p)).sum::<f64>();
            let (m2, m3, m4) = (s(2) / nf, s(3) / nf, s(4) / nf);
            let k2 = nf * m2 / (nf - 1.0);
            let k3 = nf * nf * m3 / ((nf - 1.0) * (nf - 2.0));
            let k4 = nf * nf * ((nf + 1.0) * m4 - 3.0 * (nf - 1.0) * m2 * m2)
                / ((nf - 1.0) * (nf - 2.0) * (nf - 3.0));
            let g1 = (nf * (nf - 1.0)).sqrt() / (nf - 2.0) * m3 / m2.powf(1.5);
            let r = m.read();
            for (got, want) in r.iter().zip([k2, k3, k4, g1]) {
                assert!(
                    gap(got.unwrap(), want) < 1e-13,
                    "{n}: {got:?} against {want}"
                );
            }
        }
    }
}

/// `min_weight` withholds below it and reports at it.
#[test]
fn min_weight_withholds_below_and_reports_at_it() {
    let mut c = cfg(
        Decay::Lam(1.0),
        vec![term("c2", 2, MomentKind::Central, true)],
    );
    c.min_weight = 2.0;
    let mut m = EwmMoments::new(c).unwrap();
    m.update(1.0, 1.0, 1.0);
    m.update(2.0, 1.0, 0.999);
    assert_eq!(m.read(), [Err(Withheld::BelowMinWeight)]);
    m.update(2.0, 1.0, 0.001);
    assert!(m.n_eff() >= 2.0 && m.read()[0].is_ok(), "{}", m.n_eff());
}

/// A moment past a double's range is withheld as such, and a lower order
/// of the same column still reports.
#[test]
fn an_overflowed_moment_is_withheld() {
    use MomentKind::*;
    let terms = vec![term("c2", 2, Central, true), term("c4", 4, Central, true)];
    let mut m = EwmMoments::new(cfg(Decay::Lam(1.0), terms)).unwrap();
    for x in [1e100, -1e100, 1e100] {
        m.update(x, 1.0, 1.0);
    }
    let r = m.read();
    assert!(r[0].is_ok(), "{r:?}");
    assert_eq!(r[1], Err(Withheld::Overflow));
}

/// The numbers a state's encoding holds outside its settings, each a
/// double's worth: a double or an integer one, a temporal stamp's 128-bit
/// integer two.
pub(super) fn saved_numbers(m: &EwmMoments) -> u64 {
    fn count(v: &rmpv::Value) -> u64 {
        match v {
            rmpv::Value::Array(a) => a.iter().map(count).sum(),
            rmpv::Value::Map(m) => m.iter().map(|(_, v)| count(v)).sum(),
            rmpv::Value::Binary(b) => b.len() as u64 / 8,
            rmpv::Value::F64(_) | rmpv::Value::F32(_) | rmpv::Value::Integer(_) => 1,
            _ => 0,
        }
    }
    let bytes = rmp_serde::to_vec_named(m).unwrap();
    let v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
    v.as_map()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != Some("cfg"))
        .map(|(_, v)| count(v))
        .sum()
}

/// The state's bytes are `8 (1 + 3 + (P − 1) + n_r)` for one column -- the
/// weight, and a summary of `W`, the mean's pair, the moments of orders 2
/// to `P` and the `r_k` -- and its encoding holds exactly that many
/// numbers beside the settings. Orders 1 to 20, with each count of `r_k`.
#[test]
fn the_state_is_its_formula_in_bytes() {
    use MomentKind::*;
    for top in 1..=20u32 {
        for unbiased in [0, 2, 3, 4] {
            let mut terms = vec![term("top", top, Central, true)];
            if unbiased > 0 {
                terms.push(term("u", unbiased, Central, false));
            }
            let mut m = EwmMoments::new(cfg(Decay::Halflife(9.0), terms)).unwrap();
            for &(x, w, d) in &stream(1, &[], true)[..5] {
                m.update(x, Decay::Halflife(9.0).factor(d), w);
            }
            let p = top.max(unbiased);
            let n_r = match unbiased {
                3 => 2,
                4 => 3,
                _ => 1,
            };
            let doubles = 4 + (p as u64 - 1) + n_r;
            assert_eq!(
                m.state_bytes(),
                8 * doubles,
                "order {top}, unbiased {unbiased}"
            );
            assert_eq!(
                8 * saved_numbers(&m),
                m.state_bytes(),
                "order {top}, unbiased {unbiased}"
            );
        }
    }
    // Orders 2 to 4, plug-in: 48 to 64 bytes; order 6, 80.
    for (p, bytes) in [(2, 48), (3, 56), (4, 64), (6, 80)] {
        let m = EwmMoments::new(cfg(Decay::Lam(0.9), vec![term("c", p, Central, true)])).unwrap();
        assert_eq!(m.state_bytes(), bytes, "order {p}");
    }
}

/// The state is checked against `max_state_bytes` before the first row:
/// a state of exactly the limit builds, one double past it is refused,
/// naming the bytes and the limit. There is no other cap on the order.
#[test]
fn the_state_is_held_to_max_state_bytes() {
    let at = |p: u32, cap: u64| {
        let mut c = cfg(
            Decay::Lam(0.9),
            vec![term("c", p, MomentKind::Central, true)],
        );
        c.max_state_bytes = Some(cap);
        EwmMoments::new(c)
    };
    // Order 40: 4 + 39 + 1 doubles, 352 bytes.
    assert!(at(40, 352).is_ok());
    let err = at(41, 352).unwrap_err();
    assert!(
        err.contains("360 bytes") && err.contains("max_state_bytes (352)"),
        "{err}"
    );
    // The default, 256 KB: 32,764 orders fit and the next does not.
    let mut c = cfg(
        Decay::Lam(0.9),
        vec![term("c", 32_764, MomentKind::Central, true)],
    );
    assert_eq!(c.state_bytes(), 256 * 1024);
    assert!(c.validate().is_ok());
    c.terms[0].powers[0].1 += 1;
    assert!(
        c.validate()
            .unwrap_err()
            .contains("over max_state_bytes (262144)")
    );
}

/// Each refusal names its term: an unbiased form above order 4 (and order
/// 4 builds), a standardized moment of order 1 (order 2 builds, and reads
/// 1), an order of 0, a name given twice.
#[test]
fn the_refusals_name_the_term() {
    use MomentKind::*;
    let build = |terms| EwmMoments::new(cfg(Decay::Lam(0.9), terms));
    let err = build(vec![term("k5", 5, Cumulant, false)]).unwrap_err();
    assert!(
        err.contains("\"k5\"") && err.contains("bias=False at order 5"),
        "{err}"
    );
    assert!(build(vec![term("k4", 4, Cumulant, false)]).is_ok());
    // A raw moment is unbiased whatever `bias` says, so it is never refused.
    assert!(build(vec![term("r9", 9, Raw, false)]).is_ok());
    let err = build(vec![term("s1", 1, Standardized, true)]).unwrap_err();
    assert!(err.contains("\"s1\"") && err.contains("order 1"), "{err}");
    let mut m = build(vec![term("s2", 2, Standardized, true)]).unwrap();
    m.update(1.0, 1.0, 1.0);
    m.update(2.0, 1.0, 1.0);
    assert_eq!(m.read(), [Ok(1.0)]);
    let err = build(vec![term("z", 0, Central, true)]).unwrap_err();
    assert!(err.contains("\"z\" has order 0"), "{err}");
    let err = build(vec![
        term("a", 2, Central, true),
        term("a", 3, Central, true),
    ])
    .unwrap_err();
    assert!(err.contains("\"a\" is named twice"), "{err}");
    assert!(build(vec![]).unwrap_err().contains("names no term"));
}

/// Each statistic's default `bias` is Polars': a variance unbiased, every
/// other order and kind the plug-in.
#[test]
fn the_default_bias_is_polars() {
    use MomentKind::*;
    assert!(!MomentTerm::default_bias(Central, 2));
    assert!(!MomentTerm::default_bias(Cumulant, 2));
    for (kind, p) in [
        (Central, 3),
        (Central, 4),
        (Standardized, 3),
        (Standardized, 4),
        (Raw, 2),
        (Cumulant, 4),
    ] {
        assert!(MomentTerm::default_bias(kind, p), "{kind:?} {p}");
    }
}

/// The cumulants' polynomials, against the closed forms to order 6.
#[test]
fn the_cumulant_polynomials_are_the_closed_forms() {
    let p = |n| cumulant_poly(n);
    assert_eq!(p(2), [(vec![2], 1.0)]);
    assert_eq!(p(4), [(vec![4], 1.0), (vec![2, 2], -3.0)]);
    assert_eq!(p(5), [(vec![5], 1.0), (vec![2, 3], -10.0)]);
    let six = p(6);
    for (mono, coef) in [
        (vec![6], 1.0),
        (vec![2, 4], -15.0),
        (vec![3, 3], -10.0),
        (vec![2, 2, 2], 30.0),
    ] {
        assert!(
            six.contains(&(mono.clone(), coef)),
            "{mono:?} {coef}: {six:?}"
        );
    }
    assert_eq!(six.len(), 4);
}
