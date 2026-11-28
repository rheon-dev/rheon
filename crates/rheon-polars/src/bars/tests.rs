//! The stream operator over frames: chunk invariance, the limited feed, a
//! save and load to the bit, the cuts, and the bytes a group holds against
//! its serialized size. The crossing rule itself is `rheon_core::bars`'s,
//! tested there against its definitions.

use super::*;

fn cfg(kind: BarKind, quantity: Option<&str>, theta: f64) -> BarsConfig {
    BarsConfig {
        kind,
        clock: "t".into(),
        threshold: Threshold::Value(theta),
        quantity: quantity.map(str::to_string),
        group: Some("g".into()),
        weight: None,
        session: None,
        gap_cap: None,
        max_span: None,
        restart_after_step_back: None,
        on_step_back: None,
        keep_partial: true,
        monotone: false,
        max_state_bytes: 256 * 1024,
        max_total_bytes: 1 << 30,
        aggregates: vec![
            AggSpec {
                kind: AggKind::First,
                column: Some("x".into()),
                name: "open".into(),
            },
            AggSpec {
                kind: AggKind::Max,
                column: Some("x".into()),
                name: "high".into(),
            },
            AggSpec {
                kind: AggKind::Sum,
                column: Some("x".into()),
                name: "sx".into(),
            },
            AggSpec {
                kind: AggKind::Sum,
                column: Some("n".into()),
                name: "sn".into(),
            },
            AggSpec {
                kind: AggKind::Len,
                column: None,
                name: "rows".into(),
            },
        ],
        call: Default::default(),
    }
}

/// A seeded frame: values over a wide range of exponents, so the exact
/// sums hold several parts.
fn frame(n: usize, seed: u64) -> DataFrame {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 11
    };
    let mut t = Vec::new();
    let mut g = Vec::new();
    let mut x = Vec::new();
    let mut q = Vec::new();
    let mut nn = Vec::new();
    for i in 0..n {
        t.push(i as i64);
        g.push(["a", "b", "c"][(next() % 3) as usize]);
        let e = (next() % 60) as i32 - 30;
        x.push((next() % 1000) as f64 * 2f64.powi(e));
        q.push((next() % 1000) as f64 / 7.0);
        nn.push((next() % 100) as i64 - 50);
    }
    df!("t" => t, "g" => g, "x" => x, "q" => q, "n" => nn).unwrap()
}

fn run_chunks(cfg: &BarsConfig, df: &DataFrame, chunk: usize) -> (DataFrame, Vec<u8>) {
    let mut b = Bars::new(cfg.clone(), df.schema()).unwrap();
    let mut outs = b.feed(&df.slice(0, 0), None).unwrap();
    let mut at = 0;
    while at < df.height() {
        let part = df.slice(at as i64, chunk);
        outs.vstack_mut(&b.feed(&part, None).unwrap()).unwrap();
        at += chunk;
    }
    (outs, b.save_bytes().unwrap())
}

#[test]
fn any_chunking_gives_the_same_bars_and_state() {
    let df = frame(400, 7);
    for (kind, q) in [
        (BarKind::Count, None),
        (BarKind::Sum, Some("q")),
        (BarKind::Imbalance, Some("n")),
    ] {
        let theta = match kind {
            BarKind::Count => 9.0,
            BarKind::Sum => 300.0,
            BarKind::Imbalance => 60.0,
        };
        let c = cfg(kind, q, theta);
        let (one, state) = run_chunks(&c, &df, df.height());
        assert!(one.height() > 10);
        for chunk in [1, 2, 7, 64] {
            let (many, s) = run_chunks(&c, &df, chunk);
            assert!(one.equals_missing(&many), "{kind:?} chunk {chunk}");
            assert_eq!(state, s);
        }
    }
}

#[test]
fn a_saved_state_goes_on_to_the_bit() {
    let df = frame(300, 3);
    let c = cfg(BarKind::Sum, Some("q"), 250.0);
    let (whole, _) = run_chunks(&c, &df, df.height());
    for cut in [0, 1, 50, 151, 299, 300] {
        let mut b = Bars::new(c.clone(), df.schema()).unwrap();
        let mut out = b.feed(&df.slice(0, cut), None).unwrap();
        let bytes = b.save_bytes().unwrap();
        let mut r = Bars::load_bytes(&bytes, c.clone(), df.schema()).unwrap();
        assert_eq!(r.save_bytes().unwrap(), bytes, "re-saved byte for byte");
        out.vstack_mut(&r.feed(&df.slice(cut as i64, df.height()), None).unwrap())
            .unwrap();
        assert!(out.equals_missing(&whole), "cut at {cut}");
    }
    // Another call is refused, naming what differs.
    let bytes = run_chunks(&c, &df, 50).1;
    let mut other = c.clone();
    other.kind = BarKind::Imbalance;
    other.quantity = Some("n".into());
    let e = Bars::load_bytes(&bytes, other, df.schema()).unwrap_err();
    assert!(e.contains("kind"), "{e}");
    let mut other = c.clone();
    other.gap_cap = Some(3.0);
    let e = Bars::load_bytes(&bytes, other, df.schema()).unwrap_err();
    assert!(e.contains("gap_cap"), "{e}");
    // A damaged state is refused.
    let mut bad = bytes.clone();
    let mid = bad.len() / 2;
    bad[mid] ^= 0x40;
    assert!(Bars::load_bytes(&bad, c, df.schema()).is_err());
}

#[test]
fn a_limited_feed_stops_at_the_row_of_its_last_bar() {
    let df = frame(300, 5);
    let c = cfg(BarKind::Count, None, 4.0);
    let (whole, _) = run_chunks(&c, &df, df.height());
    for n in [1, 5, 17] {
        let mut b = Bars::new(c.clone(), df.schema()).unwrap();
        let head = b.feed(&df, Some(n)).unwrap();
        assert_eq!(head.height(), n);
        assert!(head.equals_missing(&whole.head(Some(n))));
        // The rows read are those up to the n-th bar's close.
        let close = head
            .column("time_close")
            .unwrap()
            .i64()
            .unwrap()
            .get(n - 1)
            .unwrap();
        assert_eq!(b.rows_fed, close as usize + 1);
        let rest = b.feed(&df.slice(close + 1, df.height()), None).unwrap();
        let mut all = head.clone();
        all.vstack_mut(&rest).unwrap();
        assert!(all.equals_missing(&whole));
    }
}

#[test]
fn a_cut_done_twice_does_nothing_the_second_time() {
    let df = frame(10, 1);
    let c = cfg(BarKind::Count, None, 100.0);
    let mut b = Bars::new(c, df.schema()).unwrap();
    b.feed(&df, None).unwrap();
    let key = b.groups.keys().next().unwrap().clone();
    let mut g = b.groups.remove(&key).unwrap();
    assert!(g.rows > 0);
    assert!(b.cut(&key, &mut g).is_some());
    let after = g.clone();
    assert!(b.cut(&key, &mut g).is_none());
    assert_eq!(g, after);
}

#[test]
fn the_byte_formula_bounds_the_saved_size() {
    let df = frame(600, 11);
    let c = cfg(BarKind::Sum, Some("q"), 1e9);
    let mut b = Bars::new(c, df.schema()).unwrap();
    b.feed(&df, None).unwrap();
    for (key, g) in &b.groups {
        // The sum of `x` holds several parts: values over 60 binades.
        let parts = g
            .aggs
            .iter()
            .map(|a| match a {
                Acc::SumFloat { exact, .. } => exact.parts().len(),
                _ => 0,
            })
            .max()
            .unwrap();
        assert!(parts >= 2, "{parts}");
        let saved = rmp_serde::to_vec_named(&(key, g)).unwrap().len() as u64;
        let bound = b.group_bytes(key.as_str().unwrap().len());
        assert!(saved <= bound, "saved {saved} > bound {bound}");
        // And the bound is not loose by more than the parts it allows.
        assert!(bound < saved + 8 * (2 * MAX_PARTS + RUNNING_PARTS) as u64 + 1024);
    }
    // The call's worst case, against the cap.
    let mut small = cfg(BarKind::Sum, Some("q"), 1.0);
    small.max_state_bytes = 100;
    assert!(
        Bars::new(small, df.schema())
            .unwrap_err()
            .contains("max_state_bytes")
    );
}

#[test]
fn the_total_cap_refuses_a_group_and_warns_at_half() {
    let df = frame(30, 2);
    let mut c = cfg(BarKind::Count, None, 100.0);
    let one = Bars::new(c.clone(), df.schema()).unwrap().group_bytes(1);
    c.max_total_bytes = 2 * one + one / 2;
    let mut b = Bars::new(c.clone(), df.schema()).unwrap();
    let e = b.feed(&df, None).unwrap_err().to_string();
    assert!(e.contains("max_total_bytes"), "{e}");
    assert_eq!(b.take_notices().len(), 1);
    c.max_total_bytes = 3 * one;
    let mut b = Bars::new(c, df.schema()).unwrap();
    b.feed(&df, None).unwrap();
}

#[test]
fn span_past_is_exact_on_each_clock() {
    use Instant::*;
    assert!(!span_past(Int(10), Int(8), 2.0));
    assert!(span_past(Int(11), Int(8), 2.0));
    assert!(span_past(Nanos(i64::MAX), Nanos(i64::MAX - 3), 2.5));
    assert!(!span_past(Nanos(i64::MAX), Nanos(i64::MAX - 2), 2.0));
    // The exact step from the double 0.1 to the double 0.3 is just below
    // the double 0.2, where a double subtraction gives 0.19999999999999998.
    assert!(!span_past(Number(0.3), Number(0.1), 0.2));
    assert!(span_past(Number(0.3), Number(0.1), 0.19999999999999996));
    assert!(span_past(Int(3), Number(0.5), 2.4));
    assert!(!span_past(Int(3), Number(0.5), 2.5));
}

#[test]
fn finish_emits_each_open_bar_and_leaves_none() {
    let df = frame(20, 4);
    let c = cfg(BarKind::Count, None, 100.0);
    let mut b = Bars::new(c, df.schema()).unwrap();
    assert_eq!(b.feed(&df, None).unwrap().height(), 0);
    let out = b.finish().unwrap();
    assert_eq!(out.height(), b.groups.len());
    let rows: i64 = out
        .column("rows")
        .unwrap()
        .cast(&DataType::Int64)
        .unwrap()
        .i64()
        .unwrap()
        .sum()
        .unwrap();
    assert_eq!(rows, 20);
    assert!(b.groups.values().all(|g| g.rows == 0));
    assert_eq!(b.finish().unwrap().height(), 0);
}

/// STATE-3: `on_step_back` mismatch and malformed `skipped_step_back`
/// tables in a bars state file are refused cleanly.
#[test]
fn damaged_step_back_state_and_policy_mismatch_are_refused() {
    let df = df!(
        "t" => [1i64, 5, 3, 6],
        "g" => ["a", "a", "a", "a"],
        "x" => [1.0, 2.0, 9.0, 3.0],
        "q" => [1.0, 1.0, 1.0, 1.0],
        "n" => [1i64, 1, 1, 1],
    )
    .unwrap();
    let mut skip_cfg = cfg(BarKind::Count, None, 10.0);
    skip_cfg.on_step_back = Some("skip".into());
    let mut b = Bars::new(skip_cfg.clone(), df.schema()).unwrap();
    let _ = b.feed(&df, None).unwrap();
    assert_eq!(b.total_skipped_step_back(), 1);
    let bytes = b.save_bytes().unwrap();

    // Policy mismatch on load.
    let err_cfg = cfg(BarKind::Count, None, 10.0);
    let e = Bars::load_bytes(&bytes, err_cfg.clone(), df.schema()).unwrap_err();
    assert!(e.contains("on_step_back"), "{e}");

    let reseal = |mut f: BarsFile, load_cfg: BarsConfig, schema: &Schema| -> String {
        f.checksum = Some(crate::checksum::Trailer::Patched);
        let mut raw = rmp_serde::to_vec_named(&f).unwrap();
        crate::checksum::seal(&mut raw);
        Bars::load_bytes(&raw, load_cfg, schema)
            .err()
            .unwrap_or_default()
    };

    // 1) Zero count.
    let mut f: BarsFile = rmp_serde::from_slice(&bytes).unwrap();
    f.skipped_step_back = vec![(GroupKey(Some("a".into())), 0)];
    assert!(reseal(f, skip_cfg.clone(), df.schema()).contains("the state is damaged"));

    // 2) Duplicate group key.
    let mut f: BarsFile = rmp_serde::from_slice(&bytes).unwrap();
    f.skipped_step_back = vec![
        (GroupKey(Some("a".into())), 1),
        (GroupKey(Some("a".into())), 2),
    ];
    assert!(reseal(f, skip_cfg.clone(), df.schema()).contains("the state is damaged"));

    // 3) Non-empty skipped_step_back under on_step_back = "error".
    let mut f: BarsFile = rmp_serde::from_slice(&bytes).unwrap();
    f.config.on_step_back = None;
    assert!(reseal(f, err_cfg, df.schema()).contains("the state is damaged"));

    // 4) Non-null group key in ungrouped bars.
    let mut u_cfg = cfg(BarKind::Count, None, 10.0);
    u_cfg.group = None;
    u_cfg.on_step_back = Some("skip".into());
    let mut bu = Bars::new(u_cfg.clone(), df.schema()).unwrap();
    let _ = bu.feed(&df, None).unwrap();
    let mut f: BarsFile = rmp_serde::from_slice(&bu.save_bytes().unwrap()).unwrap();
    f.skipped_step_back = vec![(GroupKey(Some("a".into())), 1)];
    assert!(reseal(f, u_cfg, df.schema()).contains("the state is damaged"));
}
