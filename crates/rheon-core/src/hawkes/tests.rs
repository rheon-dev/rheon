use super::*;
use crate::oracle;
use crate::seconds_of_ns;

fn lcg(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
}

/// A row: its clock step and one target value per node.
type Row = (f64, Vec<Option<f64>>);

/// A stream with every form a row takes: one event, events of several nodes
/// on one row, a count of 2, a fractional mark, rows with no event (a value
/// of 0, a negative one, none), and steps of 0.
fn stream(n: usize, rows: usize, seed: u64) -> Vec<Row> {
    let mut s = seed;
    (0..rows)
        .map(|i| {
            let d = if i == 0 || lcg(&mut s) < 0.05 {
                0.0
            } else {
                -lcg(&mut s).ln() * 0.4
            };
            let y = (0..n)
                .map(|_| {
                    let u = lcg(&mut s);
                    if u < 0.30 {
                        Some(1.0)
                    } else if u < 0.33 {
                        Some(2.0)
                    } else if u < 0.36 {
                        Some(0.5 + lcg(&mut s))
                    } else if u < 0.5 {
                        Some(0.0)
                    } else if u < 0.6 {
                        Some(-1.0)
                    } else {
                        None
                    }
                })
                .collect();
            (d, y)
        })
        .collect()
}

fn cfg(n: usize, betas: &[f64]) -> HawkesCfg {
    HawkesCfg {
        n_targets: n,
        kernel_decays: betas.to_vec(),
        decay: Decay::Halflife(f64::INFINITY),
        min_weight: 0.0,
        window_events: None,
        window_clock: None,
        closed: WindowClosed::Right,
        max_state_bytes: None,
    }
}

/// The clock and marks of each row, as the longhand reads them.
fn timeline(rows: &[Row]) -> Vec<(f64, Vec<f64>)> {
    let mut t = 0.0;
    rows.iter()
        .map(|(d, y)| {
            t += d;
            (t, y.iter().map(|v| Hawkes::mark(*v)).collect())
        })
        .collect()
}

/// `max |a − b| / max |b|`.
fn rel(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b
        .iter()
        .fold(0.0f64, |m, v| m.max(v.abs()))
        .max(f64::MIN_POSITIVE);
    a.iter()
        .zip(b)
        .fold(0.0f64, |m, (x, y)| m.max((x - y).abs()))
        / scale
}

/// The longhand's `θ`, one row of `p` per node.
fn oracle_theta(g: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let p = b.len() / n;
    // `b` transposed: `p` rows of one column per node.
    let bt: Vec<f64> = (0..p * n).map(|ic| b[(ic % n) * p + ic / n]).collect();
    let x = oracle::solve_columns(g, &bt, n);
    (0..n * p).map(|ip| x[(ip % p) * n + ip / p]).collect()
}

/// The model's `G` and `b`, read from its linear terms, against the
/// definition's double sums; and its fit against the longhand's solve. This
/// is the identity the state rests on: `G`'s event part is a function of `b`
/// and `q`.
#[test]
fn the_system_is_the_definitions_double_sums() {
    for (n, betas, gamma) in [
        (1, vec![1.0, 5.0], 0.0),
        (1, vec![0.5, 3.0, 20.0], 0.0),
        (2, vec![1.0, 8.0], 0.0),
        (3, vec![0.7, 4.0], 0.0),
        (2, vec![0.7, 4.0, 15.0], 0.0),
        (1, vec![1.0, 5.0], 0.01),
        (2, vec![0.7, 4.0, 15.0], 0.05),
        (2, vec![1.0, 8.0], 1.5),
        (1, vec![1.0, 5.0], 12.0),
    ] {
        let rows = stream(n, 400, 7 + n as u64);
        let mut c = cfg(n, &betas);
        if gamma > 0.0 {
            c.decay = Decay::Halflife(std::f64::consts::LN_2 / gamma);
        }
        let mut m = Hawkes::new(c).unwrap();
        for (d, y) in &rows {
            m.step(&[], y, *d, 1.0);
        }
        let line = timeline(&rows);
        let end = line.last().unwrap().0;
        let (g_want, b_want) = oracle::hawkes_system(&line, &betas, gamma, 0.0, end);
        let (g, b) = m.system();
        let what = format!("{n} nodes, decays {betas:?}, gamma {gamma}");
        assert!(rel(&g, &g_want) < 1e-12, "{what}: G {:e}", rel(&g, &g_want));
        assert!(rel(&b, &b_want) < 1e-12, "{what}: b {:e}", rel(&b, &b_want));
        eprintln!(
            "{what}: G {:.1e}, b {:.1e}",
            rel(&g, &g_want),
            rel(&b, &b_want)
        );
        let theta: Vec<f64> = m.coefficients().concat();
        let want = oracle_theta(&g_want, &b_want, n);
        assert!(
            rel(&theta, &want) < 1e-10,
            "{what}: theta {:e}",
            rel(&theta, &want)
        );
        eprintln!("  theta {:.1e}", rel(&theta, &want));
    }
}

/// Each row reports every node's intensity before the row's events, from
/// the fit on the rows before it alone (hard rule 2): the longhand's `θ` at
/// the row's clock times its `φ(t−)` from the earlier rows' events.
#[test]
fn each_row_reports_the_intensity_before_its_events_out_of_sample() {
    let (n, betas) = (2, vec![1.0, 6.0]);
    let rows = stream(n, 160, 31);
    let mut m = Hawkes::new(cfg(n, &betas)).unwrap();
    let line = timeline(&rows);
    let mut reported = 0;
    for (i, (d, y)) in rows.iter().enumerate() {
        let got = m.step(&[], y, *d, 1.0).pred;
        if i < 40 || i % 7 != 0 {
            continue;
        }
        let now = line[i].0;
        let before = &line[..i];
        let (g, b) = oracle::hawkes_system(before, &betas, 0.0, 0.0, now);
        let theta = oracle_theta(&g, &b, n);
        let p = 1 + n * betas.len();
        let phi: Vec<f64> = (0..n * betas.len())
            .map(|a| {
                let (j, beta) = (a / betas.len(), betas[a % betas.len()]);
                before
                    .iter()
                    .map(|(t, m)| beta * m[j] * (-beta * (now - t)).exp())
                    .sum()
            })
            .collect();
        for node in 0..n {
            let th = &theta[node * p..(node + 1) * p];
            let want = th[0] + th[1..].iter().zip(&phi).map(|(a, f)| a * f).sum::<f64>();
            assert!(
                (got[node] - want).abs() <= 1e-10 * (1.0 + want.abs()),
                "row {i}, node {node}: {} vs {want}",
                got[node]
            );
        }
        reported += 1;
    }
    assert!(reported > 10);
}

/// The window's fit is the longhand's over the last `W` rows with an
/// event, observed from the clock of the row dropped last, at every row of
/// a stream many windows long: across the older block running out, its
/// rebuild, and rows of several nodes at once.
#[test]
fn the_window_is_the_definitions_fit_on_its_rows() {
    for (n, betas, cap) in [
        (1, vec![1.0, 5.0], 1usize),
        (1, vec![1.0, 5.0], 7),
        (2, vec![0.7, 4.0, 15.0], 25),
        (3, vec![2.0], 12),
    ] {
        let rows = stream(n, 360, 101 + cap as u64);
        let mut c = cfg(n, &betas);
        c.window_events = Some(cap);
        let mut m = Hawkes::new(c).unwrap();
        let line = timeline(&rows);
        let mut kept: VecDeque<usize> = VecDeque::new();
        let mut start = 0.0;
        let mut worst: f64 = 0.0;
        for (i, (d, y)) in rows.iter().enumerate() {
            m.step(&[], y, *d, 1.0);
            if line[i].1.iter().any(|v| *v > 0.0) {
                kept.push_back(i);
                if kept.len() > cap {
                    start = line[kept.pop_front().unwrap()].0;
                }
            }
            assert_eq!(m.window_rows(), kept.len(), "row {i}");
            if i % 3 != 0 {
                continue;
            }
            let window: Vec<(f64, Vec<f64>)> = kept.iter().map(|&r| line[r].clone()).collect();
            // Nothing observed yet: `G` is rounding residue against zeros.
            if window.is_empty() || line[i].0 == start {
                continue;
            }
            let (g_want, b_want) = oracle::hawkes_system(&window, &betas, 0.0, start, line[i].0);
            let (g, b) = m.system();
            let err = rel(&g, &g_want).max(rel(&b, &b_want));
            assert!(err < 1e-12, "{n} nodes, window {cap}, row {i}: G/b {err:e}");
            worst = worst.max(err);
        }
        eprintln!("window {cap}, {n} nodes: worst G/b {worst:.2e}");
    }
}

/// A row of weight 0 learns none of its events: the state is the one a row
/// with no event leaves, to the bit, and the weight is the row's to the
/// contract.
#[test]
fn a_row_of_weight_zero_learns_none_of_its_events() {
    let rows = stream(2, 120, 5);
    for window in [None, Some(9)] {
        let mut c = cfg(2, &[1.0, 4.0]);
        c.window_events = window;
        let mut a = Hawkes::new(c.clone()).unwrap();
        let mut b = Hawkes::new(c).unwrap();
        for (i, (d, y)) in rows.iter().enumerate() {
            if i % 5 == 3 {
                a.step(&[], y, *d, 0.0);
                b.step(&[], &[None, None], *d, 0.0);
            } else {
                a.step(&[], y, *d, 1.0);
                b.step(&[], y, *d, 1.0);
            }
        }
        assert_eq!(a, b, "window {window:?}");
    }
    // The first row at weight 0: nothing divides by it.
    let mut m = Hawkes::new(cfg(1, &[1.0])).unwrap();
    m.step(&[], &[Some(1.0)], 0.0, 0.0);
    let s = m.step(&[], &[Some(1.0)], 1.0, 1.0);
    assert_eq!(s.n_eff, 0.0);
    assert!(m.r.iter().chain(&m.b).chain(&m.q).all(|v| v.is_finite()));
}

/// A positive weight learns the row's events at their marks alone: the
/// weight is not a multiplier of an event, so scaling every weight moves
/// nothing but the weight.
#[test]
fn a_weight_scales_nothing_but_the_weight() {
    let rows = stream(2, 150, 9);
    let mut a = Hawkes::new(cfg(2, &[1.0, 4.0])).unwrap();
    let mut b = Hawkes::new(cfg(2, &[1.0, 4.0])).unwrap();
    for (d, y) in &rows {
        let sa = a.step(&[], y, *d, 1.0);
        let sb = b.step(&[], y, *d, 1e-3);
        assert_eq!(sa.pred.len(), sb.pred.len());
        for (x, z) in sa.pred.iter().zip(&sb.pred) {
            assert!(x == z || (x.is_nan() && z.is_nan()));
        }
    }
    assert_eq!(a.r, b.r);
    assert_eq!(a.b, b.b);
}

/// A save mid-window, in either block, goes on to the bit.
#[test]
fn a_saved_window_goes_on_to_the_bit() {
    let rows = stream(2, 300, 77);
    let mut c = cfg(2, &[1.0, 6.0]);
    c.window_events = Some(11);
    for cut in [0, 1, 5, 40, 41, 97, 150, 211] {
        let mut m = Hawkes::new(c.clone()).unwrap();
        for (d, y) in &rows[..cut] {
            m.step(&[], y, *d, 1.0);
        }
        let bytes = rmp_serde::to_vec(&m.state()).unwrap();
        let mut back = Hawkes::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
        for (d, y) in &rows[cut..] {
            let a = m.step(&[], y, *d, 1.0);
            let b = back.step(&[], y, *d, 1.0);
            assert_eq!(
                a.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "cut {cut}"
            );
        }
        assert_eq!(m, back, "cut {cut}");
    }
}

/// The state's size: `8 (k + D (1 + k) + D (D + 1) / 2 + 2)` bytes of numbers, whatever
/// the stream's length, and the window's at most `8 W (2 + D + k)` beside.
#[test]
fn the_state_is_bounded_by_its_shape_and_window() {
    for (n, betas) in [(1usize, vec![1.0, 5.0]), (2, vec![0.7, 4.0, 15.0])] {
        let k = n * betas.len();
        let numbers = k + n * (1 + k) + n * (n + 1) / 2 + 2;
        let mut m = Hawkes::new(cfg(n, &betas)).unwrap();
        let mut sizes = Vec::new();
        for (d, y) in stream(n, 4000, 3) {
            m.step(&[], &y, d, 1.0);
            sizes.push(rmp_serde::to_vec(&m.state()).unwrap().len());
        }
        let (first, last) = (sizes[100], sizes[3999]);
        assert_eq!(first, last, "{n} nodes: the state grew with the stream");
        // msgpack writes each f64 in 9 bytes; the cfg and the tags are the rest.
        eprintln!(
            "{n} nodes, {} decays: {numbers} numbers ({} bytes as f64), {last} bytes of msgpack",
            betas.len(),
            8 * numbers
        );
        assert!(last < 9 * numbers + 200, "{last}");
        let cap = 50;
        let mut c = cfg(n, &betas);
        c.window_events = Some(cap);
        let mut m = Hawkes::new(c).unwrap();
        let mut most = 0;
        for (d, y) in stream(n, 4000, 3) {
            m.step(&[], &y, d, 1.0);
            most = most.max(rmp_serde::to_vec(&m.state()).unwrap().len());
        }
        let window = cap * (2 + n + k);
        eprintln!(
            "  window {cap}: at most {} bytes of msgpack, bound {} numbers",
            most,
            numbers + window
        );
        assert!(most < 9 * (numbers + window) + 400, "{most}");
    }
}

/// Simulated from known parameters by Ogata's thinning, one node, the fit
/// recovers them as the events grow.
#[test]
fn the_fit_recovers_a_simulated_process() {
    let (mu, alpha, beta) = (0.5, 0.6, 2.0);
    let mut s = 21u64;
    let (mut t, mut excite) = (0.0f64, 0.0f64);
    let mut times = Vec::new();
    while t < 20_000.0 {
        let bound = mu + excite;
        let w = -lcg(&mut s).ln() / bound;
        excite *= (-beta * w).exp();
        t += w;
        if lcg(&mut s) * bound <= mu + excite {
            times.push(t);
            excite += alpha * beta;
        }
    }
    let mut m = Hawkes::new(cfg(1, &[beta])).unwrap();
    let mut last = 0.0;
    for &ti in &times {
        m.step(&[], &[Some(1.0)], ti - last, 1.0);
        last = ti;
    }
    let theta = &m.coefficients()[0];
    eprintln!(
        "{} events: mu {}, alpha {}",
        times.len(),
        theta[0],
        theta[1]
    );
    assert!((theta[0] - mu).abs() < 0.03, "{theta:?}");
    assert!((theta[1] - alpha).abs() < 0.03, "{theta:?}");
}

#[test]
fn a_state_of_the_wrong_shape_is_refused() {
    let mut c = cfg(2, &[1.0, 4.0]);
    c.window_events = Some(5);
    let mut m = Hawkes::new(c).unwrap();
    for (d, y) in stream(2, 50, 1) {
        m.step(&[], &y, d, 1.0);
    }
    let edits: [fn(&mut Hawkes); 5] = [
        |h| {
            h.r.pop();
        },
        |h| {
            h.b.pop();
        },
        |h| {
            h.q.push(0.0);
        },
        |h| h.window = None,
        |h| {
            h.window.as_mut().unwrap().new_sum.pop();
        },
    ];
    for (i, edit) in edits.iter().enumerate() {
        let mut s = m.state();
        let ModelState::Hawkes(inner) = &mut s.model else {
            unreachable!()
        };
        edit(inner);
        match Hawkes::restore(&s) {
            Err(StateError::Invalid(e)) => assert!(e.contains("wrong shape"), "{i}: {e}"),
            other => panic!("{i}: {other:?}"),
        }
    }
    let mut s = m.state();
    let ModelState::Hawkes(inner) = &mut s.model else {
        unreachable!()
    };
    inner.cfg.kernel_decays = vec![1.0, 1.0];
    match Hawkes::restore(&s) {
        Err(StateError::Invalid(e)) => assert!(e.contains("hawkes refuses"), "{e}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_settings_new_refuses() {
    let refuse = |c: HawkesCfg, says: &str| {
        let e = Hawkes::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    refuse(cfg(0, &[1.0]), "at least one target");
    refuse(cfg(1, &[]), "at least one decay");
    refuse(cfg(1, &[1.0, -2.0]), "finite number above 0");
    refuse(cfg(1, &[1.0, f64::INFINITY]), "finite number above 0");
    refuse(cfg(1, &[3.0, 3.0]), "distinct");
    let near = |gamma: f64, betas: &[f64]| HawkesCfg {
        decay: Decay::Halflife(std::f64::consts::LN_2 / gamma),
        ..cfg(1, betas)
    };
    refuse(near(1.0005, &[1.0, 5.0]), "within 0.1% of 1");
    refuse(near(6.001, &[1.0, 5.0]), "within 0.1% of 1 + 5");
    assert!(Hawkes::new(near(1.01, &[1.0, 5.0])).is_ok());
    let mut c = cfg(1, &[1.0]);
    c.window_events = Some(0);
    refuse(c, "at least 1");
    let mut c = near(0.01, &[1.0]);
    c.window_events = Some(10);
    refuse(c, "takes no decay");
    let mut c = cfg(1, &[1.0]);
    c.min_weight = f64::NAN;
    refuse(c, "min_weight");
}

/// `predict` is the step's answer for the row, the state untouched; and
/// before every node has had an event the system is singular, so nothing
/// is reported rather than a fit of a jitter.
#[test]
fn predict_is_the_step_and_a_singular_system_reports_nothing() {
    let rows = stream(2, 200, 13);
    let mut m = Hawkes::new(cfg(2, &[1.0, 4.0])).unwrap();
    let first = m.step(&[], &[Some(1.0), None], 0.0, 1.0);
    assert!(first.pred.iter().all(|v| v.is_nan()));
    let only_one = m.step(&[], &[Some(1.0), None], 1.0, 1.0);
    assert!(
        only_one.pred.iter().all(|v| v.is_nan()),
        "{:?}",
        only_one.pred
    );
    for (d, y) in &rows {
        let before = m.clone();
        let p = m.predict(&[], *d);
        let s = m.step(&[], y, *d, 1.0);
        assert_eq!(
            p.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(p.n_eff, s.n_eff);
        let again = before.predict(&[], *d);
        assert_eq!(
            again.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            p.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(m.coefficients().iter().flatten().all(|v| v.is_finite()));
}

/// Two nodes whose events fall on the same rows, of nearly the same marks,
/// cannot be told apart: their `G` has a scaled condition number near
/// `1e14`, and its exact solution is digits of rounding (coefficients of
/// ±450,000 for intensities near 1,000). Such a fit is not reported; the
/// same nodes on rows of their own are.
#[test]
fn a_fit_the_events_cannot_identify_is_not_reported() {
    let mut c = cfg(2, &[1.0, 4.0]);
    c.decay = Decay::Halflife(20.0);
    let mut m = Hawkes::new(c.clone()).unwrap();
    let mut s = 5u64;
    for i in 0..17 {
        let v = 1000.0 + 3.0 * lcg(&mut s) - 1.5 * lcg(&mut s);
        let d = if i == 0 { 0.0 } else { 1.0 };
        m.step(&[], &[Some(v + 0.5), Some(v + 1.0)], d, 1.0);
    }
    let (g, _) = m.system();
    let p = 5;
    let scaled: Vec<f64> = (0..p * p)
        .map(|ij| g[ij] / (g[(ij / p) * (p + 1)] * g[(ij % p) * (p + 1)]).sqrt())
        .collect();
    let kappa = oracle::condition(&scaled);
    assert!(kappa > 1e13, "{kappa:e}");
    assert!(m.coefficients().concat().iter().all(|v| v.is_nan()));
    assert!(m.predict(&[], 1.0).pred.iter().all(|v| v.is_nan()));
    let mut m = Hawkes::new(c).unwrap();
    for (d, y) in stream(2, 200, 17) {
        m.step(&[], &y, d, 1.0);
    }
    assert!(m.coefficients().concat().iter().all(|v| v.is_finite()));
}

/// The numbers a state holds, cfg aside, driven far past a full window:
/// never more than `state_bytes` says, and exactly that once the window is
/// full. The window's rows are the one part of the state the data could
/// size, and `window_events` caps them.
#[test]
fn the_state_never_passes_the_bytes_its_settings_give() {
    fn floats(v: &rmpv::Value) -> u64 {
        match v {
            rmpv::Value::Array(a) => a.iter().map(floats).sum(),
            rmpv::Value::Map(m) => m
                .iter()
                .filter(|(k, _)| k.as_str() != Some("cfg"))
                .map(|(_, v)| floats(v))
                .sum(),
            rmpv::Value::F64(_) => 1,
            _ => 0,
        }
    }
    for (n, betas, window) in [
        (1usize, vec![1.0, 5.0], None),
        (2, vec![0.7, 4.0, 15.0], None),
        (1, vec![1.0, 5.0], Some(7usize)),
        (3, vec![2.0, 9.0], Some(40)),
    ] {
        let mut c = cfg(n, &betas);
        c.window_events = window;
        let bound = Hawkes::state_bytes(n, betas.len(), window);
        let mut m = Hawkes::new(c).unwrap();
        let mut most = 0;
        for (d, y) in stream(n, 3000, 41) {
            m.step(&[], &y, d, 1.0);
            let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
            let v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
            most = most.max(8 * floats(&v));
        }
        assert_eq!(most, bound, "{n} nodes, {betas:?}, window {window:?}");
    }
}

// ---- the window of clock units (PLAN 498/538) ----

/// A stream whose steps are multiples of a quarter of a unit, so that sums
/// of them are exact and a row is often exactly a span old: the same rows
/// as [`stream`], with the steps rounded.
fn grid_stream(n: usize, rows: usize, seed: u64) -> Vec<Row> {
    let mut s = seed ^ 0x9e37;
    stream(n, rows, seed)
        .into_iter()
        .enumerate()
        .map(|(i, (_, y))| {
            let d = if i == 0 {
                0.0
            } else {
                (lcg(&mut s) * 6.0).floor() * 0.25
            };
            (d, y)
        })
        .collect()
}

fn clock_cfg(n: usize, betas: &[f64], span: f64, closed: WindowClosed) -> HawkesCfg {
    HawkesCfg {
        window_clock: Some(span),
        closed,
        ..cfg(n, betas)
    }
}

/// The rows of `line` up to `i` that a window of `span` holds at `line[i]`'s
/// clock, from the definition: those with an event, less than `span` old
/// (or exactly that old, under `Both`).
fn in_clock_window(
    line: &[(f64, Vec<f64>)],
    i: usize,
    span: f64,
    closed: WindowClosed,
) -> Vec<(f64, Vec<f64>)> {
    line[..=i]
        .iter()
        .filter(|(t, m)| {
            let age = line[i].0 - t;
            m.iter().any(|v| *v > 0.0)
                && match closed {
                    WindowClosed::Right => age < span,
                    WindowClosed::Both => age <= span,
                }
        })
        .cloned()
        .collect()
}

/// The window of clock units is the definition's fit on the events of the
/// last `span`, observed over `(T − span, T]` -- from the first row until
/// there is that much clock -- at every row of a stream many spans long,
/// under either edge, over steps that put rows exactly a span old, across the
/// older block running out and rebuilding.
#[test]
fn the_clock_window_is_the_definitions_fit_on_the_events_in_it() {
    for (n, betas, span, closed) in [
        (1, vec![1.0, 5.0], 2.5, WindowClosed::Right),
        (1, vec![1.0, 5.0], 2.5, WindowClosed::Both),
        (2, vec![0.7, 4.0, 15.0], 6.0, WindowClosed::Right),
        (2, vec![0.7, 4.0, 15.0], 6.0, WindowClosed::Both),
        (3, vec![2.0], 0.75, WindowClosed::Right),
        (2, vec![1.0, 8.0], 40.0, WindowClosed::Right),
    ] {
        let rows = grid_stream(n, 400, 211 + (span * 4.0) as u64);
        let line = timeline(&rows);
        let mut m = Hawkes::new(clock_cfg(n, &betas, span, closed)).unwrap();
        let mut worst: f64 = 0.0;
        let mut at_edge = 0;
        for (i, (d, y)) in rows.iter().enumerate() {
            m.step(&[], y, *d, 1.0);
            let window = in_clock_window(&line, i, span, closed);
            assert_eq!(m.window_rows(), window.len(), "{span} {closed:?} row {i}");
            at_edge += usize::from(
                line[..i]
                    .iter()
                    .any(|(t, mk)| line[i].0 - t == span && mk.iter().any(|v| *v > 0.0)),
            );
            let start = (line[i].0 - span).max(line[0].0);
            if window.is_empty() || line[i].0 == start {
                continue;
            }
            let (g_want, b_want) = oracle::hawkes_system(&window, &betas, 0.0, start, line[i].0);
            let (g, b) = m.system();
            let err = rel(&g, &g_want).max(rel(&b, &b_want));
            assert!(
                err < 1e-12,
                "{n} nodes, {span} {closed:?}, row {i}: G/b {err:e}"
            );
            worst = worst.max(err);
        }
        if span < 10.0 {
            assert!(at_edge > 5, "{span}: the stream put few rows at the edge");
        }
        eprintln!("clock window {span} {closed:?}, {n} nodes: worst G/b {worst:.2e}");
    }
}

/// A row exactly a span old has left under `Right` and stays under `Both`,
/// the edge Polars' `rolling_*_by(closed=)` draws; and a row that leaves
/// does so at the first row at or past its age, before that row's
/// intensity is read.
#[test]
fn a_row_exactly_a_span_old_is_on_the_edge_closed_says() {
    for (closed, kept) in [(WindowClosed::Right, 3), (WindowClosed::Both, 4)] {
        let mut m = Hawkes::new(clock_cfg(1, &[1.0], 3.0, closed)).unwrap();
        // Events at the clocks 0, 1, 2 and 3.
        for d in [0.0, 1.0, 1.0, 1.0] {
            m.step(&[], &[Some(1.0)], d, 1.0);
        }
        assert_eq!(m.window_rows(), kept, "{closed:?}");
    }
    // Rows at 0, 0.5, 3.0 and 3.5: at 3.5 the first two are 3.5 and 3.0 old,
    // both out under `Right`.
    let mut m = Hawkes::new(clock_cfg(1, &[1.0], 3.0, WindowClosed::Right)).unwrap();
    for d in [0.0, 0.5, 2.5, 0.5] {
        m.step(&[], &[Some(1.0)], d, 1.0);
    }
    assert_eq!(m.window_rows(), 2);
}

/// The clock window decides its edge on the stamps it is handed, exact in
/// integer nanoseconds, where a clock summed in doubles drifts: steps of
/// 100 ms against a span of 700 ms. The window's members are those the
/// integer clock gives at every row.
#[test]
fn the_clock_window_reads_its_edge_off_the_stamps() {
    const MS: i128 = 1_000_000;
    let n = 2;
    let betas = [3.0, 12.0];
    let span_ns = 700 * MS;
    let span = seconds_of_ns(span_ns);
    let rows = stream(n, 500, 31);
    let mut ns = 0i128;
    let mut stamps = Vec::new();
    let mut s = 19u64;
    for i in 0..rows.len() {
        if i > 0 {
            ns += 100 * MS * (lcg(&mut s) * 4.0).floor() as i128;
        }
        stamps.push(ns);
    }
    let mut m = Hawkes::new(clock_cfg(n, &betas, span, WindowClosed::Right)).unwrap();
    let mut prev = 0i128;
    let mut flipped = 0;
    let mut own = 0.0;
    for (i, ((_, y), &at)) in rows.iter().zip(&stamps).enumerate() {
        let d = seconds_of_ns(at - prev);
        prev = at;
        own += d;
        m.stamp_next(Stamp::Ns(at));
        m.step(&[], y, d, 1.0);
        let has_event = |y: &[Option<f64>]| y.iter().any(|v| Hawkes::mark(*v) > 0.0);
        let want = rows[..=i]
            .iter()
            .zip(&stamps)
            .filter(|(row, t)| at - **t < span_ns && has_event(&row.1))
            .count();
        assert_eq!(m.window_rows(), want, "row {i}");
        // The summed clock, which the stamps are there to not read, puts a
        // row exactly 700 ms old on the other side of the edge.
        flipped += rows[..i]
            .iter()
            .zip(&stamps)
            .filter(|(row, t)| {
                at - **t == span_ns && has_event(&row.1) && own - seconds_of_ns(**t) < span
            })
            .count();
    }
    assert!(m.window_rows() > 0);
    assert!(
        flipped > 0,
        "the stream had no row the summed clock misreads"
    );
}

/// Under a window `n_eff` is the weight inside it (hard rule 8): the weights
/// of the event rows held as the last row left them, before the row's own
/// clock step, for either kind of window. A row with no event, or of
/// weight 0, is no part of the fit and adds none.
#[test]
fn under_a_window_n_eff_is_the_weight_inside_it() {
    let rows = grid_stream(2, 300, 59);
    let line = timeline(&rows);
    let weight = |i: usize| match i % 7 {
        0 => 0.0,
        1 => 2.5,
        2 => 0.5,
        _ => 1.0,
    };
    for kind in 0..3 {
        let c = match kind {
            0 => HawkesCfg {
                window_events: Some(9),
                ..cfg(2, &[1.0, 4.0])
            },
            1 => clock_cfg(2, &[1.0, 4.0], 3.0, WindowClosed::Right),
            _ => clock_cfg(2, &[1.0, 4.0], 3.0, WindowClosed::Both),
        };
        let mut m = Hawkes::new(c).unwrap();
        let mut held: VecDeque<(f64, f64)> = VecDeque::new();
        for (i, (d, y)) in rows.iter().enumerate() {
            // The weight inside the window as the last row left it, before
            // this row's own clock step (hard rule 8).
            let want: f64 = held.iter().map(|(_, w)| w).sum();
            assert!(
                (m.n_eff() - want).abs() < 1e-12,
                "kind {kind} row {i}: accessor"
            );
            let s = m.step(&[], y, *d, weight(i));
            assert!(
                (s.n_eff - want).abs() < 1e-12,
                "kind {kind} row {i}: {} against {want}",
                s.n_eff
            );
            // A row of weight 0 handed to the model ages it as any row
            // does: the stream hands it none (hard rule 9).
            match kind {
                0 => {}
                1 => held.retain(|(t, _)| line[i].0 - t < 3.0),
                _ => held.retain(|(t, _)| line[i].0 - t <= 3.0),
            }
            if weight(i) > 0.0 && line[i].1.iter().any(|v| *v > 0.0) {
                held.push_back((line[i].0, weight(i)));
                if kind == 0 && held.len() > 9 {
                    held.pop_front();
                }
            }
        }
    }
}

/// A row of weight 0 learns none of its events under a window of clock
/// units either, and the window holds none of the row: its state is that of
/// a row with no event, to the bit.
#[test]
fn a_zero_weight_row_is_no_part_of_a_clock_window() {
    let rows = grid_stream(2, 160, 5);
    let c = clock_cfg(2, &[1.0, 4.0], 4.0, WindowClosed::Right);
    let mut a = Hawkes::new(c.clone()).unwrap();
    let mut b = Hawkes::new(c.clone()).unwrap();
    for (i, (d, y)) in rows.iter().enumerate() {
        if i % 5 == 3 {
            a.step(&[], y, *d, 0.0);
            b.step(&[], &[None, None], *d, 0.0);
        } else {
            a.step(&[], y, *d, 1.0);
            b.step(&[], y, *d, 1.0);
        }
    }
    assert_eq!(a, b);
    // The first row at weight 0: nothing divides by it.
    let mut m = Hawkes::new(c).unwrap();
    m.step(&[], &[Some(1.0), Some(1.0)], 0.0, 0.0);
    let s = m.step(&[], &[Some(1.0), Some(1.0)], 1.0, 1.0);
    assert_eq!(s.n_eff, 0.0);
    assert!(m.r.iter().chain(&m.b).chain(&m.q).all(|v| v.is_finite()));
}

/// A save mid-window goes on to the bit, stamped or not, in either block,
/// and a read ahead from a restored state is the live one's.
#[test]
fn a_saved_clock_window_goes_on_to_the_bit() {
    const MS: i128 = 1_000_000;
    let rows = grid_stream(2, 300, 77);
    let c = clock_cfg(2, &[1.0, 6.0], 2.0, WindowClosed::Right);
    for stamped in [false, true] {
        let mut at = 0i128;
        let stamps: Vec<i128> = rows
            .iter()
            .map(|(d, _)| {
                at += (d * 1000.0).round() as i128 * MS;
                at
            })
            .collect();
        let feed = |m: &mut Hawkes, i: usize| {
            if stamped {
                m.stamp_next(Stamp::Ns(stamps[i]));
            }
            m.step(&[], &rows[i].1, rows[i].0, 1.0)
        };
        for cut in [0, 1, 5, 40, 41, 97, 150, 211] {
            let mut m = Hawkes::new(c.clone()).unwrap();
            for i in 0..cut {
                feed(&mut m, i);
            }
            let bytes = rmp_serde::to_vec(&m.state()).unwrap();
            let mut back = Hawkes::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
            assert_eq!(m, back, "stamped {stamped}, cut {cut}: the state");
            for (i, row) in rows.iter().enumerate().skip(cut) {
                let ahead = back.predict(&[], row.0);
                let a = feed(&mut m, i);
                let b = feed(&mut back, i);
                let bits = |s: &Step| s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&a), bits(&b), "stamped {stamped}, cut {cut}, row {i}");
                assert_eq!(bits(&a), bits(&ahead), "predict, cut {cut}, row {i}");
                assert_eq!(a.n_eff, ahead.n_eff);
            }
            assert_eq!(m, back, "stamped {stamped}, cut {cut}");
        }
    }
}

/// `predict` is the step's answer at the row, across the rows that leave: the
/// window is cut at the row's clock on a copy, the state untouched.
#[test]
fn a_read_ahead_is_the_step_across_the_rows_that_leave() {
    for closed in [WindowClosed::Right, WindowClosed::Both] {
        let rows = grid_stream(2, 250, 13);
        let mut m = Hawkes::new(clock_cfg(2, &[1.0, 4.0], 1.5, closed)).unwrap();
        let mut cut_ahead = 0;
        for (d, y) in &rows {
            let before = m.clone();
            let p = m.predict(&[], *d);
            let ahead = m.stamp_ahead(*d).unwrap();
            cut_ahead += usize::from(m.oldest_has_left(ahead));
            assert_eq!(before, m, "a read moved the state");
            let s = m.step(&[], y, *d, 1.0);
            assert_eq!(
                p.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                s.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
            assert_eq!(p.n_eff, s.n_eff);
        }
        assert!(cut_ahead > 20, "{closed:?}: few reads cut the window");
    }
}

/// A clock window's size is the data's, so it is counted as the rows go in:
/// the first row to pass half of the cap raises a warning, once, and the
/// state is never taken past the cap, the row that would being the one not
/// learned.
#[test]
fn a_burst_of_events_is_warned_of_at_half_the_cap_and_stopped_at_it() {
    assert_eq!(
        std::mem::size_of::<Stamp>(),
        32,
        "the byte counts, and the docs, say 32"
    );
    let (n, betas) = (2usize, vec![1.0, 4.0]);
    let one_row = Hawkes::clock_state_bytes(n, 2, 1) - Hawkes::clock_state_bytes(n, 2, 0);
    // 100 rows fit; a span that holds more than that is the burst.
    let cap = Hawkes::clock_state_bytes(n, 2, 100);
    let mut c = clock_cfg(n, &betas, 1e6, WindowClosed::Right);
    c.max_state_bytes = Some(cap);
    let step = |m: &mut Hawkes, i: usize| {
        m.step(
            &[],
            &[Some(1.0), Some(1.0)],
            if i == 0 { 0.0 } else { 0.01 },
            1.0,
        )
    };
    let mut m = Hawkes::new(c.clone()).unwrap();
    let mut warned_at = None;
    let mut over_at = None;
    let mut high = 0u64;
    for i in 0..300usize {
        step(&mut m, i);
        let held = Hawkes::clock_state_bytes(n, 2, m.window_rows());
        high = high.max(held);
        assert!(held <= cap, "row {i}: {held} bytes past the cap {cap}");
        if let Some((bytes, said_cap)) = m.take_state_warning() {
            assert!(warned_at.replace(i).is_none(), "warned twice");
            assert_eq!(said_cap, cap);
            assert_eq!(bytes, held);
            assert!(2 * bytes > cap && 2 * (bytes - one_row) <= cap, "{bytes}");
        }
        if let Some((bytes, said_cap)) = m.state_over_cap() {
            over_at.get_or_insert(i);
            assert_eq!(said_cap, cap);
            assert_eq!(bytes, cap + one_row, "the row that would have passed it");
        }
    }
    // The first row to hold more than half of the cap's bytes warns (the
    // sums are part of them); the 101st (index 100) is the first not learned.
    let half = (1..=100)
        .find(|&r| 2 * Hawkes::clock_state_bytes(n, 2, r) > cap)
        .unwrap();
    assert_eq!(warned_at, Some(half - 1));
    assert_eq!(over_at, Some(100));
    assert_eq!(m.window_rows(), 100);
    assert_eq!(high, cap);
    // A span that holds few rows stays under the cap and says nothing.
    let mut quiet = Hawkes::new(HawkesCfg {
        window_clock: Some(0.05),
        ..c.clone()
    })
    .unwrap();
    for i in 0..300usize {
        step(&mut quiet, i);
        assert!(quiet.take_state_warning().is_none());
        assert!(quiet.state_over_cap().is_none());
    }
    // The warning is once a group: a save and load after it goes on without
    // saying it again, and the overrun is held across a save.
    let mut m = Hawkes::new(c).unwrap();
    for i in 0..60usize {
        step(&mut m, i);
    }
    assert!(m.take_state_warning().is_some());
    let bytes = rmp_serde::to_vec(&m.state()).unwrap();
    let mut back = Hawkes::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    for i in 60..300usize {
        step(&mut back, i);
        assert!(back.take_state_warning().is_none(), "row {i}");
    }
    assert!(back.state_over_cap().is_some());
    let bytes = rmp_serde::to_vec(&back.state()).unwrap();
    let again = Hawkes::restore(&rmp_serde::from_slice(&bytes).unwrap()).unwrap();
    assert_eq!(again.state_over_cap(), back.state_over_cap());
}

/// The numbers a clock window's state holds are the bytes it counts: the
/// sums and the window's rows, the clock the model tracks and a stamp a row.
#[test]
fn a_clock_windows_numbers_are_the_bytes_it_counts() {
    fn floats(v: &rmpv::Value) -> u64 {
        match v {
            rmpv::Value::Array(a) => a.iter().map(floats).sum(),
            rmpv::Value::Map(m) => m
                .iter()
                .filter(|(k, _)| k.as_str() != Some("cfg"))
                .map(|(_, v)| floats(v))
                .sum(),
            rmpv::Value::F64(_) => 1,
            _ => 0,
        }
    }
    let (n, betas) = (2usize, vec![0.7, 4.0, 15.0]);
    let mut m = Hawkes::new(clock_cfg(n, &betas, 5.0, WindowClosed::Right)).unwrap();
    let mut most = 0;
    for (d, y) in grid_stream(n, 1500, 3) {
        m.step(&[], &y, d, 1.0);
        let bytes = rmp_serde::to_vec_named(&m.state()).unwrap();
        let v = rmpv::decode::read_value(&mut bytes.as_slice()).unwrap();
        let rows = m.window_rows() as u64;
        // Unstamped, a row's stamp is a raw clock and 0 removed: two
        // numbers, where the model counts a stamp as 32 bytes.
        let want = Hawkes::state_bytes(n, betas.len(), Some(m.window_rows())) + 8 + 16 * rows;
        assert_eq!(8 * floats(&v), want, "{rows} rows");
        most = most.max(rows);
    }
    assert!(most > 10);
}

#[test]
fn the_clock_windows_settings_are_refused_where_they_conflict() {
    let refuse = |c: HawkesCfg, says: &str| {
        let e = Hawkes::new(c).unwrap_err();
        assert!(e.contains(says), "{e}");
    };
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        refuse(
            clock_cfg(1, &[1.0], bad, WindowClosed::Right),
            "finite number of clock units above 0",
        );
    }
    let mut c = clock_cfg(1, &[1.0], 5.0, WindowClosed::Right);
    c.window_events = Some(5);
    refuse(c, "two windows");
    let mut c = cfg(1, &[1.0]);
    c.closed = WindowClosed::Both;
    refuse(c, "no clock window");
    let mut c = clock_cfg(1, &[1.0], 5.0, WindowClosed::Right);
    c.decay = Decay::Halflife(100.0);
    refuse(c, "takes no decay");
    let mut c = clock_cfg(1, &[1.0], 5.0, WindowClosed::Right);
    c.max_state_bytes = Some(0);
    refuse(c, "max_state_bytes must be above 0");
    let mut c = clock_cfg(1, &[1.0], 5.0, WindowClosed::Right);
    c.max_state_bytes = Some(10);
    refuse(c, "past max_state_bytes of 10");
}

/// A row of weight 0 counts in no clock (hard rule 9): the stream never
/// hands one to a model, and steps the next row from the last that counted.
/// That is held for `hawkes` under no window, a window of events and a
/// window of clock units at either edge, to the bit against the stream
/// without such rows, by `crates/rheon-polars/tests/zero_weight.rs`; here,
/// a row of weight 0 handed over anyway ages the state over its step and
/// reads what `predict` reads, as every model's does.
#[test]
fn a_zero_weight_row_handed_over_reads_as_predict_and_ages_the_state() {
    let base = grid_stream(2, 120, 41);
    for config in [
        cfg(2, &[1.0, 4.0]),
        clock_cfg(2, &[1.0, 4.0], 3.0, WindowClosed::Both),
    ] {
        let mut a = Hawkes::new(config.clone()).unwrap();
        let mut b = Hawkes::new(config).unwrap();
        for (i, (d, y)) in base.iter().enumerate() {
            let w = if i % 4 == 2 { 0.0 } else { 1.0 };
            let ahead = a.predict(&[], *d);
            let step = a.step(&[], y, *d, w);
            assert_eq!(
                step.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                ahead.pred.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "row {i}"
            );
            if w == 0.0 {
                b.step(&[], &[None, None], *d, 1.0);
            } else {
                b.step(&[], y, *d, 1.0);
            }
            assert_eq!(a.r, b.r, "row {i}: R");
            assert_eq!(a.g0.to_bits(), b.g0.to_bits(), "row {i}: G00");
        }
    }
}
