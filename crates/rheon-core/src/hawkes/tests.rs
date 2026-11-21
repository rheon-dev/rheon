use super::*;
use crate::oracle;

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
/// the stream's length, and the window's at most `8 W (1 + D + k)` beside.
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
        let window = cap * (1 + n + k);
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
