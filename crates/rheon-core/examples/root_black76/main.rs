//! The root-finder over rows read from stdin, one solve a line, for
//! `tests/test_second_opinion_root.py` to hold to scipy's `brentq`.
//!
//! A line is `kind lo hi increasing xtol rtol p...`, whitespace-separated:
//!
//! - `black76 lo hi inc xtol rtol F K tau df price`: the volatility at which
//!   Black-76's call is `price`; with `inc = 0` the solve is handed the
//!   negation, `price − call`, declared decreasing;
//! - `cubic lo hi inc xtol rtol c t`: the root of `x³ + c·x − t`, negated
//!   likewise.
//!
//! Each output line is `root lo hi evaluations bound reason`: the root
//! (`NaN` where withheld), the bracket, the evaluations against their bound,
//! and `ok` or the reason's name. Every number is written in Rust's shortest
//! form that reads back to the same double.

use std::io::{BufRead, Write};

use rheon_core::{Bounded, RootTolerance, find_root};

mod black76;

fn main() {
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    for line in stdin.lock().lines() {
        let line = line.expect("stdin is readable");
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        let num = |i: usize| -> f64 {
            fields[i]
                .parse()
                .unwrap_or_else(|_| panic!("field {i} of {line:?} is not a number"))
        };
        let (lo, hi, increasing) = (num(1), num(2), fields[3] == "1");
        let tol = RootTolerance::new(num(4), num(5)).expect("a valid tolerance");
        let sign = if increasing { 1.0 } else { -1.0 };
        let report = match fields[0] {
            "black76" => {
                let (fwd, strike, tau, df, price) = (num(6), num(7), num(8), num(9), num(10));
                let f = |s: f64| {
                    let (v, e) = black76::call_less(s, fwd, strike, tau, df, price);
                    Bounded::new(sign * v, e)
                };
                find_root(f, lo, hi, increasing, &tol)
            }
            "cubic" => {
                let (c, t) = (num(6), num(7));
                let f = |x: f64| {
                    let (v, e) = black76::cubic_less(x, c, t);
                    Bounded::new(sign * v, e)
                };
                find_root(f, lo, hi, increasing, &tol)
            }
            other => panic!("unknown kind {other:?}"),
        };
        let (root, reason) = match report.root {
            Ok(x) => (x, "ok"),
            Err(why) => (f64::NAN, why.name()),
        };
        writeln!(
            out,
            "{root} {} {} {} {} {reason}",
            report.lo, report.hi, report.evaluations, report.bound
        )
        .expect("stdout is writable");
    }
}
