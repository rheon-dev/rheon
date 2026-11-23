"""Print the reference numbers `crates/rheon-core/src/ewm_moments/tests.rs`
holds the Rust state to (PLAN 554, parts 3 and 4).

The stream is the one the Rust test generates: the integer generator of
`tests/model_contract.rs` and arithmetic alone, so both sides form the same
doubles. The reference (`tests/reference_ewm_moments.py`) is run on it with
the output read after each row (`emit="after"`), and the numbers at a few
rows are printed as Rust literals, each with its bits.

    python3 scripts/ewm_moments_literals_554.py > crates/rheon-core/src/ewm_moments/reference_values.rs

Needs nothing but the standard library: the reference is pure Python.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tests"))

import reference_ewm_moments as rm  # noqa: E402

N = 60
CHECK_ROWS = (2, 31, 59)


def lcg(state: int) -> tuple[int, float]:
    state = (state * 6364136223846793005 + 1442695040888963407) % 2**64
    return state, (state >> 11) / 2**53


def stream(seed: int, zero_rows: tuple[int, ...], steps: bool) -> dict[str, list[float]]:
    """`x = 3 + 4u^3 + v/2` (skewed), `w = 1/4 + u'`, a weight of 0 on
    `zero_rows`, and a clock stepping 1, 1.5, 2, ... (sums of halves, exact)
    under `steps`, else 1 a row."""
    s = seed
    x, w, t = [], [], []
    clock = 0.0
    for i in range(N):
        s, u = lcg(s)
        s, v = lcg(s)
        s, q = lcg(s)
        x.append(3.0 + 4.0 * u * u * u + 0.5 * v)
        w.append(0.0 if i in zero_rows else 0.25 + q)
        clock += 1.0 + (i % 3) * 0.5 if steps else 1.0
        t.append(clock)
    return {"x": x, "w": w, "t": t}


def terms(bias: bool) -> tuple[dict, dict]:
    top = 6 if bias else 4
    tm: dict[str, dict[str, int]] = {}
    kd: dict[str, str] = {}
    for p in range(2, top + 1):
        tm[f"c{p}"] = {"x": p}
        kd[f"c{p}"] = "central"
        tm[f"k{p}"] = {"x": p}
        kd[f"k{p}"] = "cumulant"
    for p in range(3, top + 1):
        tm[f"s{p}"] = {"x": p}
        kd[f"s{p}"] = "standardized"
    if bias:
        for p in range(1, 7):
            tm[f"r{p}"] = {"x": p}
            kd[f"r{p}"] = "raw"
    return tm, kd


def run(data, bias: bool, **decay) -> None:
    tm, kd = terms(bias)
    spec = rm.EwmMoments(tm, kind=kd, bias=bias, clocked=True, emit="after", **decay)
    for i in range(N):
        part = {k: [v[i]] for k, v in data.items()}
        out = spec.feed(part)
        if i in CHECK_ROWS:
            (summ,) = spec.sums.values()
            rows = [("W", summ.W)] if bias else []
            rows += [("kish_r2", summ.r[0])] if bias else []
            rows += [(k + ("" if bias else "u"), o[0].value) for k, o in out.items()]
            for k, v in rows:
                lit = "f64::NAN" if v is None else repr(v)
                print(f'    ({i}, "{k}", {lit}),')


def stream2(seed: int, zero_rows: tuple[int, ...], gap_row: int | None) -> dict[str, list[float]]:
    """Part 4's stream: `x` as `stream`'s, `y = 1/2 + x/2 + q'` (each from
    the same generator, four draws a row), the weight as `stream`'s, and a
    clock stepping 1, 1.5, 2, ... with a step of 9.5 on `gap_row`."""
    s = seed
    x, y, w, t = [], [], [], []
    clock = 0.0
    for i in range(N):
        s, u = lcg(s)
        s, v = lcg(s)
        s, q = lcg(s)
        s, q2 = lcg(s)
        xi = 3.0 + 4.0 * u * u * u + 0.5 * v
        x.append(xi)
        y.append(0.5 + 0.5 * xi + q2)
        w.append(0.0 if i in zero_rows else 0.25 + q)
        clock += 9.5 if i == gap_row else 1.0 + (i % 3) * 0.5
        t.append(clock)
    return {"x": x, "y": y, "w": w, "t": t}


#: Part 4's terms: (name, legs, kind).
LAG_TERMS = [
    ("ac1", {"x": 1, "x@1": 1}, "central"),
    ("acor2", {"x": 1, "x@2": 1}, "standardized"),
    ("xy", {"x": 1, "y": 1}, "central"),
    ("xy3", {"x": 1, "y@3": 2}, "central"),
    ("x3y1", {"x": 3, "y@1": 1}, "central"),
    ("c22", {"x": 2, "x@1": 2}, "central"),
    ("cov1", {"x": 1, "y@1": 1}, "central"),
    ("m22", {"x": 2, "y": 2}, "central"),
    ("raw2", {"x": 1, "y@2": 1}, "raw"),
    ("lagmean", {"y@3": 1}, "raw"),
    ("sk1", {"x@1": 3}, "standardized"),
    ("ku1", {"x@1": 4}, "standardized"),
]
CLOCK_TERMS = [
    ("ac2", {"x": 1, "x@2": 1}, "central"),
    ("acor2", {"x": 1, "x@2": 1}, "standardized"),
    ("xy35", {"x": 1, "y@3.5": 1}, "central"),
    ("sk2", {"x@2": 3}, "standardized"),
    ("lm35", {"y@3.5": 1}, "raw"),
]
WINDOW_TERMS = [
    ("c2", {"x": 2}, "central"),
    ("c4", {"x": 4}, "central"),
    ("sk", {"x": 3}, "standardized"),
    ("ac1", {"x": 1, "x@1": 1}, "central"),
    ("corr", {"x": 1, "y": 1}, "standardized"),
    ("k4", {"x": 4}, "cumulant"),
]
SE_TERMS = [
    ("c2", {"x": 2}, "central"),
    ("c3", {"x": 3}, "central"),
    ("s3", {"x": 3}, "standardized"),
    ("s4", {"x": 4}, "standardized"),
    ("r2", {"x": 2}, "raw"),
    ("k4", {"x": 4}, "cumulant"),
    ("mean", {"x": 1}, "raw"),
    ("xy", {"x": 1, "y": 1}, "central"),
    ("corr", {"x": 1, "y": 1}, "standardized"),
    ("ac1", {"x": 1, "x@1": 1}, "central"),
]
ROWS4 = (2, 13, 31, 44, 59)


def print_terms(const: str, terms) -> None:
    print(f"const {const}: &Spec = &[")
    for name, legs, kind in terms:
        ls = ", ".join(f'("{k}", {v})' for k, v in legs.items())
        print(f'    ("{name}", &[{ls}], "{kind}"),')
    print("];")


def run4(const: str, data, terms, biases=(True,), suffix=("",), **kw) -> None:
    """Every term (each `bias` in turn, names suffixed) at `ROWS4`, read
    after each row; NaN where withheld."""
    print(f"const {const}: &[(usize, &str, f64)] = &[")
    for bias, suf in zip(biases, suffix, strict=True):
        tm = {
            n: legs
            for n, legs, k in terms
            if bias or (k != "raw" and sum(legs.values()) <= 4)
        }
        kd = {n: k for n, _, k in terms if n in tm}
        spec = rm.EwmMoments(tm, kind=kd, bias=bias, emit="after", **kw)
        cols = sorted({leg.split("@")[0] for legs in tm.values() for leg in legs})
        for i in range(N):
            part = {c: [data[c][i]] for c in [*cols, "w", "t"]}
            out = spec.feed(part)
            if i in ROWS4:
                for k, o in out.items():
                    lit = "f64::NAN" if o[0].value is None else repr(o[0].value)
                    print(f'    ({i}, "{k}{suf}", {lit}),')
    print("];")


def main4() -> None:
    print()
    print("// Part 4: lags, windows and the standard error, on `stream2`.")
    print_terms("LAG_TERMS", LAG_TERMS)
    print_terms("CLOCK_TERMS", CLOCK_TERMS)
    print_terms("WINDOW_TERMS", WINDOW_TERMS)
    print_terms("SE_TERMS", SE_TERMS)
    rows = stream2(5542, (0, 17, 40), None)
    run4("REF_LAG_ROWS", rows, LAG_TERMS, (True, False), ("", "u"), lam=0.9)
    clock = stream2(5543, (0, 17, 18, 40), 30)
    run4("REF_LAG_CLOCK", clock, CLOCK_TERMS, clocked=True, half_life=7.0)
    run4(
        "REF_LAG_TOL",
        clock,
        CLOCK_TERMS,
        clocked=True,
        half_life=7.0,
        lag_tolerance=0.5,
    )
    run4("REF_WIN_ROWS", rows, WINDOW_TERMS, lam=0.95, window_rows=12)
    run4(
        "REF_WIN_CLOCK",
        clock,
        WINDOW_TERMS,
        clocked=True,
        half_life=20.0,
        window_clock=9.0,
    )
    run4("REF_SE", rows, SE_TERMS, lam=0.97, se=True)


def main() -> None:
    print("// Generated by `python3 scripts/ewm_moments_literals_554.py`: the Python")
    print("// reference (`tests/reference_ewm_moments.py`) on the streams `tests.rs`'s")
    print("// `stream` and `stream2` make. NaN where the reference withholds.")
    print()
    # A: a zero-weight first row and no other, one clock unit a row under
    # lam = 0.9, so the weight and Kish's ratio go through no libm and are
    # the Rust's to the bit on every platform.
    a = stream(554, (0,), False)
    print("const REF_A: &[(usize, &str, f64)] = &[")
    run(a, True, lam=0.9)
    run(a, False, lam=0.9)
    print("];")
    # B: zero-weight rows inside, a half-life of 7 clock units.
    b = stream(5541, (0, 17, 18, 40), True)
    print("const REF_B: &[(usize, &str, f64)] = &[")
    run(b, True, half_life=7.0)
    run(b, False, half_life=7.0)
    print("];")
    main4()


if __name__ == "__main__":
    main()
