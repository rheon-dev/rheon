"""Study 508's checks against ``river.drift`` and against definitions; each returns plain numbers.

``python scripts/study_508.py`` prints the tables the record quotes. A flag
comparison is exact (the rows where each side flags); a continuous output is
compared in ulps (``ulps``: the count of doubles between the two values).
"""

from __future__ import annotations

import math
import pickle
import random
import statistics
import struct
import sys
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
from river import drift as rdrift
from river.drift import binary as rbinary
from scipy import stats as sstats

sys.path.insert(0, str(Path(__file__).parent))
import data  # noqa: E402
from detectors import DDM, EDDM, KSWIN, Adwin, PageHinkley, ks_pvalue_equal, run  # noqa: E402


def ulps(a: float, b: float) -> int:
    if a == b:
        return 0
    if math.isnan(a) or math.isnan(b) or math.isinf(a) or math.isinf(b):
        return -1
    ia = struct.unpack("<q", struct.pack("<d", a))[0]
    ib = struct.unpack("<q", struct.pack("<d", b))[0]
    if ia < 0:
        ia = -(ia & 0x7FFFFFFFFFFFFFFF)
    if ib < 0:
        ib = -(ib & 0x7FFFFFFFFFFFFFFF)
    return abs(ia - ib)


def bits(a: float, b: float) -> bool:
    return struct.pack("<d", a) == struct.pack("<d", b)


# ---------------------------------------------------------------------------
# 1, 2. Flag for flag and bit for bit against river.

# (name, ours, river's, stream, field pairs compared bit for bit)
def _adwin_fields(m: Adwin, r) -> list[tuple[float, float]]:
    return [(m.width, r.width), (m.total, r.total), (m.var, r.variance), (m.estimation, r.estimation)]


def _ph_fields(m: PageHinkley, r) -> list[tuple[float, float]]:
    return [
        (m.mean, r._x_mean.get()),
        (m.s_inc, r._sum_increase),
        (m.s_dec, r._sum_decrease),
        (m.min_inc, r._min_increase),
        (m.max_dec, r._max_decrease),
    ]


def _ddm_fields(m: DDM, r) -> list[tuple[float, float]]:
    return [(m.p, r._p.get()), (m.p_min, r._p_min), (m.s_min, r._s_min), (m.ps_min, r._ps_min)]


def _eddm_fields(m: EDDM, r) -> list[tuple[float, float]]:
    v = r._error_distances
    return [(m.dmean, v.mean.get()), (m.dS, v._S), (m.p2s_max, r._p2s_prime_max)]


def _kswin_fields(m: KSWIN, r) -> list[tuple[float, float]]:
    return [(m.p_value, float(r.p_value))]


PARITY_CASES = [
    # label, ours factory, river factory, stream kind, binary?, fields, warning?
    ("ADWIN", lambda: Adwin(), lambda: rdrift.ADWIN(), "real", _adwin_fields, False),
    ("ADWIN clock=1, delta=0.05", lambda: Adwin(clock=1, delta=0.05), lambda: rdrift.ADWIN(clock=1, delta=0.05), "real", _adwin_fields, False),
    ("ADWIN on bits, M=3", lambda: Adwin(max_buckets=3), lambda: rdrift.ADWIN(max_buckets=3), "binary", _adwin_fields, False),
    ("Page-Hinkley, threshold 25", lambda: PageHinkley(threshold=25.0), lambda: rdrift.PageHinkley(threshold=25.0), "real", _ph_fields, False),
    ("Page-Hinkley up, alpha=1", lambda: PageHinkley(threshold=25.0, alpha=1.0, mode="up"), lambda: rdrift.PageHinkley(threshold=25.0, alpha=1.0, mode="up"), "real", _ph_fields, False),
    ("DDM", lambda: DDM(), lambda: rbinary.DDM(), "binary", _ddm_fields, True),
    ("EDDM", lambda: EDDM(), lambda: rbinary.EDDM(), "binary", _eddm_fields, True),
    ("KSWIN", lambda: KSWIN(seed=7), lambda: rdrift.KSWIN(seed=7), "real", _kswin_fields, False),
    ("KSWIN W=300, r=50", lambda: KSWIN(seed=7, window_size=300, stat_size=50), lambda: rdrift.KSWIN(seed=7, window_size=300, stat_size=50), "real", _kswin_fields, False),
]

KINDS = ("none", "step", "gradual")


def stream_for(kind: str, binary: bool, n: int, seed: int) -> np.ndarray:
    at = n // 2
    if binary:
        return data.binary(kind, n, seed, at=at, p0=0.1, p1=0.25, ramp=1000)
    return data.real(kind, n, seed, at=at, size=1.0, ramp=1000)


def parity(n: int = 10_000, seeds: int = 3) -> list[dict]:
    """Each case on each stream kind and seed: flags and warnings row for row, fields bit for bit."""
    out = []
    for label, mk, mkr, sk, fields, warn in PARITY_CASES:
        binary = sk == "binary"
        for kind in KINDS:
            res = {"case": label, "kind": kind, "rows": 0, "ours": 0, "river": 0, "flag_rows_differ": 0,
                   "warn_rows_differ": 0, "fields_not_bit_equal": 0, "max_ulps": 0}
            for seed in range(seeds):
                xs = stream_for(kind, binary, n, seed)
                m, r = mk(), mkr()
                is_bin = isinstance(r, (rbinary.DDM, rbinary.EDDM))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for x in xs:
                        x = float(x)
                        a = m.update(x)
                        r.update(bool(x) if is_bin else x)
                        res["rows"] += 1
                        res["ours"] += a
                        res["river"] += r.drift_detected
                        res["flag_rows_differ"] += a != r.drift_detected
                        if warn:
                            res["warn_rows_differ"] += m.warning != r.warning_detected
                        for p, q in fields(m, r):
                            if not bits(p, q):
                                res["fields_not_bit_equal"] += 1
                                res["max_ulps"] = max(res["max_ulps"], ulps(p, q))
            out.append(res)
    return out


# The paper's forms where river departs from them: how many flags move.

PAPER_FORMS = [
    ("ADWIN keeps W1 after a cut", lambda: Adwin(reset_on_drift=False), "real"),
    ("ADWIN clock=1 (every row)", lambda: Adwin(clock=1), "real"),
    ("ADWIN Hoeffding bound (Thm 3.1), bits", lambda: Adwin(bound="hoeffding"), "binary"),
    ("Page-Hinkley alpha=1", lambda: PageHinkley(threshold=25.0, alpha=1.0), "real"),
    ("Page-Hinkley mean as sum/n", lambda: PageHinkley(threshold=25.0, mean="sum"), "real"),
    ("DDM p = errors/n", lambda: DDM(mean="count"), "binary"),
    ("KSWIN D > sqrt(-ln(alpha)/r)", lambda: KSWIN(seed=7, test="paper"), "real"),
    ("KSWIN newest r vs all older (no draw)", lambda: KSWIN(seed=7, draw="all"), "real"),
]

RIVER_FORM = {
    "ADWIN": lambda: Adwin(),
    "Page-Hinkley": lambda: PageHinkley(threshold=25.0),
    "DDM": lambda: DDM(),
    "KSWIN": lambda: KSWIN(seed=7),
}


def paper_forms(n: int = 10_000, seeds: int = 5) -> list[dict]:
    out = []
    for label, mk, sk in PAPER_FORMS:
        base = RIVER_FORM[next(k for k in RIVER_FORM if label.startswith(k))]
        for kind in KINDS:
            fr = fp = differ = 0
            first_r: list[int] = []
            first_p: list[int] = []
            for seed in range(seeds):
                xs = stream_for(kind, sk == "binary", n, seed)
                a = run(base(), xs)
                b = run(mk(), xs)
                fr += int(a.sum())
                fp += int(b.sum())
                differ += int((a != b).sum())
                if kind != "none":
                    at = n // 2
                    ia = np.flatnonzero(a[at:])
                    ib = np.flatnonzero(b[at:])
                    if len(ia):
                        first_r.append(int(ia[0]))
                    if len(ib):
                        first_p.append(int(ib[0]))
            out.append({
                "form": label, "kind": kind, "river_flags": fr, "paper_flags": fp, "rows_differ": differ,
                "delay_river": statistics.median(first_r) if first_r else None,
                "delay_paper": statistics.median(first_p) if first_p else None,
            })
    return out


# KSWIN's p-value: river (scipy's Horner evaluation) against the integer sum rounded once.

def kswin_pvalues() -> list[dict]:
    out = []
    for r in (10, 30, 50, 100, 300):
        worst = 0
        worst_h = 0
        for h in range(1, r + 1):
            exact = ks_pvalue_equal(r, h)
            a = np.zeros(r)
            b = np.zeros(r)
            # two samples whose D is h/r: b shifted so that h of its values sit above all of a
            a[:] = np.arange(r, dtype=float)
            b[:] = np.arange(r, dtype=float) + h - 0.5
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = sstats.ks_2samp(a, b, method="exact")
            assert round(res.statistic * r) == h, (r, h, res.statistic)
            u = ulps(exact, float(res.pvalue))
            if u > worst:
                worst, worst_h = u, h
        crit = min(h for h in range(1, r + 1) if ks_pvalue_equal(r, h) <= 0.005) if r >= 10 else None
        out.append({"r": r, "max_ulps": worst, "at_h": worst_h, "h_crit(alpha=0.005)": crit,
                    "size": ks_pvalue_equal(r, crit) if crit else None,
                    "paper_h_crit": math.floor(r * math.sqrt(-math.log(0.005) / r)) + 1})
    return out


# ---------------------------------------------------------------------------
# 3. False alarms and delay over seeds.

def _alarm_stats(make, gen, n: int, at: int | None, seeds: int, count_checks=None) -> dict:
    fa = 0
    null_rows = 0
    checks = 0
    delays = []
    missed = 0
    for seed in range(seeds):
        xs = gen(seed)
        det = make()
        flags = np.zeros(n, dtype=bool)
        ck = 0
        for i, x in enumerate(xs):
            flags[i] = det.update(float(x))
            if count_checks is not None:
                ck += count_checks(det)
        checks += ck
        end = n if at is None else at
        fa += int(flags[:end].sum())
        null_rows += end
        if at is not None:
            after = np.flatnonzero(flags[at:])
            if len(after):
                delays.append(int(after[0]) + 1)  # rows after the change, the change row counted as 1
            else:
                missed += 1
    d = np.array(delays) if delays else np.array([np.nan])
    return {
        "false_alarms": fa,
        "null_rows": null_rows,
        "fa_per_row": fa / null_rows,
        "checks": checks,
        "delay_median": float(np.median(d)),
        "delay_p10": float(np.percentile(d, 10)),
        "delay_p90": float(np.percentile(d, 90)),
        "missed": missed,
        "delay_max": float(np.max(d)),
        "delays": delays,
    }


def _adwin_checks(det: Adwin) -> int:
    # a check ran on this row (the tick was just advanced)
    return int(det.tick % det.clock == 0 and det.width > det.grace)


ALARM_CASES = [
    # label, factory, binary?, change size, checks counter
    ("ADWIN delta=0.002", lambda: Adwin(), False, 1.0, _adwin_checks),
    ("ADWIN delta=0.002, clock=1", lambda: Adwin(clock=1), False, 1.0, _adwin_checks),
    ("ADWIN delta=0.1", lambda: Adwin(delta=0.1), False, 1.0, _adwin_checks),
    ("ADWIN delta=0.002, bits", lambda: Adwin(), True, None, _adwin_checks),
    ("ADWIN Hoeffding, delta=0.002, bits", lambda: Adwin(bound="hoeffding"), True, None, _adwin_checks),
    ("Page-Hinkley lambda=25", lambda: PageHinkley(threshold=25.0), False, 1.0, None),
    ("Page-Hinkley lambda=50 (default)", lambda: PageHinkley(), False, 1.0, None),
    ("DDM", lambda: DDM(), True, None, None),
    ("EDDM", lambda: EDDM(), True, None, None),
    ("KSWIN alpha=0.005", lambda: KSWIN(seed=11), False, 1.0, None),
]


def alarms(seeds: int = 200, n: int = 10_000, labels: list[str] | None = None) -> list[dict]:
    out = []
    at = n // 2
    for label, mk, binary, size, ck in ALARM_CASES:
        if labels and label not in labels:
            continue
        if binary:
            def null(s, n=n):
                return data.binary("none", n, s, p0=0.1)

            def step(s, n=n):
                return data.binary("step", n, s, at=at, p0=0.1, p1=0.25)

            def grad(s, n=n):
                return data.binary("gradual", n, s, at=at, p0=0.1, p1=0.25, ramp=1000)
        else:
            def null(s, n=n):
                return data.real("none", n, s)

            def step(s, n=n, size=size):
                return data.real("step", n, s, at=at, size=size)

            def grad(s, n=n, size=size):
                return data.real("gradual", n, s, at=at, size=size, ramp=1000)
        t0 = time.time()
        r0 = _alarm_stats(mk, null, n, None, seeds, ck)
        r1 = _alarm_stats(mk, step, n, at, seeds)
        r2 = _alarm_stats(mk, grad, n, at, seeds)
        out.append({"case": label, "null": r0, "step": r1, "gradual": r2, "secs": time.time() - t0})
    return out


def kswin_size(seeds: int = 200, n: int = 10_000, decimals: int | None = None) -> dict:
    """KSWIN on the null: share of tests (full-window rows) that reject, against the test's exact size."""
    rejects = tests = 0
    det_proto = KSWIN(seed=11)
    for seed in range(seeds):
        xs = data.real("none", n, seed)
        if decimals is not None:
            xs = np.round(xs, decimals)
        det = KSWIN(seed=11 + seed)
        for x in xs:
            f = det.update(float(x))
            tests += det.tested
            rejects += f
    r = det_proto.r
    hc = min(h for h in range(1, r + 1) if ks_pvalue_equal(r, h) <= det_proto.alpha)
    return {"tests": tests, "rejects": rejects, "rate": rejects / tests, "exact_size": ks_pvalue_equal(r, hc),
            "alpha": det_proto.alpha, "decimals": decimals}


# ---------------------------------------------------------------------------
# 4. State size.

def adwin_levels_bound(width: float, M: int) -> int:
    """Most levels a window of ``width`` rows can hold: the top level and ``M - 1`` buckets on each below it."""
    # width >= M 2^(L-1) - (M - 1)  =>  L <= 1 + log2((width + M - 1) / M)
    return 1 + math.floor(math.log2((width + M - 1) / M))


def adwin_bytes(levels: int, M: int, compact: bool = False) -> int:
    per = (2 * M * 8 + 1) if compact else (2 * (M + 1) * 8 + 8)
    return levels * per + 6 * 8


def adwin_state(widths=(1_000, 10_000, 100_000, 1_000_000), M: int = 5, seed: int = 0) -> list[dict]:
    """Grow an ADWIN on a stationary stream and record levels and buckets as the width passes each mark."""
    out = []
    n = max(widths) + 1
    xs = data.real("none", n, seed)
    det = Adwin(max_buckets=M, delta=1e-12)  # a tiny delta: no false cut, so the width is the row count
    max_lv_seen = 0
    over_bound = 0
    marks = list(widths)
    for i, x in enumerate(xs):
        det.update(float(x))
        lv = len(det.levels)
        if lv > adwin_levels_bound(det.width, M):
            over_bound += 1
        max_lv_seen = max(max_lv_seen, lv)
        if marks and det.width == marks[0]:
            arr = det.state_arrays()
            counted = sum(a.nbytes for a in arr.values())
            out.append({
                "width": int(det.width), "levels": lv, "max_levels_so_far": max_lv_seen,
                "levels_bound": adwin_levels_bound(det.width, M), "buckets": det.n_buckets(),
                "bytes_counted": counted, "bytes_formula": adwin_bytes(lv, M),
                "bytes_bound": adwin_bytes(adwin_levels_bound(det.width, M), M),
                "rows_over_bound": over_bound,
            })
            marks.pop(0)
    return out


def adwin_capped_state(cap: int, M: int = 5, rows: int = 50_000) -> dict:
    det = Adwin(max_buckets=M, delta=1e-12, max_width=cap)
    xs = data.real("none", rows, 3)
    mx = 0
    mxw = 0
    for x in xs:
        det.update(float(x))
        mx = max(mx, len(det.levels))
        mxw = max(mxw, det.width)
    return {"cap": cap, "max_levels": mx, "max_width": mxw, "bound": adwin_levels_bound(cap, M),
            "bytes_bound": adwin_bytes(adwin_levels_bound(cap, M), M)}


def river_adwin_pickle(width: int) -> int:
    r = rdrift.ADWIN(delta=1e-12)
    for x in data.real("none", width, 0):
        r.update(float(x))
    return len(pickle.dumps(r._helper))


def small_states() -> list[dict]:
    rows = []
    xs = data.real("step", 2_000, 0, at=1_000)
    bs = data.binary("step", 2_000, 0, at=1_000)
    for label, det, s in (
        ("Page-Hinkley", PageHinkley(), xs),
        ("DDM", DDM(), bs),
        ("EDDM", EDDM(), bs),
    ):
        run(det, s)
        arr = det.state_arrays()
        rows.append({"detector": label, "bytes": sum(a.nbytes for a in arr.values()), "arrays": {k: a.shape for k, a in arr.items()}})
    for W, r in ((100, 30), (1_000, 30), (10_000, 30)):
        for draw in ("random", "all"):
            det = KSWIN(window_size=W, stat_size=r, draw=draw)
            run(det, data.real("none", W + 5, 0))
            arr = det.state_arrays()
            rows.append({"detector": f"KSWIN W={W}, r={r}, draw={draw}", "bytes": sum(a.nbytes for a in arr.values()),
                         "arrays": {k: a.shape for k, a in arr.items()}})
    return rows


# ---------------------------------------------------------------------------
# 5. Cost.

def _time_rows(det, xs) -> float:
    upd = det.update
    t0 = time.perf_counter()
    for x in xs:
        upd(x)
    return (time.perf_counter() - t0) / len(xs)


def cost(timed: int = 20_000) -> list[dict]:
    out = []
    for W in (1_000, 10_000):
        fill = [float(v) for v in data.real("none", W + 200, 1)]
        tail = [float(v) for v in data.real("none", timed, 2)]
        # ADWIN held at a width of W (the cap), no cut on the null
        ours = Adwin(max_width=W, delta=1e-12)
        for x in fill:
            ours.update(x)
        t_ours = _time_rows(ours, tail)
        ours1 = Adwin(max_width=W, delta=1e-12, clock=1)
        for x in fill:
            ours1.update(x)
        t_ours1 = _time_rows(ours1, tail[: timed // 4])
        # river has no cap: time it from width W on (its width grows by `timed` rows meanwhile)
        rv = rdrift.ADWIN(delta=1e-12)
        for x in fill[:W]:
            rv.update(x)
        t_rv = _time_rows(rv, tail)
        out.append({"detector": "ADWIN", "W": W, "ours_us": t_ours * 1e6, "ours_clock1_us": t_ours1 * 1e6, "river_us": t_rv * 1e6})
        for r in (30, W // 10):
            k = KSWIN(window_size=W, stat_size=r, seed=1, alpha=0.0)  # alpha 0: never flags, every row a test
            for x in fill:
                k.update(x)
            nt = 2_000 if r == 30 else max(100, 200_000 // W)
            t_k = _time_rows(k, tail[:nt])
            kr = rdrift.KSWIN(window_size=W, stat_size=r, seed=1, alpha=0.0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for x in fill:
                    kr.update(x)
                t_kr = _time_rows(kr, tail[:nt])
            out.append({"detector": f"KSWIN r={r}", "W": W, "ours_us": t_k * 1e6, "ours_clock1_us": None, "river_us": t_kr * 1e6})
    xs = [float(v) for v in data.real("none", timed, 3)]
    bs = [float(v) for v in data.binary("none", timed, 3)]
    for label, ours, theirs, s, conv in (
        ("Page-Hinkley", PageHinkley(), rdrift.PageHinkley(), xs, float),
        ("DDM", DDM(), rbinary.DDM(), bs, bool),
        ("EDDM", EDDM(), rbinary.EDDM(), bs, bool),
    ):
        t_o = _time_rows(ours, s)
        sc = [conv(v) for v in s]
        t_r = _time_rows(theirs, sc)
        out.append({"detector": label, "W": None, "ours_us": t_o * 1e6, "ours_clock1_us": None, "river_us": t_r * 1e6})
    return out


# ---------------------------------------------------------------------------
# 6. Zero weights; 7. chunks and save / load.

FACTORIES = {
    "ADWIN": (lambda: Adwin(), lambda: rdrift.ADWIN(), False),
    "Page-Hinkley": (lambda: PageHinkley(threshold=25.0), lambda: rdrift.PageHinkley(threshold=25.0), False),
    "DDM": (lambda: DDM(), lambda: rbinary.DDM(), True),
    "EDDM": (lambda: EDDM(), lambda: rbinary.EDDM(), True),
    "KSWIN": (lambda: KSWIN(seed=5), lambda: rdrift.KSWIN(seed=5), False),
}


def zero_weights(n: int = 20_000, seed: int = 4) -> list[dict]:
    """Ours with zero-weight rows (the first among them) against river on the stream without them."""
    out = []
    w = data.weights(n, seed)
    for label, (mk, mkr, binary) in FACTORIES.items():
        xs = stream_for("step", binary, n, seed)
        det = mk()
        flags = np.zeros(n, dtype=bool)
        state_moved = 0
        for i, (x, wi) in enumerate(zip(xs, w)):
            if wi == 0.0:
                before = pickle.dumps(det.save())
            flags[i] = det.update(float(x), float(wi))
            if wi == 0.0 and pickle.dumps(det.save()) != before:
                state_moved += 1
        keep = w > 0
        r = mkr()
        rf = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for x in xs[keep]:
                r.update(bool(x) if binary else float(x))
                rf.append(r.drift_detected)
        rf = np.array(rf)
        out.append({
            "detector": label, "rows": n, "zero_rows": int((~keep).sum()), "first_row_zero": bool(w[0] == 0.0),
            "flags": int(flags.sum()), "kept_rows_differ": int((flags[keep] != rf).sum()),
            "zero_rows_flagged": int(flags[~keep].sum()), "zero_rows_moving_state": state_moved,
        })
    return out


def chunks(n: int = 20_000, seed: int = 6) -> list[dict]:
    """One pass against random chunks with a save and a load into a fresh detector at every boundary."""
    out = []
    rng = random.Random(seed)
    w = data.weights(n, seed, frac_zero=0.05)
    for label, (mk, _, binary) in FACTORIES.items():
        xs = stream_for("step", binary, n, seed)
        one = mk()
        f1 = np.array([one.update(float(x), float(wi)) for x, wi in zip(xs, w)])
        bounds = sorted(set(rng.sample(range(1, n), 200)))
        det = mk()
        f2 = []
        prev = 0
        n_chunks = 0
        for b in bounds + [n]:
            for x, wi in zip(xs[prev:b], w[prev:b]):
                f2.append(det.update(float(x), float(wi)))
            s = pickle.loads(pickle.dumps(det.save()))
            det = mk()
            det.load(s)
            prev = b
            n_chunks += 1
        out.append({
            "detector": label, "chunks": n_chunks, "flags": int(f1.sum()),
            "rows_differ": int((f1 != np.array(f2)).sum()),
            "final_state_equal": pickle.dumps(one.save()) == pickle.dumps(det.save()),
        })
    return out


# ---------------------------------------------------------------------------
# 8. Numerical risk.

def _window_exact(xs: list[float]) -> tuple[Fraction, Fraction]:
    s = sum(Fraction(x) for x in xs)
    n = len(xs)
    mu = s / n
    v = sum((Fraction(x) - mu) ** 2 for x in xs)
    return s, v


def adwin_drift_of_sums(offset: float, cap: int, rows: int, seed: int = 8) -> dict:
    """ADWIN's running total and sum of squares against exact rationals over its window (a suffix of the stream)."""
    xs = [float(v) for v in data.real("none", rows, seed) + offset]
    det = Adwin(max_width=cap, delta=1e-12)
    for x in xs:
        det.update(x)
    w = int(det.width)
    s, v = _window_exact(xs[-w:])
    return {
        "offset": offset, "cap": cap, "rows": rows, "width": w,
        "total_abs": float(abs(Fraction(det.total) - s)),
        "total_rel": float(abs(Fraction(det.total) - s) / abs(s)),
        "var_rel": float(abs(Fraction(det.var) - v) / v),
        "var_per_row": float(v / w),
    }


def mean_forms(rows: int, offset: float, seed: int = 9) -> dict:
    """The running mean as river keeps it, ``m += (x - m)/n``, and as ``sum/n``, against the exact mean.

    The reference is ``fsum`` of the prefix (the sum correctly rounded) divided in rationals and
    rounded: within one ulp of the exact mean, so gaps of a few ulps are at the reference's
    resolution.
    """
    xs = (data.real("none", rows, seed) + offset).tolist()
    m = 0.0
    s = 0.0
    n = 0.0
    marks = {}
    for i, x in enumerate(xs):
        n += 1.0
        m += (1.0 / n) * (x - m)
        s += x
        k = i + 1
        if k in (10**3, 10**4, 10**5, 10**6, 10**7):
            exact = float(Fraction(math.fsum(xs[:k])) / k)
            marks[k] = (ulps(m, exact), ulps(s / n, exact))
    return {"offset": offset, "marks": marks}


def ph_sum_error(rows: int, seed: int = 10, alpha: float = 1.0) -> dict:
    """Page-Hinkley's cumulative sum ``m_t`` against a more precise run of the same recursion.

    Each row adds ``x_t - xbar_t - delta`` (``xbar_t`` as the detector holds it, so the mean's own
    rounding is left out). With ``alpha = 1`` the reference is ``fsum`` of those increments (exact,
    rounded once); with ``alpha < 1`` it is the recursion in ``np.longdouble`` (x86-64's 64-bit
    significand, 11 bits more than a double).
    """
    xs = data.real("none", rows, seed).tolist()
    det = PageHinkley(alpha=alpha, threshold=1e300)
    devs = []
    worst = 0.0
    la = np.longdouble(alpha)
    ld = np.longdouble(0.0)
    dl = np.longdouble(det.delta)
    for i, x in enumerate(xs):
        det.update(x)
        dev = x - det.mean
        if alpha == 1.0:
            devs.append(dev)
        else:
            ld = la * ld + np.longdouble(dev) - dl
        k = i + 1
        if k in (10**3, 10**4, 10**5, 10**6) or k == rows:
            ref = math.fsum(devs + [-det.delta] * k) if alpha == 1.0 else float(ld)
            worst = max(worst, abs(det.s_inc - ref))
    return {"rows": rows, "alpha": alpha, "final_m": det.s_inc, "worst_abs": worst, "rel_to_final": worst / abs(det.s_inc)}


def ddm_count_vs_recursive(rows: int = 1_000_000, seed: int = 12) -> dict:
    bs = data.binary("none", rows, seed, p0=0.1).tolist()
    p = 0.0
    n = 0.0
    e = 0.0
    worst = 0
    for b in bs:
        n += 1.0
        p += (1.0 / n) * (b - p)
        e += b
        worst = max(worst, ulps(p, e / n))
    return {"rows": rows, "max_ulps": worst, "final_ulps": ulps(p, e / n)}


# The ADWIN delay the bound implies for a step at an exact split.

def adwin_n1(delta: float, n0: int, dmu: float, sd0: float, bound: str, factor: float, L: float = 5.0) -> int:
    """Smallest ``n1`` with ``|dmu| > factor * eps_cut`` for a split at the change (``n0`` rows before it).

    ``factor = 2`` is Theorem 3.1's false-negative statement (the window shrinks with probability at
    least ``1 - delta``); ``factor = 1`` is where the statistic's expected value first crosses the
    threshold. The window's variance is the mixture's, ``sd0^2 + n0 n1 dmu^2 / (n0 + n1)^2``.
    """
    for n1 in range(1, 10**7):
        width = n0 + n1
        if bound == "hoeffding":
            m_recip = 1.0 / n0 + 1.0 / n1
            eps = math.sqrt(m_recip / 2.0 * math.log(4.0 * width / delta))
        else:
            if n1 < L:
                continue
            var = sd0 * sd0 + n0 * n1 * dmu * dmu / (width * width)
            dp = math.log(2.0 * math.log(width) / delta)
            m_recip = 1.0 / (n0 - L + 1.0) + 1.0 / (n1 - L + 1.0)
            eps = math.sqrt(2.0 * m_recip * var * dp) + 2.0 / 3.0 * dp * m_recip
        if abs(dmu) > factor * eps:
            return n1
    return -1
