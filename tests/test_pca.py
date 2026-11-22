"""``pca`` through the bank: the definitions of CCIPCA and Oja's rule written
out longhand in numpy (no library computes either), the eigengap gate on
both sides of its threshold, and what the bank owes every model -- chunk
invariance, a save mid-stream, zero weights, the clock in rows and as a
duration, its refusals and the state's cap. The third-party oracles
(scikit-learn's ``IncrementalPCA`` and ``PCA``, scipy) are in
``tests/test_second_opinion_pca.py``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"

INF = float("inf")
CRITICAL = 9.210340371976184  # chi-squared, 2 df, 99th percentile


def stream(n: int, spectrum: list[float], seed: int, weights: bool = False) -> pl.DataFrame:
    """Gaussian rows whose covariance has eigenvalues ``spectrum`` along a
    random basis, a mean near 5, and (with ``weights``) unequal weights with
    one row in ten at 0."""
    rng = np.random.default_rng(seed)
    k = len(spectrum)
    basis = np.linalg.qr(rng.standard_normal((k, k)))[0]
    x = (rng.standard_normal((n, k)) * np.sqrt(spectrum)) @ basis.T + 5.0
    df = pl.DataFrame(x, schema=[f"x{i}" for i in range(k)])
    if weights:
        w = rng.uniform(0.5, 1.5, n)
        w[rng.uniform(size=n) < 0.1] = 0.0
        df = df.with_columns(w=pl.Series(w))
    return df


def spec(k: int, **kw):
    if "lam" not in kw:
        kw.setdefault("half_life", INF)
    return rp.spec.pca("p", features=[f"x{i}" for i in range(k)], **kw)


def run(df: pl.DataFrame, s) -> pl.DataFrame:
    return rp.ModelBank([s]).fit_predict(df).unnest("p")


# --- the definitions, longhand ----------------------------------------------


def longhand(x: np.ndarray, w: np.ndarray, lam: float, q: int, method: str, gain: float = 4.0):
    """The recursion of the module docs, written out from the papers: the
    mean form, ``a = lam W / W'``, ``b = w / W'``, ``d = x - m`` before the
    row; the directions as unit rows and their variances. Returns each row's
    directions and variances *before* the row (what the row reads)."""
    n, k = x.shape
    W = 0.0
    m = np.zeros(k)
    tr = 0.0
    E = np.zeros((q, k))
    L = np.zeros(q)
    seen = []
    for t in range(n):
        seen.append((E.copy(), L.copy()))
        wt = w[t]
        if wt == 0:
            # In no clock: the next row decays as if this one were not there.
            continue
        Wn = lam * W + wt
        if Wn <= 0:
            W = Wn
            continue
        a, b = lam * W / Wn, wt / Wn
        d = x[t] - m
        m = m + b * d
        tr = a * tr + a * b * d @ d
        W = Wn
        ab = a * b
        if a <= 0:
            E[:] = 0.0
            L[:] = 0.0
            continue
        if ab <= 0:
            continue
        old = E.copy()
        floor = (np.finfo(float).eps * k) ** 2 * (d @ d)
        if method == "ccipca":
            u = d.copy()
            for i in range(q):
                if L[i] > 0:
                    v = a * L[i] * E[i] + ab * (u @ E[i]) * u
                else:
                    if not u @ u > floor:
                        continue
                    v = ab * math.sqrt(u @ u) * u
                nv = math.sqrt(v @ v)
                E[i] = v / nv
                L[i] = nv
                u = u - (u @ E[i]) * E[i]
        else:  # oja
            eta = gain * b / tr
            for i in range(q):
                if np.any(E[i] != 0):
                    p = E[i] @ d
                    E[i] = E[i] + eta * p * d
                    L[i] = a * L[i] + ab * p * p
            started = 0
            for i in range(q):
                if not np.any(E[i] != 0):
                    continue
                for j in range(i):
                    if np.any(E[j] != 0):
                        E[i] = E[i] - (E[i] @ E[j]) * E[j]
                E[i] = E[i] / math.sqrt(E[i] @ E[i])
                started = i + 1
            if started < q and not np.any(E[started] != 0):
                u = d.copy()
                for j in range(started):
                    u = u - (u @ E[j]) * E[j]
                if u @ u > floor:
                    E[started] = u / math.sqrt(u @ u)
                    L[started] = ab * (u @ u)
        for i in range(q):
            along = E[i] @ old[i]
            if along < 0 or (along == 0 and E[i][np.argmax(np.abs(E[i]))] < 0):
                E[i] = -E[i]
    return seen


@pytest.mark.parametrize("method", ["ccipca", "oja"])
@pytest.mark.parametrize("lam", [1.0, 0.99])
def test_the_stochastic_methods_are_their_definitions(method, lam):
    """Each emitted loading against the recursion written out in numpy, row
    by row, under decay and unequal weights with zeros among them: to
    rounding (largest measured 2.2e-13 over 3,000 rows)."""
    df = stream(3000, [9.0, 4.0, 1.0, 0.5], seed=7, weights=True)
    x = df.select(pl.exclude("w")).to_numpy()
    s = spec(4, method=method, n_components=2, lam=lam, weight="w")
    out = run(df, s)
    seen = longhand(x, df["w"].to_numpy(), lam, 3, method)
    worst = 0.0
    compared = 0
    for t in range(0, 3000, 7):
        E, _ = seen[t]
        for j in range(2):
            got = [out[f"pc{j}_loading_x{i}"][t] for i in range(4)]
            if got[0] is None:
                continue
            worst = max(worst, float(np.max(np.abs(np.array(got) - E[j]))))
            compared += 1
    assert compared > 500
    assert worst < 1e-10, worst


def test_the_gate_is_the_test_of_equal_eigenvalues():
    """The two sides of the threshold, read off the bank's own fields: a
    component is written exactly where ``(n - 1) ln(((l + m)/2)^2 / (l m))``
    passes 9.21 against both neighbours, with ``l`` and ``m`` the variances
    the full SVD keeps (every direction: three columns, three components)."""
    df = stream(400, [25.0, 4.0, 1.0], seed=3)
    out = run(df, spec(3, n_components=3))
    n = np.arange(400, dtype=float)
    # Bessel's factor is shared, so the ratio of written variances is the
    # ratio of the kept ones.
    seen_both = 0
    for t in range(3, 400):
        v = [out[f"pc{j}_var"][t] for j in range(3)]
        if None in v:
            continue
        for a, b in [(v[0], v[1]), (v[1], v[2])]:
            stat = (n[t] - 1) * math.log(((a + b) / 2) ** 2 / (a * b))
            assert stat > CRITICAL
        seen_both += 1
    assert seen_both > 300
    # And at least one row is withheld at the start, with its reason.
    assert out["pc0_var"][2] is None
    assert out["withheld_reason"][2] == "not_separated"


def test_a_tie_is_withheld_and_said():
    """Two columns of one spread, independent: their components are tied
    and never told apart; the third, of a larger spread, is written."""
    df = stream(3000, [9.0, 1.0, 1.0], seed=11)
    out = run(df, spec(3, n_components=3))
    late = out.slice(1000)
    assert late["pc0_var"].null_count() == 0
    assert late["pc1_var"].null_count() > 1900
    assert late["pc2_var"].null_count() > 1900
    assert (late["withheld_reason"] == "not_separated").sum() > 1900


# --- what the bank owes every model ------------------------------------------

KINDS = [
    {"method": "svd"},
    {"method": "ccipca"},
    {"method": "oja", "learning_rate": 1.5},
    {"method": "svd", "n_components": 4},
]


@pytest.mark.parametrize("kw", KINDS)
def test_chunk_invariance(kw):
    df = stream(600, [9.0, 4.0, 1.0, 0.5], seed=5, weights=True).with_columns(
        g=pl.Series([i % 2 for i in range(600)])
    )
    s = spec(4, group="g", weight="w", half_life=50.0, **kw)
    one = rp.ModelBank([s]).fit_predict(df)
    for size in (1, 7, 97):
        bank = rp.ModelBank([s])
        many = pl.concat([bank.fit_predict(df.slice(i, size)) for i in range(0, 600, size)])
        assert one.equals(many, null_equal=True), size


@pytest.mark.parametrize("kw", KINDS)
def test_a_save_goes_on_to_the_bit(kw):
    df = stream(500, [9.0, 4.0, 1.0, 0.5], seed=6, weights=True)
    s = spec(4, weight="w", half_life=40.0, **kw)
    for cut in (1, 3, 250, 499):
        a = rp.ModelBank([s])
        a.fit_predict(df.slice(0, cut))
        buf = io.BytesIO()
        a.save(buf)
        b = rp.ModelBank.load_bytes(buf.getvalue(), specs=[s])
        rest = df.slice(cut)
        assert a.fit_predict(rest).equals(b.fit_predict(rest), null_equal=True), cut


@pytest.mark.parametrize("kw", KINDS)
def test_a_zero_weight_row_counts_in_no_clock(kw):
    """The rows of weight above 0 read the same with and without the rows
    of weight 0 between them, to the bit, on a numeric clock with gaps; the
    zero-weight rows are still read."""
    df = stream(400, [9.0, 4.0, 1.0, 0.5], seed=8, weights=True).with_columns(
        t=pl.Series(np.cumsum(np.where(np.arange(400) % 13 == 5, 4.0, 1.0)))
    )
    s = spec(4, weight="w", clock="t", gap_cap=10.0, half_life=30.0, **kw)
    full = run(df, s)
    kept = run(df.filter(pl.col("w") > 0), s)
    assert full.filter(pl.col("w") > 0).equals(kept, null_equal=True)
    zero = full.filter(pl.col("w") == 0)
    assert zero["pc0_score"].null_count() < len(zero)


def test_a_weight_scales_nothing_but_the_weight():
    """Every weight doubled moves no component, share or score: a weight
    is relative. ``weight_sum`` doubles."""
    df = stream(400, [9.0, 4.0, 1.0], seed=9, weights=True)
    s = spec(3, weight="w", half_life=30.0)
    a = run(df, s)
    b = run(df.with_columns(pl.col("w") * 2.0), s)
    for c in a.columns:
        if not c.startswith("pc"):
            continue
        if a[c].dtype == pl.Float64:
            diff = (a[c] - b[c]).abs().max()
            assert diff is None or diff <= 1e-12, c
    assert np.allclose(b["weight_sum"].to_numpy(), 2 * a["weight_sum"].to_numpy())


def test_the_half_life_is_rows_or_a_duration():
    """On a temporal clock a duration half-life reads the same components
    as the same number of units on a numeric clock of the same steps."""
    df = stream(300, [9.0, 4.0, 1.0], seed=12)
    secs = np.cumsum(np.where(np.arange(300) % 17 == 3, 5, 1)).astype(np.int64)
    num = df.with_columns(t=pl.Series(secs.astype(float)))
    tem = df.with_columns(t=pl.Series(secs * 1_000_000).cast(pl.Datetime("us")))
    a = run(num, spec(3, clock="t", gap_cap=10.0, half_life=20.0))
    b = run(tem, spec(3, clock="t", gap_cap="10s", half_life="20s"))
    for c in a.columns:
        if c.startswith("pc"):
            diff = (a[c] - b[c]).abs().max()
            assert diff is None or diff <= 1e-12, c


def test_scores_are_read_before_the_row():
    """A row's score is its deviation from the mean before it, along the
    loadings before it: the row's own value moves neither."""
    df = stream(300, [9.0, 1.0, 0.25], seed=13)
    out = run(df, spec(3))
    x = df.to_numpy()
    for t in (50, 150, 299):
        m = x[:t].mean(axis=0)
        e = np.array([out[f"pc0_loading_x{i}"][t] for i in range(3)])
        assert abs(out["pc0_score"][t] - e @ (x[t] - m)) <= 1e-9


def test_the_signs_keep_their_side():
    """A loading never flips: each row's is within 90 degrees of the row
    before's."""
    df = stream(2000, [9.0, 4.0, 1.0], seed=14)
    for method in ("svd", "ccipca", "oja"):
        out = run(df, spec(3, method=method, half_life=50.0))
        e = out.select([f"pc0_loading_x{i}" for i in range(3)]).drop_nulls().to_numpy()
        assert (np.einsum("ij,ij->i", e[1:], e[:-1]) >= 0).all(), method


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"n_components": 0}, "at least 1"),
        ({"n_components": 4}, "at most the number of features"),
        ({"method": "nope"}, "method must be"),
        ({"learning_rate": 1.0}, "takes no rate"),
        ({"method": "oja", "learning_rate": -1.0}, "above 0"),
        ({"max_state_bytes": 0}, "above 0"),
        ({"max_state_bytes": 100}, "past max_state_bytes"),
        ({"emit_sigma": True}, "emit_sigma"),
    ],
)
def test_refusals(kw, says):
    with pytest.raises(Exception, match=says):
        rp.ModelBank([spec(3, **kw)])


def test_targets_are_refused():
    with pytest.raises(TypeError, match="takes no targets"):
        rp.spec.pca("p", features=["x0"], targets=["x0"])


def test_the_state_is_held_to_its_cap_at_the_call():
    """``8 (2 + 3k + q k + q)`` bytes: 2,000 features and 12 components (13
    kept) fit the default 256 KB; 13 components do not, until the cap is
    raised."""
    cols = [f"c{i}" for i in range(2000)]

    def build(**kw):
        return rp.ModelBank([rp.spec.pca("p", features=cols, half_life=INF, **kw)])

    build(n_components=12)
    with pytest.raises(Exception, match="272128 bytes"):
        build(n_components=13)
    build(n_components=13, max_state_bytes=300_000)


def test_names_match_the_realized_struct():
    df = stream(50, [9.0, 4.0, 1.0], seed=1)
    s = spec(3, n_components=2)
    out = rp.ModelBank([s]).fit_predict(df)
    assert [f.name for f in out.schema["p"].fields] == rp.spec.output_fields(s)
