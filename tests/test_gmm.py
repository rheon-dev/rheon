"""``gmm`` through the bank: its start's withheld reasons by name, the fit a
separated pair starts at, chunk invariance, its refusals, every row after
the start held to the weighted EM step under decay written out in numpy, and
the fit of 50 seeds held to bands about scikit-learn's ``GaussianMixture``
(each type's one-seed comparison is in ``tests/test_second_opinion_gmm.py``).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp
from test_second_opinion_gmm import ours, theirs
from test_second_opinion_gmm import pair as sk_pair

TIER = "mixed"

INF = float("inf")


def pair(n: int, seed: int = 1) -> pl.DataFrame:
    """Two round clusters, 65 and 35 percent of the rows, at (4, 3) and the
    origin, spread 0.5."""
    rng = np.random.default_rng(seed)
    far = rng.random(n) < 0.65
    x = rng.standard_normal((n, 2)) * 0.5 + np.where(far[:, None], [4.0, 3.0], [0.0, 0.0])
    return pl.DataFrame(x, schema=["a", "b"])


def one(n: int, seed: int = 2) -> pl.DataFrame:
    """One round cluster."""
    x = np.random.default_rng(seed).standard_normal((n, 2))
    return pl.DataFrame(x, schema=["a", "b"])


def fields(df: pl.DataFrame, **kw) -> pl.DataFrame:
    spec = rp.spec.gmm("g", features=["a", "b"], half_life=INF, **kw)
    return rp.ModelBank([spec]).fit_predict(df)["g"].struct.unnest()


def test_a_separated_pair_starts_and_reads_its_components():
    out = fields(pair(2000), n_components=2)
    started = out["cluster"].is_not_null()
    first = int(started.arg_true()[0])
    # The start is tried at each doubling of the buffer, and never forced.
    assert first & (first - 1) == 0, first
    assert out["withheld_reason"][0] == "below_warm_rows"
    assert out["withheld_reason"][first] is None
    late = out[first:]
    total = late["membership_0"] + late["membership_1"]
    assert np.allclose(total.to_numpy(), 1.0)
    # The larger component first: most rows read component 0.
    assert (late["cluster"] == 0).mean() > 0.55


def test_one_cluster_never_starts_two_components_and_says_why():
    out = fields(one(1000), n_components=2)
    assert out["cluster"].is_null().all()
    # A second component on one cluster is either too light to identify or
    # not told apart from the first, by the try.
    assert out["withheld_reason"][-1] in ("not_identified", "not_separated")


def test_a_buffer_at_its_bound_is_withheld_by_its_own_reason():
    out = fields(pair(400), n_components=2, max_state_bytes=24 * 10)
    assert out["cluster"].is_null().all()
    assert out["withheld_reason"][-1] == "above_max_state_bytes"


def test_one_chunk_or_many_give_the_same_rows():
    df = pair(600)
    spec = rp.spec.gmm("g", features=["a", "b"], n_components=2, half_life=200.0)
    whole = rp.ModelBank([spec]).fit_predict(df)
    bank = rp.ModelBank([spec])
    parts = pl.concat([bank.fit_predict(df[i : i + 37]) for i in range(0, 600, 37)])
    # `coef` is written on each chunk's last row by default, so it is left
    # out: every field the rows read is the same.
    a, b = (t["g"].struct.unnest().drop("coef") for t in (whole, parts))
    assert a.equals(b, null_equal=True)


@pytest.mark.parametrize(
    ("kw", "says"),
    [
        ({"n_components": 0}, "n_components must be at least 1"),
        ({"n_components": 2, "covariance_type": "box"}, "covariance_type"),
        ({"n_components": 2, "reg_covar": 0.0}, "reg_covar must be"),
        ({"n_components": 2, "delta_bic": -1.0}, "delta_bic must be"),
        ({"n_components": 2, "tol": 0.0}, "tol must be"),
        ({"n_components": 2, "n_init": 0}, "n_init and max_iter"),
        ({"n_components": 2, "max_state_bytes": 0}, "max_state_bytes must be"),
    ],
)
def test_each_setting_is_refused_by_name(kw, says):
    with pytest.raises(Exception, match=says):
        rp.ModelBank([rp.spec.gmm("g", features=["a", "b"], half_life=INF, **kw)])


def test_it_takes_no_targets():
    with pytest.raises(TypeError, match="targets"):
        rp.spec.gmm("g", targets=["a"], features=["a", "b"], n_components=2)  # type: ignore[call-arg]


# -- the weighted EM step under decay, written out longhand -------------------

H = 50.0
CAP = 10.0
REG = 1e-6


def clocked_pair(n: int, seed: int = 3) -> pl.DataFrame:
    """``pair``'s rows on an irregular clock, with weights from 0.5 to 1.5
    and one row in ten at 0."""
    rng = np.random.default_rng(seed)
    df = pair(n, seed)
    w = rng.uniform(0.5, 1.5, n)
    w[rng.random(n) < 0.1] = 0.0
    t = np.cumsum(rng.integers(1, 4, n)).astype(float)
    return df.with_columns(w=pl.Series(w), t=pl.Series(t))


def longhand(x, w, lam, pi, mu, cov, n, kind):
    """Online EM from the parameters after a row (``cov`` with ``reg_covar``,
    ``n`` the weight then): each row read before it is learned, then the
    mean form at weight ``w r_j`` after the decay ``lam``. Returns each
    row's log-density and the parameters at the end."""
    k = x.shape[1]
    big_w = np.asarray(pi) * n
    mu = np.array(mu, dtype=float)
    c = np.array(cov, dtype=float) - REG * np.eye(k)
    logf = []
    for xi, wi, li in zip(x, w, lam, strict=True):
        s = c + REG * np.eye(k)
        if kind == "diag":
            s = s * np.eye(k)
        logp = np.empty(len(big_w))
        for j in range(len(big_w)):
            d = xi - mu[j]
            sign, ld = np.linalg.slogdet(s[j])
            q = d @ np.linalg.solve(s[j], d)
            logp[j] = np.log(big_w[j] / big_w.sum()) - 0.5 * (k * np.log(2 * np.pi) + ld + q)
        top = logp.max()
        lf = top + np.log(np.exp(logp - top).sum())
        r = np.exp(logp - lf)
        logf.append(lf)
        for j in range(len(big_w)):
            new = li * big_w[j] + wi * r[j]
            if new > 0.0:
                d = xi - mu[j]
                b = wi * r[j] / new
                mu[j] = mu[j] + b * d
                c[j] = (li * big_w[j] / new) * (c[j] + b * np.outer(d, d))
            big_w[j] = new
    return np.array(logf), big_w / big_w.sum(), mu, c + REG * np.eye(k)


def components(row: list[float], kind: str, k: int = 2):
    """A component's weight, mean and covariance (``k x k``) from its
    ``coef`` row."""
    weight, mu = row[0], np.array(row[1 : 1 + k])
    rest = row[1 + k :]
    full = np.array([[rest[0], rest[1]], [rest[1], rest[2]]]) if kind == "full" else None
    return weight, mu, np.diag(rest) if full is None else full


@pytest.mark.parametrize("kind", ["full", "diag"])
def test_each_row_is_the_weighted_em_step_under_decay_written_out(kind: str):
    df = clocked_pair(1500)
    spec = rp.spec.gmm(
        "g", features=["a", "b"], n_components=2, covariance_type=kind, clock="t",
        gap_cap=CAP, half_life=H, weight="w", coef_every=0,
    )  # fmt: skip
    out = rp.ModelBank([spec]).fit_predict(df)["g"].struct.unnest()
    started = out["cluster"].is_not_null().arg_true()
    t0 = int(started[0])
    # The parameters after row t0 - 1, the start, and the weight then.
    rows = out["coef"][t0 - 1].to_list()
    per = 3 + (3 if kind == "full" else 2)
    parts = [components(rows[j * per : (j + 1) * per], kind) for j in range(2)]
    pi = [p[0] for p in parts]
    mu = [p[1] for p in parts]
    cov = [p[2] for p in parts]
    n = out["weight_sum"][t0]
    x = df.select("a", "b").to_numpy()[t0:]
    w = df["w"].to_numpy()[t0:]
    # A zero-weight row leaves the clock where it was: the next row decays
    # across the span from the last row learned, `gap_cap` and all.
    t = df["t"].to_numpy()
    span, last = np.zeros(len(t)), t[0]
    for i, wi in enumerate(df["w"].to_numpy()):
        if wi > 0.0:
            span[i], last = min(t[i] - last, CAP), t[i]
    lam = (0.5 ** (span / H))[t0:]
    logf, pi_end, mu_end, cov_end = longhand(x, w, lam, pi, mu, cov, n, kind)
    got = out["log_density"].to_numpy()[t0:]
    assert np.allclose(got, logf, rtol=1e-9, atol=1e-9), np.abs(got - logf).max()
    last = out["coef"][-1].to_list()
    for j in range(2):
        weight, m, c = components(last[j * per : (j + 1) * per], kind)
        assert abs(weight - pi_end[j]) < 1e-9
        assert np.allclose(m, mu_end[j], rtol=1e-9, atol=1e-9)
        want = cov_end[j] * np.eye(2) if kind == "diag" else cov_end[j]
        assert np.allclose(c, want, rtol=1e-9, atol=1e-9)


# -- scikit-learn's fit over 50 seeds -------------------------------------------

SEEDS = range(50)
#: The worst over the 50 seeds of each type's largest difference from
#: ``GaussianMixture``'s fit of the same 4,000 rows, measured at f192cceb
#: and rounded up by about a third: a weight, a mean coordinate and a
#: covariance entry. ``full``'s is one seed's (11): its start, at 16 rows,
#: gave the narrow component a variance of 0.04 along ``a``, the wide one
#: took the narrow one's far rows, and one pass of 4,000 rows at no decay
#: leaves its mean 0.15 and a covariance entry 0.78 off scikit-learn's,
#: whose fit scores higher (a mean log-density of -3.093 against -3.141).
#: Every other seed of every type is within 0.01 of a covariance entry.
BANDS = {
    "full": (0.023, 0.2, 1.0),
    "diag": (0.0003, 0.0019, 0.009),
    "spherical": (0.0003, 0.0023, 0.0066),
    "tied": (0.0003, 0.0029, 0.0023),
}


@pytest.mark.extended(reason="50 seeds of 4,000 rows, a scikit-learn fit each, four types")
@pytest.mark.parametrize("kind", sorted(BANDS))
def test_fifty_seeds_end_within_the_bands_of_scikit_learns_fit(kind: str):
    worst = np.zeros(3)
    for seed in SEEDS:
        x = sk_pair(4000, seed)
        w, m, c = ours(x, kind)
        tw, tm, tc = theirs(x, kind)
        a, b = np.argsort(m[:, 0]), np.argsort(tm[:, 0])
        worst = np.maximum(
            worst,
            [
                np.abs(w[a] - tw[b]).max(),
                np.abs(m[a] - tm[b]).max(),
                np.abs(c[a] - tc[b]).max(),
            ],
        )
    print(kind, worst)
    assert (worst <= BANDS[kind]).all(), (kind, worst, BANDS[kind])
