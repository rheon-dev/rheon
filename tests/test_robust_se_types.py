"""Heteroskedasticity-consistent standard-error variants (HC0-HC3) on ewridge and rls.

Holds the row-time HC0-HC3 estimators to:
- exact algebraic identity against the sequential leverage recursion on the
  stream's own out-of-sample residuals (to 1e-12);
- exact zero-regression parity for legacy ``emit_robust_se=True`` alone;
- strict ordering HC0 < HC1 < HC2 < HC3 and Kish scaling HC1 / HC0;
- statsmodels ``OLS.fit(cov_type="HC0".."HC3")`` over a 50-seed oracle grid
  recorded in the oracle ledger;
- hard rules: chunk invariance, zero-weight invariance, save/load bit-identity,
  feature-set subsets, HAC alongside HC variants, and ``max_state_bytes`` cap
  enforcement (refusal at k=100 with HAC past 256 KB, warning past half).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import statsmodels.api as sm

import ledger
from rheon import polars as rp

TIER = "mixed"
GRID_SEEDS = range(50)


def _heteroskedastic_frame(seed: int, n: int = 1_500, k: int = 3) -> pl.DataFrame:
    rng = np.random.default_rng(10_512 + seed)
    x = rng.normal(size=(n, k))
    beta = np.linspace(1.5, -0.8, k)
    sigma = 0.4 + 0.8 * np.abs(x[:, 0]) + 0.3 * x[:, 1] ** 2
    y = 0.5 + x @ beta + sigma * rng.normal(size=n)
    cols = {f"x{j}": x[:, j] for j in range(k)}
    cols["y"] = y
    return pl.DataFrame(cols)


def _last_list(col: pl.Series) -> np.ndarray:
    vals = col.drop_nulls()
    assert vals.len() > 0
    return np.asarray(vals[-1].to_list(), dtype=float)


def _run_variants(
    df: pl.DataFrame,
    *,
    kind: str = "ewridge",
    types: list[str] | None = None,
    **kw,
) -> pl.DataFrame:
    features = [c for c in df.columns if c.startswith("x")]
    builder = rp.spec.ewridge if kind == "ewridge" else rp.spec.rls
    extra = {"ridge": 0.0} if kind == "ewridge" else {"delta": 1e-6}
    extra.update(kw)
    if types is not None:
        extra["robust_se_types"] = types
    spec = builder(
        "m",
        targets=["y"],
        features=features,
        half_life=extra.pop("half_life", float("inf")),
        emit_robust_se=True,
        coef_every=0.0,
        **extra,
    )
    return rp.ModelBank([spec]).fit_predict(df).unnest("m")


def test_legacy_emit_robust_se_matches_hc3_to_the_bit():
    """Without ``robust_se_types``, ``se_coef_hc0`` keeps the exact legacy
    ``e^2`` sandwich (identical to ``robust_se_types=["hc3"]``)."""
    df = _heteroskedastic_frame(0, n=400, k=3)
    legacy = _run_variants(df, half_life=80.0, ridge=1e-6)
    hc3 = _run_variants(df, half_life=80.0, ridge=1e-6, types=["hc3"])
    assert "se_coef_hc0" in legacy.columns
    assert "se_coef_hc3" not in legacy.columns
    assert "se_coef_hc3" in hc3.columns
    assert "se_coef_hc0" not in hc3.columns
    assert legacy["se_coef_hc0"].equals(hc3["se_coef_hc3"], null_equal=True)


@pytest.mark.parametrize("kind", ["ewridge", "rls"])
def test_hc0_to_hc3_exact_recursion_and_ordering(kind: str):
    """On a weighted, decayed stream, HC0-HC3 match the exact row-time leverage
    recursion to 1e-12 and satisfy HC0 < HC1 < HC2 < HC3."""
    rng = np.random.default_rng(51201)
    n, k = 300, 3
    x = rng.normal(size=(n, k))
    y = 0.4 + x @ np.array([1.2, -0.7, 0.5]) + (0.5 + np.abs(x[:, 0])) * rng.normal(size=n)
    w = rng.uniform(0.5, 2.0, size=n)
    w[::23] = 0.0
    df = pl.DataFrame({f"x{j}": x[:, j] for j in range(k)} | {"y": y, "w": w})

    hl = 60.0
    lam = 0.5 ** (1.0 / hl)
    out = _run_variants(
        df,
        kind=kind,
        half_life=hl,
        weight="w",
        types=["hc0", "hc1", "hc2", "hc3"],
    )
    # Recompute exact row-time sandwich from the stream's own out-of-sample residuals.
    resids = out["resid_y"].to_numpy()
    settled = out["settled_frac"].to_numpy()
    p = k + 1
    bread = np.zeros((p, p))
    m0 = np.zeros((p, p))
    m2 = np.zeros((p, p))
    m3 = np.zeros((p, p))
    w_sum, q_sum, folded = 0.0, 0.0, 0
    for i in range(n):
        wi = w[i]
        ei = resids[i]
        # Diagnostics age on every learned row with positive weight and fold once ready
        # (>= 95% settled).
        if wi == 0.0:
            continue
        bread *= lam
        m0 *= lam * lam
        m2 *= lam * lam
        m3 *= lam * lam
        w_sum *= lam
        q_sum *= lam * lam
        ready = np.isfinite(settled[i]) and settled[i] >= 0.95 - 1e-15 and np.isfinite(ei)
        if ready:
            z = np.concatenate([[1.0], x[i]])
            if folded >= p:
                g = wi * float(z @ np.linalg.solve(bread, z))
                denom = 1.0 + g
                m0 += (wi * ei / denom) ** 2 * np.outer(z, z)
                m2 += (wi * ei) ** 2 / denom * np.outer(z, z)
            w_sum += wi
            q_sum += wi * wi
            m3 += (wi * ei) ** 2 * np.outer(z, z)
            bread += wi * np.outer(z, z)
            folded += 1

    binv = np.linalg.inv(bread)
    ref_hc0 = np.sqrt(np.diag(binv @ m0 @ binv))
    n_kish = (w_sum * w_sum) / q_sum
    ref_hc1 = ref_hc0 * np.sqrt(n_kish / (n_kish - p))
    ref_hc2 = np.sqrt(np.diag(binv @ m2 @ binv))
    ref_hc3 = np.sqrt(np.diag(binv @ m3 @ binv))

    got = {t: _last_list(out[f"se_coef_{t}"]) for t in ("hc0", "hc1", "hc2", "hc3")}
    np.testing.assert_allclose(got["hc0"], ref_hc0, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(got["hc1"], ref_hc1, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(got["hc2"], ref_hc2, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(got["hc3"], ref_hc3, rtol=1e-11, atol=1e-12)
    assert np.all(got["hc0"] < got["hc1"])
    assert np.all(got["hc0"] < got["hc2"])
    assert np.all(got["hc2"] < got["hc3"])


def test_chunk_invariance_save_load_and_feature_sets():
    """Chunk invariance, save/load bit-identity, feature_sets subset leverage,
    and HAC alongside HC0-HC3."""
    df = _heteroskedastic_frame(3, n=350, k=3)
    spec = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x1", "x2"],
        half_life=50.0,
        feature_sets={"sub": ["x0", "x2"]},
        emit_robust_se=True,
        robust_se_types=["hc0", "hc1", "hc2", "hc3"],
        robust_se_lags=3,
        coef_every=0.0,
    )
    whole = rp.ModelBank([spec]).fit_predict(df).unnest("m")
    # Chunk invariance across small uneven chunks.
    b_chunk = rp.ModelBank([spec])
    parts = [b_chunk.fit_predict(df.slice(i, 37)) for i in range(0, df.height, 37)]
    chunked = pl.concat(parts).unnest("m")
    cols = ["se_coef_hc0", "se_coef_hc1", "se_coef_hc2", "se_coef_hc3", "se_coef_hac"]
    for c in cols:
        assert chunked[c].equals(whole[c], null_equal=True), c

    # Save/load continuation bit-identity.
    b1 = rp.ModelBank([spec])
    p1 = b1.fit_predict(df.slice(0, 200))
    b2 = rp.ModelBank.load_bytes(b1.save_bytes())
    p2 = b2.fit_predict(df.slice(200))
    resumed = pl.concat([p1, p2]).unnest("m")
    for c in cols:
        assert resumed[c].equals(whole[c], null_equal=True), f"save/load {c}"

    # Feature-set subset [x0, x2] must match a standalone 2-feature spec.
    spec_2 = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=["x0", "x2"],
        half_life=50.0,
        emit_robust_se=True,
        robust_se_types=["hc0", "hc1", "hc2", "hc3"],
        coef_every=0.0,
    )
    out_2 = rp.ModelBank([spec_2]).fit_predict(df).unnest("m")
    for t in ("hc0", "hc1", "hc2", "hc3"):
        v_sub = _last_list(whole[f"se_coef_{t}"])
        v_2 = _last_list(out_2[f"se_coef_{t}"])
        assert np.isnan(v_sub[2])
        np.testing.assert_allclose([v_sub[0], v_sub[1], v_sub[3]], v_2, rtol=1e-12, atol=1e-13)


def test_max_state_bytes_cap_refusal_warning_and_raise():
    """At k=100 with HAC lags, robust SE state exceeds the 256 KB default and
    is refused; raising max_state_bytes allows it; crossing half warns once."""
    f100 = [f"x{i}" for i in range(100)]
    # k=100, intercept=True -> d=101; 1 HAC lag alone is > 800 KB -> refused under default 256 KB.
    with pytest.raises(ValueError, match="past max_state_bytes of 262144"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=f100,
            half_life=50.0,
            emit_robust_se=True,
            robust_se_types=["hc0"],
            robust_se_lags=1,
        )
    # Raising max_state_bytes permits the same spec.
    spec_big = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=f100,
        half_life=50.0,
        emit_robust_se=True,
        robust_se_types=["hc0"],
        robust_se_lags=1,
        max_state_bytes=2 * 1024 * 1024,
    )
    assert spec_big["max_state_bytes"] == 2 * 1024 * 1024

    # k=100, lags=0, HC0-HC3 is ~286 KB -> refused at 256 KB; at 350 KB it is past half -> warns!
    with pytest.raises(ValueError, match="past max_state_bytes of 262144"):
        rp.spec.ewridge(
            "m",
            targets=["y"],
            features=f100,
            half_life=50.0,
            emit_robust_se=True,
            robust_se_types=["hc0", "hc1", "hc2", "hc3"],
        )
    spec_half = rp.spec.ewridge(
        "m",
        targets=["y"],
        features=f100,
        half_life=50.0,
        emit_robust_se=True,
        robust_se_types=["hc0", "hc1", "hc2", "hc3"],
        max_state_bytes=400_000,
    )
    df100 = pl.DataFrame({c: [0.1, 0.2] for c in f100} | {"y": [1.0, 2.0]})
    with pytest.warns(rp.ReadinessWarning, match="past half of max_state_bytes"):
        rp.ModelBank([spec_half]).fit_predict(df100)


def test_robust_se_types_and_max_state_bytes_refusals():
    base = dict(targets=["y"], features=["x0", "x1"], half_life=50.0)
    with pytest.raises(ValueError, match="emit_robust_se"):
        rp.spec.ewridge("m", **base, robust_se_types=["hc0"])
    with pytest.raises(ValueError, match="empty"):
        rp.spec.ewridge("m", **base, emit_robust_se=True, robust_se_types=[])
    with pytest.raises(ValueError, match="more than once"):
        rp.spec.ewridge("m", **base, emit_robust_se=True, robust_se_types=["hc1", "hc1"])
    with pytest.raises(ValueError, match='"hc0"'):
        rp.spec.ewridge("m", **base, emit_robust_se=True, robust_se_types=["hc4"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.ewridge("m", **base, max_state_bytes=100_000)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.ewridge("m", **base, emit_robust_se=True, max_state_bytes=0)
    with pytest.raises(ValueError, match="64 MiB"):
        rp.spec.ewridge("m", **base, emit_robust_se=True, max_state_bytes=64 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="max_state_bytes"):
        rp.spec.lasso("m", **base, penalties=[0.1], max_state_bytes=100_000)


# ---------------------------------------------------------------------------
# 50-seed oracle grid + ledger guard against statsmodels HC0-HC3
# ---------------------------------------------------------------------------


def _grid_rheon_outputs() -> list[tuple[list[float], list[float], list[float], list[float]]]:
    out = []
    for seed in GRID_SEEDS:
        df = _heteroskedastic_frame(seed, n=3_000, k=3)
        res = _run_variants(df, types=["hc0", "hc1", "hc2", "hc3"])
        out.append(
            (
                _last_list(res["se_coef_hc0"]).tolist(),
                _last_list(res["se_coef_hc1"]).tolist(),
                _last_list(res["se_coef_hc2"]).tolist(),
                _last_list(res["se_coef_hc3"]).tolist(),
            )
        )
    return out


ledger.guard(
    "robust_se_hc0_hc3_oracle",
    _grid_rheon_outputs,
    oracles=["statsmodels"],
    bits=52,
)


@pytest.mark.oracle_grid(
    reason="50 seeds of 3,000 heteroskedastic rows comparing streaming HC0-HC3 to statsmodels OLS"
)
def test_robust_se_hc0_hc3_matches_statsmodels_over_50_seeds():
    rheon_vals = _grid_rheon_outputs()
    errs = {t: [] for t in ("hc0", "hc1", "hc2", "hc3")}
    for seed, (g0, g1, g2, g3) in zip(GRID_SEEDS, rheon_vals, strict=True):
        df = _heteroskedastic_frame(seed, n=3_000, k=3)
        x = sm.add_constant(df.select(["x0", "x1", "x2"]).to_numpy())
        y = df["y"].to_numpy()
        for name, got in (("hc0", g0), ("hc1", g1), ("hc2", g2), ("hc3", g3)):
            want = sm.OLS(y, x).fit(cov_type=name.upper()).bse
            rel = float(np.max(np.abs(np.asarray(got) - want) / want))
            errs[name].append(rel)

    for name in ("hc0", "hc1", "hc2", "hc3"):
        ledger.record("robust_se_hc0_hc3_oracle", f"{name}_rel_gap", errs[name])
        assert max(errs[name]) < 0.05, f"{name}: max rel gap {max(errs[name])}"
