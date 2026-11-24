"""Shrinkage on ``ew_cov`` and ``ew_class`` held to scikit-learn's
``LedoitWolf``, ``OAS``, ``ShrunkCovariance`` and
``LinearDiscriminantAnalysis``, to scipy's Gaussian density, and, under decay
and unequal weights, to the papers' definitions written two-pass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
import pytest

from rheon import polars as rp

TIER = "essential"


def _shrunk_rows(n: int, p: int, seed: int) -> np.ndarray:
    """``n`` rows of ``p`` correlated features of unequal spread about 3."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.0, (n, p))
    x = z + 0.6 * np.roll(z, 1, axis=1)
    return 3.0 + x * (1.0 + 0.5 * np.arange(p))


def _shrunk_out(spec: dict[str, Any], x: np.ndarray, w: np.ndarray | None = None) -> Any:
    cols = {f"x{i}": x[:, i] for i in range(x.shape[1])}
    if w is not None:
        cols["w"] = w
    return rp.ModelBank([spec]).fit_predict(pl.DataFrame(cols))["m"].struct.unnest()


def _shrunk_matrix(out: Any, i: int, p: int) -> np.ndarray:
    """Row ``i``'s ``var`` and ``cov`` fields as a matrix."""
    m = np.empty((p, p))
    for a in range(p):
        m[a, a] = out[f"var_x{a}"][i]
        for b in range(a + 1, p):
            m[a, b] = m[b, a] = out[f"cov_x{a}_x{b}"][i]
    return m


def _paper_lw(z: np.ndarray, omega: np.ndarray, n: float) -> float:
    """Ledoit and Wolf (2004), section 3.3, weighted: ``S = Σ ω z z'``, the
    paper's norm ``|A|^2 = tr(A A') / p``, ``m = <S, I>``, ``d^2 = |S - m I|^2``,
    ``b̄^2 = (1/n) Σ ω |z z' - S|^2`` (the paper's ``(1/n^2) Σ`` at
    ``ω = 1/n``), ``b^2 = min(b̄^2, d^2)``, ``s = b^2 / d^2``."""
    p = z.shape[1]
    s = (omega[:, None, None] * z[:, :, None] * z[:, None, :]).sum(0)
    m = np.trace(s) / p
    d2 = ((s - m * np.eye(p)) ** 2).sum() / p
    outer = z[:, :, None] * z[:, None, :] - s
    b_bar2 = (omega * (outer**2).sum(axis=(1, 2)) / p).sum() / n
    return min(b_bar2, d2) / d2 if d2 > 0 else 0.0


def _paper_oas(s: np.ndarray, n: float) -> float:
    """Chen, Wiesel, Eldar and Hero (2010), Eq. 23, without its ``2/p``
    terms, as scikit-learn documents it."""
    p = s.shape[0]
    tr, tr_s2 = np.trace(s), (s**2).sum()
    den = (n + 1.0) * (tr_s2 - tr**2 / p)
    return 1.0 if den == 0 else min((tr_s2 + tr**2) / den, 1.0)


class TestShrinkageIsScikitLearns:
    """``ew_cov(shrinkage=...)`` against scikit-learn's ``LedoitWolf``,
    ``OAS`` and ``ShrunkCovariance`` on the rows read, at unit weights with
    no decay (``bias=True``, whose divisor ``n`` is theirs); and, under decay
    and unequal weights, against the papers' definitions written two-pass
    at Kish's ``n``, which scikit-learn, having no sample weights here,
    cannot give.

    Tolerances, measured on these streams on Linux x86-64: the intensity
    agrees to 4.4e-16 at worst, each shrunk entry to 1.6e-15 of the
    matrix's largest (the window's, 1.0e-15 unwindowed, 8.6e-16 weighted),
    and the means to 7.0e-16 relative. Each bound is a few hundred to a
    thousand times its worst: a one-pass sum against scikit-learn's and
    NumPy's two-pass ones, whose last bits follow the CPU's paths."""

    P = 5

    def _est(self, kind: str):
        from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance

        return {
            "ledoit_wolf": LedoitWolf(),
            "oas": OAS(),
            "fixed": ShrunkCovariance(shrinkage=0.3),
        }[kind]

    @pytest.mark.parametrize("kind", ["ledoit_wolf", "oas", "fixed"])
    def test_unit_weights_are_scikit_learns_estimators(self, kind):
        p, n = self.P, 120
        x = _shrunk_rows(n, p, 21)
        spec = rp.spec.ew_cov(
            "m",
            features=[f"x{i}" for i in range(p)],
            stats=["mean", "var", "cov", "shrinkage"],
            half_life=float("inf"),
            min_weight=0,
            bias=True,
            shrinkage=0.3 if kind == "fixed" else kind,
        )
        out = _shrunk_out(spec, x)
        for i in [3, 4, 6, 10, 25, 60, n - 1]:
            est = self._est(kind).fit(x[:i])
            got = _shrunk_matrix(out, i, p)
            scale = np.abs(est.covariance_).max()
            np.testing.assert_allclose(got, est.covariance_, rtol=0, atol=1e-12 * scale)
            np.testing.assert_allclose(
                [out[f"mean_x{a}"][i] for a in range(p)], est.location_, rtol=1e-13
            )
            want_s = 0.3 if kind == "fixed" else est.shrinkage_
            assert abs(out["shrinkage"][i] - want_s) < 1e-13, (i, out["shrinkage"][i], want_s)
            if kind == "ledoit_wolf":
                from sklearn.covariance import ledoit_wolf_shrinkage

                assert abs(out["shrinkage"][i] - ledoit_wolf_shrinkage(x[:i])) < 1e-13

    def test_bias_false_multiplies_the_shrunk_matrix_and_not_the_intensity(self):
        from sklearn.covariance import OAS

        p, n = self.P, 80
        x = _shrunk_rows(n, p, 22)
        spec = rp.spec.ew_cov(
            "m",
            features=[f"x{i}" for i in range(p)],
            stats=["var", "cov", "shrinkage"],
            half_life=float("inf"),
            min_weight=0,
            shrinkage="oas",
        )
        out = _shrunk_out(spec, x)
        for i in [5, 30, n - 1]:
            est = OAS().fit(x[:i])
            want = est.covariance_ * i / (i - 1)
            got = _shrunk_matrix(out, i, p)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())
            assert abs(out["shrinkage"][i] - est.shrinkage_) < 1e-13

    @pytest.mark.parametrize("kind", ["oas", "fixed"])
    def test_a_window_at_unit_weights_is_scikit_learn_on_its_rows(self, kind):
        p, n, window = self.P, 150, 40.0
        x = _shrunk_rows(n, p, 23)
        spec = rp.spec.ew_cov(
            "m",
            features=[f"x{i}" for i in range(p)],
            stats=["var", "cov", "shrinkage"],
            half_life=float("inf"),
            min_weight=0,
            bias=True,
            window_size=window,
            shrinkage=0.3 if kind == "fixed" else kind,
        )
        out = _shrunk_out(spec, x)
        for i in [10, 39, 40, 41, 75, n - 1]:
            # The rows inside on row i: ages from row i - 1 below the window.
            rows = x[:i][(i - 1) - np.arange(i) < window]
            est = self._est(kind).fit(rows)
            got = _shrunk_matrix(out, i, p)
            np.testing.assert_allclose(
                got, est.covariance_, rtol=0, atol=1e-12 * np.abs(est.covariance_).max()
            )
            want_s = 0.3 if kind == "fixed" else est.shrinkage_
            assert abs(out["shrinkage"][i] - want_s) < 1e-13, (i, out["shrinkage"][i], want_s)

    @pytest.mark.parametrize("kind", ["ledoit_wolf", "oas"])
    def test_decay_and_weights_are_the_papers_definitions_at_kishs_n(self, kind):
        p, n, half_life = self.P, 300, 30.0
        x = _shrunk_rows(n, p, 24)
        w = np.random.default_rng(25).uniform(0.2, 2.0, n)
        spec = rp.spec.ew_cov(
            "m",
            features=[f"x{i}" for i in range(p)],
            stats=["var", "cov", "shrinkage"],
            half_life=half_life,
            min_weight=0,
            bias=True,
            weight="w",
            shrinkage=kind,
        )
        out = _shrunk_out(spec, x, w)
        for i in [20, 100, n - 1]:
            raw = w[:i] * 0.5 ** (((i - 1) - np.arange(i)) / half_life)
            omega = raw / raw.sum()
            kish = raw.sum() ** 2 / (raw**2).sum()
            m = omega @ x[:i]
            z = x[:i] - m
            s = (omega[:, None, None] * z[:, :, None] * z[:, None, :]).sum(0)
            want_s = _paper_lw(z, omega, kish) if kind == "ledoit_wolf" else _paper_oas(s, kish)
            assert abs(out["shrinkage"][i] - want_s) < 1e-13, (i, out["shrinkage"][i], want_s)
            mu = np.trace(s) / p
            want = (1 - want_s) * s + want_s * mu * np.eye(p)
            got = _shrunk_matrix(out, i, p)
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-12 * np.abs(want).max())

    def test_a_given_target_is_the_mixture(self):
        p, n = 3, 60
        x = _shrunk_rows(n, p, 26)
        target = [[2.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 3.0]]
        spec = rp.spec.ew_cov(
            "m",
            features=[f"x{i}" for i in range(p)],
            stats=["var", "cov"],
            half_life=float("inf"),
            min_weight=0,
            bias=True,
            shrinkage=0.25,
            shrinkage_target=target,
        )
        out = _shrunk_out(spec, x)
        for i in [5, n - 1]:
            s = np.cov(x[:i].T, ddof=0)
            want = 0.75 * s + 0.25 * np.array(target)
            np.testing.assert_allclose(_shrunk_matrix(out, i, p), want, rtol=0, atol=1e-12 * 4)


class TestClassShrinkageIsScikitLearns:
    """``ew_class(shrinkage=...)`` against scikit-learn at unit weights and
    no decay, the class priors the class proportions: ``tied`` is
    ``LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=...)``,
    which shrinks each class's covariance and pools them by the priors, and
    ``full`` is each class's ``LedoitWolf`` or ``OAS`` covariance through
    ``scipy.stats.multivariate_normal.logpdf`` plus ``ln pi_c``, softmaxed.
    ``precision_prior`` is required and added after shrinkage: at 1e-12 it
    moves the posteriors by under 1e-12 here.

    Measured on these streams on Linux x86-64, the posteriors agree to
    2.1e-15 at worst; the bound, 1e-10, leaves room for the ridge and for
    scikit-learn's ``lstsq`` against the model's Cholesky on another CPU's
    paths."""

    P, NC = 4, 3

    def _rows(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = np.arange(n) % self.NC
        z = rng.normal(0.0, 1.0, (n, self.P))
        x = z + 0.5 * np.roll(z, 1, axis=1) + y[:, None] * 0.8
        return x, y

    def _out(self, x: np.ndarray, y: np.ndarray, shape: str, kind: str) -> Any:
        spec = rp.spec.ew_class(
            "m",
            features=[f"x{i}" for i in range(self.P)],
            label="c",
            classes=[str(c) for c in range(self.NC)],
            covariance_type=shape,
            precision_prior=1e-12,
            half_life=float("inf"),
            min_weight=0,
            shrinkage=0.3 if kind == "fixed" else kind,
        )
        cols = {f"x{i}": x[:, i] for i in range(self.P)}
        cols["c"] = [str(v) for v in y]
        return rp.ModelBank([spec]).fit_predict(pl.DataFrame(cols))["m"].struct.unnest()

    def _est(self, kind: str):
        from sklearn.covariance import OAS, LedoitWolf, ShrunkCovariance

        return {
            "ledoit_wolf": LedoitWolf(),
            "oas": OAS(),
            "fixed": ShrunkCovariance(shrinkage=0.3),
        }[kind]

    @pytest.mark.parametrize("kind", ["ledoit_wolf", "oas", "fixed"])
    def test_tied_is_lda_with_the_estimator(self, kind):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

        x, y = self._rows(240, 31)
        out = self._out(x, y, "tied", kind)
        for i in [30, 31, 90, 239]:
            lda = LinearDiscriminantAnalysis(solver="lsqr", covariance_estimator=self._est(kind))
            want = lda.fit(x[:i], y[:i]).predict_proba(x[i : i + 1])[0]
            got = [out[f"p_{c}"][i] for c in range(self.NC)]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)

    @pytest.mark.parametrize("kind", ["ledoit_wolf", "oas", "fixed"])
    def test_full_is_each_class_estimator_through_scipy(self, kind):
        from scipy.stats import multivariate_normal

        x, y = self._rows(240, 32)
        out = self._out(x, y, "full", kind)
        for i in [30, 31, 90, 239]:
            ell = []
            for c in range(self.NC):
                xc = x[:i][y[:i] == c]
                cov = self._est(kind).fit(xc).covariance_
                prior = len(xc) / i
                ell.append(np.log(prior) + multivariate_normal(xc.mean(0), cov).logpdf(x[i]))
            ell = np.array(ell)
            want = np.exp(ell - ell.max()) / np.exp(ell - ell.max()).sum()
            got = [out[f"p_{c}"][i] for c in range(self.NC)]
            np.testing.assert_allclose(got, want, rtol=0, atol=1e-10)
