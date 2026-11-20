# Oracle verification for PLAN tasks 511-525

Read-only check of every oracle library the tasks cite, against current official docs or
source (fetched 2026-11-20). Verdicts: CONFIRMED (name, spelling and meaning as cited),
CORRECTED (the right name or setting is given), MISSING (the library has no such thing; an
alternative is given). "Exact for the comparison" says whether the cited setting makes the
oracle equal to rheon's proposed quantity, or only a reference.

Versions seen: scikit-learn 1.9.1 docs, SciPy 1.18.0 docs, river 0.26.1 source and 0.27.0 docs
(riverml.xyz/latest), filterpy source on master and PyPI 2.0.0 (the PyPI JSON read).

## Task 511. Shrinkage covariance on `ew_cov` and `ew_class`

- scikit-learn `OAS(*, store_precision=True, assume_centered=False)`: CONFIRMED (1.9.1).
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.OAS.html
  Exact for the comparison only if rheon follows sklearn's formula, which is NOT the paper's:
  the docs say it "differs from Eq. 23" of Chen et al. (2010): the `2/p` factor on `tr(S^2)` is
  omitted. Source (`_oas`, 1.9.X `sklearn/covariance/_shrunk_covariance.py`):
  `alpha = mean(S**2)` (= tr(S^2)/p^2 for symmetric S), `mu = tr(S)/p`,
  `num = alpha + mu^2`, `den = (n+1)*(alpha - mu^2/p)`, `shrinkage = 1.0 if den == 0 else min(num/den, 1.0)`;
  `p == 1` returns `S` with shrinkage 0.0. `S` is the biased (divisor n) covariance, centered
  unless `assume_centered=True`. The task must say "OAS as scikit-learn computes it" and
  replicate the omitted factor; an implementation from Eq. 23 will differ at small p.
- scikit-learn `LedoitWolf(*, store_precision=True, assume_centered=False, block_size=1000)`:
  CONFIRMED. `block_size` is memory-only. `shrinkage_` attribute. Source
  (`ledoit_wolf_shrinkage`): divisor n throughout, `X - X.mean(0)` unless `assume_centered`,
  `beta = min(beta, delta)`, `shrinkage = 0 if beta == 0 else beta/delta`; a single feature
  returns 0.0. Target `mu*I`, `mu = tr(S)/p`. Exact in a window and at `half_life=inf`
  (same divisor n). The scan's reduction to `sum||d||^4 - n||S||_F^2` is the same
  term sklearn uses (`X2` sums); not independently re-derived here.
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html
- scikit-learn `ShrunkCovariance(*, store_precision=True, assume_centered=False, shrinkage=0.1)`:
  CONFIRMED. `(1-shrinkage)*S + shrinkage*mu*I`, `mu = tr(S)/p`. Exact.
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ShrunkCovariance.html
- river `LedoitWolfCovariance(fading_factor=0.5)`, `OASCovariance(fading_factor=0.5)`,
  `ShrunkCovariance(fading_factor=0.5, delta=0.1, target="constant_correlation")`: CONFIRMED at
  0.26.1 (`river/covariance/ewa.py`; docs at 0.27.0 show the same signature).
  CORRECTED detail: river's ShrunkCovariance uses `delta` (not `shrinkage`) and defaults to
  `target="constant_correlation"` (other: `"identity"`); only `target="identity"` compares to
  sklearn's `ShrunkCovariance`. River's LW is a heuristic (n ~ 1/fading_factor), its OAS sets
  n = max(1/fading_factor, 2): references only, as the task says. Not exact.
  https://raw.githubusercontent.com/online-ml/river/0.26.1/river/covariance/ewa.py
  https://riverml.xyz/latest/api/covariance/LedoitWolfCovariance/

## Task 517. `ew_cov` outputs: uncentered covariance and a Gaussian log-likelihood

- scikit-learn `assume_centered=True` (on `EmpiricalCovariance`, also OAS, LedoitWolf,
  ShrunkCovariance): CONFIRMED. `empirical_covariance`: `X.T @ X / n` when centered-assumed,
  `np.cov(X.T, bias=1)` otherwise: divisor n. Exact vs an undecayed or windowed uncentered
  second moment `sum w x x^T / sum w` (with sklearn's n, i.e. unit weights).
  https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html
- scipy `scipy.stats.multivariate_normal.logpdf(x, mean=None, cov=1, allow_singular=False)`:
  CONFIRMED (SciPy 1.18.0). Exact per row: one row's `logpdf(x, mean, cov)` equals the
  Gaussian log-density `-0.5*(p*ln(2 pi) + ln|S| + mahal^2)`. Use `allow_singular=True` for a
  rank-deficient early `S`.
  https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html
- The scan's "scikit-learn `score_samples`-style" log-likelihood: MISSING as a per-row method.
  `EmpiricalCovariance` has `score(X_test, y=None)`, which returns the log-likelihood of the
  test set's empirical covariance under the fit, a single average per-sample value
  (`log_likelihood(emp_cov, precision) = (-sum(emp_cov*precision) + logdet(precision) - p*ln(2 pi))/2`),
  and `mahalanobis(X)` (squared distances). No `score_samples`. Use scipy `logpdf` per row
  as the oracle; `score` on a one-row `X_test` equals the per-row log-density only when the
  row is centered at `location_` (it uses `X_test - location_`), which holds, since a
  one-row test covariance is `d d^T`: `-d^T P d` equals `-sum(emp_cov*precision)`. So `score`
  on one row is a valid second check.
  https://raw.githubusercontent.com/scikit-learn/scikit-learn/1.9.X/sklearn/covariance/_empirical_covariance.py

## Task 519. `kalman`: `coef_prior`, innovation log-likelihood and Mahalanobis distance

- filterpy `KalmanFilter` attributes `x`, `log_likelihood`, `mahalanobis`: CONFIRMED
  (master `filterpy/kalman/kalman_filter.py`; PyPI reports 2.0.0 released 2026-10-26).
  Read-only, lazily computed from the last `update()`: `log_likelihood = logpdf(x=y, cov=S)`
  (innovation `y` and its covariance `S = H P H^T + R`, both before the update): exact for
  the proposed innovation log-likelihood. CORRECTED detail: `mahalanobis` is
  `sqrt(y^T S^-1 y)`, the distance, NOT the squared form. The test must square it (or rheon
  emit the distance) before comparing with `mahal`-style squared outputs. `update(z, R=None, H=None)`.
  `x`/`P` are the initial state and covariance for `coef_prior`'s comparison (set `x`, `P`
  before the first update).
  https://raw.githubusercontent.com/rlabbe/filterpy/master/filterpy/kalman/kalman_filter.py
  https://pypi.org/pypi/filterpy/json
