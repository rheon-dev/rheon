"""Seeded tick data for study 499: no file is read or written.

One trading day is the unit interval [0, 1], cut into a fine grid of
``FINE`` steps (a quarter second over 6.5 hours). Two efficient log-prices
follow a diffusion with a deterministic U-shaped volatility and correlation
``RHO``, so the integrated variance and covariance of the day are known
exactly: ``IV = sum_i sigma_i^2 dt`` on the fine grid (the volatility is
constant inside a fine step). Each observation adds an independent
Gaussian noise of standard deviation ``OMEGA`` (the microstructure noise).

Three tick sets come from one path:

* ``sync``: both assets every ``SYNC_EVERY`` fine steps (one second): a
  regular, synchronous grid of ``n + 1`` prices.
* ``async``: asset A at each fine step with probability ``1 / MEAN_A``
  steps, asset B with ``1 / MEAN_B`` (Poisson-like, independent): the
  non-synchronous pair that refresh time and Hayashi-Yoshida are for.
* ``irregular``: asset A's own irregular ticks alone (the univariate
  irregular-time series).

Stamps are integer nanoseconds from midnight; a stamp occurs at most once
per asset (hfhd's ``refresh_time`` reads only the first of a repeated
stamp, see the write-up's caveats).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FINE = 93_600  # fine steps a day: a quarter second over 6.5 hours
SYNC_EVERY = 4  # the synchronous grid: one second, 23,400 returns
MEAN_A = 8.0  # asset A ticks every 8 fine steps on average (2 s)
MEAN_B = 20.0  # asset B every 20 (5 s)
SIGMA = 0.02  # daily volatility level
RHO = 0.5
OMEGA = 6e-4  # noise standard deviation on the log-price
DAY_NS = 23_400 * 10**9  # 6.5 hours in nanoseconds
OPEN_NS = 34_200 * 10**9  # 09:30


@dataclass
class Day:
    """One simulated day: the truth and the three tick sets."""

    iv: np.ndarray  # 2 x 2 integrated covariance of the efficient prices
    sync_t: np.ndarray  # stamps (int64 ns) of the synchronous grid
    sync_p: np.ndarray  # (n + 1) x 2 noisy log-prices on it
    a_t: np.ndarray  # asset A's stamps
    a_p: np.ndarray  # asset A's noisy log-prices
    b_t: np.ndarray
    b_p: np.ndarray
    omega2: float  # the noise variance used

    def series(self, which: str) -> list[pd.Series]:
        """The tick sets as hfhd takes them: one pd.Series of log-prices per
        asset, on a nanosecond DatetimeIndex."""
        if which == "sync":
            idx = to_index(self.sync_t)
            return [pd.Series(self.sync_p[:, j], index=idx) for j in range(2)]
        if which == "async":
            return [
                pd.Series(self.a_p, index=to_index(self.a_t)),
                pd.Series(self.b_p, index=to_index(self.b_t)),
            ]
        if which == "irregular":
            return [pd.Series(self.a_p, index=to_index(self.a_t))]
        raise ValueError(which)


def to_index(t: np.ndarray) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.to_datetime(t.astype(np.int64), unit="ns"))


def vol_path(u: np.ndarray) -> np.ndarray:
    """The U-shaped intraday volatility at day fraction ``u``: 1.6x the level
    at the open and the close, 0.8x at midday."""
    return SIGMA * (0.8 + 0.8 * (2.0 * u - 1.0) ** 2)


def simulate(seed: int, fine: int = FINE, omega: float = OMEGA) -> Day:
    rng = np.random.default_rng(seed)
    dt = 1.0 / fine
    u = (np.arange(fine) + 0.5) * dt
    sig = vol_path(u)
    z = rng.standard_normal((fine, 2))
    z[:, 1] = RHO * z[:, 0] + np.sqrt(1.0 - RHO**2) * z[:, 1]
    dx = sig[:, None] * np.sqrt(dt) * z
    x = np.vstack([np.zeros((1, 2)), np.cumsum(dx, axis=0)]) + np.log(100.0)
    s2 = float(np.sum(sig**2) * dt)
    iv = s2 * np.array([[1.0, RHO], [RHO, 1.0]])
    stamps = OPEN_NS + np.arange(fine + 1, dtype=np.int64) * (DAY_NS // fine)

    def noisy(rows: np.ndarray, col: int) -> np.ndarray:
        return x[rows, col] + omega * rng.standard_normal(len(rows))

    sync_rows = np.arange(0, fine + 1, SYNC_EVERY)
    sync_p = np.column_stack([noisy(sync_rows, 0), noisy(sync_rows, 1)])
    a_rows = np.flatnonzero(rng.random(fine + 1) < 1.0 / MEAN_A)
    b_rows = np.flatnonzero(rng.random(fine + 1) < 1.0 / MEAN_B)
    return Day(
        iv=iv,
        sync_t=stamps[sync_rows],
        sync_p=sync_p,
        a_t=stamps[a_rows],
        a_p=noisy(a_rows, 0),
        b_t=stamps[b_rows],
        b_p=noisy(b_rows, 1),
        omega2=omega**2,
    )
