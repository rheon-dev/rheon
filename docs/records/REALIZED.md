# `realized`: the design note (task 499)

The model, the alternatives weighed and the measurements behind its
choices. The update equations are in `crates/rheon-core/src/realized.rs`
and in [MODELS.md](../MODELS.md#realized--how-much-did-it-move-and-did-it-jump);
the study they follow is `docs/records/STUDY-499-JUMPS.md` on the branch
`vm/study-499b` (commit 64ef06d; R `highfrequency` 1.0.3, `yuima` 1.15.34,
`hfvol` 0.1.0, mpmath), built on branch `vm/task-499-realized-vol`
(2026-11-22, request A22).

## What was built

- **One kind, `realized`, seven fields a target**: `rv`, `bv`, `tv`, `tq`
  and `bns` of the returns before the row; `lm`, the row's own return over
  the volatility of the returns before it, and `lm_jump`, its flag. No
  features, no `coef`. The kind name and the one-kind layout are the
  coordinator's (answers of 2026-11-22).
- **The papers' estimators, no small-sample factor** (Barndorff-Nielsen and
  Shephard 2004, 2006; Lee and Mykland 2008). The study found the
  libraries' factors (`n/(n-1)` on BV in `realized-library`'s default and
  `hf.econometrics`' `adjust`; `n/(n-2)` on TQ in `highfrequency` and
  `realized-library`; `(n-1)/(n-2)` in `mfe-toolbox`'s debiased BV) and
  that the papers have none; a factor of `n` has no meaning on a decayed
  window of weights. `mfe-toolbox`'s TQ is wrong by `n - 2` and
  `realized-library`'s ratio test and Lee and Mykland statistic depart by a
  factor of order `n` or with the returns' scale: neither is an oracle.
- **The BNS test is the adjusted ratio `J` with the tripower quarticity**,
  as the study's engine, positive under jumps (the sign of
  `highfrequency`, `yuima` and the study).
- **Forms**: one pass, a decay (the default form; `half_life`), a window of
  rows (`window_rows`) or of the clock (`window_size`, `closed`); a window
  takes no decay, as `hawkes` and `expectile`. `lm_window = K`, the
  paper's window, is Lee and Mykland's own, independent of the measures'
  form; without it `σ_i` is read from the measures' sums.

## The window's convention: `W` returns hold `W - 1` products

A product (a triple) is in the window only while every return in it is, so
`W` returns hold `W - 1` products and `W - 2` triples, as the papers' sums
over `W` returns do (the coordinator's ruling; the study's prototype held
`W` products, the first reaching outside the window). Each target keeps a
ring of its terms in the window, each with how many of its products were
taken; when its oldest return leaves, its own term, the next return's
product and the second return's triple leave with it. The window's measures
are then the papers' on its returns, to the rounding of the terms (the
tests: RV 0, BV 0, TV 10, TQ 21 ulps, `bns` 2.1 in units of `cond · ε`, on
windows of 50 to 1,000 rows and of clock units at both edges).

## The count `n`

`TQ`, `bns` and the flag read the count of returns the measures are of: the
returns a window holds, or Kish's effective count `(Σ w)² / Σ w²` of the
returns' decayed weights (the coordinator's ruling, 2026-11-22; the study
used the returns' weight `Σ w`). At equal weights without a decay it is the
count itself, exactly; under a decay it is the number of returns the
weights are worth, about `(1 + λ)/(1 − λ)` at unit weights, and it does not
scale with the weights, which the sums do. It costs one more compensated
sum a target, the decayed squared weights (decayed by `f²`), 16 bytes.

## Lee and Mykland's flag

The flag is the paper's test at confidence `β` (`lm_beta`, 0.99, a test at
1%): `(|L| − C_n)/S_n > β*`, `β* = −ln(−ln β)` (the coordinator's reading,
confirmed 2026-11-22). `n` is the target's own count above.

**The flag's `n`, decided (the coordinator, 2026-11-22): the returns.**
The paper's `n` is the number of returns of the testing period the maximum
is taken over; `realized` reads the count of returns the measures are of
(a window's, or the decayed count), which on a one pass is the returns
before the row. `highfrequency`'s `intradayJumpTest` takes `N` as the number
of *prices* of the day, `n + 1`, for every row. On the oracle's days the two
thresholds differ by `C_{n+1} − C_i`: at the end of a day of 390 returns, at
`β = 0.95`, 3.8905 against 3.8899 in the `c L` scale. The flag keeps the
returns; the R test compares every row against `intradayJumpTest` by
applying its `N = n + 1` to `realized`'s statistic, which it holds to 8
ulps (measured 3): the flags are the same on every row. `realized`'s own
flag is held to them wherever the two thresholds agree.

## Compensated sums: TwoSum, not Kahan

Every running sum is a pair, its high part and what that leaves out. The
study measured plain sums at 40 to 132 ulps over 23,400 rows (a random walk
of the roundings) against 0 to 5 compensated. The first build used the
library's Kahan step (`crate::comp::add`), which is exact only while the
step is smaller than the sum: a window taking back a jump larger than the
rest of its sum lost 7 to 58 ulps. Knuth's TwoSum finds the rounding
whatever the sizes and brought the windows to the rounding of their terms
(above). When a target's ring empties its sums are set to exactly 0, so
their rounding is that of one window, not of every row the stream dropped.
Under a decay the error is the decay's multiplication a row, as in every
decaying model: RV 21, BV 17, TV 14, TQ 32 ulps against the definition with
each term's age taken in one power.

The snapshot ring the other windowed models keep was not used: as the
difference of two cumulative sums it lost 3 to 4 digits on an undecayed
window (up to 14,000 ulps of TQ at 23,400 rows over 100, the study's §6.4).

## The cap

A group's state is checked against `max_state_bytes` (256 KB by default, at
most 64 MiB), through the library's `clock_span` rules:

| form | bytes |
|---|---|
| sums, any decay | `8 + 144 m` for `m` targets |
| `lm_window = K` | `16 (K − 2) + 32` a target more |
| window of rows | `32`, and `16 + 56 m` a row (its sequence and weight; each target's sequence, weight, four terms and links) |
| window of the clock | as a window of rows, and a 32-byte stamp a row |

A window of rows is fixed by its settings, so it is checked when the spec is
built and refused naming the largest window that fits (2,032 rows for two
targets with `lm_window = 50` at 256 KB). A window of the clock holds the
rows its span has, so it is checked as the rows go in: the first row past
half of the cap raises a warning, once a group; the row that would pass it
is not learned, nor any after it, and the bank refuses the run naming the
group and the bytes.

## Zero weights, gaps and sessions

A row of weight 0 counts in no clock (hard rule 9): the stream hands it to
no model, so it is in no product, no window and no weight, and the next row
steps from the last that counted; it is still read. A null return is that
target's alone, its products reaching across the row to its last return. A
gap past `gap_cap` or a session's edge is a step of the clock like any
other, its products reaching across it, so a session's gap is the clock
rewritten and a capped gap is the step cut to the cap, as for every kind
whose outputs are not a block; `session_gap="reset"` measures each
session on its own. (A first build cleared the last returns at those two
events, `clear_lags`; that broke both identities, which the bank's
session and cap tests hold every kind to, and was dropped.)

## The oracles and their gaps

| quantity | oracle | convention divided out | gap measured |
|---|---|---|---|
| RV, BV | `highfrequency` `rRVar`, `rBPCov`; `hfvol` | none | ≤ 4 and 8 ulps (bounds) |
| TV | `highfrequency` `rMPVar(m = 3, p = 2)` | `N / (N − 2)` | ≤ 64 ulps (bound) |
| TQ | `highfrequency` `rTPQuar` | `N / (N − 2)` | ≤ 64 ulps (bound) |
| `bns` | `highfrequency` `BNSjumpTest(ratio, max = TRUE, TP)` | `N / (N − 2)` on TQ | ≤ 64 units of `cond · ε` |
| `bns` | `yuima` `bns.test(r = 4/3 ×3, ratio, adj = TRUE)` | `sqrt((n + 1) / n)` | to about 1e-14, relative |
| `lm` | `highfrequency` `intradayJumpTest(dontIncludeLast = TRUE)` | `c = sqrt(2/π)` | 3 ulps (bound 8) |
| every field | the papers in numpy, `math.fsum` | none | as the core tests above |

The Rust tests hold every form to the papers' definitions in double-double
(`realized/paper.rs`); the Python tests to the same definitions in numpy on
the simulated days, Binance's BTCUSDT minutes and ETCUSDT trades in 10
seconds; the R tests run only in the `r-oracles` job (`-m r_oracle`).
