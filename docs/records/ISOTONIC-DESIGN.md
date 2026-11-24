<!-- kind: record -->
# `isotonic`: the design note

The bank's `isotonic` kind (PLAN 573; the user, 2026-11-23, from a
production workload report: "we want the building blocks and not the final
business implementation"). Isotonic regression of each target on one
feature, kept online: the monotone calibrator. The model is
`crates/rheon-core/src/isotonic.rs`; the builder is
`python/rheon/_spec_isotonic.py`.

## The fit, and the oracle it is held to

For target `t`, the rows learned so far `(x_i, y_i, w_i)` (the window's,
under a window), each weight times its decay `d_i` (1 without a decay):

```text
f_t = argmin_f  sum_i w_i d_i (y_i - f(x_i))^2   subject to  f non-decreasing in x
                                                 (non-increasing with increasing = False)
pred_t(x) = f_t interpolated linearly between the distinct x it was fitted at,
            read before the row is learned (out of sample)
```

The oracle is scikit-learn's `IsotonicRegression(increasing, y_min, y_max,
out_of_bounds)` fitted with `sample_weight = w_i d_i` on the same rows, and
the model computes what it computes in the order it computes it, so that
the two agree to the bit where the arithmetic allows:

1. Rows of weight 0 are dropped (scikit-learn's mask; here, hard rule 9).
2. Ties in `x` are pooled: weight `W = sum w`, value `(sum w y) / W`
   (scikit-learn's `_make_unique`, `y * w` summed in its sort order).
3. Pool-adjacent-violators over the distinct `x` in order: scipy's
   `optimize.isotonic_regression`, Busing's (2022) Algorithm 1, transcribed
   line for line (`>=` on both violation tests, the block sum `w_prev
   x_prev + w x`, the backward fill); decreasing runs it over the reversed
   sequence and reverses back, as scipy does.
4. Clip to `[y_min, y_max]`.
5. Predict by `numpy.interp` over the distinct `x` and their fitted values
   (scikit-learn's `interp1d` hands a one-dimensional float fit to it):
   the exact value at the last `x`, else `slope (x - x_j) + y_j` on the
   interval `x_j <= x < x_{j+1}`, with numpy's two NaN fallbacks. Outside
   `[x_min, x_max]`: NaN under `out_of_bounds = "nan"`, the end value under
   `"clip"` (`x` clamped first, as scikit-learn clamps it). One distinct `x`
   predicts its value everywhere, as scikit-learn's `len(y) == 1` branch
   does.

Interpolating over every distinct `x`, where scikit-learn first drops the
interior points of a flat run, gives the same bits: a dropped point sits
inside a flat interval, whose slope is 0, so either reading is `y_j`.

### Where the bits cannot agree, and why

Measured on this Mac against scikit-learn 1.9.1 and scipy 1.18.1:

- **scipy's and scikit-learn's arm64 wheels contract `a * b + c` into a
  fused multiply-add.** A transcription that fuses scipy's lines 13, 18 and
  25 (`fma(w_prev, x_prev, w x)`, `fma(w_i, x_i, s)`) and `_make_unique`'s
  `current_y += y * w` matched `IsotonicRegression.predict` to the bit on
  3,000 of 3,000 random fits (ties, weights, both directions, `y_min` and
  `y_max`); the unfused transcription differed in the last bit on about
  half of them. Rust does not contract, and an `x86-64` build without FMA
  cannot, so the model computes the source's unfused arithmetic on every
  platform, and its state and outputs are the same bits on every OS. The
  Python tests hold the bank to a longhand transcription to the bit, and
  scikit-learn to whichever of the fused or the unfused transcription its
  wheel computes, to the bit (so the only difference is the contraction),
  and record the worst ulps between the bank and scikit-learn.
- **Ties under no window** are pooled as they arrive (`W += w`, `S += w y`),
  where scikit-learn sums a tie's `y w` in ascending `y`: the same sum in
  another order, within a few ulps. Under a window each row is held, sorted
  by `(x, y)` with the older row first among equals (`numpy.lexsort` is
  stable), and the tie is summed in scikit-learn's order: bit for bit.
- **Under a decay** a row's weight is `w` times `decay.factor(-(T_i - T0))`,
  `T` the model's clock and `T0` an origin moved forward when the factor
  passes `2^128`, where scikit-learn is handed `w * 0.5^((t - t_i) / h)`:
  each is one `exp2` from the clock, so the weights agree within a few
  ulps, and the fit by as much.
- **`_make_unique` pools `x` within `1e-15` absolute** (`x - current_x >=
  eps`, `eps = finfo.resolution`), so two distinct floats above about 4.5
  that are adjacent, or within 1e-15, are one point to scikit-learn. Here a
  tie is exact equality. The tests draw `x` that are equal or far apart.

## Decisions taken (the rules settle them)

- **Out of sample** (hard rule 2): the row is predicted from the fit of the
  rows before it, then learned.
- **A row of weight 0** learns nothing and counts in no clock (hard rule
  9); it is predicted as any row is, the window read as it would stand.
- **`n_eff`** (hard rule 8): `w_sum <- f w_sum + w` before the row's own
  decay, the window's weight under a window; each target its own weight
  (`target_n_eff_into`) over the rows it was present on.
- **No row count gates the fit.** A fit is identified at `x` once a row has
  been learned at or on both sides of it: the prediction at `x` is NaN
  outside the range of the learned `x` under `out_of_bounds = "nan"`, and
  before any row under either, with the withheld reason `not_identified`
  (the rows do not identify the fit there; a reason of its own would have
  changed every frozen bank fixture's output schema, whose `withheld_reason`
  enum lists the reasons). The kind's default `min_weight` is 0, as for
  every model whose own rule is the gate.
- **Chunk invariance** (hard rule 3) holds by construction: the state is
  the rows or the pooled points, and nothing reads a chunk's edge.
- **A window** is `window_rows` (the last rows of positive weight) or
  `window_size` (the rows of the last span of the clock, under `closed`),
  the library's two spellings. It holds each row exactly, so a row leaving
  takes nothing but itself out, and the fit is rebuilt from the rows left.
  A window and a decay may be given together: the weights are the decay
  factors within the window, scikit-learn's weighted window.
- **The bound** (`max_state_bytes`, default 256 KB, at most 64 MiB): the
  state the data sizes, the distinct `x` pooled without a window and the
  rows inside a `window_size`, is checked as the rows go in, by the
  library's one rule for such a state: the first row past half of the cap
  is said, and a row that would pass the cap is not learned, nor any after
  it, and the bank refuses the run naming the group and the bytes. A
  `window_rows` window's bytes are fixed by the settings and checked at the
  call. A distinct `x` is 32 bytes a target (`x`, the weight, the sum,
  the fitted value); a windowed row 8 bytes, a stamp under `window_size`,
  and 40 bytes a target present.
- **The refit is the whole fit, each learned row**, over the distinct `x`
  (`m` of them): `O(m)` a row, `m` bounded by the cap. It is what makes the
  fit scipy's bits; an update local to the block the row lands in would be
  `O(block)` and agree only within rounding (question 3).

## Questions for the user, with a recommendation

1. **The kind's name and inputs.** `isotonic` (scikit-learn's module
   `sklearn.isotonic`; Polars has no such concept), with exactly one
   feature (the score) and any number of targets, each fitted on its own
   against it. Keywords as scikit-learn's: `increasing` (default true;
   `"auto"` is not offered, since a direction read from the data could
   flip from row to row), `y_min`, `y_max`, `out_of_bounds` (`"nan"`, the
   default, or `"clip"`; `"raise"` has no meaning in a stream).
   *Recommendation*: as built.
2. **`out_of_bounds`' default.** scikit-learn's is `"nan"`, which withholds
   a score beyond any seen; a calibrator usually wants `"clip"`.
   *Recommendation*: keep scikit-learn's default (a withheld row says why),
   and the docs show `"clip"` for a calibrator.
3. **Speed against the bits.** The whole refit costs `O(m)` a learned row
   (the cost section of the CHANGELOG entry gives the measured figures). A
   local update, PAV rerun over the block the new row lands in and merged
   outward, is exact in real arithmetic (a block of the old fit that does
   not hold the new row never splits: its prefix means are at least its
   value and its suffix means at most) and costs about the block's size,
   but its block means are summed in another order than scipy's, so it
   would agree within a few ulps rather than to the bit. *Recommendation*:
   keep the whole refit (exact) now; build the local update if a workload
   needs more than about `m = 2,000` distinct scores at a high row rate.
4. **Fused arithmetic.** Fusing (`f64::mul_add`) would match the arm64
   wheels to the bit and is the same bits on every OS (a correctly rounded
   FMA), but an `x86-64` build without FMA computes it in software, several
   times slower a refit. *Recommendation*: unfused, as built.
5. **The bound past the cap.** Built: the refusal every data-sized state
   keeps. A continuous score under a decay with no window grows one point a
   row until the cap (about 8,000 distinct scores at 256 KB), so such a
   stream must round the score in the query (`pl.col("score").round(3)`
   bounds it at 1,001 points, and the fit is then exactly scikit-learn's on
   the rounded score) or take a window. The alternative is a **block-merge
   rule**: at the cap, merge the two adjacent points of one fitted block
   with the least weight into one point spanning both `x` (no prediction
   moves, since the block's value is unchanged, but a later row can no
   longer split them), falling back to the least-weight adjacent pair when
   every block is one point. It never refuses, and agrees with
   scikit-learn only until the first merge. *Recommendation*: keep the
   refusal, and document rounding; add the merge rule as an option
   (`on_cap = "merge"`) only if a workload asks for an unrounded score
   under a decay.
6. **The fitted curve as a frame.** The thresholds and fitted values (a
   reliability curve) are state; a binding like `ModelBank.marginal` could
   return them per group. *Recommendation*: not now; the prediction at any
   score is what a calibrator reads.
