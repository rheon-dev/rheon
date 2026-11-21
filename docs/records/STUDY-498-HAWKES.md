# Study 498: exact online least-squares fit of a Hawkes process

Status: complete (parts 1 and 2). Part 1: the math, two online accumulators
(forms A and B), a batch double sum, tick. Part 2: a third form with `O(k)` work
per event (form C), exponential forgetting, a rolling window, parameter
recovery.

Question: can tick's least-squares fit of a Hawkes process with sum-of-exponential
kernels and given decays be computed exactly online, one event at a time, with
state bounded independently of the number of events? Does it match tick? Can it
forget, or keep only a window?

Answer: yes on every count. The fit is a quadratic form whose Gram matrix and
linear terms update in closed form at each event; the accumulated statistics
equal a batch double sum over pairs of events to about 1e-14 relative, and the
loss and minimiser equal tick's to at most 3.9e-13 relative, once tick's
normalisation (division by the number of events) is applied. The Gram matrix's
event part is a function of the linear terms, so the state is only the linear
terms and the excitations, `O(k)` work per event (section 5). Forgetting at a
rate per unit time keeps that cost and matches an `O(n^2)` longhand to 1.5e-14;
a rolling window matches tick refit on the window's events to 1.2e-13 (loss)
and 4.1e-14 absolute (minimiser), but needs the window's event times as state
(section 7).

Script: `scripts/study_498.py` (numpy; tick and scipy only for the comparison;
`python scripts/study_498.py part2` runs part 2 alone, about 90 s). Data: an
Ogata thinning simulator in the script, seeded numpy generators; nothing is
stored.

## 1. The math

Nodes `i, j = 1..D`, given decays `beta_u`, `u = 1..U`, events `t_k^j` of node `j`.
tick's parametrisation (`ModelHawkesSumExpKernLeastSq`):

    lambda_i(t) = mu_i + sum_j sum_u alpha_iju beta_u sum_{t_k^j < t} exp(-beta_u (t - t_k^j)).

Write the kernel state per node and decay

    R_ju(t) = sum_{t_k^j < t} beta_u exp(-beta_u (t - t_k^j)),

and the feature vector `phi(t) = (1, R_11, ..., R_DU)` of length `p = 1 + k`,
`k = D U`. With `theta_i = (mu_i, alpha_i11, ..., alpha_iDU)`,
`lambda_i(t) = theta_i' phi(t)`: the intensity is linear in the parameters.

tick's least-squares contrast (its docstring) is

    L(theta) = sum_i [ int_0^T lambda_i(t)^2 dt - 2 int_0^T lambda_i(t) dN_i(t) ]
             = sum_i [ theta_i' G theta_i - 2 theta_i' b_i ],

    G   = int_0^T phi(t) phi(t)' dt                 (p x p, one Gram for every node)
    b_i = sum_{events t_k^i of node i} phi(t_k^i -)   (left limit: the event's own
                                                     jump is not in its own intensity)

so `L` is a quadratic form in `theta`, and the unconstrained minimiser is
`theta_i = G^{-1} b_i`, one solve with `D` right-hand sides. (This is the
contrast `int lambda^2 - 2 int lambda dN`, the squared error
`int (lambda - dN/dt)^2` with the parameter-free `int (dN/dt)^2` dropped.)

### Streaming update, form A (gap integrals, O(p^2) per event)

Between events every `R_a` decays: `R_a(t) = R_a(s) e^{-beta_a (t-s)}`. Over a gap
of length `d` with no events, in closed form:

    int 1          = d
    int R_a        = R_a(s) (1 - e^{-beta_a d}) / beta_a
    int R_a R_b    = R_a(s) R_b(s) (1 - e^{-(beta_a+beta_b) d}) / (beta_a + beta_b)

(`1 - e^{-x}` as `-expm1(-x)`). At an event of node `j` at time `t`: decay `R` to
`t`, add `phi(t-)` to `b_j`, then `R_ju += beta_u` for each `u`. Reading at any
time `T` integrates the last gap on a copy. Cost per event `O(p^2)` (the outer
product), `U` distinct exponentials for `R` plus `O(k^2)` for the `expm1` factors
unless they are tabulated per decay pair (`U(U+1)/2` of them).

### Streaming update, form B (jump sums, O(U p) per event)

Between events `d/dt (R_a R_b) = -(beta_a + beta_b) R_a R_b` and
`d/dt R_a = -beta_a R_a`; all `R` start at 0. Integrating over `[0, T]`:

    int_0^T R_a R_b dt = (J_ab - R_a(T) R_b(T)) / (beta_a + beta_b),
    int_0^T R_a dt     = (beta_a N_a - R_a(T)) / beta_a = N_a - R_a(T) / beta_a,

where `N_a` is the event count of `a`'s node and `J_ab` the sum over events of
the jump in `R_a R_b`. An event of node `j` raises `R_a` by `beta_a` only for the
`U` features `a` of node `j`, so with `R` read before the jump

    J_ab += beta_a 1[a in j] R_b + beta_b 1[b in j] R_a + beta_a beta_b 1[a in j] 1[b in j].

That touches the `U` rows and columns of node `j`: `O(U k)` = `O(D U^2)` work and
`U` exponentials (`e^{-beta_u d}` is shared by every node) per event, against
`O((DU)^2)` for form A. `b_j += phi(t-)` is `O(k)`. Reading at `T` decays `R` to
`T` (`U` exponentials) and assembles `G` in `O(k^2)`, without moving the state.
No integral is evaluated per gap; the only libm results in the state are the
`R` decays. `J_ab - R_a(T) R_b(T)` is a difference of positive numbers; it
cancels only when nearly all of `J` is the recent jumps, i.e. `T` within about
`1/beta` of the start, which is when the fit is not yet usable anyway.

Form B is `O(U k)` per event, not `O(k)` (`k` = decays times dimension): `G`
holds the products `R_a R_b` across nodes, and an event of node `j` moves the
`U` rows of `G` that involve node `j`, `k` entries each. No form that keeps `G`
(or `J`) explicitly can do better. Section 5 shows that `J` need not be kept:
form C is `O(k)`.

### State (form B)

`J` symmetric `k(k+1)/2`, `b` `D p`, `R` `k`, counts `D`, last time 1, in f64:

    bytes = 8 [ k(k+1)/2 + D(1+k) + k + D + 1 ],   k = D U.

| D | U | form B bytes | form A bytes (`G` p(p+1)/2, `b`, `R`, t) |
|---|---|---|---|
| 1 | 2 | 80 | 96 |
| 1 | 3 | 120 | 144 |
| 2 | 2 | 216 | 240 |
| 2 | 3 | 352 | 392 |
| 10 | 3 | 6,528 | 6,696 |

Bounded by the shape (`O(D^2 U^2)`), never by the number of events.

## 2. Online equals batch

The batch accumulator in the script computes `G` and `b` from the definitions
as double sums over pairs of events (`int R_a R_b = beta_a beta_b sum_{k,l}
int_{max(t_k, t_l)}^T ...` in closed form), `O(n^2)`, sharing no recursion with
either online form.

| case | events | G: A vs B | G: B vs batch | b: B vs batch | theta: B vs batch |
|---|---|---|---|---|---|
| 1-dim, U=2 (decays 1, 5) | 3,951 | 4.8e-15 | 5.2e-15 | 2.2e-15 | 1.8e-15 |
| 1-dim, U=3 (0.5, 3, 20) | 4,412 | 3.3e-15 | 3.2e-15 | 1.1e-15 | 1.0e-14 |
| 2-dim, U=2 (1, 8) | 5,789 | 1.6e-15 | 4.1e-15 | 2.1e-15 | 1.2e-14 |
| 2-dim, U=3 (0.7, 4, 15) | 3,963 | 8.9e-15 | 8.3e-15 | 8.5e-16 | 5.3e-14 |

(max absolute difference over the max absolute entry; `b` of form A and B are
bit-identical, as they share the `R` recursion.) The online accumulator equals
the batch one to rounding, and so does the minimiser.

## 3. Against tick

tick 0.8.0.2, `ModelHawkesSumExpKernLeastSq(decays).fit(events, end_times=T)`,
coefficients laid out `[mu_1..mu_D, adjacency (D, D, U) raveled]`.

* **Normalisation.** tick's `loss` is the contrast above **divided by the total
  number of events** (`n_jumps`). The ratio ours / tick is `n_jumps` to 12
  digits in every case. With that division, the loss at five random coefficient
  vectors per case agrees to at most 3.9e-13 relative (5.98e-15, 2.27e-14,
  8.15e-14, 3.86e-13). No boundary term differs: tick integrates to the given
  `end_times`, as here. The division does not move the minimiser.
* **Minimiser.** tick's sum-exponential model has no `hessian` (it raises
  `NotImplementedError`), so its exact minimiser is recovered from its own
  gradient: the loss is quadratic, `H e_c = grad(e_c) - grad(0)`, minimiser
  `-H^{-1} grad(0)`. Against ours: max absolute 1.0e-13, max relative 1.9e-13
  (per case 1.3e-14, 1.0e-14, 1.3e-14, 1.9e-13). Within 1e-6 relative with
  seven orders to spare.
* **Learner.** `HawkesSumExpKern(decays, penalty='none', solver='agd',
  tol=1e-13, max_iter=20000)`. Two differences from the model, both
  documented behaviour, not mismatches: (1) its `fit` takes no end time and
  ends the observation at the last event, and (2) even with `penalty='none'`
  it projects onto `theta >= 0` (`ProxPositive`). Compared with our
  non-negative fit (scipy `nnls` on the Cholesky factor of our `G`, read at
  the last event time) it agrees to 8.5e-7, 1.0e-6, 2.3e-6, 2.2e-6 absolute
  (parameters 0.05 to 0.7, so up to about 5e-6 relative). That residual is the
  learner's iterative solver stopping, not the model: the exact minimiser from
  tick's model above agrees to 1e-13. In the 2-dim, U=3 case the unconstrained
  minimiser has entries down to -0.015 (a small true `alpha` split over three
  decays); there the projection matters, and the non-negative fit is what
  matches the learner.

## 4. Cost

Time per event, this machine, in the Python loop of the script (numpy on arrays
of 2 to 7 entries, so interpreter overhead dominates and the figures say
nothing about a compiled build): form B 13 to 21 us, form A 14 to 15 us. Counted
work, which is what a compiled build would pay, per event:

* form B: `U` exponentials, about `U k` multiply-adds into `J` (stored
  symmetric) and `k` into `b` (6 for 1-dim U=2; 24 for 2-dim U=3; 120 for
  D=10, U=3);
* form A: `U` exponentials plus `k^2` products and `k(k+1)/2` `expm1` factors
  (or a table of `U(U+1)/2`): 900 products for D=10, U=3.

A read (assembling `G` and solving) is `O(p^3 / 3 + D p^2)`, paid only when a
fit is wanted, not per event.

## Notes on the tools

* tick 0.8.0.2 installs from wheels on CPython 3.11 to 3.14 but imports
  `numpydoc` and `six` without declaring them. Without `numpydoc` every model
  fails at construction with `object has no settable attribute 'dtype'` (its
  attribute registry is parsed from docstrings); without `six` the learners
  fail building their prox. Installed both into the study's venv (CPython
  3.13, numpy 2.5.5).
* `ModelHawkesExpKernLeastSq` (one decay per pair, `D x D` decays) was not
  run: every case here has two or three decays, which is the sum-exponential
  model. With a single shared decay it is the `U = 1` case of the same
  quadratic form.

## 5. Form C: `O(k)` per event

Form B's `J_ab` is the sum over events of the jump in `R_a R_b`. At an event of
node `j` that jump is `beta_a 1[a in j] R_b(t-) + beta_b 1[b in j] R_a(t-) +
beta_a beta_b 1[a, b in j]`. Summed over node `j`'s events, `sum R_b(t-)` is
exactly `b_j[b]` (the linear term the fit already keeps) and the count is
`b_j[0] = N_j`. So, with `j(a)` the node of feature `a`,

    J_ab = beta_a b_{j(a)}[b] + beta_b b_{j(b)}[a] + beta_a beta_b 1[j(a) = j(b)] N_{j(a)},

    G_00 = T,   G_0a = N_{j(a)} - R_a(T) / beta_a,
    G_ab = (J_ab - R_a(T) R_b(T)) / (beta_a + beta_b).

`J` is not state. The state is `R` (`k`), `b` (`D x p`) and the last event time.
Per event: `U` exponentials (decay `R`, shared across nodes), `b_j += phi(t-)`
(`k + 1` adds, one row of `b`), `R_ju += beta_u` (`U` adds): `O(k)`. The read
assembles `J` in `O(k^2)` and solves, as before. This is the identity behind
"each event costs `O(U k)`": the `U k` entries an event moves in `J` are all
linear in one row of `b`, so moving the row moves them.

Against form B on part 1's four cases: `G` agrees to 8.4e-15 relative at most,
`b` is bit-identical, the minimiser agrees to 6.0e-14. Part 1's checks of form B
against the batch sum and tick therefore carry over.

State, f64, with the scalar `s` that forgetting adds (section 6):

    bytes = 8 [ k + D (1 + k) + 2 ],   k = D U   (O(D^2 U), against form B's O(D^2 U^2))

| D | U | form C bytes | form B bytes |
|---|---|---|---|
| 1 | 2 | 56 | 80 |
| 1 | 3 | 72 | 120 |
| 2 | 2 | 128 | 216 |
| 2 | 3 | 176 | 352 |
| 10 | 3 | 2,736 | 6,528 |

In the script's Python loop form C costs 6 to 9 us per event against form B's
14 to 20 us (interpreter overhead dominates both; the counted work is the point:
`k + 1` adds against `U k`).

## 6. Exponential forgetting

**Choice: forgetting per unit time.** The loss at time `t` and an event at `t`
carry weight `w(t) = e^{-gamma (T - t)}`, `gamma >= 0` (a factor `lambda =
e^{-gamma}` in (0, 1] per unit of time; `lambda = 1` is the unforgotten fit):

    L_gamma(theta) = sum_i [ int_0^T w lambda_i^2 dt - 2 int_0^T w lambda_i dN_i ]
                   = sum_i [ theta_i' G theta_i - 2 theta_i' b_i ],
    G = int_0^T w phi phi' dt,   b_i = sum_{events of i} w(t_k) phi(t_k -).

**Recursion.** Before each new event at gap `d`, everything accumulated decays
by `e^{-gamma d}`; the event then adds `phi(t-)` to `b_j` with weight 1. In form
C only `b` is accumulated, so this is `b <- e^{-gamma d} b; b_j += phi(t-)`. To
keep the per-event cost `O(k)` the decay is not applied to all of `b`: `b = s
bt`, the scalar `s` decays (`s <- s e^{-gamma d}`) and the event adds `phi(t-) /
s` to `bt`; when `s < 2^-500`, `bt` is multiplied by `s` and `s` reset to 1
(once per `500 ln 2 / gamma` time units). At a read at `T`,
`b = bt s e^{-gamma (T - t_last)}`.

**Read.** Integrate `w d(R_a R_b)` by parts. Between events `d(R_a R_b) = -(beta_a
+ beta_b) R_a R_b dt`, and at events it jumps; `int_0^T w d(R_a R_b) = R_a R_b(T)
- gamma int w R_a R_b`. Equating, with `J^gamma` the weighted jump sum (form C's
identity holds unchanged with the weighted `b`):

    G_00 = (1 - e^{-gamma T}) / gamma
    G_0a = (beta_a N_{j(a)} - R_a(T)) / (beta_a - gamma)          (N = b[:, 0], weighted)
    G_ab = (J_ab - R_a(T) R_b(T)) / (beta_a + beta_b - gamma).

With `gamma = 0` this is section 5.

**Forgetting per event** (`lambda` per event, `w = lambda^{N(T) - N(t)}`) was
not taken: its weight steps at each event, so integration by parts leaves
`(w_k - w_{k-1}) R(t_k-) R(t_k-)'` per event, a full outer product: `O(k^2)` per
event, the cost of form A, whose recursion it then is (`G <- lambda (G + gap
integral)`).

**Oracles.** tick's least-squares Hawkes models take no weights (`fit(events,
end_times)`; several realisations are summed with equal weight), so tick cannot
check a forgotten fit. The oracle is a longhand, `batch_stats_weighted`: the
weighted `G` and `b` written from their definitions as double sums over pairs of
events (`O(n^2)`, closed-form integrals of products of exponentials). A second
online recursion, form A with forgetting (`OnlineIntervalForget`: `G` and `b`
decay by `e^{-gamma d}` per gap and each gap's integral is taken with the
weight inside), shares no read formula with form C. On part 1's four cases with
`gamma` = 0.002, 0.01 and 0.3 (effective weight 1 to 951 events of 3,951 to
5,789):

| case | gamma | form C vs longhand: G / b / theta | form A vs longhand: G / b / theta |
|---|---|---|---|
| 1d-U2 | 0.002 (weight 494) | 3.6e-15 / 3.9e-15 / 8.8e-15 | 2.4e-15 / 4.3e-15 / 1.3e-14 |
| 1d-U2 | 0.01 (weight 94) | 4.1e-16 / 4.8e-16 / 1.2e-15 | 2.1e-16 / 6.5e-16 / 4.4e-15 |
| 1d-U2 | 0.3 (weight 1) | 1.2e-16 / 1.8e-16 / 3.1e-16 | 1.2e-16 / 1.8e-16 / 2.4e-15 |
| 1d-U3 | 0.002 (weight 733) | 3.9e-15 / 6.1e-15 / 6.1e-15 | 3.1e-15 / 6.5e-15 / 1.3e-14 |
| 1d-U3 | 0.01 (weight 136) | 9.3e-16 / 3.9e-16 / 5.1e-15 | 1.9e-15 / 6.3e-16 / 8.2e-15 |
| 1d-U3 | 0.3 (weight 5) | 1.2e-16 / 2.0e-16 / 7.9e-15 | 1.2e-16 / 2.0e-16 / 8.7e-15 |
| 2d-U2 | 0.002 (weight 951) | 2.9e-15 / 2.6e-15 / 9.9e-15 | 3.9e-15 / 4.4e-15 / 1.4e-14 |
| 2d-U2 | 0.01 (weight 187) | 1.7e-15 / 1.6e-15 / 6.1e-15 | 1.1e-15 / 1.6e-15 / 4.0e-15 |
| 2d-U2 | 0.3 (weight 4) | 2.2e-16 / 3.7e-16 / 3.3e-15 | 2.9e-16 / 1.9e-16 / 1.0e-15 |
| 2d-U3 | 0.002 (weight 788) | 4.0e-15 / 4.8e-15 / 1.5e-14 | 4.4e-15 / 3.6e-15 / 9.1e-15 |
| 2d-U3 | 0.01 (weight 173) | 4.9e-16 / 5.0e-16 / 6.0e-15 | 6.1e-16 / 8.8e-16 / 7.5e-15 |
| 2d-U3 | 0.3 (weight 11) | 3.4e-16 / 2.8e-16 / 5.2e-15 | 1.7e-16 / 5.5e-16 / 6.1e-15 |

Largest gap of either form to the longhand: 1.5e-14 relative.

## 7. Rolling window

**Definition.** At read time `T`, the window is `[T - W, T]` (`W` time units);
the fit is the least-squares fit of the window's events alone, observed over the
window: excitation from events before `T - W` is dropped, `G_00 = W`. This is
what tick computes when refit on the window's events shifted by `T - W`, with
end time `W`.

**Removal.** Adding an event is form C's update, with `R^w` the excitation of
the window's events. Removing the oldest event `e` (time `t_e`, node `j_e`), at
any time after it leaves the window and before the next event is added, takes
back exactly its contributions:

    b_{j_e}[0]       -= 1                           (its phi(t_e -) is (1, 0, ..): nothing older remains)
    b_i[(j_e, u)]    -= beta_u F_e[i, u]            for every node i and decay u,
                         F_e[i, u] = sum_{later window events l of node i} e^{-beta_u (t_l - t_e)}
    R^w_{(j_e, u)}   -= beta_u e^{-beta_u (t_last - t_e)}.

`J` follows from `b` (section 5), so nothing else moves. In form B, `J` would
lose `beta_a beta_b F_e`-terms over `U k` entries instead.

**What the removal needs.** `F_e` is the pair term of `e` with every later event
in the window. It cannot be carried as one running sum: `sum_l e^{-beta (t_l -
s_0)}` over the window, rescaled as `s_0` moves, multiplies its rounding error
by up to `e^{beta W}`. So the state holds the window's events (time and node,
16 bytes each) besides `R`, `b`, `t`:

    bytes = 8 [ k + D (1 + k) + 1 ] + 16 n_W.

The script computes `F_e` by a direct sum over the window, `O(n_W U)` per
removal. An `O(k)`-amortised version (derived, not run) splits the window in
two: for the older block each event stores its forward sum within the block
(`k` floats, formed once by the backward recursion `F_e = e^{-beta (t_{e+1} -
t_e)} (1[node e+1] + F_{e+1})` when the block is formed); the newer block keeps
one `k`-vector `B = sum_l e^{-beta (t_l - t_m)}` relative to its start `t_m`;
then `F_e = F_e^block + e^{-beta (t_m - t_e)} B`, every factor at most 1. That
stores `k + 2` numbers per window event.

**Bounded?** For a window of `W` events, yes: by `W`. For a window of `W` time
units the state grows with the number of events in the window, which the shape
does not bound (only the stream's intensity does, in distribution). A time
window therefore needs a cap on its event count, as rheon's other windows have.

**Oracle and result.** For each of part 1's cases, `W = 300`, six read times
spread over the stream (269 to 615 events in a window, 3,951 to 5,789 events
streamed, so 10 to 13 full turnovers of the window by the last read): tick's
`ModelHawkesSumExpKernLeastSq` refit on the window's shifted events with
`end_times = W` (its loss at five random coefficient vectors, divided by its
`n_jumps`, and its exact minimiser from its gradient, as in section 3), and the
batch double sum of section 2 on the same events. Largest gaps over 24 reads:
loss 1.2e-13 relative; minimiser 4.1e-14 absolute, 7.9e-14 relative; `G`, `b`
against the batch sum 7.5e-15 relative. The gap does not grow over the stream
(first read 3e-16 to 1e-14, last 6e-15 to 6e-14).

## 8. Parameter recovery

Simulated with the Ogata sampler from known parameters, fitted with form C (the
unconstrained minimiser, the true decay given, read at `T`). One seed per row
(21 for `T = 2,000`, 22 for `T = 20,000`); `alpha` is tick's (the kernel
`alpha beta e^{-beta t}` integrates to `alpha`).

| model | T | events | truth (mu; alpha) | fitted | max abs error | max rel error |
|---|---|---|---|---|---|---|
| 1-dim, beta 2 | 2,000 | 2,268 | 0.5; 0.6 | 0.482; 0.575 | 0.025 | 0.042 |
| 1-dim, beta 2 | 20,000 | 25,342 | 0.5; 0.6 | 0.496; 0.608 | 0.008 | 0.014 |
| 2-dim, beta 3 | 2,000 | 3,439 | (0.3, 0.4); [[0.4, 0.2], [0.1, 0.5]] | (0.270, 0.384); [[0.401, 0.221], [0.123, 0.480]] | 0.030 | 0.227 |
| 2-dim, beta 3 | 20,000 | 36,122 | (0.3, 0.4); [[0.4, 0.2], [0.1, 0.5]] | (0.298, 0.405); [[0.410, 0.194], [0.101, 0.502]] | 0.010 | 0.030 |

The error falls about as `1/sqrt(n)` (a factor 3 for 10 times the events). The
largest relative error at `T = 2,000` is on the smallest parameter (`alpha_21 =
0.1`, fitted 0.123: 0.023 absolute).

## 9. Numerical risks

* **`J - R(T) R(T)'`** (forms B and C): a difference of positive numbers; it
  cancels only when nearly all of `J` is the latest jumps (a read within about
  `1/beta` of the start). Benign in every case here (1e-14).
* **Forgetting near a decay.** Form C divides by `beta_a - gamma` and
  `beta_a + beta_b - gamma`, which vanish when `gamma` equals a decay or a sum
  of two. Measured on the 1-dim case (decays 1 and 5) against form A, whose
  `expm1(-x d) / x` factors have the finite limit `d`: 1.4e-14 at `gamma = 0.99`,
  1.8e-13 at 0.999, 1.8e-9 at `1 - 1e-6`, 9.7e-7 at `1 - 1e-9` (the minimiser,
  relative); `gamma` above every decay (1.5, 3) is exact again (4e-15). About
  `log10(beta / |beta - gamma|)` digits are lost. The intended use, a horizon
  `1/gamma` much longer than the kernel's memory `1/min beta`, is far from it;
  a model would refuse `gamma` within a margin of a decay or pair sum.
* **Lazy scale.** `1/s` reaches `2^500` before the rescale; f64 holds it with
  its full relative precision, and the rescale costs `D p` multiplies once per
  `500 ln 2 / gamma` time units.
* **Window drift.** `b^w` is a running sum of additions and removals, so its
  rounding error is that of every contribution ever added, not of the window's.
  It did not show over 10 to 13 turnovers (6e-14 at most); on a much longer
  stream it grows, and a rebuild from the stored window events (one form C pass,
  `O(n_W k)`) resets it whenever wanted. `R^w` after removing every event can be
  a rounding residue below 0 (about 1e-17), harmless.
* **Libm in the state.** `R` (all forms) and `s` (forgetting) are products of
  `exp` results; `F_e` in the window is a sum of them. They continue to the bit
  on one OS, and are what a cross-OS comparison would need a tolerance for.
* **Singular `G`.** A node with no events in a window, or a fit read before
  any event, gives a singular `G`; a model would gate the read on a minimum
  weight, as rheon's other models do.

## Open questions

* Is form C's state (`O(D^2 U)`) and per-event cost (`O(D U)`) acceptable at
  the dimensions rheon would offer? It is below form B's on both counts.
* A time window's state grows with the events in it; a model would cap the
  count (or offer only a window of `W` events), and choose between the direct
  `F_e` sum (`O(n_W U)` per removal, 16 bytes per event) and the two-block form
  (`O(k)` amortised, `8 (k + 2)` bytes per event).
* Non-negativity: tick's learners always project onto `theta >= 0`. An exact
  online fit can do the same at read time with a `p`-dimensional NNLS per node
  (section 3); whether rheon reports the unconstrained or the projected fit is
  still a design choice. The forgotten and windowed fits read the same way.
