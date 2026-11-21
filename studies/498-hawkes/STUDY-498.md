# Study 498, part 1: exact online least-squares fit of a Hawkes process

Status: part 1 complete (math, online and batch accumulators, comparison with
tick, cost). Part 2 not started.

Question: can tick's least-squares fit of a Hawkes process with sum-of-exponential
kernels and given decays be computed exactly online, one event at a time, with
state bounded independently of the number of events? Does it match tick?

Answer: yes on both counts. The fit is a quadratic form whose Gram matrix and
linear terms update in closed form at each event; the accumulated statistics
equal a batch double sum over pairs of events to about 1e-14 relative, and the
loss and minimiser equal tick's to at most 3.9e-13 relative, once tick's
normalisation (division by the number of events) is applied.

Script: `studies/498-hawkes/hawkes_ls.py` (numpy; tick, scipy only for the
comparison). Data: an Ogata thinning simulator in the script, seeded numpy
generators; nothing is stored.

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

The task asked for O(k) per event, k counting decays times dimension. Form B is
`O(U k)`, not `O(k)`: `G` holds the products `R_a R_b` across nodes, and an event
of node `j` moves the `U` rows of `G` that involve node `j`, `k` entries each. I
see no way below that while `G` is kept explicitly; it is `O(k)` per event for
`U = 1`.

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

## Open questions

* Per-event cost is `O(U k)`, not `O(k)`; see section 1. Is `O(D U^2)` per
  event, `O(D^2 U^2)` state, acceptable at the dimensions rheon would offer?
* Form B reads `G` from `J - R(T) R(T)'`. The cancellation is benign in these
  cases (agreement 1e-14); a stream of very bursty events at a long decay is the
  case to probe before choosing it over form A.
* Non-negativity: tick's learners always project onto `theta >= 0`. An exact
  online fit can do the same at read time with a `p`-dimensional NNLS per node
  (as here); whether rheon reports the unconstrained or the projected fit is a
  design choice for part 2 or later.
