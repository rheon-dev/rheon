# Hoeffding trees against river under a memory cap: a numpy study (PLAN task 503)

*Status, 2026-11-21: a record, in progress for request A14. Part 1: the
classifier written from the definitions and compared with river's, row by
row, on both of river's code paths (§2, §3). Part 2: the regressor and its
observers against river's (§4). Part 3: the state table, river's `max_size`,
a cap a build can enforce and its accuracy cost (§5). Nothing in the package or the
Rust crates was changed. The scan this builds on is
`vm-tools/research/494-MODELS-SCAN.md` on branch `vm-tools/oracle-gap-scan`,
section 9 ("scan §9"). The runnable study is `scripts/study_503.py`.*

## 1. The question

Scan §9 proposes online trees, river's `HoeffdingTreeRegressor` and
`HoeffdingTreeClassifier` first, and says their state "must be raised with the
user first": a regression leaf keeps per feature a table of bins, "tens of MB"
per group at 1,000 leaves. Before any Rust:

1. Does a tree written from the definitions (Domingos and Hulten 2000 for the
   classifier; Ikonomovska, Gama and Dzeroski 2011 for the regressor's split
   rule and leaf) give river's tree on the same rows, node for node and
   prediction for prediction, and to the bit where river's arithmetic allows?
2. How large is the state, as a function of features, classes, leaves and
   observer bins, and what per-group cap can a build enforce exactly? What
   does a cap of 64 KB, 256 KB or 1 MB cost in accuracy?
3. Is the tree deterministic under chunking and under save and resume?
4. What do zero-weight rows, weights, and a row cost?

river is 0.27.0 (installed from PyPI). Two facts about it shape everything
below:

* **river 0.27's `HoeffdingTreeClassifier` runs in Rust by default.** When the
  settings allow (numeric features only, the default `GaussianSplitter`,
  `split_criterion` not `'hellinger'`, `max_size == 100.0`,
  `memory_estimate_period == 1_000_000`, `stop_mem_management` and
  `remove_poor_attrs` off), `learn_one` goes to `NativeHoeffdingTreeClassifier`
  (`rust_src/hoeffding_tree.rs` in its sdist); any other setting runs the
  Python path (`tree/hoeffding_tree_classifier.py`). The two paths are not
  the same algorithm to the bit, and in four respects not at all (§3.3). Both
  are compared here. The Rust path has **no memory management**: `max_size`
  is not passed to it.
* **`HoeffdingTreeRegressor` has only the Python path.**

## 2. The trees

**Classifier** (`HTClassifier` in the script). A leaf keeps the class weights
`n_c`, and, once it has learned a row, a Gaussian observer per feature and
class: weight `W`, mean `m`, sum of squared deviations `S` (Welford, weighted;
variance `S / (W - 1)`, river's `ddof = 1` on the weight), min and max. Every
`grace_period` of learned weight the leaf tries a split: per feature,
`n_splits` (10) thresholds equally spaced strictly between the feature's min
and max over classes; at threshold `t` a class's weight goes left in the
proportion of its Gaussian CDF at `t` (all of it when `t` is above the class's
max, none when below its min); a threshold leaving either side with at most
`min_branch_fraction` of the weight is skipped; the criterion `G` scores the
rest. The leaf splits on the best feature when

```text
G(best) - G(second) > eps   or   eps < tau
eps = sqrt(R^2 ln(1/delta) / (2 n)),  n = the leaf's weight
R = log2(max(classes seen at the leaf, 2))   (information gain, Hellinger);  1 (Gini)
```

A leaf with one class is not tried; a leaf whose majority share exceeds
`max_share_to_split` is deactivated (it keeps its class weights and predicts
the majority class). Predictions: `mc` (class weights over their sum), `nb`
(naive Bayes over the Gaussian observers, `ln(n_c / n) + sum_f ln pdf_c(x_f)`,
normalised), `nba` (naive Bayes while it has been right on at least as much of
the leaf's learned weight as the majority class, scored before each row is
learned).

**The null split.** Domingos and Hulten's pre-pruning compares the best
attribute with `X_0`, "not splitting", whose merit is the leaf's own (an
information gain of 0). river gives the null split a merit of `-inf`
(`BranchFactory()`'s default, in both paths), so it only wins when no feature
has a candidate, and with a single usable feature the second-best is `-inf`
and the first attempt always splits. The script has `null_merit` (`-inf`,
river; `0.0`, the paper) and §3.4 measures the difference.

**Regressor** (`HTRegressor`): §4 (part 2).

The per-row arithmetic runs on Python floats (IEEE doubles, the rounding a
Rust build has); numpy holds the data and, from part 3, the packed state. The
order of operations is river's wherever it is a choice: the Welford update of
its Rust `Gaussian` (`n += w; m += (w/n)(x - m); S += max(w (x - m_old)(x -
m_new), 0)`), its entropy form `(-sum n_c log2 n_c + n log2 n) / n`, its Gini
form, its stable sort of the candidates with the null split first, and its
naive Bayes normalisation. Where river's two paths differ the script takes a
switch (`nb_form`, `gini_form`, `depth_rule`, all `'native'` or `'python'`).

**`erf`.** The Gaussian CDF is `0.5 (1 + erf((t - m) / sd / sqrt 2))`. river's
Rust links the `libm` crate (0.2.16), a port of FreeBSD msun's `s_erf.c` and
`e_exp.c`; Python's `math.erf` is glibc's. On 200,000 arguments glibc's CDF
differs from river's on 3,004 (1.5%), by at most 1.1e-16 absolute (1 ulp of
`erf`). The script ports msun's `erf` and `exp` operation for operation
(`msun_erf`); the port equals river's CDF on all 200,000. Every other libm
call (`ln`, `log2`, `exp`, `sqrt`) is the platform's in both river's Rust (std
`f64` methods) and Python, so they agree on the same OS.

## 3. Part 1: the classifier against river, row by row

Run: `.venv-503/bin/python scripts/study_503.py part1 --out <dir>` (30 s).

Prequential: each row is predicted (`predict_proba_one`, `predict_one`) and
then learned, by river and by the tree here, on the same rows. Streams: `k`
features N(0, 1); the label from a random axis-aligned tree of depth 3 to 9
over them, with 0.4% to 5% of labels redrawn at random; 20,000 rows, river's
Python path on all 20,000 too. Every comparison is exact (`==` on doubles).

### 3.1 Against river's Rust path (its default)

| case | k | classes | settings | splits | nodes | probabilities differ | labels differ | split rows, structure, thresholds |
|---|---|---|---|---|---|---|---|---|
| A | 5 | 2 | river's defaults (`grace_period` 200, `delta` 1e-7, `tau` 0.05, info gain, `nba`) | 12 | 25 | 0 of 20,000 | 0 | identical, thresholds 0 ulps |
| B | 10 | 2 | `grace_period` 50, Gini | 8 | 17 | 0 | 0 | identical |
| C | 5 | 3 | `grace_period` 50, `mc` | 6 | 13 | 0 | 0 | identical |
| D | 8 | 2 | `grace_period` 100, `nb` | 7 | 15 | 0 | 0 | identical |
| E | 5 | 2 | `grace_period` 50, weights U(0.2, 3) | 19 | 39 | 0 | 0 | identical |
| F | 6 | 3 | `grace_period` 50, `delta` 1e-5, `nba` | 6 | 13 | 0 | 0 | identical |
| G | 10 | 2 | `grace_period` 50, `tau` 0.1, depth-9 concept | 28 | 57 | 0 | 0 | identical |
| I | 5 | 2 | `grace_period` 50, `max_depth` 2 | 3 | 7 | 0 | 0 | identical |
| J | 5 | 2 | `grace_period` 50, 0.4% noise (leaves reach `max_share_to_split`) | 14 | 29 | 0 | 0 | identical |

**The tree matches river's Rust classifier to the bit**: every probability of
every row, every predicted label, the row of every split, the feature and the
threshold of every node. With glibc's `erf` in place of the port, the splits
and the structure are still identical in every case, but the probabilities
differ on 0 to 422 rows per case (B, C, E, F, G), by at most 1.3e-15
absolute: the CDF's last bit moves the class weights handed to the children at
a split, and through them the leaf's counts and its naive Bayes prior.

### 3.2 Against river's Python path

The same nine cases plus H (`split_criterion='hellinger'`, which only the
Python path has), with the switches set to river's Python forms:

| case | probabilities differ | labels differ | split rows, structure, thresholds |
|---|---|---|---|
| A to J (10 cases, 20,000 rows each) | 0 | 0 | identical, 0 ulps |

To get there the tree takes, from river's Python path: the majority-class
probabilities as `n_c * (1 / n)` (its `normalize_values_in_dict` multiplies by
the reciprocal: 6/10 is 0.6, 6 * (1/10) is 0.6000000000000001); naive Bayes
normalised by log-sum-exp (`exp(v - (max + ln sum exp(v - max)))`) instead of
`exp(v - max) / sum exp(v - max)`; Python's `sum()` for class-weight totals,
which since 3.12 is compensated (Neumaier) and differs from adding in order
for three or more classes; the Gini as `1 - sum_side (W_side / W) gini_side`
with `gini_side = 1 - p_1^2 - p_2^2 ...` in order; a leaf at `max_depth`
deactivated when it learns; Hellinger returning 0 whenever a class is absent
from a side (its `KeyError` path); and the memory manager below.

### 3.3 river's two paths disagree

On the same rows and settings river's Python and Rust paths differ on 9,284 to
18,880 of 20,000 rows. In seven cases the gap is rounding (at most 6.7e-15
absolute, 234 ulps), from the normalisation and the `1/n` above. In three it
is behaviour:

* **Three classes (C).** The compensated sum against the in-order sum moved a
  near tie in a split attempt: the split rows differ, and probabilities by up
  to 0.40.
* **`max_depth` (I).** The Python path deactivates a leaf at the depth, which
  then predicts its majority class; the Rust path keeps it active (naive
  Bayes). Probabilities differ by up to 0.85.
* **Near-pure leaves (J).** The Python path calls its memory manager after
  every decided split attempt. Until the first size estimate (at
  `memory_estimate_period` rows, 1,000,000 by default) its estimates are 0, so
  the manager deactivates nothing and **reactivates every inactive leaf above
  `max_depth`**, with empty observers. A leaf deactivated because one class
  passed `max_share_to_split`, or because the null split won, is therefore
  active again at once, with its observers reset; in the Rust path it stays
  inactive and predicts the majority class. Probabilities differ by up to
  0.997.

The tree here follows the Rust path by default, because it is river's
default and it is the algorithm as written (a deactivated leaf stays
deactivated).

### 3.4 The null split's merit

With river's `-inf` and one feature (4,000 rows, 45% label noise, a stream a
tree should barely split), river split 60 times and the tree here split at
the same 60 rows; with the paper's 0 it split twice. With several features
the two agree while every feature has a candidate; they part when only one
feature has a threshold passing `min_branch_fraction`: E split 19 times under
`-inf` and 9 under 0 (accuracy 0.781 against 0.777), G 28 and 19 (0.671
against 0.667), J 14 and 12. A build that matches river takes `-inf`; the
choice should be stated in the model's docs.

## 4. Part 2: the regressor and its observers

Run: `.venv-503/bin/python scripts/study_503.py part2 --out <dir>` (50 s).

**The regressor** (`HTRegressor`). A leaf keeps the target's weight `W`, mean
and sum of squared deviations `S` (river's Python `stats.Var`: `n += w; m +=
(w/n)(y - m); S += w (y - m_old)(y - m_new)`, variance `S / (W - 1)`), and per
feature an observer that can return, for each candidate threshold, the target
statistics of the leaf's rows at or below it. The merit of a threshold is the
variance reduction (Ikonomovska et al.):

```text
VR = var(S) - (W_l / W) var(S_l) - (W_r / W) var(S_r),   0 when a side has < min_samples_split (5)
split when  VR(second) / VR(best) < 1 - eps   or   eps < tau,   with VR(best) > 0 and R = 1
```

Sides are combined and subtracted with Chan et al.'s parallel formulas, in
river's order (`S = S_a + S_b + (m_a - m_b)^2 W_a W_b / (W_a + W_b)`, the
mean as `(W_a m_a + W_b m_b) / W`; subtraction as river's `__isub__`, which
uses the updated mean). The leaf predicts its mean (`leaf_prediction='mean'`).
A failed attempt on E-BST and TE-BST observers prunes their bad split points
(FIMT-DD's rule as river has it: a node whose merit ratio falls below `r - 2
eps`, only where its left subtree is all bad). river's memory manager runs
after every decided attempt, with the reactivation of §3.3.

**Observers.**

* **E-BST** (`EBSTSplitter`): a binary search tree with a node per distinct
  feature value seen at the leaf, each holding the target statistics of the
  rows at or below its value in its subtree; split candidates are every
  stored value, by an in-order walk. **Its size is the number of distinct
  values**: at one leaf that never splits, 1,000 nodes after 1,000 rows,
  40,000 after 40,000. It grows with the stream: by the user's rule it is
  out (N).
* **TE-BST** (`TEBSTSplitter(digits=1)`, river's default): E-BST on values
  rounded to `digits` decimals with Python's `round`. Bounded by the number
  of distinct rounded values, which is the feature's range times `10^digits`:
  on N(0, 1) data 60, 69, 76, 81 nodes after 1,000, 5,000, 20,000, 40,000
  rows at 1 digit, and 378 to 672 at 2 digits. The bound holds only for a
  bounded feature; a heavy tail keeps adding nodes. The rounding is part of
  the arithmetic: on 200,000 N(0, 9) values `floor(|x| 10^d + 0.5) / 10^d`
  (with the sign) gave the same doubles as Python's `round(x, d)` for d = 1,
  2, 3, but they are different functions at decimal ties, and a build must
  pin one.
* **QO** (`QOSplitter(radius=0.25)`): slot `floor(x / radius)` keeps the
  weight and mean of `x` and the target statistics of its rows; candidates
  are the midpoints of consecutive slots' `x` means. Bounded by range over
  radius: 26, 30, 32, 34 slots after 1,000 to 40,000 N(0, 1) rows. A slot cap
  (`max_slots`, a value whose slot would be new goes to the nearest existing
  slot) makes the bound hard; part 3 uses it.

### 4.1 Against river, row by row

Prequential, 20,000 rows each, river's `HoeffdingTreeRegressor` with
`leaf_prediction='mean'` and the same splitter:

| case | k | stream | observer | settings | splits | predictions differ | split rows, structure, thresholds | observer size, mean (max) | MSE |
|---|---|---|---|---|---|---|---|---|---|
| RA | 5 | random tree, depth 6 | TE-BST 1 | river's defaults | 68 | 0 of 20,000 | identical, 0 ulps | 18.4 (48) nodes | 0.766 |
| RB | 8 | Friedman #1 | TE-BST 1 | `grace_period` 50 | 170 | 0 | identical | 13.0 (39) | 0.304 |
| RC | 5 | random tree | E-BST | `grace_period` 100 | 108 | 0 | identical | 47.0 (243) | 0.488 |
| RD | 5 | random tree | QO 0.25 | `grace_period` 100 | 107 | 0 | identical | 7.7 (22) slots | 0.641 |
| RE | 8 | Friedman #1 | QO 0.25 | `grace_period` 50 | 152 | 0 | identical | 8.8 (24) | 0.348 |
| RF | 5 | random tree, weights U(0.2, 3) | TE-BST 1 | `grace_period` 50 | 297 | 0 | identical | 8.1 (35) | 0.444 |
| RG | 5 | random tree, weights U(0.2, 3) | QO 0.25 | `grace_period` 50 | 272 | 0 | identical | 5.6 (25) | 0.689 |

**The regressor matches river's to the bit**: every prediction, every split
row, every threshold, and the number of nodes or slots in every observer at
the end. river's single feature quirk is the classifier's: with one feature
the second-best merit is the null split's `-inf`, so `-inf / VR(best) < 1 -
eps` and every attempt with a positive merit splits (36 splits in 3,000 rows
of noise sd 3, same rows here); in the regressor a null merit of 0 gives the
same (0 / VR < 1 - eps), so the choice does not matter there.

### 4.2 Where river's QO departs from its definition

Two things in river 0.27's `QOSplitter.best_evaluated_split_suggestion`,
confirmed on river's own first split (3,000 rows, radius 0.25, first split at
row 100):

1. **The partition scored is not the one the threshold makes.** At the
   midpoint between slots `i-1` and `i` the running sum already holds slot
   `i`, so the merit is that of `{slots <= i} | {slots > i}` while the
   threshold sends slot `i` right. At river's split the right child started
   with 62 rows; the threshold sends 65 right.
2. **The left child gets the whole leaf's statistics.** The candidate keeps a
   reference to the running accumulator, which by the end of the walk holds
   every slot. River's left child started with `n = 100` (all rows) and the
   parent's mean, 0.1077; the 35 rows the threshold sends left have mean
   -0.2451. The left leaf then predicts the parent's mean until it has seen
   enough rows to move it, and its `last_split_attempt_at` is the parent's
   weight.

The script's `river_quirks=True` reproduces both (that is how §4.1 matched);
`False` scores the partition the threshold makes and copies the left
statistics. On the same rows the definition lowered the prequential MSE from
0.641 to 0.516 (RD), 0.348 to 0.331 (RE) and 0.689 to 0.613 (RG). **A build
should implement the definition, and its oracle test against river must use
TE-BST or E-BST, or a QO oracle written here, not river's QO.** With TE-BST
and E-BST the definition and river are the same tree.

### 4.3 What was not reproduced: river's leaf models

river's regressor defaults to `leaf_prediction='adaptive'`: each leaf also
trains river's `LinearRegression` (SGD) and predicts with whichever of the
mean and the model has the lower faded squared error (`model_selector_decay =
0.95`). The tree's splits do not depend on the leaf model (the split rows were
the same as with `'mean'` in all seven cases), so the structure above holds
for it; the leaf model was not reproduced here and its predictions were not
compared. On these streams the adaptive leaves lowered the MSE (RA 0.766 to
0.683, RB 0.304 to 0.182, RE 0.348 to 0.201; RG 0.689 to 0.630). A rheon leaf
model would be one of this library's own models (`sgd`'s state per leaf),
held to its own oracle; it adds `(k + 1)` floats or more per leaf. Two
details for whoever builds it: river's adaptive leaf updates its faded
errors **without the row's weight** (a zero-weight row moves them, §6), and
a child copies its parent's leaf model and faded errors at a split.

### 4.4 The classification observers

river's classifier offers `GaussianSplitter(n_splits=10)` (5 floats per class
and feature, the default, §2) and `HistogramSplitter(n_bins=256,
n_splits=32)` (a streaming histogram per class and feature, up to `n_bins`
bins of three numbers each). On 10,000 rows (5 features, 2 classes,
`grace_period` 50, `mc` leaves, river's Python path) the histograms split
more and scored better: accuracy 0.644 Gaussian, 0.670 with 32 bins, 0.681
with 256 bins, at up to 32 and 256 bins per observer (90 on average for 256).
The Gaussian observer is the small one and the one river's Rust path has; the
histogram is a later option, and its bytes are `24 n_bins` per class and
feature.

## 5. Part 3: state size, river's `max_size`, and a cap a build can enforce

Run: `.venv-503/bin/python scripts/study_503.py part3 --out <dir>` (670 s, of
which 330 s are river's uncapped Python-path runs).

### 5.1 river's memory management

river bounds a tree by `max_size` in MiB of **Python objects**
(`calculate_object_size`, which walks the object graph), not by any layout a
saved state has. Every `memory_estimate_period` learned weight (1,000,000 by
default) it measures the active and inactive leaves, and when the tree is over
`max_size` it ranks the leaves by promise (the classifier: the weight the
majority class gets wrong; the regressor: minus the depth, so shallow leaves
first), keeps as many active as the estimate allows and deactivates the rest
(they keep their class weights or target statistics and stop feeding their
observers, so they can no longer split; a classifier's inactive leaf predicts
its majority class). Three consequences:

* Until the first estimate the manager does nothing at all (§3.3), so with
  the default period no tree is bounded before a million rows.
* The Rust classifier, river's default, has no memory management.
* Python object sizes are 9 to 80 times a packed layout (§5.2): river's tree
  of 284 leaves over 50 features is 94.9 MB of objects and 1.15 MB packed. At
  `max_size` = 1 MiB river kept 2 of 158 leaves active; at 256 KB and 64 KB
  it kept none (9 leaves, all inactive), on the stream of §5.3.

So river's number cannot be matched and should not be: the scan's advice
holds (a cap in the build's own units, with river's management off and the
cap not binding in the oracle runs).

### 5.2 The state table

The packed layout (`pack_clf`, `pack_reg` in the script; little-endian, no
padding) is what a build would save. Its length equals the formula below on
every tree measured (three classifiers, three regressors, all to the byte),
and `unpack` of it continues to the bit (§6). `L` leaves, `B = L - 1`
branches, `A` leaves holding observers, `k` features, `C` classes:

| tree, observer | bytes | per active leaf, with its branch | per inactive leaf | bounded by |
|---|---|---|---|---|
| classifier, Gaussian | `40 + 8C + 36B + (8C + 27)L + 40kC A` | `40kC + 8C + 63` | `8C + 63` | the shape: E |
| classifier, histogram (river's `HistogramSplitter`) | as above with `24 n_bins kC` per active leaf | | | `n_bins`: E |
| regressor, QO, at most `S` slots a feature | `40 + 20B + 35L + sum over observers (2 + 36 slots)`, at most `... + k(2 + 36S) A` | up to `k(2 + 36S) + 55` | `55` | `S`: E (river's QO has no `S`: bounded by the feature's range over the radius) |
| regressor, TE-BST | `... + sum (4 + 33 nodes)`, nodes <= distinct rounded values | | `55` | the feature's range times `10^digits` only |
| regressor, E-BST | `... + sum (4 + 33 nodes)`, nodes = distinct values seen at the leaf | | `55` | **the stream: N** |

The branch keeps its left and right weights in the classifier (river's Rust
tree routes a missing value to the heavier side); the regressor's needs none
here. A build's msgpack adds a type byte per float unless it stores the
arrays as binary, about 12% more.

Some numbers (packed, and river's Python-object size of river's tree grown on
the same 20,000 rows, the same tree in every case):

| tree | k | C | leaves | packed | river's objects | ratio |
|---|---|---|---|---|---|---|
| classifier | 5 | 2 | 32 | 14.5 KB | 1.07 MB | 73 |
| classifier | 20 | 2 | 30 | 50.4 KB | 3.98 MB | 79 |
| classifier | 20 | 5 | 6 | 24.7 KB | 1.91 MB | 77 |
| regressor, QO 0.25 (mean 10, max 23 slots) | 10 | | 113 | 426 KB | 5.41 MB | 13 |
| regressor, TE-BST 1 (mean 16, max 42 nodes) | 10 | | 123 | 606 KB | 5.21 MB | 9 |
| regressor, E-BST (mean 44, max 190 nodes) | 10 | | 110 | 1.53 MB | 13.0 MB | 8.5 |

The unit costs: a classifier's active leaf at `k = 20, C = 2` is 1.7 KB, at
`k = 50, C = 2` 4.1 KB; a regressor's active leaf at `k = 20` with QO capped
at 32 slots is up to 23 KB (16 slots: 11.6 KB). An inactive leaf is 55 to 103
bytes. **Uncapped, the trees of §5.3 reached 1.1 MB (classifier, 284 leaves,
200,000 rows, `k = 50`) and 10.6 to 11.5 MB (regressor, 1,545 to 1,806
leaves, `k = 20`)** per group; the regression models' state today is `O(k^2)`
doubles per target (3.2 KB for a 20 x 20 matrix), so even the classifier is
two orders of magnitude above today's typical state, and the regressor three
to four. That is the growth the user's rule says to raise first.

### 5.3 The cap, and what it costs

**The rule** (`set_cap` in the script). A per-group `max_state_bytes`
counted in the packed layout above:

1. A split is refused when the structure alone (every leaf inactive, `40 +
   8C + 36B + (8C + 27)L` or `40 + 20B + 35L`) would pass the cap; the tree
   then stops growing.
2. After every split, and every 1,000 learned rows, the leaves are ranked by
   promise and only as many keep observers as the bytes left by the structure
   pay for **at their largest** (`40kC` for the Gaussian observers, `k(2 +
   36S)` for QO with at most `S` slots); the rest are deactivated (observers
   dropped, statistics kept). A leaf the cap deactivated is reactivated, with
   empty observers, when it ranks back in; a leaf the algorithm deactivated
   (a dominated or pre-pruned leaf) is not.
3. Because observers are reserved at their largest, **the packed state never
   passes the cap**, whatever the rows: checked every 2,000 rows and at the
   end of every capped run below (worst 65,500 bytes under 64 KB, 262,111
   under 256 KB, 476,853 under 1 MB). It is deterministic: the ranking is a
   stable sort with ties in leaf order, and the recheck counts learned rows,
   which are in the state.

Promise: the classifier uses river's (the weight the majority class gets
wrong); for the regressor river's (minus the depth) is compared with the
leaf's sum of squared deviations `S` (the error a split could remove), which
is better at every cap below.

Prequential runs, 200,000 rows, accuracy (classifier) or MSE (regressor)
over all rows and over the last 50,000; `grace_period` 50, `tau` 0.1 for the
classifier, `grace_period` 50 for the regressor:

**Classifier**, `k = 50`, two classes, Gaussian observers, `nba`:

| stream | cap | accuracy (last 50k) | cost | leaves (active) | packed bytes at the end |
|---|---|---|---|---|---|
| random tree, depth 10, 5% noise | none | 0.6002 (0.6198) | | 284 (284) | 1,146,456 |
| | 1 MB | 0.6002 (0.6198) | 0.0000 (0.0000) | 284 (256) | 1,046,456 |
| | 256 KB | 0.6005 (0.6187) | +0.0002 (-0.0011) | 270 (60) | 261,350 |
| | 64 KB | 0.5880 (0.5953) | -0.0122 (-0.0245) | 208 (12) | 64,452 |
| | river `max_size` 1 MiB | 0.5647 (0.5707) | -0.0356 (-0.0491) | 158 (2) | (river: 0.88 MB of objects) |
| | river 256 KB, 64 KB | 0.5406 (0.5401) | -0.0596 (-0.0797) | 9 (0) | |
| hyperplane, 10 of 50 features informative, 5% noise | none | 0.8410 (0.8526) | | 281 (274) | 1,118,219 |
| | 1 MB | 0.8410 (0.8528) | 0.0000 (+0.0002) | 281 (256) | 1,046,219 |
| | 256 KB | 0.8365 (0.8422) | -0.0044 (-0.0104) | 215 (61) | 257,005 |
| | 64 KB | 0.8123 (0.8104) | -0.0286 (-0.0422) | 119 (14) | 65,421 |
| | river 1 MiB | 0.7802 (0.7862) | -0.0608 (-0.0664) | 108 (2) | |
| | river 256 KB, 64 KB | 0.7023 (0.6987) | -0.1387 (-0.1540) | 6 (0) | |

(River's uncapped run on the hyperplane differs from the tree here by
0.0001: river's Python path ran it, which reactivates the 7 dominated leaves
that the Rust path, followed here, keeps inactive; §3.3.)

**Regressor**, `k = 20`, QO radius 0.25 with at most 32 slots a feature,
`'mean'` leaves; cost is the MSE's rise over the uncapped tree:

| stream | cap | promise | MSE (last 50k) | cost | leaves (active) | packed bytes, end (worst) |
|---|---|---|---|---|---|---|
| Friedman #1, 5 of 20 features informative | none | | 0.2499 (0.2142) | | 1,545 (1,545) | 10,619,919 |
| | 1 MB | `S` | 0.2672 (0.2411) | +7% (+12%) | 551 (44) | 372,925 (476,853) |
| | 1 MB | depth | 0.2979 (0.2826) | +19% (+32%) | 237 (44) | 397,303 |
| | 256 KB | `S` | 0.2914 (0.2667) | +17% (+25%) | 298 (10) | 101,806 (119,019) |
| | 256 KB | depth | 0.3593 (0.3460) | +44% (+62%) | 118 (11) | 97,454 |
| | 64 KB | `S` | 0.3483 (0.3242) | +39% (+51%) | 119 (2) | 24,681 (30,799) |
| | 64 KB | depth | 0.4040 (0.3774) | +62% (+76%) | 56 (2) | 15,672 |
| | river 1 MiB | depth | 0.3230 (0.3097) | +29% (+45%) | 126 (10) | |
| | river 256 KB | depth | 0.3404 (0.3277) | +36% (+53%) | 57 (1) | |
| | river 64 KB | depth | 0.3486 (0.3408) | +39% (+59%) | 33 (0) | |
| random tree, depth 10, noise sd 0.3 | none | | 1.0785 (1.0484) | | 1,806 (1,806) | 11,526,962 |
| | 1 MB | `S` | 1.0966 (1.0781) | +1.7% (+2.8%) | 614 (43) | 365,226 |
| | 256 KB | `S` | 1.1114 (1.0963) | +3.1% (+4.6%) | 309 (10) | 102,231 |
| | 64 KB | `S` | 1.1399 (1.1298) | +5.7% (+7.8%) | 141 (2) | 18,327 |
| | river 1 MiB, 256 KB, 64 KB | depth | 1.141 to 1.146 | +5.8% to +6.3% | 41 to 84 | |

(The depth-10 random tree over 20 features is barely learnable by a mean
tree in 200,000 rows, so its costs are small because its gains are; the
Friedman stream is the informative one.)

**What the numbers say.** For the classifier the cap is cheap: an active leaf
is a few KB, inactive leaves keep predicting, and 1 MB cost nothing, 256 KB
at most half a point of accuracy (one point on the last quarter), 64 KB one
to three points. For the regressor it is expensive: an active QO leaf is tens
of KB at `k = 20`, a leaf without observers can never split, and the tree
that fits in 1 MB is a third of the uncapped one; the cost is 7% of MSE at 1
MB and 17% at 256 KB with the `S` promise, worse with river's. The worst-case
reservation is the price of the guarantee: the capped regressors used at most
46% of their cap, because a leaf fills on average 10 of its 32 slots. A
build could reserve less (count the slots in use and evict when a leaf's
observers grow), at the cost of a rule that must handle a leaf growing past
the budget mid-stream; the reservation here is the simple rule that cannot
fail.

### 5.4 The proposed per-group cap

* **The unit is the packed state's bytes**, per group (and per target where a
  spec has several), computed from counts the state holds; the docstring
  states the formula of §5.2.
* **The cap is a required spec setting with a default of 256 KB**, refused
  above a ceiling (64 MiB, the order of the spec layer's other allocation
  ceilings) and below the size of a one-leaf tree. At 256 KB the classifier
  costs under one point of accuracy on these streams and the regressor 17%
  of MSE; 1 MB is the setting to recommend for a regressor with `k >= 20`.
* **The observers it enforces are bounded ones only**: Gaussian (classifier),
  QO with a slot cap `S` (regressor; default 32), optionally the histogram
  with `n_bins`. E-BST is refused (it grows with the stream); TE-BST is
  bounded only for bounded features and is better left out.
* **The bank's memory is groups times the cap**: 10,000 groups at 256 KB are
  2.5 GB, at 1 MB 10 GB. The state is not preallocated (a tree grows into its
  cap), so the figure is a ceiling, but it is the figure to raise with the
  user: it is two to three orders of magnitude above today's per-group state.
