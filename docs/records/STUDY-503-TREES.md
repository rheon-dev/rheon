# Hoeffding trees against river under a memory cap: a numpy study (PLAN task 503)

*Status, 2026-11-21: a record, in progress for request A14. Part 1: the
classifier written from the definitions and compared with river's, row by
row, on both of river's code paths (§2, §3). Nothing in the package or the
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
