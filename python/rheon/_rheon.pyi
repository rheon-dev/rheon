"""Type stub for the pyo3 module (crates/rheon-py/src/lib.rs). The wrappers in
this package are what users call; this is what lets a type checker follow them
into the native calls. ``tests/test_kwargs_typing.py`` checks it names every
attribute the built module has, so it cannot go stale silently."""

from collections.abc import Callable, Iterator

import polars as pl

class RefreshTime:
    """Refresh-time sampling, fed frames of the long input in order."""

    def __init__(
        self,
        names: list[str],
        series: str,
        clock: str,
        value: str,
        group: str | None = None,
        pairs: bool = False,
        keep: list[str] | None = None,
        age: bool = False,
    ) -> None: ...
    @staticmethod
    def load_bytes(
        state: bytes,
        names: list[str],
        series: str,
        clock: str,
        value: str,
        group: str | None = None,
        pairs: bool = False,
        keep: list[str] | None = None,
        age: bool = False,
    ) -> RefreshTime: ...
    def save(self, path: str) -> None: ...
    def feed(self, df: pl.DataFrame, limit: int | None = None) -> pl.DataFrame: ...

class Bars:
    """``rp.stream.bars``, fed a stream's chunks in order."""

    def __init__(self, config: str, input: pl.DataFrame) -> None: ...
    @staticmethod
    def load_bytes(state: bytes, config: str, input: pl.DataFrame) -> Bars: ...
    def save(self, path: str) -> None: ...
    def save_bytes(self) -> bytes: ...
    def feed(self, df: pl.DataFrame, limit: int | None = None) -> pl.DataFrame: ...
    def finish(self) -> pl.DataFrame: ...
    def take_notices(self) -> list[str]: ...
    def group_bytes(self, key_len: int) -> int: ...

class Merge:
    """``rp.stream.merge``, fed each input's chunks in order."""

    def __init__(
        self,
        names: list[str],
        frames: list[pl.DataFrame],
        clock: str,
        source: str,
        by: str | None = None,
        restart: float | None = None,
        restart_text: str | None = None,
        carry: list[str] | None = None,
    ) -> None: ...
    @staticmethod
    def load_bytes(
        state: bytes,
        names: list[str],
        frames: list[pl.DataFrame],
        clock: str,
        source: str,
        by: str | None = None,
        restart: float | None = None,
        restart_text: str | None = None,
        carry: list[str] | None = None,
    ) -> Merge: ...
    def empty(self) -> pl.DataFrame: ...
    def push(self, i: int, df: pl.DataFrame) -> None: ...
    def end(self, i: int) -> None: ...
    def pull(
        self, limit: int | None = None, hold: bool = False
    ) -> tuple[pl.DataFrame, int | None]: ...
    def save_bytes(self) -> bytes: ...
    def save(self, path: str) -> None: ...

class Windows:
    """``rp.stream.with_windows``, fed a stream's chunks in order."""

    def __init__(self, config: str, input: pl.DataFrame) -> None: ...
    @staticmethod
    def load_bytes(state: bytes, config: str, input: pl.DataFrame) -> Windows: ...
    def save(self, path: str, skip_on_resume: int = 0, input_ended: bool = False) -> None: ...
    def feed(self, df: pl.DataFrame, limit: int | None = None) -> pl.DataFrame: ...
    def finish(self) -> pl.DataFrame: ...
    def settle(self) -> pl.DataFrame: ...
    def needed(self) -> list[str]: ...
    def consumed(self) -> int: ...
    def take_warning(self) -> str | None: ...

class ArrowStruct:
    """One spec's output struct, over the Arrow PyCapsule interface.

    Handed back by ``ModelBank.fit_predict_arrow``. A reader of
    ``__arrow_c_array__`` takes it directly -- ``pl.Series(obj)``,
    ``pa.array(obj)`` -- and duckdb, which reads streams, through
    ``pl.Series(obj)``. Exporting consumes it, so it can be read once.

    .. warning::
        The Arrow output is considered **unstable**: it may change in any
        release without that counting as a breaking change. Under
        ``RHEON_WARN_UNSTABLE=1`` the calls that return one raise
        ``rheon.UnstableWarning``.
    """

    @property
    def name(self) -> str: ...
    def __arrow_c_array__(
        self, requested_schema: object | None = None
    ) -> tuple[object, object]: ...

class ModelBank:
    def __init__(self, specs_json: str, threads: int | None = None) -> None: ...
    @property
    def threads(self) -> int | None: ...
    def fit_predict(
        self, df: pl.DataFrame, row_base: int = 0, learn_only: bool = False
    ) -> list[pl.Series]: ...
    def predict(self, df: pl.DataFrame, row_base: int = 0) -> list[pl.Series]: ...
    def fit_predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]: ...
    def predict_arrow(self, df: pl.DataFrame) -> list[ArrowStruct]: ...
    def save(self, path: str) -> None: ...
    def save_bytes(self) -> bytes: ...
    def save_json_string(self, pretty: bool = True) -> str: ...
    @staticmethod
    def load_bytes(
        bytes: bytes, specs_json: str | None = None, threads: int | None = None
    ) -> ModelBank: ...
    def take_notices(self) -> list[str]: ...
    def output_fields(self) -> list[list[str]]: ...
    def solve_failures(self) -> list[list[tuple[str | None, int]]]: ...
    def gram(
        self,
        spec: int,
        group: list[str | None] | None = None,
        float32: bool = False,
        packed: bool = False,
        columns: list[str] | None = None,
    ) -> list[
        tuple[
            tuple[
                str | None,  # group
                str,  # instance
                int,  # k
                float,  # weight_sum
                float | None,  # n_kish
                list[float],  # means
                # comoments: the bytes of k*k row-major, or k(k+1)/2 packed,
                # float64s or float32s (docs/PLAN.md task 229)
                bytearray,
                list[list[float]],  # cross_moments, one row per target
                list[float],  # target_weights
                list[float],  # target_means
                list[float],  # target_vars
                list[float | None],  # target_n_kish
            ],
            # (lags, L*k*k cross-moments as bytes), or None without lags (E56)
            tuple[list[int], bytearray] | None,
            # this Gram's targets, as indices into the spec's, each one's
            # column means over its own rows (docs/PLAN.md task 81), and its
            # cross-moments centred at them (review 2026-09-12, N4)
            tuple[list[int], list[list[float]], list[list[float]]],
        ]
    ]: ...
    def coef(
        self, spec: int, group: list[str | None] | None = None
    ) -> list[tuple[str | None, str, float, list[float] | None]]: ...
    def last_row(
        self, spec: int, group: list[str | None] | None = None
    ) -> tuple[list[str | None], pl.Series]: ...
    def summary(self, specs: list[int], group: list[str | None] | None = None) -> pl.DataFrame: ...
    def describe(self, spec: int, group: list[str | None] | None = None) -> pl.DataFrame: ...
    def marginal(self, spec: int, group: list[str | None] | None = None) -> pl.DataFrame: ...
    def smoothed_tail(self, spec: int, group: list[str | None] | None = None) -> pl.DataFrame: ...
    def audit(
        self,
        spec: int,
        group: list[str | None] | None = None,
        table: str = "columns",
        pooled: bool = False,
    ) -> pl.DataFrame: ...
    def vector_ar(
        self,
        spec: int,
        group: list[str | None] | None = None,
        table: str = "irf",
        periods: int = 10,
        stderr: bool = False,
        caused: list[str] | None = None,
        causing: list[str] | None = None,
        kind: str | None = None,
    ) -> pl.DataFrame: ...
    def closed_groups(self, spec: int | None = None, drop: bool = True) -> pl.DataFrame: ...
    def spec_names(self) -> list[str]: ...
    def specs_json(self) -> str: ...
    def groups(self, specs: list[int]) -> pl.DataFrame: ...
    def group_counts(self) -> list[int]: ...
    def drop_groups(self, keys: list[str | None], spec: int | None = None) -> int: ...
    def last_clocks(
        self,
    ) -> list[list[tuple[str | None, float | None, int | None, int | None]]]: ...
    def group_marks(self) -> list[tuple[str | None, bool] | None]: ...
    def rows_fed(self) -> int: ...

def validate_spec(spec_json: str) -> None: ...
def parse_duration(text: str) -> int: ...
def format_duration(ns: int) -> str: ...
def psquared_reads(values: list[float], levels: list[float]) -> list[list[float | None]]: ...
def tdigest_quantiles(
    values: list[float], compression: float, levels: list[float]
) -> list[float | None]: ...
def tdigest_tail_means(
    values: list[float], compression: float, levels: list[float]
) -> list[float | None]: ...
def tdigest_centroids(values: list[float], compression: float) -> list[tuple[float, float]]: ...
def ddsketch_buckets(
    values: list[float], alpha: float
) -> tuple[list[tuple[int, float]], list[tuple[int, float]], float]: ...
def ddsketch_decayed_buckets(
    values: list[float], alpha: float, half_life: float
) -> tuple[list[tuple[int, float]], float]: ...
def ddsketch_values(keys: list[int], alpha: float) -> list[float]: ...
def ddsketch_reads(
    values: list[float], weights: list[float], alpha: float, levels: list[float]
) -> list[tuple[float, float] | None]: ...
def spec_clock_fields() -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]]]: ...
def format_of_path(path: str, who: str = ...) -> str: ...
def csv_flat(df: pl.DataFrame, who: str = ...) -> pl.DataFrame: ...
def write_frame(df: pl.DataFrame, path: str, who: str = ...) -> None: ...
def default_chunk_size() -> int: ...
def spec_output_index(spec_json: str) -> pl.DataFrame: ...
def spec_coef_fields(spec_json: str) -> pl.DataFrame: ...
def spec_output_fields(spec_json: str) -> list[str]: ...
def resolved_defaults(spec_json: str) -> str: ...
def native_version() -> str: ...
def schema_version() -> int: ...
def thread_pool_size(threads: int | None = None) -> int: ...
def max_threads() -> int: ...
def threads_too_many(shown: str) -> str: ...
def model_kinds() -> list[str]: ...
def save_checkpoint(path: str, stages: list[tuple[str, bytes]]) -> None: ...
def load_checkpoint(data: bytes) -> list[tuple[str, str, bytes]]: ...

# A Gram as the fits in `rheon.gram` hand it over: `k` and
# `weight_sum`, then `means`, `comoments`, `cross_moments`, `means_by_target`,
# `cross_centred`, `target_weights`, `target_means`, `target_vars` and
# `target_n_kish`, each a C-contiguous float64 numpy array, and last
# `target_within_parts`, per target the parts a Gram pooled within holds, or
# None (docs/PLAN.md task 240).
_GramIn = tuple[
    int,
    float,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    object,
    list[int] | None,
]
# A lasso path: the knots' penalties, their coefficients (the bytes of
# native-endian float64s, flat), the active columns at each, and why it stopped.
_PathOut = tuple[list[float], bytearray, list[list[int]], str]

def gram_lars_paths(
    grams: list[_GramIn],
    targets: list[list[int]],
    slots: list[int],
    icept: int | None,
    weights: list[float],
    max_steps: int | None,
    max_active: int | None,
    threads: int | None = None,
) -> list[list[_PathOut]]: ...
def gram_cd_path(
    gram: _GramIn,
    target: int,
    slots: list[int],
    icept: int | None,
    penalties: list[float],
    l1_ratio: float,
    weights: list[float],
    max_iter: int,
    tol: float,
) -> tuple[list[float], list[bool]]: ...

# A ridge fit: `coef`, `se` and `t` (each the bytes of native-endian float64s),
# then `resid_var`, `sigma2`, `r2` and `n`.
_RidgeOut = tuple[bytearray, bytearray, bytearray, float, float, float, float]

def gram_ridge_subsets(
    grams: list[_GramIn],
    subsets: list[list[int]],
    targets: list[int],
    slots: list[int],
    icept: int | None,
    ridge: float,
    standardize: bool,
    within: bool,
    signs: list[list[float]],
    threads: int | None = None,
) -> list[list[list[_RidgeOut]]]: ...
def gram_path_subsets(
    grams: list[_GramIn],
    subsets: list[list[int]],
    targets: list[int],
    slots: list[int],
    icept: int | None,
    weights: list[float],
    max_steps: int | None,
    max_active: int | None,
    within: bool,
    signs: list[list[float]],
    threads: int | None = None,
) -> list[list[list[_PathOut]]]: ...

# A job: its subset, target, draw and column set, each a position. Each job's
# fit, in order: its subset merged, its draw's signs on the cross-moments, its
# target fitted on `slots[set]` (a path weighted by `weights[set]`).
_JobIn = tuple[int, int, int, int]

def gram_ridge_jobs(
    grams: list[_GramIn],
    subsets: list[list[int]],
    slots: list[list[int]],
    jobs: list[_JobIn],
    icept: int | None,
    ridge: float,
    standardize: bool,
    within: bool,
    signs: list[list[float]],
    threads: int | None = None,
) -> list[_RidgeOut]: ...
def gram_path_jobs(
    grams: list[_GramIn],
    subsets: list[list[int]],
    slots: list[list[int]],
    weights: list[list[float]],
    jobs: list[_JobIn],
    icept: int | None,
    max_steps: int | None,
    max_active: int | None,
    within: bool,
    signs: list[list[float]],
    threads: int | None = None,
) -> list[_PathOut]: ...

# `rheon.polars._gbdt`'s native engine (python/rheon/polars/_gbdt.py, docs/PLAN.md task
# 242). `batches` returns a fresh iterator of frames for each pass; each
# build returns the library's handle, which the caller owns, and its counts
# and timings.
class GbdtXgboost:
    def __init__(self, path: str) -> None: ...
    @property
    def version(self) -> tuple[int, int, int]: ...
    def matrix(
        self,
        batches: Callable[[], Iterator[pl.DataFrame]],
        names: list[str],
        kinds: list[str],
        levels: list[int],
        label: str,
        weight: str | None,
        max_bin: int,
        threads: int,
        sample_rows: int | None,
        seed: int,
        ref_handle: int | None,
        read_ahead: int,
        ipc: tuple[list[str], list[list[str] | None]] | None = None,
        margin: str | None = None,
        margin_width: int | None = None,
    ) -> tuple[int, dict[str, float]]: ...

class GbdtLightgbm:
    def __init__(self, path: str, version: str) -> None: ...
    @property
    def has_arrow(self) -> bool: ...
    def dataset(
        self,
        batches: Callable[[], Iterator[pl.DataFrame]],
        names: list[str],
        kinds: list[str],
        levels: list[int],
        label: str,
        weight: str | None,
        params: str,
        threads: int,
        rows: int,
        in_memory: bool,
        reference: int | None,
        read_ahead: int,
        ipc: tuple[list[str], list[list[str] | None]] | None = None,
        margin: str | None = None,
        margin_width: int | None = None,
    ) -> tuple[int, dict[str, float]]: ...

class GbdtSampler:
    def __init__(
        self,
        names: list[str],
        kinds: list[str],
        levels: list[int],
        label: str,
        weight: str | None,
        sample_rows: int,
        seed: int,
        threads: int,
        margin: str | None = None,
    ) -> None: ...
    def push(self, df: pl.DataFrame) -> None: ...
    @property
    def rows_seen(self) -> int: ...
    def ref_matrix(self, max_bin: int) -> tuple[bytes, int, int]: ...

def gbdt_available_memory() -> int: ...
