<!-- kind: record -->
# Linking and the interfaces a change rides on (CLAUDE.md rules 12 and 13)

Moved verbatim from CLAUDE.md on 2026-11-14 (the user: cut the size of the
docs read most). CLAUDE.md keeps the rules; this file keeps the evidence
that settled static linking and rule 13's account of each interface.

**Raised and resolved (2026-08-31): static linking here is what Polars
prescribes, not a shortcut.** Checked against Polars' own documentation rather
than reasoned about:

- The [User Guide](https://docs.pola.rs/user-guide/plugins/expr_plugins/) calls
  expression plugins "the preferred way to create user defined functions... The
  Polars engine will **dynamically link your function at runtime**". The dynamic
  link is the engine `dlopen`-ing our `cdylib`; the plugin itself carries its
  own Polars. The pyo3-polars README says so outright: "The plugin functions are
  **compiled separately**."
- Their canonical `Cargo.toml` is ours: `crate-type = ["cdylib"]`, a plain
  `polars` dependency, `pyo3` with `abi3`, `pyo3-polars` with `derive`.
- Their canonical `lib.rs` opens with
  `#[global_allocator] static ALLOC: PolarsAllocator = PolarsAllocator::new();`
  — **this is prescribed, not an optimisation.** It is the mechanism that keeps
  allocation coherent between the two copies of Polars, and the reason a
  statically linked plugin is safe rather than a double-free waiting to happen.
  Without it we silently ran on a second heap (and 43% slower).

So rule 12 is satisfied, and **stays** satisfied now that the plugin is gone
(task 85). The quotations above are the evidence that settled the question,
not a description of a surface we still ship; what they established still
holds of the wheel itself. `online-py` is a `cdylib` that Python `dlopen`s, it
carries its own statically linked Polars, and the `PolarsAllocator` above is
still what keeps allocation coherent between the two copies — which is why
that `#[global_allocator]` line must stay whatever else changes. Nothing on
crates.io publishes a `dylib` to link against anyway (0 of our 453
dependencies; `crate-type` is the publisher's choice).

**Rule 12, 2026-11-23: Polars' `timezones` feature, approved by the
user.** The workspace's `polars` dependency gains the features `timezones`
and `dtype-date`. Without `timezones`, polars-core 0.55.2 casts a column to
`Datetime(unit, Some(zone))` and returns it naive, without a word
(`chunked_array/cast.rs`, lines 106 and 327: the zoned arm is behind
`#[cfg(feature = "timezones")]`), while a column of nulls alone keeps the
zone (`series/mod.rs`, line 458, the all-null path of `Series::cast`), so
the two then fail to find a supertype. PLAN 556's age met it: a zoned clock
cast to its own dtype, less a column of nulls. `timezones` alone does not
compile polars-io, which then needs `dtype-date` too. It links nothing new:
`chrono-tz` 0.10.4 (the IANA zone data) was already linked, through
polars-arrow's own features (`cargo tree -i chrono-tz`); the lockfile gains
three dependency edges and no package (polars-io on `chrono-tz` and
`polars-time`, polars-json on `chrono-tz`). Measured on arm64, clean release
builds of the CLI and the wheel, one after the other at a load of 4.6 to
5.9: the CLI binary 64,440,384 to 65,440,672 bytes (+1.55%), the wheel
26,527,484 to 26,965,871 bytes to download (+1.65%) and 76,210,258 to
77,319,466 installed (+1.46%), the two builds 559 s to 566 s (+1.3%). No
output moved: the zoned-clock tests (`tests/test_zoned_clocks.py`) pass
before and after, every path that returns a clock keeping its zone without
a cast (merge, refresh_time, a bank's clocks, the window operators, a
checkpoint); the two casts to a zoned dtype the code makes, a held row's
value (`refresh/held.rs`) and a carried value (`merge.rs`), already set the
zone again.

13. **Know which of the interfaces a change rides on.** Four, and the three
    that are py-polars' to change carry no guarantee. (The expression plugin —
    which had a version handshake — was removed in task 85: it read every row
    by construction, which is the opposite of what this library is for.)
    - **PyO3 extension types** (`PyDataFrame`/`PySeries`, i.e. `ModelBank`) —
      the README states these "are however only provided for convenience and
      **do not have stability guarantees** beyond that the latest definitions
      should work for the latest version of Polars."

    - **IO plugin** (`lf.online.fit_predict(...)`, `polars.io.plugins.register_io_source`)
      — documented in the user guide but decorated `@unstable` in py-polars:
      "may be changed at any point without it being considered a breaking
      change". The bank is the source; polars does not re-apply the
      projection, predicate or slice it pushes into a Python source, so the
      source honours all three (`python/polars_online/_frame.py`).

    - **An expression's serialized form** (`expr.meta.serialize(format="json")`,
      the window formulas, task 143) — unstable across Polars versions, by
      Polars' own account. `python/polars_online/_formula.py` reads seven node
      kinds from it, under the caller's Polars, into this library's own tree,
      and refuses any other node by name. Its shapes were measured the same on
      1.34.0 and 1.44.2 (task 154).

    - **The Arrow PyCapsule interface** (`ModelBank.fit_predict_arrow`, task
      86) — the *output* side only, and the one path whose contract is not
      py-polars' to change: `__arrow_c_array__` is an Arrow specification,
      py-polars consumes it through the public `PySeries.from_arrow_c_array`,
      and pyarrow and duckdb consume it too. A break here would be an
      Arrow-level break rather than a polars one. Two caveats, both real: the
      *input* still arrives as a `PyDataFrame`, so a `fit_predict_arrow` call
      rides on this **and** on the first entry above; and which py-polars
      versions expose `from_arrow_c_array` has not been measured here, so the
      floor below is not known to hold for it.

    `polars>=1.37.0,<3` in `pyproject.toml` is therefore *measured* for all
    four but *guaranteed* for none below the latest — and the two that stream
    are among those — see `docs/RELEASE-READINESS.md`.
    Treat a `ModelBank`, IO-plugin or formula break on a new Polars as expected
    maintenance, not a surprise, and check those paths first. The floor is
    py-polars 1.37.0, the first whose `LazyFrame.collect_batches` (added in
    1.34.0), which the IO plugin reads with, stops reading ahead of a slower
    consumer: 1.34.0-1.36.1 held a whole 1 GB source (2026-10-12). That
    holds only where a source feeds `collect_batches` directly. Polars
    bounds what it keeps in flight by frames, not bytes, about ten frames a
    thread between a source and a node that runs on every thread
    (`with_columns`, a computing `select`), on 1.37.0 and 2.0 alike
    (2026-11-06, task 348): so a scan followed by `with_columns` in front of
    a slower `fit_predict` reads ahead (848 MB at 4M rows, against 279 MB
    for the same model in a loop over the bare scan), and this library's
    own sources hand Polars slices of at most 1 MiB so that two stages in
    series stay bounded. The loop is the remedy for a Polars scan;
    `docs/PERFORMANCE.md` §47 measures both.
    `ModelBank` alone works from 1.28.1.
