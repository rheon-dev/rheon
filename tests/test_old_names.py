"""The spellings the library had before it was renamed `rheon` appear only where
they are meant to: in the history, in the environment fallback that still reads
them, and in the files that test that they are still read.

A rule is a spelling, the trees it is held over, and the files that may keep it.
`git ls-files` and the untracked files that are not ignored are scanned, so a
new file is held to the rule before it is committed.

The rules start with the names the code carries (the warnings, the frame
namespace, the environment variables, the `po.` that prefixed a message). The
reader-facing prose (the README, the guides, `llms.txt`), the command line's
own name and the package's URLs follow in their own commits, and each adds its
rule here as it lands: `DEFERRED` names what is not yet scanned, so that the
list of what remains is in one place.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parent.parent

#: Paths no rule scans: the history, the maintainers' coordination files, and
#: what is not text.
HISTORY = (
    "docs/records/*",
    # The records filed beside the guides (docs/README.md, *Records*), kept as
    # they were written.
    "docs/ARROW-SOURCES.md",
    "docs/BOOSTED-TREES.md",
    "docs/CLUSTERING.md",
    "docs/ENHANCEMENTS.md",
    "docs/GBDT-ADAPTER.md",
    "docs/HUBER.md",
    "docs/IMPROVEMENTS.md",
    "docs/MARGINAL-LAGS-AND-BINS.md",
    "docs/PLAN.md",
    "docs/README-ITERATIONS.md",
    "docs/REVIEW-*.md",
    "docs/SIMPLIFICATION.md",
    "docs/WARMUP-AND-CONVERGENCE.md",
    ".claude/*",
    "uv.lock",
    "Cargo.lock",
    "*.parquet",
    "*.arrow",
    "*.so",
    "*.pyc",
)

#: Not yet scanned by any rule: the prose that explains the library to a reader
#: (rewritten in its own commit), and the changelog's past sections.
DEFERRED = (
    "CHANGELOG.md",
    "CLAUDE.md",
)

#: The guides. Their links, namespace, variables and warnings are renamed and
#: held; the prose that names the project `polars-online` is not yet, so the
#: two rules of the package's and the project's names do not read them.
GUIDES = (
    "docs/*.md",
    "docs/how-to/*",
    "docs/tutorials/*",
    "docs/explanation/*",
    "docs/contributing/*",
)

#: The guide that lists the old names beside the new ones.
NAMES_THE_OLD = ("docs/VERSIONS.md",)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    #: Globs of the files held to the rule (everything else is not scanned).
    trees: tuple[str, ...]
    #: Globs of the files that may keep the spelling, and why.
    allowed: tuple[str, ...] = ()


#: The files that read the old environment variables by their old names, and
#: the tests of that.
ENV_FALLBACK = (
    "python/rheon/_env.py",
    "python/rheon/_warnings.py",  # RheonDeprecationWarning's docstring names them
    "crates/rheon-polars/src/env.rs",
    "crates/rheon-polars/src/pool.rs",  # THREADS_VAR's doc names the old one
    "crates/rheon-polars/src/timing.rs",  # so does TIMING_VAR's
    "crates/rheon-cli/tests/surface.rs",
    "tests/test_env_rename.py",
    "tests/test_api_surface.py",  # reads the pairs out of the two tables
    "tests/api_surface.txt",  # and lists them
    "tests/test_old_names.py",
    "scripts/rheon_import.py",  # picks the old name for a benchmark run on a release before 2.0
)

#: The files that name a release before 2.0 on purpose: it is the project
#: `polars-online` and the package `polars_online`, and the scripts and tests
#: that compare with one, or read one's state, import it by that name.
RELEASE_BEFORE_2 = (
    "scripts/compare_release.py",
    "scripts/release_probe.py",
    "scripts/rheon_import.py",
    "scripts/bench_header.py",
    "scripts/release_version.py",  # reads either name in the docs' pins
    "tests/data.py",
    "tests/test_released_state.py",
    "tests/test_release_workflow.py",
    "tests/test_old_names.py",
)

#: The file formats keep the magic bytes 1.x wrote, so every 1.x state loads
#: under 2.0 with no loader: these files write, read or corrupt them.
MAGIC = (
    "crates/rheon-polars/src/bank.rs",
    "crates/rheon-polars/src/refresh.rs",
    "crates/rheon-polars/src/windows_frame.rs",
    "crates/rheon-polars/tests/state_portability.rs",
    "tests/test_production_hardening.py",
    "tests/test_windows.py",
)

#: Not renamed yet: each still names the old project or links its old home.
NOT_YET = (
    ".github/ISSUE_TEMPLATE/config.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/polars-canary.yml",
    "examples/bank.toml",
    "examples/pathway_integration.py",
    "scripts/clustering_proto.py",
    "scripts/ogbt_proto.py",
    "scripts/doc_review.py",
    "scripts/validate.py",
    "tests/test_reference_pages.py",
)

CODE = (
    "python/*",
    "crates/*",
    "tests/*",
    "scripts/*",
    "examples/*",
    ".github/*",
    "docs/reference/*",
    "pyproject.toml",
    "Cargo.toml",
    "budgets.toml",
    "writing_ratchet.toml",
)

#: The reader-facing files rewritten under the new name, held to the rules
#: of the package's and the project's names beside the code.
PROSE = ("README.md", "llms.txt", "CONTRIBUTING.md", "SECURITY.md", "CITATION.cff")

RULES = (
    Rule(
        "the package is imported as `rheon`; `polars_online` names only a release before 2.0",
        re.compile(r"polars_online"),
        CODE + PROSE,
        allowed=RELEASE_BEFORE_2,
    ),
    Rule(
        "the project is `rheon`; `polars-online` names only a release before 2.0",
        re.compile(r"polars-online"),
        CODE + PROSE,
        allowed=(
            *RELEASE_BEFORE_2,
            *MAGIC,
            *NOT_YET,
            # The note under the title says what the package was called.
            "README.md",
            "llms.txt",
        ),
    ),
    Rule(
        "the crates are `rheon-core`, `rheon-polars`, `rheon-cli`, `rheon-py` and `rheon-bench`",
        re.compile(r"(?<![A-Za-z0-9_-])online-(?:core|polars|py|cli|bench)(?![A-Za-z0-9_])"),
        CODE + PROSE + GUIDES,
        allowed=("tests/test_old_names.py", *NAMES_THE_OLD),
    ),
    Rule(
        "the project's home is github.com/rheon-dev/rheon and rheon-dev.github.io/rheon",
        re.compile(r"hgilde"),
        CODE + PROSE + GUIDES,
        allowed=("tests/test_old_names.py", *NOT_YET, *NAMES_THE_OLD),
    ),
    Rule(
        "the warning classes were renamed RheonWarning and RheonDeprecationWarning",
        re.compile(r"PolarsOnline(?:Deprecation)?Warning|__polars_online_renamed__"),
        CODE + PROSE + GUIDES,
        allowed=("tests/test_old_names.py", *NAMES_THE_OLD),
    ),
    Rule(
        "the frame namespaces are LazyFrameRheonNamespace and DataFrameRheonNamespace",
        re.compile(r"(?:Lazy|Data)FrameOnlineNamespace"),
        CODE,
        allowed=("tests/test_old_names.py",),
    ),
    Rule(
        "the frame namespace is `.rheon`, registered by `rheon.polars`",
        # An accessor: `.online` after a name, a call or a subscript, as
        # `lf.online.fit_predict`, `df.online[...]`, `scan(...).online.`.
        re.compile(r"""(?<=[A-Za-z0-9_)\]"'])\\?\.online(?![A-Za-z0-9_-])"""),
        CODE + PROSE + GUIDES,
        allowed=(
            "tests/test_old_names.py",
            *NAMES_THE_OLD,
            # Lets a benchmark run under a release before 2.0, whose namespace is `.online`.
            "scripts/rheon_import.py",
        ),
    ),
    Rule(
        "the environment variables are RHEON_*; only the fallback reads the old names",
        re.compile(r"POLARS_ONLINE_"),
        CODE + PROSE + GUIDES,
        allowed=(*ENV_FALLBACK, *NAMES_THE_OLD),
    ),
    Rule(
        "a message names the path a user calls, `rp.` or `rheon.`, not `po.`",
        # `po.` at the start of a word: not `repo.`, `expo.` or a variable
        # called `po` in a Rust test (`po.link`).
        re.compile(r"(?<![A-Za-z0-9_])po\\?\.[A-Za-z_]"),
        ("python/*", "crates/*", "tests/test_*.py"),
        allowed=(
            "crates/rheon-core/src/sgd.rs",  # a local variable called `po`
            "tests/test_old_names.py",
            # Child interpreters that import the package as `po` (code in a
            # string), and the tests of documents that still name the alias.
            "tests/test_budgets_memory.py",
            "tests/test_eval_query.py",
            "tests/test_ffi_memory.py",
            "tests/test_hardening.py",
            "tests/test_kwargs_typing.py",
            "tests/test_marginal_shards.py",
            "tests/test_portability.py",
            "tests/test_production_hardening.py",
            "tests/test_pyarrow_interop.py",
            "tests/test_solve_pool_size.py",
            "tests/test_threads.py",
            "tests/test_released_state.py",
            "tests/test_api_links.py",
            "tests/test_llms_txt.py",
            "tests/test_calibration_page.py",
            "tests/test_diagnostics_page.py",
            "tests/test_doc_review.py",
            "tests/test_doc_structure.py",
            "tests/test_feature_index.py",
            "tests/test_readiness.py",
            "tests/test_gram_page.py",
            "tests/test_gbdt_page.py",
        ),
    ),
)


def tracked_and_new() -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        cwd=str(REPO),
        check=True,
    ).stdout
    return [p for p in out.splitlines() if (REPO / p).is_file()]


def matches(path: str, globs: tuple[str, ...]) -> bool:
    return any(fnmatch(path, g) for g in globs)


def hits(rule: Rule) -> list[str]:
    found = []
    for path in tracked_and_new():
        if matches(path, HISTORY) or matches(path, DEFERRED):
            continue
        if not matches(path, rule.trees) or matches(path, rule.allowed):
            continue
        try:
            text = (REPO / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for n, line in enumerate(text.splitlines(), 1):
            if rule.pattern.search(line):
                found.append(f"{path}:{n}: {line.strip()[:110]}")
    return found


@pytest.mark.parametrize("rule", RULES, ids=lambda r: r.name.split(" ")[1])
def test_an_old_spelling_appears_only_where_it_is_allowed(rule):
    found = hits(rule)
    assert not found, f"{rule.name}:\n" + "\n".join(found[:40])


def test_the_scan_sees_what_it_is_meant_to():
    """A scan that finds nothing because it scans nothing proves nothing: each
    rule's pattern matches a spelling it is meant to catch, and each allowed
    file exists, so a renamed file does not leave an exemption behind."""
    samples = (
        "import polars_online as po",
        "pip install polars-online",
        "crates/online-polars/src",
        "https://github.com/hgilde/polars-online",
        "PolarsOnlineWarning",
        "__polars_online_renamed__",
        "LazyFrameOnlineNamespace",
        "lf.online.fit_predict",
        "df\\.online\\.predict",
        "POLARS_ONLINE_TIMING",
        "po.spec.ewridge",
        r"po\.stream\.embargo",
    )
    caught = [s for s in samples if any(r.pattern.search(s) for r in RULES)]
    assert caught == list(samples)
    for s in (
        "repo.spec",
        "rheon.polars",
        "self.online_mode",
        "x.online-prefetch1",
        "rheon-polars",
        "the online-learning kernel",
        "github.com/rheon-dev/rheon",
    ):
        assert not any(r.pattern.search(s) for r in RULES), s
    files = set(tracked_and_new())
    for rule in RULES:
        for allowed in rule.allowed:
            assert allowed in files, f"{allowed} is allowed by a rule and is not a file"
    assert len(files) > 500
