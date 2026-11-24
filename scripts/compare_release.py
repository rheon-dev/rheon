"""This build's outputs against a released version's, bit for bit.

    uv run python scripts/compare_release.py                  # against the newest release on PyPI
    uv run python scripts/compare_release.py --against 0.9.1
    uv run python scripts/compare_release.py --report         # print; exit 0 whatever differs

CI runs it on every push and pull request, on Linux, macOS and Windows
(`.github/workflows/ci.yml`, `bits`), and step 1 of a release runs it
again (docs/contributing/cut-a-release.md). A test with a tolerance lets
a small numeric change through, so every output field is compared with
the last release's, bit for bit, on the same machine, where both builds
call the same math library. A field that differs is either a change the
CHANGELOG declares, or a regression, whose cause then gets a test.

**A declaration** is a clause in a CHANGELOG entry above the release
compared against (`[Unreleased]`, and while a release is cut, its own
section): ``Moves:`` and the fields it moves, each backticked as
``model.field``, the model by its builder's name, with ``*`` and ``?`` as
wildcards (`ewridge.pred_*`, `*.weight_sum`, `rcov.closed.*`). The rest of
the entry is the reason, and an entry with none fails. A field that moved,
went, or whose spec this build refuses, fails the run unless a declaration
covers it; a field or a spec that only this build has is an addition,
listed and passed. The probe also counts each hard path its workload is
built to reach (`release_probe.PATHS`), and a path counted 0 fails the
run: a comparison that no longer reaches a path does not compare it.

The released version is installed from PyPI into a cached venv under
`.cache/release-compare/`, beside the Polars this environment has, and
`scripts/release_probe.py` runs under both builds. A release before 2.0 is
the project `polars-online` and imports as `polars_online`; from 2.0 it is
`rheon` (`tests/data.py`, `release_project`), and the probe imports whichever
the venv has. Each run reports which
package it imported, so the comparison cannot quietly compare a
build with itself. A spec the released version refuses is listed as not
comparable, an addition, not as a difference.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from data import release_project  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PROBE = REPO / "scripts" / "release_probe.py"
CACHE = REPO / ".cache" / "release-compare"


def newest_release() -> str:
    """The newest release: `rheon`'s newest from 2.0 on if it has one, else
    `polars-online`'s newest (`rheon` also holds a 1.x forwarder, which is no
    build of this code)."""
    for project in ("rheon", "polars-online"):
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/json", timeout=30) as r:
                doc = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        version = doc["info"]["version"]
        if release_project(version) == project:
            return version
    raise SystemExit("no release found on PyPI under `rheon` or `polars-online`")


def _python_in(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def released_python(version: str) -> Path:
    """A venv holding the release `version` (`polars-online`, or `rheon` from
    2.0) and this environment's Polars.

    Built under a name of this process's own and renamed into place once
    installed, so a venv at the cached path is always a whole one: the test
    suite runs in several processes at once, and two of them may want the
    same release's venv. The first rename wins, and a later builder's copy
    is deleted. A venv's interpreter finds its packages from `pyvenv.cfg`
    beside it, so the rename moves a working environment."""
    venv = CACHE / version
    python = _python_in(venv)
    if not python.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        building = CACHE / f"{version}.{os.getpid()}.building"
        shutil.rmtree(building, ignore_errors=True)
        wanted = f"{sys.version_info.major}.{sys.version_info.minor}"
        subprocess.run(["uv", "venv", "-q", "--python", wanted, str(building)], check=True)
        subprocess.run(
            [
                "uv", "pip", "install", "-q", "--python", str(_python_in(building)),
                f"{release_project(version)}=={version}", f"polars=={pl.__version__}", "numpy",
            ],
            check=True,
        )  # fmt: skip
        try:
            building.rename(venv)
        except OSError:
            if not python.exists():
                raise
        shutil.rmtree(building, ignore_errors=True)
    return python


def probe(python: Path, tag: str) -> tuple[pl.DataFrame, dict]:
    out, manifest = CACHE / f"{tag}.parquet", CACHE / f"{tag}.json"
    # `-I`: no PYTHONPATH and no script directory on the path, so each build
    # imports the package its own environment installed.
    subprocess.run([str(python), "-I", str(PROBE), str(out), str(manifest)], check=True)
    return pl.read_parquet(out), json.loads(manifest.read_text(encoding="utf-8"))


def float_bits(s: pl.Series) -> tuple[np.ndarray, np.ndarray]:
    """Null mask and the raw bits, so -0.0 and a NaN's payload count."""
    return s.is_null().to_numpy(), s.fill_null(0.0).to_numpy().view(np.uint64)


def same(a: pl.Series, b: pl.Series) -> bool:
    # `coef` and the fields laid out like it were an `Array` in the builds
    # between task 244 and task 250, a `List` of the same numbers before and
    # after: compared as the list either holds.
    if isinstance(a.dtype, pl.Array) and isinstance(b.dtype, pl.List):
        a = a.cast(b.dtype)
    elif isinstance(b.dtype, pl.Array) and isinstance(a.dtype, pl.List):
        b = b.cast(a.dtype)
    if a.dtype != b.dtype:
        return False
    if a.dtype == pl.Float64:
        (na, va), (nb, vb) = float_bits(a), float_bits(b)
        return bool(np.array_equal(na, nb) and np.array_equal(va, vb))
    if isinstance(a.dtype, pl.List) and a.dtype.inner == pl.Float64:
        if not a.list.len().equals(b.list.len(), null_equal=True):
            return False
        return same(_explode(a), _explode(b))
    return a.equals(b, null_equal=True)


def _explode(s: pl.Series) -> pl.Series:
    """A list column's values, an empty list as one null: the keyword keeps
    that behaviour past polars 2.0, and py-polars before it, down to the
    floor 1.37.0, has no keyword and already does it."""
    try:
        return s.explode(empty_as_null=True)
    except TypeError:
        return s.explode()


def _dtype_change(a: pl.Series, b: pl.Series) -> str | None:
    """How the dtype moved, or None where `same` would compare the values
    as they are (an `Array` against a `List` of the same numbers)."""
    if {type(a.dtype), type(b.dtype)} == {pl.Array, pl.List}:
        return None
    if a.dtype == b.dtype:
        return None
    if isinstance(a.dtype, pl.Enum) and isinstance(b.dtype, pl.Enum):
        old, new = list(a.dtype.categories), list(b.dtype.categories)
        gone = [c for c in old if c not in new]
        added = [c for c in new if c not in old]
        if not gone and not added:
            return "Enum categories reordered"
        return f"Enum categories: gone {gone}, new {added}"
    return f"dtype {a.dtype} -> {b.dtype}"


def first_difference(a: pl.Series, b: pl.Series) -> str:
    dtype = _dtype_change(a, b)
    for i, (x, y) in enumerate(zip(a.to_list(), b.to_list(), strict=True)):
        if x != y and not (x != x and y != y):  # NaN == NaN here
            return f"row {i}: {x!r} -> {y!r}" + (f" ({dtype})" if dtype else "")
    # Every value equal as Python reads it: either the dtype moved (an
    # integer width, an Enum's categories), which the release compares too,
    # or the bits differ where `==` does not look (-0.0 against 0.0).
    if dtype:
        return f"{dtype}, every value equal"
    return "bits differ (e.g. -0.0 against 0.0)"


#: A declaration: ``Moves:`` and one or more backticked ``model.field``
#: patterns, inside a CHANGELOG entry whose own text is the reason.
MOVES = re.compile(r"Moves: (`[^`]+`(?:(?:,? and |, )`[^`]+`)*)")
#: A CHANGELOG section's heading, with its version or `Unreleased`.
HEADING = re.compile(r"^## \[([^\]]+)\]")


class Declaration(NamedTuple):
    """One ``Moves:`` clause: the patterns it names, as ``model.field``
    with ``*`` and ``?`` as wildcards, the entry's text without the clause,
    and the CHANGELOG line the entry starts on. A named tuple, not a
    dataclass: the tests load this script by its path, outside
    `sys.modules`, where a dataclass cannot be made."""

    patterns: tuple[str, ...]
    reason: str
    line: int

    def covers(self, kind: str, field: str) -> bool:
        return any(fnmatch.fnmatchcase(f"{kind}.{field}", p) for p in self.patterns)


def _key(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(p) for p in version.split("."))
    except ValueError:
        return None


#: A wildcard or a character class: what is left of a pattern's part once
#: these are taken out is what it names.
_WILD = re.compile(r"\[[^\]]*\]|[*?]")
#: Below this share of this build's specs also run by the release, the
#: comparison holds too little to pass on: most of the workload would go
#: unseen (review S18).
MIN_SHARE_COMPARED = 0.5


def names_nothing(pattern: str) -> bool:
    """Whether a ``model.field`` pattern is wildcards alone on both sides of
    its dot (``*.*``, ``?*.?*``, ``[a-z]*.[a-z]*``): it would declare every
    move of every field (review S18). ``*.weight_sum`` and ``kmeans.*`` each
    name something."""
    model, _, field = pattern.partition(".")
    return not _WILD.sub("", model) and not _WILD.sub("", field)


def unreleased(changelog: str, since: str) -> list[tuple[int, str]]:
    """The lines, with their numbers from 1, of every section above the
    one for the release `since`: `[Unreleased]`, and a version not yet on
    PyPI while a release is cut. The sections are newest first, so it stops
    at the first heading of `since` or any version before it."""
    lines: list[tuple[int, str]] = []
    reading = False
    released = _key(since)
    for number, line in enumerate(changelog.splitlines(), 1):
        heading = HEADING.match(line)
        if heading:
            version = _key(heading.group(1))
            if version is not None and released is not None and version <= released:
                break
            reading = True
            continue
        if reading:
            lines.append((number, line))
    return lines


def _entries(lines: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Each list item or paragraph of `lines`, joined to one line, with the
    number of the line it starts on. Fenced code is left out."""
    entries: list[tuple[int, list[str]]] = []
    fenced = False
    for number, line in lines:
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        if not line.strip() or line.startswith(("#", "- ", "* ")):
            entries.append((number, []))
        if line.strip() and not line.startswith("#"):
            if not entries:
                entries.append((number, []))
            entries[-1][1].append(line.strip())
    return [(n, " ".join(text)) for n, text in entries if text]


def declarations(changelog: str, since: str) -> tuple[list[Declaration], list[str]]:
    """The ``Moves:`` declarations above the release `since`, and what is
    wrong with any of them: a pattern that is not ``model.field``, or an
    entry that gives no reason beside the clause."""
    found: list[Declaration] = []
    errors: list[str] = []
    # A version the read cannot order never ends it, so every past
    # section's clauses would count (review S18).
    if _key(since) is None:
        errors.append(f"the release {since!r} is not a version of dotted whole numbers")
    for number, line in enumerate(changelog.splitlines(), 1):
        heading = HEADING.match(line)
        if heading and heading.group(1) != "Unreleased" and _key(heading.group(1)) is None:
            errors.append(
                f"CHANGELOG.md:{number}: the section {heading.group(1)!r} is neither "
                "`Unreleased` nor a version of dotted whole numbers"
            )
    for number, text in _entries(unreleased(changelog, since)):
        for clause in MOVES.finditer(text):
            patterns = tuple(re.findall(r"`([^`]+)`", clause.group(1)))
            for pattern in patterns:
                if "." not in pattern or " " in pattern:
                    errors.append(
                        f"CHANGELOG.md:{number}: `{pattern}` is not `model.field` "
                        "(a model's builder, a dot, and an output field; * and ? as wildcards)"
                    )
                elif names_nothing(pattern):
                    errors.append(
                        f"CHANGELOG.md:{number}: `{pattern}` is wildcards alone, so it would "
                        "declare every field of every model: name the model or the field"
                    )
            reason = MOVES.sub("", text).lstrip("-* ").strip(" .")
            words = re.findall(r"[A-Za-z]{2,}", reason)
            if len(words) < 3:
                errors.append(
                    f"CHANGELOG.md:{number}: the entry declaring {', '.join(patterns)} "
                    "gives no reason: say what moved the numbers, and by how much"
                )
            found.append(Declaration(patterns, reason, number))
    return found, errors


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--against", help="the released version; default the newest on PyPI")
    ap.add_argument("--report", action="store_true", help="exit 0 whatever differs")
    ap.add_argument(
        "--changelog",
        type=Path,
        default=REPO / "CHANGELOG.md",
        help="where the declarations are read (default: the checkout's CHANGELOG.md)",
    )
    args = ap.parse_args()
    version = args.against or newest_release()
    CACHE.mkdir(parents=True, exist_ok=True)

    old, old_meta = probe(released_python(version), f"released-{version}")
    new, new_meta = probe(Path(sys.executable), "this-build")
    declared, errors = declarations(args.changelog.read_text(encoding="utf-8"), version)
    lines: list[str] = []
    code = compare(version, old, old_meta, new, new_meta, lines, declared, errors)
    text = "\n".join(lines)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            verdict = {0: "passed", 1: "FAILED", 2: "compared nothing"}[code]
            title = f"## Against {release_project(version)} {version}, bit for bit: {verdict}"
            fh.write(f"{title}\n\n```text\n{text}\n```\n")
    return 0 if args.report and code == 1 else code


def compare(
    version: str,
    old: pl.DataFrame,
    old_meta: dict,
    new: pl.DataFrame,
    new_meta: dict,
    out: list[str],
    declared: list[Declaration] | None = None,
    errors: list[str] | None = None,
) -> int:
    """Write the comparison into `out`: 0 identical, or every field that
    moved declared; 1 a field moved or went that no declaration covers, a
    declaration is malformed or names no model the workload runs, or a path
    of the workload went unreached, or the release ran under
    ``MIN_SHARE_COMPARED`` of this build's specs; 2 nothing compared (no
    field compared and none gone). A field or a spec
    this build has and the release does not is an addition, listed and
    passed."""
    print = out.append  # noqa: A001 -- every line goes to the report
    declared = declared or []
    errors = list(errors or [])
    old_name = release_project(old_meta["version"])
    print(f"released: {old_name} {old_meta['version']} from {old_meta['file']}")
    print(f"this:     rheon {new_meta['version']} from {new_meta['file']}")
    if old_meta["file"] == new_meta["file"]:
        print("both runs imported the same package; nothing was compared")
        return 2

    kinds = {**old_meta.get("kinds", {}), **new_meta.get("kinds", {})}
    known = set(kinds.values())
    for d in declared:
        for pattern in d.patterns:
            kind = pattern.split(".", 1)[0]
            if "." in pattern and not any(fnmatch.fnmatchcase(k, kind) for k in known):
                errors.append(
                    f"CHANGELOG.md:{d.line}: `{pattern}` names no model the workload runs "
                    f"(its models: {', '.join(sorted(known))})"
                )

    specs = [s for s in new_meta["ran"] if s in old_meta["ran"]]
    not_comparable = sorted(set(new_meta["ran"]) - set(old_meta["ran"]))
    # Each a (spec, field, what happened) that a declaration must cover.
    moved: list[tuple[str, str, str]] = []
    added: list[str] = []
    # A spec the release ran and this build refuses is every field of it
    # gone, not a silence: refused here, it would drop out of `specs` above
    # and the verdict would say "identical" (task 160, SC1).
    for spec in [s for s in old_meta["ran"] if s not in new_meta["ran"]]:
        why = new_meta.get("refused", {}).get(spec, "not run by this build")
        fields = [c.split(".", 1)[1] for c in old.columns if c.split(".", 1)[0] == spec]
        for field in fields or ["*"]:
            moved.append((spec, field, f"refused by this build: {why}"))
    compared = 0
    for col in new.columns:
        spec, field = col.split(".", 1)
        if spec not in specs:
            continue
        if col not in old.columns:
            added.append(col)
            continue
        compared += 1
        if not same(old[col], new[col]):
            moved.append((spec, field, first_difference(old[col], new[col])))
    for col in old.columns:
        spec, field = col.split(".", 1)
        if spec in specs and col not in new.columns:
            moved.append((spec, field, "gone from this build"))

    print(f"\n{compared} fields of {len(specs)} specs compared bit for bit over {new.height} rows")
    ours = len(new_meta["ran"])
    if ours and len(specs) < MIN_SHARE_COMPARED * ours:
        errors.append(
            f"the release ran {len(specs)} of this build's {ours} specs: under "
            f"{MIN_SHARE_COMPARED:.0%}, too few to compare"
        )
    if added or not_comparable:
        print(f"\nadded, not compared ({len(added)} fields, {len(not_comparable)} specs):")
        for col in added:
            print(f"  {col}: new in this build")
        for spec in not_comparable:
            print(f"  {spec}: refused by {version}: {old_meta['refused'].get(spec, '?')}")

    reached = new_meta.get("activations")
    if reached is not None:
        was = old_meta.get("activations") or {}
        print(f"\npaths the workload reached, counted (this build / {version}):")
        for path, count in reached.items():
            print(f"  {path}: {count} / {was.get(path, '-')}")
            if not count:
                errors.append(f"the workload no longer reaches {path}: counted {count}")
        for spec, why in new_meta.get("unmeasured", {}).items():
            errors.append(f"{spec}: its paths could not be counted: {why}")

    covered: dict[tuple[str, str, str], Declaration] = {}
    for item in moved:
        spec, field, _ = item
        for d in declared:
            if d.covers(kinds.get(spec, spec), field):
                covered[item] = d
                break
    undeclared = [m for m in moved if m not in covered]
    if covered:
        print(f"\nmoved, as the CHANGELOG declares ({len(covered)} fields):")
        for (spec, field, what), d in covered.items():
            print(f"  {spec}.{field} ({kinds.get(spec, spec)}): {what}")
            print(f"    CHANGELOG.md:{d.line}: {d.reason[:160]}")
    unused = [d for d in declared if d not in covered.values()]
    for d in unused:
        print(f"\ndeclared, nothing moved: {', '.join(d.patterns)} (CHANGELOG.md:{d.line})")
    if errors:
        print("\nFAILED:")
        for e in errors:
            print(f"  {e}")
    if compared == 0 and not moved:
        # Nothing on either side was read: no field compared, none gone.
        print(f"\nno field compared with {version}: nothing was compared")
        return 2
    if not undeclared:
        if not errors:
            print(f"\nidentical to {version}" if not covered else "\nevery move declared")
        return 1 if errors else 0
    print(f"\nDIFFERENT from {version}, not declared, in {len(undeclared)} fields:")
    by_spec: dict[str, list[str]] = {}
    for spec, field, what in undeclared:
        by_spec.setdefault(spec, []).append(f"{spec}.{field}: {what}")
    for spec, found in by_spec.items():
        print(f"  {spec} ({kinds.get(spec, spec)})")
        for line in found:
            print(f"    {line}")
    print(
        "\nDeclare each in the CHANGELOG entry that moved it, ending the entry with "
        "`Moves:` and the fields as `model.field` (CHANGELOG.md's header), "
        "or find the cause and pin it with a test."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
