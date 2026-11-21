"""Yanked versions in the two lock files.

A yank withdraws one release from a registry: the maintainer says it is
broken and nobody should resolve to it, though a lock that already names it
keeps building. GitHub's advisory database (Dependabot alerts) covers
security advisories only and has no yank alerts, so nothing else says so.

Every registry package in ``Cargo.lock`` is looked up in crates.io's sparse
index (``https://index.crates.io/``, one JSON line per version with a
``yanked`` flag), and every registry package in ``uv.lock`` in PyPI's JSON
API (``https://pypi.org/pypi/<name>/<version>/json``, whose ``info.yanked``
is the flag). It prints each yanked locked version and exits 1 if there is
one; a registry that cannot be reached is an error, not a pass. Standard
library only::

    python3 scripts/yanks.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CRATES_INDEX = "https://index.crates.io"
PYPI = "https://pypi.org/pypi"

#: ``fetch(url)`` returns the body, or None for a 404 (a package the registry
#: does not hold, a git or path dependency's name clashing with none).
Fetch = Callable[[str], str | None]


def http_fetch(url: str) -> str | None:
    """The body of ``url``; None on a 404; an exception on anything else."""
    req = urllib.request.Request(url, headers={"User-Agent": "lock-yank-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return None
        raise


def _locked(text: str, registry_prefix: str) -> list[tuple[str, str]]:
    """The (name, version) of every package whose source is the registry."""
    out = []
    for pkg in tomllib.loads(text).get("package", []):
        source = pkg.get("source")
        registry = source if isinstance(source, str) else (source or {}).get("registry", "")
        if registry.startswith(registry_prefix) and "version" in pkg:
            out.append((pkg["name"], pkg["version"]))
    return sorted(set(out))


def cargo_locked(text: str) -> list[tuple[str, str]]:
    """Registry crates in a ``Cargo.lock``."""
    return _locked(text, "registry+https://github.com/rust-lang/crates.io-index")


def uv_locked(text: str) -> list[tuple[str, str]]:
    """Registry packages in a ``uv.lock`` (PyPI)."""
    return _locked(text, "https://pypi.org/simple")


def index_path(name: str) -> str:
    """The crates.io sparse-index path of a crate."""
    n = name.lower()
    if len(n) <= 2:
        return f"{len(n)}/{n}"
    if len(n) == 3:
        return f"3/{n[0]}/{n}"
    return f"{n[:2]}/{n[2:4]}/{n}"


def cargo_yanked(name: str, version: str, fetch: Fetch) -> bool:
    body = fetch(f"{CRATES_INDEX}/{index_path(name)}")
    if body is None:
        raise RuntimeError(f"crates.io index has no crate {name}")
    for line in body.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["vers"] == version:
            return bool(row.get("yanked"))
    raise RuntimeError(f"crates.io index has no {name} {version}")


def pypi_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def pypi_yanked(name: str, version: str, fetch: Fetch) -> bool:
    body = fetch(f"{PYPI}/{pypi_name(name)}/{version}/json")
    if body is None:
        raise RuntimeError(f"PyPI has no {name} {version}")
    return bool(json.loads(body)["info"].get("yanked"))


def find_yanks(cargo_lock: str, uv_lock: str, fetch: Fetch, workers: int = 16) -> list[str]:
    """Each yanked locked version as ``"<registry> <name> <version>"``, sorted."""
    jobs = [("crates.io", n, v, cargo_yanked) for n, v in cargo_locked(cargo_lock)] + [
        ("PyPI", n, v, pypi_yanked) for n, v in uv_locked(uv_lock)
    ]

    def one(job: tuple[str, str, str, Callable[[str, str, Fetch], bool]]) -> str | None:
        registry, name, version, check = job
        return f"{registry} {name} {version}" if check(name, version, fetch) else None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sorted(r for r in pool.map(one, jobs) if r)


def main() -> int:
    cargo = (REPO / "Cargo.lock").read_text(encoding="utf-8")
    uv = (REPO / "uv.lock").read_text(encoding="utf-8")
    yanked = find_yanks(cargo, uv, http_fetch)
    n = len(cargo_locked(cargo)) + len(uv_locked(uv))
    lines = [f"checked {n} locked registry packages; {len(yanked)} yanked"]
    lines += [f"YANKED: {y}" for y in yanked]
    print("\n".join(lines))
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    return 1 if yanked else 0


if __name__ == "__main__":
    sys.exit(main())
