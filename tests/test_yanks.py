"""`scripts/yanks.py` and `.github/workflows/yanks.yml`: a locked version the
registry has yanked fails the check. Every test runs on canned registry
answers; none reaches the network."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("yanks", REPO / "scripts/yanks.py")
assert _spec and _spec.loader
yk = importlib.util.module_from_spec(_spec)
sys.modules["yanks"] = yk
_spec.loader.exec_module(yk)

CARGO_LOCK = """
version = 4

[[package]]
name = "rheon-py"
version = "2.0.0"

[[package]]
name = "good"
version = "1.0.0"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "bad-crate"
version = "0.2.1"
source = "registry+https://github.com/rust-lang/crates.io-index"

[[package]]
name = "gitdep"
version = "0.1.0"
source = "git+https://github.com/x/y#abc"
"""

UV_LOCK = """
version = 1

[[package]]
name = "rheon"
version = "2.0.0"
source = { editable = "." }

[[package]]
name = "Some_Pkg"
version = "3.1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "okpkg"
version = "1.0"
source = { registry = "https://pypi.org/simple" }
"""


def canned(url: str) -> str | None:
    if url == "https://index.crates.io/go/od/good":
        return json.dumps({"vers": "1.0.0", "yanked": False}) + "\n"
    if url == "https://index.crates.io/ba/d-/bad-crate":
        return (
            json.dumps({"vers": "0.2.0", "yanked": False})
            + "\n"
            + json.dumps({"vers": "0.2.1", "yanked": True})
            + "\n"
        )
    if url == "https://pypi.org/pypi/some-pkg/3.1/json":
        return json.dumps({"info": {"yanked": True}})
    if url == "https://pypi.org/pypi/okpkg/1.0/json":
        return json.dumps({"info": {"yanked": False}})
    raise AssertionError(url)


def test_locked_packages_are_registry_only() -> None:
    assert yk.cargo_locked(CARGO_LOCK) == [("bad-crate", "0.2.1"), ("good", "1.0.0")]
    assert yk.uv_locked(UV_LOCK) == [("Some_Pkg", "3.1"), ("okpkg", "1.0")]


@pytest.mark.parametrize(
    ("name", "path"),
    [("a", "1/a"), ("ab", "2/ab"), ("abc", "3/a/abc"), ("serde", "se/rd/serde")],
)
def test_index_path(name: str, path: str) -> None:
    assert yk.index_path(name) == path


def test_finds_exactly_the_yanked_versions() -> None:
    assert yk.find_yanks(CARGO_LOCK, UV_LOCK, canned) == [
        "PyPI Some_Pkg 3.1",
        "crates.io bad-crate 0.2.1",
    ]


def test_nothing_yanked_is_empty() -> None:
    clean = CARGO_LOCK.replace('"bad-crate"', '"good"').replace("0.2.1", "1.0.0")
    assert yk.find_yanks(clean, "version = 1\n", canned) == []


def test_unknown_version_is_an_error_not_a_pass() -> None:
    with pytest.raises(RuntimeError, match="no good 9.9.9"):
        yk.cargo_yanked("good", "9.9.9", canned)
    with pytest.raises(RuntimeError, match="no crate"):
        yk.cargo_yanked("good", "1.0.0", lambda _u: None)
    with pytest.raises(RuntimeError, match="PyPI has no"):
        yk.pypi_yanked("okpkg", "1.0", lambda _u: None)


def test_workflow_triggers_and_permissions() -> None:
    wf = yaml.safe_load((REPO / ".github/workflows/yanks.yml").read_text(encoding="utf-8"))
    on = wf.get("on", wf.get(True))
    assert "schedule" in on and "workflow_dispatch" in on
    assert set(on["push"]["paths"]) == {"Cargo.lock", "uv.lock"}
    assert wf["permissions"] == {}
    for job in wf["jobs"].values():
        assert job["permissions"] == {"contents": "read"}
        for step in job["steps"]:
            if "uses" in step:
                assert len(step["uses"].split("@")[1]) == 40, step
