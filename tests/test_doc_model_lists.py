"""The lists of models the guides and docstrings spell out, held to the builders.

A sentence that names "the models that take X" goes stale the day a new
model takes X, and nothing fails: the 2026-11-23 review found the hard
window's five models beside fourteen builders that take ``window_size``,
"the ten linear models" beside thirteen that take the residual switches,
and a list of models with no features that left six out (DOC-3, DOC-6).
Each list here is derived from the builders themselves -- the parameters
their signatures take, or the switches they accept when called -- and every
model so derived must be named where the text lists them.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from rheon import polars as rp
from rheon import spec as spec_module
from test_model_registry import MINIMAL

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
MODELS = REPO / "docs" / "MODELS.md"
FEATURES = REPO / "docs" / "FEATURES.md"
DIAGNOSTICS = REPO / "docs" / "DIAGNOSTICS.md"
FORGET = REPO / "docs" / "how-to" / "forget-rows-past-a-window.md"
TIME_AND_DECAY = REPO / "docs" / "explanation" / "time-and-decay.md"

_HELPERS = {"output_fields", "output_index", "coef_fields", "coef_index"}


def _builders() -> list[str]:
    return [
        name
        for name in spec_module.__all__
        if name not in _HELPERS and callable(getattr(spec_module, name))
    ]


def _taking(param: str) -> set[str]:
    """The builders whose signature names ``param``."""
    return {
        name
        for name in _builders()
        if param in inspect.signature(getattr(spec_module, name)).parameters
    }


def _named(text: str) -> set[str]:
    """Every name in backticks (single or double) in ``text``."""
    return set(re.findall(r"`+([a-z_]+)`+", text))


def _section(text: str, heading: str) -> str:
    """The body of the section under ``heading``, to the next heading."""
    start = text.index(heading)
    rest = text[start + len(heading) :]
    nxt = re.search(r"\n#+ ", rest)
    return rest[: nxt.start()] if nxt else rest


def test_the_hard_window_names_every_model_that_takes_one():
    """MODELS.md's *A hard window* names every builder that takes
    ``window_size``; it, FEATURES.md's row, the how-to and the explanation
    name every one that cuts it from snapshots (``snapshot_every``)."""
    windowed = _taking("window_size")
    snapshots = _taking("snapshot_every")
    assert snapshots <= windowed
    assert len(snapshots) > 1 and windowed - snapshots, "the split this test reads has changed"
    section = _section(MODELS.read_text(encoding="utf-8"), "#### A hard window\n")
    assert windowed <= _named(section), sorted(windowed - _named(section))
    row = next(
        line
        for line in FEATURES.read_text(encoding="utf-8").splitlines()
        if line.startswith("| a hard window")
    )
    assert windowed <= _named(row), sorted(windowed - _named(row))
    for page in (FORGET, TIME_AND_DECAY):
        text = page.read_text(encoding="utf-8")
        assert snapshots <= _named(text), (page.name, sorted(snapshots - _named(text)))


def test_the_spec_module_names_every_model_without_targets_or_features():
    """``rheon.spec``'s ``targets``, ``features`` entry names each builder
    that takes no ``targets`` (``audit`` and ``ew_class`` aside, which it
    names by the column they take instead) and each that takes no
    ``features``."""
    doc = " ".join((spec_module.__doc__ or "").split())
    no_target = doc[doc.index("A model with no target") :]
    no_target = no_target[: no_target.index("alone")]
    untargeted = (
        set(_builders()) - _taking("targets") - {"audit", "ew_class", "hoeffding_tree_classifier"}
    )
    assert untargeted <= _named(no_target), sorted(untargeted - _named(no_target))
    end = doc.index("take no features")
    no_features = doc[doc.rindex(". ", 0, end) : end]
    featureless = set(_builders()) - _taking("features") - {"audit"}
    assert featureless <= _named(no_features), sorted(featureless - _named(no_features))


def test_the_linear_models_of_the_diagnostics_are_those_that_take_the_residual_switches():
    """DIAGNOSTICS.md's list of *the linear models* is every builder that
    takes ``emit_sigma``, the switch every residual diagnostic shares."""
    takes = set()
    for name in sorted(MINIMAL):
        kw: dict[str, object] = {"targets": ["y"], "features": ["x0"], "half_life": 50.0}
        kw.update(MINIMAL[name])
        kw = {k: v for k, v in kw.items() if v is not None}
        try:
            getattr(rp.spec, name)("m", **kw, emit_sigma=True)
        except ValueError:
            continue
        takes.add(name)
    text = DIAGNOSTICS.read_text(encoding="utf-8")
    sentence = text[text.index("The linear models are") :]
    sentence = sentence[: sentence.index(". ")]
    assert takes == _named(sentence), (
        sorted(takes - _named(sentence)),
        sorted(_named(sentence) - takes),
    )
