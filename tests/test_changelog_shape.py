"""The CHANGELOG's shape: each release section has each kind of heading
once, and no paragraph is a fragment of another entry (a continuation line
after a blank line, cut from an entry copied twice). Both were in
`[Unreleased]` once: a second `### Changed` 800 lines under the first, and
two entries' tails left behind when the entries were rewritten (PLAN 578j,
DOC-2)."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

TIER = "essential"
CHANGELOG = Path(__file__).resolve().parent.parent / "CHANGELOG.md"


def sections(text: str) -> dict[str, list[str]]:
    """Each `## ` section's lines, by its heading."""
    out: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            current = line
            out[current] = []
        elif current is not None:
            out[current].append(line)
    return out


def repeated_headings(text: str) -> list[str]:
    found = []
    for release, lines in sections(text).items():
        counts = Counter(line for line in lines if line.startswith("### "))
        found += [f"{release}: {h} x{n}" for h, n in counts.items() if n > 1]
    return found


def fragments(text: str) -> list[str]:
    """Lines that continue a list entry's text (indented, starting with a
    lower-case word) right after a blank line: an entry's own second
    paragraph starts a sentence."""
    lines = text.splitlines()
    return [
        f"{i + 1}: {line.strip()[:80]}"
        for i, line in enumerate(lines)
        if i and not lines[i - 1].strip() and re.match(r"  +[a-z]", line)
    ]


def test_the_unreleased_section_has_each_heading_once():
    """The released sections stay as they were published (0.3.0 has two of
    three headings); the section a release will publish has each once."""
    found = repeated_headings(CHANGELOG.read_text(encoding="utf-8"))
    assert not [f for f in found if f.startswith("## [Unreleased]")], found


def test_no_paragraph_is_a_fragment_of_an_entry():
    assert not fragments(CHANGELOG.read_text(encoding="utf-8"))


def test_the_checks_see_what_they_are_meant_to():
    text = "## [1]\n\n### Changed\n\n- an entry\n  goes on.\n\n  goes on.\n\n### Changed\n"
    assert repeated_headings(text) == ["## [1]: ### Changed x2"]
    assert fragments(text) == ["8: goes on."]
    assert not fragments("## [1]\n\n- an entry:\n\n  ```\n  code\n  ```\n")
