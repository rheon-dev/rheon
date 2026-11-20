"""The API reference keeps every address a released README links.

The reference gives each public function and class a page of its own,
grouped by family on its module's page (``docs/reference/families.py``).
Before that, each module page documented its objects in place, so the README
of every release on PyPI links ids such as
``spec.html#rheon.spec.ewridge``, and keeps linking them after this
repository has moved on. Each module page therefore carries an anchor with
every id it once gave, beside a link to the object's own page. This builds
the reference and holds the README's links, and every placed object's
anchor, to the HTML.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

TIER = "essential"

ROOT = Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "docs" / "reference"
LINK = re.compile(r"https://rheon-dev\.github\.io/rheon/([\w-]+)\.html(?:#([\w.\-]+))?")

sys.path.insert(0, str(REFERENCE))
import families  # noqa: E402 -- the reference's own list, beside conf.py


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("html")
    run = subprocess.run(
        [sys.executable, "-m", "sphinx", "-W", "-q", str(REFERENCE), str(out)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return out


def _ids(page: Path) -> set[str]:
    return set(re.findall(r' id="([^"]+)"', page.read_text(encoding="utf-8")))


def _readme_links() -> list[tuple[str, str]]:
    """The links into the reference from the README and the pages that took
    its lists of calls: the feature index and MODELS.md's model table (task
    338 stage 8)."""
    texts = [
        (ROOT / path).read_text(encoding="utf-8")
        for path in ("README.md", "docs/FEATURES.md", "docs/MODELS.md")
    ]
    return [(page, anchor) for text in texts for page, anchor in LINK.findall(text)]


def test_the_readme_links_the_reference():
    """The links into the reference are found: the README's 36 when this was
    written, and with the feature index well over a hundred, every one to a
    module page."""
    links = _readme_links()
    assert len(links) >= 100, links
    assert sum(1 for _, anchor in links if anchor) >= 100, links


def test_every_link_the_readme_gives_lands_in_the_built_reference(built):
    missing = []
    for page, anchor in _readme_links():
        html = built / f"{page}.html"
        if not html.exists() or (anchor and anchor not in _ids(html)):
            missing.append(f"{page}.html#{anchor}")
    assert not missing, missing


#: A link into the reference anywhere in a guide: the page may sit under
#: ``api/``, the id is read to the first character no id holds, and a
#: template (``<name>``) is not a link.
ANY_LINK = re.compile(r"https://rheon-dev\.github\.io/rheon/([\w./-]+?)\.html#([\w.\-<>]+)")


def _guide_links() -> list[tuple[str, str, str]]:
    """Every ``page.html#id`` link in the README, ``llms.txt`` and every file
    under ``docs/`` that is not a record, the built reference or its output:
    (the file, the page, the id)."""
    files = [ROOT / "README.md", ROOT / "llms.txt"]
    for path in sorted((ROOT / "docs").rglob("*")):
        rel = path.relative_to(ROOT / "docs").parts
        if rel[0] in {"records", "reference", "_build"} or path.suffix not in {".md", ".txt"}:
            continue
        files.append(path)
    found = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        for page, anchor in ANY_LINK.findall(text):
            if "<" not in anchor:
                found.append((path.relative_to(ROOT).as_posix(), page, anchor))
    return found


def test_every_anchor_a_guide_links_exists_in_the_built_reference(built):
    """Names that need Polars are anchored ``rheon.polars.<name>``, which the
    flat ``rheon.<name>`` is not: a link written the flat way lands at the
    top of its page and no build says so. Every ``...html#id`` link in the
    README, ``llms.txt`` and the guides (not the records) is held to the
    built HTML."""
    links = _guide_links()
    assert len(links) >= 120, len(links)  # the scan reads the guides it is meant to
    ids: dict[str, set[str] | None] = {}
    missing = []
    for where, page, anchor in links:
        if page not in ids:
            html = built / f"{page}.html"
            ids[page] = _ids(html) if html.exists() else None
        if ids[page] is None or anchor not in ids[page]:
            missing.append(f"{where}: {page}.html#{anchor}")
    assert not missing, missing


def test_the_scan_of_the_guides_reads_what_it_is_meant_to():
    where = {w for w, _, _ in _guide_links()}
    assert {"README.md", "docs/FEATURES.md", "docs/how-to/run-a-bank-in-a-query.md"} <= where
    assert not any(w.startswith("docs/records/") for w in where), where
    assert ANY_LINK.findall(
        "(https://rheon-dev.github.io/rheon/api/rheon.spec.ewridge.html#rheon.spec.ewridge)"
    )
    assert not [a for _, _, a in _guide_links() if "<" in a]


def test_each_module_page_keeps_an_anchor_for_every_object_it_lists(built):
    """Every placed object, and every method and property of a class, keeps
    its old id on its module page, and that page links the object's own."""
    sys.path.insert(0, str(REFERENCE))
    import conf  # noqa: PLC0415 -- the build's own reading of the objects

    problems = []
    for fam in families.FAMILIES:
        page = built / f"{fam.page}.html"
        text = page.read_text(encoding="utf-8")
        ids = _ids(page)
        for name in fam.entries:
            obj = conf._object(name)
            kind = "class" if isinstance(obj, type) else "function"
            wanted = [name] + ([m for m, _ in conf._members(name)] if kind == "class" else [])
            problems += [f"{fam.page}.html#{i}" for i in wanted if i not in ids]
            target = f'href="api/{families.page_name(name, kind)}.html'
            if target not in text:
                problems.append(f"{fam.page}.html links no page for {name}")
            own = built / "api" / f"{families.page_name(name, kind)}.html"
            if not own.exists() or name not in _ids(own):
                problems.append(f"{name} has no page of its own")
    assert not problems, problems


def test_twins_of_another_case_get_two_files():
    """``Target`` and ``target`` would be one file on macOS and Windows."""
    assert families.page_name("rheon.Target", "class") == "rheon.Target-class"
    assert families.page_name("rheon.target", "function") == "rheon.target"
    assert families.page_name("rheon.polars.ModelBank", "class") == "rheon.polars.ModelBank"
