"""No POSIX-only call in `scripts/` or `tests/` outside a platform guard.

Windows has no `os.getloadavg`, `os.uname`, `os.fork`, `os.wait4`,
`signal.SIGKILL` or `signal.alarm`, and no `resource`, `fcntl`, `pwd`,
`termios` or `pty` module. A script or test that reaches one unguarded fails
there with an AttributeError or ImportError, and CI's Windows legs report
about ninety minutes after a push: `scripts/compare_outputs.py`'s
`os.getloadavg` (review 459, A5 and C1) kept all four Windows legs of CI red
for two days. This walks the syntax trees on any platform, so the Mac's
tests catch what only Windows would have run into.

A use is guarded when it sits:

- inside an ``if``, ``while``, conditional expression or ``and`` whose test
  reads the platform (``sys.platform``, ``os.name``, ``platform.system()``)
  or asks ``hasattr``;
- after an ``if`` of that kind in the same block whose body returns, raises
  or continues (an early exit for the other platform);
- in the body of a ``try`` that catches ImportError, AttributeError, OSError
  or Exception;
- in a function or class skipped by a ``skipif`` that reads the platform, or
  in a module whose ``pytestmark`` does.

``getattr(os, "getloadavg", None)`` is not an attribute read and needs no guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]

#: Attributes Windows' `os` and `signal` lack.
POSIX_ATTRS = {
    "os": {
        "getloadavg", "uname", "fork", "forkpty", "setsid", "setpgrp", "setpgid", "getpgid",
        "killpg", "getuid", "geteuid", "getgid", "getegid", "getgroups", "wait3", "wait4",
        "WNOHANG", "sysconf", "nice", "mkfifo", "pathconf", "statvfs", "chown", "lchown",
        "getpriority", "setpriority", "sched_getaffinity", "sched_setaffinity",
    },
    "signal": {
        "SIGKILL", "SIGALRM", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGCHLD", "SIGSTOP",
        "SIGCONT", "SIGPIPE", "SIGTSTP", "alarm", "setitimer", "getitimer", "pause",
        "siginterrupt", "ITIMER_REAL", "pthread_kill", "pthread_sigmask",
    },
}  # fmt: skip
#: Modules Windows' Python does not have.
POSIX_MODULES = {"resource", "fcntl", "pwd", "grp", "termios", "tty", "pty", "posix", "crypt",
                 "syslog", "spwd", "nis", "readline"}  # fmt: skip
#: What an exception handler must catch for its ``try`` to guard a use.
GUARD_EXCEPTIONS = {"ImportError", "ModuleNotFoundError", "AttributeError", "OSError", "Exception",
                    "BaseException"}  # fmt: skip
PLATFORM_WORDS = ("sys.platform", "os.name", "platform.system", "hasattr(")


def _reads_platform(node: ast.AST) -> bool:
    text = ast.unparse(node)
    return any(w in text for w in PLATFORM_WORDS)


def _exits(body: list[ast.stmt]) -> bool:
    return bool(body) and isinstance(body[-1], (ast.Return, ast.Raise, ast.Continue))


def _catches(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = handler.type.elts if isinstance(handler.type, ast.Tuple) else [handler.type]
    return any(isinstance(n, ast.Name) and n.id in GUARD_EXCEPTIONS for n in names)


def _skipped(node: ast.AST) -> bool:
    decorators = getattr(node, "decorator_list", [])
    return any("skipif" in ast.unparse(d) and _reads_platform(d) for d in decorators)


def _module_skipped(tree: ast.Module) -> bool:
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in stmt.targets)
            and "skipif" in ast.unparse(stmt.value)
            and _reads_platform(stmt.value)
        ):
            return True
    return False


def _uses(tree: ast.AST) -> list[tuple[ast.AST, str]]:
    """Every POSIX-only use in ``tree``: the node and what it names."""
    found: list[tuple[ast.AST, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.attr in POSIX_ATTRS.get(node.value.id, ()):
                found.append((node, f"{node.value.id}.{node.attr}"))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in POSIX_MODULES:
                    found.append((node, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in POSIX_MODULES and node.level == 0:
                found.append((node, f"from {node.module} import"))
            for alias in node.names:
                if alias.name in POSIX_ATTRS.get(top, ()):
                    found.append((node, f"from {top} import {alias.name}"))
    return found


def _parents(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    return {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}


def _guarded(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    child = node
    parent = parents.get(child)
    while parent is not None:
        if isinstance(parent, (ast.If, ast.While, ast.IfExp)):
            if child is not parent.test and _reads_platform(parent.test):
                return True
        elif isinstance(parent, ast.BoolOp) and isinstance(parent.op, ast.And):
            at = parent.values.index(child) if child in parent.values else 0
            if any(_reads_platform(v) for v in parent.values[:at]):
                return True
        elif isinstance(parent, ast.Try) and child in parent.body:
            if any(_catches(h) for h in parent.handlers):
                return True
        elif isinstance(parent, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and _skipped(
            parent
        ):
            return True
        # An earlier statement of the same block that leaves on the other platform.
        for field in ("body", "orelse", "finalbody"):
            block = getattr(parent, field, None)
            if isinstance(block, list) and child in block:
                for stmt in block[: block.index(child)]:
                    if (
                        isinstance(stmt, ast.If)
                        and _reads_platform(stmt.test)
                        and _exits(stmt.body)
                    ):
                        return True
        child, parent = parent, parents.get(parent)
    return False


def unguarded(source: str) -> list[str]:
    """The POSIX-only uses of ``source`` outside a platform guard, as ``line: name``."""
    tree = ast.parse(source)
    if _module_skipped(tree):
        return []
    parents = _parents(tree)
    return [
        f"{getattr(n, 'lineno', '?')}: {name}"
        for n, name in _uses(tree)
        if not _guarded(n, parents)
    ]


def _files() -> list[Path]:
    out = []
    for top in ("scripts", "tests"):
        out += [p for p in sorted((REPO / top).rglob("*.py")) if "__pycache__" not in p.parts]
    return out


def test_no_posix_only_call_is_unguarded_in_scripts_or_tests():
    files = _files()
    assert len(files) > 200, f"only {len(files)} files found: the walk lost its way"
    bad = []
    for path in files:
        for hit in unguarded(path.read_text(encoding="utf-8")):
            bad.append(f"{path.relative_to(REPO).as_posix()}:{hit}")
    assert not bad, "POSIX-only calls outside a platform guard (Windows lacks them):\n" + "\n".join(
        bad
    )


@pytest.mark.parametrize(
    "source",
    [
        # The call that kept CI's Windows legs red (0fb0d16f).
        "import os\ndef load():\n    return os.getloadavg()[0]\n",
        "import resource\n",
        "from fcntl import flock\n",
        "import os, signal\nos.kill(1, signal.SIGKILL)\n",
        "import signal\nsignal.alarm(5)\n",
        "import os\nprint(os.uname().machine)\n",
        "from os import getloadavg\n",
        # A try that catches something else is no guard.
        "import os\ntry:\n    os.fork()\nexcept KeyError:\n    pass\n",
        # The guard's other branch is not guarded.
        "import os, sys\nif sys.platform == 'win32':\n    pass\nelse:\n    pass\nos.getloadavg()\n",
    ],
)
def test_the_walk_finds_an_unguarded_call(source):
    assert unguarded(source), source


@pytest.mark.parametrize(
    "source",
    [
        "import os, sys\nif sys.platform != 'win32':\n    os.getloadavg()\n",
        "import os\nif hasattr(os, 'getloadavg'):\n    x = os.getloadavg()\n",
        "import os\nx = os.getloadavg() if hasattr(os, 'getloadavg') else None\n",
        "import os\nroot = hasattr(os, 'geteuid') and os.geteuid() == 0\n",
        "try:\n    import resource\nexcept ImportError:\n    resource = None\n",
        "import os\ntry:\n    x = os.getloadavg()\n"
        "except (AttributeError, OSError):\n    x = None\n",
        "import os, sys\ndef f():\n    if sys.platform == 'win32':\n        return 0\n"
        "    return os.wait4(1, os.WNOHANG)\n",
        "import sys, pytest\npytestmark = pytest.mark.skipif(sys.platform == 'win32', reason='x')\n"
        "import pty\n",
        "import sys, pytest\n@pytest.mark.skipif(sys.platform == 'win32', reason='x')\n"
        "def test_x():\n    import termios\n",
        "import os\nf = getattr(os, 'getloadavg', None)\n",
    ],
)
def test_the_walk_passes_a_guarded_call(source):
    assert unguarded(source) == [], source
