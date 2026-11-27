"""The import-time CPU check of the x86-64 wheels.

The x86-64 wheels of ``rheon`` are compiled for CPUs with AVX2, FMA
and the rest of Polars' feature set, and a CPU without one dies on an
illegal instruction. ``rheon/_cpu_check.py`` reads the CPU's
features in pure Python, first thing in ``__init__``, and warns. These
tests hold the pieces: the build script that records what a build needs
(``crates/rheon-py/build.rs``, compiled and run for real), the comparison
and the warning (against a faked CPU, since the failing path cannot be
reached on a CPU that has every feature), the real reader (on an x86-64
host, against ``/proc/cpuinfo`` where there is one), and the order of the
imports, which is the point: the check runs before the extension module is
asked for.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import warnings
from pathlib import Path

import pytest
import yaml

from rheon import _cpu_check

TIER = "essential"

REPO = Path(__file__).resolve().parents[1]
BUILD_RS = REPO / "crates" / "rheon-py" / "build.rs"
RELEASE = yaml.safe_load((REPO / ".github" / "workflows" / "release.yml").read_text("utf-8"))

#: The flags the release gives the x86-64 wheels of `rheon`.
X86_FEATURES = RELEASE["env"]["X86_FEATURES"]

ON_X86 = platform.machine().lower() in {"x86_64", "amd64", "x64"} and sys.maxsize > 2**32


# --- the build script ----------------------------------------------------------


@pytest.fixture(scope="module")
def build_script(tmp_path_factory) -> Path:
    """build.rs compiled on its own: it uses the standard library alone."""
    rustc = shutil.which("rustc") or shutil.which("rustc", path=str(Path.home() / ".cargo" / "bin"))
    if rustc is None:
        pytest.fail("rustc is not on PATH; `source scripts/env.sh` first")
    out = tmp_path_factory.mktemp("build-rs") / ("build.exe" if os.name == "nt" else "build")
    res = subprocess.run(
        [rustc, "--edition", "2024", "-o", str(out), str(BUILD_RS)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert res.returncode == 0, res.stderr
    return out


def _run_build_script(
    exe: Path, root: Path, flags: list[str] | None, arch: str = "x86_64", python_dir: bool = True
) -> tuple[subprocess.CompletedProcess, Path]:
    """Run it in a copy of the layout (crates/rheon-py/, python/rheon/)
    and return the process and the file it writes."""
    manifest = root / "crates" / "rheon-py"
    manifest.mkdir(parents=True, exist_ok=True)
    package = root / "python" / "rheon"
    if python_dir:
        package.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CARGO", "RUSTFLAGS"))}
    env["CARGO_MANIFEST_DIR"] = str(manifest)
    env["CARGO_CFG_TARGET_ARCH"] = arch
    if flags is not None:
        env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join(flags)
    res = subprocess.run(
        [str(exe)], env=env, capture_output=True, text=True, encoding="utf-8", check=False
    )
    return res, package / "_build_flags.py"


#: The x86-64 psABI levels' features, in rustc's names, as ``build.rs``
#: records each (the levels of the System V AMD64 ABI supplement; ``lahf_lm``
#: has no rustc feature, and v4's are AVX-512's).
V2 = "+cmpxchg16b,+popcnt,+sse3,+sse4.1,+sse4.2,+ssse3"
V3 = V2 + ",+avx,+avx2,+bmi1,+bmi2,+f16c,+fma,+lzcnt,+movbe,+xsave"
V4 = V3 + ",+avx512f,+avx512bw,+avx512cd,+avx512dq,+avx512vl"
V3_NO_AVX2 = V3.replace(",+avx2", "")


def _recorded(path: Path) -> str:
    (value,) = re.findall(r'^BUILD_FEATURE_FLAGS = "([^"]*)"$', path.read_text("utf-8"), re.M)
    return value


@pytest.mark.parametrize(
    "flags, want",
    [
        (["-C", "target-feature=+avx2,+fma"], "+avx2,+fma"),
        (["-Ctarget-feature=+avx2,+fma"], "+avx2,+fma"),
        (["--codegen", "target-feature=+avx2"], "+avx2"),
        (["--codegen=target-feature=+avx2"], "+avx2"),
        # Every flag adds; the order is the order given.
        (["-C", "target-feature=+bmi1", "-C", "target-feature=+avx2,+fma"], "+bmi1,+avx2,+fma"),
        # A feature turned off is not needed, and crt-static is no CPU feature.
        (["-C", "target-feature=+avx2,-sse4a,+crt-static,-crt-static"], "+avx2"),
        (["-C", "target-feature=-crt-static"], ""),
        # Other flags do not name features.
        (["-C", "link-arg=-fuse-ld=lld", "-Zthreads=4"], ""),
        (["-C", "target-feature=+avx2", "-C", "target-cpu=native", "--cfg", "x"], "+avx2"),
        # A psABI level names its features (review 2026-11-23, PYO3-2): a
        # build for `x86-64-v3` recorded none, so its check said nothing.
        (["-C", "target-cpu=x86-64-v2"], V2),
        (["-C", "target-cpu=x86-64-v3", "-C", "link-arg=-fuse-ld=lld", "-Zthreads=4"], V3),
        (["-Ctarget-cpu=x86-64-v4"], V4),
        # A feature turned off after it was turned on is not needed, whatever
        # turned it on; one turned on again is.
        (["-C", "target-cpu=x86-64-v3", "-C", "target-feature=-avx2"], V3_NO_AVX2),
        (["-C", "target-feature=+avx2,+fma", "-C", "target-feature=-avx2"], "+fma"),
        (["-C", "target-feature=+avx2,-avx2,+avx2"], "+avx2"),
        (["-C", "target-cpu=x86-64-v2", "-C", "target-feature=+avx2"], V2 + ",+avx2"),
        ([], ""),
        (None, ""),
    ],
)
def test_the_build_script_records_the_features_rustc_was_told(build_script, tmp_path, flags, want):
    res, path = _run_build_script(build_script, tmp_path, flags)
    assert res.returncode == 0, res.stderr
    assert _recorded(path) == want
    # The three conditions that make cargo write it again.
    assert "cargo:rerun-if-changed=build.rs" in res.stdout
    assert "cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS" in res.stdout
    assert any(
        line.startswith("cargo:rerun-if-changed=") and line.endswith("_build_flags.py")
        for line in res.stdout.splitlines()
    ), res.stdout


@pytest.mark.parametrize("arch", ["aarch64", "arm", "riscv64"])
def test_a_build_that_is_not_for_x86_records_nothing(build_script, tmp_path, arch):
    res, path = _run_build_script(build_script, tmp_path, ["-C", "target-feature=+neon"], arch)
    assert res.returncode == 0, res.stderr
    assert _recorded(path) == ""


def test_the_recorded_file_names_the_features_and_the_python_reads_them_back(
    build_script, tmp_path
):
    """What the script writes is what ``_cpu_check`` reads: the release's own
    flag list, through both."""
    res, path = _run_build_script(build_script, tmp_path, ["-C", f"target-feature={X86_FEATURES}"])
    assert res.returncode == 0, res.stderr
    namespace: dict = {}
    exec(path.read_text("utf-8"), namespace)
    assert namespace["BUILD_FEATURE_FLAGS"] == X86_FEATURES
    assert _cpu_check.required_features(namespace["BUILD_FEATURE_FLAGS"]) == [
        f.removeprefix("+") for f in X86_FEATURES.split(",")
    ]


def test_a_file_that_is_current_is_left_alone_and_a_stale_one_is_rewritten(build_script, tmp_path):
    """Cargo must not find the file newer than its own record each time, so a
    build that records the same flags does not write it; one that records
    other flags does, and so does one whose file is gone."""
    flags = ["-C", "target-feature=+avx2"]
    _, path = _run_build_script(build_script, tmp_path, flags)
    os.utime(path, (1_000_000_000, 1_000_000_000))
    _run_build_script(build_script, tmp_path, flags)
    assert path.stat().st_mtime == 1_000_000_000
    _run_build_script(build_script, tmp_path, ["-C", "target-feature=+avx2,+fma"])
    assert _recorded(path) == "+avx2,+fma"
    path.unlink()
    _run_build_script(build_script, tmp_path, flags)
    assert _recorded(path) == "+avx2"


def test_a_layout_without_the_python_source_builds_and_records_nothing(build_script, tmp_path):
    res, path = _run_build_script(
        build_script, tmp_path, ["-C", "target-feature=+avx2"], python_dir=False
    )
    assert res.returncode == 0, res.stderr
    assert not path.exists()


# --- the comparison and the warning -----------------------------------------------


@pytest.fixture
def cpu(monkeypatch):
    """A CPU whose features a test sets, and a build whose flags it sets."""

    class Fake:
        def build(self, flags: str) -> None:
            monkeypatch.setattr(_cpu_check, "baked_flags", lambda: flags)

        def has(self, features: dict[str, bool] | None) -> None:
            monkeypatch.setattr(_cpu_check, "read_cpu_features", lambda: features)

    monkeypatch.delenv(_cpu_check.SKIP_ENV, raising=False)
    return Fake()


def _all(**off: bool) -> dict[str, bool]:
    return {name: off.get(name.replace(".", "_"), True) for name in _cpu_check.FEATURE_BITS}


def test_required_features_drop_the_plus_and_the_features_turned_off():
    assert _cpu_check.required_features("+sse3,+avx2,-sse4a,,+fma") == ["sse3", "avx2", "fma"]
    assert _cpu_check.required_features("") == []


def test_every_missing_feature_is_named_in_the_builds_order(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all(bmi1=False, fma=False, avx2=False))
    # The build names avx2, then fma, then bmi1.
    assert _cpu_check.missing_features() == ["avx2", "fma", "bmi1"]


def test_a_cpu_with_every_feature_gets_no_warning(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_missing_feature_warns_with_the_feature_the_remedy_and_the_way_out(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(_all(avx2=False, fma=False))
    with pytest.warns(RuntimeWarning, match="Missing required CPU features") as caught:
        _cpu_check.run()
    (warning,) = caught
    text = str(warning.message)
    assert "avx2, fma" in text
    # The package to install, which one to remove, Rosetta, and the variable.
    assert "rheon-compat" in text
    assert "pip uninstall rheon" in text
    assert "Rosetta" in text
    assert "RHEON_SKIP_CPU_CHECK" in text
    assert "crash" in text
    assert warning.category is RuntimeWarning


def test_the_variable_is_the_documented_one():
    assert _cpu_check.SKIP_ENV == "RHEON_SKIP_CPU_CHECK"


def test_the_variable_skips_the_check(cpu, monkeypatch):
    cpu.build(X86_FEATURES)
    cpu.has(_all(avx2=False))
    monkeypatch.setenv(_cpu_check.SKIP_ENV, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()
    # An empty value is no request to skip, as in Polars.
    monkeypatch.setenv(_cpu_check.SKIP_ENV, "")
    with pytest.warns(RuntimeWarning):
        _cpu_check.run()


@pytest.mark.parametrize("flags", ["", "-avx2", ","])
def test_a_build_that_needs_nothing_reads_nothing(cpu, monkeypatch, flags):
    """No CPU read at all on ARM, on the compat build and on a source build:
    the reader would raise if asked."""
    cpu.build(flags)

    def refuse() -> None:
        raise AssertionError("the CPU was read for a build that needs no feature")

    monkeypatch.setattr(_cpu_check, "read_cpu_features", refuse)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_cpu_that_cannot_be_read_is_no_reason_to_warn(cpu):
    cpu.build(X86_FEATURES)
    cpu.has(None)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _cpu_check.run()


def test_a_feature_the_check_cannot_read_is_not_reported_missing(cpu):
    cpu.build("+avx2,+avx512f")
    cpu.has(_all())
    assert _cpu_check.missing_features() == []


def test_a_build_with_no_target_feature_flag_has_nothing_to_check(monkeypatch):
    """Every development build, the compat wheel and a source build record no
    feature, whatever CPU they run on."""
    monkeypatch.setattr(_cpu_check, "baked_flags", lambda: "")
    assert _cpu_check.missing_features() == []


def test_no_record_at_all_is_no_check(monkeypatch):
    monkeypatch.setitem(sys.modules, "rheon._build_flags", None)  # import fails
    assert _cpu_check.baked_flags() == ""


def test_the_releases_features_are_all_features_the_check_can_read():
    """A feature the release builds for but the check cannot read would be a
    feature nothing checks."""
    names = [f.removeprefix("+") for f in X86_FEATURES.split(",")]
    assert set(names) == set(_cpu_check.FEATURE_BITS), set(names) ^ set(_cpu_check.FEATURE_BITS)
    assert len(set(names)) == len(names)


def test_the_cpu_is_never_read_off_x86(monkeypatch):
    monkeypatch.setattr(platform, "machine", lambda: "arm64")
    assert _cpu_check.read_cpu_features() is None
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    assert _cpu_check.read_cpu_features() is None


# --- OSXSAVE / XGETBV and Windows W^X (PYO3-2) -----------------------------------


def _all_cpuid_registers(
    *, osxsave: bool, off: set[str] | None = None
) -> dict[int, dict[str, int]]:
    """CPUID leaf values reporting every feature in ``FEATURE_BITS`` except
    ``off``, with ``OSXSAVE`` (leaf 1 ECX bit 27) set per ``osxsave``."""
    off = off or set()
    regs: dict[int, dict[str, int]] = {
        0: {"eax": 7, "ebx": 0, "ecx": 0, "edx": 0},
        0x80000000: {"eax": 0x80000001, "ebx": 0, "ecx": 0, "edx": 0},
        1: {"eax": 0, "ebx": 0, "ecx": (1 << 27) if osxsave else 0, "edx": 0},
        7: {"eax": 0, "ebx": 0, "ecx": 0, "edx": 0},
        0x80000001: {"eax": 0, "ebx": 0, "ecx": 0, "edx": 0},
    }
    for name, (leaf, reg, bit) in _cpu_check.FEATURE_BITS.items():
        if name not in off:
            regs[leaf][reg] |= 1 << bit
    return regs


def test_osxsave_off_hides_ymm_features_and_never_calls_xgetbv(monkeypatch):
    """When CPUID reports AVX/AVX2/FMA but ``OSXSAVE`` is 0, ``xgetbv`` must
    not be executed (it traps without OSXSAVE), YMM-dependent features read
    False, and non-YMM features (including SSE4.2, BMI1, BMI2, LZCNT) stay True."""
    regs = _all_cpuid_registers(osxsave=False)
    closed: list[str] = []

    class FakeCpuid:
        def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
            return regs[leaf]

        def close(self) -> None:
            closed.append("cpuid")

    def refuse_xgetbv():
        raise AssertionError("xgetbv must not be invoked when OSXSAVE is 0")

    monkeypatch.setattr(_cpu_check, "_is_x86_64", lambda: True)
    monkeypatch.setattr(_cpu_check, "_Cpuid", FakeCpuid)
    monkeypatch.setattr(_cpu_check, "_Xgetbv", refuse_xgetbv)

    got = _cpu_check.read_cpu_features()
    assert got is not None
    assert closed == ["cpuid"]
    for name in ("avx", "avx2", "fma"):
        assert got[name] is False, name
    for name in ("sse4.2", "bmi1", "bmi2", "lzcnt", "sse3", "ssse3", "sse4.1", "popcnt"):
        assert got[name] is True, name


@pytest.mark.parametrize("xcr0", [0x0, 0x1, 0x2, 0x3, 0x4, 0x5])
def test_xcr0_without_both_xmm_and_ymm_hides_ymm_features(monkeypatch, xcr0):
    """With ``OSXSAVE = 1`` but either XMM (bit 1) or YMM (bit 2) absent in
    XCR0, AVX/AVX2/FMA read False while non-YMM features stay True."""
    regs = _all_cpuid_registers(osxsave=True)
    xgetbv_calls: list[int] = []
    closed: list[str] = []

    class FakeCpuid:
        def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
            return regs[leaf]

        def close(self) -> None:
            closed.append("cpuid")

    class FakeXgetbv:
        def __call__(self, xcr: int = 0) -> int:
            xgetbv_calls.append(xcr)
            return xcr0

        def close(self) -> None:
            closed.append("xgetbv")

    monkeypatch.setattr(_cpu_check, "_is_x86_64", lambda: True)
    monkeypatch.setattr(_cpu_check, "_Cpuid", FakeCpuid)
    monkeypatch.setattr(_cpu_check, "_Xgetbv", FakeXgetbv)

    got = _cpu_check.read_cpu_features()
    assert got is not None
    assert xgetbv_calls == [0]
    assert set(closed) == {"cpuid", "xgetbv"}
    for name in ("avx", "avx2", "fma"):
        assert got[name] is False, (hex(xcr0), name)
    for name in ("sse4.2", "bmi1", "bmi2", "lzcnt"):
        assert got[name] is True, (hex(xcr0), name)


@pytest.mark.parametrize("xcr0", [0x6, 0x7, 0xE7])
def test_xcr0_with_xmm_and_ymm_enables_cpuid_ymm_features(monkeypatch, xcr0):
    """With ``OSXSAVE = 1`` and ``(XCR0 & 0x6) == 0x6``, AVX/AVX2/FMA match
    their CPUID bits (still False if CPUID itself omits one)."""
    regs = _all_cpuid_registers(osxsave=True, off={"avx2"})
    xgetbv_calls: list[int] = []
    closed: list[str] = []

    class FakeCpuid:
        def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
            return regs[leaf]

        def close(self) -> None:
            closed.append("cpuid")

    class FakeXgetbv:
        def __call__(self, xcr: int = 0) -> int:
            xgetbv_calls.append(xcr)
            return xcr0

        def close(self) -> None:
            closed.append("xgetbv")

    monkeypatch.setattr(_cpu_check, "_is_x86_64", lambda: True)
    monkeypatch.setattr(_cpu_check, "_Cpuid", FakeCpuid)
    monkeypatch.setattr(_cpu_check, "_Xgetbv", FakeXgetbv)

    got = _cpu_check.read_cpu_features()
    assert got is not None
    assert xgetbv_calls == [0]
    assert set(closed) == {"cpuid", "xgetbv"}
    assert got["avx"] is True
    assert got["fma"] is True
    assert got["avx2"] is False
    for name in ("sse4.2", "bmi1", "bmi2", "lzcnt"):
        assert got[name] is True, name


def test_xgetbv_machine_code_bytes_match_abis():
    """Verify the exact assembled System V AMD64 and Windows x64 ``xgetbv`` stubs."""
    # System V: mov %edi, %ecx (89 f9); xgetbv (0f 01 d0); ret (c3)
    assert _cpu_check._POSIX_XGETBV_CODE == b"\x89\xf9\x0f\x01\xd0\xc3"
    # Windows x64: xcr is already in %ecx; xgetbv (0f 01 d0); ret (c3)
    assert _cpu_check._WINDOWS_XGETBV_CODE == b"\x0f\x01\xd0\xc3"


class _FakeKernel32:
    """In-memory mock of ``kernel32.dll`` for testing Windows W^X allocation."""

    def __init__(self, *, protect_succeeds: bool = True) -> None:
        import ctypes

        self._ctypes = ctypes
        self._protect_succeeds = protect_succeeds
        self.allocated: dict[int, object] = {}
        self.alloc_calls: list[tuple[object, int, int, int]] = []
        self.protect_calls: list[tuple[int, int, int, bytes]] = []
        self.free_calls: list[tuple[int, int, int]] = []

        class _Fn:
            def __init__(self, fn):
                self._fn = fn
                self.argtypes = None
                self.restype = None

            def __call__(self, *args):
                return self._fn(*args)

        self.VirtualAlloc = _Fn(self._alloc)
        self.VirtualProtect = _Fn(self._protect)
        self.VirtualFree = _Fn(self._free)

    def _alloc(self, addr: object, size: int, alloc_type: int, protect: int) -> int:
        buf = self._ctypes.create_string_buffer(size)
        ptr = self._ctypes.addressof(buf)
        self.allocated[ptr] = buf
        self.alloc_calls.append((addr, size, alloc_type, protect))
        return ptr

    def _protect(self, addr: int, size: int, new_protect: int, old_protect_ptr: object) -> int:
        buf = self.allocated[addr]
        written = bytes(self._ctypes.string_at(addr, size))
        self.protect_calls.append((addr, size, new_protect, written))
        assert buf is not None
        assert old_protect_ptr is not None
        return 1 if self._protect_succeeds else 0

    def _free(self, addr: int, size: int, free_type: int) -> int:
        self.free_calls.append((addr, size, free_type))
        self.allocated.pop(addr, None)
        return 1


@pytest.mark.parametrize(
    ("cls", "expected_code"),
    [
        (_cpu_check._Cpuid, _cpu_check._WINDOWS_CODE),
        (_cpu_check._Xgetbv, _cpu_check._WINDOWS_XGETBV_CODE),
    ],
)
def test_windows_wx_allocates_rw_writes_code_then_protects_rx(monkeypatch, cls, expected_code):
    """Windows allocates ``PAGE_READWRITE`` (0x04), copies the code, then calls
    ``VirtualProtect`` with ``PAGE_EXECUTE_READ`` (0x20), and frees on close."""
    import ctypes

    kernel32 = _FakeKernel32(protect_succeeds=True)

    def fake_cdll(name: str, *args, **kwargs):
        assert name == "kernel32.dll"
        return kernel32

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)

    stub = cls()
    try:
        assert len(kernel32.alloc_calls) == 1
        _, size, alloc_type, fl_protect = kernel32.alloc_calls[0]
        assert size == len(expected_code)
        assert alloc_type == 0x3000
        assert fl_protect == 0x04  # PAGE_READWRITE, never PAGE_EXECUTE_READWRITE (0x40)

        assert len(kernel32.protect_calls) == 1
        p_addr, p_size, new_protect, written = kernel32.protect_calls[0]
        assert p_addr == stub.address
        assert p_size == len(expected_code)
        assert new_protect == 0x20  # PAGE_EXECUTE_READ
        assert written == expected_code  # memmove ran before VirtualProtect
        assert kernel32.VirtualProtect.argtypes is not None
        assert kernel32.VirtualProtect.restype == ctypes.c_long
        assert kernel32.free_calls == []
    finally:
        stub.close()

    assert kernel32.free_calls == [(stub.address, 0, 0x8000)]
    assert kernel32.allocated == {}


def test_windows_virtualprotect_failure_frees_memory_and_returns_none(monkeypatch):
    """If ``VirtualProtect`` fails on Windows, ``VirtualFree`` is called before
    raising so nothing leaks, and ``read_cpu_features()`` returns ``None`` cleanly."""
    import ctypes

    kernel32 = _FakeKernel32(protect_succeeds=False)

    def fake_cdll(name: str, *args, **kwargs):
        assert name == "kernel32.dll"
        return kernel32

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(_cpu_check, "_is_x86_64", lambda: True)

    with pytest.raises(OSError, match="executable"):
        _cpu_check._Cpuid()
    assert len(kernel32.free_calls) == 1
    assert kernel32.allocated == {}

    assert _cpu_check.read_cpu_features() is None
    assert len(kernel32.free_calls) == 2
    assert kernel32.allocated == {}


def test_windows_xgetbv_virtualprotect_failure_closes_cpuid_and_returns_none(monkeypatch):
    """If ``_Xgetbv`` fails during Windows ``VirtualProtect`` after ``_Cpuid``
    succeeded, both ``_Xgetbv``'s buffer and ``_Cpuid`` are cleaned up cleanly."""
    import ctypes

    regs = _all_cpuid_registers(osxsave=True)
    cpuid_closed = False

    class FakeCpuid:
        def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
            return regs[leaf]

        def close(self) -> None:
            nonlocal cpuid_closed
            cpuid_closed = True

    kernel32 = _FakeKernel32(protect_succeeds=False)

    def fake_cdll(name: str, *args, **kwargs):
        assert name == "kernel32.dll"
        return kernel32

    monkeypatch.setattr(os, "name", "nt")
    monkeypatch.setattr(ctypes, "CDLL", fake_cdll)
    monkeypatch.setattr(_cpu_check, "_is_x86_64", lambda: True)
    monkeypatch.setattr(_cpu_check, "_Cpuid", FakeCpuid)

    assert _cpu_check.read_cpu_features() is None
    assert cpuid_closed is True
    assert len(kernel32.free_calls) == 1
    assert kernel32.allocated == {}


# --- the real reader, where there is an x86-64 CPU ------------------------------------

#: /proc/cpuinfo's name for each feature the check reads.
CPUINFO = {
    "sse3": "pni",
    "pclmulqdq": "pclmulqdq",
    "ssse3": "ssse3",
    "fma": "fma",
    "cmpxchg16b": "cx16",
    "sse4.1": "sse4_1",
    "sse4.2": "sse4_2",
    "movbe": "movbe",
    "popcnt": "popcnt",
    "avx": "avx",
    "bmi1": "bmi1",
    "avx2": "avx2",
    "bmi2": "bmi2",
    "lzcnt": "abm",
}

x86_only = pytest.mark.skipif(not ON_X86, reason="cpuid exists on x86-64 CPUs only")


@x86_only
def test_the_real_reader_answers_for_every_feature_and_the_baseline_is_there():
    features = _cpu_check.read_cpu_features()
    assert features is not None, "cpuid could not be run on an x86-64 host"
    assert set(features) == set(_cpu_check.FEATURE_BITS)
    # Every x86-64 CPU the wheels run on, v1 or v3, has these.
    for name in ("sse3", "ssse3", "sse4.1", "sse4.2", "popcnt", "cmpxchg16b"):
        assert features[name], name


@x86_only
@pytest.mark.skipif(
    not Path("/proc/cpuinfo").exists(), reason="/proc/cpuinfo is the oracle, and Linux's"
)
def test_the_real_reader_agrees_with_the_kernels_flags():
    """The kernel's list is read by the kernel's own cpuid code, which shares
    nothing with these bytes."""
    text = Path("/proc/cpuinfo").read_text("utf-8")
    (line,) = [ln for ln in text.splitlines() if ln.startswith("flags")][:1]
    flags = set(line.split(":", 1)[1].split())
    features = _cpu_check.read_cpu_features()
    assert features is not None
    for name, flag in CPUINFO.items():
        assert features[name] == (flag in flags), (name, flag)


@x86_only
def test_reading_twice_gives_the_same_answer_and_frees_its_memory():
    first = _cpu_check.read_cpu_features()
    assert all(_cpu_check.read_cpu_features() == first for _ in range(50))


@x86_only
def test_the_real_xgetbv_reports_x87_and_xmm_when_osxsave_is_set():
    """On a real x86-64 host with OSXSAVE enabled, ``_Xgetbv()(0)`` reads XCR0
    cleanly and reports at least x87 (bit 0) and SSE/XMM (bit 1)."""
    cpuid = _cpu_check._Cpuid()
    try:
        osxsave = bool(cpuid(1)["ecx"] & (1 << _cpu_check._OSXSAVE_BIT))
    finally:
        cpuid.close()
    if not osxsave:
        pytest.skip("host CPU/OS does not report OSXSAVE")
    xgetbv = _cpu_check._Xgetbv()
    try:
        xcr0 = xgetbv(0)
    finally:
        xgetbv.close()
    assert (xcr0 & 0x3) == 0x3, hex(xcr0)


# --- the order of the imports -----------------------------------------------------------


def test_the_check_runs_before_the_extension_module_is_asked_for():
    """A fresh interpreter imports the package with a finder that logs which
    of its modules the import machinery is asked for, in order: the build's
    record (read by the check) comes before the compiled module."""
    code = textwrap.dedent(
        """
        import sys

        asked = []

        class Log:
            def find_spec(self, name, path=None, target=None):
                if name.startswith("rheon"):
                    asked.append(name)
                return None

        sys.meta_path.insert(0, Log())
        import rheon

        print("\\n".join(asked))
        """
    )
    env = {k: v for k, v in os.environ.items() if k != _cpu_check.SKIP_ENV}
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    asked = res.stdout.split()
    # The package, then the check, its reader of the environment, and what the
    # check reads, and only then anything else of the package: the compiled
    # module among it.
    assert asked[:4] == ["rheon", "rheon._cpu_check", "rheon._env", "rheon._build_flags"]
    assert "rheon._rheon" in asked, asked


def test_the_skip_variable_skips_the_whole_check_in_a_fresh_interpreter():
    """With the variable set, the build's record is never asked for."""
    code = textwrap.dedent(
        """
        import sys

        asked = []

        class Log:
            def find_spec(self, name, path=None, target=None):
                asked.append(name)
                return None

        sys.meta_path.insert(0, Log())
        import rheon

        print("rheon._build_flags" in asked)
        """
    )
    env = {**os.environ, _cpu_check.SKIP_ENV: "1"}
    res = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=env,
        check=False,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "False"
