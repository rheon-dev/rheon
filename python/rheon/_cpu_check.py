"""Warn, before the compiled module loads, that the CPU lacks a feature the wheel needs.

The x86-64 wheels of ``rheon`` are compiled for CPUs with AVX2, FMA
and the rest of the feature set in ``BUILD_FEATURE_FLAGS``; the compiler may
use any of them anywhere. On a CPU without one the first such instruction
kills the process with an illegal-instruction signal, and no Python frame
says why. This module reads the CPU's features with the ``cpuid``
instruction and, when one is missing, issues a ``RuntimeWarning`` that names
it and the package to install instead (``rheon-compat``, the same
code built for every x86-64 CPU). It runs in pure Python, first in the
package's ``__init__``: a check inside the extension would run after the
compiler had already emitted instructions it cannot execute.

The check does nothing, and loads nothing, where there is nothing to check:
on ARM, in ``rheon-compat``, in a build from source with no
``target-feature`` flag (all of which record an empty ``BUILD_FEATURE_FLAGS``),
and when ``RHEON_SKIP_CPU_CHECK`` is set. A CPU it cannot read
(an operating system that refuses executable memory, say) is not a reason to
fail the import: it reads nothing and says nothing.

The design, the 14 feature names and the machine code are Polars'
(``polars/_cpu_check.py``, MIT), which vendors Anders Høst's ``cpuid.py``
(MIT, Copyright (c) 2014 Anders Høst): a few bytes that run ``cpuid`` from
Python through ``ctypes``. The bytes here were assembled again from the
instructions listed beside them, and the code around them is written afresh,
with four additions: a leaf is read only when the CPU reports it, AVX/AVX2/FMA
also check ``OSXSAVE`` and ``XCR0`` YMM state through ``xgetbv``, executable
memory is allocated write-then-execute on both POSIX and Windows, and a failure
to read is silent.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import platform
import sys
import warnings
from collections.abc import Callable

from rheon import _env

#: Set it to anything non-empty to skip the check.
SKIP_ENV = "RHEON_SKIP_CPU_CHECK"

#: The other package, built for every x86-64 CPU.
COMPAT_PACKAGE = "rheon-compat"

#: Each feature the check can read: the ``cpuid`` leaf, the register and the
#: bit, from the Intel SDM (vol. 2A, ``CPUID``) and AMD's APM (vol. 3).
#: The names are the ones rustc's ``-C target-feature`` takes.
FEATURE_BITS: dict[str, tuple[int, str, int]] = {
    "sse3": (1, "ecx", 0),
    "pclmulqdq": (1, "ecx", 1),
    "ssse3": (1, "ecx", 9),
    "fma": (1, "ecx", 12),
    "cmpxchg16b": (1, "ecx", 13),
    "sse4.1": (1, "ecx", 19),
    "sse4.2": (1, "ecx", 20),
    "movbe": (1, "ecx", 22),
    "popcnt": (1, "ecx", 23),
    "avx": (1, "ecx", 28),
    "bmi1": (7, "ebx", 3),
    "avx2": (7, "ebx", 5),
    "bmi2": (7, "ebx", 8),
    "lzcnt": (0x80000001, "ecx", 5),
}

_X86_MACHINES = {"x86_64", "amd64", "x64"}

#: CPUID leaf 1 ECX bit 27: the OS has enabled XSAVE/XGETBV (``CR4.OSXSAVE``).
_OSXSAVE_BIT = 27

#: XCR0 bits 1 (XMM) and 2 (YMM): both must be set for AVX/AVX2/FMA
#: instructions to execute without trapping (``#UD`` / ``SIGILL``).
_XCR0_YMM_MASK = 0x6

#: Features in ``FEATURE_BITS`` that require OS-enabled YMM state.
#: General-purpose VEX-encoded extensions (``bmi1``, ``bmi2``) and SSE/XMM
#: extensions do not require YMM state in XCR0.
_YMM_FEATURES = frozenset({"avx", "avx2", "fma"})

# ``void cpuid(uint32_t out[4], uint32_t leaf, uint32_t subleaf)``: leaf in
# eax, subleaf in ecx, and eax, ebx, ecx, edx stored to ``out``. System V
# (Linux, macOS): out, leaf, subleaf arrive in rdi, esi, edx.
# fmt: off
_POSIX_CODE = bytes([
    0x53,                    # push   %rbx
    0x89, 0xf0,              # mov    %esi,%eax
    0x89, 0xd1,              # mov    %edx,%ecx
    0x0f, 0xa2,              # cpuid
    0x89, 0x07,              # mov    %eax,(%rdi)
    0x89, 0x5f, 0x04,        # mov    %ebx,0x4(%rdi)
    0x89, 0x4f, 0x08,        # mov    %ecx,0x8(%rdi)
    0x89, 0x57, 0x0c,        # mov    %edx,0xc(%rdi)
    0x5b,                    # pop    %rbx
    0xc3,                    # ret
])
# Windows x64: out, leaf, subleaf arrive in rcx, edx, r8d.
_WINDOWS_CODE = bytes([
    0x53,                    # push   %rbx
    0x89, 0xd0,              # mov    %edx,%eax
    0x49, 0x89, 0xc9,        # mov    %rcx,%r9
    0x44, 0x89, 0xc1,        # mov    %r8d,%ecx
    0x0f, 0xa2,              # cpuid
    0x41, 0x89, 0x01,        # mov    %eax,(%r9)
    0x41, 0x89, 0x59, 0x04,  # mov    %ebx,0x4(%r9)
    0x41, 0x89, 0x49, 0x08,  # mov    %ecx,0x8(%r9)
    0x41, 0x89, 0x51, 0x0c,  # mov    %edx,0xc(%r9)
    0x5b,                    # pop    %rbx
    0xc3,                    # ret
])
# ``uint32_t xgetbv(uint32_t xcr)``: XCR index in ecx, low 32 bits of XCR
# returned in eax (high 32 bits written to edx; eax, ecx and edx are volatile
# on both System V AMD64 and Windows x64, and no callee-saved register or stack
# slot is touched).
# System V (Linux, macOS): xcr arrives in edi.
_POSIX_XGETBV_CODE = bytes([
    0x89, 0xf9,              # mov    %edi,%ecx
    0x0f, 0x01, 0xd0,        # xgetbv
    0xc3,                    # ret
])
# Windows x64: xcr arrives in ecx.
_WINDOWS_XGETBV_CODE = bytes([
    0x0f, 0x01, 0xd0,        # xgetbv
    0xc3,                    # ret
])
# fmt: on


def baked_flags() -> str:
    """The features this build was compiled to need, as ``"+avx2,+fma,..."``.

    Empty when the build recorded none, or recorded nothing at all (no
    ``_build_flags`` module: a checkout that was never built through cargo).
    """
    try:
        module = importlib.import_module("rheon._build_flags")
    except ImportError:
        return ""
    return str(getattr(module, "BUILD_FEATURE_FLAGS", ""))


def required_features(flags: str) -> list[str]:
    """The feature names in a flag string, without their ``+``; a feature
    switched off (``-name``) and an empty item are not required."""
    return [f.lstrip("+") for f in flags.split(",") if f and not f.startswith("-")]


def _is_x86_64() -> bool:
    return platform.machine().lower() in _X86_MACHINES and sys.maxsize > 2**32


class _ExecCode:
    """Machine code allocated write-then-execute (W^X) and freed on ``close``."""

    def __init__(self, code: bytes) -> None:
        import ctypes

        self._ctypes = ctypes
        self._release: Callable[[], object] | None = None
        if os.name == "nt":
            kernel32 = ctypes.CDLL("kernel32.dll")
            kernel32.VirtualAlloc.restype = ctypes.c_void_p
            kernel32.VirtualAlloc.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_ulong,
                ctypes.c_ulong,
            ]
            kernel32.VirtualProtect.restype = ctypes.c_long
            kernel32.VirtualProtect.argtypes = [
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_ulong,
                ctypes.POINTER(ctypes.c_ulong),
            ]
            kernel32.VirtualFree.restype = ctypes.c_long
            kernel32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ulong]
            mem_commit_reserve = 0x3000
            page_readwrite, page_execute_read = 0x04, 0x20
            mem_release = 0x8000
            # Written first and made executable after (W^X): policies such as
            # ACG refuse memory that is writable and executable at once
            # (review 2026-11-23, PYO3-2).
            address = kernel32.VirtualAlloc(None, len(code), mem_commit_reserve, page_readwrite)
            if not address:
                raise OSError("no memory for the CPU check")
            try:
                ctypes.memmove(address, code, len(code))
                old_protect = ctypes.c_ulong(0)
                if not kernel32.VirtualProtect(
                    address, len(code), page_execute_read, ctypes.byref(old_protect)
                ):
                    raise OSError(
                        "rheon CPU check: could not make memory executable. "
                        "Set RHEON_SKIP_CPU_CHECK=1 to skip the check."
                    )
            except BaseException:
                kernel32.VirtualFree(address, 0, mem_release)
                raise
            self._release = lambda: kernel32.VirtualFree(address, 0, mem_release)
        else:
            import mmap

            # Written first and made executable after, since some systems
            # refuse memory that is writable and executable at once.
            self._map = mmap.mmap(
                -1,
                len(code),
                mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS,
                mmap.PROT_READ | mmap.PROT_WRITE,
            )
            try:
                self._map.write(code)
                address = ctypes.addressof(ctypes.c_char.from_buffer(self._map))
                mprotect = ctypes.CDLL(None, use_errno=True).mprotect
                mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
                mprotect.restype = ctypes.c_int
                if mprotect(address, len(code), mmap.PROT_READ | mmap.PROT_EXEC) != 0:
                    # The map is closed, not left to the collector (review
                    # 2026-11-23, PYO3-2).
                    raise OSError("could not make memory executable for the CPU check")
            except BaseException:
                self._close_map()
                raise
            self._release = self._close_map
        self.address: int = int(address)

    def _close_map(self) -> None:
        # A view still alive is left to the garbage collector.
        with contextlib.suppress(BufferError):
            self._map.close()

    def close(self) -> None:
        if self._release is not None:
            self._release()
            self._release = None


class _Cpuid(_ExecCode):
    """``cpuid`` as a Python callable: a few bytes in executable memory."""

    def __init__(self) -> None:
        super().__init__(_WINDOWS_CODE if os.name == "nt" else _POSIX_CODE)
        prototype = self._ctypes.CFUNCTYPE(
            None,
            self._ctypes.POINTER(self._ctypes.c_uint32),
            self._ctypes.c_uint32,
            self._ctypes.c_uint32,
        )
        self._function = prototype(self.address)

    def __call__(self, leaf: int, subleaf: int = 0) -> dict[str, int]:
        out = (self._ctypes.c_uint32 * 4)()
        self._function(out, leaf, subleaf)
        return dict(zip(("eax", "ebx", "ecx", "edx"), out, strict=True))


class _Xgetbv(_ExecCode):
    """``xgetbv`` as a Python callable: returns the low 32 bits of XCR ``xcr``.

    Only call when ``cpuid(1)["ecx"]`` reports ``OSXSAVE`` (bit 27):
    executing ``xgetbv`` without ``OSXSAVE`` traps with an illegal instruction.
    """

    def __init__(self) -> None:
        super().__init__(_WINDOWS_XGETBV_CODE if os.name == "nt" else _POSIX_XGETBV_CODE)
        prototype = self._ctypes.CFUNCTYPE(self._ctypes.c_uint32, self._ctypes.c_uint32)
        self._function = prototype(self.address)

    def __call__(self, xcr: int = 0) -> int:
        return int(self._function(xcr))


def read_cpu_features() -> dict[str, bool] | None:
    """Which of ``FEATURE_BITS`` this CPU has, or ``None`` when it cannot be
    read: not an x86-64 CPU, or memory it may not execute."""
    if not _is_x86_64():
        return None
    try:
        cpuid = _Cpuid()
    except Exception:
        return None
    try:
        registers: dict[int, dict[str, int]] = {}
        highest = {0: cpuid(0)["eax"], 0x80000000: cpuid(0x80000000)["eax"]}
        for leaf in sorted({leaf for leaf, _, _ in FEATURE_BITS.values()}):
            # A leaf past the highest the CPU reports returns the highest
            # one's data, which is no answer: its features are absent.
            top = highest[0x80000000 if leaf >= 0x80000000 else 0]
            registers[leaf] = cpuid(leaf) if leaf <= top else {}
        # A CPU may advertise AVX/AVX2/FMA in CPUID even when the OS or
        # hypervisor has not enabled XSAVE or YMM state in XCR0; executing
        # an AVX/FMA instruction then traps with #UD / SIGILL (review
        # 2026-11-23, PYO3-2). Never execute xgetbv unless OSXSAVE is set.
        osxsave = bool(registers.get(1, {}).get("ecx", 0) & (1 << _OSXSAVE_BIT))
        if osxsave:
            xgetbv = _Xgetbv()
            try:
                ymm_enabled = (xgetbv(0) & _XCR0_YMM_MASK) == _XCR0_YMM_MASK
            finally:
                xgetbv.close()
        else:
            ymm_enabled = False
    except Exception:
        return None
    finally:
        cpuid.close()
    return {
        name: bool(registers[leaf].get(register, 0) & (1 << bit))
        and (ymm_enabled or name not in _YMM_FEATURES)
        for name, (leaf, register, bit) in FEATURE_BITS.items()
    }


def missing_features(flags: str | None = None) -> list[str]:
    """The features of ``flags`` (this build's, by default) the CPU lacks, in
    the order the build names them. A feature the check cannot read is not
    reported, and neither is anything when the CPU cannot be read."""
    wanted = required_features(baked_flags() if flags is None else flags)
    if not wanted:
        return []
    have = read_cpu_features()
    if have is None:
        return []
    return [f for f in wanted if f in have and not have[f]]


def message(missing: list[str]) -> str:
    """The warning's text."""
    return (
        "Missing required CPU features.\n"
        "\n"
        "The following CPU features, which this build of rheon needs, were not "
        "detected:\n"
        f"    {', '.join(missing)}\n"
        "Continuing to use this build on this processor will likely crash it with an "
        "illegal instruction.\n"
        f"Install `{COMPAT_PACKAGE}` instead of `rheon`: it is the same code, built "
        "for every x86-64 CPU, and it is imported as `rheon` too. Install one or the "
        "other: run `pip uninstall rheon` first.\n"
        "\n"
        "Hint: If you are on an Apple silicon Mac (M1 or later), this is likely due to "
        "running Python under Rosetta. Install a native Python that does not run under "
        "Rosetta x86-64 emulation.\n"
        "\n"
        "If you believe this warning to be a false positive, set the "
        f"`{SKIP_ENV}` environment variable to bypass this check.\n"
    )


def run() -> None:
    """The check: warn if this CPU lacks a feature the build needs."""
    if _env.get(SKIP_ENV):
        return
    missing = missing_features()
    if missing:
        warnings.warn(message(missing), RuntimeWarning, stacklevel=3)
