"""Local hardware / performance profile for Reality Painter's Asset Optimizer.

Block 6 describes the machine Reality Engine is currently running on -
CPU, RAM, platform, and (best-effort) GPU - and derives a conservative,
explicitly-thresholded performance tier plus a starting FPS policy
recommendation from it. It performs no benchmarking, no rendering, no
GLB loading, and no network access, and it never calls into Blocks
1-5 (`analyzer.py`, `candidate_generator.py`, `benchmark.py`,
`selector.py`, `cache.py`) - this module only produces data; a later
integration layer is responsible for wiring a `HardwareProfile`'s
recommendation into `apps.reality_painter.optimization.selector`.

    local machine -> HardwareProfile -> RecommendedPerformancePolicy
                                              (starting point only -
                                               Block 3's real, measured
                                               benchmark FPS remains the
                                               performance source of
                                               truth once available)

DETECTION PHILOSOPHY
----------------------
Every detection step is best-effort and fails safe: an unavailable
library, an unrecognized platform, a failed subprocess call, or a
malformed value never raises out of `build_hardware_profile()` - the
corresponding field is simply `None` (or `PerformanceTier.UNKNOWN`
once too many fields are unknown to classify at all). No value is
ever fabricated or guessed to fill a gap.

SHARED-MEMORY HARDWARE
-------------------------
GPU memory is deliberately never reported as a fabricated size. Many
laptops use integrated graphics sharing system RAM rather than a
dedicated VRAM pool, so `GPUInfo.memory_kind` only ever takes one of
three honest values - `DEDICATED`, `SHARED`, or `UNKNOWN` - and
`GPUInfo.memory_mb` is populated only when a dedicated size is
actually known (which, with only OS-provided command-line utilities
and no vendor SDK, is effectively never in this implementation - see
"Genuine limitations" in the accompanying report). Defaulting to
`UNKNOWN`/`None` is the honest behavior, not a bug.

NO VENDOR SDKS, NO NETWORK
-----------------------------
GPU name/vendor detection (when it succeeds at all) uses only
general-purpose OS command-line utilities that are commonly present
(`wmic` on Windows, `lspci` on Linux, `system_profiler` on macOS) via
the standard library's `subprocess` - never CUDA, PyTorch, GPUtil,
pynvml, or any other vendor SDK, and never a network call of any kind.
Every such call is wrapped so its absence or failure degrades to
`GPUInfo(name=None, ...)` rather than raising.

DEPENDENCY INJECTION FOR TESTABILITY
----------------------------------------
Every detection entry point accepts an optional override (a fake CPU
count function, a fake memory detector, a fake `platform`-shaped
module, a fake GPU command runner) - the same dependency-injection
convention already used elsewhere in this repository (e.g. Block 2's
`CommandRunner`, Block 3's `Loader`/`RendererFactory`), so tests never
need to depend on the real, current machine's actual hardware.
"""

from __future__ import annotations

import ctypes
import os
import platform as _platform_module
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Sequence

# --- GPU ------------------------------------------------------------------


class GPUMemoryKind(str, Enum):
    """How (if at all) a GPU's memory capacity is known.

    Never inferred beyond what can be determined honestly - see the
    module docstring's "SHARED-MEMORY HARDWARE" section.
    """

    DEDICATED = "dedicated"
    SHARED = "shared"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GPUInfo:
    """Best-effort, gracefully-degrading GPU information.

    Attributes:
        name: The GPU's reported device name, or `None` if detection
            was unavailable/unsupported/failed.
        vendor: A coarse vendor label ("NVIDIA", "AMD", "Intel",
            "Apple"), inferred only from `name`'s own text - or `None`
            if `name` is `None` or unrecognized. Never guessed beyond
            that.
        memory_kind: See `GPUMemoryKind`. Defaults to `UNKNOWN`.
        memory_mb: Known dedicated VRAM in megabytes, only ever set
            when `memory_kind` is `DEDICATED` and a real value was
            obtained. `None` otherwise - never fabricated.
    """

    name: Optional[str] = None
    vendor: Optional[str] = None
    memory_kind: GPUMemoryKind = GPUMemoryKind.UNKNOWN
    memory_mb: Optional[int] = None


#: Runs a GPU-detection command and returns its captured stdout text.
#: Mirrors Block 2's `CommandRunner` convention. Injectable so tests
#: never invoke a real subprocess.
GPUCommandRunner = Callable[[Sequence[str]], str]

_WINDOWS_GPU_COMMAND: Sequence[str] = ("wmic", "path", "win32_VideoController", "get", "name")
_LINUX_GPU_COMMAND: Sequence[str] = ("lspci",)
_MAC_GPU_COMMAND: Sequence[str] = ("system_profiler", "SPDisplaysDataType")
_GPU_COMMAND_TIMEOUT_SECONDS = 5.0


def _default_gpu_command_runner(args: Sequence[str]) -> str:
    """Runs `args` as a real, local subprocess. The default `GPUCommandRunner`.

    Never touches the network - every command here is a local OS
    utility. A short timeout keeps a hung/missing utility from ever
    blocking profile construction.
    """
    completed = subprocess.run(list(args), capture_output=True, timeout=_GPU_COMMAND_TIMEOUT_SECONDS, check=False)
    stdout = completed.stdout
    return stdout.decode("utf-8", "replace") if isinstance(stdout, bytes) else str(stdout or "")


def _parse_windows_gpu_output(output: str) -> Optional[str]:
    """Extracts a device name from `wmic ... get name` output."""
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    lines = [line for line in lines if line.lower() != "name"]
    return lines[0] if lines else None


def _parse_linux_gpu_output(output: str) -> Optional[str]:
    """Extracts a device name from `lspci` output."""
    for line in output.splitlines():
        if "VGA compatible controller" in line or "3D controller" in line:
            parts = line.split(":", 2)
            if len(parts) == 3:
                return parts[2].strip()
            return line.strip() or None
    return None


def _parse_mac_gpu_output(output: str) -> Optional[str]:
    """Extracts a device name from `system_profiler SPDisplaysDataType` output."""
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Chipset Model:"):
            value = stripped.split(":", 1)[1].strip()
            return value or None
    return None


def _infer_vendor(name: str) -> Optional[str]:
    """Infers a coarse vendor label from a GPU device name's own text."""
    lowered = name.lower()
    if "nvidia" in lowered:
        return "NVIDIA"
    if "amd" in lowered or "radeon" in lowered:
        return "AMD"
    if "intel" in lowered:
        return "Intel"
    if "apple" in lowered:
        return "Apple"
    return None


def detect_gpu(system: Optional[str] = None, runner: Optional[GPUCommandRunner] = None) -> GPUInfo:
    """Best-effort GPU detection via a general-purpose OS command, if any is known.

    Never raises: an unrecognized `system`, a missing/failing command,
    or a timeout all degrade to `GPUInfo(name=None, ...)`. Memory is
    always reported as `GPUMemoryKind.UNKNOWN`/`memory_mb=None` - see
    module docstring; no dedicated/shared distinction is ever inferred
    from these commands' output.

    Args:
        system: The platform name (`"Windows"`/`"Linux"`/`"Darwin"`),
            as `platform.system()` would return. Defaults to the real
            current platform.
        runner: Optional `GPUCommandRunner` override, injected in
            tests instead of a real subprocess call.

    Returns:
        A `GPUInfo`, possibly with every field `None`/`UNKNOWN`.
    """
    active_system = system if system is not None else _platform_module.system()
    active_runner = runner if runner is not None else _default_gpu_command_runner

    name: Optional[str] = None
    try:
        if active_system == "Windows":
            name = _parse_windows_gpu_output(active_runner(_WINDOWS_GPU_COMMAND))
        elif active_system == "Linux":
            name = _parse_linux_gpu_output(active_runner(_LINUX_GPU_COMMAND))
        elif active_system == "Darwin":
            name = _parse_mac_gpu_output(active_runner(_MAC_GPU_COMMAND))
    except Exception:
        name = None

    if not name:
        return GPUInfo(name=None, vendor=None, memory_kind=GPUMemoryKind.UNKNOWN, memory_mb=None)

    return GPUInfo(name=name, vendor=_infer_vendor(name), memory_kind=GPUMemoryKind.UNKNOWN, memory_mb=None)


# --- CPU --------------------------------------------------------------


@dataclass(frozen=True)
class CPUInfo:
    """CPU information available without any external dependency.

    Attributes:
        logical_cores: `os.cpu_count()`'s result, or `None` if it
            could not be determined (the function itself can return
            `None` on some platforms) or was non-positive/invalid.
    """

    logical_cores: Optional[int] = None


def detect_cpu(cpu_count_fn: Callable[[], Optional[int]] = os.cpu_count) -> CPUInfo:
    """Detects logical CPU core count. Never raises.

    Args:
        cpu_count_fn: Override for `os.cpu_count`, injected in tests.

    Returns:
        A `CPUInfo`. `logical_cores` is `None` if `cpu_count_fn` raised,
        returned `None`, or returned a non-positive value.
    """
    try:
        logical_cores = cpu_count_fn()
    except Exception:
        logical_cores = None

    if not isinstance(logical_cores, int) or logical_cores <= 0:
        logical_cores = None

    return CPUInfo(logical_cores=logical_cores)


# --- Memory -----------------------------------------------------------


@dataclass(frozen=True)
class MemoryInfo:
    """System RAM information.

    Attributes:
        total_ram_mb: Total physical RAM in megabytes, or `None` if it
            could not be determined.
    """

    total_ram_mb: Optional[int] = None


class _WindowsMemoryStatusEx(ctypes.Structure):
    """Mirrors Win32's `MEMORYSTATUSEX` structure for `GlobalMemoryStatusEx`."""

    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _detect_ram_mb_windows() -> Optional[int]:
    """Reads total physical RAM via the Win32 User32/Kernel32 API. Never raises."""
    try:
        status = _WindowsMemoryStatusEx()
        status.dwLength = ctypes.sizeof(_WindowsMemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):  # type: ignore[attr-defined]
            return None
        return int(status.ullTotalPhys // (1024 * 1024))
    except Exception:
        return None


def _detect_ram_mb_posix() -> Optional[int]:
    """Reads total physical RAM via `os.sysconf` on Linux/macOS. Never raises."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
        return int((page_size * page_count) // (1024 * 1024))
    except (ValueError, OSError, AttributeError):
        return None


def detect_memory(
    system: Optional[str] = None, detector_override: Optional[Callable[[], Optional[int]]] = None
) -> MemoryInfo:
    """Detects total system RAM. Never raises.

    Args:
        system: The platform name, as `platform.system()` would
            return. Defaults to the real current platform. Ignored if
            `detector_override` is given.
        detector_override: A zero-argument callable returning total
            RAM in megabytes (or `None`), injected in tests instead of
            a real platform-specific detection call.

    Returns:
        A `MemoryInfo`. `total_ram_mb` is `None` if detection failed,
        was unsupported for the platform, or returned a non-positive
        value.
    """
    if detector_override is not None:
        try:
            total = detector_override()
        except Exception:
            total = None
    else:
        active_system = system if system is not None else _platform_module.system()
        if active_system == "Windows":
            total = _detect_ram_mb_windows()
        elif active_system in ("Linux", "Darwin"):
            total = _detect_ram_mb_posix()
        else:
            total = None

    if not isinstance(total, int) or total <= 0:
        total = None

    return MemoryInfo(total_ram_mb=total)


# --- Platform -----------------------------------------------------------


@dataclass(frozen=True)
class PlatformInfo:
    """Operating system / architecture information.

    Attributes:
        system: `platform.system()`'s result (e.g. `"Windows"`,
            `"Linux"`, `"Darwin"`), or `"Unknown"` if it could not be
            determined.
        release: `platform.release()`'s result, or `None` if
            unavailable.
        architecture: `platform.machine()`'s result (e.g. `"AMD64"`,
            `"x86_64"`, `"arm64"`), or `None` if unavailable.
    """

    system: str = "Unknown"
    release: Optional[str] = None
    architecture: Optional[str] = None


def detect_platform(platform_module: object = _platform_module) -> PlatformInfo:
    """Detects OS/architecture information. Never raises.

    Args:
        platform_module: An object exposing `system()`, `release()`,
            and `machine()` methods, matching the standard library's
            `platform` module. Defaults to the real `platform` module;
            tests inject a fake object instead.

    Returns:
        A `PlatformInfo`. Any field whose lookup raises falls back to
        `"Unknown"` (`system`) or `None` (`release`/`architecture`).
    """
    try:
        system = platform_module.system() or None  # type: ignore[attr-defined]
    except Exception:
        system = None

    try:
        release = platform_module.release() or None  # type: ignore[attr-defined]
    except Exception:
        release = None

    try:
        architecture = platform_module.machine() or None  # type: ignore[attr-defined]
    except Exception:
        architecture = None

    return PlatformInfo(system=system or "Unknown", release=release, architecture=architecture)


# --- Performance tier -------------------------------------------------


class PerformanceTier(str, Enum):
    """A conservative, application-level machine-capability classification.

    This is an explicit heuristic for Reality Painter's own asset
    selection, not a scientifically universal hardware rating - see
    module/`TierThresholds` docstrings. `UNKNOWN` is returned whenever
    there isn't enough detected data to classify at all, rather than
    guessing.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TierThresholds:
    """Explicit, configurable thresholds driving `classify_tier`.

    Deliberately plain, named fields rather than a hardcoded claim
    buried in logic (e.g. "8 GB RAM = high-end") - a caller can
    construct a different `TierThresholds` to change the policy
    without editing this module.

    Attributes:
        min_cores_for_medium: Minimum logical core count to qualify
            for at least `MEDIUM`.
        min_cores_for_high: Minimum logical core count to qualify for
            `HIGH`.
        min_ram_mb_for_medium: Minimum total RAM (MB) to qualify for
            at least `MEDIUM`.
        min_ram_mb_for_high: Minimum total RAM (MB) to qualify for
            `HIGH`.
    """

    min_cores_for_medium: int = 4
    min_cores_for_high: int = 8
    min_ram_mb_for_medium: int = 8 * 1024
    min_ram_mb_for_high: int = 16 * 1024


DEFAULT_TIER_THRESHOLDS = TierThresholds()


def classify_tier(cpu: CPUInfo, memory: MemoryInfo, thresholds: TierThresholds = DEFAULT_TIER_THRESHOLDS) -> PerformanceTier:
    """Classifies a conservative performance tier from CPU/RAM data alone.

    Deterministic: the same `(cpu, memory, thresholds)` always
    produces the same tier. Both CPU core count and RAM must clear a
    tier's thresholds for that tier to apply - a machine strong in one
    dimension but unknown/weak in the other is never over-classified.
    GPU information is intentionally not a tier input in this block:
    reliable GPU capability (as opposed to just a device name) is not
    safely available without a vendor SDK - see module docstring.

    Args:
        cpu: Detected CPU information.
        memory: Detected memory information.
        thresholds: The thresholds to classify against. Defaults to
            `DEFAULT_TIER_THRESHOLDS`.

    Returns:
        `PerformanceTier.UNKNOWN` if either `cpu.logical_cores` or
        `memory.total_ram_mb` is `None` (insufficient data to
        classify); otherwise the highest tier both dimensions clear,
        defaulting to `LOW`.
    """
    if cpu.logical_cores is None or memory.total_ram_mb is None:
        return PerformanceTier.UNKNOWN

    if cpu.logical_cores >= thresholds.min_cores_for_high and memory.total_ram_mb >= thresholds.min_ram_mb_for_high:
        return PerformanceTier.HIGH
    if (
        cpu.logical_cores >= thresholds.min_cores_for_medium
        and memory.total_ram_mb >= thresholds.min_ram_mb_for_medium
    ):
        return PerformanceTier.MEDIUM
    return PerformanceTier.LOW


# --- Recommended performance policy -----------------------------------


@dataclass(frozen=True)
class RecommendedPerformancePolicy:
    """A starting-point FPS policy recommendation for a given tier.

    Explicitly NOT a claim that hardware specs predict actual renderer
    FPS - this is only a conservative starting point. Once Block 3's
    real, measured benchmark results are available for a specific
    asset/candidate, those remain the performance source of truth (see
    module docstring). This is a plain data contract, independent of
    (and never constructed from) `apps.reality_painter.optimization
    .selector.PerformancePolicy` - Block 6 never imports Block 4.

    Attributes:
        target_fps: Recommended target FPS to aim for.
        minimum_fps: Recommended minimum acceptable FPS.
    """

    target_fps: float
    minimum_fps: float


#: Deterministic, explicit per-tier recommendations. A machine
#: classified `UNKNOWN` gets the same conservative policy as `LOW` -
#: the safest assumption when there isn't enough data to know better.
_TIER_POLICY_TABLE = {
    PerformanceTier.HIGH: RecommendedPerformancePolicy(target_fps=90.0, minimum_fps=45.0),
    PerformanceTier.MEDIUM: RecommendedPerformancePolicy(target_fps=60.0, minimum_fps=30.0),
    PerformanceTier.LOW: RecommendedPerformancePolicy(target_fps=30.0, minimum_fps=15.0),
    PerformanceTier.UNKNOWN: RecommendedPerformancePolicy(target_fps=30.0, minimum_fps=15.0),
}


def recommended_policy_for_tier(tier: PerformanceTier) -> RecommendedPerformancePolicy:
    """Returns the deterministic, explicit FPS recommendation for `tier`.

    Args:
        tier: The tier to look up.

    Returns:
        The matching `RecommendedPerformancePolicy`, or the same
        conservative policy as `PerformanceTier.LOW` for any tier not
        explicitly in the table (defensive default; every current
        `PerformanceTier` member is explicitly listed).
    """
    return _TIER_POLICY_TABLE.get(tier, _TIER_POLICY_TABLE[PerformanceTier.LOW])


# --- Aggregate profile --------------------------------------------------


@dataclass(frozen=True)
class HardwareProfile:
    """A complete, best-effort snapshot of the local machine.

    Attributes:
        cpu: Detected CPU information.
        memory: Detected memory information.
        platform: Detected OS/architecture information.
        gpu: Best-effort GPU information.
        tier: The conservative `PerformanceTier` classification
            derived from `cpu`/`memory` (see `classify_tier`).
    """

    cpu: CPUInfo
    memory: MemoryInfo
    platform: PlatformInfo
    gpu: GPUInfo
    tier: PerformanceTier


def build_hardware_profile(
    cpu_count_fn: Callable[[], Optional[int]] = os.cpu_count,
    memory_detector: Optional[Callable[[], Optional[int]]] = None,
    platform_module: object = _platform_module,
    gpu_runner: Optional[GPUCommandRunner] = None,
    tier_thresholds: TierThresholds = DEFAULT_TIER_THRESHOLDS,
) -> HardwareProfile:
    """Builds a complete `HardwareProfile` for the local machine. Never raises.

    Every detection step below already fails safe on its own (see each
    function's docstring); this function performs no additional
    validation beyond composing their results and deriving `tier`.

    Args:
        cpu_count_fn: Override for CPU detection, injected in tests.
        memory_detector: Override for memory detection, injected in
            tests.
        platform_module: Override for platform detection (an object
            shaped like the standard library's `platform` module),
            injected in tests.
        gpu_runner: Override for the GPU-detection command runner,
            injected in tests.
        tier_thresholds: Thresholds to classify `tier` against.
            Defaults to `DEFAULT_TIER_THRESHOLDS`.

    Returns:
        A fully-populated `HardwareProfile`. Individual fields may
        still be `None`/`UNKNOWN` where detection was unavailable.
    """
    cpu = detect_cpu(cpu_count_fn)
    plat = detect_platform(platform_module)
    memory = detect_memory(system=plat.system, detector_override=memory_detector)
    gpu = detect_gpu(system=plat.system, runner=gpu_runner)
    tier = classify_tier(cpu, memory, tier_thresholds)

    return HardwareProfile(cpu=cpu, memory=memory, platform=plat, gpu=gpu, tier=tier)