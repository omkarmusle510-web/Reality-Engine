"""Real render-performance benchmarking for Reality Painter's Asset Optimizer.

Block 3 answers one question only: given several candidate GLBs (e.g.
the `ORIGINAL`/`LIGHT`/`HEAVY_OPTIMIZED`/`COMBINED` outputs of Block 2's
`apps.reality_painter.optimization.candidate_generator`), which ones
actually load and render successfully, and how fast?

This is measurement only - see the package docstring for what Block 3
deliberately does NOT do (quality selection, caching, runtime LOD
switching, hardware detection).

REAL RENDER PATH - NOT SIMULATED
---------------------------------
`benchmark_candidate()` drives the *actual*, unmodified engine stack:

    candidate path
        -> engine.scene.loader.load_glb()      (real GLB parsing)
        -> engine.scene.scene.Scene            (real scene container)
        -> engine.rendering.renderer.Renderer3D.render()   (real pyrender draw)
        -> engine.rendering.renderer.composite_rgba_onto() (real compositing)

repeated over N measured frames and timed with `time.perf_counter`.
Nothing here estimates FPS from triangle count, fabricates a GPU score,
or returns a hardcoded performance value - a candidate's reported FPS
is always derived from real, measured `Renderer3D.render()` wall-clock
time.

`composite_rgba_onto()` is exercised every measured frame against an
in-memory dummy BGR buffer (not a live camera frame), because that is
the exact per-frame workload `apps.reality_painter.asset_render`'s
pipeline stage already performs on every real pipeline cycle
(`renderer.render(scene)` then `composite_rgba_onto(frame.image,
rendered)`) - this is the deepest part of the real "GLB -> Renderer3D
-> composited frame" path that can be exercised without a live camera
device or `app.py`/pipeline wiring.

WHAT THIS BLOCK DOES NOT COVER
-------------------------------
The full "GLB -> Renderer3D -> webcam-composited Reality Painter frame"
path additionally involves `engine.vision.camera.Camera`, the running
`Engine`/`Pipeline`, hand tracking, and the rest of
`apps/reality_painter/app.py`'s registered stages. Driving that would
require either a live camera device or invasive changes to `app.py` to
extract a headless entry point - both are explicitly out of scope for
this block (see the accompanying task description). Compositing onto a
real camera frame, inside the running application pipeline, is left
for a later integration phase.

FAILURE HANDLING
-----------------
`benchmark_candidate()` never raises for an expected failure (missing
file, malformed/unsupported file, load failure, renderer
initialization failure, mid-render failure) - every case is reported
as a `BenchmarkResult(success=False, error=...)`, matching Block 1/2's
"typed result over uncaught exception" convention, so a caller
benchmarking several candidates never has one bad candidate abort the
batch.

INDEPENDENCE FROM RENDERER3D/LOAD_GLB (BY INJECTION, NOT DUPLICATION)
-----------------------------------------------------------------------
This module never reimplements loading or rendering. `load_glb` and a
`Renderer3D`-shaped factory are both injectable (`loader`/
`renderer_factory` parameters) purely so offline unit tests can supply
a fast, deterministic fake instead of requiring a GPU/EGL context or a
real GLB parse - the default values are the real
`engine.scene.loader.load_glb` and a real `Renderer3D`. Production
files (`engine/scene/loader.py`, `engine/rendering/renderer.py`, and
everything else under `engine/`) are never modified by this module.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np

# `Renderer3D` only imports/requires `pyrender` (and, transitively, an
# EGL-capable OpenGL context) inside its own `__init__` - not at module
# import time - so this default-platform hint is harmless to set here
# even when a fake renderer_factory is used and pyrender is never
# touched. Matches the documented reason this exists in
# `engine/rendering/renderer.py`: EGL must be selected before pyrender
# is first imported anywhere in the process. `setdefault` never
# overrides a platform the caller (or the real renderer module) has
# already chosen.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from engine.core.logger import get_logger
from engine.scene.loader import ModelLoadError, load_glb
from engine.scene.objects import SceneObject
from engine.scene.scene import Scene
from engine.rendering.renderer import RenderError, Renderer3D, composite_rgba_onto

logger = get_logger(__name__)

_SUPPORTED_EXTENSIONS = (".glb", ".gltf")
_DEFAULT_WARMUP_FRAMES = 5
_DEFAULT_MEASURED_FRAMES = 30
_DEFAULT_WIDTH = 320
_DEFAULT_HEIGHT = 240

#: Loads a model file into a `SceneObject`, matching
#: `engine.scene.loader.load_glb`'s signature. Injectable so offline
#: tests never need a real GLB parse.
Loader = Callable[[Path, str], SceneObject]

#: Constructs a renderer exposing `.render(scene) -> np.ndarray` (RGBA
#: uint8) and `.close()`, matching `Renderer3D`. Injectable so offline
#: tests never need a GPU/EGL context.
RendererFactory = Callable[[int, int], Any]

#: A monotonic time source, matching `time.perf_counter`. Injectable
#: for deterministic frame-timing tests.
Clock = Callable[[], float]


class BenchmarkError(Exception):
    """Base class for errors raised directly by the benchmark's public API."""


@dataclass(frozen=True)
class BenchmarkResult:
    """The outcome of benchmarking one candidate GLB against the real renderer.

    Attributes:
        candidate_path: The benchmarked file's path, as a string.
        candidate_name: A short identifier for the candidate, derived
            from the file's stem unless the caller supplied one
            explicitly (e.g. Block 2's `CandidateResult.candidate_name`).
        success: Whether the candidate loaded and rendered every
            measured frame without error.
        error: A human-readable failure reason, or `None` on success.
        load_time_seconds: Wall-clock time spent in `load_glb()` for
            this candidate. `None` if loading never completed.
        average_fps: `1 / average_frame_time_seconds`, or `0.0` if no
            frame completed. Always derived from real measured render
            calls - never estimated or fabricated.
        average_frame_time_seconds: Mean wall-clock time per measured
            `Renderer3D.render()` + `composite_rgba_onto()` call.
        measured_frames: Number of frames actually measured and
            included in the average (may be less than the requested
            count if a render call failed partway through - the
            candidate is still reported as a failure in that case, but
            the frames that did complete are preserved for inspection).
        warmup_frames: Number of warm-up frames run (and excluded from
            timing) before measurement began.
        render_width: Renderer output width used for this benchmark.
        render_height: Renderer output height used for this benchmark.
    """

    candidate_path: str
    candidate_name: str
    success: bool
    error: Optional[str]
    load_time_seconds: Optional[float]
    average_fps: float
    average_frame_time_seconds: float
    measured_frames: int
    warmup_frames: int
    render_width: int
    render_height: int

    def to_dict(self):
        """Returns a plain, JSON-serializable dict of this result."""
        return asdict(self)


def _default_renderer_factory(width: int, height: int) -> Renderer3D:
    """Constructs the real `Renderer3D`. The default `RendererFactory`."""
    return Renderer3D(width=width, height=height)


def _failure(
    candidate_path: Path,
    candidate_name: str,
    error: str,
    width: int,
    height: int,
    warmup_frames: int,
    load_time_seconds: Optional[float] = None,
    measured_frames: int = 0,
) -> BenchmarkResult:
    """Builds a failed `BenchmarkResult`. Never raises - see module docstring."""
    logger.warning("Benchmark failed for candidate '%s': %s", candidate_name, error)
    return BenchmarkResult(
        candidate_path=str(candidate_path),
        candidate_name=candidate_name,
        success=False,
        error=error,
        load_time_seconds=load_time_seconds,
        average_fps=0.0,
        average_frame_time_seconds=0.0,
        measured_frames=measured_frames,
        warmup_frames=warmup_frames,
        render_width=width,
        render_height=height,
    )


def benchmark_candidate(
    candidate_path: Union[str, Path],
    candidate_name: Optional[str] = None,
    warmup_frames: int = _DEFAULT_WARMUP_FRAMES,
    measured_frames: int = _DEFAULT_MEASURED_FRAMES,
    width: int = _DEFAULT_WIDTH,
    height: int = _DEFAULT_HEIGHT,
    loader: Optional[Loader] = None,
    renderer_factory: Optional[RendererFactory] = None,
    clock: Optional[Clock] = None,
) -> BenchmarkResult:
    """Benchmarks a single candidate GLB through the real render path.

    Never raises: every expected failure mode (missing file,
    unsupported extension, load failure, renderer initialization
    failure, mid-render failure) is reported as
    `BenchmarkResult(success=False, error=...)`. Never modifies
    `candidate_path` - it is only ever read (via `load_glb`).

    Sequence:
        1. Validate the candidate file exists and has a supported
           extension.
        2. Load it via `loader` (default: the real `load_glb`), timing
           the call as `load_time_seconds`.
        3. Construct a renderer via `renderer_factory` (default: a
           real `Renderer3D`) and build a single-object `Scene`.
        4. Run `warmup_frames` unmeasured render+composite cycles, so
           first-frame initialization (shader/context setup) never
           pollutes the measured average.
        5. Run `measured_frames` timed render+composite cycles.
        6. Compute `average_frame_time_seconds` and `average_fps` from
           the measured cycles, and always close the renderer.

    Args:
        candidate_path: Path to the candidate `.glb`/`.gltf` file.
        candidate_name: Short identifier for this candidate. Defaults
            to `candidate_path`'s file stem.
        warmup_frames: Unmeasured render cycles run before timing
            begins.
        measured_frames: Timed render cycles to average over.
        width: Renderer output width, in pixels.
        height: Renderer output height, in pixels.
        loader: Optional `Loader` override. Defaults to the real
            `engine.scene.loader.load_glb`. Tests inject a fake here so
            no real GLB parse is required.
        renderer_factory: Optional `RendererFactory` override. Defaults
            to constructing a real `Renderer3D`. Tests inject a fake
            here so no GPU/EGL context is required.
        clock: Optional `Clock` override. Defaults to
            `time.perf_counter`. Tests inject a fake for deterministic
            frame-time assertions.

    Returns:
        A `BenchmarkResult` describing the outcome.
    """
    path = Path(candidate_path)
    name = candidate_name or path.stem
    active_clock = clock if clock is not None else time.perf_counter
    active_loader = loader if loader is not None else load_glb
    active_renderer_factory = renderer_factory if renderer_factory is not None else _default_renderer_factory

    if path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        return _failure(
            path, name, f"Unsupported asset format {path.suffix!r}: {path}.", width, height, warmup_frames
        )
    if not path.is_file():
        return _failure(path, name, f"Candidate file not found: {path}.", width, height, warmup_frames)

    load_start = active_clock()
    try:
        scene_object = active_loader(path, name)
    except ModelLoadError as exc:
        return _failure(path, name, f"Failed to load candidate: {exc}", width, height, warmup_frames)
    except Exception as exc:
        return _failure(path, name, f"Unexpected error loading candidate: {exc}", width, height, warmup_frames)
    load_time_seconds = active_clock() - load_start

    try:
        renderer = active_renderer_factory(width, height)
    except RenderError as exc:
        return _failure(
            path, name, f"Renderer initialization failed: {exc}", width, height, warmup_frames, load_time_seconds
        )
    except Exception as exc:
        return _failure(
            path,
            name,
            f"Unexpected error initializing renderer: {exc}",
            width,
            height,
            warmup_frames,
            load_time_seconds,
        )

    scene = Scene()
    scene.add(scene_object)
    dummy_frame = np.full((height, width, 3), 127, dtype=np.uint8)
    frame_times: List[float] = []

    try:
        for _ in range(max(0, warmup_frames)):
            rendered = renderer.render(scene)
            composite_rgba_onto(dummy_frame.copy(), rendered)

        for _ in range(max(0, measured_frames)):
            frame_start = active_clock()
            rendered = renderer.render(scene)
            composite_rgba_onto(dummy_frame.copy(), rendered)
            frame_times.append(active_clock() - frame_start)
    except RenderError as exc:
        return _failure(
            path,
            name,
            f"Render failed: {exc}",
            width,
            height,
            warmup_frames,
            load_time_seconds,
            measured_frames=len(frame_times),
        )
    except Exception as exc:
        return _failure(
            path,
            name,
            f"Unexpected error during render: {exc}",
            width,
            height,
            warmup_frames,
            load_time_seconds,
            measured_frames=len(frame_times),
        )
    finally:
        try:
            renderer.close()
        except Exception:
            logger.exception("Renderer.close() raised while benchmarking candidate '%s'.", name)

    if not frame_times:
        return _failure(
            path, name, "No frames were measured (measured_frames <= 0).", width, height, warmup_frames, load_time_seconds
        )

    average_frame_time_seconds = sum(frame_times) / len(frame_times)
    average_fps = (1.0 / average_frame_time_seconds) if average_frame_time_seconds > 0 else 0.0

    return BenchmarkResult(
        candidate_path=str(path),
        candidate_name=name,
        success=True,
        error=None,
        load_time_seconds=load_time_seconds,
        average_fps=average_fps,
        average_frame_time_seconds=average_frame_time_seconds,
        measured_frames=len(frame_times),
        warmup_frames=warmup_frames,
        render_width=width,
        render_height=height,
    )


def benchmark_candidates(
    candidate_paths: Sequence[Union[str, Path]],
    **kwargs: Any,
) -> List[BenchmarkResult]:
    """Benchmarks several candidates independently.

    One candidate failing (for any reason `benchmark_candidate` already
    handles) never prevents the remaining candidates from being
    attempted - this function only ever returns, never raises, for
    per-candidate failures.

    Args:
        candidate_paths: Paths to the candidate `.glb`/`.gltf` files.
        **kwargs: Forwarded to `benchmark_candidate` for every
            candidate (e.g. `warmup_frames`, `measured_frames`,
            `renderer_factory`).

    Returns:
        One `BenchmarkResult` per entry in `candidate_paths`, in order.
    """
    results: List[BenchmarkResult] = []
    for candidate_path in candidate_paths:
        results.append(benchmark_candidate(candidate_path, **kwargs))
    return results


# --- Deterministic comparison -----------------------------------------


def rank_candidates(results: Sequence[BenchmarkResult]) -> List[BenchmarkResult]:
    """Ranks candidates by measured performance, without inventing data.

    Only `success=True` results are ever ranked - a failed candidate
    can never win, regardless of what (if anything) its partial timing
    data looks like. Successful results are sorted by `average_fps`
    descending; ties are broken by `candidate_path` for a fully
    deterministic order given the same input.

    Args:
        results: Benchmark results to rank, e.g. from
            `benchmark_candidates`.

    Returns:
        The successful subset of `results`, sorted best-first. Empty if
        no candidate succeeded.
    """
    successful = [result for result in results if result.success]
    return sorted(successful, key=lambda result: (-result.average_fps, result.candidate_path))


def best_candidate(results: Sequence[BenchmarkResult]) -> Optional[BenchmarkResult]:
    """Returns the best-performing successful candidate, or `None`.

    A thin convenience wrapper over `rank_candidates` - see its
    docstring for the ranking rule. Never returns a failed candidate.

    Args:
        results: Benchmark results to select from.

    Returns:
        The top-ranked `BenchmarkResult`, or `None` if every candidate
        failed.
    """
    ranked = rank_candidates(results)
    return ranked[0] if ranked else None


# --- Manual/real-hardware entry point -------------------------------------
#
# Deliberately separate from the offline unit-test-facing API above:
# this is the "run it for real, on my machine" path, using the real
# `load_glb`/`Renderer3D` defaults end to end (no injected fakes). Not
# imported or exercised by `tests/test_asset_benchmark.py` - that file
# only ever calls `benchmark_candidate`/`benchmark_candidates` with
# injected fakes, so it never requires a GPU/EGL context.
#
# What this does NOT cover: a live camera device, the running
# `Engine`/`Pipeline`, or `app.py`'s registered stages - see the module
# docstring's "WHAT THIS BLOCK DOES NOT COVER" section. This exercises
# the deepest real `GLB -> Renderer3D -> composited frame` path
# reachable without modifying or invoking `app.py`.


def _print_manual_report(results: List[BenchmarkResult]) -> None:
    """Prints a human-readable report for the manual benchmark entry point."""
    print("REALITY ENGINE ASSET BENCHMARK")
    print()
    for result in results:
        print(f"Candidate: {result.candidate_name}")
        if result.load_time_seconds is not None:
            print(f"Load time: {result.load_time_seconds:.4f}s")
        else:
            print("Load time: N/A")
        if result.success:
            print(f"Average FPS: {result.average_fps:.2f}")
            print(f"Average frame time: {result.average_frame_time_seconds * 1000:.2f}ms")
            print("Result: PASS")
        else:
            print("Average FPS: N/A")
            print("Average frame time: N/A")
            print(f"Result: FAIL ({result.error})")
        print()

    winner = best_candidate(results)
    if winner is not None:
        print(f"BEST PERFORMING VALID CANDIDATE: {winner.candidate_name} ({winner.average_fps:.2f} FPS)")
    else:
        print("BEST PERFORMING VALID CANDIDATE: none (every candidate failed)")


def _run_manual_benchmark(paths: List[str]) -> int:
    """Runs the real benchmark against real candidate files and prints a report.

    Uses the real `load_glb`/`Renderer3D` defaults - no fakes. Intended
    to be run manually (`python -m
    apps.reality_painter.optimization.benchmark <glb> [<glb> ...]`)
    against real candidate files, e.g. Block 2's generated candidates.

    Args:
        paths: Candidate GLB/GLTF file paths to benchmark, in order.

    Returns:
        Process exit code: 0 if at least one candidate succeeded, 1
        otherwise (including "no paths given").
    """
    if not paths:
        print("Usage: python -m apps.reality_painter.optimization.benchmark <candidate.glb> [more.glb ...]")
        return 1

    results = benchmark_candidates(paths)
    _print_manual_report(results)
    return 0 if any(result.success for result in results) else 1


if __name__ == "__main__":
    sys.exit(_run_manual_benchmark(sys.argv[1:]))
