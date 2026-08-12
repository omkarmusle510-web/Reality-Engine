"""Offline, deterministic tests for the Asset Optimizer's Block 3 benchmarker.

No GPU, no EGL/pyrender context, no camera, no network. Every loader
and renderer call is injected via fakes (the same dependency-injection
pattern already used by `tests/test_asset_retriever.py` and
`tests/test_candidate_generator.py`), so `Renderer3D` and `load_glb`
are never actually invoked here - only
`apps.reality_painter.optimization.benchmark`'s own contract and logic
are exercised.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

from apps.reality_painter.optimization.benchmark import (
    BenchmarkResult,
    benchmark_candidate,
    benchmark_candidates,
    best_candidate,
    rank_candidates,
)
from engine.rendering.renderer import RenderError
from engine.scene.loader import ModelLoadError
from engine.scene.objects import SceneObject

passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"PASS: {name}")
    else:
        failed += 1
        print(f"FAIL: {name}")


# --- Fakes -----------------------------------------------------------


def _fake_loader_ok(path, name):
    """Fake `Loader`: returns a SceneObject without touching trimesh/GLB parsing."""
    return SceneObject(mesh=object(), name=name)


def _fake_loader_raises_model_load_error(path, name):
    raise ModelLoadError(f"fake load failure for {path}")


class _FakeClock:
    """Deterministic, injectable clock: advances by a fixed step each call."""

    def __init__(self, step=0.01):
        self._time = 0.0
        self._step = step

    def __call__(self):
        current = self._time
        self._time += self._step
        return current


class _FakeRenderer:
    """Fake `Renderer3D`-shaped renderer: returns a small RGBA buffer instantly."""

    def __init__(self, width, height, fail_after=None):
        self._width = width
        self._height = height
        self._calls = 0
        self._fail_after = fail_after
        self.closed = False

    def render(self, scene):
        self._calls += 1
        if self._fail_after is not None and self._calls > self._fail_after:
            raise RenderError("fake render failure")
        return np.zeros((self._height, self._width, 4), dtype=np.uint8)

    def close(self):
        self.closed = True


def _make_renderer_factory(fail_after=None, raise_on_init=False):
    created = []

    def factory(width, height):
        if raise_on_init:
            raise RenderError("fake renderer init failure")
        renderer = _FakeRenderer(width, height, fail_after=fail_after)
        created.append(renderer)
        return renderer

    factory.created = created
    return factory


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    # A real file is enough for path-validation tests - its bytes are
    # never parsed since the loader is always faked.
    candidate_path = tmp_path / "candidate.glb"
    candidate_path.write_bytes(b"not-real-glb-bytes-never-parsed-by-fake-loader")
    original_bytes = candidate_path.read_bytes()

    # 1. Result contract: a successful run returns a well-formed BenchmarkResult.
    ok_factory = _make_renderer_factory()
    result = benchmark_candidate(
        candidate_path,
        warmup_frames=2,
        measured_frames=5,
        loader=_fake_loader_ok,
        renderer_factory=ok_factory,
        clock=_FakeClock(step=0.01),
    )
    check("benchmark_candidate returns a BenchmarkResult", isinstance(result, BenchmarkResult))
    check("successful result has success=True", result.success is True)
    check("successful result has error=None", result.error is None)
    check("successful result reports candidate_path", result.candidate_path == str(candidate_path))
    check("successful result reports measured_frames == requested", result.measured_frames == 5)
    check("successful result reports warmup_frames == requested", result.warmup_frames == 2)
    check("successful result reports a positive load_time_seconds", result.load_time_seconds > 0)
    check("successful result closes the renderer", ok_factory.created[0].closed is True)

    # 2. FPS / frame-time calculation, using the deterministic fake clock.
    #    Each render call advances the clock by exactly 0.01s, so every
    #    measured frame takes exactly 0.01s -> 100 FPS, deterministically.
    check("average_frame_time_seconds matches the fake clock's fixed step", abs(result.average_frame_time_seconds - 0.01) < 1e-9)
    check("average_fps == 1 / average_frame_time_seconds", abs(result.average_fps - 100.0) < 1e-6)

    # 3. to_dict() round-trip / JSON-serializable representation.
    as_dict = result.to_dict()
    check("to_dict() returns a plain dict", isinstance(as_dict, dict))
    check("to_dict() round-trips average_fps", as_dict["average_fps"] == result.average_fps)

    # 4. Missing candidate fails cleanly (never raises).
    missing_result = benchmark_candidate(
        tmp_path / "does_not_exist.glb",
        loader=_fake_loader_ok,
        renderer_factory=_make_renderer_factory(),
    )
    check("missing candidate returns success=False (no exception)", missing_result.success is False)
    check("missing candidate carries a clear error message", bool(missing_result.error))
    check("missing candidate reports 0.0 average_fps (never fabricated)", missing_result.average_fps == 0.0)

    # 5. Unsupported/malformed candidate extension fails cleanly.
    unsupported_path = tmp_path / "model.obj"
    unsupported_path.write_text("dummy")
    unsupported_result = benchmark_candidate(
        unsupported_path, loader=_fake_loader_ok, renderer_factory=_make_renderer_factory()
    )
    check("unsupported extension returns success=False", unsupported_result.success is False)
    check("unsupported extension result never reaches the loader/renderer", unsupported_result.load_time_seconds is None)

    # 6. Load failure (malformed candidate, from the loader's point of view).
    load_failure_result = benchmark_candidate(
        candidate_path, loader=_fake_loader_raises_model_load_error, renderer_factory=_make_renderer_factory()
    )
    check("load failure returns success=False", load_failure_result.success is False)
    check("load failure error message mentions the failure", "fake load failure" in (load_failure_result.error or ""))
    check("load failure reports average_fps 0.0 (never estimated)", load_failure_result.average_fps == 0.0)

    # 7. Renderer initialization failure.
    init_failure_result = benchmark_candidate(
        candidate_path, loader=_fake_loader_ok, renderer_factory=_make_renderer_factory(raise_on_init=True)
    )
    check("renderer init failure returns success=False", init_failure_result.success is False)
    check("renderer init failure still reports load_time_seconds (loading succeeded)", init_failure_result.load_time_seconds is not None)
    check("renderer init failure error message is clear", "init" in (init_failure_result.error or "").lower())

    # 8. Mid-render failure (renderer works for a while, then fails).
    render_failure_factory = _make_renderer_factory(fail_after=3)
    render_failure_result = benchmark_candidate(
        candidate_path,
        warmup_frames=0,
        measured_frames=10,
        loader=_fake_loader_ok,
        renderer_factory=render_failure_factory,
        clock=_FakeClock(),
    )
    check("mid-render failure returns success=False", render_failure_result.success is False)
    check("mid-render failure still closes the renderer", render_failure_factory.created[0].closed is True)
    check("mid-render failure never fabricates a nonzero FPS", render_failure_result.average_fps == 0.0)

    # 9. Multiple candidates can be benchmarked independently; one
    #    failure never crashes the batch or prevents the others.
    second_candidate_path = tmp_path / "second.glb"
    second_candidate_path.write_bytes(b"also-fake-bytes")

    def _mixed_loader(path, name):
        if "second" in str(path):
            raise ModelLoadError("second candidate intentionally fails to load")
        return SceneObject(mesh=object(), name=name)

    batch_results = benchmark_candidates(
        [candidate_path, second_candidate_path, tmp_path / "missing.glb"],
        loader=_mixed_loader,
        renderer_factory=_make_renderer_factory(),
        clock=_FakeClock(),
    )
    check("benchmark_candidates never raises for mixed outcomes", isinstance(batch_results, list))
    check("benchmark_candidates returns one result per input path", len(batch_results) == 3)
    check("first candidate in the batch succeeds", batch_results[0].success is True)
    check("second candidate in the batch fails cleanly (load error)", batch_results[1].success is False)
    check("third candidate in the batch fails cleanly (missing file)", batch_results[2].success is False)

    # 10. Candidate ranking: deterministic, by measured FPS, descending.
    fast_clock = _FakeClock(step=0.01)  # 100 FPS
    slow_clock = _FakeClock(step=0.05)  # 20 FPS
    fast_result = benchmark_candidate(
        candidate_path, candidate_name="fast", loader=_fake_loader_ok, renderer_factory=_make_renderer_factory(), clock=fast_clock
    )
    slow_result = benchmark_candidate(
        second_candidate_path,
        candidate_name="slow",
        loader=_fake_loader_ok,
        renderer_factory=_make_renderer_factory(),
        clock=slow_clock,
    )
    failed_result = benchmark_candidate(
        tmp_path / "missing2.glb", candidate_name="missing", loader=_fake_loader_ok, renderer_factory=_make_renderer_factory()
    )

    ranked = rank_candidates([slow_result, fast_result, failed_result])
    check("rank_candidates orders successful candidates by descending FPS", [r.candidate_name for r in ranked] == ["fast", "slow"])
    check("rank_candidates excludes failed candidates entirely", "missing" not in [r.candidate_name for r in ranked])

    # 11. A failed candidate can never win, even if it were somehow
    #     the only entry, or ranked among only-failed entries.
    check("best_candidate on an all-failed list returns None", best_candidate([failed_result]) is None)
    check("best_candidate picks the fastest successful candidate", best_candidate([slow_result, fast_result, failed_result]).candidate_name == "fast")

    # Ranking is stable/deterministic across repeated calls with the same input.
    check(
        "rank_candidates is deterministic across repeated calls",
        rank_candidates([slow_result, fast_result, failed_result]) == rank_candidates([fast_result, slow_result, failed_result]),
    )

    # 12. Source files are never modified by benchmarking.
    check("candidate file bytes are unchanged after all benchmark runs", candidate_path.read_bytes() == original_bytes)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
