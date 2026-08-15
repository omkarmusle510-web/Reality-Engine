"""Offline, deterministic tests for the Asset Optimizer's Block 8 integration layer.

No network access, no GPU/EGL, no real `gltfpack` binary. Blocks 1-7
are exercised through their own real, unmodified APIs - this file only
tests the orchestration in `apps.reality_painter.optimization.pipeline`.
Block 1 (`analyze_asset`) and Block 5 (`OptimizationCache`) run for
real against small local GLB fixtures generated with `trimesh` (the
same fixture pattern already used by `tests/test_asset_analyzer.py`
and `tests/test_candidate_generator.py`); Blocks 2/3/6 use the same
dependency-injection points those blocks already define, so no real
`gltfpack` binary, GPU, or hardware probing is ever required.
"""
import inspect
import sys
import tempfile
from pathlib import Path

import numpy as np
import trimesh

from apps.reality_painter.optimization import pipeline as pipeline_module
from apps.reality_painter.optimization.cache import CacheKey, OptimizationCache
from apps.reality_painter.optimization.hardware_profile import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    MemoryInfo,
    PerformanceTier,
    PlatformInfo,
)
from apps.reality_painter.optimization.pipeline import PipelineStatus, optimize_asset
from apps.reality_painter.optimization.selector import PerformancePolicy
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


# --- Shared fakes ----------------------------------------------------


class _FakeCompletedProcess:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


class _RealBytesRunner:
    """Fake Block 2 CommandRunner: writes real, valid GLB bytes as its 'optimized' output.

    This is what lets Block 1's analyze_asset() succeed against the
    candidate outputs too (not just the source), so quality scores are
    real, not skipped.
    """

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        output_path = Path(args[args.index("-o") + 1])
        output_path.write_bytes(self.payload)
        return _FakeCompletedProcess(returncode=0)


def _fake_benchmark_loader(path, name):
    return SceneObject(mesh=object(), name=name)


class _FakeRenderer:
    def __init__(self, width, height):
        self.closed = False

    def render(self, scene):
        return np.zeros((4, 4, 4), dtype=np.uint8)

    def close(self):
        self.closed = True


def _fake_renderer_factory(width, height):
    return _FakeRenderer(width, height)


class _FakeClock:
    def __init__(self, step=0.01):
        self._t = 0.0
        self._step = step

    def __call__(self):
        current = self._t
        self._t += self._step
        return current


def _fixed_hardware_profile(**kwargs):
    """A deterministic HardwareProfile stand-in - no real detection."""
    return HardwareProfile(
        cpu=CPUInfo(logical_cores=8),
        memory=MemoryInfo(total_ram_mb=16 * 1024),
        platform=PlatformInfo(system="Windows", release="11", architecture="AMD64"),
        gpu=GPUInfo(name="Fake GPU", vendor=None),
        tier=PerformanceTier.MEDIUM,  # target=60, minimum=30 (see hardware_profile.py)
    )


def _benchmark_kwargs():
    return dict(
        loader=_fake_benchmark_loader,
        renderer_factory=_fake_renderer_factory,
        clock=_FakeClock(step=0.01),  # -> 100 FPS, deterministic
        warmup_frames=1,
        measured_frames=3,
    )


def _run_pipeline(source_path, output_dir, cache_dir, **overrides):
    kwargs = dict(
        cache_dir=cache_dir,
        candidate_runner=overrides.pop("candidate_runner"),
        tool_path=overrides.pop("tool_path"),
        benchmark_kwargs=overrides.pop("benchmark_kwargs", _benchmark_kwargs()),
        hardware_profile_fn=overrides.pop("hardware_profile_fn", _fixed_hardware_profile),
    )
    kwargs.update(overrides)
    return optimize_asset(source_path, output_dir, **kwargs)


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source_path = tmp_path / "source.glb"
    box.export(source_path, file_type="glb")
    real_glb_bytes = source_path.read_bytes()

    fake_tool_path = tmp_path / "fake_gltfpack"
    fake_tool_path.write_text("#!/bin/sh\n")

    output_dir = tmp_path / "candidates"
    cache_dir = tmp_path / "opt_cache"

    def _fresh_runner():
        return _RealBytesRunner(real_glb_bytes)

    # ===========================================================================
    # 1. Complete successful pipeline.
    # ===========================================================================
    result = _run_pipeline(
        source_path, output_dir, cache_dir,
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
    )
    check("full pipeline returns SUCCESS", result.status == PipelineStatus.SUCCESS)
    check("full pipeline result has no error", result.error is None)
    check("full pipeline populates analysis", result.analysis is not None and result.analysis.triangle_count > 0)
    check("full pipeline populates candidates", len(result.candidates) == 4)
    check("full pipeline populates benchmark_results", len(result.benchmark_results) == 4)
    check("full pipeline populates hardware_profile", result.hardware_profile is not None)
    check("full pipeline populates decision", result.decision is not None)
    check("full pipeline selects a candidate", result.decision.selected_candidate is not None)
    check("full pipeline reports MEETS_TARGET (100 FPS beats MEDIUM's 60 target)", result.decision.status.value == "meets_target")
    check("full pipeline populates cache_metadata", result.cache_metadata is not None)
    check("full pipeline selected_asset_path exists on disk", result.selected_asset_path is not None and result.selected_asset_path.is_file())

    # ===========================================================================
    # 2. Block 1 analysis failure (corrupt source file).
    # ===========================================================================
    corrupt_source = tmp_path / "corrupt.glb"
    corrupt_source.write_bytes(b"not a real glb")
    result = _run_pipeline(
        corrupt_source, tmp_path / "candidates_2", tmp_path / "cache_2",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
    )
    check("analysis failure -> ANALYSIS_FAILED", result.status == PipelineStatus.ANALYSIS_FAILED)
    check("analysis failure carries an error message", bool(result.error))
    check("analysis failure never reaches candidate generation", result.candidates == [])

    # ===========================================================================
    # 3. Candidate generation failure (source vanishes before Block 2 runs).
    # ===========================================================================
    def _raising_generate_candidates(*args, **kwargs):
        raise RuntimeError("candidate generation exploded")

    result = _run_pipeline(
        source_path, tmp_path / "candidates_3", tmp_path / "cache_3",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        generate_candidates_fn=_raising_generate_candidates,
    )
    check("candidate generation failure -> CANDIDATE_GENERATION_FAILED", result.status == PipelineStatus.CANDIDATE_GENERATION_FAILED)
    check("candidate generation failure carries an error message", "exploded" in (result.error or ""))
    check("candidate generation failure never reaches benchmarking", result.benchmark_results == [])

    # No successful candidates at all (tool never resolves) is also CANDIDATE_GENERATION_FAILED.
    result = _run_pipeline(
        source_path, tmp_path / "candidates_3b", tmp_path / "cache_3b",
        candidate_runner=_fresh_runner(), tool_path="/definitely/not/a/real/tool",
    )
    check(
        "unavailable tool: only the no-tool ORIGINAL candidate succeeds",
        [c.candidate_name for c in result.candidates if c.success] == ["ORIGINAL"],
    )
    check(
        "unavailable tool: every tool-requiring candidate reports a clear unavailable-tool error",
        all(bool(c.error) for c in result.candidates if not c.success),
    )
    check(
        "unavailable tool: pipeline still proceeds using the one successful (ORIGINAL) candidate",
        result.status not in (PipelineStatus.ANALYSIS_FAILED, PipelineStatus.CANDIDATE_GENERATION_FAILED),
    )

    # ===========================================================================
    # 4. Benchmark failure (every candidate fails to benchmark).
    # ===========================================================================
    def _always_failing_benchmark(path, candidate_name=None, **kwargs):
        from apps.reality_painter.optimization.benchmark import BenchmarkResult
        return BenchmarkResult(
            candidate_path=str(path), candidate_name=candidate_name or "x", success=False,
            error="forced failure", load_time_seconds=None, average_fps=0.0,
            average_frame_time_seconds=0.0, measured_frames=0, warmup_frames=0,
            render_width=1, render_height=1,
        )

    result = _run_pipeline(
        source_path, tmp_path / "candidates_4", tmp_path / "cache_4",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        benchmark_fn=_always_failing_benchmark,
    )
    check("all-benchmarks-fail -> NO_VALID_CANDIDATE", result.status == PipelineStatus.NO_VALID_CANDIDATE)
    check("all-benchmarks-fail never fabricates a nonzero FPS", all(b.average_fps == 0.0 for b in result.benchmark_results))
    check("all-benchmarks-fail never reaches a decision", result.decision is None)

    # ===========================================================================
    # 5. Hardware detection failure.
    # ===========================================================================
    def _raising_hardware_profile(**kwargs):
        raise RuntimeError("hardware probe exploded")

    result = _run_pipeline(
        source_path, tmp_path / "candidates_5", tmp_path / "cache_5",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        hardware_profile_fn=_raising_hardware_profile,
    )
    check("hardware detection failure -> HARDWARE_DETECTION_FAILED", result.status == PipelineStatus.HARDWARE_DETECTION_FAILED)
    check("hardware detection failure carries an error message", "exploded" in (result.error or ""))
    check("hardware detection failure never reaches a decision", result.decision is None)

    # ===========================================================================
    # 6. Decision (Block 7) failure.
    # ===========================================================================
    def _raising_decide(*args, **kwargs):
        raise RuntimeError("decision exploded")

    result = _run_pipeline(
        source_path, tmp_path / "candidates_6", tmp_path / "cache_6",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        decide_fn=_raising_decide,
    )
    check("decision failure -> DECISION_FAILED", result.status == PipelineStatus.DECISION_FAILED)
    check("decision failure carries an error message", "exploded" in (result.error or ""))
    check("decision failure never reaches cache store", result.cache_metadata is None)

    # ===========================================================================
    # 7 & 16. No valid candidate / rejected optimization path (policy too strict).
    # ===========================================================================
    impossible_policy = PerformancePolicy(target_fps=99999.0, minimum_fps=99999.0)
    result = _run_pipeline(
        source_path, tmp_path / "candidates_7", tmp_path / "cache_7",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        policy_override=impossible_policy,
    )
    check("impossible policy -> NO_VALID_CANDIDATE", result.status == PipelineStatus.NO_VALID_CANDIDATE)
    check("impossible policy carries Block 4/7's own rejection reason", bool(result.error))
    check("impossible policy never selects an asset path", result.selected_asset_path is None)
    check("impossible policy still reports the underlying decision for inspection", result.decision is not None)

    # ===========================================================================
    # 8. Cache (Block 5) failure.
    # ===========================================================================
    class _RaisingCache:
        def lookup(self, key):
            from apps.reality_painter.optimization.cache import CacheLookupResult, CacheStatus
            return CacheLookupResult(CacheStatus.MISS, key.value, None, None, "no entry")

        def store(self, *args, **kwargs):
            raise RuntimeError("cache store exploded")

    result = _run_pipeline(
        source_path, tmp_path / "candidates_8", tmp_path / "cache_8",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        cache=_RaisingCache(),
    )
    check("cache store failure -> CACHE_FAILED", result.status == PipelineStatus.CACHE_FAILED)
    check("cache store failure carries an error message", "exploded" in (result.error or ""))
    check("cache store failure still carries the successful decision", result.decision is not None and result.decision.selected_candidate is not None)

    # ===========================================================================
    # 9. Deterministic result across repeated runs (same inputs -> same outcome).
    # ===========================================================================
    result_a = _run_pipeline(
        source_path, tmp_path / "candidates_9a", tmp_path / "cache_9a",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
    )
    result_b = _run_pipeline(
        source_path, tmp_path / "candidates_9b", tmp_path / "cache_9b",
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
    )
    check("repeated runs produce the same status", result_a.status == result_b.status)
    check("repeated runs select the same candidate", result_a.decision.selected_candidate == result_b.decision.selected_candidate)
    check("repeated runs produce the same decision reason", result_a.decision.reason == result_b.decision.reason)
    check("repeated runs produce the same quality scores", result_a.quality_scores == result_b.quality_scores)

    # ===========================================================================
    # 10. Correct propagation of Block 1-7 results into the structured result.
    # ===========================================================================
    result = result_a
    check("propagated analysis matches a real Block 1 AssetAnalysis", result.analysis.file_path == str(source_path) or Path(result.analysis.file_path) == source_path)
    check("propagated candidates are real CandidateResult objects", all(hasattr(c, "operations_applied") for c in result.candidates))
    check("propagated benchmark_results are real BenchmarkResult objects", all(hasattr(b, "average_frame_time_seconds") for b in result.benchmark_results))
    check("propagated decision.evaluated_candidates is non-empty on success", len(result.decision.evaluated_candidates) > 0)
    check("propagated cache_result reflects the fresh store", result.cache_result.status.value == "hit")

    # ===========================================================================
    # 11. No fabricated FPS/quality values.
    # ===========================================================================
    # All four candidates are byte-identical to the source (see _RealBytesRunner),
    # so every quality score must be exactly 1.0 - never a placeholder/guessed value.
    check("quality scores are derived from real geometry (all == 1.0 for identical candidates)", all(abs(v - 1.0) < 1e-9 for v in result.quality_scores.values()))
    # FakeClock step is fixed at 0.01s -> exactly 100 FPS; never estimated.
    check("benchmark FPS matches the deterministic fake clock exactly (no fabrication)", all(abs(b.average_fps - 100.0) < 1e-6 for b in result.benchmark_results if b.success))

    # ===========================================================================
    # 12. No network access (source inspection).
    # ===========================================================================
    source_text = inspect.getsource(pipeline_module)
    import_lines = [
        line.strip() for line in source_text.splitlines()
        if line.strip().startswith("import ") or line.strip().startswith("from ")
    ]
    forbidden_network_tokens = ("requests", "socket", "http.client", "urllib")
    check(
        "pipeline module has no network-related import statements",
        not any(any(token in line for token in forbidden_network_tokens) for line in import_lines),
    )

    # ===========================================================================
    # 13. No direct Renderer3D implementation/instantiation.
    # ===========================================================================
    check(
        "pipeline module never imports or instantiates Renderer3D/pyrender directly",
        "Renderer3D(" not in source_text and "import pyrender" not in source_text and "OffscreenRenderer" not in source_text,
    )

    # ===========================================================================
    # 14. No duplicate optimization logic (selection/benchmark/analysis not reimplemented).
    # ===========================================================================
    check(
        "pipeline module never imports Block 4's select_candidate (only decide_optimization/PerformancePolicy)",
        "from apps.reality_painter.optimization.selector import" in source_text
        and "select_candidate" not in inspect.getsource(pipeline_module.optimize_asset),
    )
    check(
        "pipeline module never defines its own FPS/benchmark computation",
        "perf_counter" not in source_text and "time.time()" not in source_text,
    )

    # ===========================================================================
    # 15. Already-cached asset path (cache hit short-circuits the pipeline).
    # ===========================================================================
    cache_15_dir = tmp_path / "cache_15"
    call_counter = {"count": 0}

    def _counting_generate_candidates(*args, **kwargs):
        call_counter["count"] += 1
        return generate_candidates_real(*args, **kwargs)

    from apps.reality_painter.optimization.candidate_generator import generate_candidates as generate_candidates_real

    first = _run_pipeline(
        source_path, tmp_path / "candidates_15", cache_15_dir,
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        generate_candidates_fn=_counting_generate_candidates,
    )
    check("cache-miss run performs real candidate generation", call_counter["count"] == 1)
    check("cache-miss run succeeds", first.status == PipelineStatus.SUCCESS)

    second = _run_pipeline(
        source_path, tmp_path / "candidates_15b", cache_15_dir,
        candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path),
        generate_candidates_fn=_counting_generate_candidates,
    )
    check("cache-hit run returns CACHED status", second.status == PipelineStatus.CACHED)
    check("cache-hit run never re-invokes candidate generation", call_counter["count"] == 1)
    check("cache-hit run still returns a usable selected_asset_path", second.selected_asset_path is not None and second.selected_asset_path.is_file())
    check("cache-hit run performs no analysis/benchmarking/decision work", second.analysis is None and second.benchmark_results == [] and second.decision is None)

    # ===========================================================================
    # 17. Structured error/result on every failure path (never a bare exception).
    # ===========================================================================
    from apps.reality_painter.optimization.pipeline import OptimizationPipelineResult

    all_failure_results = [
        result_a if False else None,  # placeholder to keep list literal simple
    ]
    del all_failure_results
    for label, res, expected_status in [
        ("analysis failure", _run_pipeline(corrupt_source, tmp_path / "c17a", tmp_path / "cache17a", candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path)), PipelineStatus.ANALYSIS_FAILED),
        ("hardware failure", _run_pipeline(source_path, tmp_path / "c17b", tmp_path / "cache17b", candidate_runner=_fresh_runner(), tool_path=str(fake_tool_path), hardware_profile_fn=_raising_hardware_profile), PipelineStatus.HARDWARE_DETECTION_FAILED),
    ]:
        check(f"{label} returns an OptimizationPipelineResult, never raises", isinstance(res, OptimizationPipelineResult))
        check(f"{label} status matches expectation", res.status == expected_status)
        check(f"{label} carries a non-empty error", bool(res.error))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
