"""Final integration layer for Reality Painter's Asset Optimizer (Block 8).

`optimize_asset()` is the single high-level entry point that wires the
already-completed Blocks 1-7 into one deterministic workflow:

    source GLB
        -> Block 1  analyze_asset()              (analysis)
        -> Block 2  generate_candidates()          (candidate GLBs)
        -> Block 3  benchmark_candidate()           (real render timing, per candidate)
        -> Block 6  build_hardware_profile()        (local machine profile)
        -> Block 7  decide_optimization()           (delegates to Block 4's
                                                        select_candidate() internally)
        -> Block 5  OptimizationCache.store()/lookup() (persist/reuse the result)

This module never reimplements any block's logic:
    - It never re-derives FPS, benchmarks, or renders anything itself
      (Block 3/`engine.rendering.renderer.Renderer3D` are only reached
      through `benchmark_candidate`).
    - It never calls `select_candidate` directly - `decide_optimization`
      (Block 7) already does that internally; calling it a second time
      here would be duplicate selection logic.
    - It never detects hardware itself - `build_hardware_profile`
      (Block 6) is the sole source of that data.
    - It never implements a second cache - `OptimizationCache` (Block 5)
      is the sole persistence layer.
    - It performs no network access and no direct GLB rendering.

QUALITY SCORES
---------------
Block 7/Block 4 require a `quality_scores` mapping but define no
opinion on how it's computed (see `decision.py`'s docstring: "e.g.
derived from Block 1's AssetAnalysis"). This module derives one
deterministically, from real Block 1 analysis of each candidate output:
the fraction of the original asset's triangle count the candidate
still has (clamped to [0, 1]). This is never a fabricated/hardcoded
number - it is always a pure function of two real `AssetAnalysis`
results. A candidate whose output can't be analyzed (e.g. a corrupt
optimizer output) is excluded from `quality_scores` entirely - Block 7
already treats a missing score as "exclude, never fabricate."

FAILURE SAFETY
----------------
Every stage is wrapped so an expected or unexpected failure produces a
structured `OptimizationPipelineResult` with a `status` and `error`
rather than an uncaught exception. Every block function used here
already returns/raises according to its own documented contract; this
module only translates those outcomes into one unified result shape.

DEPENDENCY INJECTION
-----------------------
Every stage's underlying callable (`analyze_fn`, `generate_candidates_fn`,
`benchmark_fn`, `hardware_profile_fn`, `decide_fn`) is overridable,
matching the injection convention already used throughout this
package (`CommandRunner`, `Loader`/`RendererFactory`,
`GPUCommandRunner`, ...). Defaults are always the real Block 1-7
functions; overrides exist purely so callers/tests can exercise a
specific failure path without needing real GPU/EGL/network access.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from apps.reality_painter.optimization.analyzer import (
    AssetAnalysis,
    AssetAnalysisError,
    analyze_asset,
)
from apps.reality_painter.optimization.benchmark import BenchmarkResult, benchmark_candidate
from apps.reality_painter.optimization.cache import (
    CacheEntryMetadata,
    CacheKey,
    CacheLookupResult,
    CacheStatus,
    OptimizationCache,
)
from apps.reality_painter.optimization.candidate_generator import (
    CandidateGeneratorError,
    CandidateResult,
    CandidateSpec,
    generate_candidates,
)
from apps.reality_painter.optimization.decision import OptimizationDecision, decide_optimization
from apps.reality_painter.optimization.hardware_profile import HardwareProfile, build_hardware_profile
from apps.reality_painter.optimization.selector import PerformancePolicy


class PipelineStatus(str, Enum):
    """The outcome category of one `optimize_asset()` call."""

    #: A fresh optimization ran end to end and a candidate was cached.
    SUCCESS = "success"
    #: A previously cached, still-valid result was returned; no
    #: analysis/generation/benchmarking/decision work was re-run.
    CACHED = "cached"
    ANALYSIS_FAILED = "analysis_failed"
    CANDIDATE_GENERATION_FAILED = "candidate_generation_failed"
    HARDWARE_DETECTION_FAILED = "hardware_detection_failed"
    DECISION_FAILED = "decision_failed"
    #: Every candidate was rejected (below-minimum, all benchmarks
    #: failed, or nothing scoreable) - Block 7's own verdict.
    NO_VALID_CANDIDATE = "no_valid_candidate"
    CACHE_FAILED = "cache_failed"


@dataclass
class OptimizationPipelineResult:
    """The complete, structured outcome of one `optimize_asset()` call.

    Every block's real output is carried through unmodified so a
    caller (or a test) can inspect exactly what each stage produced,
    never a summarized/re-derived version of it.
    """

    source_path: str
    source_identity: str
    status: PipelineStatus
    error: Optional[str] = None
    analysis: Optional[AssetAnalysis] = None
    candidates: List[CandidateResult] = field(default_factory=list)
    benchmark_results: List[BenchmarkResult] = field(default_factory=list)
    hardware_profile: Optional[HardwareProfile] = None
    quality_scores: Dict[str, float] = field(default_factory=dict)
    decision: Optional[OptimizationDecision] = None
    cache_result: Optional[CacheLookupResult] = None
    cache_metadata: Optional[CacheEntryMetadata] = None
    selected_asset_path: Optional[Path] = None


def _quality_score(candidate_analysis: AssetAnalysis, original_analysis: AssetAnalysis) -> float:
    """Deterministically derives a quality score from two real Block 1 analyses.

    The fraction of the original triangle count the candidate retains,
    clamped to [0, 1]. Never a hardcoded/guessed value - always a pure
    function of `analyze_asset()`'s own real output for both files.
    """
    if original_analysis.triangle_count <= 0:
        return 0.0
    ratio = candidate_analysis.triangle_count / original_analysis.triangle_count
    return max(0.0, min(1.0, ratio))


def optimize_asset(
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    source_identity: Optional[str] = None,
    cache: Optional[OptimizationCache] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    candidate_specs: Optional[Tuple[CandidateSpec, ...]] = None,
    tool_path: Optional[str] = None,
    candidate_runner: Optional[Callable[[List[str]], Any]] = None,
    benchmark_kwargs: Optional[Dict[str, Any]] = None,
    hardware_kwargs: Optional[Dict[str, Any]] = None,
    policy_override: Optional[PerformancePolicy] = None,
    analyze_fn: Callable[..., AssetAnalysis] = analyze_asset,
    generate_candidates_fn: Callable[..., List[CandidateResult]] = generate_candidates,
    benchmark_fn: Callable[..., BenchmarkResult] = benchmark_candidate,
    hardware_profile_fn: Callable[..., HardwareProfile] = build_hardware_profile,
    decide_fn: Callable[..., OptimizationDecision] = decide_optimization,
) -> OptimizationPipelineResult:
    """Runs the full Block 1-7 optimization workflow for one source asset.

    Args:
        source_path: Path to the source `.glb`/`.gltf` file to optimize.
        output_dir: Directory Block 2 writes candidate files into.
        source_identity: Stable identity used for the Block 5 cache key
            (see `CacheKey.build`). Defaults to `str(source_path)`.
        cache: An `OptimizationCache` instance to use. Defaults to a
            new one bound to `cache_dir`.
        cache_dir: Passed to `OptimizationCache(cache_dir=...)` if
            `cache` is not supplied.
        use_cache: If False, skips the cache lookup and always runs the
            full pipeline (the result is still stored afterward unless
            selection fails).
        candidate_specs: Forwarded to `generate_candidates` (Block 2).
        tool_path: Forwarded to `generate_candidates` (Block 2).
        candidate_runner: Forwarded to `generate_candidates` as its
            `runner` (Block 2's `CommandRunner` injection point).
        benchmark_kwargs: Extra keyword arguments forwarded to
            `benchmark_fn` for every candidate (e.g. `loader`,
            `renderer_factory`, `clock`, `warmup_frames`,
            `measured_frames`, `width`, `height`).
        hardware_kwargs: Extra keyword arguments forwarded to
            `hardware_profile_fn`.
        policy_override: Forwarded to `decide_fn` (Block 7).
        analyze_fn: Override for Block 1's `analyze_asset`.
        generate_candidates_fn: Override for Block 2's `generate_candidates`.
        benchmark_fn: Override for Block 3's `benchmark_candidate`.
        hardware_profile_fn: Override for Block 6's `build_hardware_profile`.
        decide_fn: Override for Block 7's `decide_optimization`.

    Returns:
        A structured `OptimizationPipelineResult`. Never raises: every
        expected or unexpected failure at any stage is translated into
        a `PipelineStatus` plus a human-readable `error`.
    """
    source = Path(source_path)
    identity = source_identity if source_identity is not None else str(source)
    active_cache = cache if cache is not None else OptimizationCache(cache_dir=cache_dir)
    active_benchmark_kwargs = dict(benchmark_kwargs) if benchmark_kwargs else {}
    active_hardware_kwargs = dict(hardware_kwargs) if hardware_kwargs else {}

    result = OptimizationPipelineResult(
        source_path=str(source), source_identity=identity, status=PipelineStatus.ANALYSIS_FAILED
    )

    cache_key = CacheKey.build(identity)

    if use_cache:
        try:
            cache_lookup = active_cache.lookup(cache_key)
        except Exception:
            cache_lookup = None
        if cache_lookup is not None:
            result.cache_result = cache_lookup
            if cache_lookup.status == CacheStatus.HIT:
                result.status = PipelineStatus.CACHED
                result.selected_asset_path = cache_lookup.asset_path
                result.cache_metadata = cache_lookup.metadata
                return result

    # --- Block 1: analysis --------------------------------------------
    try:
        analysis = analyze_fn(source)
    except AssetAnalysisError as exc:
        result.status = PipelineStatus.ANALYSIS_FAILED
        result.error = str(exc)
        return result
    except Exception as exc:
        result.status = PipelineStatus.ANALYSIS_FAILED
        result.error = f"Unexpected analysis error: {exc}"
        return result
    result.analysis = analysis

    # --- Block 2: candidate generation ---------------------------------
    try:
        candidates = generate_candidates_fn(
            source, output_dir, specs=candidate_specs, tool_path=tool_path, runner=candidate_runner
        )
    except CandidateGeneratorError as exc:
        result.status = PipelineStatus.CANDIDATE_GENERATION_FAILED
        result.error = str(exc)
        return result
    except Exception as exc:
        result.status = PipelineStatus.CANDIDATE_GENERATION_FAILED
        result.error = f"Unexpected candidate generation error: {exc}"
        return result
    result.candidates = candidates

    successful_candidates = [c for c in candidates if c.success]
    if not successful_candidates:
        result.status = PipelineStatus.CANDIDATE_GENERATION_FAILED
        result.error = "No candidate was generated successfully."
        return result

    # --- Block 3: benchmarking (real, per candidate) --------------------
    benchmark_results: List[BenchmarkResult] = []
    try:
        for candidate in successful_candidates:
            benchmark_results.append(
                benchmark_fn(
                    candidate.output_path, candidate_name=candidate.candidate_name, **active_benchmark_kwargs
                )
            )
    except Exception as exc:
        result.status = PipelineStatus.NO_VALID_CANDIDATE
        result.error = f"Unexpected benchmarking error: {exc}"
        result.benchmark_results = benchmark_results
        return result
    result.benchmark_results = benchmark_results

    if not any(b.success for b in benchmark_results):
        result.status = PipelineStatus.NO_VALID_CANDIDATE
        result.error = "No candidate completed benchmarking successfully."
        return result

    # --- Block 6: hardware profile --------------------------------------
    try:
        hardware_profile = hardware_profile_fn(**active_hardware_kwargs)
    except Exception as exc:
        result.status = PipelineStatus.HARDWARE_DETECTION_FAILED
        result.error = f"Hardware detection failed: {exc}"
        return result
    result.hardware_profile = hardware_profile

    # --- Quality scores: real, derived from Block 1 on each candidate ---
    quality_scores: Dict[str, float] = {}
    for candidate in successful_candidates:
        try:
            candidate_analysis = analyze_fn(candidate.output_path)
        except Exception:
            continue
        quality_scores[candidate.candidate_name] = _quality_score(candidate_analysis, analysis)
    result.quality_scores = quality_scores

    # --- Block 7 (delegates to Block 4 internally): decision -------------
    try:
        decision = decide_fn(hardware_profile, benchmark_results, quality_scores, policy_override=policy_override)
    except Exception as exc:
        result.status = PipelineStatus.DECISION_FAILED
        result.error = f"Optimization decision failed: {exc}"
        return result
    result.decision = decision

    if decision.selected_candidate is None:
        result.status = PipelineStatus.NO_VALID_CANDIDATE
        result.error = decision.reason
        return result

    selected = next(
        (c for c in successful_candidates if c.candidate_name == decision.selected_candidate), None
    )
    if selected is None:
        result.status = PipelineStatus.NO_VALID_CANDIDATE
        result.error = f"Selected candidate {decision.selected_candidate!r} has no generated output."
        return result

    # --- Block 5: cache store --------------------------------------------
    try:
        metadata = active_cache.store(
            cache_key, identity, selected.output_path, selected_candidate=decision.selected_candidate
        )
        stored_lookup = active_cache.lookup(cache_key)
    except Exception as exc:
        result.status = PipelineStatus.CACHE_FAILED
        result.error = f"Cache store failed: {exc}"
        return result

    result.cache_metadata = metadata
    result.cache_result = stored_lookup
    result.selected_asset_path = stored_lookup.asset_path
    result.status = PipelineStatus.SUCCESS
    return result
