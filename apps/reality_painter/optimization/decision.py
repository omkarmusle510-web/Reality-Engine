"""Optimization decision/orchestration layer for Reality Painter's Asset Optimizer.

Block 7 sits downstream of every other optimization block but calls
none of them - it performs no analysis (Block 1), no candidate
generation (Block 2), no benchmark execution (Block 3), no hardware
detection (Block 6), no caching (Block 5), no rendering, no GLB
loading, and no network access of any kind. It only orchestrates
results those blocks have already produced:

    HardwareProfile (Block 6, already built)
        -> recommended_policy_for_tier() (Block 6, already implemented)
        -> PerformancePolicy
    already-measured BenchmarkResults (Block 3, already run)
    + already-computed candidate quality scores (caller-supplied metadata)
        -> SelectionCandidate list (Block 4's own input contract)
        -> select_candidate() (Block 4, the sole selection authority)
        -> OptimizationDecision

This module never re-implements Block 4's selection algorithm (target
vs minimum FPS, quality-aware tie-breaking, deterministic ordering) -
`select_candidate` is called directly and its `SelectionResult` is
carried through unchanged into `OptimizationDecision`. It never
predicts FPS from hardware specs and never fabricates a quality score:
a candidate whose quality score is not supplied by the caller is
excluded from selection entirely (see `excluded_candidates`) rather
than defaulted to some invented value. Actual measured benchmark FPS
(`BenchmarkResult.average_fps`) is always the source of truth for
performance; `HardwareProfile`/Block 6 only ever supplies a *policy*
(target/minimum FPS to select against), never a substitute for a real
measurement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

from apps.reality_painter.optimization.benchmark import BenchmarkResult
from apps.reality_painter.optimization.hardware_profile import (
    HardwareProfile,
    PerformanceTier,
    recommended_policy_for_tier,
)
from apps.reality_painter.optimization.selector import (
    PerformancePolicy,
    SelectionCandidate,
    SelectionResult,
    SelectionStatus,
    select_candidate,
)


@dataclass(frozen=True)
class OptimizationDecision:
    """The final, structured outcome of one Block 7 decision.

    Carries `SelectionResult` (Block 4's own verdict) through
    unchanged, plus the hardware/policy context that produced the
    `PerformancePolicy` it was evaluated against, plus any candidates
    this layer itself excluded before selection ever ran (see
    `excluded_candidates`).

    Attributes:
        hardware_tier: The `PerformanceTier` the supplied
            `HardwareProfile` was classified as.
        policy: The `PerformancePolicy` selection was evaluated
            against - either `policy_override`, if supplied to
            `decide_optimization`, or Block 6's recommendation for
            `hardware_tier`.
        status: Mirrors `SelectionResult.status` - see
            `apps.reality_painter.optimization.selector.SelectionStatus`.
        selected_candidate: Mirrors `SelectionResult.selected_candidate`.
        reason: Mirrors `SelectionResult.reason` - always non-empty.
        evaluated_candidates: Mirrors `SelectionResult.evaluated_candidates`
            - candidates Block 4 actually considered.
        rejected_candidates: Mirrors `SelectionResult.rejected_candidates`
            - candidates Block 4 considered but did not select.
        excluded_candidates: Candidate names present in
            `benchmark_results` but never passed to Block 4 at all,
            because no quality score was supplied for them in
            `quality_scores`. Disjoint from
            `evaluated_candidates`/`rejected_candidates`. Never
            populated by fabricating a quality score.
        selection: The underlying `SelectionResult` from Block 4, for
            a caller that wants the untouched original.
    """

    hardware_tier: PerformanceTier
    policy: PerformancePolicy
    status: SelectionStatus
    selected_candidate: Optional[str]
    reason: str
    evaluated_candidates: Tuple[str, ...]
    rejected_candidates: Tuple[str, ...]
    excluded_candidates: Tuple[str, ...]
    selection: SelectionResult


def _build_policy(hardware_profile: HardwareProfile, policy_override: Optional[PerformancePolicy]) -> PerformancePolicy:
    """Resolves the `PerformancePolicy` to select against.

    Args:
        hardware_profile: The already-built hardware profile (Block 6).
        policy_override: An explicit policy supplied by the caller,
            taking precedence over the hardware-tier recommendation
            when given.

    Returns:
        `policy_override` if supplied, otherwise Block 6's
        `recommended_policy_for_tier(hardware_profile.tier)` converted
        into a `PerformancePolicy`.
    """
    if policy_override is not None:
        return policy_override

    recommendation = recommended_policy_for_tier(hardware_profile.tier)
    return PerformancePolicy(target_fps=recommendation.target_fps, minimum_fps=recommendation.minimum_fps)


def _build_candidates(
    benchmark_results: Sequence[BenchmarkResult],
    quality_scores: Mapping[str, float],
) -> Tuple[list, Tuple[str, ...]]:
    """Converts already-measured `BenchmarkResult`s into `SelectionCandidate`s.

    A candidate is excluded (never handed to Block 4) only when no
    quality score was supplied for its name - this layer never
    fabricates one. A candidate whose benchmark itself failed IS still
    passed through (as `benchmark_success=False`), since Block 4 is
    already responsible for correctly rejecting those - duplicating
    that check here would risk diverging from Block 4's own logic.

    Args:
        benchmark_results: Already-run `BenchmarkResult`s (Block 3).
        quality_scores: Caller-supplied quality metric per candidate
            name (e.g. derived from Block 1's `AssetAnalysis`).

    Returns:
        `(selection_candidates, excluded_names)` - the candidates to
        pass to `select_candidate`, and the names excluded for lacking
        a quality score, in input order.
    """
    selection_candidates = []
    excluded_names = []

    for result in benchmark_results:
        quality_score = quality_scores.get(result.candidate_name)
        if quality_score is None:
            excluded_names.append(result.candidate_name)
            continue
        selection_candidates.append(
            SelectionCandidate(
                name=result.candidate_name,
                average_fps=result.average_fps,
                quality_score=quality_score,
                benchmark_success=result.success,
            )
        )

    return selection_candidates, tuple(excluded_names)


def decide_optimization(
    hardware_profile: HardwareProfile,
    benchmark_results: Sequence[BenchmarkResult],
    quality_scores: Mapping[str, float],
    policy_override: Optional[PerformancePolicy] = None,
) -> OptimizationDecision:
    """Decides which already-benchmarked candidate (if any) to use.

    Pure orchestration over already-computed inputs: performs no
    analysis, generation, benchmarking, hardware detection, caching,
    rendering, or network access itself. Never raises for malformed or
    empty input - degenerate cases (no results, no quality scores, all
    benchmarks failed, nothing above the minimum) all resolve to a
    `SelectionStatus` from Block 4, exactly as `select_candidate`
    itself already guarantees.

    Args:
        hardware_profile: An already-built `HardwareProfile` (Block 6).
        benchmark_results: Already-measured `BenchmarkResult`s for the
            candidates to decide among (Block 3). Real, measured
            `average_fps` values are always the performance source of
            truth - never re-estimated here.
        quality_scores: Caller-supplied, already-computed quality
            metric per candidate name (e.g. derived from Block 1's
            `AssetAnalysis`). A candidate whose name is absent here is
            excluded from selection - see `OptimizationDecision
            .excluded_candidates` - rather than assigned a fabricated
            score.
        policy_override: An explicit `PerformancePolicy` to select
            against, overriding the hardware-tier recommendation. Only
            the recommendation's *source* changes; the underlying
            decision authority is still exclusively Block 4's
            `select_candidate`.

    Returns:
        A structured `OptimizationDecision`. Deterministic: the same
        inputs always produce the same decision, since both this
        function and `select_candidate` are pure and side-effect-free.
    """
    policy = _build_policy(hardware_profile, policy_override)
    selection_candidates, excluded_names = _build_candidates(benchmark_results, quality_scores)

    selection = select_candidate(selection_candidates, policy)

    return OptimizationDecision(
        hardware_tier=hardware_profile.tier,
        policy=policy,
        status=selection.status,
        selected_candidate=selection.selected_candidate,
        reason=selection.reason,
        evaluated_candidates=selection.evaluated_candidates,
        rejected_candidates=selection.rejected_candidates,
        excluded_candidates=excluded_names,
        selection=selection,
    )
