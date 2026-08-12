"""Candidate selection / performance policy for Reality Painter's Asset Optimizer.

Block 4 sits downstream of Block 1 (`analyzer.py`), Block 2
(`candidate_generator.py`), and Block 3 (`benchmark.py`), but imports
none of them and performs no rendering, network access, or file I/O of
its own. It operates purely on typed, already-computed inputs
(`SelectionCandidate`) and a declarative `PerformancePolicy`, and
deterministically decides which candidate - if any - should be used.

    Benchmark results + candidate metadata -> PerformancePolicy -> SelectionResult

This module never invents hardware information, never estimates FPS,
and never blindly picks the highest-FPS candidate: selection is
quality-aware, driven entirely by `SelectionCandidate.quality_score` -
a deterministic metric the caller derives however it likes (e.g. from
Block 1's `AssetAnalysis`, such as texture resolution or triangle
count) - so this module stays independent of any specific quality
metric's definition.

Selection policy, in order:
    1. Malformed candidates (missing/invalid name, non-finite or
       negative `average_fps`/`quality_score`) are filtered out first
       and never considered for selection.
    2. Candidates whose benchmark did not succeed
       (`benchmark_success=False`) are filtered out next.
    3. Among the remainder, candidates below `policy.minimum_fps` are
       excluded entirely - performance below the minimum is never an
       acceptable trade for quality.
    4. Among candidates at or above `policy.minimum_fps`:
        - If any also meet `policy.target_fps`, the one with the
          highest `quality_score` wins (ties broken by higher
          `average_fps`, then by name for full determinism) - this is
          the quality-aware step: a candidate need not be the fastest
          to be selected, only fast enough.
        - If none meet `policy.target_fps`, the fastest candidate
          (ties broken by `quality_score`, then name) is selected as a
          documented fallback, and the result is flagged
          `BELOW_TARGET` rather than `MEETS_TARGET` so a caller can
          distinguish "best available" from "as requested."

Every `SelectionResult` carries a human-readable `reason` explaining
why its `selected_candidate` (or lack of one) was chosen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple


class SelectionStatus(str, Enum):
    """The outcome category of one `select_candidate` call."""

    #: A candidate was selected and meets `policy.target_fps`.
    MEETS_TARGET = "meets_target"
    #: A candidate was selected (at or above `policy.minimum_fps`) but
    #: none reached `policy.target_fps`.
    BELOW_TARGET = "below_target"
    #: No candidate satisfied the minimum performance requirement (or
    #: none completed benchmarking successfully, or none were given).
    NO_VALID_CANDIDATE = "no_valid_candidate"
    #: Every candidate supplied had malformed/incomplete data.
    INVALID_DATA = "invalid_data"


@dataclass(frozen=True)
class SelectionCandidate:
    """One benchmarked, analyzable candidate, as typed input to the selector.

    This is intentionally a small, independent contract - not
    `apps.reality_painter.optimization.benchmark.BenchmarkResult` or
    `apps.reality_painter.optimization.analyzer.AssetAnalysis` - so
    Block 4 never imports Blocks 1-3. A caller builds one of these per
    candidate from whatever those blocks already produced.

    Attributes:
        name: Candidate identifier (e.g. a `BenchmarkResult.candidate_name`).
        average_fps: Measured average FPS (e.g.
            `BenchmarkResult.average_fps`). Must be finite and >= 0.
        quality_score: A deterministic, caller-defined quality metric
            where higher is always better (e.g. derived from
            `AssetAnalysis` - texture resolution, absence of
            aggressive simplification, etc.). Must be finite and >= 0.
            This module never computes or interprets its units - it
            only compares candidates by it.
        benchmark_success: Whether this candidate's benchmark run
            completed successfully (mirrors `BenchmarkResult.success`).
            A candidate with `False` here is never selectable.
    """

    name: str
    average_fps: float
    quality_score: float
    benchmark_success: bool = True


@dataclass(frozen=True)
class PerformancePolicy:
    """Declarative, configurable selection thresholds.

    Attributes:
        target_fps: The desired performance level. A candidate meeting
            this is preferred and selected by quality among its peers.
        minimum_fps: The hard floor. A candidate below this is never
            selected, regardless of quality.

    Raises:
        ValueError: If `minimum_fps` exceeds `target_fps`, or either is
            negative/non-finite - an inconsistent policy is rejected at
            construction rather than producing confusing results later.
    """

    target_fps: float
    minimum_fps: float

    def __post_init__(self) -> None:
        for field_name, value in (("target_fps", self.target_fps), ("minimum_fps", self.minimum_fps)):
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"PerformancePolicy.{field_name} must be a finite, non-negative number.")
        if self.minimum_fps > self.target_fps:
            raise ValueError(
                f"PerformancePolicy.minimum_fps ({self.minimum_fps}) cannot exceed target_fps ({self.target_fps})."
            )


@dataclass(frozen=True)
class SelectionResult:
    """The outcome of one `select_candidate` call.

    Attributes:
        status: The outcome category.
        selected_candidate: The chosen candidate's `name`, or `None` if
            nothing was selected (`NO_VALID_CANDIDATE`/`INVALID_DATA`).
        reason: A human-readable explanation of the outcome - always
            non-empty, whether a candidate was selected or not.
        evaluated_candidates: Names of every well-formed candidate that
            was actually considered (post-malformed-filtering), in
            input order.
        rejected_candidates: Names of every candidate (malformed or
            well-formed) that was not selected.
    """

    status: SelectionStatus
    selected_candidate: Optional[str]
    reason: str
    evaluated_candidates: Tuple[str, ...]
    rejected_candidates: Tuple[str, ...]


def _malformed_reason(candidate: SelectionCandidate) -> Optional[str]:
    """Returns a description of what's wrong with `candidate`, or `None` if it's valid."""
    if not isinstance(candidate.name, str) or not candidate.name.strip():
        return "name must be a non-empty string"

    for field_name, value in (("average_fps", candidate.average_fps), ("quality_score", candidate.quality_score)):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return f"{field_name} must be a number"
        if not math.isfinite(value):
            return f"{field_name} must be finite"
        if value < 0:
            return f"{field_name} must be >= 0"

    return None


def select_candidate(candidates: Sequence[SelectionCandidate], policy: PerformancePolicy) -> SelectionResult:
    """Deterministically selects the best candidate under `policy`, or explains why none qualifies.

    Pure function: performs no rendering, no file I/O, and no network
    access - it only compares the data it was given. Never raises for
    malformed candidate data; that is reported via
    `SelectionStatus.INVALID_DATA` instead.

    Args:
        candidates: Candidates to choose among, in any order. An empty
            sequence is a valid input (reported as
            `NO_VALID_CANDIDATE`, not an error).
        policy: The thresholds to select against.

    Returns:
        A `SelectionResult` describing the outcome and, on success,
        which candidate was chosen and why.
    """
    if not candidates:
        return SelectionResult(
            status=SelectionStatus.NO_VALID_CANDIDATE,
            selected_candidate=None,
            reason="No candidates were provided.",
            evaluated_candidates=(),
            rejected_candidates=(),
        )

    malformed_names = []
    well_formed = []
    for candidate in candidates:
        reason = _malformed_reason(candidate)
        if reason is not None:
            label = candidate.name if isinstance(candidate.name, str) and candidate.name.strip() else "<unnamed>"
            malformed_names.append(label)
        else:
            well_formed.append(candidate)

    if not well_formed:
        return SelectionResult(
            status=SelectionStatus.INVALID_DATA,
            selected_candidate=None,
            reason=f"All {len(candidates)} candidate(s) had malformed or incomplete data.",
            evaluated_candidates=(),
            rejected_candidates=tuple(malformed_names),
        )

    evaluated_names = tuple(candidate.name for candidate in well_formed)

    benchmarked_ok = [candidate for candidate in well_formed if candidate.benchmark_success]
    if not benchmarked_ok:
        return SelectionResult(
            status=SelectionStatus.NO_VALID_CANDIDATE,
            selected_candidate=None,
            reason="No candidate completed benchmarking successfully.",
            evaluated_candidates=evaluated_names,
            rejected_candidates=evaluated_names + tuple(malformed_names),
        )

    meets_minimum = [c for c in benchmarked_ok if c.average_fps >= policy.minimum_fps]
    if not meets_minimum:
        return SelectionResult(
            status=SelectionStatus.NO_VALID_CANDIDATE,
            selected_candidate=None,
            reason=(
                f"No candidate met the minimum required {policy.minimum_fps:g} FPS "
                f"(best measured: {max(c.average_fps for c in benchmarked_ok):g} FPS)."
            ),
            evaluated_candidates=evaluated_names,
            rejected_candidates=evaluated_names + tuple(malformed_names),
        )

    meets_target = [c for c in meets_minimum if c.average_fps >= policy.target_fps]

    if meets_target:
        best = sorted(meets_target, key=lambda c: (-c.quality_score, -c.average_fps, c.name))[0]
        status = SelectionStatus.MEETS_TARGET
        reason = (
            f"'{best.name}' meets the target {policy.target_fps:g} FPS "
            f"({best.average_fps:g} FPS) and has the highest quality score "
            f"({best.quality_score:g}) among candidates that also meet target."
        )
    else:
        best = sorted(meets_minimum, key=lambda c: (-c.average_fps, -c.quality_score, c.name))[0]
        status = SelectionStatus.BELOW_TARGET
        reason = (
            f"No candidate reached the target {policy.target_fps:g} FPS; "
            f"'{best.name}' is the best available at {best.average_fps:g} FPS "
            f"(meets the minimum {policy.minimum_fps:g} FPS)."
        )

    rejected_names = tuple(name for name in evaluated_names if name != best.name) + tuple(malformed_names)

    return SelectionResult(
        status=status,
        selected_candidate=best.name,
        reason=reason,
        evaluated_candidates=evaluated_names,
        rejected_candidates=rejected_names,
    )
