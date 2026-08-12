"""Offline, deterministic tests for the Asset Optimizer's Block 4 selector.

No network access, no GitHub, no camera, no pyrender/OpenGL, no
`trimesh`. Only `apps.reality_painter.optimization.selector` is
exercised here - Blocks 1-3 (`analyzer.py`, `candidate_generator.py`,
`benchmark.py`) are never imported or touched, matching this module's
own independence from them.
"""
import inspect
import sys

from apps.reality_painter.optimization import selector
from apps.reality_painter.optimization.selector import (
    PerformancePolicy,
    SelectionCandidate,
    SelectionStatus,
    select_candidate,
)

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


def expect_raises(name, exception_type, func):
    try:
        func()
        check(name, False)
    except exception_type:
        check(name, True)


DEFAULT_POLICY = PerformancePolicy(target_fps=60.0, minimum_fps=30.0)

# ===========================================================================
# 1. Target FPS selection: a candidate meeting target is selected over one
#    that doesn't, and status is reported as MEETS_TARGET.
# ===========================================================================
candidates = [
    SelectionCandidate(name="light", average_fps=123.6, quality_score=0.4),
    SelectionCandidate(name="heavy", average_fps=46.6, quality_score=0.9),
]
result = select_candidate(candidates, DEFAULT_POLICY)
check("target-FPS selection picks a candidate meeting target", result.selected_candidate == "light")
check("target-FPS selection reports MEETS_TARGET", result.status == SelectionStatus.MEETS_TARGET)
check("MEETS_TARGET result names the runner-up as rejected", "heavy" in result.rejected_candidates)

# ===========================================================================
# 2. Minimum FPS rejection: a below-minimum candidate is never selected,
#    even if it has the best quality score.
# ===========================================================================
candidates = [
    SelectionCandidate(name="too_slow", average_fps=10.0, quality_score=1.0),
    SelectionCandidate(name="acceptable", average_fps=35.0, quality_score=0.2),
]
result = select_candidate(candidates, DEFAULT_POLICY)
check("below-minimum candidate is never selected despite high quality", result.selected_candidate == "acceptable")
check("below-minimum candidate is excluded even from evaluation influence", "too_slow" in result.rejected_candidates)

# All candidates below minimum -> NO_VALID_CANDIDATE, nothing selected.
all_too_slow = [
    SelectionCandidate(name="a", average_fps=5.0, quality_score=0.9),
    SelectionCandidate(name="b", average_fps=12.0, quality_score=0.1),
]
result = select_candidate(all_too_slow, DEFAULT_POLICY)
check("all-below-minimum -> NO_VALID_CANDIDATE", result.status == SelectionStatus.NO_VALID_CANDIDATE)
check("all-below-minimum -> nothing selected", result.selected_candidate is None)

# ===========================================================================
# 3. Quality vs performance tradeoff: among candidates that BOTH meet
#    target FPS, the higher-quality one wins even though it is slower.
# ===========================================================================
candidates = [
    SelectionCandidate(name="fast_low_quality", average_fps=200.0, quality_score=0.3),
    SelectionCandidate(name="fast_high_quality", average_fps=75.0, quality_score=0.95),
]
result = select_candidate(candidates, DEFAULT_POLICY)
check("selector is quality-aware, not fastest-wins", result.selected_candidate == "fast_high_quality")
check("quality-aware selection still reports MEETS_TARGET", result.status == SelectionStatus.MEETS_TARGET)

# ===========================================================================
# 4. Deterministic tie handling: equal FPS and quality resolve by name.
# ===========================================================================
tied = [
    SelectionCandidate(name="zeta", average_fps=90.0, quality_score=0.5),
    SelectionCandidate(name="alpha", average_fps=90.0, quality_score=0.5),
]
result_1 = select_candidate(tied, DEFAULT_POLICY)
result_2 = select_candidate(list(reversed(tied)), DEFAULT_POLICY)
check("exact ties resolve to the alphabetically-first name", result_1.selected_candidate == "alpha")
check("tie resolution is independent of input order", result_1.selected_candidate == result_2.selected_candidate)

# ===========================================================================
# 5. No valid candidates: empty input, and a failed-benchmark-only input.
# ===========================================================================
result = select_candidate([], DEFAULT_POLICY)
check("empty candidate list -> NO_VALID_CANDIDATE", result.status == SelectionStatus.NO_VALID_CANDIDATE)
check("empty candidate list -> no reason omitted", bool(result.reason))

failed_only = [SelectionCandidate(name="broken", average_fps=100.0, quality_score=1.0, benchmark_success=False)]
result = select_candidate(failed_only, DEFAULT_POLICY)
check("benchmark_success=False candidate is never selected", result.status == SelectionStatus.NO_VALID_CANDIDATE)
check("failed-benchmark candidate is still listed as evaluated", "broken" in result.evaluated_candidates)

# ===========================================================================
# 6. Malformed/incomplete candidate data.
# ===========================================================================
malformed_all = [
    SelectionCandidate(name="", average_fps=90.0, quality_score=0.5),
    SelectionCandidate(name="neg_fps", average_fps=-5.0, quality_score=0.5),
]
result = select_candidate(malformed_all, DEFAULT_POLICY)
check("all-malformed candidates -> INVALID_DATA", result.status == SelectionStatus.INVALID_DATA)
check("all-malformed candidates -> nothing selected", result.selected_candidate is None)
check("all-malformed reason mentions malformed data", "malformed" in result.reason.lower())

mixed = [
    SelectionCandidate(name="nan_quality", average_fps=90.0, quality_score=float("nan")),
    SelectionCandidate(name="good", average_fps=90.0, quality_score=0.5),
]
result = select_candidate(mixed, DEFAULT_POLICY)
check("one malformed among valid candidates is filtered, not fatal", result.status == SelectionStatus.MEETS_TARGET)
check("filtered malformed candidate is not the selection", result.selected_candidate == "good")
check("malformed candidate appears in rejected_candidates", "nan_quality" in result.rejected_candidates)

# PerformancePolicy itself rejects an inconsistent configuration.
expect_raises(
    "PerformancePolicy rejects minimum_fps > target_fps",
    ValueError,
    lambda: PerformancePolicy(target_fps=30.0, minimum_fps=60.0),
)
expect_raises(
    "PerformancePolicy rejects a negative target_fps",
    ValueError,
    lambda: PerformancePolicy(target_fps=-1.0, minimum_fps=0.0),
)

# ===========================================================================
# 7. SelectionResult always carries a non-empty reason.
# ===========================================================================
for test_candidates, test_policy, label in [
    (candidates, DEFAULT_POLICY, "successful selection"),
    ([], DEFAULT_POLICY, "empty input"),
    (all_too_slow, DEFAULT_POLICY, "below-minimum"),
    (malformed_all, DEFAULT_POLICY, "malformed input"),
]:
    outcome = select_candidate(test_candidates, test_policy)
    check(f"reason is non-empty for {label}", bool(outcome.reason and outcome.reason.strip()))

# ===========================================================================
# 8. Selector performs no rendering/network/file I/O - verified by source
#    inspection (no forbidden imports) and by confirming pure, in-memory
#    computation (repeated calls are side-effect-free and idempotent).
# ===========================================================================
source = inspect.getsource(selector)
forbidden_tokens = ("import requests", "import trimesh", "Renderer3D", "open(", "socket", "subprocess")
check(
    "selector module contains no network/render/file-I/O imports",
    not any(token in source for token in forbidden_tokens),
)

repeat_a = select_candidate(candidates, DEFAULT_POLICY)
repeat_b = select_candidate(candidates, DEFAULT_POLICY)
check("select_candidate is idempotent/side-effect-free across repeated calls", repeat_a == repeat_b)

# ===========================================================================
# 9. Boundary conditions: FPS exactly at target/minimum thresholds.
# ===========================================================================
exact_target = [SelectionCandidate(name="exact", average_fps=60.0, quality_score=0.5)]
result = select_candidate(exact_target, DEFAULT_POLICY)
check("FPS exactly equal to target_fps counts as meeting target", result.status == SelectionStatus.MEETS_TARGET)

exact_minimum = [SelectionCandidate(name="exact_min", average_fps=30.0, quality_score=0.5)]
result = select_candidate(exact_minimum, DEFAULT_POLICY)
check("FPS exactly equal to minimum_fps is accepted (not rejected)", result.status == SelectionStatus.BELOW_TARGET)
check("boundary-minimum candidate is the one selected", result.selected_candidate == "exact_min")

just_below_minimum = [SelectionCandidate(name="just_under", average_fps=29.999, quality_score=0.5)]
result = select_candidate(just_below_minimum, DEFAULT_POLICY)
check("FPS just below minimum_fps is rejected", result.status == SelectionStatus.NO_VALID_CANDIDATE)

# target_fps == minimum_fps is a legal (degenerate) policy: meeting the
# floor always also means meeting target.
equal_policy = PerformancePolicy(target_fps=50.0, minimum_fps=50.0)
result = select_candidate([SelectionCandidate(name="c", average_fps=50.0, quality_score=0.1)], equal_policy)
check("degenerate policy (target == minimum) at the boundary meets target", result.status == SelectionStatus.MEETS_TARGET)

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
