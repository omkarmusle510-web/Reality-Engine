"""Offline, deterministic tests for the Asset Optimizer's Block 7 decision layer.

No network access, no GitHub, no camera, no pyrender/OpenGL, no
`trimesh`, no real hardware detection, no real benchmarking. Every
`HardwareProfile` and `BenchmarkResult` used here is constructed
directly, in-memory, as already-computed input data - matching this
module's own contract of orchestrating existing results rather than
producing them. Blocks 1, 2, 3, 5, and 6's own detection/measurement
logic are never invoked; only Block 6's `recommended_policy_for_tier`
(a pure lookup) and Block 4's `select_candidate` (the selection
authority Block 7 delegates to) are exercised indirectly through
`decide_optimization`.
"""
import sys

from apps.reality_painter.optimization.benchmark import BenchmarkResult
from apps.reality_painter.optimization.decision import OptimizationDecision, decide_optimization
from apps.reality_painter.optimization.hardware_profile import (
    CPUInfo,
    GPUInfo,
    HardwareProfile,
    MemoryInfo,
    PerformanceTier,
    PlatformInfo,
    recommended_policy_for_tier,
)
from apps.reality_painter.optimization.selector import PerformancePolicy, SelectionStatus

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


def _profile(tier: PerformanceTier) -> HardwareProfile:
    """Builds a fake HardwareProfile with the given tier, no real detection."""
    return HardwareProfile(
        cpu=CPUInfo(logical_cores=8),
        memory=MemoryInfo(total_ram_mb=16 * 1024),
        platform=PlatformInfo(system="Windows", release="11", architecture="AMD64"),
        gpu=GPUInfo(name="Fake GPU", vendor=None),
        tier=tier,
    )


def _bench(name, average_fps, success=True):
    """Builds a fake, already-measured BenchmarkResult - no real render call."""
    return BenchmarkResult(
        candidate_path=f"/fake/{name}.glb",
        candidate_name=name,
        success=success,
        error=None if success else "fake render failure",
        load_time_seconds=0.01,
        average_fps=average_fps,
        average_frame_time_seconds=(1.0 / average_fps) if average_fps > 0 else 0.0,
        measured_frames=30 if success else 0,
        warmup_frames=5,
        render_width=320,
        render_height=240,
    )


# ===========================================================================
# 1. Hardware tier -> policy: HIGH/MEDIUM/LOW/UNKNOWN each resolve to
#    Block 6's own recommended policy for that tier.
# ===========================================================================
for tier in (PerformanceTier.HIGH, PerformanceTier.MEDIUM, PerformanceTier.LOW, PerformanceTier.UNKNOWN):
    profile = _profile(tier)
    decision = decide_optimization(profile, [], {})
    expected = recommended_policy_for_tier(tier)
    check(
        f"{tier.value} tier resolves to Block 6's recommended target/minimum FPS",
        decision.policy.target_fps == expected.target_fps and decision.policy.minimum_fps == expected.minimum_fps,
    )
    check(f"{tier.value} tier is carried through to hardware_tier", decision.hardware_tier == tier)

# ===========================================================================
# 2. Target reached: a candidate meeting the tier's target FPS is
#    selected, status MEETS_TARGET - delegated entirely to Block 4.
# ===========================================================================
medium_profile = _profile(PerformanceTier.MEDIUM)  # target=60, minimum=30
results = [_bench("LIGHT", 123.6), _bench("HEAVY_OPTIMIZED", 46.6)]
quality = {"LIGHT": 0.4, "HEAVY_OPTIMIZED": 0.9}
decision = decide_optimization(medium_profile, results, quality)
check("target-reaching candidate is selected", decision.selected_candidate == "LIGHT")
check("status is MEETS_TARGET when a candidate reaches target", decision.status == SelectionStatus.MEETS_TARGET)
check("OptimizationDecision.selection carries the underlying SelectionResult", decision.selection.selected_candidate == "LIGHT")

# ===========================================================================
# 3. Fallback between minimum and target: no candidate reaches target,
#    but one clears minimum -> BELOW_TARGET, best available selected.
# ===========================================================================
results = [_bench("HEAVY_OPTIMIZED", 40.0)]
quality = {"HEAVY_OPTIMIZED": 0.5}
decision = decide_optimization(medium_profile, results, quality)
check("below-target but above-minimum candidate is still selected", decision.selected_candidate == "HEAVY_OPTIMIZED")
check("status is BELOW_TARGET as a documented fallback", decision.status == SelectionStatus.BELOW_TARGET)

# ===========================================================================
# 4. Below-minimum candidates are never selected, regardless of quality.
# ===========================================================================
results = [_bench("TOO_SLOW", 10.0)]
quality = {"TOO_SLOW": 1.0}
decision = decide_optimization(medium_profile, results, quality)
check("below-minimum-only input -> NO_VALID_CANDIDATE", decision.status == SelectionStatus.NO_VALID_CANDIDATE)
check("below-minimum-only input selects nothing", decision.selected_candidate is None)

# ===========================================================================
# 5. Failed benchmarks are never selected (Block 4's own rule; Block 7
#    never duplicates it, only passes benchmark_success through).
# ===========================================================================
results = [_bench("BROKEN", 200.0, success=False)]
quality = {"BROKEN": 1.0}
decision = decide_optimization(medium_profile, results, quality)
check("failed-benchmark-only input -> NO_VALID_CANDIDATE", decision.status == SelectionStatus.NO_VALID_CANDIDATE)
check("failed candidate is still listed as evaluated (not silently dropped)", "BROKEN" in decision.evaluated_candidates)

# ===========================================================================
# 6. No candidates at all -> NO_VALID_CANDIDATE, never a crash.
# ===========================================================================
decision = decide_optimization(medium_profile, [], {})
check("empty benchmark_results -> NO_VALID_CANDIDATE", decision.status == SelectionStatus.NO_VALID_CANDIDATE)
check("empty benchmark_results -> non-empty reason", bool(decision.reason))
check("empty benchmark_results -> no excluded candidates", decision.excluded_candidates == ())

# ===========================================================================
# 7. Malformed input: a candidate with no supplied quality score is
#    excluded from selection entirely - never assigned a fabricated score.
# ===========================================================================
results = [_bench("LIGHT", 90.0), _bench("UNSCORED", 95.0)]
quality = {"LIGHT": 0.6}  # UNSCORED intentionally has no quality score
decision = decide_optimization(medium_profile, results, quality)
check("candidate missing a quality score is excluded, not fabricated", decision.excluded_candidates == ("UNSCORED",))
check("excluded candidate never appears in evaluated_candidates", "UNSCORED" not in decision.evaluated_candidates)
check("excluded candidate never appears in rejected_candidates", "UNSCORED" not in decision.rejected_candidates)
check("remaining scored candidate is still selected normally", decision.selected_candidate == "LIGHT")

# All candidates unscored -> nothing reaches Block 4 at all.
results = [_bench("A", 90.0), _bench("B", 95.0)]
decision = decide_optimization(medium_profile, results, {})
check("no quality scores supplied -> every candidate excluded", set(decision.excluded_candidates) == {"A", "B"})
check("no quality scores supplied -> NO_VALID_CANDIDATE (nothing to select from)", decision.status == SelectionStatus.NO_VALID_CANDIDATE)
check("no quality scores supplied -> nothing evaluated by Block 4", decision.evaluated_candidates == ())

# ===========================================================================
# 8. Deterministic identical inputs -> identical decisions.
# ===========================================================================
results_a = [_bench("LIGHT", 123.6), _bench("HEAVY_OPTIMIZED", 46.6)]
results_b = [_bench("LIGHT", 123.6), _bench("HEAVY_OPTIMIZED", 46.6)]
quality = {"LIGHT": 0.4, "HEAVY_OPTIMIZED": 0.9}
decision_1 = decide_optimization(medium_profile, results_a, quality)
decision_2 = decide_optimization(medium_profile, results_b, quality)
check("identical inputs produce an identical selected_candidate", decision_1.selected_candidate == decision_2.selected_candidate)
check("identical inputs produce an identical status", decision_1.status == decision_2.status)
check("identical inputs produce an identical reason", decision_1.reason == decision_2.reason)
check("identical inputs produce an identical OptimizationDecision", decision_1 == decision_2)

# ===========================================================================
# 9. policy_override takes precedence over the hardware-tier recommendation.
# ===========================================================================
custom_policy = PerformancePolicy(target_fps=200.0, minimum_fps=150.0)
results = [_bench("LIGHT", 123.6)]
quality = {"LIGHT": 0.4}
decision = decide_optimization(medium_profile, results, quality, policy_override=custom_policy)
check("policy_override replaces the tier-recommended policy", decision.policy == custom_policy)
check("candidate below an overridden minimum is rejected accordingly", decision.status == SelectionStatus.NO_VALID_CANDIDATE)

# Without override, the same input meets MEDIUM's default target instead.
decision_default = decide_optimization(medium_profile, results, quality)
check("without override, the tier's own recommended policy applies", decision_default.status == SelectionStatus.MEETS_TARGET)

# ===========================================================================
# 10. SelectionResult fields are carried through verbatim (no divergence
#     from Block 4's own algorithm/output).
# ===========================================================================
results = [_bench("LIGHT", 123.6), _bench("HEAVY_OPTIMIZED", 46.6)]
quality = {"LIGHT": 0.4, "HEAVY_OPTIMIZED": 0.9}
decision = decide_optimization(medium_profile, results, quality)
check(
    "OptimizationDecision fields mirror the underlying SelectionResult exactly",
    (decision.status, decision.selected_candidate, decision.reason, decision.evaluated_candidates, decision.rejected_candidates)
    == (
        decision.selection.status,
        decision.selection.selected_candidate,
        decision.selection.reason,
        decision.selection.evaluated_candidates,
        decision.selection.rejected_candidates,
    ),
)

# ===========================================================================
# 11. Every decision carries a non-empty, human-readable reason.
# ===========================================================================
for label, test_results, test_quality in [
    ("normal selection", [_bench("LIGHT", 123.6)], {"LIGHT": 0.4}),
    ("empty input", [], {}),
    ("all excluded (no quality scores)", [_bench("A", 90.0)], {}),
    ("all below minimum", [_bench("SLOW", 1.0)], {"SLOW": 0.5}),
]:
    outcome = decide_optimization(medium_profile, test_results, test_quality)
    check(f"reason is non-empty for {label}", bool(outcome.reason and outcome.reason.strip()))

# ===========================================================================
# 12. OptimizationDecision is the expected type, and orchestration never
#     raises for any of the scenarios exercised above.
# ===========================================================================
check("decide_optimization returns an OptimizationDecision", isinstance(decision, OptimizationDecision))

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
