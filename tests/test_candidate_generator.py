"""Offline, deterministic tests for the Asset Optimizer's Block 2 candidate generator.

No network access, no GitHub, no NVIDIA, no camera, no pyrender/OpenGL,
and no real `gltfpack` binary is required - every tool invocation is
mocked via an injected `runner` callable (the same dependency-
injection pattern already used in `tests/test_github_asset_source.py`
and `tests/test_asset_retriever.py`). Only
`apps.reality_painter.optimization.candidate_generator` is exercised
here; Block 1's analyzer, the runtime application, `AssetRegistry`,
`AssetRetriever`, `load_glb`, and `Renderer3D` are never imported or
touched.
"""
import sys
import tempfile
from pathlib import Path

import trimesh

from apps.reality_painter.optimization.candidate_generator import (
    CandidateResult,
    CandidateSpec,
    SourceAssetNotFoundError,
    UnsupportedAssetFormatError,
    generate_candidate,
    generate_candidates,
    get_default_candidate_specs,
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


class _FakeCompletedProcess:
    """A minimal stand-in for `subprocess.CompletedProcess`."""

    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = b""


class _SucceedingRunner:
    """Fake runner that simulates a successful gltfpack invocation.

    Real gltfpack writes its output file as a side effect of running;
    this fake reproduces that by writing a small deterministic payload
    to the `-o` argument's path before reporting success.
    """

    def __init__(self, payload=b"optimized-glb-bytes"):
        self.payload = payload
        self.calls = []

    def __call__(self, args):
        self.calls.append(args)
        output_path = Path(args[args.index("-o") + 1])
        output_path.write_bytes(self.payload)
        return _FakeCompletedProcess(returncode=0)


class _FailingRunner:
    """Fake runner that simulates the tool running but exiting with failure."""

    def __call__(self, args):
        return _FakeCompletedProcess(returncode=1, stderr=b"simplification failed: bad input")


class _RaisingRunner:
    """Fake runner that simulates the tool invocation itself blowing up."""

    def __call__(self, args):
        raise OSError("could not exec tool")


with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)

    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    source_path = tmp_path / "source.glb"
    box.export(source_path, file_type="glb")
    original_bytes = source_path.read_bytes()

    output_dir = tmp_path / "candidates"

    # A fake "installed" tool path: _resolve_tool_path accepts any
    # existing file as a literal tool path, so this stands in for a
    # real gltfpack binary without requiring one to actually be
    # installed in the test environment. The fake runner (never a real
    # subprocess) is what's actually invoked - this file's contents are
    # irrelevant, only its existence matters for resolution.
    fake_tool_path = tmp_path / "fake_gltfpack"
    fake_tool_path.write_text("#!/bin/sh\n")

    # 1. Candidate specification is deterministic.
    specs_a = get_default_candidate_specs()
    specs_b = get_default_candidate_specs()
    check("default candidate specs are deterministic across calls", specs_a == specs_b)
    check("default candidate specs include ORIGINAL, LIGHT, HEAVY_OPTIMIZED, COMBINED",
          {s.name for s in specs_a} == {"ORIGINAL", "LIGHT", "HEAVY_OPTIMIZED", "COMBINED"})
    check("ORIGINAL spec does not require the external tool",
          next(s for s in specs_a if s.name == "ORIGINAL").requires_tool is False)
    check("LIGHT/HEAVY_OPTIMIZED/COMBINED specs require the external tool",
          all(s.requires_tool for s in specs_a if s.name != "ORIGINAL"))

    # 2. Missing source fails cleanly (typed error, not a crash/traceback).
    expect_raises(
        "missing source raises SourceAssetNotFoundError",
        SourceAssetNotFoundError,
        lambda: generate_candidates(tmp_path / "does_not_exist.glb", output_dir, runner=_SucceedingRunner()),
    )

    expect_raises(
        "unsupported source extension raises UnsupportedAssetFormatError",
        UnsupportedAssetFormatError,
        lambda: generate_candidates(tmp_path / "model.obj", output_dir, runner=_SucceedingRunner()),
    )

    # 3. Tool unavailable fails cleanly (per-candidate, not a crash).
    light_spec = next(s for s in specs_a if s.name == "LIGHT")
    unavailable_result = generate_candidate(
        light_spec, source_path, output_dir, tool_path="/definitely/not/a/real/tool", runner=_SucceedingRunner()
    )
    check("tool-unavailable candidate reports success=False", unavailable_result.success is False)
    check("tool-unavailable candidate carries a clear error message", bool(unavailable_result.error))
    check("tool-unavailable candidate output_file_size_bytes is None", unavailable_result.output_file_size_bytes is None)
    check(
        "tool-unavailable candidate still reports its attempted operations",
        unavailable_result.operations_applied == light_spec.operations,
    )

    # 4. Original source is never overwritten by any candidate path.
    original_spec = next(s for s in specs_a if s.name == "ORIGINAL")
    original_result = generate_candidate(original_spec, source_path, output_dir, runner=_SucceedingRunner())
    check("ORIGINAL candidate output path differs from the source path", original_result.output_path != str(source_path))
    check("source file bytes are unchanged after ORIGINAL candidate generation", source_path.read_bytes() == original_bytes)

    # 5. Successful candidate generation is represented correctly.
    check("ORIGINAL candidate succeeds via plain copy", original_result.success is True)
    check("ORIGINAL candidate reports a positive output size", (original_result.output_file_size_bytes or 0) > 0)
    check("ORIGINAL candidate output file actually exists", Path(original_result.output_path).is_file())
    check("ORIGINAL candidate copied content matches source", Path(original_result.output_path).read_bytes() == original_bytes)

    succeeding_runner = _SucceedingRunner()
    heavy_spec = next(s for s in specs_a if s.name == "HEAVY_OPTIMIZED")
    heavy_result = generate_candidate(heavy_spec, source_path, output_dir, tool_path=str(fake_tool_path), runner=succeeding_runner)
    check("HEAVY_OPTIMIZED candidate succeeds with a working fake tool", heavy_result.success is True)
    check("HEAVY_OPTIMIZED candidate reports correct output size", heavy_result.output_file_size_bytes == len(succeeding_runner.payload))
    check("HEAVY_OPTIMIZED candidate invoked the tool with -i/-o", "-i" in succeeding_runner.calls[0] and "-o" in succeeding_runner.calls[0])
    check(
        "HEAVY_OPTIMIZED candidate passed its declared tool_args",
        all(arg in succeeding_runner.calls[0] for arg in heavy_spec.tool_args),
    )

    # 6. Multiple candidates can be represented (a full batch).
    batch_runner = _SucceedingRunner()
    batch_results = generate_candidates(source_path, output_dir, tool_path=str(fake_tool_path), runner=batch_runner)
    check("generate_candidates returns one result per spec", len(batch_results) == len(specs_a))
    check("every batch result is a CandidateResult", all(isinstance(r, CandidateResult) for r in batch_results))
    check("batch result names match the spec set", {r.candidate_name for r in batch_results} == {s.name for s in specs_a})
    check("every batch candidate succeeded with a working fake tool", all(r.success for r in batch_results))
    check("batch output paths are all distinct", len({r.output_path for r in batch_results}) == len(batch_results))

    # 7. Failed candidate generation does not crash the caller.
    failing_results = generate_candidates(source_path, output_dir, tool_path=str(fake_tool_path), runner=_FailingRunner())
    check("failing tool run does not raise out of generate_candidates", isinstance(failing_results, list))
    non_original_failures = [r for r in failing_results if r.candidate_name != "ORIGINAL"]
    check("tool-exit-failure candidates report success=False", all(r.success is False for r in non_original_failures))
    check("ORIGINAL still succeeds even when the tool would fail (no tool needed)",
          next(r for r in failing_results if r.candidate_name == "ORIGINAL").success is True)
    check("tool-exit-failure error message includes stderr detail",
          all("simplification failed" in (r.error or "") for r in non_original_failures))

    raising_results = generate_candidates(source_path, output_dir, tool_path=str(fake_tool_path), runner=_RaisingRunner())
    check("runner raising an exception does not propagate out of generate_candidates", isinstance(raising_results, list))
    check(
        "runner exception is captured as a failed candidate result",
        all(r.success is False for r in raising_results if r.candidate_name != "ORIGINAL"),
    )

    # 8. Metadata is correct.
    combined_spec = next(s for s in specs_a if s.name == "COMBINED")
    combined_result = generate_candidate(combined_spec, source_path, output_dir, tool_path=str(fake_tool_path), runner=_SucceedingRunner())
    check("metadata.source_path matches input", combined_result.source_path == str(source_path))
    check("metadata.candidate_name matches spec name", combined_result.candidate_name == "COMBINED")
    check("metadata.operations_applied matches spec operations", combined_result.operations_applied == combined_spec.operations)
    as_dict = combined_result.to_dict()
    check("to_dict() returns a plain, JSON-serializable dict", isinstance(as_dict, dict))
    check("to_dict() round-trips candidate_name", as_dict["candidate_name"] == "COMBINED")

print(f"\n{passed} passed, {failed} failed")
sys.exit(1 if failed else 0)
