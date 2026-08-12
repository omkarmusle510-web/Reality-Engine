"""Offline optimized-candidate generation for Reality Painter's Asset Optimizer.

Block 2 sits directly on top of Block 1
(`apps.reality_painter.optimization.analyzer`) but does not depend on
it or modify it - this module only *produces* candidate GLB files;
measuring them is `analyze_asset()`'s job, and benchmarking them
against the real renderer is explicitly out of scope for this block
(see module docstring notes below).

Given a source `.glb`/`.gltf` file, `generate_candidates()` produces a
small, explicit set of `CandidateResult`s:

    - ORIGINAL:        an untouched copy of the source (no tool needed).
    - LIGHT:            mild geometry simplification only.
    - HEAVY_OPTIMIZED:  aggressive geometry simplification + compression.
    - COMBINED:         aggressive geometry simplification + texture
                        resizing, combined.

The exact operations each named candidate applies are declared once,
as data, in `DEFAULT_CANDIDATE_SPECS` - there is no per-asset branching
logic and no hidden "one size fits all" behavior. Every candidate is
independently measurable: every `CandidateResult` reports its own
success/failure, the operations actually attempted, and (on success)
the output file size.

Geometry simplification and texture resizing are delegated entirely to
an external, already-established glTF CLI tool (`gltfpack` by default -
see https://github.com/zeux/meshoptimizer) rather than reimplemented
here. This module never decimates a mesh or resizes a texture itself.
If the configured tool cannot be located on the system, every candidate
that requires it fails cleanly with a typed error message - optimization
is never silently faked, and no candidate output is ever fabricated.

This module never converts to KTX2 and never applies Draco compression
- gltfpack's default output remains plain glTF/GLB, which the existing
`engine.scene.loader.load_glb`/Block 1 analyzer can already inspect
unmodified. It also never claims a texture-resize candidate improves
FPS; that claim can only be established later by an actual benchmark
against the real renderer (out of scope here).

Independence, matching Block 1's conventions:
    - Never imports `engine.scene.loader`, `engine.rendering.renderer`,
      `apps.reality_painter.assets.registry.AssetRegistry`, or
      `apps.reality_painter.assets.retriever.AssetRetriever`.
    - Never downloads, retrieves, or discovers assets.
    - Never overwrites or otherwise mutates the source file - every
      candidate is written to a distinct path inside the caller-
      supplied output directory.
    - No runtime/application integration, no FPS benchmarking, no
      hardware detection, no GitHub or cache integration.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

_SUPPORTED_EXTENSIONS = (".glb", ".gltf")
_DEFAULT_TOOL_NAME = "gltfpack"
_DEFAULT_TIMEOUT_SECONDS = 120.0


class CandidateGeneratorError(Exception):
    """Base class for errors raised directly by the candidate generator's public API."""


class SourceAssetNotFoundError(CandidateGeneratorError):
    """Raised when the source file does not exist."""


class UnsupportedAssetFormatError(CandidateGeneratorError):
    """Raised when the source file's extension is not a supported 3D format."""


# --- Candidate specification (data, not branching logic) -----------------


@dataclass(frozen=True)
class CandidateSpec:
    """A named, declarative description of one optimization candidate.

    Purely data: this module never special-cases a candidate by name
    anywhere in its generation logic - `generate_candidate()` treats
    every `CandidateSpec` identically, driven only by its fields.

    Attributes:
        name: Stable candidate identifier (e.g. "LIGHT"). Used to build
            the candidate's output filename and reported verbatim in
            its `CandidateResult`.
        operations: Human-readable operation labels describing what
            this candidate attempts (e.g. `("simplify_light",)`),
            reported verbatim in `CandidateResult.operations_applied`
            regardless of success - so a failed candidate still records
            what it *attempted*.
        requires_tool: Whether generating this candidate needs the
            external glTF CLI tool. `False` only for a plain copy
            (ORIGINAL).
        tool_args: Extra CLI arguments appended after the tool's own
            `-i <source> -o <output>` arguments. Ignored if
            `requires_tool` is `False`.
    """

    name: str
    operations: Tuple[str, ...]
    requires_tool: bool = True
    tool_args: Tuple[str, ...] = ()


# Declared once, as data. Every entry is independently measurable via
# Block 1's `analyze_asset()` once generated - this module makes no
# claim about which candidate is "best," only what was attempted.
DEFAULT_CANDIDATE_SPECS: Tuple[CandidateSpec, ...] = (
    CandidateSpec(name="ORIGINAL", operations=("copy",), requires_tool=False),
    CandidateSpec(
        name="LIGHT",
        operations=("simplify_light",),
        tool_args=("-si", "0.85"),
    ),
    CandidateSpec(
        name="HEAVY_OPTIMIZED",
        operations=("simplify_heavy", "compress"),
        tool_args=("-si", "0.4", "-c"),
    ),
    CandidateSpec(
        name="COMBINED",
        operations=("simplify_heavy", "compress", "texture_resize"),
        tool_args=("-si", "0.4", "-c", "-tc"),
    ),
)


def get_default_candidate_specs() -> Tuple[CandidateSpec, ...]:
    """Returns the default candidate specification set.

    Deterministic: repeated calls return an equal (value-wise) tuple of
    frozen `CandidateSpec` entries, since `DEFAULT_CANDIDATE_SPECS` is
    itself declared once, as static data, and never mutated.
    """
    return DEFAULT_CANDIDATE_SPECS


# --- Result metadata -------------------------------------------------


@dataclass(frozen=True)
class CandidateResult:
    """The outcome of generating one optimization candidate.

    Attributes:
        source_path: The original source file's path, as a string.
        candidate_name: The `CandidateSpec.name` this result is for.
        output_path: Where this candidate's file was (or would have
            been) written, as a string.
        operations_applied: The operations this candidate attempted,
            copied verbatim from its `CandidateSpec` - present even on
            failure, so a failure report still says what was tried.
        success: Whether generation completed and produced a usable
            output file.
        error: A human-readable failure reason, or `None` on success.
        output_file_size_bytes: Size of the generated file in bytes, or
            `None` if generation did not succeed.
    """

    source_path: str
    candidate_name: str
    output_path: str
    operations_applied: Tuple[str, ...]
    success: bool
    error: Optional[str] = None
    output_file_size_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this result."""
        return asdict(self)


# --- Tool resolution ----------------------------------------------------


def _resolve_tool_path(tool_path: Optional[str]) -> Optional[str]:
    """Locates the external glTF CLI tool, or returns `None` if unavailable.

    Never raises: absence of the tool is an expected, reportable
    condition (see `generate_candidate`), not an exceptional one.

    Args:
        tool_path: An explicit path/command name to use instead of the
            default (`gltfpack`). Resolved the same way: via `PATH`
            lookup first, falling back to treating it as a literal
            existing file path.

    Returns:
        A usable path/command string, or `None` if it could not be
        located.
    """
    candidate = tool_path or _DEFAULT_TOOL_NAME
    resolved = shutil.which(candidate)
    if resolved:
        return resolved
    if Path(candidate).is_file():
        return str(candidate)
    return None


# --- Command execution (injectable for testing) ---------------------------

#: A runner executes a fully-built CLI command and returns an object
#: exposing `.returncode` (and, ideally, `.stdout`/`.stderr` for error
#: reporting). Tests inject a fake runner so no real `gltfpack` binary
#: is ever required to exercise this module's logic - the same
#: dependency-injection pattern already used elsewhere in this
#: repository (e.g. `apps.reality_painter.assets.github.discover_assets`'s
#: `session` parameter).
CommandRunner = Callable[[List[str]], Any]


def _default_runner(args: List[str]) -> Any:
    """Runs `args` as a real subprocess. The default `CommandRunner`."""
    return subprocess.run(args, capture_output=True, timeout=_DEFAULT_TIMEOUT_SECONDS)


# --- Generation -----------------------------------------------------------


def generate_candidate(
    spec: CandidateSpec,
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    tool_path: Optional[str] = None,
    runner: Optional[CommandRunner] = None,
) -> CandidateResult:
    """Generates a single optimization candidate.

    Never raises for an expected failure (missing tool, tool exit
    failure, I/O failure) - every such case is reported as a
    `CandidateResult(success=False, error=...)` so a caller generating
    several candidates never has one failure abort the rest (see
    `generate_candidates`). Only a missing/unsupported *source* file is
    treated as a hard precondition failure - see `generate_candidates`,
    which validates that once for the whole batch.

    The source file is never modified or overwritten: the output always
    lands at a distinct path inside `output_dir`, named from
    `spec.name` and the source's own stem/suffix.

    Args:
        spec: The candidate to generate.
        source_path: Path to the source `.glb`/`.gltf` file. Assumed to
            already exist and be a supported format - callers going
            through `generate_candidates` get that validated once,
            up front.
        output_dir: Directory to write the candidate file into. Created
            if it does not already exist.
        tool_path: Optional explicit path/command for the external glTF
            CLI tool, overriding the default (`gltfpack`).
        runner: Optional `CommandRunner` used instead of a real
            subprocess call. Defaults to `_default_runner`.

    Returns:
        A `CandidateResult` describing the outcome.
    """
    source = Path(source_path)
    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / f"{source.stem}__{spec.name.lower()}{source.suffix}"

    if not spec.requires_tool:
        try:
            shutil.copy2(source, output_path)
        except OSError as exc:
            return CandidateResult(
                source_path=str(source),
                candidate_name=spec.name,
                output_path=str(output_path),
                operations_applied=spec.operations,
                success=False,
                error=f"Failed to copy source for candidate {spec.name!r}: {exc}",
            )
        return CandidateResult(
            source_path=str(source),
            candidate_name=spec.name,
            output_path=str(output_path),
            operations_applied=spec.operations,
            success=True,
            output_file_size_bytes=output_path.stat().st_size,
        )

    resolved_tool = _resolve_tool_path(tool_path)
    if resolved_tool is None:
        return CandidateResult(
            source_path=str(source),
            candidate_name=spec.name,
            output_path=str(output_path),
            operations_applied=spec.operations,
            success=False,
            error=(
                f"Required tool {tool_path or _DEFAULT_TOOL_NAME!r} was not found on this system. "
                "Install it (e.g. gltfpack from meshoptimizer) or pass an explicit tool_path. "
                "No optimization was faked or fabricated."
            ),
        )

    command = [resolved_tool, "-i", str(source), "-o", str(output_path), *spec.tool_args]
    active_runner = runner if runner is not None else _default_runner

    try:
        result = active_runner(command)
    except Exception as exc:
        return CandidateResult(
            source_path=str(source),
            candidate_name=spec.name,
            output_path=str(output_path),
            operations_applied=spec.operations,
            success=False,
            error=f"Running {resolved_tool!r} for candidate {spec.name!r} failed: {exc}",
        )

    return_code = getattr(result, "returncode", None)
    if return_code != 0:
        stderr = getattr(result, "stderr", b"")
        detail = stderr.decode("utf-8", "replace") if isinstance(stderr, bytes) else str(stderr)
        return CandidateResult(
            source_path=str(source),
            candidate_name=spec.name,
            output_path=str(output_path),
            operations_applied=spec.operations,
            success=False,
            error=f"{resolved_tool} exited with code {return_code} for candidate {spec.name!r}: {detail[:300]}",
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        return CandidateResult(
            source_path=str(source),
            candidate_name=spec.name,
            output_path=str(output_path),
            operations_applied=spec.operations,
            success=False,
            error=f"{resolved_tool} reported success but produced no usable output for candidate {spec.name!r}.",
        )

    return CandidateResult(
        source_path=str(source),
        candidate_name=spec.name,
        output_path=str(output_path),
        operations_applied=spec.operations,
        success=True,
        output_file_size_bytes=output_path.stat().st_size,
    )


def generate_candidates(
    source_path: Union[str, Path],
    output_dir: Union[str, Path],
    specs: Optional[Tuple[CandidateSpec, ...]] = None,
    tool_path: Optional[str] = None,
    runner: Optional[CommandRunner] = None,
) -> List[CandidateResult]:
    """Generates a batch of optimization candidates for one source file.

    Validates the source file once, up front, so a missing or
    unsupported source fails clearly and immediately rather than
    producing a batch of misleading per-candidate failures. Once that
    precondition passes, each candidate in `specs` is generated
    independently via `generate_candidate` - one candidate failing
    (e.g. the external tool being unavailable) never prevents the
    others from being attempted and reported.

    Args:
        source_path: Path to the source `.glb`/`.gltf` file.
        output_dir: Directory to write candidate files into.
        specs: The candidates to generate. Defaults to
            `DEFAULT_CANDIDATE_SPECS`.
        tool_path: Optional explicit path/command for the external glTF
            CLI tool.
        runner: Optional `CommandRunner` for tool invocation, injected
            in tests instead of a real subprocess call.

    Returns:
        One `CandidateResult` per entry in `specs`, in the same order.

    Raises:
        SourceAssetNotFoundError: If `source_path` does not exist.
        UnsupportedAssetFormatError: If `source_path`'s extension is
            not `.glb`/`.gltf`.
    """
    source = Path(source_path)

    if source.suffix.lower() not in _SUPPORTED_EXTENSIONS:
        raise UnsupportedAssetFormatError(f"Unsupported asset format {source.suffix!r}: {source}.")
    if not source.is_file():
        raise SourceAssetNotFoundError(f"Source asset not found: {source}.")

    active_specs = specs if specs is not None else DEFAULT_CANDIDATE_SPECS
    return [
        generate_candidate(spec, source, output_dir, tool_path=tool_path, runner=runner) for spec in active_specs
    ]
