"""Prompt construction for Reality Painter's AI subsystem.

`PromptBuilder` has exactly one responsibility: turning user intent -
plus whatever optional context happens to be available - into a single,
finished prompt string. It satisfies the structural `PromptBuilder`
Protocol already declared in `apps.reality_painter.ai.manager`
(`build(capability, user_input, sketch_analysis, context) -> str`), so
it plugs into `AIManager` unmodified.

This module never talks to a provider, never calls an API, never
analyzes a sketch, and never knows Gemini/Groq/OpenAI/Meshy exist - it
only composes text. Sketch analysis is expected to already be a plain
`Dict[str, Any]` (as produced by an `AIManager`-configured
`SketchAnalyzer`) by the time it reaches `build()`.

Prompts are assembled from small, independent sections - one per
concern (capability, user intent, sketch, canvas, tool, drawing
context, style, quality, safety, metadata) - each built by its own
method and composed in a fixed order. Every section is optional: a
missing piece of context simply omits that section rather than
breaking generation. New sections are added by registering an
additional section builder via `register_section()`, never by editing
existing section logic - see the module docstring's extensibility
goals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from apps.reality_painter.ai.manager import AICapability

# A single section builder: given the same inputs `build()` receives,
# returns either the section's text or `None`/empty string if this
# section has nothing to contribute for this request. Signature is
# identical to `PromptBuilder.build()`'s own arguments (minus
# `capability` duplication) so both built-in and custom sections are
# interchangeable.
SectionBuilder = Callable[
    [AICapability, Optional[str], Optional[Dict[str, Any]], Dict[str, Any]],
    Optional[str],
]


@dataclass
class PromptBuilderConfig:
    """Static, reusable defaults for a `PromptBuilder` instance.

    Anything here can still be overridden per-request via `context`
    (see individual `_build_*_section` docstrings) - these are only the
    fallback values used when a request's `context` doesn't supply its
    own.

    Attributes:
        default_style_modifiers: Fallback style text (e.g. "digital
            painting, vibrant colors") used when a request's `context`
            has no `"style"` entry. `None` means no default style.
        default_quality_modifiers: Fallback quality text (e.g. "highly
            detailed, sharp focus") used when a request's `context` has
            no `"quality"` entry. `None` means no default quality text.
        default_safety_instructions: Fallback safety/content-policy
            text appended to every prompt unless a request's `context`
            supplies its own `"safety_instructions"`. `None` means no
            default safety text.
        include_capability_hint: Whether to prefix the prompt with a
            short line naming the requested capability (e.g. "Task:
            image_generation"). Some providers benefit from this
            framing; others ignore it harmlessly.
        section_separator: String used to join non-empty sections into
            the final prompt.
    """

    default_style_modifiers: Optional[str] = None
    default_quality_modifiers: Optional[str] = None
    default_safety_instructions: Optional[str] = None
    include_capability_hint: bool = True
    section_separator: str = "\n\n"


class PromptBuilder:
    """Composes a finished prompt string from optional, independent inputs.

    Satisfies `apps.reality_painter.ai.manager.PromptBuilder`
    structurally, so any instance can be passed directly to
    `AIManager(prompt_builder=...)`.

    Sections run in registration order, each contributing zero or one
    piece of text; empty contributions are dropped, and the survivors
    are joined with `config.section_separator`. The default section
    order is: capability hint, user prompt, sketch analysis, canvas
    context, tool context, drawing context, style modifiers, quality
    modifiers, application metadata, safety instructions.

    Extending the builder with a new section (lighting, camera,
    materials, art style, negative prompt, scene description, 3D
    generation context, object constraints, reference images, ...)
    never requires editing an existing section method - call
    `register_section()` instead.
    """

    def __init__(
        self,
        config: Optional[PromptBuilderConfig] = None,
        extra_sections: Optional[List[SectionBuilder]] = None,
    ) -> None:
        """Creates a prompt builder.

        Args:
            config: Static defaults for this builder. Defaults to
                `PromptBuilderConfig()` (no defaults, capability hint
                on) if omitted.
            extra_sections: Additional section builders appended after
                the built-in sections, in the given order. Equivalent
                to calling `register_section()` for each, after
                construction.
        """
        self._config = config if config is not None else PromptBuilderConfig()
        self._sections: List[SectionBuilder] = [
            self._build_capability_section,
            self._build_user_prompt_section,
            self._build_sketch_section,
            self._build_canvas_section,
            self._build_tool_context_section,
            self._build_drawing_context_section,
            self._build_style_section,
            self._build_quality_section,
            self._build_metadata_section,
            self._build_safety_section,
        ]
        if extra_sections:
            for section in extra_sections:
                self.register_section(section)

    # --- Extensibility --------------------------------------------------

    def register_section(self, section: SectionBuilder, position: Optional[int] = None) -> None:
        """Adds a section builder without touching any existing section.

        This is the sanctioned way to extend `PromptBuilder` - a future
        section (lighting, camera, negative prompt, reference images,
        ...) is added by writing one function matching `SectionBuilder`
        and registering it, never by editing `build()` or an existing
        `_build_*_section` method.

        Args:
            section: A callable matching the `SectionBuilder` signature.
                Called with `(capability, user_input, sketch_analysis,
                context)` on every `build()` and expected to return the
                section's text, or `None`/empty string to contribute
                nothing.
            position: Index to insert at. Appended to the end (i.e. runs
                last) if omitted.
        """
        if position is None:
            self._sections.append(section)
        else:
            self._sections.insert(position, section)

    # --- Protocol entry point -------------------------------------------

    def build(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> str:
        """Builds the finished prompt text for one request.

        Deterministic: the same inputs always produce the same output,
        since sections run in a fixed order and none carry state across
        calls. Missing inputs (no `user_input`, no `sketch_analysis`,
        an empty `context`) never raise - the corresponding sections
        simply contribute nothing.

        Args:
            capability: The kind of work being requested.
            user_input: Raw user-provided text, if any.
            sketch_analysis: Structured sketch data, if any.
            context: Arbitrary caller-supplied context. See each
                `_build_*_section` method for which keys it reads.

        Returns:
            The finished prompt string. May be empty if every section
            contributed nothing (e.g. no user input, no context, and no
            configured defaults).
        """
        pieces: List[str] = []
        for section in self._sections:
            text = section(capability, user_input, sketch_analysis, context)
            if text:
                pieces.append(text.strip())
        return self._config.section_separator.join(pieces)

    # --- Built-in sections ------------------------------------------------
    #
    # Each method below builds exactly one section and returns `None`
    # (never raises) when it has nothing to contribute. None reads or
    # mutates instance state beyond `self._config`, so they remain easy
    # to unit test in isolation - call any one directly with arbitrary
    # arguments and check its return value.

    def _build_capability_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Short line naming the requested capability, if enabled.

        Controlled by `config.include_capability_hint`. Independent of
        `context`, since the capability is always known.
        """
        if not self._config.include_capability_hint:
            return None
        return f"Task: {capability.value}"

    def _build_user_prompt_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """The user's own prompt text, verbatim, if any was given."""
        if not user_input:
            return None
        return user_input.strip()

    def _build_sketch_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Describes the analyzed sketch, if `sketch_analysis` was provided.

        `sketch_analysis` is treated as an opaque mapping - this method
        never assumes a specific schema beyond "a dict of describable
        key/value pairs" so it stays compatible with any future
        `SketchAnalyzer` implementation.
        """
        if not sketch_analysis:
            return None
        details = self._format_mapping(sketch_analysis)
        if not details:
            return None
        return f"Sketch context: {details}"

    def _build_canvas_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Describes canvas context, read from `context["canvas"]`.

        Expected to be a mapping (e.g. `{"width": 1920, "height":
        1080}`), a plain string, or absent entirely.
        """
        return self._format_context_entry(context, "canvas", "Canvas context")

    def _build_tool_context_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Describes the active tool, read from `context["tool"]`.

        Expected to be a mapping (e.g. `{"name": "brush", "size": 12}`),
        a plain string, or absent entirely.
        """
        return self._format_context_entry(context, "tool", "Tool context")

    def _build_drawing_context_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Describes general drawing context, read from `context["drawing"]`.

        Distinct from `"canvas"`/`"tool"`: intended for freeform,
        higher-level context about what's being drawn (e.g. subject,
        composition notes) rather than canvas geometry or tool state.
        """
        return self._format_context_entry(context, "drawing", "Drawing context")

    def _build_style_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Style modifiers, from `context["style"]` or the configured default.

        A per-request `context["style"]` entry always takes precedence
        over `config.default_style_modifiers`.
        """
        style = context.get("style") or self._config.default_style_modifiers
        if not style:
            return None
        return f"Style: {self._format_value(style)}"

    def _build_quality_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Quality modifiers, from `context["quality"]` or the configured default.

        A per-request `context["quality"]` entry always takes
        precedence over `config.default_quality_modifiers`.
        """
        quality = context.get("quality") or self._config.default_quality_modifiers
        if not quality:
            return None
        return f"Quality: {self._format_value(quality)}"

    def _build_metadata_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Forward-compatible application metadata, from `context["metadata"]`.

        A deliberately generic, opaque pass-through - the same pattern
        `overlay.py` already uses for `brush_type_name`/`shape_type` -
        so a future application-specific value can flow into the
        prompt without this module needing to know it exists ahead of
        time.
        """
        return self._format_context_entry(context, "metadata", "Additional context")

    def _build_safety_section(
        self,
        capability: AICapability,
        user_input: Optional[str],
        sketch_analysis: Optional[Dict[str, Any]],
        context: Dict[str, Any],
    ) -> Optional[str]:
        """Safety/content-policy instructions, appended last.

        A per-request `context["safety_instructions"]` entry always
        takes precedence over `config.default_safety_instructions`.
        Placed last so it reads as a final constraint rather than being
        buried among descriptive sections.
        """
        safety = context.get("safety_instructions") or self._config.default_safety_instructions
        if not safety:
            return None
        return self._format_value(safety)

    # --- Formatting helpers -----------------------------------------------

    def _build_context_entry_text(self, key: str, label: str, context: Dict[str, Any]) -> Optional[str]:
        """Deprecated alias retained for internal consistency; unused directly."""
        return self._format_context_entry(context, key, label)

    def _format_context_entry(self, context: Dict[str, Any], key: str, label: str) -> Optional[str]:
        """Formats `context[key]` as a labeled section, if present.

        Args:
            context: The request's context mapping.
            key: The key to read from `context`.
            label: Human-readable label prefixed to the formatted value.

        Returns:
            `"{label}: {formatted value}"`, or `None` if `key` is
            absent or its value is empty.
        """
        value = context.get(key)
        if not value:
            return None
        formatted = self._format_value(value)
        if not formatted:
            return None
        return f"{label}: {formatted}"

    def _format_value(self, value: Any) -> str:
        """Formats an arbitrary section value as text.

        Mappings are formatted as `key: value` pairs (see
        `_format_mapping`); anything else is converted via `str()` and
        stripped. Centralizing this keeps every section's formatting
        consistent without duplicating the mapping-vs-scalar branch in
        each `_build_*_section` method.
        """
        if isinstance(value, dict):
            return self._format_mapping(value)
        return str(value).strip()

    def _format_mapping(self, mapping: Dict[str, Any]) -> str:
        """Formats a mapping as a comma-separated `key: value` list.

        Empty values (`None`, `""`, empty containers) are skipped so a
        sparsely populated mapping (e.g. a sketch analysis with only
        some fields set) never produces dangling `key: ` fragments.

        Args:
            mapping: The mapping to format.

        Returns:
            A comma-separated string, or an empty string if `mapping`
            has no non-empty values.
        """
        parts = [f"{key}: {value}" for key, value in mapping.items() if value not in (None, "", [], {})]
        return ", ".join(parts)