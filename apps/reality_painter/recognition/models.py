"""Typed, provider-agnostic data contracts for Reality Painter's object-recognition layer.

This module holds only the shape of recognition data - never a
provider, never an HTTP call, never asset-resolution or retrieval
logic. `apps.reality_painter.recognition.provider.RecognitionProvider`
implementations return `RecognitionResult`; the integration controller
in `apps.reality_painter.inspection.controller` is the only consumer
that interprets it further.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class RecognizedObject:
    """One object a recognition provider believes it saw in a drawing.

    Attributes:
        label: The recognized object's name (e.g. "flower"), exactly
            as reported by the provider - never invented or guessed by
            any downstream consumer.
        confidence: A relative score used only for deterministic
            selection among multiple candidates (see
            `apps.reality_painter.inspection.controller.select_object`).
            Not guaranteed to be a calibrated probability - a provider
            that has no real confidence signal is expected to document
            what it puts here (see e.g. a future NVIDIA NIM adapter),
            rather than this module inventing meaning for the field.
        reasoning: Optional free-text explanation from the provider, if
            it gave one. Never fabricated by this module.
    """

    label: str
    confidence: float
    reasoning: Optional[str] = None


@dataclass(frozen=True)
class RecognitionResult:
    """The outcome of one recognition request.

    Attributes:
        succeeded: Whether the provider completed a recognition attempt
            without error. `False` means `objects` should be treated as
            empty/meaningless regardless of its contents.
        objects: Recognized candidates, in the provider's own reported
            order. May be empty even when `succeeded` is True (e.g. the
            provider genuinely found nothing recognizable).
        error: A human-readable error message if `succeeded` is False,
            otherwise `None`.
    """

    succeeded: bool
    objects: List[RecognizedObject] = field(default_factory=list)
    error: Optional[str] = None
