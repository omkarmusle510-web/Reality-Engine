"""Recognition provider contract for Reality Painter's object-recognition layer.

Defines the interface any recognition backend (NVIDIA NIM, or any
future vision model) must implement to plug into
`apps.reality_painter.inspection.controller.InspectionController`.
Structural (`Protocol`), not an abstract base class - matching the
existing convention already used by
`apps.reality_painter.ai.manager.AIProvider`/`PromptBuilder`/
`SketchAnalyzer`: a concrete provider satisfies this simply by having
the right method, with no inheritance and no import-time dependency on
this module.

No concrete provider is implemented here - this module contains no
HTTP calls, no API keys, and no model-specific logic of any kind.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from apps.reality_painter.recognition.models import RecognitionResult


@runtime_checkable
class RecognitionProvider(Protocol):
    """A single recognition backend capable of classifying a drawing."""

    def recognize(self, image: Any) -> RecognitionResult:
        """Classifies `image` and returns a structured result.

        Args:
            image: Opaque drawing/canvas data (e.g. a `numpy.ndarray`,
                or already-encoded image bytes). Never inspected or
                interpreted by any caller of this protocol - only the
                concrete provider needs to know its shape.

        Returns:
            A `RecognitionResult`. A provider should catch its own
            expected failures (network, auth, malformed response, ...)
            internally and report them via
            `RecognitionResult(succeeded=False, error=...)` rather than
            raising - matching the "no uncaught exception" contract
            `InspectionController` relies on - though the controller
            also guards against a provider that raises anyway.
        """
        ...
