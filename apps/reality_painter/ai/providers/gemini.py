"""Gemini provider adapter for Reality Painter's AI subsystem.

`GeminiProvider` is a concrete `AIProviderBase` implementation that
translates already-prepared, provider-agnostic `AIRequest`s into calls
against Google's Gemini API via the modern `google-genai` SDK
(`from google import genai`), and translates results back into the
repository's `AIResponse` model.

This is a TRUE image-to-image integration: when an `AIRequest` carries
`canvas_image` (raw sketch pixels, populated by `AIManager.generate()`
from the caller's `sketch` argument - see `apps.reality_painter.ai.models`),
that image is encoded and sent to Gemini alongside the generated text
prompt as multimodal input, and the model's image output is decoded
back out of the response - never approximated as text-only.

SDK NOTE: this module now depends on the `google-genai` package (see
`requirements.txt`), not the deprecated `google-generativeai` package.
Import is deferred to construction time (see `__init__`), so importing
this module never fails just because `google-genai` isn't installed
yet - only constructing a `GeminiProvider` does.

MODEL NOTE: the default model is `gemini-2.5-flash-image`, a stable
image-generation model that accepts image + text input and returns
image output via `inline_data` parts. `gemini-1.5-flash` is a
text-only legacy model and is never used here.

This module builds no prompts (that remains `PromptBuilder`'s job),
selects no providers (that remains `AIManager`'s job), and performs no
caching or history recording. It only executes a request it is given
and reports the outcome.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Tuple

import numpy as np

from apps.reality_painter.ai.models import (
    AICapability,
    AIRequest,
    AIResponse,
    ProviderHealth,
    ProviderHealthStatus,
)
from apps.reality_painter.ai.provider import AIProviderBase
from engine.core.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_PROVIDER_NAME = "gemini"
_DEFAULT_IMAGE_MODEL = "gemini-2.5-flash-image"
_DEFAULT_ENCODE_EXTENSION = ".png"
_DEFAULT_ENCODE_MIME_TYPE = "image/png"


class GeminiProvider(AIProviderBase):
    """Adapts Google's Gemini image-generation API to Reality Painter's `AIProvider` contract.

    Configuration (API key, model name, declared capabilities) is
    entirely caller-supplied at construction - this class never
    hard-codes a key, never reads one from a hard-coded location, and
    never assumes which capabilities a given Gemini model/tier
    actually supports.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = _DEFAULT_IMAGE_MODEL,
        capabilities: FrozenSet[AICapability] = frozenset(
            {AICapability.IMAGE_GENERATION, AICapability.IMAGE_EDIT}
        ),
        name: str = _DEFAULT_PROVIDER_NAME,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a Gemini provider bound to one image-generation model and API key.

        Args:
            api_key: Gemini API key. Supplied by the caller (e.g. read
                from an environment variable upstream of this class) -
                never hard-coded or logged here.
            model_name: The Gemini model to call. Defaults to
                `gemini-2.5-flash-image`, a stable image-generation
                model supporting image + text input and image output.
            capabilities: The `AICapability` values this configured
                model/account combination actually supports. Defaults
                to both `IMAGE_GENERATION` and `IMAGE_EDIT`, since a
                single `generate()` call here services both: with a
                `canvas_image` present the request behaves as an edit
                of that image; without one, as fresh generation from
                the prompt alone.
            name: Registration name used with
                `AIManager.register_provider`. Defaults to "gemini".
            generation_config: Optional extra `types.GenerateContentConfig`
                keyword overrides (e.g. `temperature`), merged into the
                image-output config this provider always sets.
        """
        super().__init__(name=name, capabilities=capabilities)

        # Imported here, not at module level, so importing this module
        # never fails just because `google-genai` isn't installed yet -
        # only constructing a `GeminiProvider` does.
        from google import genai
        from google.genai import types

        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._generation_config = generation_config or {}

    # --- Core generation ---------------------------------------------

    def generate(self, request: AIRequest) -> AIResponse:
        """Executes `request` against the configured Gemini image model.

        Sends `request.prompt` (already built upstream by
        `PromptBuilder`) and, if present, `request.canvas_image` (the
        raw sketch pixels `AIManager` attached to the request) as
        multimodal input to `generate_content`, requesting image
        output. Never inspects or rebuilds the prompt itself, and never
        approximates the image as a text description - the actual
        pixels are encoded and sent.

        Args:
            request: The provider-agnostic request. `request.capability`
                is guaranteed by `AIManager` to be one this provider
                was selected for.

        Returns:
            A successful `AIResponse` with
            `data={"image_bytes": ..., "mime_type": ..., "model": ...}`,
            or a failed `AIResponse` with `error` set for any expected
            provider/API failure or missing input.
        """
        contents: List[Any] = []
        if request.prompt:
            contents.append(request.prompt)

        image_part = self._encode_image(request.canvas_image)
        if image_part is not None:
            contents.append(image_part)

        if not contents:
            return self._failure(request.request_id, "Empty request: no prompt or canvas image to send to Gemini.")

        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    **self._generation_config,
                ),
            )
        except Exception as exc:  # Expected provider/API failure surface.
            return self._failure(request.request_id, self._describe_error(exc))

        image_bytes, mime_type = self._extract_image(response)
        if not image_bytes:
            return self._failure(request.request_id, "Gemini returned no image data.")

        return AIResponse(
            request_id=request.request_id,
            success=True,
            provider_name=self.name,
            data={
                "image_bytes": image_bytes,
                "mime_type": mime_type or _DEFAULT_ENCODE_MIME_TYPE,
                "model": self._model_name,
            },
        )

    # --- Optional capability hooks -----------------------------------

    def analyze_sketch(self, sketch: Any) -> Dict[str, Any]:
        """Describes a sketch image using Gemini's multimodal input support.

        `sketch` is expected to be a `numpy.ndarray` (BGR, e.g. from
        `Canvas.export_snapshot()`) or an object with an `.image`
        attribute that is one. This is a provider-backed alternative
        to the local `SketchAnalyzer` - it performs an actual Gemini
        call and is not free the way local OpenCV analysis is.

        Args:
            sketch: Opaque image data understood by `_encode_image`.

        Returns:
            A dict with a single `"description"` key holding Gemini's
            text description of the sketch.

        Raises:
            RuntimeError: If the Gemini call fails.
        """
        image_part = self._encode_image(sketch)
        contents: List[Any] = ["Describe the shapes, objects, and composition in this sketch."]
        if image_part is not None:
            contents.append(image_part)

        try:
            response = self._client.models.generate_content(model=self._model_name, contents=contents)
        except Exception as exc:
            raise RuntimeError(self._describe_error(exc)) from exc

        text = self._extract_text(response)
        return {"description": text} if text else {}

    def optimize_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Asks Gemini to rewrite `prompt` for better generation quality.

        Args:
            prompt: The prompt text to optimize.
            context: Optional caller-supplied context, included as
                plain text guidance if present.

        Returns:
            The optimized prompt text, or the original `prompt`
            unchanged if Gemini's response was empty/unusable.

        Raises:
            RuntimeError: If the Gemini call fails.
        """
        instruction = (
            "Rewrite the following image-generation prompt to be more vivid, "
            "specific, and effective, without changing its intent. "
            "Return only the rewritten prompt.\n\n"
            f"Prompt: {prompt}"
        )
        if context:
            instruction += f"\n\nAdditional context: {context}"

        try:
            response = self._client.models.generate_content(model=self._model_name, contents=instruction)
        except Exception as exc:
            raise RuntimeError(self._describe_error(exc)) from exc

        text = self._extract_text(response)
        return text.strip() if text else prompt

    # --- Health -----------------------------------------------------------

    def check_health(self) -> ProviderHealth:
        """Performs a lightweight reachability/auth check against Gemini.

        Lists available models as a minimal, low-cost call that
        exercises authentication without performing a real generation.
        """
        try:
            list(self._client.models.list())
        except Exception as exc:
            return ProviderHealth(status=ProviderHealthStatus.UNAVAILABLE, message=self._describe_error(exc))
        return ProviderHealth(status=ProviderHealthStatus.HEALTHY)

    # --- Internal helpers -------------------------------------------------

    def _encode_image(self, image_source: Any) -> Optional[Any]:
        """Encodes opaque image data into a Gemini `types.Part`, or `None`.

        Accepts a raw `numpy.ndarray` (BGR, as produced by
        `Canvas.export_snapshot()`), or an object exposing an `.image`
        attribute that is one (e.g. `Frame`). PNG is used as the wire
        format - lossless, so strokes/edges stay crisp for the model.

        Args:
            image_source: The opaque value from `AIRequest.canvas_image`
                (or a direct `analyze_sketch` argument).

        Returns:
            A `types.Part` ready to include in `contents`, or `None` if
            `image_source` couldn't be resolved to a usable image.
        """
        if image_source is None:
            return None

        image_array = image_source
        if not isinstance(image_array, np.ndarray):
            image_array = getattr(image_source, "image", None)

        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            return None

        import cv2

        success, encoded = cv2.imencode(_DEFAULT_ENCODE_EXTENSION, image_array)
        if not success:
            logger.warning("Failed to encode canvas image for Gemini request.")
            return None

        return self._types.Part.from_bytes(data=encoded.tobytes(), mime_type=_DEFAULT_ENCODE_MIME_TYPE)

    def _extract_image(self, response: Any) -> Tuple[Optional[bytes], Optional[str]]:
        """Extracts the first inline image (`inline_data`) from a Gemini response.

        Args:
            response: The SDK's `GenerateContentResponse`.

        Returns:
            A `(image_bytes, mime_type)` tuple. Both are `None` if no
            candidate contained inline image data (e.g. the model
            replied with text only, or generation was blocked).
        """
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                data = getattr(inline_data, "data", None) if inline_data is not None else None
                if data:
                    return data, getattr(inline_data, "mime_type", None)
        return None, None

    def _extract_text(self, response: Any) -> Optional[str]:
        """Safely extracts text from a Gemini `GenerateContentResponse`.

        Guards against `response.text` raising when a response has no
        valid text candidate (e.g. an image-only response, or blocked
        by safety filters) - that case is treated as "no text" rather
        than propagating the SDK's exception past this adapter.
        """
        try:
            text = response.text
        except Exception:
            return None
        return text if text else None

    def _failure(self, request_id: str, message: str) -> AIResponse:
        """Builds a failed `AIResponse` tagged with this provider's name."""
        logger.warning("Gemini provider '%s' request %s failed: %s", self.name, request_id, message)
        return AIResponse(request_id=request_id, success=False, provider_name=self.name, error=message)

    def _describe_error(self, exc: Exception) -> str:
        """Formats an exception for `AIResponse.error` without leaking secrets.

        Never includes the API key or full request/auth headers - only
        the exception type and message, which Gemini's SDK does not
        populate with credential material.
        """
        return f"{type(exc).__name__}: {exc}"