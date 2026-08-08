"""Gemini provider adapter for Reality Painter's AI subsystem.

`GeminiProvider` is a concrete `AIProviderBase` implementation that
translates already-prepared, provider-agnostic `AIRequest`s into calls
against Google's Gemini API via the `google-generativeai` SDK, and
translates results back into the repository's `AIResponse` model.

NOTE ON DEPENDENCIES: this repository's `requirements.txt` does not
currently list `google-generativeai`. This module is written against
that package's stable, documented entry point
(`google.generativeai.GenerativeModel.generate_content`) rather than
any experimental/preview API, since no installed version is pinned in
the repo to verify against. Add `google-generativeai` to
`requirements.txt` before this module can actually be imported/run.

This module builds no prompts (that remains `PromptBuilder`'s job),
selects no providers (that remains `AIManager`'s job), and performs no
caching or history recording. It only executes a request it is given
and reports the outcome.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional

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


class GeminiProvider(AIProviderBase):
    """Adapts Google's Gemini API to Reality Painter's `AIProvider` contract.

    Configuration (API key, model name, declared capabilities) is
    entirely caller-supplied at construction - this class never
    hard-codes a key, never reads one from a hard-coded location, and
    never assumes which capabilities a given Gemini model/tier
    actually supports. The caller declares `capabilities` based on the
    model and account they've configured; `AIProviderBase.supports()`
    (inherited, unmodified) is what `AIManager.select_provider()`
    consults when routing a request here.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        capabilities: FrozenSet[AICapability],
        name: str = _DEFAULT_PROVIDER_NAME,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a Gemini provider bound to one model and API key.

        Args:
            api_key: Gemini API key. Supplied by the caller (e.g. read
                from an environment variable or secrets store upstream
                of this class) - never hard-coded or logged here.
            model_name: The Gemini model to call (e.g.
                "gemini-1.5-flash"). No default is assumed, since the
                repository does not currently declare one.
            capabilities: The `AICapability` values this configured
                model/account combination actually supports. Passed
                straight through to `AIProviderBase`.
            name: Registration name used with
                `AIManager.register_provider`. Defaults to "gemini".
            generation_config: Optional Gemini `generation_config`
                overrides (e.g. temperature, max output tokens),
                forwarded to the SDK as-is. `AIManager`/`PromptBuilder`
                never construct this - it is Gemini-specific and stays
                entirely inside this adapter.
        """
        super().__init__(name=name, capabilities=capabilities)

        # Imported here, not at module level, so importing this module
        # never fails just because `google-generativeai` isn't
        # installed yet - only constructing a `GeminiProvider` does.
        import google.generativeai as genai

        self._genai = genai
        self._model_name = model_name
        self._generation_config = generation_config or {}

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model_name=model_name)

    # --- Core generation ---------------------------------------------

    def generate(self, request: AIRequest) -> AIResponse:
        """Executes `request` against the configured Gemini model.

        Sends `request.prompt` (already built upstream by
        `PromptBuilder`, or passed through verbatim by `AIManager`) to
        `GenerativeModel.generate_content`. Never inspects or rebuilds
        the prompt itself.

        Args:
            request: The provider-agnostic request. `request.capability`
                is guaranteed by `AIManager` to be one this provider
                was selected for.

        Returns:
            A successful `AIResponse` with the generated text/data in
            `data`, or a failed `AIResponse` with `error` set for any
            expected provider/API failure.
        """
        if not request.prompt:
            return AIResponse(
                request_id=request.request_id,
                success=False,
                provider_name=self.name,
                error="Empty prompt: nothing to send to Gemini.",
            )

        try:
            response = self._model.generate_content(
                request.prompt,
                generation_config=self._generation_config or None,
            )
        except self._genai.types.StopCandidateException as exc:
            return self._failure(request.request_id, f"Gemini generation stopped early: {exc}")
        except Exception as exc:  # Expected provider/API failure surface.
            return self._failure(request.request_id, self._describe_error(exc))

        text = self._extract_text(response)
        if not text:
            return self._failure(request.request_id, "Gemini returned an empty or malformed response.")

        return AIResponse(
            request_id=request.request_id,
            success=True,
            provider_name=self.name,
            data={"text": text, "model": self._model_name},
        )

    # --- Optional capability hooks -----------------------------------

    def analyze_sketch(self, sketch: Any) -> Dict[str, Any]:
        """Describes a sketch image using Gemini's multimodal input support.

        `sketch` is expected to be raw image bytes or a PIL-compatible
        image object, as accepted directly by `generate_content`'s
        multimodal content list. This is a provider-backed alternative
        to the local `SketchAnalyzer` - it performs an actual Gemini
        call and is not free the way local OpenCV analysis is.

        Args:
            sketch: Opaque image data understood by the Gemini SDK's
                multimodal content parameter.

        Returns:
            A dict with a single `"description"` key holding Gemini's
            text description of the sketch.

        Raises:
            RuntimeError: If the Gemini call fails.
        """
        try:
            response = self._model.generate_content(
                ["Describe the shapes, objects, and composition in this sketch.", sketch]
            )
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
            response = self._model.generate_content(instruction)
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
            list(self._genai.list_models())
        except Exception as exc:
            return ProviderHealth(status=ProviderHealthStatus.UNAVAILABLE, message=self._describe_error(exc))
        return ProviderHealth(status=ProviderHealthStatus.HEALTHY)

    # --- Internal helpers -------------------------------------------------

    def _extract_text(self, response: Any) -> Optional[str]:
        """Safely extracts text from a Gemini `GenerateContentResponse`.

        Guards against Gemini's `response.text` raising when a
        response has no valid candidate (e.g. blocked by safety
        filters) - that case is treated as "no text" rather than
        propagating the SDK's exception past this adapter.
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