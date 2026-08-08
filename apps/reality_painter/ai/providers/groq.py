"""Groq provider adapter for Reality Painter's AI subsystem.

`GroqProvider` is a concrete `AIProviderBase` implementation for
Groq's chat-completion API via the `groq` Python SDK.

NOTE ON DEPENDENCIES: this repository's `requirements.txt` does not
currently list `groq`. This module is written against that package's
stable, documented entry point
(`groq.Groq().chat.completions.create`) rather than any
experimental API, since no installed version is pinned in the repo to
verify against. Add `groq` to `requirements.txt` before this module
can actually be imported/run.

CAPABILITY NOTE: `apps.reality_painter.ai.models.AICapability` only
defines `IMAGE_GENERATION`, `IMAGE_EDIT`, and `MODEL_3D_GENERATION`.
Groq's chat-completion API is a text LLM and does not genuinely
perform any of those - so, honoring the existing rule that a provider
must not advertise capabilities it doesn't implement, `GroqProvider`
declares no `AICapability` membership by default and will never be
selected by `AIManager.select_provider()` for a `generate()` call
under normal operation. Its real, working integration is the
`optimize_prompt()` hook on `AIProviderBase`, which is independent of
the capability-gated `generate()`/`supports()` path and lets Reality
Painter use Groq for fast prompt rewriting without requiring a new
`AICapability` member (which would mean redesigning `models.py`).
`generate()` is still implemented, defensively, since it's abstract
on `AIProviderBase` - it returns a clean failure response rather than
raising, in the unlikely case it's ever invoked directly.
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

_DEFAULT_PROVIDER_NAME = "groq"


class GroqProvider(AIProviderBase):
    """Adapts Groq's chat-completion API to Reality Painter's `AIProvider` contract.

    Configuration (API key, model name, declared capabilities) is
    entirely caller-supplied at construction. `capabilities` defaults
    to empty - see the module docstring's "CAPABILITY NOTE" - but a
    caller is free to pass a non-empty set if a future `AICapability`
    member is added that this provider genuinely supports.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        capabilities: FrozenSet[AICapability] = frozenset(),
        name: str = _DEFAULT_PROVIDER_NAME,
        generation_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Creates a Groq provider bound to one model and API key.

        Args:
            api_key: Groq API key. Supplied by the caller - never
                hard-coded or logged here.
            model_name: The Groq-hosted model to call (e.g.
                "llama-3.1-8b-instant"). No default is assumed, since
                the repository does not currently declare one.
            capabilities: The `AICapability` values this provider
                should advertise via `supports()`. Defaults to empty;
                see module docstring.
            name: Registration name used with
                `AIManager.register_provider`. Defaults to "groq".
            generation_config: Optional Groq completion parameter
                overrides (e.g. temperature, max_tokens), forwarded to
                the SDK as-is.
        """
        super().__init__(name=name, capabilities=capabilities)

        # Imported here, not at module level, so importing this module
        # never fails just because `groq` isn't installed yet - only
        # constructing a `GroqProvider` does.
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model_name = model_name
        self._generation_config = generation_config or {}

    # --- Core generation (defensive; see module docstring) --------------

    def generate(self, request: AIRequest) -> AIResponse:
        """Defensively rejects generation, since Groq supports no declared `AICapability`.

        `AIManager.select_provider()` only routes a request here for a
        capability this provider's `supports()` returns True for - and
        by default that set is empty, so this path should not be
        reachable in normal operation. It is implemented (rather than
        left raising `NotImplementedError`) so a misconfiguration fails
        as a clean `AIResponse`, not an uncaught exception.

        Args:
            request: The provider-agnostic request.

        Returns:
            A failed `AIResponse` naming the unsupported capability.
        """
        return AIResponse(
            request_id=request.request_id,
            success=False,
            provider_name=self.name,
            error=f"Groq provider does not support capability {request.capability.value!r}.",
        )

    # --- Optional capability hooks -----------------------------------

    def optimize_prompt(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Uses Groq's chat-completion API to rewrite `prompt` for better generation quality.

        This is Groq's real, working integration in this provider -
        fast text rewriting via chat completion, distinct from and
        never overlapping with `PromptBuilder`'s section composition.

        Args:
            prompt: The prompt text to optimize.
            context: Optional caller-supplied context, included as
                plain text guidance if present.

        Returns:
            The optimized prompt text, or the original `prompt`
            unchanged if Groq's response was empty/unusable.

        Raises:
            RuntimeError: If the Groq call fails.
        """
        instruction = (
            "Rewrite the following image-generation prompt to be more vivid, "
            "specific, and effective, without changing its intent. "
            "Return only the rewritten prompt, with no preamble."
        )
        user_content = f"Prompt: {prompt}"
        if context:
            user_content += f"\n\nAdditional context: {context}"

        try:
            completion = self._client.chat.completions.create(
                model=self._model_name,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": user_content},
                ],
                **self._generation_config,
            )
        except Exception as exc:
            raise RuntimeError(self._describe_error(exc)) from exc

        text = self._extract_text(completion)
        return text.strip() if text else prompt

    # --- Health -----------------------------------------------------------

    def check_health(self) -> ProviderHealth:
        """Performs a lightweight reachability/auth check against Groq.

        Lists available models as a minimal, low-cost call that
        exercises authentication without performing a real completion.
        """
        try:
            self._client.models.list()
        except Exception as exc:
            return ProviderHealth(status=ProviderHealthStatus.UNAVAILABLE, message=self._describe_error(exc))
        return ProviderHealth(status=ProviderHealthStatus.HEALTHY)

    # --- Internal helpers -------------------------------------------------

    def _extract_text(self, completion: Any) -> Optional[str]:
        """Safely extracts text from a Groq chat-completion response.

        Guards against an empty/malformed `choices` list rather than
        letting an `IndexError`/`AttributeError` propagate.
        """
        choices = getattr(completion, "choices", None)
        if not choices:
            return None
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None) if message is not None else None
        return content if content else None

    def _describe_error(self, exc: Exception) -> str:
        """Formats an exception for error reporting without leaking secrets.

        Never includes the API key or full request/auth headers - only
        the exception type and message.
        """
        return f"{type(exc).__name__}: {exc}"