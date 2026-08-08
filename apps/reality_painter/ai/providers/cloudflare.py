"""Cloudflare Workers AI provider adapter for Reality Painter's AI subsystem.

`CloudflareProvider` is a concrete `AIProviderBase` implementation that
sends already-prepared, provider-agnostic `AIRequest`s to Cloudflare
Workers AI's REST API, running:

    @cf/black-forest-labs/flux-2-klein-4b

This is a TRUE image-to-image integration, matching the same contract
`GeminiProvider` already established: when an `AIRequest` carries
`canvas_image` (raw sketch pixels, populated by `AIManager.generate()`
from the caller's `sketch` argument - see
`apps.reality_painter.ai.models.AIRequest.canvas_image` and
`AIManager.generate()`), that image is encoded and sent to Cloudflare
alongside the generated prompt as `input_image_0` - never approximated
as a text-only description.

This module builds no prompts (that remains `PromptBuilder`'s job),
selects no providers (that remains `AIManager`'s job), performs no
caching or history recording, and never touches the camera pipeline
thread itself - `generate()` is a plain synchronous call, exactly like
`GeminiProvider.generate()`; whatever calls this provider is
responsible for keeping it off the real-time loop.

Credentials (account id, API token) are supplied entirely by the
caller at construction - this class never reads environment variables
itself (that happens once, at the application boundary - see
`apps/reality_painter/app.py`), never hardcodes a credential, and
never logs or embeds one in an exception message.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, FrozenSet, Optional, Tuple

import cv2
import numpy as np
import requests

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

_DEFAULT_PROVIDER_NAME = "cloudflare"
_DEFAULT_MODEL = "@cf/black-forest-labs/flux-2-klein-4b"
_API_BASE_URL = "https://api.cloudflare.com/client/v4"
_DEFAULT_TIMEOUT_SECONDS = 60.0

# FLUX.2 Klein 4B's documented reference-image constraint.
_MAX_REFERENCE_DIMENSION_PX = 512


def _resize_to_fit(image: np.ndarray, max_dimension: int) -> np.ndarray:
    """Downscales `image` to fit within `max_dimension` on its longest side.

    Preserves aspect ratio; never upscales - a canvas already within
    the limit is returned unchanged, so this never needlessly degrades
    a small drawing.

    Args:
        image: BGR image array.
        max_dimension: Maximum allowed width/height in pixels.

    Returns:
        `image` unchanged, or a resized copy.
    """
    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_dimension:
        return image

    scale = max_dimension / float(longest_side)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


class CloudflareProvider(AIProviderBase):
    """Adapts Cloudflare Workers AI's FLUX.2 Klein 4B to Reality Painter's `AIProvider` contract.

    Configuration (account id, API token, model, declared capabilities)
    is entirely caller-supplied at construction - this class never
    hardcodes or reads credentials itself. Defaults to advertising
    `AICapability.IMAGE_GENERATION`, since that is the capability this
    model actually performs.
    """

    def __init__(
        self,
        account_id: str,
        api_token: str,
        capabilities: FrozenSet[AICapability] = frozenset({AICapability.IMAGE_GENERATION}),
        name: str = _DEFAULT_PROVIDER_NAME,
        model: str = _DEFAULT_MODEL,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Creates a Cloudflare Workers AI provider bound to one account and model.

        Args:
            account_id: Cloudflare account id. Supplied by the caller
                (read from `CLOUDFLARE_ACCOUNT_ID` upstream, in
                `app.py`) - never hardcoded or logged here.
            api_token: Cloudflare API token, scoped for Workers AI.
                Supplied by the caller (read from
                `CLOUDFLARE_API_TOKEN` upstream) - never hardcoded,
                logged, or included in any exception message raised by
                this class.
            capabilities: The `AICapability` values this provider
                should advertise via `supports()`. Defaults to
                `{IMAGE_GENERATION}`.
            name: Registration name used with
                `AIManager.register_provider`. Defaults to "cloudflare".
            model: The Cloudflare Workers AI model path to call.
                Defaults to `@cf/black-forest-labs/flux-2-klein-4b`.
            timeout_seconds: Per-request HTTP timeout.

        Raises:
            ValueError: If `account_id` or `api_token` is empty.
        """
        if not account_id or not account_id.strip():
            raise ValueError("CloudflareProvider requires a non-empty account_id.")
        if not api_token or not api_token.strip():
            raise ValueError("CloudflareProvider requires a non-empty api_token.")

        super().__init__(name=name, capabilities=capabilities)

        self._account_id = account_id
        self._api_token = api_token
        self._model = model
        self._timeout_seconds = timeout_seconds
        self._endpoint = f"{_API_BASE_URL}/accounts/{account_id}/ai/run/{model}"

    # --- Core generation ---------------------------------------------

    def generate(self, request: AIRequest) -> AIResponse:
        """Executes `request` against Cloudflare Workers AI.

        Sends `request.prompt` (already built upstream by
        `PromptBuilder`) as the required `prompt` field, and - if
        present - `request.canvas_image` (the raw sketch pixels
        `AIManager` attached to the request, the same field
        `GeminiProvider` already reads) as `input_image_0`. Never
        inspects or rebuilds the prompt itself, and never approximates
        the image as a text description.

        Args:
            request: The provider-agnostic request. `request.capability`
                is guaranteed by `AIManager` to be one this provider
                was selected for.

        Returns:
            A successful `AIResponse` with the generated PNG bytes in
            `data`, or a failed `AIResponse` with `error` set for any
            expected provider/API/network failure. Never raises.
        """
        if not request.prompt:
            return self._failure(request.request_id, "Empty prompt: nothing to send to Cloudflare.")

        try:
            reference_png = self._encode_image(request.canvas_image)
        except Exception as exc:
            return self._failure(request.request_id, f"Could not prepare reference image: {exc}")

        form_data = {"prompt": request.prompt}
        files: Optional[Dict[str, Tuple[str, bytes, str]]] = None
        if reference_png is not None:
            files = {"input_image_0": ("reference.png", reference_png, "image/png")}

        headers = {"Authorization": f"Bearer {self._api_token}"}

        try:
            http_response = requests.post(
                self._endpoint,
                headers=headers,
                data=form_data,
                files=files,
                timeout=self._timeout_seconds,
            )
        except requests.exceptions.Timeout:
            return self._failure(request.request_id, "Cloudflare request timed out.")
        except requests.exceptions.ConnectionError as exc:
            return self._failure(request.request_id, f"Could not reach Cloudflare: {type(exc).__name__}.")
        except requests.exceptions.RequestException as exc:
            return self._failure(request.request_id, f"Cloudflare request failed: {type(exc).__name__}.")

        return self._handle_response(request.request_id, http_response)

    # --- Response handling -------------------------------------------

    def _handle_response(self, request_id: str, http_response: "requests.Response") -> AIResponse:
        """Converts a Cloudflare HTTP response into an `AIResponse`.

        Handles both response shapes Cloudflare Workers AI image models
        may return: a raw `image/*` body, or a JSON envelope carrying a
        base64-encoded image. Never leaks the API token - it is never
        present in Cloudflare's response body or in anything this
        method reads.

        Args:
            request_id: The originating `AIRequest.request_id`.
            http_response: The raw HTTP response from `requests.post`.

        Returns:
            A successful or failed `AIResponse`.
        """
        status_code = http_response.status_code

        if status_code == 200:
            try:
                image_bytes = self._extract_image_bytes(http_response)
            except Exception as exc:
                return self._failure(request_id, f"Malformed Cloudflare response: {exc}")

            if not image_bytes:
                return self._failure(request_id, "Cloudflare returned a response with no image data.")

            return AIResponse(
                request_id=request_id,
                success=True,
                provider_name=self.name,
                data={"image_bytes": image_bytes, "mime_type": "image/png", "model": self._model},
            )

        if status_code in (401, 403):
            return self._failure(request_id, "Cloudflare authentication failed (check account id / API token).")
        if status_code == 400:
            return self._failure(request_id, f"Cloudflare rejected the request: {self._safe_body(http_response)}")
        if status_code == 429:
            return self._failure(request_id, "Cloudflare rate limit exceeded. Try again shortly.")
        if status_code >= 500:
            return self._failure(request_id, f"Cloudflare service error (HTTP {status_code}).")

        return self._failure(request_id, f"Unexpected Cloudflare response (HTTP {status_code}).")

    def _extract_image_bytes(self, http_response: "requests.Response") -> Optional[bytes]:
        """Extracts raw PNG bytes from a successful Cloudflare response.

        Cloudflare Workers AI image models may respond either with the
        image bytes directly (`content-type: image/*`) or with a JSON
        envelope wrapping a base64-encoded image under a
        `result.image` (or `result.images[0]`) key. Both are handled;
        an unrecognized shape raises so the caller reports a clear
        "malformed response" error rather than silently returning
        nothing.

        Args:
            http_response: The raw HTTP response.

        Returns:
            Raw image bytes, or `None` if the body genuinely carries no
            image (e.g. `success: false` in the JSON envelope).
        """
        content_type = http_response.headers.get("content-type", "")

        if content_type.startswith("image/"):
            return http_response.content

        payload = http_response.json()

        if isinstance(payload, dict) and payload.get("success") is False:
            errors = payload.get("errors")
            raise ValueError(str(errors) if errors else "Cloudflare reported an unsuccessful result.")

        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise ValueError("Response JSON has no usable 'result' object.")

        encoded_image = result.get("image")
        if encoded_image is None:
            images = result.get("images")
            if isinstance(images, list) and images:
                encoded_image = images[0]

        if not encoded_image or not isinstance(encoded_image, str):
            raise ValueError("Response JSON 'result' has no image data.")

        return base64.b64decode(encoded_image)

    def _safe_body(self, http_response: "requests.Response") -> str:
        """Returns a short, credential-free snippet of a response body for error messages."""
        try:
            text = http_response.text
        except Exception:
            return "<unreadable response body>"
        return text[:200]

    # --- Reference image preparation ----------------------------------

    def _encode_image(self, image_source: Any) -> Optional[bytes]:
        """Encodes opaque canvas image data into PNG bytes, or `None`.

        Accepts a raw `numpy.ndarray` (BGR, as produced by
        `Canvas.export_snapshot()`), or an object exposing an `.image`
        attribute that is one (e.g. `Frame`) - the same acceptance
        rule `GeminiProvider._encode_image` already uses, so both
        providers behave identically given the same
        `AIRequest.canvas_image`. `None` input (no canvas image on the
        request) returns `None`, letting `generate()` fall back to a
        text-to-image call.

        Args:
            image_source: The opaque value from `AIRequest.canvas_image`.

        Returns:
            PNG-encoded bytes, resized to fit within
            `_MAX_REFERENCE_DIMENSION_PX` on its longest side (aspect
            ratio preserved, never upscaled), or `None` if
            `image_source` couldn't be resolved to a usable image.

        Raises:
            ValueError: If `image_source` is a `numpy.ndarray` (or
                resolves to one) but PNG encoding itself fails.
        """
        if image_source is None:
            return None

        image_array = image_source
        if not isinstance(image_array, np.ndarray):
            image_array = getattr(image_source, "image", None)

        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            return None

        resized = _resize_to_fit(image_array, _MAX_REFERENCE_DIMENSION_PX)
        success, encoded = cv2.imencode(".png", resized)
        if not success:
            raise ValueError("cv2.imencode failed to encode the reference image.")
        return encoded.tobytes()

    # --- Health -----------------------------------------------------------

    def check_health(self) -> ProviderHealth:
        """Reports health without making a billed generation call.

        Cloudflare Workers AI has no dedicated no-cost reachability
        endpoint for a specific model, so this intentionally does not
        perform a network call (unlike `GeminiProvider`/`GroqProvider`,
        which list models as a cheap auth check) - it only confirms
        this instance was constructed with credentials, which the
        constructor already guarantees. A real failure still surfaces
        normally, as a failed `AIResponse` from `generate()`.
        """
        return ProviderHealth(status=ProviderHealthStatus.HEALTHY)

    # --- Internal helpers -------------------------------------------------

    def _failure(self, request_id: str, message: str) -> AIResponse:
        """Builds a failed `AIResponse` tagged with this provider's name."""
        logger.warning("Cloudflare provider '%s' request %s failed: %s", self.name, request_id, message)
        return AIResponse(request_id=request_id, success=False, provider_name=self.name, error=message)
