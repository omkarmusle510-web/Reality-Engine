"""NVIDIA NIM vision recognition provider for Reality Painter.

Implements `apps.reality_painter.recognition.provider.RecognitionProvider`
against NVIDIA's hosted OpenAI-compatible chat-completions endpoint
(`meta/llama-3.2-11b-vision-instruct`), the same endpoint/model already
proven by the standalone diagnostic at
`tests/diagnostic_nvidia_flower_recognition.py`. This module owns every
NVIDIA-specific concern - endpoint, model id, request payload, response
parsing - so `InspectionController` and the generic `RecognitionProvider`
contract never see any of it and remain provider-agnostic.

Never raises: every expected failure (network, auth, rate limit,
malformed/empty response) is caught here and reported as a failed
`RecognitionResult`, matching the contract `RecognitionProvider`
documents. The API key is supplied by the caller at construction and is
never logged, printed, or included in any error message.
"""

from __future__ import annotations

import base64
from typing import Any, Tuple

import cv2
import numpy as np
import requests

from apps.reality_painter.recognition.models import RecognitionResult, RecognizedObject
from engine.core.logger import get_logger

logger = get_logger(__name__)

_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
_MODEL = "meta/llama-3.2-11b-vision-instruct"
_DEFAULT_TIMEOUT_SECONDS = 30.0
_MAX_TOKENS = 256
_DEFAULT_CONFIDENCE = 0.6

_PROMPT = (
    "You are looking at a simple hand-drawn sketch. "
    "On the first line, respond with ONLY the single object name "
    "this drawing most likely depicts (one or two words, lowercase, "
    "e.g. 'flower', 'car', 'house', 'unknown'). "
    "On the following lines, briefly explain the visual reasoning "
    "for your answer (shapes, lines, composition you can see)."
)


class NvidiaRecognitionProvider:
    """Adapts NVIDIA NIM's hosted vision model to the `RecognitionProvider` contract.

    Configuration (API key) is entirely caller-supplied at construction -
    this class never reads an environment variable or hardcodes a key
    itself (that happens once, at the application boundary).
    """

    def __init__(self, api_key: str, timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS) -> None:
        """Creates a provider bound to one NVIDIA API key.

        Args:
            api_key: NVIDIA API key, supplied by the caller (read from
                `NVIDIA_API_KEY` upstream) - never hardcoded, logged, or
                included in any exception message raised by this class.
            timeout_seconds: Per-request HTTP timeout.

        Raises:
            ValueError: If `api_key` is empty.
        """
        if not api_key or not api_key.strip():
            raise ValueError("NvidiaRecognitionProvider requires a non-empty api_key.")
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    # --- RecognitionProvider protocol ---------------------------------

    def recognize(self, image: Any) -> RecognitionResult:
        """Classifies `image` via NVIDIA NIM and returns a `RecognitionResult`.

        Args:
            image: A `numpy.ndarray` (BGR), or an object exposing an
                `.image` attribute that is one (e.g. `Frame`).

        Returns:
            A `RecognitionResult`. Never raises - every expected failure
            is converted to `RecognitionResult(succeeded=False,
            error=...)`, matching `RecognitionProvider`'s contract.
        """
        try:
            data_uri = self._encode_data_uri(image)
        except ValueError as exc:
            return RecognitionResult(succeeded=False, error=str(exc))

        payload = {
            "model": _MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                }
            ],
            "max_tokens": _MAX_TOKENS,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        try:
            response = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=self._timeout_seconds)
        except requests.exceptions.Timeout:
            return RecognitionResult(succeeded=False, error="NVIDIA NIM request timed out.")
        except requests.exceptions.ConnectionError as exc:
            return RecognitionResult(succeeded=False, error=f"Could not reach NVIDIA NIM: {type(exc).__name__}.")
        except requests.exceptions.RequestException as exc:
            return RecognitionResult(succeeded=False, error=f"NVIDIA NIM request failed: {type(exc).__name__}.")

        return self._parse_response(response)

    # --- Image encoding -------------------------------------------------

    def _encode_data_uri(self, image_source: Any) -> str:
        """Encodes opaque image data into a base64 PNG data URI.

        Accepts a raw `numpy.ndarray` (BGR), or an object exposing an
        `.image` attribute that is one (e.g. `Frame`) - the same
        acceptance rule already used by `GeminiProvider._encode_image`
        and `CloudflareProvider._encode_image`.

        Raises:
            ValueError: If `image_source` can't be resolved to a usable
                image, or PNG encoding fails.
        """
        image_array = image_source
        if not isinstance(image_array, np.ndarray):
            image_array = getattr(image_source, "image", None)

        if not isinstance(image_array, np.ndarray) or image_array.size == 0:
            raise ValueError("No usable image data to send to NVIDIA NIM.")

        success, encoded = cv2.imencode(".png", image_array)
        if not success:
            raise ValueError("Failed to encode image for NVIDIA NIM.")

        encoded_b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        return f"data:image/png;base64,{encoded_b64}"

    # --- Response parsing -------------------------------------------------

    def _parse_response(self, response: "requests.Response") -> RecognitionResult:
        """Converts a raw NVIDIA HTTP response into a `RecognitionResult`.

        Never leaks the API key - it is never present in NVIDIA's
        response body or in anything this method reads.
        """
        if response.status_code in (401, 403):
            return RecognitionResult(succeeded=False, error="NVIDIA NIM authentication failed.")
        if response.status_code == 429:
            return RecognitionResult(succeeded=False, error="NVIDIA NIM rate limit exceeded.")
        if response.status_code != 200:
            return RecognitionResult(succeeded=False, error=f"NVIDIA NIM returned HTTP {response.status_code}.")

        try:
            body = response.json()
            text = body["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            return RecognitionResult(succeeded=False, error="Malformed response from NVIDIA NIM.")

        if not text or not str(text).strip():
            return RecognitionResult(succeeded=False, error="NVIDIA NIM returned an empty response.")

        label, reasoning = self._split_label_and_reasoning(str(text))
        if not label:
            return RecognitionResult(succeeded=False, error="NVIDIA NIM response had no parsable label.")

        return RecognitionResult(
            succeeded=True,
            objects=[RecognizedObject(label=label, confidence=_DEFAULT_CONFIDENCE, reasoning=reasoning or None)],
        )

    def _split_label_and_reasoning(self, raw_text: str) -> Tuple[str, str]:
        """Splits the model's raw response into `(label, reasoning)`.

        Mirrors `tests/diagnostic_nvidia_flower_recognition.py`'s own
        parsing convention: first non-empty line is the label, the rest
        is reasoning.
        """
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        if not lines:
            return "", ""
        label = lines[0].strip(".:- ").lower()
        reasoning = " ".join(lines[1:]) if len(lines) > 1 else ""
        return label, reasoning
