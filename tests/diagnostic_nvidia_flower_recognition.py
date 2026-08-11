"""TEMPORARY diagnostic - NOT part of the application.

Purpose: prove (or disprove) that NVIDIA NIM's hosted vision model can
genuinely recognize a hand-drawn flower sketch.

Pipeline under test:

    local drawing file (PNG/JPG)
        -> base64 data URI
        -> POST https://integrate.api.nvidia.com/v1/chat/completions
           model = meta/llama-3.2-11b-vision-instruct
        -> model's own text response
        -> parsed into a recognized-object label + reasoning

Uses `requests` only - no new SDK dependency. The API key is read from
the NVIDIA_API_KEY environment variable and is never printed, logged,
or included in any error message.

Not integrated into Reality Painter's pipeline, AssetRegistry,
AssetRetriever, the 3D renderer, or app.py. Standalone only. Delete
this file once the diagnostic is done.

Setup (key must already be set in your terminal environment):
    set NVIDIA_API_KEY=nvapi-...          (Windows cmd)
    $env:NVIDIA_API_KEY="nvapi-..."       (PowerShell)

Usage (from C:\\Reality Engine):
    python tools\\diagnostic_nvidia_flower_recognition.py flower1.png flower2.png ...

Each argument is a path to a local drawing/image file. The script
sends every image to the real NVIDIA NIM vision model and prints a
structured PASS/FAIL result based on the model's own answer - no
filename matching, no keyword-only shortcut, no fake classifier. The
target label ("flower") is only used to check whether it appears
in the model's actual returned text; it is never returned as a
hardcoded result.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import requests

_ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
_MODEL = "meta/llama-3.2-11b-vision-instruct"
_TARGET_OBJECT = "flower"
_TIMEOUT_SECONDS = 60.0
_MAX_TOKENS = 256

# NVIDIA's hosted NIM endpoint has historically enforced a payload-size
# limit on inline base64 images (roughly ~180KB) before requiring a
# separate large-asset upload API. This is just a heads-up printed to
# the console, not a hard block - a real 4xx from the API is still
# what determines pass/fail.
_INLINE_IMAGE_WARNING_BYTES = 180_000


def _build_data_uri(image_path: Path) -> str:
    """Reads `image_path` and encodes it as a base64 data URI."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if mime_type not in ("image/png", "image/jpeg"):
        mime_type = "image/png"

    raw_bytes = image_path.read_bytes()
    if len(raw_bytes) > _INLINE_IMAGE_WARNING_BYTES:
        print(
            f"  NOTE: {image_path.name} is {len(raw_bytes)} bytes - NVIDIA's hosted "
            f"endpoint may reject inline base64 images above ~180KB. If the request "
            f"fails with a size-related 4xx, that's why."
        )

    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _call_nvidia_nim(api_key: str, data_uri: str) -> Tuple[Optional[str], Optional[str]]:
    """Sends one image + classification prompt to NVIDIA NIM.

    Returns:
        (response_text, error_message) - exactly one is None.
    """
    prompt = (
        "You are looking at a simple hand-drawn sketch. "
        "On the first line, respond with ONLY the single object name "
        "this drawing most likely depicts (one or two words, lowercase, "
        "e.g. 'flower', 'car', 'house', 'unknown'). "
        "On the following lines, briefly explain the visual reasoning "
        "for your answer (shapes, lines, composition you can see)."
    )

    # No system message here: NVIDIA's NIM VLM docs note that, following
    # Meta's guidance for this model family, system messages are not
    # allowed alongside an image, and only one image per request is
    # supported - both honored below.
    payload = {
        "model": _MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }
        ],
        "max_tokens": _MAX_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    try:
        response = requests.post(_ENDPOINT, headers=headers, json=payload, timeout=_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return None, "Request timed out."
    except requests.exceptions.ConnectionError as exc:
        return None, f"Could not reach NVIDIA NIM endpoint: {type(exc).__name__}."
    except requests.exceptions.RequestException as exc:
        return None, f"Request failed: {type(exc).__name__}."

    if response.status_code == 401 or response.status_code == 403:
        return None, f"Authentication failed (HTTP {response.status_code}) - check NVIDIA_API_KEY is valid for this endpoint/model."
    if response.status_code != 200:
        # Response body may contain useful detail (never the key itself).
        return None, f"NVIDIA NIM returned HTTP {response.status_code}: {response.text[:300]}"

    try:
        body = response.json()
        text = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        return None, f"Malformed response from NVIDIA NIM: {exc}"

    if not text:
        return None, "NVIDIA NIM returned an empty response."

    return text.strip(), None


def _split_label_and_reasoning(raw_text: str) -> Tuple[str, str]:
    """Splits the model's raw response into (label, reasoning)."""
    lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
    if not lines:
        return "", raw_text
    label = lines[0].strip(".:- ").lower()
    reasoning = " ".join(lines[1:]) if len(lines) > 1 else raw_text
    return label, reasoning


def _run_one(api_key: str, image_path: Path) -> None:
    print("OBJECT RECOGNITION TEST")
    print("-----------------------")
    print(f"Input: {image_path.name}")

    try:
        data_uri = _build_data_uri(image_path)
    except OSError as exc:
        print("Recognized object: N/A")
        print(f"Confidence/reasoning: Could not read file ({exc}).")
        print("Result: FAIL")
        print()
        return

    raw_text, error = _call_nvidia_nim(api_key, data_uri)
    if error is not None:
        print("Recognized object: N/A")
        print(f"Confidence/reasoning: NVIDIA NIM call failed - {error}")
        print("Result: FAIL")
        print()
        return

    label, reasoning = _split_label_and_reasoning(raw_text)
    passed = _TARGET_OBJECT in label or _TARGET_OBJECT in raw_text.lower()

    print(f"Recognized object: {label or '(unparsed - see reasoning)'}")
    print(f"Confidence/reasoning: {reasoning}")
    print(f"Result: {'PASS' if passed else 'FAIL'}")
    print()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python tools\\diagnostic_nvidia_flower_recognition.py <image1> [image2] [...]"
        )

    api_key = os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY environment variable is not set.")

    for raw_path in sys.argv[1:]:
        image_path = Path(raw_path)
        if not image_path.is_file():
            print("OBJECT RECOGNITION TEST")
            print("-----------------------")
            print(f"Input: {raw_path}")
            print("Result: FAIL (file not found)")
            print()
            continue
        _run_one(api_key, image_path)


if __name__ == "__main__":
    main()