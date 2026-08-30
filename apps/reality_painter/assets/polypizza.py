"""Poly Pizza API client for search-on-demand 3D asset discovery.

Connects to Poly Pizza's official public REST API (v1.1) to search for
low-poly 3D models by keyword. Models are returned as ``Asset`` objects
compatible with the existing ``AssetRegistry``/``AssetRetriever``
architecture — no second registry or retrieval path is introduced.

API details:
    Base URL: ``https://api.poly.pizza/v1.1``
    Auth: ``x-auth-token`` header with API key from ``POLY_PIZZA_API_KEY``
    Search: ``GET /search/{keyword}?limit=N``
    Response: JSON with ``results`` array, each containing ``ID``,
        ``Title``, ``Download`` (direct GLB CDN URL), ``Licence``,
        ``Creator``, ``Tags``, ``TriangleCount``, etc.

This module performs metadata search and candidate ranking only. It
never downloads model files itself — that is ``AssetRetriever``'s
responsibility via the ``poly_pizza`` source type. It never crawls,
scrapes, or bulk-downloads the Poly Pizza catalogue.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.schema import Asset, AssetSource

logger = logging.getLogger(__name__)

_API_BASE_URL = "https://api.poly.pizza/v1.1"
_ENV_VAR_API_KEY = "POLY_PIZZA_API_KEY"
_DEFAULT_TIMEOUT_SECONDS = 10.0
_DEFAULT_SEARCH_LIMIT = 10
_MAX_TRIANGLE_PREFERENCE = 5000  # prefer models under this count


# --- Errors ---------------------------------------------------------------

class PolyPizzaError(Exception):
    """Base class for errors raised by the Poly Pizza client."""


class PolyPizzaAuthError(PolyPizzaError):
    """Raised when API key is missing or invalid."""


class PolyPizzaRateLimitError(PolyPizzaError):
    """Raised when rate limit is exceeded."""


class PolyPizzaNetworkError(PolyPizzaError):
    """Raised on network-level failures."""


# --- API key ---------------------------------------------------------------

def get_api_key() -> Optional[str]:
    """Returns the Poly Pizza API key from environment, or None."""
    key = os.environ.get(_ENV_VAR_API_KEY, "").strip()
    return key if key else None


def is_available() -> bool:
    """Returns True if a Poly Pizza API key is configured."""
    return get_api_key() is not None


# --- HTTP helpers -----------------------------------------------------------

def _get(url: str, api_key: str, params: Optional[Dict[str, str]] = None,
         session: Optional[Any] = None) -> Any:
    """Issues one authenticated GET request to Poly Pizza."""
    http = session if session is not None else requests
    headers = {"x-auth-token": api_key}
    try:
        return http.get(url, headers=headers, params=params,
                        timeout=_DEFAULT_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout as exc:
        raise PolyPizzaNetworkError("Poly Pizza request timed out.") from exc
    except requests.exceptions.ConnectionError as exc:
        raise PolyPizzaNetworkError("Could not reach Poly Pizza API.") from exc
    except requests.exceptions.RequestException as exc:
        raise PolyPizzaNetworkError(
            f"Poly Pizza request failed ({type(exc).__name__}).") from exc


def _raise_for_status(response: Any) -> None:
    """Raises appropriate error for non-200 responses."""
    if response.status_code == 429:
        raise PolyPizzaRateLimitError("Poly Pizza rate limit exceeded.")
    if response.status_code in (401, 403):
        raise PolyPizzaAuthError(
            f"Poly Pizza authentication failed (HTTP {response.status_code}).")
    if response.status_code == 404:
        return  # empty results, not an error
    if response.status_code != 200:
        raise PolyPizzaError(
            f"Unexpected Poly Pizza response (HTTP {response.status_code}).")


# --- Search -----------------------------------------------------------------

def search_models(
    keyword: str,
    limit: int = _DEFAULT_SEARCH_LIMIT,
    session: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Searches Poly Pizza for models matching ``keyword``.

    Args:
        keyword: Search term (e.g. "tree", "car", "dog").
        limit: Maximum number of results to return.
        session: Optional requests-compatible session for testing.

    Returns:
        List of raw model dicts from the API response, each containing
        at minimum: ID, Title, Download, Licence, Tags.

    Raises:
        PolyPizzaAuthError: If API key is missing or invalid.
        PolyPizzaRateLimitError: If rate limited.
        PolyPizzaNetworkError: On network failure.
        PolyPizzaError: On unexpected API response.
    """
    api_key = get_api_key()
    if not api_key:
        raise PolyPizzaAuthError(
            "No Poly Pizza API key configured. Set POLY_PIZZA_API_KEY.")

    clean_keyword = keyword.strip()
    if not clean_keyword:
        return []

    url = f"{_API_BASE_URL}/search/{requests.utils.quote(clean_keyword)}"
    response = _get(url, api_key, params={"limit": str(limit)},
                    session=session)
    _raise_for_status(response)

    if response.status_code == 404:
        return []

    try:
        payload = response.json()
    except ValueError:
        raise PolyPizzaError("Malformed JSON response from Poly Pizza.")

    if not isinstance(payload, dict):
        return []

    results = payload.get("results", [])
    if not isinstance(results, list):
        return []

    return results


# --- Asset construction -----------------------------------------------------

def _make_asset_id(model_id: str) -> str:
    """Builds a deterministic, stable asset id for a Poly Pizza model."""
    slug = re.sub(r"[^a-z0-9]+", "_", f"polypizza_{model_id}".lower()).strip("_")
    return slug


def _normalize_license(raw_licence: Optional[str]) -> Optional[str]:
    """Normalizes Poly Pizza licence strings to SPDX-like identifiers."""
    if not raw_licence:
        return None
    normalized = raw_licence.strip().lower()
    # Common Poly Pizza licence values
    licence_map = {
        "cc0 1.0": "CC0-1.0",
        "cc0": "CC0-1.0",
        "cc-by 4.0": "CC-BY-4.0",
        "cc-by 3.0": "CC-BY-3.0",
        "cc-by-4.0": "CC-BY-4.0",
        "cc-by-3.0": "CC-BY-3.0",
        "public domain": "CC0-1.0",
    }
    return licence_map.get(normalized, raw_licence.strip())


def _is_word_match(query: str, text: str) -> bool:
    """True if query appears as a whole word in text (case-insensitive)."""
    pattern = re.compile(
        r"(?:^|[^a-z0-9])" + re.escape(query.lower()) + r"(?:[^a-z0-9]|$)")
    return bool(pattern.search(text.lower()))


def _rank_candidate(model: Dict[str, Any], query: str) -> Tuple[int, int, str]:
    """Ranks a Poly Pizza model candidate.

    Returns a tuple for sorting (lower is better):
        (match_quality, triangle_count, title)
    """
    title = str(model.get("Title", "")).lower()
    tags = [str(t).lower() for t in model.get("Tags", []) if isinstance(t, str)]
    query_lower = query.lower()

    # Match quality
    if query_lower == title:
        quality = 0  # exact title match
    elif any(query_lower == tag for tag in tags):
        quality = 1  # exact tag match
    elif _is_word_match(query, title):
        quality = 2  # word boundary match in title
    elif any(_is_word_match(query, tag) for tag in tags):
        quality = 3  # word boundary match in tags
    else:
        quality = 4  # weaker match

    tri_count = model.get("TriangleCount", 99999)
    if not isinstance(tri_count, (int, float)):
        tri_count = 99999

    return (quality, int(tri_count), title)


def build_asset_from_model(model: Dict[str, Any]) -> Optional[Asset]:
    """Converts a Poly Pizza API model dict into an ``Asset``.

    Returns None if the model lacks required fields (ID, Title, Download).
    """
    model_id = model.get("ID")
    title = model.get("Title")
    download_url = model.get("Download")

    if not model_id or not title or not download_url:
        return None

    asset_id = _make_asset_id(str(model_id))
    licence = _normalize_license(model.get("Licence"))

    # Extract creator info for attribution
    creator = model.get("Creator", {})
    creator_name = creator.get("Username", "") if isinstance(creator, dict) else ""

    tags_raw = model.get("Tags", [])
    tags = [str(t).lower() for t in tags_raw if isinstance(t, str)]

    # Category inference from tags
    known_categories = {
        "furniture", "plants", "vehicles", "characters", "props",
        "architecture", "nature", "animals", "weapons", "food",
    }
    category = "unknown"
    for tag in tags:
        if tag in known_categories:
            category = tag
            break

    source_details: Dict[str, Any] = {
        "download_url": str(download_url),
        "model_id": str(model_id),
    }
    if creator_name:
        source_details["creator"] = creator_name

    data = {
        "id": asset_id,
        "name": str(title).strip().title() or str(model_id),
        "category": category,
        "format": "glb",
        "source": {"type": "poly_pizza", **source_details},
        "tags": sorted(set(tags)),
    }
    if licence:
        data["license"] = licence

    try:
        return Asset.from_dict(data)
    except Exception as exc:
        logger.debug("Failed to build Asset from Poly Pizza model %s: %s",
                      model_id, exc)
        return None


# --- Search + register flow -----------------------------------------------

def search_and_register(
    query: str,
    registry: AssetRegistry,
    limit: int = _DEFAULT_SEARCH_LIMIT,
    session: Optional[Any] = None,
) -> List[Asset]:
    """Searches Poly Pizza for ``query`` and registers results into ``registry``.

    Results are ranked by match quality and triangle count (preferring
    lower-poly models). Already-registered assets are skipped.

    Args:
        query: Clean object label to search for.
        registry: AssetRegistry to register found assets into.
        limit: Max results to fetch from API.
        session: Optional requests-compatible session for testing.

    Returns:
        List of newly registered Assets, sorted by relevance.
        Empty list if no API key, search fails, or no valid results.
    """
    if not is_available():
        logger.debug("Poly Pizza search skipped: no API key configured.")
        return []

    try:
        raw_results = search_models(query, limit=limit, session=session)
    except PolyPizzaError as exc:
        logger.warning("Poly Pizza search for '%s' failed: %s", query, exc)
        return []

    if not raw_results:
        return []

    # Build assets and rank them
    candidates: List[Tuple[Tuple[int, int, str], Asset]] = []
    for model in raw_results:
        asset = build_asset_from_model(model)
        if asset is not None:
            rank = _rank_candidate(model, query)
            candidates.append((rank, asset))

    # Sort by rank
    candidates.sort(key=lambda x: x[0])

    registered: List[Asset] = []
    for _rank, asset in candidates:
        if registry.register(asset):
            registered.append(asset)

    return registered

