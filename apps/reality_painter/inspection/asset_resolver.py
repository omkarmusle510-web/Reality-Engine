"""Recognized-label -> registered-asset resolution for Reality Painter.

`resolve_asset` is the sole bridge between a recognition label (e.g.
"flower") and the existing `AssetRegistry` (Phase 12A/12B) - it
performs no retrieval and no caching itself; it only decides whether a
registered asset exists for a label, using the registry's own existing,
deterministic `search_assets` lookup. It never invents an asset for an
unmatched label, and it never introduces AI/embedding-based or fuzzy
matching beyond what `AssetRegistry` already provides.

Block 11C Phase 2 adds exactly one fallback, in this order:
    1. Already-registered asset (Phase 1's normalized-label lookup,
       unchanged).
    2. On a miss, run the existing Phase 1 auto-discovery
       (`apps.reality_painter.assets.auto_discovery.ensure_discovered`)
       across the configured GitHub repositories, and retry the same
       normalized-label lookup exactly once more.
No second discovery implementation, registry, or matching algorithm is
introduced - this only calls the existing Phase 1 orchestration
(itself unmodified) on a miss. `ensure_discovered` already isolates a
failing repository (never raises `GitHubSourceError` past it) and
never re-scans a repository already represented in the registry, so a
label that remains unresolved after discovery triggers no repeated
network calls on subsequent lookups.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apps.reality_painter.assets import auto_discovery
from apps.reality_painter.assets import polypizza
from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetrievalError, AssetRetriever
from apps.reality_painter.assets.schema import Asset
from engine.core.logger import get_logger

logger = get_logger(__name__)


class AssetResolutionStatus(str, Enum):
    """Outcome of resolving a recognized label against the asset registry."""

    RESOLVED = "resolved"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class AssetResolution:
    """The result of resolving one recognized label.

    Attributes:
        status: Whether a matching registered asset was found.
        asset: The matched `Asset`, or `None` if `status` is
            `UNAVAILABLE`.
        label: The clean label that was resolved, carried through for
            logging/diagnostics.
    """

    status: AssetResolutionStatus
    asset: Optional[Asset]
    label: str


def extract_clean_label(label: str) -> str:
    """Extracts a clean, normalized object label from formatted or natural-language recognition output.

    Deterministic, conservative string normalization:
    - Parses JSON payloads (raw or within markdown code fences) extracting 'label'
    - Strips markdown formatting (**, *, _, `, quotes, etc.) and list bullets (*, -, 1.)
    - Extracts value from label patterns like 'object: flower', 'label: dog'
    - Extracts noun phrase from descriptive carrier phrases like 'the drawing depicts a dog'
    """
    if not label:
        return ""
    text = label.strip()

    # 1. JSON payload / markdown code block extraction
    json_candidate = text
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE)
    if fence_match:
        json_candidate = fence_match.group(1).strip()
    elif text.startswith("{") and text.endswith("}"):
        json_candidate = text
    else:
        obj_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if obj_match:
            json_candidate = obj_match.group(0).strip()

    if json_candidate.startswith("{") and json_candidate.endswith("}"):
        try:
            data = json.loads(json_candidate)
            if isinstance(data, dict):
                for key in ("label", "object", "name", "item", "class", "prediction"):
                    val = data.get(key)
                    if isinstance(val, str) and val.strip():
                        text = val.strip()
                        break
        except (ValueError, TypeError, json.JSONDecodeError):
            pass

    for _ in range(3):
        text = text.strip()
        text = re.sub(r"^[\*\-\+\•\>\#]+\s*", "", text)
        text = re.sub(r"^\d+[\.\)]\s*", "", text)
        text = text.strip("*_`'\"~()[]{}")

    text = text.strip()

    # Key-value pattern: "object: flower", "label: dog", "category: car", etc.
    kv_match = re.search(
        r"(?:^|\b)(?:object|label|class|item|entity|category|name|prediction)\s*:\s*([^,;\n]+)",
        text,
        re.IGNORECASE,
    )
    if kv_match:
        text = kv_match.group(1).strip()
    else:
        # Natural language carrier phrases
        carrier_patterns = [
            r"^(?:this\s+)?(?:is\s+|appears\s+to\s+be\s+|looks\s+like\s+)(?:a|an|the)?\s*(.+)$",
            r"^(?:the\s+)?(?:drawing|image|sketch|picture|canvas|user|photo)\s+(?:depicts|shows|illustrates|represents|is(?:\s+of)?)\s+(?:a|an|the)?\s*(.+)$",
            r"^(?:a|an|the)?\s*(?:drawing|sketch|picture|illustration|image)\s+of\s+(?:a|an|the)?\s*(.+)$",
            r"^it\s+is\s+(?:a|an|the)?\s*(.+)$",
            r"^(?:i\s+see\s+)(?:a|an|the)?\s*(.+)$",
        ]
        for pattern in carrier_patterns:
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                text = m.group(1).strip()
                break

    for _ in range(3):
        text = text.strip()
        text = re.sub(r"^[\*\-\+\•\>\#]+\s*", "", text)
        text = text.strip(".*_`'\"~()[]{},;!?:")

    return text.strip().lower()


_NON_PLURAL_S_WORDS = {
    "bus", "gas", "canvas", "plus", "lens", "news", "walrus", "cactus",
    "focus", "status", "grass", "glass", "dress", "cross", "moss", "chess",
    "scissors", "glasses", "sunglasses", "jeans", "pants", "shorts", "trousers",
    "species", "series", "chaos", "basis", "axis", "analysis",
}


def _singularize_word(word: str) -> Optional[str]:
    """Conservatively converts a regular English plural noun to singular.

    Returns None if the word is already singular or not a recognizable regular plural.
    """
    w = word.lower().strip()
    if len(w) <= 3 or w in _NON_PLURAL_S_WORDS:
        return None

    # -ies -> -y (e.g. cherries -> cherry, berries -> berry, butterflies -> butterfly)
    if w.endswith("ies") and len(w) > 4 and w[-4] not in "aeiou":
        return w[:-3] + "y"

    # -sses -> -ss (e.g. dresses -> dress, glasses -> glass)
    if w.endswith("sses") and len(w) > 4:
        return w[:-2]

    # -xes, -ches, -shes, -zes -> strip -es (e.g. boxes -> box, watches -> watch, dishes -> dish)
    if (w.endswith("xes") or w.endswith("ches") or w.endswith("shes") or w.endswith("zes")) and len(w) > 4:
        return w[:-2]

    # -oes -> -o (e.g. tomatoes -> tomato, potatoes -> potato, heroes -> hero)
    if w.endswith("oes") and len(w) > 4 and w[-4] not in "aeiou":
        return w[:-2]

    # Regular -s (e.g. trees -> tree, flowers -> flower, cars -> car, dogs -> dog, mushrooms -> mushroom)
    # Exclude words ending in 'ss'
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]

    return None


def _singularize_phrase(phrase: str) -> Optional[str]:
    """Singularizes the final word of a noun phrase if applicable."""
    words = phrase.split()
    if not words:
        return None
    last_singular = _singularize_word(words[-1])
    if last_singular and last_singular != words[-1]:
        return " ".join(words[:-1] + [last_singular])
    return None


def _normalized_label_candidates(label: str) -> List[str]:
    """Generates deterministic label variants to try against the registry.

    Broadest-first: the clean (lowercased, stripped) label and its singular
    variant if plural, then progressively shorter word-suffixes of it (and
    their singular variants) - e.g. "red flowers" ->
    ["red flowers", "red flower", "flowers", "flower"] - so a descriptive
    or plural recognition label still resolves against a registry entry
    named/tagged with just its singular base noun.
    """
    clean = extract_clean_label(label)
    if not clean:
        return []

    words = clean.split()
    candidates: List[str] = []

    def _add(cand: Optional[str]) -> None:
        if cand and cand not in candidates:
            candidates.append(cand)

    # 1. Full phrase + singular of full phrase
    _add(clean)
    _add(_singularize_phrase(clean))

    # 2. Suffix phrases + singular of suffix phrases
    for start in range(1, len(words)):
        suffix = " ".join(words[start:])
        _add(suffix)
        _add(_singularize_phrase(suffix))

    return candidates


def _candidate_sort_key(
    asset: Asset,
    query_candidates: List[str],
    repo_order: List[str],
    retriever: Optional[AssetRetriever] = None,
) -> tuple:
    """Builds a deterministic priority key for ordering asset candidates.

    Order:
    1. Local cached valid asset (0 if cached, 1 if not)
    2. Repository order (our primary repository is index 0, configured external repos in order)
    3. Match quality (1=exact name, 2=exact tag, 3=word boundary)
    4. Candidate specificity (index in query_candidates, e.g. 'red flower' before 'flower')
    5. Asset ID for determinism.
    """
    is_cached = 1
    if retriever is not None:
        try:
            cache_path = retriever._cache_path_for(asset)
            if retriever._is_valid_cached_file(cache_path):
                is_cached = 0
        except Exception:
            is_cached = 1

    repo = asset.source.details.get("repository") if asset.source.type == "github" else None
    if asset.source.type == "poly_pizza":
        # Poly Pizza sits between primary repository (index 0) and
        # external GitHub repositories (index 1+) in priority.
        repo_idx = 0.5
    else:
        try:
            repo_idx = repo_order.index(repo)
        except ValueError:
            repo_idx = len(repo_order)

    name_lower = asset.name.lower()
    tags_lower = [t.lower() for t in asset.tags]
    best_candidate_idx = len(query_candidates)
    best_match_prio = 999

    for c_idx, cand in enumerate(query_candidates):
        norm_cand = cand.strip().lower()
        if not norm_cand:
            continue
        if norm_cand == name_lower:
            prio = 1
        elif any(norm_cand == t for t in tags_lower):
            prio = 2
        else:
            prio = 3
        if (prio, c_idx) < (best_match_prio, best_candidate_idx):
            best_match_prio = prio
            best_candidate_idx = c_idx

    return (is_cached, repo_idx, best_match_prio, best_candidate_idx, asset.id)


def _collect_and_sort_candidates(
    query_candidates: List[str],
    registry: AssetRegistry,
    repo_order: List[str],
    retriever: Optional[AssetRetriever] = None,
) -> List[Asset]:
    """Finds and prioritizes all registered assets matching any candidate string."""
    matched_by_id: Dict[str, Asset] = {}
    for cand in query_candidates:
        for asset in registry.search_assets(cand):
            if asset.id not in matched_by_id:
                matched_by_id[asset.id] = asset

    candidates = list(matched_by_id.values())
    candidates.sort(key=lambda a: _candidate_sort_key(a, query_candidates, repo_order, retriever))
    return candidates


def _try_candidates(
    candidates: List[Asset],
    retriever: Optional[AssetRetriever],
    tried_ids: Set[str],
) -> Optional[Asset]:
    """Tries candidates in order, returning the first successfully retrievable asset."""
    for asset in candidates:
        if asset.id in tried_ids:
            continue
        tried_ids.add(asset.id)
        if retriever is not None:
            try:
                retriever.retrieve(asset)
                return asset
            except AssetRetrievalError as exc:
                logger.warning(
                    "Asset candidate '%s' failed retrieval (%s); attempting next candidate.",
                    asset.id,
                    exc,
                )
                continue
            except Exception as exc:
                logger.warning(
                    "Unexpected error retrieving candidate '%s' (%s); attempting next candidate.",
                    asset.id,
                    exc,
                )
                continue
        else:
            return asset
    return None


def resolve_asset(
    label: str,
    registry: AssetRegistry,
    session: Optional[Any] = None,
    registry_path: Optional[Path] = None,
    retriever: Optional[AssetRetriever] = None,
) -> AssetResolution:
    """Resolves a recognized label to a registered asset, if one exists and can be retrieved.

    Deterministic:
    1. Normalizes/extracts the clean object label from markdown or natural language.
    2. Collects matching candidates from `registry` and sorts them by:
       - local cache hit (valid cached file on disk)
       - repository priority (our primary repository first, then configured external fallbacks)
       - match quality (exact name > exact tag > word boundary) and candidate specificity.
    3. Tries each candidate in order. If `retriever` is provided, verifies that
       the candidate can actually be retrieved (catching retrieval errors and
       falling through to the next candidate).
    4. If no candidate from the existing registry succeeds, triggers discovery
       on the primary repository and attempts retrieval on newly found candidates.
    5. If still unresolved, triggers discovery on configured external repositories
       in order and attempts retrieval.
    6. Returns `AssetResolutionStatus.RESOLVED` on the first retrievable match, or
       `AssetResolutionStatus.UNAVAILABLE` if all fail.

    Args:
        label: A recognized object label (e.g. "**object: flower**", or "red flower").
        registry: The `AssetRegistry` to search.
        session: Optional `requests`-compatible session forwarded to discovery.
        registry_path: Forwarded to `ensure_discovered`.
        retriever: Optional `AssetRetriever` used to verify downloadability.

    Returns:
        `AssetResolution(status=RESOLVED, asset=...)` if a retrievable candidate
        matched, or `AssetResolution(status=UNAVAILABLE, asset=None)` otherwise.
    """
    clean_label = extract_clean_label(label)
    if not clean_label:
        return AssetResolution(status=AssetResolutionStatus.UNAVAILABLE, asset=None, label=label)

    configured = auto_discovery.configured_repositories()
    repo_order = [r.get("repository") for r in configured]
    query_candidates = _normalized_label_candidates(clean_label)
    tried_ids: Set[str] = set()

    # Step 1: Search candidates already in registry
    existing_candidates = _collect_and_sort_candidates(query_candidates, registry, repo_order, retriever)
    match = _try_candidates(existing_candidates, retriever, tried_ids)
    if match is not None:
        return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=clean_label)

    # Fallback 1: Primary repository discovery
    primary = auto_discovery.primary_repository()
    auto_discovery.ensure_discovered(
        registry,
        repositories=[primary],
        session=session,
        registry_path=registry_path,
    )
    primary_candidates = _collect_and_sort_candidates(query_candidates, registry, repo_order, retriever)
    match = _try_candidates(primary_candidates, retriever, tried_ids)
    if match is not None:
        return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=clean_label)

    # Fallback 1.5: Poly Pizza search-on-demand
    # Searches the Poly Pizza API for the clean label and registers any
    # compatible results into the existing registry. Only attempted when
    # an API key is configured. This never bulk-downloads the catalogue —
    # it issues a single keyword search for the specific label.
    if polypizza.is_available():
        try:
            pp_assets = polypizza.search_and_register(
                clean_label, registry, session=session)
            if not pp_assets:
                singular = _singularize_phrase(clean_label)
                if singular and singular != clean_label:
                    pp_assets = polypizza.search_and_register(
                        singular, registry, session=session)
            if pp_assets:
                logger.info(
                    "Poly Pizza search for '%s' found %d candidates.",
                    clean_label, len(pp_assets))
                pp_candidates = _collect_and_sort_candidates(
                    query_candidates, registry, repo_order, retriever)
                match = _try_candidates(pp_candidates, retriever, tried_ids)
                if match is not None:
                    return AssetResolution(
                        status=AssetResolutionStatus.RESOLVED,
                        asset=match, label=clean_label)
        except Exception as exc:
            logger.warning(
                "Poly Pizza fallback for '%s' failed: %s", clean_label, exc)

    # Fallback 2: Configured external repositories discovery (in order)
    for repo in configured:
        if repo.get("repository") == primary.get("repository"):
            continue
        auto_discovery.ensure_discovered(
            registry,
            repositories=[repo],
            session=session,
            registry_path=registry_path,
        )
        repo_candidates = _collect_and_sort_candidates(query_candidates, registry, repo_order, retriever)
        match = _try_candidates(repo_candidates, retriever, tried_ids)
        if match is not None:
            return AssetResolution(status=AssetResolutionStatus.RESOLVED, asset=match, label=clean_label)

    return AssetResolution(status=AssetResolutionStatus.UNAVAILABLE, asset=None, label=clean_label)


