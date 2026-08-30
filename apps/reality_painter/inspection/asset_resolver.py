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

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from apps.reality_painter.assets import auto_discovery
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
    - Strips markdown formatting (**, *, _, `, quotes, etc.) and list bullets (*, -, 1.)
    - Extracts value from label patterns like 'object: flower', 'label: dog'
    - Extracts noun phrase from descriptive carrier phrases like 'the drawing depicts a dog'
    """
    if not label:
        return ""
    text = label.strip()
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


def _normalized_label_candidates(label: str) -> List[str]:
    """Generates deterministic label variants to try against the registry.

    Broadest-first: the clean (lowercased, stripped) label, then
    progressively shorter word-suffixes of it - e.g. "red flower" ->
    ["red flower", "flower"] - so a descriptive recognition label
    still resolves against a registry entry named/tagged with just its
    final noun. Purely string manipulation: no NLP model, no stemming,
    no embeddings. A single-word label (e.g. "car") yields exactly one
    candidate, identical to the label itself.

    Args:
        label: The raw or cleaned recognized label.

    Returns:
        Ordered, deduplicated candidate strings to try, most specific
        first. Empty if `label` is empty/whitespace-only.
    """
    clean = extract_clean_label(label)
    words = clean.split()
    candidates = [clean] if clean else []
    for start in range(1, len(words)):
        candidate = " ".join(words[start:])
        if candidate and candidate not in candidates:
            candidates.append(candidate)
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

    # Fallback 2: Configured external repositories discovery (in order)
    external_repos = auto_discovery.external_repositories()
    for repo in external_repos:
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


