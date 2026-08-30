"""JSON-backed asset registry for Reality Painter.

`AssetRegistry` holds validated `Asset` metadata in memory and exposes
small, deterministic lookup APIs (`get_asset`, `list_assets`,
`search_assets`). It performs no network access, no file retrieval, no
AI/embedding-based matching, and no rendering - it only knows about
metadata that has already been loaded, either from a JSON file on disk
or from an in-memory list of dicts.

Future phases (remote retrieval, AI tool-calling, 3D loading) are
expected to sit on top of this registry, never inside it - this module
has no outward dependency on any of them.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Union

from apps.reality_painter.assets.schema import Asset, AssetValidationError

_DEFAULT_REGISTRY_PATH = Path(__file__).parent / "registry.json"


class AssetRegistry:
    """An in-memory collection of validated `Asset` metadata.

    Assets are keyed by their unique `id`; loading a registry with a
    duplicate id is a validation error, so `AssetRegistry` never
    silently drops or overwrites an entry.
    """

    def __init__(self, assets: Optional[Iterable[Asset]] = None) -> None:
        """Creates a registry, optionally pre-populated with `assets`.

        Args:
            assets: Already-validated `Asset` objects to add. Most
                callers should use `AssetRegistry.load()` or
                `AssetRegistry.from_list()` instead of constructing
                pre-validated assets directly.

        Raises:
            AssetValidationError: If `assets` contains two entries with
                the same `id`.
        """
        self._assets: Dict[str, Asset] = {}
        self._scanned_repositories: set[str] = set()
        for asset in assets or []:
            self._add(asset)

    def _add(self, asset: Asset) -> None:
        """Adds one validated asset, rejecting a duplicate id."""
        if asset.id in self._assets:
            raise AssetValidationError(f"Duplicate asset id: {asset.id!r}.")
        self._assets[asset.id] = asset

    def register(self, asset: Asset) -> bool:
        """Registers `asset`, skipping it if its id is already present.

        Idempotent ingestion entry point for asset sources (e.g. the
        GitHub source in `github.py`): unlike `_add`/`from_list`,
        re-registering an id that's already present is not a
        validation error - re-running discovery against the same
        repository is expected to happen and must not raise or
        overwrite the existing entry.

        Args:
            asset: An already-validated `Asset`.

        Returns:
            True if `asset` was newly added, False if an asset with
            this id was already registered (left unchanged).
        """
        if asset.id in self._assets:
            return False
        self._assets[asset.id] = asset
        return True

    # --- Construction -----------------------------------------------

    @classmethod
    def from_list(cls, raw_assets: List[Dict[str, Any]]) -> "AssetRegistry":
        """Builds a registry from a list of raw (unvalidated) asset dicts.

        Args:
            raw_assets: Raw asset entries, e.g. the `"assets"` array of
                a registry JSON file.

        Returns:
            A new `AssetRegistry` with every entry validated.

        Raises:
            AssetValidationError: If any entry is malformed, or two
                entries share the same `id`.
        """
        return cls(Asset.from_dict(entry) for entry in raw_assets)

    @classmethod
    def load(cls, path: Union[str, Path, None] = None) -> "AssetRegistry":
        """Loads and validates a registry from a JSON file.

        Args:
            path: Path to a JSON file shaped `{"assets": [...]}`.
                Defaults to the bundled `registry.json` next to this
                module if omitted.

        Returns:
            A new `AssetRegistry`.

        Raises:
            FileNotFoundError: If `path` does not exist.
            AssetValidationError: If the file's top-level shape is
                wrong, or any entry fails validation.
        """
        registry_path = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
        with open(registry_path, "r", encoding="utf-8") as file:
            payload = json.load(file)

        if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
            raise AssetValidationError(f"Registry file {str(registry_path)!r} must contain an 'assets' list.")

        return cls.from_list(payload["assets"])

    def save(self, path: Union[str, Path, None] = None) -> None:
        """Serializes this registry back to a JSON file, in `id` order.

        Uses an atomic write-and-replace strategy: serializes into a temporary
        file in the same directory, flushes and syncs to disk, and replaces the
        target file atomically via `os.replace()`, preventing truncation or
        corruption if interrupted mid-write.

        Args:
            path: Where to write. Defaults to the bundled
                `registry.json` next to this module if omitted.
        """
        registry_path = Path(path) if path is not None else _DEFAULT_REGISTRY_PATH
        payload = {"assets": [asset.to_dict() for asset in self]}
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        temp_file_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(registry_path.parent),
                prefix=f".{registry_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as file:
                temp_file_path = Path(file.name)
                json.dump(payload, file, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())

            os.replace(str(temp_file_path), str(registry_path))
            temp_file_path = None
        finally:
            if temp_file_path is not None and temp_file_path.is_file():
                try:
                    temp_file_path.unlink()
                except OSError:
                    pass

    # --- Lookup -----------------------------------------------------

    def get_asset(self, asset_id: str) -> Optional[Asset]:
        """Returns the asset with `asset_id`, or `None` if not registered.

        A miss is reported as `None` rather than a raised exception -
        the same convention `dict.get()` uses - since "asset not found"
        is an expected, non-exceptional outcome for callers (e.g. a
        future tool-calling layer checking whether an id exists).

        Args:
            asset_id: The asset's unique `id`.
        """
        return self._assets.get(asset_id)

    def list_assets(self, category: Optional[str] = None) -> List[Asset]:
        """Returns all registered assets, optionally filtered by category.

        Args:
            category: If given, only assets with an exact (case-
                sensitive) `category` match are returned.

        Returns:
            A list of assets, ordered by `id` for deterministic output.
        """
        assets = self._assets.values()
        if category is not None:
            assets = (asset for asset in assets if asset.category == category)
        return sorted(assets, key=lambda asset: asset.id)

    def search_assets(self, query: str) -> List[Asset]:
        """Searches assets by name or tags using deterministic token matching.

        Priority order:
        1. Exact normalized name match
        2. Exact normalized tag match
        3. Word boundary match (the query is a complete word in name or tags)

        A substring inside a larger unrelated word (e.g. "sun" in "sunglasses")
        is NOT a valid match.

        Args:
            query: Free-text search string.

        Returns:
            Matching assets, ordered by match quality (best first),
            then by `id` for determinism.
        """
        import re
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        # Safe word boundary pattern (e.g. ^sun$ or " sun " or "sun_glasses", but not "sunglasses")
        word_pattern = re.compile(r"(?:^|[^a-z0-9])" + re.escape(normalized_query) + r"(?:[^a-z0-9]|$)")

        def _get_match_priority(asset: Asset) -> int:
            name_lower = asset.name.lower()
            if normalized_query == name_lower:
                return 1
            if any(normalized_query == tag.lower() for tag in asset.tags):
                return 2
            if word_pattern.search(name_lower):
                return 3
            if any(word_pattern.search(tag.lower()) for tag in asset.tags):
                return 3
            return 999  # No match

        matches = []
        for asset in self._assets.values():
            priority = _get_match_priority(asset)
            if priority <= 3:
                matches.append((priority, asset))

        # Sort by priority first, then by id
        matches.sort(key=lambda item: (item[0], item[1].id))
        return [asset for priority, asset in matches]

    # --- Introspection ------------------------------------------------

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[Asset]:
        return iter(sorted(self._assets.values(), key=lambda asset: asset.id))

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self._assets
