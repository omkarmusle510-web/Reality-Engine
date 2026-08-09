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
        """Searches assets by name or tags using a deterministic substring match.

        Case-insensitive substring matching against `name` and each of
        `tags` - no AI model, no embeddings, no fuzzy/ranked scoring.
        An empty or whitespace-only `query` matches nothing, rather
        than returning the entire registry.

        Args:
            query: Free-text search string.

        Returns:
            Matching assets, ordered by `id` for deterministic output.
        """
        normalized_query = query.strip().lower()
        if not normalized_query:
            return []

        def _matches(asset: Asset) -> bool:
            if normalized_query in asset.name.lower():
                return True
            return any(normalized_query in tag.lower() for tag in asset.tags)

        matches = [asset for asset in self._assets.values() if _matches(asset)]
        return sorted(matches, key=lambda asset: asset.id)

    # --- Introspection ------------------------------------------------

    def __len__(self) -> int:
        return len(self._assets)

    def __iter__(self) -> Iterator[Asset]:
        return iter(sorted(self._assets.values(), key=lambda asset: asset.id))

    def __contains__(self, asset_id: object) -> bool:
        return asset_id in self._assets
