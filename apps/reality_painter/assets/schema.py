"""Data contracts for Reality Painter's asset registry.

Defines the shape of one asset entry - its identity, descriptive
metadata, and where it can eventually be retrieved from - without
assuming anything about how retrieval actually happens. This module
performs no I/O, no network access, and no rendering; it only parses
and validates plain dicts (as loaded from JSON) into typed objects, and
serializes them back.

Storage-provider independence: `AssetSource.type` is a free-form string
("github", "local", "s3", "r2", "huggingface", ...) rather than a fixed
enum. This registry is never updated just because a new storage
provider is added - a source's provider-specific fields (repository,
path, bucket, key, url, ...) live in `AssetSource.details`, which this
module never inspects the contents of.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


class AssetValidationError(ValueError):
    """Raised when a raw asset (or asset source) entry fails validation."""


def _require_non_empty_string(value: Any, field_name: str) -> str:
    """Validates that `value` is a non-empty, non-whitespace string.

    Args:
        value: The value to validate.
        field_name: Human-readable field name, used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        AssetValidationError: If `value` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise AssetValidationError(f"{field_name!r} must be a non-empty string.")
    return value


@dataclass(frozen=True)
class AssetSource:
    """Where an asset can eventually be retrieved from.

    Attributes:
        type: A free-form label identifying the storage provider (e.g.
            "github", "local", "s3", "r2", "huggingface"). Never
            validated against a fixed list, so a new provider never
            requires a change to this module.
        details: Provider-specific fields (e.g. `repository`/`path` for
            GitHub, `bucket`/`key` for S3, `url` for a direct link).
            Opaque to this module - it is never interpreted here, only
            carried through.
    """

    type: str
    details: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(data: Any) -> "AssetSource":
        """Parses and validates a raw `source` mapping.

        Args:
            data: The raw `"source"` value from an asset entry.

        Returns:
            A validated `AssetSource`.

        Raises:
            AssetValidationError: If `data` is not a mapping, or its
                `"type"` field is missing/empty.
        """
        if not isinstance(data, dict):
            raise AssetValidationError("'source' must be an object.")
        source_type = _require_non_empty_string(data.get("type"), "source.type")
        details = {key: value for key, value in data.items() if key != "type"}
        return AssetSource(type=source_type, details=details)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes back to the flat `{"type": ..., ...details}` shape."""
        return {"type": self.type, **self.details}


@dataclass(frozen=True)
class Asset:
    """A single 3D asset's metadata.

    Attributes:
        id: Unique, stable identifier (e.g. "flower_001").
        name: Human-readable display name.
        category: Coarse grouping (e.g. "plants", "furniture").
        format: Runtime file format (e.g. "glb"; "gltf" is expected to
            be a valid future value - this field is never validated
            against a fixed format list).
        source: Where the asset can eventually be retrieved from.
        tags: Freeform search tags. Empty by default.
        license: License/provenance label (e.g. "CC0"), or `None` if
            unknown. Purely informational - never enforced.
    """

    id: str
    name: str
    category: str
    format: str
    source: AssetSource
    tags: Tuple[str, ...] = field(default_factory=tuple)
    license: Optional[str] = None

    @staticmethod
    def from_dict(data: Any) -> "Asset":
        """Parses and validates a raw asset mapping.

        Args:
            data: One raw asset entry, as loaded from JSON.

        Returns:
            A validated `Asset`.

        Raises:
            AssetValidationError: If `data` is not a mapping, any
                required field (`id`, `name`, `category`, `format`,
                `source`) is missing or malformed, or `tags` (when
                present) is not a list of strings.
        """
        if not isinstance(data, dict):
            raise AssetValidationError("Asset entry must be an object.")

        asset_id = _require_non_empty_string(data.get("id"), "id")
        name = _require_non_empty_string(data.get("name"), "name")
        category = _require_non_empty_string(data.get("category"), "category")
        asset_format = _require_non_empty_string(data.get("format"), "format")
        source = AssetSource.from_dict(data.get("source"))

        raw_tags = data.get("tags", [])
        if not isinstance(raw_tags, list) or not all(isinstance(tag, str) for tag in raw_tags):
            raise AssetValidationError(f"Asset {asset_id!r}: 'tags' must be a list of strings.")

        license_value = data.get("license")
        if license_value is not None and not isinstance(license_value, str):
            raise AssetValidationError(f"Asset {asset_id!r}: 'license' must be a string or null.")

        return Asset(
            id=asset_id,
            name=name,
            category=category,
            format=asset_format,
            source=source,
            tags=tuple(raw_tags),
            license=license_value,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serializes back to the same flat shape `from_dict` accepts."""
        result: Dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "format": self.format,
            "source": self.source.to_dict(),
            "tags": list(self.tags),
        }
        if self.license is not None:
            result["license"] = self.license
        return result
