"""Recognition -> asset-resolution -> retrieval -> GLB-loading orchestration.

`InspectionController` is the single integration point that turns a
`apps.reality_painter.recognition.models.RecognitionResult` into a
loaded `engine.scene.objects.SceneObject`. It performs no recognition,
no asset-registry matching, no retrieval/caching, and no GLB parsing
itself - it only coordinates the existing components
(`apps.reality_painter.recognition.provider.RecognitionProvider`,
`apps.reality_painter.inspection.asset_resolver.resolve_asset`,
`apps.reality_painter.assets.retriever.AssetRetriever`,
`engine.scene.loader.load_glb`), none of which are modified or
reimplemented here.

Object selection (which recognized object to act on, when more than
one is returned) is isolated in `select_object` so a future, smarter
picker can replace it without touching orchestration logic.

Scope note: this controller only produces an "asset ready" result - it
never triggers 2D/3D runtime switching itself (no state machine, no
mode transition). That integration point is intentionally left for a
later phase; see the module docstring in
`apps.reality_painter.inspection` for the current package boundary.

No exception raised by any collaborator is allowed to escape `run()` -
every expected failure (recognition failure, empty result, unknown
asset, retrieval failure, GLB load failure) is caught and reported as
a clean `ControllerOutcome(success=False, ...)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from apps.reality_painter.assets.registry import AssetRegistry
from apps.reality_painter.assets.retriever import AssetRetrievalError, AssetRetriever
from apps.reality_painter.inspection.asset_resolver import AssetResolutionStatus, resolve_asset
from apps.reality_painter.recognition.models import RecognizedObject, RecognitionResult
from apps.reality_painter.recognition.provider import RecognitionProvider
from engine.core.logger import get_logger
from engine.scene.loader import ModelLoadError, load_glb
from engine.scene.objects import SceneObject

logger = get_logger(__name__)


@dataclass(frozen=True)
class ControllerOutcome:
    """The result of one `InspectionController.run()` call.

    `success=True` represents the "ASSET_READY" outcome described in
    the integration design - a `SceneObject` is loaded and available.
    This dataclass carries that outcome as plain data; wiring it onto
    an explicit runtime state (PAINTING/INSPECTING_3D/...) is left to
    a later phase and is not this controller's concern.

    Attributes:
        success: Whether a `SceneObject` was successfully produced.
        scene_object: The loaded `SceneObject`, or `None` on failure.
        selected_label: The recognized label that was acted on, or
            `None` if recognition itself failed before a label was
            available.
        error: A human-readable error message, or `None` on success.
    """

    success: bool
    scene_object: Optional[SceneObject]
    selected_label: Optional[str]
    error: Optional[str]


def select_object(objects: Sequence[RecognizedObject]) -> Optional[RecognizedObject]:
    """Deterministically selects one recognized object to act on.

    Selection rule: highest `confidence`; ties preserve the order
    `objects` was given in (the first-listed of equally-confident
    candidates wins - a later, equal-confidence candidate never
    displaces it). Isolated as a standalone function so a future,
    smarter object picker can replace it without any change to
    `InspectionController`.

    Args:
        objects: Candidate recognized objects, in recognition-result
            order.

    Returns:
        The selected `RecognizedObject`, or `None` if `objects` is
        empty.
    """
    if not objects:
        return None

    best_index = 0
    best_confidence = objects[0].confidence
    for index in range(1, len(objects)):
        if objects[index].confidence > best_confidence:
            best_confidence = objects[index].confidence
            best_index = index
    return objects[best_index]


class InspectionController:
    """Orchestrates recognition -> asset resolution -> retrieval -> GLB loading.

    Owns no collaborators itself - `provider`, `registry`, and
    `retriever` are all supplied per-call to `run()`, matching this
    repository's existing "collaborators are injected, never imported
    by concrete type" convention (see
    `apps.reality_painter.ai.manager.AIManager`).
    """

    def run(
        self,
        image: Any,
        provider: RecognitionProvider,
        registry: AssetRegistry,
        retriever: AssetRetriever,
    ) -> ControllerOutcome:
        """Runs one full recognition-to-asset-ready cycle.

        Args:
            image: Opaque drawing/canvas data passed to `provider
                .recognize()`.
            provider: The recognition backend to use.
            registry: The `AssetRegistry` to resolve recognized labels
                against.
            retriever: The `AssetRetriever` to use for cache/GitHub
                retrieval of a resolved asset's GLB file. A cache hit
                (a valid, already-downloaded file already on disk)
                never triggers a GitHub request - that invariant is
                `AssetRetriever`'s own, unmodified behavior; this
                controller performs no cache logic of its own.

        Returns:
            A `ControllerOutcome` describing the result. Never raises.
        """
        try:
            recognition_result = provider.recognize(image)
        except Exception as exc:
            logger.exception("Recognition provider raised unexpectedly.")
            return self._fail(None, f"Recognition provider error: {exc}")

        if not recognition_result.succeeded:
            return self._fail(None, recognition_result.error or "Recognition failed.")

        selected = select_object(recognition_result.objects)
        if selected is None:
            return self._fail(None, "Recognition returned no objects.")

        resolution = resolve_asset(selected.label, registry)
        if resolution.status != AssetResolutionStatus.RESOLVED or resolution.asset is None:
            return self._fail(selected.label, f"No registered asset for label {selected.label!r}.")

        try:
            local_path = retriever.retrieve(resolution.asset)
            scene_object = load_glb(local_path, name=resolution.asset.id)
        except AssetRetrievalError as exc:
            return self._fail(selected.label, f"Asset retrieval failed: {exc}")
        except ModelLoadError as exc:
            return self._fail(selected.label, f"GLB load failed: {exc}")
        except Exception as exc:
            logger.exception("Unexpected error during asset retrieval/loading.")
            return self._fail(selected.label, f"Unexpected error: {exc}")

        return ControllerOutcome(success=True, scene_object=scene_object, selected_label=selected.label, error=None)

    def _fail(self, label: Optional[str], error: str) -> ControllerOutcome:
        """Builds a failure `ControllerOutcome`."""
        return ControllerOutcome(success=False, scene_object=None, selected_label=label, error=error)
