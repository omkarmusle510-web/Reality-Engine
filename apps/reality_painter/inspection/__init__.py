"""Recognition -> asset-resolution -> retrieval -> GLB-loading integration for Reality Painter.

Bridges `apps.reality_painter.recognition` (what was drawn) to the
existing asset metadata/retrieval stack
(`apps.reality_painter.assets.registry.AssetRegistry`,
`apps.reality_painter.assets.retriever.AssetRetriever`) and
`engine.scene.loader.load_glb`. This package performs no rendering, no
2D/3D runtime switching, no hand tracking, and no camera access - see
`apps.reality_painter.inspection.controller.InspectionController` for
the single orchestration entry point this package exposes today.
"""

from __future__ import annotations
