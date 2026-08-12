"""Asset Optimizer subsystem for Reality Painter (Block 1: Analyzer only).

This package is intentionally independent from the runtime application
and from Reality Engine itself - it never imports `engine.scene.loader`,
`engine.rendering.renderer`, `AssetRegistry`, or `AssetRetriever`. It
only inspects a local `.glb`/`.gltf` file already on disk and reports
deterministic, read-only performance metrics.

Block 1 provides only `analyzer.py` (offline analysis). Optimization,
LOD generation, hardware detection, and any UI are explicitly out of
scope for this block.
"""

from __future__ import annotations
