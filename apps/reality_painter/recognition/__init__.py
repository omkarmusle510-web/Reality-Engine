"""Object-recognition contracts for Reality Painter.

Turns a user's drawing into a structured `RecognitionResult`
(`models.py`) via any backend satisfying `provider.RecognitionProvider`.
This package defines the contract only - it contains no concrete
provider implementation (e.g. NVIDIA NIM), no asset resolution, no
retrieval, and no GLB loading. See `apps.reality_painter.inspection`
for the layer that consumes a `RecognitionResult` and turns it into a
loaded 3D asset.
"""

from __future__ import annotations
