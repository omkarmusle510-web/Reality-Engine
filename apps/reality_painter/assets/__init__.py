"""Asset metadata subsystem for Reality Painter.

Maintains metadata about 3D assets (id, name, category, tags, source,
format, license) independently of where the underlying model files are
actually stored. This package knows nothing about MediaPipe, gestures,
Canvas, the AI subsystem, or rendering - it is a standalone data layer
that future phases (search, retrieval, AI tool-calling) build on top
of, never the other way around.
"""

from __future__ import annotations
