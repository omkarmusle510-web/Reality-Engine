"""Concrete AI provider adapters for Reality Painter.

Each module here implements `apps.reality_painter.ai.provider.AIProviderBase`
for a specific external AI backend. This package contains no
orchestration, prompt-building, caching, or history logic - those
remain the concern of `apps.reality_painter.ai.manager.AIManager` and
its other injected collaborators.
"""

from __future__ import annotations