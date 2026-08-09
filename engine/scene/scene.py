"""Minimal 3D scene container for Reality Engine.

`Scene` holds a flat collection of `SceneObject`s - no hierarchy, no
ECS. This is the first phase of the engine's 3D capability; a renderer
(see `engine/rendering/renderer.py`) reads a `Scene`'s objects and
draws them, and nothing else in the engine currently depends on this
module.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from engine.scene.objects import SceneObject


class Scene:
    """A flat collection of `SceneObject`s, keyed by name."""

    def __init__(self) -> None:
        self._objects: Dict[str, SceneObject] = {}

    def add(self, obj: SceneObject) -> None:
        """Adds `obj` to the scene, keyed by its `name`.

        Adding an object under a `name` that's already present replaces
        the previous object at that name.

        Args:
            obj: The `SceneObject` to add.
        """
        self._objects[obj.name] = obj

    def remove(self, name: str) -> bool:
        """Removes the object named `name`, if present.

        Args:
            name: The object's `name`, as passed to `add()`.

        Returns:
            True if an object was removed, False if `name` wasn't in
            the scene.
        """
        return self._objects.pop(name, None) is not None

    def get(self, name: str) -> Optional[SceneObject]:
        """Returns the object named `name`, or `None` if not present."""
        return self._objects.get(name)

    def objects(self) -> List[SceneObject]:
        """Returns all objects currently in the scene, order not guaranteed."""
        return list(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)
