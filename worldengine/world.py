"""ECS-lite world state. Entities are bare ids; components are plain dicts.

World state is never authored directly — it is a fold over the event log.
Events that change state carry ``data["set_components"]``, a mapping of
component name → new value for ``event.entity``. A physics body, a molecule
and a character are all just different component sets on an id.
"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from typing import Any

from .models import Event


class World:
    """Current state: a component table folded from events."""

    def __init__(self) -> None:
        self._components: dict[str, dict[int, dict[str, Any]]] = {}

    def set(self, entity: int, component: str, value: dict[str, Any]) -> None:
        """Set a component value on an entity. Only event application should call this."""
        self._components.setdefault(component, {})[entity] = deepcopy(value)

    def get(self, entity: int, component: str) -> dict[str, Any] | None:
        """Read a copy of a component value, or None if absent."""
        value = self._components.get(component, {}).get(entity)
        return deepcopy(value) if value is not None else None

    def entities(self) -> set[int]:
        """All entity ids that currently hold any component."""
        ids: set[int] = set()
        for table in self._components.values():
            ids.update(table.keys())
        return ids

    def apply(self, event: Event) -> None:
        """Fold one event into state. Diagnostic-only events are no-ops."""
        sets = event.data.get("set_components")
        if sets and event.entity is not None:
            for component, value in sets.items():
                self.set(event.entity, component, value)


def fold(events: Iterable[Event]) -> World:
    """Rebuild world state from scratch by replaying an event stream."""
    world = World()
    for event in events:
        world.apply(event)
    return world
