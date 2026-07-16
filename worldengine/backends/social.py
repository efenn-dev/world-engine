"""Placeholder for the Track-2 swap: a social backend of LLM agent minds.

Intended shape (deliberately unimplemented in this slice):

- Entities carry ``persona`` (goals, traits), ``memory`` (recent event digests)
  and ``location`` components.
- Each `advance` tick, every agent gets a *scoped* Observation (what it can
  see + retrieved-relevant memories), proposes one Intent through the same
  contract the Director uses, and validated intents fold into events
  (moved / spoke / decided).
- Determinism note: agent minds are LLM calls, so unlike physics this backend
  is only *replay*-deterministic — raw LLM outputs must be logged as events
  (the engine already logs director.decided the same way).
"""

from __future__ import annotations

from ..models import Event
from ..world import World


class SocialBackend:
    """Stub — implements the Backend protocol shape but raises on use."""

    name = "social"
    integrators: tuple[str, ...] = ()

    def reset(self, world: World) -> list[Event]:
        """Not implemented in this slice."""
        raise NotImplementedError("SocialBackend is a stub — see module docstring for the design")

    def advance(self, world: World, duration: float) -> list[Event]:
        """Not implemented in this slice."""
        raise NotImplementedError("SocialBackend is a stub — see module docstring for the design")
