"""Reality backends — each owns a slice of truth and advances it deterministically.

A backend integrates internally at fine dt and emits coarse sampled events
(world-tick granularity; see README "Event granularity"). Backends are the
only nondeterminism-free zone the verifier trusts; the Director never calls
one directly — the Engine does, after intent validation.

Swap points: `pendulum.PendulumBackend` today; MuJoCo / RDKit / social-agent
backends implement the same protocol tomorrow.
"""

from __future__ import annotations

from typing import Protocol

from ..models import Event
from ..world import World


class Backend(Protocol):
    """Protocol every reality backend implements."""

    name: str
    #: integrator names this backend supports — flows into intent validation
    #: and the Director's observation, so capabilities live with the truth
    integrators: tuple[str, ...]

    def reset(self, world: World) -> list[Event]:
        """Events that restore the backend's slice to its initial conditions."""
        ...

    def advance(self, world: World, duration: float) -> list[Event]:
        """Advance ground truth by `duration` seconds; return sampled events."""
        ...
