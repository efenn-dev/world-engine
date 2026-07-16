"""The contract between Director and world: intent validation.

The Director may only read `Observation` and write `Intent`. Everything it
proposes passes through `validate_intent` before any system acts on it — the
LLM's hallucinations never reach ground truth. Rejections become feedback
(observation notes), not state.
"""

from __future__ import annotations

from .models import Intent
from .world import World

MAX_DT = 0.1
MAX_DURATION = 300.0


class IntentError(ValueError):
    """An intent that violates the contract; the reason is fed back to the Director."""


def validate_intent(
    intent: Intent, world: World, sim_entity: int, integrators: tuple[str, ...]
) -> None:
    """Raise `IntentError` if the intent violates the contract.

    Args:
        intent: the Director's proposal.
        world: current world state (validation may depend on it).
        sim_entity: the entity the experiment runs on.
        integrators: integrator names the active backend supports (capabilities
            are backend-owned, not hardcoded here).
    """
    if intent.action == "configure":
        integrator = intent.args.get("integrator")
        dt = intent.args.get("dt")
        if integrator not in integrators:
            raise IntentError(
                f"unknown integrator {integrator!r}; choose one of {list(integrators)}"
            )
        if not isinstance(dt, (int, float)) or isinstance(dt, bool) or not 0 < float(dt) <= MAX_DT:
            raise IntentError(f"dt must be a number in (0, {MAX_DT}], got {dt!r}")
    elif intent.action == "run":
        duration = intent.args.get("duration")
        if (
            not isinstance(duration, (int, float))
            or isinstance(duration, bool)
            or not 0 < float(duration) <= MAX_DURATION
        ):
            raise IntentError(
                f"duration must be a number in (0, {MAX_DURATION}] seconds, got {duration!r}"
            )
        if world.get(sim_entity, "sim_config") is None:
            raise IntentError("no experiment configured yet; send a configure intent first")
    elif intent.action == "conclude":
        if not str(intent.args.get("summary", "")).strip():
            raise IntentError("conclude requires a non-empty args.summary")
