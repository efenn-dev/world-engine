"""The contract between Director and world: intent validation.

The Director may only read `Observation` and write `Intent`. Everything it
proposes passes through `validate_intent` before any system acts on it — the
LLM's hallucinations never reach ground truth. Rejections become feedback
(observation notes), not state.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import ExperimentResult, Intent
from .world import World

MAX_DT = 0.1
MAX_DURATION = 300.0


class IntentError(ValueError):
    """An intent that violates the contract; the reason is fed back to the Director."""



def _has_result_for(config: dict, results: Sequence[ExperimentResult] | None) -> bool:
    """True if some completed experiment matches this exact configuration.

    Compared on (integrator, dt) because that pair IS the experiment; dt is
    compared as a float so 0.01 and 1e-2 are the same experiment.
    """
    if not results:
        return False
    integrator, dt = config.get("integrator"), config.get("dt")
    if dt is None:
        return False
    return any(
        r.integrator == integrator and float(r.dt) == float(dt) for r in results
    )


def validate_intent(
    intent: Intent,
    world: World,
    sim_entity: int,
    integrators: tuple[str, ...],
    results: Sequence[ExperimentResult] | None = None,
) -> None:
    """Raise `IntentError` if the intent violates the contract.

    Args:
        intent: the Director's proposal.
        world: current world state (validation may depend on it).
        sim_entity: the entity the experiment runs on.
        integrators: integrator names the active backend supports (capabilities
            are backend-owned, not hardcoded here).
        results: experiments completed so far. Pass ``[]`` to mean "nothing has
            run yet" and enable the configure-livelock guard; omit it entirely
            to run only the stateless checks (an unknown history cannot
            distinguish a livelock from a legitimate reconfigure).
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
        # Reconfiguring an experiment that was never run is a livelock: the
        # Director can oscillate between two integrators forever, learning
        # nothing, because nothing is ever measured. Observed for real -- a
        # blinded Director burned seven consecutive rounds alternating two
        # configures and never emitted a single run. The system prompt already
        # asks for run-after-configure; the contract has to enforce it, or a
        # confused Director just spins. Rejection becomes feedback, which is how
        # every other contract violation is handled here.
        # Only when the caller supplied results: without them we cannot tell a
        # livelock from a legitimate reconfigure, and a validator that guesses
        # would block honest work. Omitting `results` keeps the stateless checks
        # and nothing more, exactly as before this guard existed.
        pending = world.get(sim_entity, "sim_config") if results is not None else None
        if pending is not None and not _has_result_for(pending, results):
            raise IntentError(
                f"{pending.get('integrator')!r} at dt={pending.get('dt')} is configured but "
                "has no result yet — send run to measure it (or conclude); "
                "reconfiguring now would discard it unmeasured"
            )
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
        # only configurable backends require a configure step before running
        if integrators and world.get(sim_entity, "sim_config") is None:
            raise IntentError("no experiment configured yet; send a configure intent first")
    elif intent.action == "conclude":
        if not str(intent.args.get("summary", "")).strip():
            raise IntentError("conclude requires a non-empty args.summary")
