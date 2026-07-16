"""Typed contract objects — everything the Director reads or writes, plus the event spine.

The Director (LLM) never touches world state directly. It reads `Observation`
and writes `Intent`; deterministic systems turn validated intents into `Event`s.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Event(BaseModel):
    """One immutable record of something that happened in the world.

    Events are the source of truth: current world state is a fold over the
    event stream (see `worldengine.world.fold`). An event that changes state
    carries a ``set_components`` mapping in ``data``; diagnostic-only events
    omit it and fold as no-ops.
    """

    seq: int | None = None  # assigned by the EventLog on append
    t: float = 0.0  # simulation time within the current run, seconds
    kind: str
    entity: int | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Intent(BaseModel):
    """A typed proposal from the Director — the only write channel it has.

    Intents are validated against the contract (`worldengine.contract`) before
    any system acts on them; a rejected intent becomes feedback, not state.
    """

    action: Literal["configure", "run", "conclude"]
    args: dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""


class ExperimentResult(BaseModel):
    """Digest of one completed experiment, as shown back to the Director."""

    experiment: int
    integrator: str
    dt: float
    duration: float
    max_drift_pct: float
    passed: bool


class Observation(BaseModel):
    """The scoped view handed to the Director — never the raw world.

    Keeping this small and typed is what keeps the loop inside any model's
    context budget; it is the perception function of the engine.
    """

    round: int
    goal: str
    #: integrators the active backend supports — capabilities come from the
    #: world, not from the Director's prompt
    available_integrators: list[str] = Field(default_factory=list)
    #: the currently configured experiment (None if nothing configured yet) —
    #: without this the Director can't tell "configured, awaiting run" from "blank"
    configured: dict[str, Any] | None = None
    results: list[ExperimentResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class Verdict(BaseModel):
    """Deterministic verifier output for one experiment (the ground-truth reward)."""

    passed: bool
    reason: str
    metrics: dict[str, float] = Field(default_factory=dict)


class Outcome(BaseModel):
    """What a whole engine session produced."""

    concluded: bool = False
    summary: str = ""
    results: list[ExperimentResult] = Field(default_factory=list)
    #: per-experiment ground-truth samples for projectors — scalar diagnostics
    #: plus whatever state the backend reports (theta/omega, qpos/qvel, ...)
    samples: dict[int, list[dict[str, Any]]] = Field(default_factory=dict)
    rounds: int = 0
