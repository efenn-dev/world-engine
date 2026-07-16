"""The Engine: observe → decide → validate → apply → advance truth → verify.

One Director decision per round (world tick); the backend runs many
deterministic substeps inside a single `run` intent — the clocks are
decoupled on purpose (an LLM call takes seconds, physics takes microseconds).
Every decision, rejection, sample and verdict lands in the event log, so a
whole session is replayable; the LLM is the only nondeterministic part and
its outputs are logged too.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from .backends import Backend
from .contract import IntentError, validate_intent
from .director import Director
from .eventlog import EventLog
from .models import Event, ExperimentResult, Observation, Outcome
from .projector import Projector
from .verifier import Verifier
from .world import World

log = logging.getLogger(__name__)

#: how many recent notes (rejections, hints) the observation carries
_NOTE_WINDOW = 3


class Engine:
    """Runs the director-over-truth loop for one goal."""

    def __init__(
        self,
        world: World,
        eventlog: EventLog,
        backend: Backend,
        verifier: Verifier,
        director: Director,
        goal: str,
        projectors: Sequence[Projector] = (),
        max_rounds: int = 12,
        sim_entity: int = 1,
    ) -> None:
        self.world = world
        self.eventlog = eventlog
        self.backend = backend
        self.verifier = verifier
        self.director = director
        self.goal = goal
        self.projectors = list(projectors)
        self.max_rounds = max_rounds
        self.sim_entity = sim_entity

    def run(self) -> Outcome:
        """Run rounds until the Director concludes or max_rounds is hit."""
        outcome = Outcome()
        notes: list[str] = []
        experiment = 0
        for round_no in range(1, self.max_rounds + 1):
            outcome.rounds = round_no
            observation = Observation(
                round=round_no,
                goal=self.goal,
                available_integrators=list(self.backend.integrators),
                configured=self.world.get(self.sim_entity, "sim_config"),
                results=list(outcome.results),
                notes=notes[-_NOTE_WINDOW:],
            )
            intent = self.director.decide(observation)
            self.eventlog.append(
                Event(
                    kind="director.decided",
                    entity=self.sim_entity,
                    data={"round": round_no, "intent": intent.model_dump()},
                )
            )
            try:
                validate_intent(intent, self.world, self.sim_entity, self.backend.integrators)
            except IntentError as exc:
                notes.append(f"intent rejected: {exc}")
                self.eventlog.append(
                    Event(
                        kind="intent.rejected",
                        entity=self.sim_entity,
                        data={"round": round_no, "reason": str(exc)},
                    )
                )
                log.warning("round %d: intent rejected: %s", round_no, exc)
                continue

            if intent.action == "configure":
                integrator = intent.args["integrator"]
                dt = float(intent.args["dt"])
                self._apply(
                    Event(
                        kind="experiment.configured",
                        entity=self.sim_entity,
                        data={
                            "rationale": intent.rationale,
                            "set_components": {"sim_config": {"integrator": integrator, "dt": dt}},
                        },
                    )
                )
                log.info("round %d: configured %s dt=%g (%s)", round_no, integrator, dt, intent.rationale)
            elif intent.action == "run":
                experiment += 1
                result, samples = self._run_experiment(experiment, float(intent.args["duration"]))
                outcome.results.append(result)
                outcome.samples[experiment] = samples
                for projector in self.projectors:
                    projector.on_result(result)
            elif intent.action == "conclude":
                outcome.concluded = True
                outcome.summary = str(intent.args["summary"])
                self.eventlog.append(
                    Event(
                        kind="session.concluded",
                        entity=self.sim_entity,
                        data={"summary": outcome.summary},
                    )
                )
                log.info("round %d: concluded — %s", round_no, outcome.summary)
                break
        else:
            log.warning("max rounds (%d) reached without a conclusion", self.max_rounds)

        for projector in self.projectors:
            projector.finalize(outcome)
        return outcome

    def _apply(self, event: Event) -> Event:
        """Append an event to the log, then fold it into world state."""
        stored = self.eventlog.append(event)
        self.world.apply(stored)
        return stored

    def _run_experiment(self, experiment: int, duration: float) -> tuple[ExperimentResult, list[dict[str, float]]]:
        """Reset, advance ground truth, verify, and digest one experiment."""
        for event in self.backend.reset(self.world):
            self._apply(event)
        self.eventlog.append(
            Event(
                kind="run.started",
                entity=self.sim_entity,
                data={"experiment": experiment, "duration": duration},
            )
        )
        samples: list[dict[str, object]] = []
        for event in self.backend.advance(self.world, duration):
            self._apply(event.model_copy(update={"data": {**event.data, "experiment": experiment}}))
            # generic sample: t + kind + every backend diagnostic except the state fold
            samples.append(
                {
                    "t": event.t,
                    "kind": event.kind,
                    **{k: v for k, v in event.data.items() if k != "set_components"},
                }
            )
        verdict = self.verifier.check(samples)
        self.eventlog.append(
            Event(
                kind="verdict.issued",
                entity=self.sim_entity,
                data={"experiment": experiment, **verdict.model_dump()},
            )
        )
        config = self.world.get(self.sim_entity, "sim_config") or {}
        result = ExperimentResult(
            experiment=experiment,
            integrator=str(config.get("integrator") or "n/a"),
            dt=float(config.get("dt") or 0.0),
            duration=duration,
            max_drift_pct=verdict.metrics.get("max_drift_pct", float("nan")),
            passed=verdict.passed,
        )
        log.info(
            "experiment %d: %s dt=%g over %gs -> %s",
            experiment,
            result.integrator,
            result.dt,
            duration,
            verdict.reason,
        )
        return result, samples
