"""Engine loop: scripted hypothesis arc, contract rejection feedback, replayability."""

from pathlib import Path

from worldengine.backends.pendulum import PendulumBackend
from worldengine.director import ScriptedDirector
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event, Intent, Observation
from worldengine.verifier import EnergyConservation
from worldengine.world import World, fold


def _seeded(eventlog: EventLog) -> World:
    world = World()
    world.apply(
        eventlog.append(
            Event(
                kind="world.seeded",
                entity=1,
                data={
                    "set_components": {
                        "pendulum_params": {"g": 9.81, "length": 1.0, "mass": 1.0},
                        "initial_conditions": {"theta": 1.0, "omega": 0.0},
                        "pendulum_state": {"theta": 1.0, "omega": 0.0},
                    }
                },
            )
        )
    )
    return world


def _engine(world: World, eventlog: EventLog, director, max_rounds: int = 12) -> Engine:
    return Engine(
        world=world,
        eventlog=eventlog,
        backend=PendulumBackend(entity=1),
        verifier=EnergyConservation(tolerance_pct=1.0),
        director=director,
        goal="conserve pendulum energy within 1%",
        max_rounds=max_rounds,
    )


def test_scripted_arc_converges_on_symplectic(tmp_path: Path) -> None:
    eventlog = EventLog(tmp_path / "events.db")
    world = _seeded(eventlog)
    outcome = _engine(world, eventlog, ScriptedDirector(run_duration=30.0)).run()

    assert outcome.concluded
    assert [r.passed for r in outcome.results] == [False, False, True]
    assert outcome.results[-1].integrator == "symplectic_euler"
    assert "symplectic_euler" in outcome.summary and "conserves energy" in outcome.summary
    assert "explicit_euler@dt=0.02" in outcome.summary  # failures are part of the story

    kinds = {e.kind for e in eventlog.read()}
    assert {"director.decided", "experiment.configured", "verdict.issued", "session.concluded"} <= kinds

    # replayability: fold of the log alone reproduces live state
    replayed = fold(eventlog.read())
    assert replayed.get(1, "pendulum_state") == world.get(1, "pendulum_state")
    eventlog.close()


class ProbeDirector:
    """Emits one contract-violating intent, then a valid arc; records observations."""

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self._step = 0

    def decide(self, observation: Observation) -> Intent:
        self.observations.append(observation)
        self._step += 1
        if self._step == 1:
            return Intent(action="configure", args={"integrator": "leapfrog_deluxe", "dt": 0.01})
        if self._step == 2:
            return Intent(action="configure", args={"integrator": "rk4", "dt": 0.01})
        if self._step == 3:
            return Intent(action="run", args={"duration": 5.0})
        return Intent(action="conclude", args={"summary": "rk4 conserves energy"})


def test_rejected_intent_becomes_feedback_not_state(tmp_path: Path) -> None:
    eventlog = EventLog(tmp_path / "events.db")
    world = _seeded(eventlog)
    director = ProbeDirector()
    outcome = _engine(world, eventlog, director).run()

    # the rejection reached the director as an observation note, not world state
    assert any("intent rejected" in note for note in director.observations[1].notes)
    assert director.observations[1].configured is None  # bogus configure never became state
    assert director.observations[2].configured == {"integrator": "rk4", "dt": 0.01}
    assert world.get(1, "sim_config") == {"integrator": "rk4", "dt": 0.01}
    assert any(e.kind == "intent.rejected" for e in eventlog.read())
    assert outcome.concluded and outcome.results[0].passed
    eventlog.close()


def test_run_before_configure_is_rejected(tmp_path: Path) -> None:
    class EagerDirector:
        def __init__(self) -> None:
            self.calls = 0

        def decide(self, observation: Observation) -> Intent:
            self.calls += 1
            if self.calls == 1:
                return Intent(action="run", args={"duration": 5.0})
            return Intent(action="conclude", args={"summary": "gave up"})

    eventlog = EventLog(tmp_path / "events.db")
    world = _seeded(eventlog)
    outcome = _engine(world, eventlog, EagerDirector(), max_rounds=3).run()
    assert not outcome.results  # the run never executed
    assert any(e.kind == "intent.rejected" for e in eventlog.read())
    assert outcome.concluded
    eventlog.close()
