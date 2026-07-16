"""MuJoCo backend: protocol conformance, energy behavior, determinism.

Skipped entirely if the `mujoco` package is not installed.
"""

from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from worldengine.backends.mujoco_backend import MuJoCoBackend
from worldengine.eventlog import EventLog
from worldengine.models import Event, Intent, Observation
from worldengine.world import World, fold

MODEL_XML = Path(__file__).resolve().parents[1] / "assets" / "pendulum.xml"


def _world(integrator: str, dt: float) -> World:
    world = World()
    world.set(1, "initial_conditions", {"qpos": [1.0], "qvel": [0.0]})
    world.set(1, "mujoco_state", {"qpos": [1.0], "qvel": [0.0]})
    world.set(1, "sim_config", {"integrator": integrator, "dt": dt})
    return world


def _max_drift_pct(events: list[Event]) -> float:
    energies = [e.data["energy"] for e in events]
    e0 = energies[0]
    return max(abs(e - e0) / abs(e0) for e in energies) * 100.0


def test_euler_drifts_at_coarse_dt() -> None:
    events = MuJoCoBackend(1, MODEL_XML).advance(_world("euler", 0.02), duration=30.0)
    assert _max_drift_pct(events) > 1.0


def test_rk4_conserves() -> None:
    events = MuJoCoBackend(1, MODEL_XML).advance(_world("rk4", 0.02), duration=30.0)
    assert _max_drift_pct(events) < 0.01


def test_advance_is_deterministic() -> None:
    a = MuJoCoBackend(1, MODEL_XML).advance(_world("euler", 0.01), duration=10.0)
    b = MuJoCoBackend(1, MODEL_XML).advance(_world("euler", 0.01), duration=10.0)
    assert [e.data for e in a] == [e.data for e in b]


def test_reset_restores_initial_conditions() -> None:
    world = _world("rk4", 0.01)
    backend = MuJoCoBackend(1, MODEL_XML)
    for event in backend.advance(world, duration=5.0):
        world.apply(event)
    assert world.get(1, "mujoco_state") != {"qpos": [1.0], "qvel": [0.0]}
    for event in backend.reset(world):
        world.apply(event)
    assert world.get(1, "mujoco_state") == {"qpos": [1.0], "qvel": [0.0]}


def test_unknown_integrator_raises() -> None:
    with pytest.raises(ValueError, match="unknown MuJoCo integrator"):
        MuJoCoBackend(1, MODEL_XML).advance(_world("symplectic_euler", 0.01), duration=1.0)


def test_full_engine_loop_on_mujoco(tmp_path: Path) -> None:
    """The whole engine runs unchanged on the MuJoCo backend (the swap thesis)."""
    from worldengine.director import ScriptedDirector
    from worldengine.engine import Engine
    from worldengine.verifier import EnergyConservation

    plan = (
        ("euler", 0.02, "baseline"),
        ("rk4", 0.02, "high-order"),
    )
    eventlog = EventLog(tmp_path / "events.db")
    world = World()
    world.apply(
        eventlog.append(
            Event(
                kind="world.seeded",
                entity=1,
                data={
                    "set_components": {
                        "initial_conditions": {"qpos": [1.0], "qvel": [0.0]},
                        "mujoco_state": {"qpos": [1.0], "qvel": [0.0]},
                    }
                },
            )
        )
    )
    outcome = Engine(
        world=world,
        eventlog=eventlog,
        backend=MuJoCoBackend(entity=1, model_path=MODEL_XML),
        verifier=EnergyConservation(tolerance_pct=1.0),
        director=ScriptedDirector(run_duration=30.0, plan=plan),
        goal="conserve energy in MuJoCo within 1%",
    ).run()

    assert outcome.concluded
    assert [r.passed for r in outcome.results] == [False, True]
    assert outcome.results[-1].integrator == "rk4"
    replayed = fold(eventlog.read())
    assert replayed.get(1, "mujoco_state") == world.get(1, "mujoco_state")
    eventlog.close()


def test_observation_advertises_backend_integrators(tmp_path: Path) -> None:
    """Capabilities flow from the backend into the Director's observation."""
    from worldengine.engine import Engine
    from worldengine.verifier import EnergyConservation

    seen: list[Observation] = []

    class ProbeDirector:
        def decide(self, observation: Observation) -> Intent:
            seen.append(observation)
            return Intent(action="conclude", args={"summary": "probe done"})

    eventlog = EventLog(tmp_path / "events.db")
    Engine(
        world=World(),
        eventlog=eventlog,
        backend=MuJoCoBackend(entity=1, model_path=MODEL_XML),
        verifier=EnergyConservation(),
        director=ProbeDirector(),
        goal="probe",
        max_rounds=1,
    ).run()
    assert seen[0].available_integrators == ["euler", "implicitfast", "implicit", "rk4"]
    eventlog.close()
