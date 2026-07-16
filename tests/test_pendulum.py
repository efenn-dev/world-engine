"""Pendulum backend: integrator energy behavior and determinism."""

from worldengine.backends.pendulum import PendulumBackend
from worldengine.models import Event
from worldengine.world import World


def _world(integrator: str, dt: float) -> World:
    world = World()
    world.set(1, "pendulum_params", {"g": 9.81, "length": 1.0, "mass": 1.0})
    world.set(1, "initial_conditions", {"theta": 1.0, "omega": 0.0})
    world.set(1, "pendulum_state", {"theta": 1.0, "omega": 0.0})
    world.set(1, "sim_config", {"integrator": integrator, "dt": dt})
    return world


def _max_drift_pct(events: list[Event]) -> float:
    energies = [e.data["energy"] for e in events]
    e0 = energies[0]
    return max(abs(e - e0) / e0 for e in energies) * 100.0


def test_explicit_euler_pumps_energy() -> None:
    events = PendulumBackend(1).advance(_world("explicit_euler", 0.02), duration=30.0)
    assert _max_drift_pct(events) > 5.0


def test_smaller_dt_does_not_save_explicit_euler() -> None:
    events = PendulumBackend(1).advance(_world("explicit_euler", 0.002), duration=30.0)
    assert _max_drift_pct(events) > 1.0  # still fails a 1% tolerance


def test_symplectic_euler_bounds_drift() -> None:
    events = PendulumBackend(1).advance(_world("symplectic_euler", 0.005), duration=30.0)
    assert _max_drift_pct(events) < 1.0


def test_rk4_nearly_conserves() -> None:
    events = PendulumBackend(1).advance(_world("rk4", 0.02), duration=30.0)
    assert _max_drift_pct(events) < 0.01


def test_advance_is_deterministic() -> None:
    a = PendulumBackend(1).advance(_world("symplectic_euler", 0.01), duration=10.0)
    b = PendulumBackend(1).advance(_world("symplectic_euler", 0.01), duration=10.0)
    assert [e.data for e in a] == [e.data for e in b]


def test_reset_restores_initial_conditions() -> None:
    world = _world("rk4", 0.01)
    backend = PendulumBackend(1)
    for event in backend.advance(world, duration=5.0):
        world.apply(event)
    assert world.get(1, "pendulum_state") != {"theta": 1.0, "omega": 0.0}
    for event in backend.reset(world):
        world.apply(event)
    assert world.get(1, "pendulum_state") == {"theta": 1.0, "omega": 0.0}
