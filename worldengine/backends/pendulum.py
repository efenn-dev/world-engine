"""Deterministic nonlinear pendulum — ground truth for the energy-conservation slice.

theta'' = -(g / L) * sin(theta), undamped. Total energy
E = 1/2 m (L w)^2 + m g L (1 - cos theta) is conserved by the true dynamics,
so any drift is integrator error — which gives the verifier real teeth:
explicit Euler pumps energy in, symplectic Euler bounds the error, RK4 nearly
eliminates it. Pure Python, no dependencies, fully deterministic.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from ..models import Event
from ..world import World

Stepper = Callable[[float, float, float, float], tuple[float, float]]


def _accel(theta: float, g_over_l: float) -> float:
    return -g_over_l * math.sin(theta)


def _step_explicit_euler(theta: float, omega: float, dt: float, g_over_l: float) -> tuple[float, float]:
    return theta + omega * dt, omega + _accel(theta, g_over_l) * dt


def _step_symplectic_euler(theta: float, omega: float, dt: float, g_over_l: float) -> tuple[float, float]:
    omega = omega + _accel(theta, g_over_l) * dt
    return theta + omega * dt, omega


def _step_rk4(theta: float, omega: float, dt: float, g_over_l: float) -> tuple[float, float]:
    def deriv(th: float, om: float) -> tuple[float, float]:
        return om, _accel(th, g_over_l)

    k1 = deriv(theta, omega)
    k2 = deriv(theta + 0.5 * dt * k1[0], omega + 0.5 * dt * k1[1])
    k3 = deriv(theta + 0.5 * dt * k2[0], omega + 0.5 * dt * k2[1])
    k4 = deriv(theta + dt * k3[0], omega + dt * k3[1])
    theta += dt / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
    omega += dt / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
    return theta, omega


STEPPERS: dict[str, Stepper] = {
    "explicit_euler": _step_explicit_euler,
    "symplectic_euler": _step_symplectic_euler,
    "rk4": _step_rk4,
}


def energy(theta: float, omega: float, g: float, length: float, mass: float) -> float:
    """Total mechanical energy of the pendulum, joules (zero at the hanging rest pose)."""
    return 0.5 * mass * (length * omega) ** 2 + mass * g * length * (1.0 - math.cos(theta))


class PendulumBackend:
    """Advances the pendulum entity deterministically per its ``sim_config``.

    Required components on the entity:
      - ``pendulum_params``: {g, length, mass}
      - ``pendulum_state``: {theta, omega}
      - ``initial_conditions``: {theta, omega} (used by `reset`)
      - ``sim_config``: {integrator, dt} (set by a validated configure intent)
    """

    name = "pendulum"

    def __init__(self, entity: int, sample_every: float = 0.1) -> None:
        self.entity = entity
        self.sample_every = sample_every

    def reset(self, world: World) -> list[Event]:
        """Restore pendulum_state from initial_conditions so runs are comparable."""
        init = world.get(self.entity, "initial_conditions")
        if init is None:
            raise ValueError(f"entity {self.entity} has no initial_conditions component")
        return [
            Event(
                t=0.0,
                kind="physics.reset",
                entity=self.entity,
                data={"set_components": {"pendulum_state": dict(init)}},
            )
        ]

    def advance(self, world: World, duration: float) -> list[Event]:
        """Integrate forward ``duration`` seconds; emit one physics.sampled event per sample."""
        params = world.get(self.entity, "pendulum_params")
        state = world.get(self.entity, "pendulum_state")
        cfg = world.get(self.entity, "sim_config")
        if params is None or state is None or cfg is None:
            raise ValueError(
                f"entity {self.entity} is missing pendulum_params/pendulum_state/sim_config"
            )
        g, length, mass = params["g"], params["length"], params["mass"]
        g_over_l = g / length
        theta, omega = state["theta"], state["omega"]
        dt = float(cfg["dt"])
        stepper = STEPPERS[cfg["integrator"]]

        steps = max(1, round(duration / dt))
        sample_stride = max(1, round(self.sample_every / dt))
        events: list[Event] = [self._sample(0.0, theta, omega, g, length, mass)]
        for i in range(1, steps + 1):
            theta, omega = stepper(theta, omega, dt, g_over_l)
            if i % sample_stride == 0 or i == steps:
                events.append(self._sample(i * dt, theta, omega, g, length, mass))
        return events

    def _sample(self, t: float, theta: float, omega: float, g: float, length: float, mass: float) -> Event:
        return Event(
            t=t,
            kind="physics.sampled",
            entity=self.entity,
            data={
                "theta": theta,
                "omega": omega,
                "energy": energy(theta, omega, g, length, mass),
                "set_components": {"pendulum_state": {"theta": theta, "omega": omega}},
            },
        )
