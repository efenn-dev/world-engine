"""MuJoCo reality backend — a real physics engine behind the same Backend protocol.

Loads any MJCF model; the entity's components hold the generalized state:
  - ``mujoco_state``: {qpos: [...], qvel: [...]}
  - ``initial_conditions``: {qpos: [...], qvel: [...]} (used by `reset`)
  - ``sim_config``: {integrator, dt} (set by a validated configure intent)

Integrators are MuJoCo's own (`euler` is semi-implicit, `implicit`/
`implicitfast` are implicit-in-velocity, `rk4` is 4th-order Runge-Kutta) —
advertised via ``integrators`` so the contract and the Director's observation
pick them up without any engine changes.

Energy is computed by MuJoCo (`mj_energyPos` + `mj_energyVel`); note its
potential-energy zero is arbitrary, so total energy can be negative — the
relative-drift verifier only assumes |E0| is not tiny.

Requires ``pip install mujoco`` (imported lazily so the package works without it).
"""

from __future__ import annotations

from pathlib import Path

from ..models import Event
from ..world import World


class MuJoCoBackend:
    """Advances an MJCF model deterministically per the entity's ``sim_config``."""

    name = "mujoco"
    integrators: tuple[str, ...] = ("euler", "implicitfast", "implicit", "rk4")

    def __init__(self, entity: int, model_path: str | Path, sample_every: float = 0.1) -> None:
        import mujoco  # lazy: only this backend needs it

        self._mujoco = mujoco
        self.entity = entity
        self.sample_every = sample_every
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._integrator_ids = {
            "euler": mujoco.mjtIntegrator.mjINT_EULER,
            "implicitfast": mujoco.mjtIntegrator.mjINT_IMPLICITFAST,
            "implicit": mujoco.mjtIntegrator.mjINT_IMPLICIT,
            "rk4": mujoco.mjtIntegrator.mjINT_RK4,
        }

    def reset(self, world: World) -> list[Event]:
        """Restore mujoco_state from initial_conditions so runs are comparable."""
        init = world.get(self.entity, "initial_conditions")
        if init is None:
            raise ValueError(f"entity {self.entity} has no initial_conditions component")
        return [
            Event(
                t=0.0,
                kind="physics.reset",
                entity=self.entity,
                data={
                    "set_components": {
                        "mujoco_state": {"qpos": list(init["qpos"]), "qvel": list(init["qvel"])}
                    }
                },
            )
        ]

    def advance(self, world: World, duration: float) -> list[Event]:
        """Integrate forward ``duration`` seconds; emit one physics.sampled event per sample."""
        mujoco = self._mujoco
        state = world.get(self.entity, "mujoco_state")
        cfg = world.get(self.entity, "sim_config")
        if state is None or cfg is None:
            raise ValueError(f"entity {self.entity} is missing mujoco_state/sim_config")
        integrator = cfg["integrator"]
        if integrator not in self._integrator_ids:
            raise ValueError(f"unknown MuJoCo integrator {integrator!r}; choose one of {list(self.integrators)}")

        self.model.opt.timestep = float(cfg["dt"])
        self.model.opt.integrator = self._integrator_ids[integrator]
        self.data.qpos[:] = state["qpos"]
        self.data.qvel[:] = state["qvel"]
        self.data.time = 0.0
        mujoco.mj_forward(self.model, self.data)

        dt = float(cfg["dt"])
        steps = max(1, round(duration / dt))
        sample_stride = max(1, round(self.sample_every / dt))
        events: list[Event] = [self._sample(0.0)]
        for i in range(1, steps + 1):
            mujoco.mj_step(self.model, self.data)
            if i % sample_stride == 0 or i == steps:
                events.append(self._sample(i * dt))
        return events

    def _sample(self, t: float) -> Event:
        mujoco = self._mujoco
        mujoco.mj_energyPos(self.model, self.data)
        mujoco.mj_energyVel(self.model, self.data)
        qpos = [float(v) for v in self.data.qpos]
        qvel = [float(v) for v in self.data.qvel]
        return Event(
            t=t,
            kind="physics.sampled",
            entity=self.entity,
            data={
                "qpos": qpos,
                "qvel": qvel,
                "energy": float(self.data.energy[0] + self.data.energy[1]),
                "set_components": {"mujoco_state": {"qpos": qpos, "qvel": qvel}},
            },
        )
