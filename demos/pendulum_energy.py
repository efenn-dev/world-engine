"""Demo: the Director hunts for integrator settings that conserve pendulum energy.

The verifier (not the Director) judges every run against ground truth, so the
loop being exercised is: observe → decide → validate → advance truth → verify.

Usage:
    python demos/pendulum_energy.py                    # deterministic scripted director
    python demos/pendulum_energy.py --director llm     # decisions via llm-router (:4000)
    python demos/pendulum_energy.py --backend mujoco   # MuJoCo as ground truth

The --backend switch is the whole thesis: swapping truth engines changes only
the backend construction and the seed components below — the engine, contract,
verifier and directors are untouched.

Exit code 0 = the session concluded with a passing configuration and the
event-log replay check held.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldengine.backends.pendulum import PendulumBackend
from worldengine.director import PENDULUM_PLAN, LLMDirector, ScriptedDirector
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event
from worldengine.projector import PlotProjector, TextProjector
from worldengine.verifier import EnergyConservation
from worldengine.world import World, fold

log = logging.getLogger("demo.pendulum_energy")

GOAL = (
    "Find integrator settings under which the pendulum's total energy is conserved "
    "within the verifier's tolerance over a full run. The true dynamics conserve "
    "energy exactly; any drift is integrator error."
)

#: hypothesis arc for the MuJoCo backend (drift numbers measured on assets/pendulum.xml)
MUJOCO_PLAN = (
    ("euler", 0.02, "baseline: MuJoCo's default semi-implicit Euler at coarse dt"),
    ("implicitfast", 0.02, "hypothesis: implicit velocity integration will damp the drift"),
    ("rk4", 0.02, "hypothesis: drift is truncation order — jump to 4th-order Runge-Kutta"),
)


def build_backend(backend_name: str, entity: int = 1):
    """Return (backend, seed components, scripted plan) for the chosen truth engine.

    This factory is the only backend-specific code in the demo — everything
    downstream (engine, contract, verifier, directors) is shared.
    """
    if backend_name == "mujoco":
        from worldengine.backends.mujoco_backend import MuJoCoBackend

        model_path = Path(__file__).resolve().parents[1] / "assets" / "pendulum.xml"
        seed = {
            "initial_conditions": {"qpos": [1.0], "qvel": [0.0]},
            "mujoco_state": {"qpos": [1.0], "qvel": [0.0]},
        }
        return MuJoCoBackend(entity=entity, model_path=model_path), seed, MUJOCO_PLAN
    seed = {
        "pendulum_params": {"g": 9.81, "length": 1.0, "mass": 1.0},
        "initial_conditions": {"theta": 1.0, "omega": 0.0},
        "pendulum_state": {"theta": 1.0, "omega": 0.0},
    }
    return PendulumBackend(entity=entity), seed, PENDULUM_PLAN


def build_parser() -> argparse.ArgumentParser:
    """CLI for the pendulum-energy hypothesis loop."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--director", choices=("scripted", "llm"), default="scripted")
    parser.add_argument("--backend", choices=("pendulum", "mujoco"), default="pendulum")
    parser.add_argument("--rounds", type=int, default=12, help="max director rounds")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds per experiment run")
    parser.add_argument("--tolerance", type=float, default=1.0, help="energy drift tolerance, percent")
    parser.add_argument("--out", type=Path, default=None, help="run directory (default runs/<timestamp>)")
    parser.add_argument("--llm-base", default=None, help="OpenAI-compatible base URL (default: llm-router)")
    parser.add_argument("--llm-model", default=None, help="model name passed through to the gateway")
    parser.add_argument("--llm-route", default=None, help="llm-router force-pin route (e.g. local_reason)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one director-over-truth session and verify replayability."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = args.out or Path(__file__).resolve().parents[1] / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    backend, seed, plan = build_backend(args.backend)
    eventlog = EventLog(out_dir / "events.db")
    world = World()
    world.apply(eventlog.append(Event(kind="world.seeded", entity=1, data={"set_components": seed})))

    if args.director == "llm":
        director = LLMDirector(base_url=args.llm_base, model=args.llm_model, route=args.llm_route)
    else:
        director = ScriptedDirector(run_duration=args.duration, plan=plan)

    engine = Engine(
        world=world,
        eventlog=eventlog,
        backend=backend,
        verifier=EnergyConservation(tolerance_pct=args.tolerance),
        director=director,
        goal=GOAL,
        projectors=[TextProjector(), PlotProjector(out_dir)],
        max_rounds=args.rounds,
    )
    outcome = engine.run()

    # Event-sourcing proof: state rebuilt from the log alone must match live state.
    state_component = "mujoco_state" if args.backend == "mujoco" else "pendulum_state"
    replayed = fold(eventlog.read())
    replay_ok = replayed.get(1, state_component) == world.get(1, state_component)
    log.info("replay check: %s (fold(event log) == live state)", "OK" if replay_ok else "MISMATCH")
    log.info("event log: %d events at %s", eventlog.last_seq(), eventlog.path)
    eventlog.close()

    success = outcome.concluded and any(r.passed for r in outcome.results) and replay_ok
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
