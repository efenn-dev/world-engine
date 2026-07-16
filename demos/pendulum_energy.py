"""Demo: the Director hunts for integrator settings that conserve pendulum energy.

The verifier (not the Director) judges every run against ground truth, so the
loop being exercised is: observe → decide → validate → advance truth → verify.

Usage:
    python demos/pendulum_energy.py                    # deterministic scripted director
    python demos/pendulum_energy.py --director llm     # decisions via llm-router (:4000)

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
from worldengine.director import LLMDirector, ScriptedDirector
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


def build_parser() -> argparse.ArgumentParser:
    """CLI for the pendulum-energy hypothesis loop."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--director", choices=("scripted", "llm"), default="scripted")
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
    eventlog = EventLog(out_dir / "events.db")
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

    if args.director == "llm":
        director = LLMDirector(base_url=args.llm_base, model=args.llm_model, route=args.llm_route)
    else:
        director = ScriptedDirector(run_duration=args.duration)

    engine = Engine(
        world=world,
        eventlog=eventlog,
        backend=PendulumBackend(entity=1),
        verifier=EnergyConservation(tolerance_pct=args.tolerance),
        director=director,
        goal=GOAL,
        projectors=[TextProjector(), PlotProjector(out_dir)],
        max_rounds=args.rounds,
    )
    outcome = engine.run()

    # Event-sourcing proof: state rebuilt from the log alone must match live state.
    replayed = fold(eventlog.read())
    replay_ok = replayed.get(1, "pendulum_state") == world.get(1, "pendulum_state")
    log.info("replay check: %s (fold(event log) == live state)", "OK" if replay_ok else "MISMATCH")
    log.info("event log: %d events at %s", eventlog.last_seq(), eventlog.path)
    eventlog.close()

    success = outcome.concluded and any(r.passed for r in outcome.results) and replay_ok
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
