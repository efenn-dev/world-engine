"""world_dream — run one grounded physics episode and emit a structured verdict.

The Director (by default the local llm-router kernel) is handed a physics
scenario, proposes integrator settings, and a deterministic verifier judges
energy conservation against ground truth. This is the same
`observe -> decide -> validate -> advance -> verify` loop as
`demos/pendulum_energy.py`, but headless and machine-readable: it reads a
scenario-seed JSON and prints ONE JSON result object to stdout (proposed
intents + the verifier's verdict), so an outside program (the kernel's
alive-mode `world_dream` job) can schedule dreams and bank the ground-truth
outcomes as self-play data.

    python scripts/world_dream.py --seed '{"id":"p1","backend":"pendulum",...}'
    echo '<seed json>' | python scripts/world_dream.py
    python scripts/world_dream.py            # a sensible default pendulum seed

stdout: a single JSON object (the result). stderr: logs. Exit 0 if the episode
ran (whether or not it found a passing configuration), 1 on a setup error.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldengine.backends.pendulum import PendulumBackend
from worldengine.director import LLMDirector, ScriptedDirector
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event, Intent, Observation
from worldengine.verifier import EnergyConservation
from worldengine.world import World

log = logging.getLogger("world_dream")

GOAL = (
    "Find integrator settings under which the pendulum's total energy is conserved "
    "within the verifier's tolerance over a full run. The true dynamics conserve "
    "energy exactly; any drift is integrator error."
)

#: default scenario when none is supplied — the canonical pendulum challenge
DEFAULT_SEED = {
    "id": "pendulum-default",
    "backend": "pendulum",
    "tolerance": 1.0,
    "duration": 30.0,
    "rounds": 12,
    "pendulum_params": {"g": 9.81, "length": 1.0, "mass": 1.0},
    "initial_conditions": {"theta": 1.0, "omega": 0.0},
}


class _RecordingDirector:
    """Wraps a Director, recording every proposed intent and never letting a
    director failure abort the episode — a failure is recorded and turned into
    a graceful `conclude` so `Engine.run()` always returns an Outcome we can
    report on (the whole point is banking the outcome, success or not)."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.intents: list[dict] = []
        self.error: str | None = None

    def decide(self, observation: Observation) -> Intent:
        try:
            intent = self.inner.decide(observation)
        except Exception as exc:  # noqa: BLE001 — director/LLM failure must not crash the dream
            self.error = f"{type(exc).__name__}: {exc}"
            log.warning("director failed: %s", self.error)
            intent = Intent(action="conclude",
                            args={"summary": "director failed to produce a valid intent"},
                            rationale="director error")
        self.intents.append(intent.model_dump())
        return intent


def _build_backend(seed: dict):
    """Return (backend, seed_components) for the scenario. Phase 1 = pendulum."""
    backend_name = seed.get("backend", "pendulum")
    if backend_name != "pendulum":
        raise ValueError(f"unsupported backend {backend_name!r} (Phase 1 is pendulum-only)")
    params = seed.get("pendulum_params", DEFAULT_SEED["pendulum_params"])
    ic = seed.get("initial_conditions", DEFAULT_SEED["initial_conditions"])
    components = {
        "pendulum_params": params,
        "initial_conditions": ic,
        "pendulum_state": dict(ic),
    }
    return PendulumBackend(entity=1), components


def run_dream(seed: dict, director_kind: str, llm: dict, out: Path | None) -> dict:
    """Run one episode and return a structured, JSON-serializable result."""
    tolerance = float(seed.get("tolerance", 1.0))
    duration = float(seed.get("duration", 30.0))
    rounds = int(seed.get("rounds", 12))
    backend, components = _build_backend(seed)

    run_dir = out or Path(tempfile.mkdtemp(prefix="world_dream_"))
    eventlog = EventLog(run_dir / "events.db")
    try:
        world = World()
        world.apply(eventlog.append(
            Event(kind="world.seeded", entity=1, data={"set_components": components})))

        if director_kind == "scripted":
            inner = ScriptedDirector(run_duration=duration)
        else:
            inner = LLMDirector(base_url=llm.get("base"), model=llm.get("model"),
                                route=llm.get("route"))
        director = _RecordingDirector(inner)

        engine = Engine(world=world, eventlog=eventlog, backend=backend,
                        verifier=EnergyConservation(tolerance_pct=tolerance),
                        director=director, goal=GOAL, projectors=(), max_rounds=rounds)
        outcome = engine.run()
    finally:
        eventlog.close()
        if out is None:
            shutil.rmtree(run_dir, ignore_errors=True)

    results = [r.model_dump() for r in outcome.results]
    passed = [r for r in results if r["passed"]]
    drifts = [r["max_drift_pct"] for r in results
              if isinstance(r.get("max_drift_pct"), (int, float)) and r["max_drift_pct"] == r["max_drift_pct"]]
    best = min(passed, key=lambda r: r["max_drift_pct"]) if passed else None
    return {
        "id": seed.get("id", "unknown"),
        "backend": seed.get("backend", "pendulum"),
        "tolerance_pct": tolerance,
        "duration_s": duration,
        "director": director_kind,
        "llm": {"model": llm.get("model"), "route": llm.get("route")} if director_kind == "llm" else None,
        "concluded": outcome.concluded,
        "summary": outcome.summary,
        "passed": bool(passed),
        "best": best,
        "min_drift_pct": min(drifts) if drifts else None,
        "rounds_used": outcome.rounds,
        "n_experiments": len(results),
        "results": results,
        "proposed_intents": director.intents,
        "director_error": director.error,
    }


def _load_seed(args) -> dict:
    if args.seed:
        return json.loads(args.seed)
    if args.seed_file:
        return json.loads(Path(args.seed_file).read_text(encoding="utf-8"))
    if not sys.stdin.isatty():
        raw = sys.stdin.read().strip()
        if raw:
            return json.loads(raw)
    return DEFAULT_SEED


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", help="scenario seed as a JSON string")
    ap.add_argument("--seed-file", help="path to a scenario seed JSON file")
    ap.add_argument("--director", choices=("llm", "scripted"), default="llm",
                    help="llm = the kernel decides (default); scripted = deterministic stand-in")
    ap.add_argument("--llm-base", default=None, help="OpenAI-compatible base URL (default llm-router)")
    ap.add_argument("--llm-model", default=None, help="model passed to the gateway (default auto)")
    ap.add_argument("--llm-route", default=None, help="llm-router force-pin route (e.g. local_reason)")
    ap.add_argument("--out", type=Path, default=None,
                    help="persist the run dir (event log) here; default = temp + discard")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)

    try:
        seed = _load_seed(args)
    except (json.JSONDecodeError, OSError) as exc:
        print(json.dumps({"error": f"bad seed: {exc}"}))
        return 1
    llm = {"base": args.llm_base, "model": args.llm_model, "route": args.llm_route}
    try:
        result = run_dream(seed, args.director, llm, args.out)
    except Exception as exc:  # noqa: BLE001 — setup failure: emit a JSON error, exit 1
        print(json.dumps({"id": seed.get("id", "unknown"), "error": f"{type(exc).__name__}: {exc}"}))
        return 1
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
