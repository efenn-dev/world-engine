"""Demo: two agents must find each other — the social backend slice.

Mira (a seeker) and Rook (a waiter) start in different places. Minds act
through room-scoped views; the Rendezvous verifier judges co-location against
ground truth (never against what agents claim). Each `run` is a fresh episode;
the director searches for a sufficient episode length in ticks.

Usage:
    python demos/rendezvous.py                    # scripted minds + scripted director
    python demos/rendezvous.py --minds llm        # each agent is an LLM via llm-router
    python demos/rendezvous.py --director llm     # LLM director orchestrates episodes

Exit code 0 = the session concluded with a rendezvous and the event-log
replay check held. Writes runs/<timestamp>/transcript.txt — the event log
rendered as a script.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from worldengine.backends.social import LLMMind, Mind, SeekerMind, SocialBackend, WaiterMind
from worldengine.director import LLMDirector
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event, Intent, Observation
from worldengine.projector import TextProjector, TranscriptProjector
from worldengine.storyboard import LLMShotEnricher, StoryboardProjector
from worldengine.verifier import Rendezvous
from worldengine.world import World, fold

log = logging.getLogger("demo.rendezvous")

GOAL = (
    "Every agent must end an episode in the same place. Each run starts a fresh "
    "episode; duration is the number of ticks. If the verifier reports no "
    "rendezvous, run again with a larger tick budget. Conclude once it passes."
)

PLACES = ["tavern", "library", "market", "docks"]
WORLD_ENTITY, MIRA, ROOK = 1, 10, 11

#: visual sheets consumed by the StoryboardProjector — style only; presence
#: in any given shot always comes from ground-truth events
CHARACTER_SHEETS = {
    "Mira": "Mira, a young scout with short copper hair, a weathered green cloak, and determined amber eyes",
    "Rook": "Rook, an elderly archivist with short gray hair, round spectacles, ink-stained gray robes, and a gentle smile",
}
PLACE_SHEETS = {
    "tavern": "a warm lantern-lit tavern with rough wooden tables and a crackling hearth",
    "library": "a dusty candlelit library with towering, overloaded shelves",
    "market": "a bustling dusk market under canvas awnings",
    "docks": "fog-covered wooden docks with moored boats and circling gulls",
}
STYLE = "anime key visual, cel shading, cinematic lighting, painterly detailed background"


class EscalatingDirector:
    """Scripted social director: fresh episodes with a doubling tick budget."""

    def __init__(self, start_ticks: int = 2, max_episodes: int = 4) -> None:
        self.start_ticks = start_ticks
        self.max_episodes = max_episodes

    def decide(self, observation: Observation) -> Intent:
        """Run longer episodes until the verifier passes, then conclude."""
        results = observation.results
        if results and results[-1].passed:
            best = results[-1]
            return Intent(
                action="conclude",
                args={
                    "summary": (
                        f"rendezvous achieved in a {best.duration:g}-tick episode "
                        f"(after {len(results)} episodes)"
                    )
                },
                rationale="verifier reports co-location",
            )
        if len(results) >= self.max_episodes:
            return Intent(
                action="conclude",
                args={"summary": "no rendezvous within the episode budget"},
                rationale="patience exhausted",
            )
        ticks = self.start_ticks * (2 ** len(results))
        return Intent(
            action="run",
            args={"duration": ticks},
            rationale="fresh episode with a longer tick budget",
        )


def seed_events() -> list[Event]:
    """World and agent components; every bit of setup flows through events."""
    return [
        Event(
            kind="world.seeded",
            entity=WORLD_ENTITY,
            data={"set_components": {"places": {"names": PLACES}}},
        ),
        Event(
            kind="world.seeded",
            entity=MIRA,
            data={
                "set_components": {
                    "persona": {
                        "name": "Mira",
                        "traits": "a restless scout — methodical, a little impatient",
                        "goal": "Find Rook, who is somewhere in these places, and stay with them.",
                    },
                    "location": {"place": "tavern"},
                    "memory": {"entries": []},
                    "initial_conditions": {"place": "tavern"},
                }
            },
        ),
        Event(
            kind="world.seeded",
            entity=ROOK,
            data={
                "set_components": {
                    "persona": {
                        "name": "Rook",
                        "traits": "a patient archivist — calm, never in a hurry",
                        "goal": "Stay where you are so Mira can find you; greet them when they arrive.",
                    },
                    "location": {"place": "docks"},
                    "memory": {"entries": []},
                    "initial_conditions": {"place": "docks"},
                }
            },
        ),
    ]


def build_parser() -> argparse.ArgumentParser:
    """CLI for the rendezvous demo."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--minds", choices=("scripted", "llm"), default="scripted")
    parser.add_argument("--director", choices=("scripted", "llm"), default="scripted")
    parser.add_argument(
        "--storyboard",
        choices=("template", "llm", "off"),
        default="template",
        help="storyboard prompts: deterministic templates, LLM-enriched, or disabled",
    )
    parser.add_argument("--ticks", type=int, default=2, help="first episode length; doubles per episode")
    parser.add_argument("--episodes", type=int, default=4, help="max episodes before conceding")
    parser.add_argument("--rounds", type=int, default=12, help="max director rounds")
    parser.add_argument("--out", type=Path, default=None, help="run directory (default runs/<timestamp>)")
    parser.add_argument("--llm-base", default=None, help="OpenAI-compatible base URL (default: llm-router)")
    parser.add_argument("--llm-model", default=None, help="model name passed through to the gateway")
    parser.add_argument("--llm-route", default=None, help="llm-router force-pin route (e.g. local_reason)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one social session and verify replayability."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    out_dir = args.out or Path(__file__).resolve().parents[1] / "runs" / datetime.now().strftime("%Y%m%d-%H%M%S")
    eventlog = EventLog(out_dir / "events.db")
    world = World()
    for event in seed_events():
        world.apply(eventlog.append(event))

    if args.minds == "llm":
        minds: dict[int, Mind] = {
            MIRA: LLMMind(base_url=args.llm_base, model=args.llm_model, route=args.llm_route),
            ROOK: LLMMind(base_url=args.llm_base, model=args.llm_model, route=args.llm_route),
        }
    else:
        minds = {MIRA: SeekerMind(), ROOK: WaiterMind()}

    if args.director == "llm":
        director = LLMDirector(base_url=args.llm_base, model=args.llm_model, route=args.llm_route)
    else:
        director = EscalatingDirector(start_ticks=args.ticks, max_episodes=args.episodes)

    projectors = [TextProjector(), TranscriptProjector(out_dir)]
    if args.storyboard != "off":
        enricher = None
        if args.storyboard == "llm":
            enricher = LLMShotEnricher(base_url=args.llm_base, model=args.llm_model, route=args.llm_route)
        projectors.append(
            StoryboardProjector(
                out_dir,
                characters=CHARACTER_SHEETS,
                places=PLACE_SHEETS,
                style=STYLE,
                title="The Rendezvous",
                enricher=enricher,
            )
        )

    engine = Engine(
        world=world,
        eventlog=eventlog,
        backend=SocialBackend(entity=WORLD_ENTITY, minds=minds),
        verifier=Rendezvous(),
        director=director,
        goal=GOAL,
        projectors=projectors,
        max_rounds=args.rounds,
        sim_entity=WORLD_ENTITY,
    )
    outcome = engine.run()

    # Event-sourcing proof: agent state rebuilt from the log alone must match live state.
    replayed = fold(eventlog.read())
    replay_ok = all(
        replayed.get(agent, component) == world.get(agent, component)
        for agent in (MIRA, ROOK)
        for component in ("location", "memory")
    )
    log.info("replay check: %s (fold(event log) == live state)", "OK" if replay_ok else "MISMATCH")
    log.info("event log: %d events at %s", eventlog.last_seq(), eventlog.path)
    eventlog.close()

    success = outcome.concluded and any(r.passed for r in outcome.results) and replay_ok
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
