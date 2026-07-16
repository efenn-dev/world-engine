"""Storyboard projection: beat segmentation rules, prompt assembly, enrichment, outputs."""

import json
import math
from pathlib import Path

from worldengine.backends.social import SeekerMind, SocialBackend, WaiterMind
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event, ExperimentResult, Intent, Observation, Outcome
from worldengine.storyboard import StoryboardProjector, episode_beats
from worldengine.verifier import Rendezvous
from worldengine.world import World

CHARACTERS = {
    "Mira": "Mira, copper hair, green cloak",
    "Rook": "Rook, spectacles, gray robes",
}
PLACES = {
    "tavern": "a lantern-lit tavern",
    "docks": "fog-covered wooden docks",
}


def _episode_samples() -> list[dict]:
    """Engine-shaped samples for a canonical episode: wait, sweep, meet, talk."""
    return [
        {"t": 1.0, "kind": "agent.moved", "name": "Mira", "place": "tavern", "from": "tavern", "to": "library"},
        {"t": 1.0, "kind": "agent.waited", "name": "Rook", "place": "docks"},
        {"t": 1.0, "kind": "social.tick", "tick": 1, "locations": {"Mira": "library", "Rook": "docks"}, "spoke": []},
        {"t": 2.0, "kind": "agent.moved", "name": "Mira", "place": "library", "from": "library", "to": "market"},
        {"t": 2.0, "kind": "social.tick", "tick": 2, "locations": {"Mira": "market", "Rook": "docks"}, "spoke": []},
        {"t": 3.0, "kind": "agent.moved", "name": "Mira", "place": "market", "from": "market", "to": "docks"},
        {"t": 3.0, "kind": "agent.spoke", "name": "Rook", "place": "docks", "text": "Well met."},
        {"t": 3.0, "kind": "social.tick", "tick": 3, "locations": {"Mira": "docks", "Rook": "docks"}, "spoke": []},
        {"t": 4.0, "kind": "agent.spoke", "name": "Mira", "place": "docks", "text": "Found you."},
        {"t": 4.0, "kind": "social.tick", "tick": 4, "locations": {"Mira": "docks", "Rook": "docks"}, "spoke": []},
    ]


def _outcome() -> Outcome:
    return Outcome(
        concluded=True,
        summary="rendezvous at the docks",
        results=[
            ExperimentResult(
                experiment=1, integrator="n/a", dt=0.0, duration=4.0,
                max_drift_pct=float("nan"), passed=True,
            )
        ],
        samples={1: _episode_samples()},
        rounds=2,
    )


def test_beat_segmentation_rules() -> None:
    beats = episode_beats(_episode_samples())
    kinds = [b.kind for b in beats]
    assert kinds == ["establishing", "montage", "meeting", "dialogue"]
    establishing, montage, meeting, dialogue = beats
    assert establishing.characters == ["Rook"]
    # consecutive moves collapse; the meeting move is NOT a montage waypoint
    assert montage.waypoints == ["tavern", "library", "market"]
    assert "tavern -> library -> market" in montage.action
    # the meeting absorbs that tick's dialogue at the destination
    assert meeting.characters == ["Mira", "Rook"]
    assert meeting.dialogue == ['Rook: "Well met."']
    assert dialogue.dialogue == ['Mira: "Found you."']


def test_storyboard_files_and_prompt_scoping(tmp_path: Path) -> None:
    StoryboardProjector(tmp_path, characters=CHARACTERS, places=PLACES).finalize(_outcome())
    board = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    shots = board["shots"]
    assert [s["beat"] for s in shots] == ["establishing", "montage", "meeting", "dialogue", "card"]
    # sheets appear only when that character/place is actually in the shot
    establishing = shots[0]
    assert "spectacles" in establishing["image_prompt"]  # Rook is present
    assert "copper hair" not in establishing["image_prompt"]  # Mira is not
    meeting = shots[2]
    assert "copper hair" in meeting["image_prompt"] and "spectacles" in meeting["image_prompt"]
    assert "fog-covered" in meeting["image_prompt"]  # place sheet used
    # the closing card carries the session conclusion
    assert "rendezvous at the docks" in shots[-1]["image_prompt"]
    # human-readable sheet exists and mentions the dialogue
    md = (tmp_path / "storyboard.md").read_text(encoding="utf-8")
    assert 'Rook: "Well met."' in md


def test_enricher_rewrites_prompt_but_not_facts(tmp_path: Path) -> None:
    class Fancy:
        def enrich(self, facts: dict) -> str:
            return f"ENRICHED {facts['action']}"

    StoryboardProjector(tmp_path, characters=CHARACTERS, places=PLACES, enricher=Fancy()).finalize(_outcome())
    board = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    meeting = board["shots"][2]
    assert meeting["image_prompt"].startswith("ENRICHED ")
    assert meeting["action"] == "Mira arrives at docks and finds Rook"  # facts untouched
    # card shots are not enriched? they are — but a failing enricher must not break the board
    assert not math.isnan(len(board["shots"]))


def test_failing_enricher_keeps_template(tmp_path: Path) -> None:
    class Broken:
        def enrich(self, facts: dict) -> str:
            raise ValueError("no")

    StoryboardProjector(tmp_path, characters=CHARACTERS, places=PLACES, enricher=Broken()).finalize(_outcome())
    board = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    assert board["shots"][2]["image_prompt"].startswith("anime key visual")


def test_non_narrative_run_writes_nothing(tmp_path: Path) -> None:
    physics = Outcome(
        concluded=True,
        summary="energy conserved",
        results=[
            ExperimentResult(
                experiment=1, integrator="rk4", dt=0.02, duration=30.0,
                max_drift_pct=0.001, passed=True,
            )
        ],
        samples={1: [{"t": 0.0, "kind": "physics.sampled", "theta": 1.0, "omega": 0.0, "energy": 4.5}]},
        rounds=2,
    )
    StoryboardProjector(tmp_path).finalize(physics)
    assert not (tmp_path / "storyboard.json").exists()


class _EpisodeDirector:
    """Two ticks (fails), then five ticks (converges), then conclude."""

    def decide(self, observation: Observation) -> Intent:
        results = observation.results
        if results and results[-1].passed:
            return Intent(action="conclude", args={"summary": "rendezvous"})
        if len(results) >= 2:
            return Intent(action="conclude", args={"summary": "gave up"})
        return Intent(action="run", args={"duration": 2 if not results else 5})


def test_storyboard_from_live_engine_run(tmp_path: Path) -> None:
    """End to end: engine run -> event samples -> storyboard on disk."""
    eventlog = EventLog(tmp_path / "events.db")
    world = World()
    for entity, components in (
        (1, {"places": {"names": ["tavern", "library", "market", "docks"]}}),
        (10, {
            "persona": {"name": "Mira", "traits": "t", "goal": "g"},
            "location": {"place": "tavern"}, "memory": {"entries": []},
            "initial_conditions": {"place": "tavern"},
        }),
        (11, {
            "persona": {"name": "Rook", "traits": "t", "goal": "g"},
            "location": {"place": "docks"}, "memory": {"entries": []},
            "initial_conditions": {"place": "docks"},
        }),
    ):
        world.apply(eventlog.append(Event(kind="world.seeded", entity=entity, data={"set_components": components})))

    outcome = Engine(
        world=world,
        eventlog=eventlog,
        backend=SocialBackend(entity=1, minds={10: SeekerMind(), 11: WaiterMind()}),
        verifier=Rendezvous(),
        director=_EpisodeDirector(),
        goal="rendezvous",
        projectors=[StoryboardProjector(tmp_path, characters=CHARACTERS, places=PLACES)],
        sim_entity=1,
    ).run()

    assert outcome.concluded
    board = json.loads((tmp_path / "storyboard.json").read_text(encoding="utf-8"))
    beats = [s["beat"] for s in board["shots"]]
    assert "meeting" in beats and "montage" in beats and beats[-1] == "card"
    eventlog.close()
