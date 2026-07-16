"""Social backend: scripted-mind convergence, perception scoping, determinism, engine loop."""

from pathlib import Path

from worldengine.backends.social import (
    AgentAction,
    AgentView,
    SeekerMind,
    SocialBackend,
    WaiterMind,
)
from worldengine.director import Director  # noqa: F401  (protocol sanity import)
from worldengine.engine import Engine
from worldengine.eventlog import EventLog
from worldengine.models import Event, Intent, Observation
from worldengine.verifier import Rendezvous
from worldengine.world import World, fold

PLACES = ["tavern", "library", "market", "docks"]
WORLD_ENTITY, A, B = 1, 10, 11


def _seeded_world() -> World:
    world = World()
    world.set(WORLD_ENTITY, "places", {"names": list(PLACES)})
    for agent, name, place in ((A, "Mira", "tavern"), (B, "Rook", "docks")):
        world.set(agent, "persona", {"name": name, "traits": "t", "goal": "g"})
        world.set(agent, "location", {"place": place})
        world.set(agent, "memory", {"entries": []})
        world.set(agent, "initial_conditions", {"place": place})
    return world


def _engine_samples(events: list[Event]) -> list[dict]:
    """Shape events the way Engine._run_experiment does for the verifier."""
    return [
        {"t": e.t, "kind": e.kind, **{k: v for k, v in e.data.items() if k != "set_components"}}
        for e in events
    ]


def test_seeker_finds_waiter() -> None:
    world = _seeded_world()
    backend = SocialBackend(entity=WORLD_ENTITY, minds={A: SeekerMind(), B: WaiterMind()})
    events = backend.advance(world, duration=len(PLACES))
    assert world.get(A, "location") == world.get(B, "location") == {"place": "docks"}
    verdict = Rendezvous().check(_engine_samples(events))
    assert verdict.passed
    assert "docks" in verdict.reason


def test_speech_is_room_scoped() -> None:
    class Speaker:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="speak", text="Hello?")

    class Idler:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="wait")

    world = _seeded_world()  # A in tavern, B in docks
    backend = SocialBackend(entity=WORLD_ENTITY, minds={A: Speaker(), B: Idler()})
    backend.advance(world, duration=1)
    assert world.get(B, "memory") == {"entries": []}  # different room: heard nothing

    world.set(B, "location", {"place": "tavern"})  # same room now
    backend.advance(world, duration=1)
    entries = world.get(B, "memory")["entries"]
    assert any('Mira said: "Hello?"' in entry for entry in entries)


def test_invalid_action_becomes_confusion_not_state() -> None:
    class Lost:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="move", to="castle")

    class Idler:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="wait")

    world = _seeded_world()
    backend = SocialBackend(entity=WORLD_ENTITY, minds={A: Lost(), B: Idler()})
    events = backend.advance(world, duration=1)
    assert world.get(A, "location") == {"place": "tavern"}  # unchanged
    assert any(e.kind == "agent.confused" and e.entity == A for e in events)
    # ... and the agent is told, through its own memory
    assert any("your action failed" in entry for entry in world.get(A, "memory")["entries"])


def test_own_actions_land_in_memory() -> None:
    class Mover:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="move", to="library")

    class Idler:
        def decide(self, view: AgentView) -> AgentAction:
            return AgentAction(action="wait")

    world = _seeded_world()
    SocialBackend(entity=WORLD_ENTITY, minds={A: Mover(), B: Idler()}).advance(world, duration=1)
    entries = world.get(A, "memory")["entries"]
    assert any("you moved from tavern to library" in entry for entry in entries)


def test_advance_is_deterministic() -> None:
    events_a = SocialBackend(
        entity=WORLD_ENTITY, minds={A: SeekerMind(), B: WaiterMind()}
    ).advance(_seeded_world(), duration=4)
    events_b = SocialBackend(
        entity=WORLD_ENTITY, minds={A: SeekerMind(), B: WaiterMind()}
    ).advance(_seeded_world(), duration=4)
    assert [(e.kind, e.entity, e.data) for e in events_a] == [
        (e.kind, e.entity, e.data) for e in events_b
    ]


def test_reset_restores_places_and_clears_memory() -> None:
    world = _seeded_world()
    backend = SocialBackend(entity=WORLD_ENTITY, minds={A: SeekerMind(), B: WaiterMind()})
    backend.advance(world, duration=len(PLACES))
    assert world.get(A, "location") != {"place": "tavern"}
    for event in backend.reset(world):
        world.apply(event)
    assert world.get(A, "location") == {"place": "tavern"}
    assert world.get(B, "location") == {"place": "docks"}
    assert world.get(A, "memory") == {"entries": []}


class EpisodeDirector:
    """Run 2 ticks (fails), then enough ticks to converge, then conclude."""

    def decide(self, observation: Observation) -> Intent:
        results = observation.results
        if results and results[-1].passed:
            return Intent(action="conclude", args={"summary": "rendezvous"})
        if len(results) >= 2:
            return Intent(action="conclude", args={"summary": "gave up"})
        ticks = 2 if not results else len(PLACES) + 1
        return Intent(action="run", args={"duration": ticks})


def test_full_engine_loop_on_social(tmp_path: Path) -> None:
    """The whole engine runs unchanged on the social backend (the swap thesis)."""
    eventlog = EventLog(tmp_path / "events.db")
    world = World()
    for entity, components in (
        (WORLD_ENTITY, {"places": {"names": list(PLACES)}}),
        (A, {
            "persona": {"name": "Mira", "traits": "t", "goal": "g"},
            "location": {"place": "tavern"}, "memory": {"entries": []},
            "initial_conditions": {"place": "tavern"},
        }),
        (B, {
            "persona": {"name": "Rook", "traits": "t", "goal": "g"},
            "location": {"place": "docks"}, "memory": {"entries": []},
            "initial_conditions": {"place": "docks"},
        }),
    ):
        world.apply(eventlog.append(Event(kind="world.seeded", entity=entity, data={"set_components": components})))

    outcome = Engine(
        world=world,
        eventlog=eventlog,
        backend=SocialBackend(entity=WORLD_ENTITY, minds={A: SeekerMind(), B: WaiterMind()}),
        verifier=Rendezvous(),
        director=EpisodeDirector(),
        goal="get the agents together",
        sim_entity=WORLD_ENTITY,
    ).run()

    assert outcome.concluded and outcome.summary == "rendezvous"
    assert [r.passed for r in outcome.results] == [False, True]
    # run needed no configure on an integrator-less backend
    assert not any(e.kind == "intent.rejected" for e in eventlog.read())
    # replayability holds with agent minds in the loop
    replayed = fold(eventlog.read())
    for agent in (A, B):
        for component in ("location", "memory"):
            assert replayed.get(agent, component) == world.get(agent, component)
    eventlog.close()
