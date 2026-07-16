"""Social reality backend — persona-driven agent minds behind the same Backend protocol.

Track 2 of the world-engine: entities with personas act in a small world of
places. Each tick, every agent receives a *scoped* `AgentView` (its place, who
is present, what it remembers), its `Mind` proposes one typed `AgentAction`,
and the backend validates and folds the result into events — the same
propose → validate → apply shape the Director uses, one level down. An invalid
action becomes an ``agent.confused`` event and a memory entry, never state.

Components:
  world entity: ``places``  {"names": [...]}
  each agent:   ``persona`` {name, traits, goal}, ``location`` {place},
                ``memory``  {entries: [...]}, ``initial_conditions`` {place}

Determinism: with `ScriptedMind`s the backend is fully deterministic; with
`LLMMind`s it is *replay*-deterministic — raw mind output lands in event data
and state changes flow only through set_components, so fold(events)
reproduces state exactly.

Perception is room-scoped: speech and movement are only witnessed by agents in
the same place at the end of the tick (a documented simultaneity fudge, fine at
this slice's fidelity). Agents act sequentially in entity order within a tick,
so later agents see earlier moves.
"""

from __future__ import annotations

import logging
from typing import Literal, Protocol

from pydantic import BaseModel, Field, ValidationError

from ..models import Event
from ..world import World

log = logging.getLogger(__name__)

#: rolling per-agent memory window (entries, not tokens) — the context budget
MEMORY_CAP = 12


class AgentView(BaseModel):
    """The scoped observation a mind receives — never the raw world."""

    tick: int
    name: str
    traits: str
    goal: str
    place: str
    places: list[str]
    others_here: list[str] = Field(default_factory=list)
    memory: list[str] = Field(default_factory=list)


class AgentAction(BaseModel):
    """A typed proposal from a mind — its only write channel."""

    action: Literal["move", "speak", "wait"]
    to: str | None = None
    text: str | None = None
    rationale: str = ""


class Mind(Protocol):
    """Protocol every agent mind implements: one action per tick."""

    def decide(self, view: AgentView) -> AgentAction:
        """Turn a scoped view into one typed action."""
        ...


class SeekerMind:
    """Deterministic: sweep the places in listed order until someone is found."""

    def __init__(self) -> None:
        self._just_spoke = False

    def decide(self, view: AgentView) -> AgentAction:
        """Greet on contact, otherwise walk the place list cyclically."""
        if view.others_here:
            if not self._just_spoke:
                self._just_spoke = True
                return AgentAction(
                    action="speak",
                    text=f"Found you, {view.others_here[0]}!",
                    rationale="goal reached — greet",
                )
            return AgentAction(action="wait", rationale="stay together")
        self._just_spoke = False
        here = view.places.index(view.place) if view.place in view.places else -1
        nxt = view.places[(here + 1) % len(view.places)]
        return AgentAction(action="move", to=nxt, rationale="sweeping the places in order")


class WaiterMind:
    """Deterministic: hold position; greet whoever arrives."""

    def __init__(self) -> None:
        self._just_spoke = False

    def decide(self, view: AgentView) -> AgentAction:
        """Wait in place; speak once per arrival."""
        if view.others_here and not self._just_spoke:
            self._just_spoke = True
            return AgentAction(
                action="speak",
                text=f"Well met, {view.others_here[0]} — I have been here all along.",
                rationale="greet the arrival",
            )
        if not view.others_here:
            self._just_spoke = False
        return AgentAction(action="wait", rationale="hold position so I can be found")


_MIND_PROMPT = """You are {name}, a character in a small simulated world.
Your persona: {traits}
Your goal: {goal}

Each turn you receive a JSON view of what you can perceive (your place, who else is here, what you remember). Respond with EXACTLY one JSON action:
  {{"action": "move", "to": "<one of view.places, not your current place>", "rationale": "<why>"}}
  {{"action": "speak", "text": "<one short line of dialogue>", "rationale": "<why>"}}
  {{"action": "wait", "rationale": "<why>"}}

Rules:
- Speech is heard ONLY by characters in your place; you cannot shout across places.
- Your memory records where you have already moved. To find someone, visit places you have
  NOT yet been to — never bounce back and forth between the same places.
- Once your goal is achieved (e.g. you are together), stay put.
- Respond with ONLY the JSON object — no prose, no code fences, no thinking text."""


class LLMMind:
    """Mind backed by an OpenAI-compatible endpoint (same env config as LLMDirector).

    Fail-safe: a reply that cannot be parsed into an `AgentAction` after one
    retry degrades to ``wait`` (with the error in the rationale) rather than
    crashing a multi-agent tick — one flaky reply should not kill an episode.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        route: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        temperature: float = 0.2,
    ) -> None:
        from ..director import llm_defaults

        self.base_url, self.model, self.route, self.api_key = llm_defaults(
            base_url, model, route, api_key
        )
        self.timeout = timeout
        self.temperature = temperature

    def decide(self, view: AgentView) -> AgentAction:
        """Ask the LLM for one action; degrade to wait on repeated parse failure."""
        from ..director import chat_completion, extract_json

        system = _MIND_PROMPT.format(name=view.name, traits=view.traits, goal=view.goal)
        user = view.model_dump_json(indent=2)
        error_note: str | None = None
        for _ in range(2):
            prompt = user if error_note is None else (
                f"{user}\n\nYour previous reply was invalid: {error_note}\n"
                "Reply with ONLY the JSON action object."
            )
            try:
                content = chat_completion(
                    self.base_url,
                    self.model,
                    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    route=self.route,
                    api_key=self.api_key,
                    timeout=self.timeout,
                    temperature=self.temperature,
                )
                return AgentAction.model_validate_json(extract_json(content))
            except (ValidationError, ValueError, OSError) as exc:
                error_note = str(exc)
                log.warning("%s: mind reply invalid, retrying: %s", view.name, exc)
        return AgentAction(action="wait", rationale=f"(mind error: {error_note})")


class SocialBackend:
    """Advances a world of persona-driven agents one tick at a time.

    ``advance(world, duration)`` treats duration as a number of ticks. It
    applies its own events to ``world`` as it generates them (so later agents
    in a tick see earlier moves); events are idempotent set_components, so the
    engine's subsequent fold is a harmless re-apply.
    """

    name = "social"
    integrators: tuple[str, ...] = ()  # nothing to configure — run directly

    def __init__(self, entity: int, minds: dict[int, Mind]) -> None:
        self.entity = entity
        self.minds = dict(minds)

    def reset(self, world: World) -> list[Event]:
        """Restore every agent to its initial place with an empty memory."""
        events: list[Event] = []
        for agent in sorted(self.minds):
            init = world.get(agent, "initial_conditions")
            if init is None:
                raise ValueError(f"agent {agent} has no initial_conditions component")
            events.append(
                Event(
                    t=0.0,
                    kind="social.reset",
                    entity=agent,
                    data={
                        "set_components": {
                            "location": {"place": init["place"]},
                            "memory": {"entries": []},
                        }
                    },
                )
            )
        return events

    def advance(self, world: World, duration: float) -> list[Event]:
        """Run ``round(duration)`` ticks; return every action/perception/tick event."""
        places_component = world.get(self.entity, "places")
        if places_component is None:
            raise ValueError(f"entity {self.entity} has no places component")
        places = list(places_component["names"])
        for agent in self.minds:
            if world.get(agent, "persona") is None or world.get(agent, "location") is None:
                raise ValueError(f"agent {agent} is missing persona/location components")

        events: list[Event] = []
        ticks = max(1, round(duration))
        for tick in range(1, ticks + 1):
            tick_events: list[Event] = []
            for agent in sorted(self.minds):
                view = self._view(world, tick, agent, places)
                action = self.minds[agent].decide(view)
                event = self._action_event(tick, agent, view, action, places)
                world.apply(event)
                tick_events.append(event)
            events.extend(tick_events)
            events.extend(self._perceive(world, tick, tick_events))
            events.append(self._tick_event(world, tick, tick_events))
        return events

    def _view(self, world: World, tick: int, agent: int, places: list[str]) -> AgentView:
        persona = world.get(agent, "persona") or {}
        place = (world.get(agent, "location") or {}).get("place", "")
        others = [
            (world.get(other, "persona") or {}).get("name", str(other))
            for other in sorted(self.minds)
            if other != agent and (world.get(other, "location") or {}).get("place") == place
        ]
        memory = (world.get(agent, "memory") or {}).get("entries", [])
        return AgentView(
            tick=tick,
            name=persona.get("name", str(agent)),
            traits=persona.get("traits", ""),
            goal=persona.get("goal", ""),
            place=place,
            places=places,
            others_here=others,
            memory=list(memory),
        )

    def _action_event(
        self, tick: int, agent: int, view: AgentView, action: AgentAction, places: list[str]
    ) -> Event:
        """Validate one action; valid ones become state, invalid ones become confusion."""
        base = {"name": view.name, "place": view.place, "rationale": action.rationale}
        if action.action == "move":
            if action.to in places and action.to != view.place:
                return Event(
                    t=float(tick),
                    kind="agent.moved",
                    entity=agent,
                    data={
                        **base,
                        "from": view.place,
                        "to": action.to,
                        "set_components": {"location": {"place": action.to}},
                    },
                )
            reason = f"cannot move to {action.to!r}; valid places: {places}"
        elif action.action == "speak":
            text = (action.text or "").strip()
            if text:
                return Event(
                    t=float(tick), kind="agent.spoke", entity=agent, data={**base, "text": text}
                )
            reason = "speak requires non-empty text"
        else:
            return Event(t=float(tick), kind="agent.waited", entity=agent, data=base)
        return Event(
            t=float(tick),
            kind="agent.confused",
            entity=agent,
            data={**base, "reason": reason, "attempted": action.model_dump()},
        )

    def _perceive(self, world: World, tick: int, tick_events: list[Event]) -> list[Event]:
        """Fold what each agent witnessed this tick into its rolling memory."""
        events: list[Event] = []
        for agent in sorted(self.minds):
            place = (world.get(agent, "location") or {}).get("place")
            entries: list[str] = []
            for event in tick_events:
                data = event.data
                if event.kind == "agent.moved" and event.entity == agent:
                    # own travel history — without it an agent cannot "avoid repeats"
                    entries.append(f"tick {tick}: you moved from {data['from']} to {data['to']}")
                elif event.kind == "agent.moved":
                    if data["to"] == place:
                        entries.append(f"tick {tick}: {data['name']} arrived at {place}")
                    elif data["from"] == place:
                        entries.append(f"tick {tick}: {data['name']} left for {data['to']}")
                elif event.kind == "agent.spoke" and event.entity == agent:
                    entries.append(f"tick {tick}: you said: \"{data['text']}\"")
                elif event.kind == "agent.spoke" and data["place"] == place:
                    entries.append(f"tick {tick}: {data['name']} said: \"{data['text']}\"")
                elif event.kind == "agent.confused" and event.entity == agent:
                    entries.append(f"tick {tick}: your action failed: {data['reason']}")
            if entries:
                memory = world.get(agent, "memory") or {"entries": []}
                merged = (list(memory["entries"]) + entries)[-MEMORY_CAP:]
                perceived = Event(
                    t=float(tick),
                    kind="agent.perceived",
                    entity=agent,
                    data={"added": entries, "set_components": {"memory": {"entries": merged}}},
                )
                world.apply(perceived)
                events.append(perceived)
        return events

    def _tick_event(self, world: World, tick: int, tick_events: list[Event]) -> Event:
        """Per-tick world snapshot — what verifiers and projectors read."""
        locations = {
            (world.get(agent, "persona") or {}).get("name", str(agent)): (
                world.get(agent, "location") or {}
            ).get("place")
            for agent in sorted(self.minds)
        }
        spoke = [
            {"name": e.data["name"], "text": e.data["text"]}
            for e in tick_events
            if e.kind == "agent.spoke"
        ]
        return Event(
            t=float(tick),
            kind="social.tick",
            entity=self.entity,
            data={"tick": tick, "locations": locations, "spoke": spoke},
        )
