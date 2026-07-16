"""Storyboard projection — Track 3: the event log rendered as shots.

Same truth, third projection: where TranscriptProjector formats one line per
event, StoryboardProjector groups events into narrative beats — establishing,
search montage, meeting, dialogue, action — and emits one image-generation
prompt per beat. Character sheets, setting sheets and style are configuration;
WHO did WHAT WHERE always comes from ground-truth events. An optional
enricher lets an LLM rewrite each prompt cinematically; the facts (action,
cast, place, dialogue) stay untouched either way.

Outputs:
  storyboard.json — machine-readable, ready to hand to an image pipeline
  storyboard.md   — human-readable panel sheet

Beat rules (deterministic, tested):
  - a character whose episode opens with a wait gets an establishing shot
  - consecutive moves by the same character collapse into one search montage
  - a move that lands where someone already is becomes a meeting beat, and
    absorbs that tick's dialogue at the destination
  - remaining speech groups into dialogue beats per place, merging across
    consecutive ticks
  - agent.confused becomes a short action beat ("X falters")
  - if the session concluded, a closing title card is appended
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, Field

from .models import Outcome
from .projector import Projector

log = logging.getLogger(__name__)

DEFAULT_STYLE = "anime key visual, cel shading, cinematic lighting, painterly detailed background"

_NARRATIVE_KINDS = {"agent.moved", "agent.spoke", "agent.waited", "agent.confused", "social.tick"}

_CAMERAS = {
    "establishing": "wide establishing shot, slow push-in",
    "montage": "travel montage, quick cuts, wide tracking shots",
    "meeting": "medium two-shot, over-the-shoulder exchange",
    "dialogue": "alternating close-ups, shallow depth of field",
    "action": "handheld medium shot, dutch angle",
    "card": "title card, centered text on a plain background",
}


class Shot(BaseModel):
    """One storyboard panel: ground-truth facts plus a renderable prompt."""

    shot: int
    episode: int
    beat: str
    tick_start: int
    tick_end: int
    place: str
    characters: list[str] = Field(default_factory=list)
    action: str
    dialogue: list[str] = Field(default_factory=list)
    camera: str
    image_prompt: str


class Storyboard(BaseModel):
    """The full board: what an image/video pipeline consumes.

    Carries the character/place sheets so the board is self-contained — a
    renderer can build per-region prompts (multi-character shots) without
    access to the demo's configuration.
    """

    title: str
    style: str
    episodes: int
    characters: dict[str, str] = Field(default_factory=dict)
    places: dict[str, str] = Field(default_factory=dict)
    shots: list[Shot] = Field(default_factory=list)


class ShotEnricher(Protocol):
    """Optionally rewrites a shot's image prompt; must not change the facts."""

    def enrich(self, facts: dict[str, Any]) -> str:
        """Return a single-line image prompt built from the given facts."""
        ...


@dataclass
class _Beat:
    kind: str
    place: str
    characters: list[str]
    action: str
    dialogue: list[str]
    tick_start: int
    tick_end: int
    waypoints: list[str] = field(default_factory=list)


def _bucket_ticks(samples: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Organize engine samples by tick, keeping only narrative event kinds."""
    ticks: dict[int, dict[str, Any]] = {}
    for sample in samples:
        kind = sample.get("kind")
        if kind not in _NARRATIVE_KINDS:
            continue
        tick = int(sample.get("t", 0))
        bucket = ticks.setdefault(
            tick, {"moves": [], "speech": [], "waits": [], "confused": [], "locations": None}
        )
        if kind == "agent.moved":
            bucket["moves"].append(sample)
        elif kind == "agent.spoke":
            bucket["speech"].append(sample)
        elif kind == "agent.waited":
            bucket["waits"].append(sample)
        elif kind == "agent.confused":
            bucket["confused"].append(sample)
        elif kind == "social.tick":
            bucket["locations"] = dict(sample.get("locations") or {})
    return ticks


def episode_beats(samples: list[dict[str, Any]]) -> list[_Beat]:
    """Segment one episode's samples into narrative beats (see module docstring)."""
    ticks = _bucket_ticks(samples)
    if not ticks:
        return []
    beats: list[_Beat] = []
    open_beat: _Beat | None = None

    def close() -> None:
        nonlocal open_beat
        if open_beat is not None:
            if open_beat.kind == "montage":
                open_beat.action = (
                    f"{open_beat.characters[0]} searches: {' -> '.join(open_beat.waypoints)}"
                )
            beats.append(open_beat)
            open_beat = None

    order = sorted(ticks)
    # establishing prelude: characters whose episode opens with a wait
    for wait in ticks[order[0]]["waits"]:
        beats.append(
            _Beat(
                kind="establishing",
                place=wait["place"],
                characters=[wait["name"]],
                action=f"{wait['name']} waits at {wait['place']}",
                dialogue=[],
                tick_start=order[0],
                tick_end=order[0],
            )
        )

    for tick in order:
        bucket = ticks[tick]
        locations: dict[str, str] = bucket["locations"] or {}
        speech_by_place: dict[str, list[dict[str, Any]]] = {}
        for speech in bucket["speech"]:
            speech_by_place.setdefault(speech["place"], []).append(speech)

        for move in bucket["moves"]:
            mover, src, dest = move["name"], move["from"], move["to"]
            others = sorted(n for n, p in locations.items() if p == dest and n != mover)
            if others:
                close()
                lines = [f'{s["name"]}: "{s["text"]}"' for s in speech_by_place.pop(dest, [])]
                beats.append(
                    _Beat(
                        kind="meeting",
                        place=dest,
                        characters=[mover, *others],
                        action=f"{mover} arrives at {dest} and finds {' and '.join(others)}",
                        dialogue=lines,
                        tick_start=tick,
                        tick_end=tick,
                    )
                )
            elif (
                open_beat is not None
                and open_beat.kind == "montage"
                and open_beat.characters == [mover]
            ):
                open_beat.waypoints.append(dest)
                open_beat.place = dest
                open_beat.tick_end = tick
            else:
                close()
                open_beat = _Beat(
                    kind="montage",
                    place=dest,
                    characters=[mover],
                    action="",
                    dialogue=[],
                    tick_start=tick,
                    tick_end=tick,
                    waypoints=[src, dest],
                )

        for place in sorted(speech_by_place):
            lines = [f'{s["name"]}: "{s["text"]}"' for s in speech_by_place[place]]
            present = sorted(n for n, p in locations.items() if p == place) or sorted(
                {s["name"] for s in speech_by_place[place]}
            )
            if open_beat is not None and open_beat.kind == "dialogue" and open_beat.place == place:
                open_beat.dialogue.extend(lines)
                open_beat.characters = sorted(set(open_beat.characters) | set(present))
                open_beat.tick_end = tick
            else:
                close()
                open_beat = _Beat(
                    kind="dialogue",
                    place=place,
                    characters=list(present),
                    action=f"{' and '.join(present)} talk at {place}",
                    dialogue=lines,
                    tick_start=tick,
                    tick_end=tick,
                )

        for confusion in bucket["confused"]:
            close()
            beats.append(
                _Beat(
                    kind="action",
                    place=confusion["place"],
                    characters=[confusion["name"]],
                    action=f"{confusion['name']} falters - {confusion['reason']}",
                    dialogue=[],
                    tick_start=tick,
                    tick_end=tick,
                )
            )
    close()
    return beats


class StoryboardProjector(Projector):
    """Groups social events into beats and emits one image prompt per beat.

    ``characters`` and ``places`` map names to visual sheets (used only when
    that character/place is actually in the shot — presence comes from
    events). Unknown names fall back to themselves, so the projector works
    with zero configuration too.
    """

    def __init__(
        self,
        out_dir: str | Path,
        characters: dict[str, str] | None = None,
        places: dict[str, str] | None = None,
        style: str = DEFAULT_STYLE,
        title: str = "World-Engine Episode",
        enricher: ShotEnricher | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.characters = dict(characters or {})
        self.places = dict(places or {})
        self.style = style
        self.title = title
        self.enricher = enricher

    def finalize(self, outcome: Outcome) -> None:
        shots: list[Shot] = []
        for result in outcome.results:
            for beat in episode_beats(outcome.samples.get(result.experiment, [])):
                shots.append(self._shot(len(shots) + 1, result.experiment, beat))
        if not shots:
            return  # nothing narrative in this run (e.g. a physics backend)
        if outcome.concluded:
            shots.append(self._card(len(shots) + 1, outcome))
        board = Storyboard(
            title=self.title,
            style=self.style,
            episodes=len(outcome.results),
            characters=dict(self.characters),
            places=dict(self.places),
            shots=shots,
        )
        self.out_dir.mkdir(parents=True, exist_ok=True)
        json_path = self.out_dir / "storyboard.json"
        json_path.write_text(board.model_dump_json(indent=2), encoding="utf-8")
        (self.out_dir / "storyboard.md").write_text(_markdown(board), encoding="utf-8")
        log.info("storyboard: %d shots written to %s", len(shots), json_path)

    def _shot(self, number: int, episode: int, beat: _Beat) -> Shot:
        camera = _CAMERAS[beat.kind]
        setting = self.places.get(beat.place, beat.place)
        cast = [self.characters.get(name, name) for name in beat.characters]
        prompt = f"{self.style}. Setting: {setting}."
        if cast:
            prompt += f" Characters: {'; '.join(cast)}."
        prompt += f" Action: {beat.action}. Camera: {camera}."
        shot = Shot(
            shot=number,
            episode=episode,
            beat=beat.kind,
            tick_start=beat.tick_start,
            tick_end=beat.tick_end,
            place=beat.place,
            characters=list(beat.characters),
            action=beat.action,
            dialogue=list(beat.dialogue),
            camera=camera,
            image_prompt=prompt,
        )
        if self.enricher is not None:
            facts = {
                "style": self.style,
                "setting": setting,
                "characters": cast,
                "action": beat.action,
                "dialogue": list(beat.dialogue),
                "camera": camera,
            }
            try:
                shot.image_prompt = self.enricher.enrich(facts)
            except (OSError, ValueError) as exc:  # keep the template on any enricher failure
                log.warning("shot %d: enricher failed, keeping template prompt: %s", number, exc)
        return shot

    def _card(self, number: int, outcome: Outcome) -> Shot:
        camera = _CAMERAS["card"]
        return Shot(
            shot=number,
            episode=outcome.results[-1].experiment if outcome.results else 0,
            beat="card",
            tick_start=0,
            tick_end=0,
            place="",
            characters=[],
            action=outcome.summary,
            dialogue=[],
            camera=camera,
            image_prompt=f"{self.style}. Closing title card: \"{outcome.summary}\". {camera}.",
        )


_ENRICH_PROMPT = """You are the storyboard artist and cinematographer for an anime production.
You receive one shot's facts as JSON: style, setting, characters present, action, dialogue, camera.

Rewrite them into ONE vivid, concrete image-generation prompt line.
Hard rules:
- Keep every named character, the setting, and the action factually intact.
- Do NOT invent new characters, places, or events.
- Output ONLY the prompt line — no quotes, no prose, no code fences, no thinking text."""


class LLMShotEnricher:
    """Rewrites shot prompts via an OpenAI-compatible endpoint (default: llm-router)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        route: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        from .director import llm_defaults

        self.base_url, self.model, self.route, self.api_key = llm_defaults(
            base_url, model, route, api_key
        )
        self.timeout = timeout

    def enrich(self, facts: dict[str, Any]) -> str:
        """One LLM call per shot; raises on empty output so the caller keeps the template."""
        import json

        from .director import chat_completion

        content = chat_completion(
            self.base_url,
            self.model,
            [
                {"role": "system", "content": _ENRICH_PROMPT},
                {"role": "user", "content": json.dumps(facts, indent=2)},
            ],
            route=self.route,
            api_key=self.api_key,
            timeout=self.timeout,
            temperature=0.4,
        )
        line = " ".join(content.split()).strip().strip('"')
        if not line:
            raise ValueError("enricher returned an empty prompt")
        return line


def _markdown(board: Storyboard) -> str:
    """Render the board as a human-readable panel sheet."""
    lines = [f"# {board.title}", "", f"*Style: {board.style}*", ""]
    for shot in board.shots:
        header = f"## Shot {shot.shot} — {shot.beat.upper()}"
        if shot.beat != "card":
            header += f" (ep {shot.episode}, t{shot.tick_start}"
            header += f"–t{shot.tick_end})" if shot.tick_end != shot.tick_start else ")"
        lines.append(header)
        if shot.place:
            lines.append(f"**Place:** {shot.place}")
        if shot.characters:
            lines.append(f"**Cast:** {', '.join(shot.characters)}")
        lines.append(f"**Action:** {shot.action}")
        if shot.dialogue:
            lines.append("**Dialogue:**")
            lines.extend(f"> {line}" for line in shot.dialogue)
        lines.append(f"**Camera:** {shot.camera}")
        lines.append(f"**Prompt:** `{shot.image_prompt}`")
        lines.append("")
    return "\n".join(lines)
