"""Authored script -> storyboard. The hand-written / LLM-generated analog of StoryboardProjector.

Where `StoryboardProjector` turns a social-sim event log into a board, this turns an authored
`Script` (a logline expanded into characters + scenes) into the same `Storyboard`. The output
is the identical dict `scripts/render_storyboard.py` and `scripts/export_cliplab.py` already
consume — nothing downstream cares whether a board came from a simulation or an author.

`generate_script` calls llm-router (reusing `director.chat_completion`, stdlib urllib) to
expand a one-line idea into a structured script; `script_to_board` is pure and deterministic.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field, ValidationError

from .director import chat_completion, extract_json, llm_defaults
from .storyboard import _CAMERAS, DEFAULT_STYLE, Shot, Storyboard

log = logging.getLogger(__name__)

#: beats export_cliplab tunes durations for; anything else falls back to 3.0s there
BEATS: tuple[str, ...] = ("establishing", "montage", "meeting", "dialogue", "action", "card")
DEFAULT_BEAT = "dialogue"


# ── authored-script models ────────────────────────────────────────────────────

class Character(BaseModel):
    """A cast member. `description` is a stable visual sheet (hair, dress, features)
    used verbatim in prompts so the character reads consistently across shots."""

    name: str
    description: str


class Line(BaseModel):
    """One spoken line. Serialized into a board as ``Name: "text"`` — the exact
    shape voice_dialogue.py parses and export_cliplab.py shows as a caption."""

    speaker: str
    text: str


class Scene(BaseModel):
    """One shot's worth of authored intent."""

    beat: str = DEFAULT_BEAT
    place: str
    place_desc: str = ""          # vivid setting sheet, reused as the board place map
    characters: list[str] = Field(default_factory=list)
    action: str
    dialogue: list[Line] = Field(default_factory=list)
    camera: str = ""              # blank -> derived from beat via _CAMERAS


class Script(BaseModel):
    """A complete authored episode."""

    title: str
    logline: str = ""
    style: str = DEFAULT_STYLE
    characters: list[Character] = Field(default_factory=list)
    scenes: list[Scene] = Field(default_factory=list)


# ── script -> board (pure) ─────────────────────────────────────────────────────

def script_to_board(script: Script) -> Storyboard:
    """Turn an authored `Script` into a `Storyboard` the render pipeline consumes.

    Mirrors `StoryboardProjector._shot`'s prompt assembly: style + setting + cast
    sheets + action + camera. Character/place sheets go into the board-level maps
    so `render_storyboard.py --regional` and `--lora` can key off them.
    """
    char_sheets = {c.name: c.description for c in script.characters}
    places: dict[str, str] = {}
    shots: list[Shot] = []

    for i, scene in enumerate(script.scenes, start=1):
        beat = scene.beat if scene.beat in BEATS else DEFAULT_BEAT
        if scene.place_desc and scene.place not in places:
            places[scene.place] = scene.place_desc
        camera = scene.camera.strip() or _CAMERAS.get(beat, "medium shot")
        setting = scene.place_desc.strip() or scene.place
        cast_sheets = [char_sheets.get(name, name) for name in scene.characters]

        prompt = f"{script.style}. Setting: {setting}."
        if cast_sheets:
            prompt += f" Characters: {'; '.join(cast_sheets)}."
        prompt += f" Action: {scene.action}. Camera: {camera}."

        shots.append(
            Shot(
                shot=i,
                episode=1,
                beat=beat,
                tick_start=i,
                tick_end=i,
                place=scene.place,
                characters=list(scene.characters),
                action=scene.action,
                dialogue=[f'{line.speaker}: "{line.text}"' for line in scene.dialogue],
                camera=camera,
                image_prompt=prompt,
            )
        )

    return Storyboard(
        title=script.title,
        style=script.style,
        episodes=1,
        characters=char_sheets,
        places=places,
        shots=shots,
    )


# ── logline -> script (LLM) ────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a screenwriter and character designer for a short anime episode.
Expand the user's one-line idea into a shootable episode as STRICT JSON — no prose, no code fences.

Schema:
{{
  "title": "<short episode title>",
  "style": "anime key visual, cel shading, cinematic lighting, painterly detailed background",
  "characters": [
    {{"name": "<one word or short name>",
      "description": "<a STABLE visual sheet: hair color+style, outfit, distinguishing features, age — the same wording will be reused every shot for consistency>"}}
  ],
  "scenes": [
    {{"beat": "<one of: establishing, montage, meeting, dialogue, action>",
      "place": "<short place id, e.g. rooftop>",
      "place_desc": "<vivid one-sentence setting description>",
      "characters": ["<names present in this shot>"],
      "action": "<what visibly happens in this shot, one sentence>",
      "dialogue": [{{"speaker": "<character name>", "text": "<one short line>"}}],
      "camera": "<short cinematographic phrase, e.g. slow push-in, over-the-shoulder two-shot>"}}
  ]
}}

Rules:
- Exactly {n_scenes} scenes. Use 2-4 characters total (more than 4 makes shots muddy).
- Give EACH shot at most 2 characters (two-person shots max) so they render cleanly.
- Keep dialogue to 0-2 short lines per scene; some scenes can have none.
- Every "speaker" and every name in "characters" MUST match a defined character "name" exactly.
- Vary the beats; a montage scene shows a character moving/searching across places.
- Respond with ONLY the JSON object."""


def generate_script(
    logline: str,
    n_scenes: int = 6,
    base_url: str | None = None,
    model: str | None = None,
    route: str | None = None,
    api_key: str | None = None,
    timeout: float = 300.0,  # a full script is a long generation; local models under
                             # GPU contention (Wan + llama-servers resident) can be slow
    chat_fn=None,
) -> Script:
    """Expand a one-line idea into a `Script` via llm-router.

    `chat_fn(system: str, user: str) -> str` is injectable for tests; by default it
    posts to llm-router through `director.chat_completion`. Retries once, feeding the
    validation error back, before raising.
    """
    if chat_fn is None:
        resolved_base, resolved_model, resolved_route, resolved_key = llm_defaults(
            base_url, model, route, api_key
        )

        def chat_fn(system: str, user: str) -> str:  # noqa: ANN001
            return chat_completion(
                resolved_base,
                resolved_model,
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                route=resolved_route,
                api_key=resolved_key,
                timeout=timeout,
                temperature=0.7,
            )

    system = _SYSTEM_PROMPT.format(n_scenes=n_scenes)
    user = logline.strip()
    error_note: str | None = None
    for _ in range(2):
        prompt = user if error_note is None else (
            f"{user}\n\nYour previous reply was invalid: {error_note}\nReply with ONLY the JSON object."
        )
        content = chat_fn(system, prompt)
        try:
            script = Script.model_validate_json(extract_json(content))
        except (ValidationError, ValueError) as exc:
            error_note = str(exc)
            log.warning("script generation parse failed, retrying: %s", exc)
            continue
        if not script.logline:
            script = script.model_copy(update={"logline": logline.strip()})
        return script
    raise RuntimeError(f"LLM failed to produce a valid script: {error_note}")
