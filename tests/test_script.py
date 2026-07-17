"""Authored script -> board: prompt assembly, consumer-key coverage, LLM parsing."""

import json

import pytest

from worldengine.script import (
    Character,
    Line,
    Scene,
    Script,
    generate_script,
    script_to_board,
)
from worldengine.storyboard import Storyboard


def _demo_script() -> Script:
    return Script(
        title="Neon Rivals",
        logline="two rival street chefs in neon Tokyo",
        characters=[
            Character(name="Ava", description="young woman, magenta bob, red apron"),
            Character(name="Ben", description="older man, shaved head, blue chef coat"),
        ],
        scenes=[
            Scene(beat="establishing", place="alley", place_desc="a rain-slick neon alley",
                  characters=["Ava"], action="Ava lights her food cart", camera="wide establishing shot"),
            Scene(beat="meeting", place="alley", characters=["Ava", "Ben"],
                  action="Ben rolls his cart up beside Ava's",
                  dialogue=[Line(speaker="Ben", text="You're on my corner.")]),
            Scene(beat="dialogue", place="alley", characters=["Ava", "Ben"],
                  action="the two chefs face off",
                  dialogue=[Line(speaker="Ava", text="Then cook better."),
                            Line(speaker="Ben", text="Watch me.")]),
        ],
    )


def test_script_to_board_roundtrips_through_model() -> None:
    board = script_to_board(_demo_script())
    assert isinstance(board, Storyboard)
    # serialize -> dict (what gets written to storyboard.json) and re-validate
    data = json.loads(board.model_dump_json())
    assert Storyboard.model_validate(data)
    assert data["title"] == "Neon Rivals"
    assert data["characters"] == {
        "Ava": "young woman, magenta bob, red apron",
        "Ben": "older man, shaved head, blue chef coat",
    }
    assert data["places"]["alley"] == "a rain-slick neon alley"  # first place_desc wins
    assert len(data["shots"]) == 3


def test_board_satisfies_downstream_consumers() -> None:
    data = json.loads(script_to_board(_demo_script()).model_dump_json())
    # every key render_storyboard.py / export_cliplab.py access by bracket
    assert isinstance(data["title"], str)
    for i, shot in enumerate(data["shots"], start=1):
        assert shot["shot"] == i
        for key in ("episode", "beat", "action", "camera", "image_prompt"):
            assert isinstance(shot[key], str) or isinstance(shot[key], int)
        assert shot["beat"] in ("establishing", "montage", "meeting", "dialogue", "action", "card")


def test_prompt_assembly_and_dialogue_format() -> None:
    board = script_to_board(_demo_script())
    establishing, meeting, dialogue = board.shots
    # cast sheets (not bare names) embedded in the prompt
    assert "magenta bob" in establishing.image_prompt
    assert "a rain-slick neon alley" in establishing.image_prompt
    # camera derived from beat when blank
    assert establishing.camera  # non-empty (from _CAMERAS[establishing])
    # dialogue serialized as Name: "text" (voice_dialogue/export_cliplab format)
    assert meeting.dialogue == ['Ben: "You\'re on my corner."']
    assert dialogue.dialogue == ['Ava: "Then cook better."', 'Ben: "Watch me."']


def test_unknown_beat_coerced() -> None:
    script = Script(title="T", scenes=[Scene(beat="explosion", place="p", action="boom")])
    assert script_to_board(script).shots[0].beat == "dialogue"


_VALID_JSON = json.dumps({
    "title": "The Test",
    "style": "anime key visual",
    "characters": [{"name": "Zed", "description": "tall, silver hair"}],
    "scenes": [{"beat": "dialogue", "place": "bridge", "place_desc": "a foggy bridge",
                "characters": ["Zed"], "action": "Zed waits",
                "dialogue": [{"speaker": "Zed", "text": "At last."}], "camera": "close-up"}],
})


def test_generate_script_parses_and_backfills_logline() -> None:
    script = generate_script("a lonely android", chat_fn=lambda system, user: _VALID_JSON)
    assert script.title == "The Test"
    assert script.characters[0].name == "Zed"
    assert script.logline == "a lonely android"  # backfilled when LLM omits it
    board = script_to_board(script)
    assert board.shots[0].image_prompt.count("silver hair") == 1


def test_generate_script_tolerates_fences_then_retries() -> None:
    # first reply is garbage, second is valid wrapped in prose + fences
    replies = iter(["not json at all", f"Here you go:\n```json\n{_VALID_JSON}\n```"])

    def flaky(system: str, user: str) -> str:
        return next(replies)

    script = generate_script("x", chat_fn=flaky)
    assert script.title == "The Test"


def test_generate_script_raises_after_two_failures() -> None:
    with pytest.raises(RuntimeError, match="valid script"):
        generate_script("x", chat_fn=lambda system, user: "still not json")
