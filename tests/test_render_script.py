"""Render script: shot selection parsing and contact-sheet output (no server needed)."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "render_storyboard", Path(__file__).resolve().parents[1] / "scripts" / "render_storyboard.py"
)
render_storyboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_storyboard)


def test_lora_injection_scoped_to_present_characters() -> None:
    loras = render_storyboard.parse_lora_specs(
        ["Mira=mira_oc:0.85:miraoc", "Rook=rook_oc:0.8:rookoc"]
    )
    both = render_storyboard.apply_loras("a meeting", ["Mira", "Rook"], loras)
    assert both.startswith("<lora:mira_oc:0.85> miraoc, <lora:rook_oc:0.8> rookoc, ")
    solo = render_storyboard.apply_loras("a walk", ["Mira"], loras)
    assert "rook_oc" not in solo and solo.endswith("a walk")
    card = render_storyboard.apply_loras("title card", [], loras)
    assert card == "title card"  # no cast, no injection


def test_parse_lora_specs_rejects_malformed() -> None:
    import pytest

    with pytest.raises(ValueError):
        render_storyboard.parse_lora_specs(["Mira=mira_oc:0.85"])  # missing trigger


def test_regional_prompt_one_region_per_character() -> None:
    board = {
        "style": "anime key visual",
        "characters": {"Mira": "Mira, copper hair, green cloak", "Rook": "Rook, spectacles, gray robes"},
        "places": {"docks": "fog-covered wooden docks"},
    }
    shot = {"characters": ["Mira", "Rook"], "place": "docks", "action": "Mira arrives and finds Rook"}
    prompt = render_storyboard.build_regional_prompt(shot, board)
    common, mira_region, rook_region = prompt.split(" BREAK ")
    assert "fog-covered wooden docks" in common and "2 people" in common
    assert "copper hair" in mira_region and "spectacles" not in mira_region
    assert "spectacles" in rook_region and "copper hair" not in rook_region


def test_regional_script_args_shape() -> None:
    args = render_storyboard.regional_script_args(3)
    assert len(args) == 20
    assert args[0] is True and args[11] == "Attention" and args[6] == "1,1,1"


def test_parse_shot_selection() -> None:
    parse = render_storyboard.parse_shot_selection
    assert parse(None, 5) == [1, 2, 3, 4, 5]
    assert parse("1,3", 5) == [1, 3]
    assert parse("2-4", 5) == [2, 3, 4]
    assert parse("1,3-5,3", 5) == [1, 3, 4, 5]  # dedup + merge
    assert parse("4-9", 5) == [4, 5]  # clamped to the board


def test_contact_sheet_pairs_frames_with_facts(tmp_path: Path) -> None:
    board = {
        "title": "The Rendezvous",
        "style": "s",
        "episodes": 1,
        "shots": [
            {"shot": 1, "episode": 1, "beat": "meeting", "action": "Mira finds Rook",
             "dialogue": ['Rook: "Well met."'], "camera": "two-shot"},
            {"shot": 2, "episode": 1, "beat": "card", "action": "the end",
             "dialogue": [], "camera": "title card"},
        ],
    }
    (tmp_path / "shot_001.png").write_bytes(b"png")
    path = render_storyboard.write_contact_sheet(board, tmp_path, {1: "shot_001.png"})
    html = path.read_text(encoding="utf-8")
    assert "The Rendezvous" in html
    assert 'src="shot_001.png"' in html
    assert "Mira finds Rook" in html and "Well met." in html
    assert "not rendered" in html  # shot 2 has no frame yet
