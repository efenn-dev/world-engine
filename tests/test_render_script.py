"""Render script: shot selection parsing and contact-sheet output (no server needed)."""

import importlib.util
import json
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "render_storyboard", Path(__file__).resolve().parents[1] / "scripts" / "render_storyboard.py"
)
render_storyboard = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(render_storyboard)


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
