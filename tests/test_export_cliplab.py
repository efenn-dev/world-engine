"""ClipLab exporter: duration model, cue anchoring, manifest shape (no ffmpeg needed)."""

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "export_cliplab", Path(__file__).resolve().parents[1] / "scripts" / "export_cliplab.py"
)
export_cliplab = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_cliplab)


def _shot(number: int, beat: str, dialogue: list[str] | None = None) -> dict:
    return {
        "shot": number,
        "episode": 1,
        "beat": beat,
        "place": "docks",
        "characters": ["Mira"],
        "action": f"action {number}",
        "dialogue": dialogue or [],
        "camera": "cam",
        "image_prompt": "p",
    }


def test_shot_duration_scales_with_dialogue() -> None:
    assert export_cliplab.shot_duration(_shot(1, "establishing")) == 3.5
    assert export_cliplab.shot_duration(_shot(2, "montage")) == 4.5
    two_lines = export_cliplab.shot_duration(_shot(3, "dialogue", ["a", "b"]))
    assert two_lines == 2.5 + 2 * export_cliplab.SECONDS_PER_LINE


def test_caption_cues_fit_inside_clip_and_anchor() -> None:
    shot = _shot(5, "meeting", ['Rook: "Well met."', 'Mira: "Found you!"'])
    duration = export_cliplab.shot_duration(shot)
    cues = export_cliplab.caption_cues(shot, "shot-005", duration)
    assert len(cues) == 2
    assert all(cue["clipUuid"] == "shot-005" for cue in cues)
    assert cues[0]["start"] >= 0 and cues[-1]["end"] <= duration
    assert cues[0]["end"] <= cues[1]["start"]  # no overlap
    assert cues[1]["text"] == 'Mira: "Found you!"'


def test_build_manifest_shape(tmp_path: Path) -> None:
    board = {"title": "t", "style": "s", "episodes": 1,
             "shots": [_shot(1, "establishing"), _shot(2, "dialogue", ["x"])]}
    files = {1: tmp_path / "shot_001.mp4", 2: tmp_path / "shot_002.mp4"}
    for path in files.values():
        path.write_bytes(b"")
    manifest = export_cliplab.build_manifest(board, files)
    assert manifest["version"] == export_cliplab.MANIFEST_VERSION
    assert [c["uuid"] for c in manifest["clips"]] == ["shot-001", "shot-002"]
    assert all(Path(c["path"]).is_absolute() for c in manifest["clips"])
    for clip in manifest["clips"]:
        assert clip["trimEnd"] == clip["sourceDuration"] > clip["trimStart"]
    assert manifest["captions"]["timelineBasis"] == "source-anchored"
    assert manifest["captions"]["cues"][0]["clipUuid"] == "shot-002"
    # a shot without a rendered clip is excluded rather than breaking the manifest
    partial = export_cliplab.build_manifest(board, {1: files[1]})
    assert [c["uuid"] for c in partial["clips"]] == ["shot-001"]
    assert partial["captions"]["cues"] == []
