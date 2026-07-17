"""Export a rendered storyboard as a ClipLab Pro Editor timeline (manifest v2).

Turns storyboard.json + its rendered frames into everything ClipLab's Editor
needs to open the episode as an editable timeline:

  - one short H.264 clip per shot (still frame + silent audio, beat-scaled
    duration), because the Editor's video track plays video files, not PNGs
  - manifest.json in ClipForge manifest v2 schema: one clip per shot in
    storyboard order, dialogue as inline caption cues anchored to their
    shot's clipUuid (source-anchored, survives reordering)

Open it in ClipLab Pro via the Editor's "Import manifest" flow.

ffmpeg resolution order: PIPELINE_FFMPEG_BIN (gaming-pipeline's convention)
-> PATH -> the imageio-ffmpeg bundled binary.

Usage:
    python scripts/export_cliplab.py runs/<ts>/storyboard.json --frames runs/<ts>/frames-lora
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("export_cliplab")

MANIFEST_VERSION = 2  # SYNC: clipforge/src/utils/manifest.js MANIFEST_VERSION

#: seconds per beat kind before dialogue extension
BEAT_BASE_SECONDS = {
    "establishing": 3.5,
    "montage": 4.5,
    "meeting": 2.5,
    "dialogue": 2.5,
    "action": 2.5,
    "card": 3.5,
}
SECONDS_PER_LINE = 2.2
CAPTION_PAD = 0.4  # cue breathing room at each end of a clip


def find_ffmpeg() -> str:
    """Resolve an ffmpeg binary (env override -> PATH -> imageio-ffmpeg)."""
    env = os.environ.get("PIPELINE_FFMPEG_BIN", "").strip()
    if env:
        return env
    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def wav_duration(path: Path) -> float:
    """Duration of a PCM WAV in seconds (stdlib only)."""
    import wave

    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


def shot_assembly(
    shot: dict, clip_uuid: str, voice_dir: Path | None = None
) -> tuple[float, list[dict], list[dict]]:
    """Timing for one shot: (clip duration, caption cues, dialogue-audio sfx).

    Dialogue lines play sequentially from CAPTION_PAD. A line's slot is its
    real audio length + 0.4s when its WAV exists (``shot_NNN_line_II.wav`` in
    voice_dir, produced by scripts/voice_dialogue.py), else SECONDS_PER_LINE —
    so spoken lines never overrun their shot. Voice lines become sfx entries
    anchored to the same clipUuid as their caption cue.
    """
    base = BEAT_BASE_SECONDS.get(shot.get("beat", ""), 3.0)
    cues: list[dict] = []
    sfx: list[dict] = []
    t = CAPTION_PAD
    for i, line in enumerate(shot.get("dialogue", [])):
        wav: Path | None = None
        if voice_dir is not None:
            candidate = voice_dir / f"shot_{shot['shot']:03d}_line_{i:02d}.wav"
            if candidate.exists():
                wav = candidate
        audio_len = wav_duration(wav) if wav else None
        slot = max(SECONDS_PER_LINE, (audio_len or 0.0) + 0.4)
        cues.append(
            {
                "clipUuid": clip_uuid,
                "start": round(t, 3),
                "end": round(t + slot - 0.1, 3),
                "text": line,
            }
        )
        if wav is not None and audio_len:
            sfx.append(
                {
                    "uuid": f"voice-{shot['shot']:03d}-{i:02d}",
                    "clipUuid": clip_uuid,
                    "start": round(t, 3),
                    "end": round(t + audio_len, 3),
                    "path": str(wav.resolve()),
                    "category": "dialogue",
                    "label": line.split(":", 1)[0].strip(),
                    "gain": 1.0,
                }
            )
        t += slot
    duration = round(base + (t - CAPTION_PAD), 3)
    return duration, cues, sfx


def shot_duration(shot: dict, voice_dir: Path | None = None) -> float:
    """Beat-scaled clip duration (see shot_assembly for the timing model)."""
    return shot_assembly(shot, "_", voice_dir)[0]


def build_manifest(board: dict, clip_files: dict[int, Path], voice_dir: Path | None = None) -> dict:
    """ClipForge manifest v2: one clip per shot, dialogue as anchored cues + voice sfx."""
    clips = []
    cues = []
    sfx = []
    for order, shot in enumerate(board.get("shots", [])):
        number = shot["shot"]
        path = clip_files.get(number)
        if path is None:
            continue
        uuid = f"shot-{number:03d}"
        duration, shot_cues, shot_sfx = shot_assembly(shot, uuid, voice_dir)
        summary = shot.get("action", "")
        if shot.get("dialogue"):
            summary += " — " + " / ".join(shot["dialogue"])
        clips.append(
            {
                "uuid": uuid,
                "path": str(path.resolve()),
                "sourcePath": str(path.resolve()),
                "trimStart": 0.0,
                "trimEnd": duration,
                "sourceDuration": duration,
                "order": order,
                "name": f"{shot.get('beat', 'shot')} — {shot.get('action', '')[:48]}",
                "summary": summary[:600],
                "category": shot.get("beat", ""),
                "tags": [t for t in (shot.get("beat"), shot.get("place")) if t],
                "audioMuted": False,
            }
        )
        cues.extend(shot_cues)
        sfx.extend(shot_sfx)
    manifest = {
        "version": MANIFEST_VERSION,
        "runId": f"world-engine-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "generator": {
            "name": "world-engine",
            "stage": "export_cliplab",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
        },
        "clips": clips,
        "overlays": [],
        "sfx": sfx,
        "captions": {
            "cues": cues,
            "timelineBasis": "source-anchored",
            "burnedIn": False,
        },
    }
    return manifest


def render_clip(ffmpeg: str, frame: Path, out_path: Path, duration: float) -> None:
    """Still frame -> H.264 mp4 with silent stereo audio (Editor-friendly)."""
    command = [
        ffmpeg, "-y", "-loglevel", "error",
        "-loop", "1", "-i", str(frame),
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
        "-t", f"{duration:.3f}",
        "-r", "30",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-c:a", "aac", "-shortest",
        "-movflags", "+faststart",
        str(out_path),
    ]
    subprocess.run(command, check=True)


def _localize_voice_audio(manifest: dict, out_dir: Path) -> None:
    """Copy referenced voice WAVs into out_dir/voice and rewrite sfx paths.

    Makes the export package self-contained: ClipLab's fs/media IPC only reads
    inside allowed roots, so every file the manifest references must live with
    the manifest (which the user places under an allowed root).
    """
    voice_out = out_dir / "voice"
    for entry in manifest.get("sfx", []):
        source = Path(entry["path"])
        voice_out.mkdir(parents=True, exist_ok=True)
        dest = voice_out / source.name
        if source.resolve() != dest.resolve():
            shutil.copyfile(source, dest)
        entry["path"] = str(dest.resolve())


def build_parser() -> argparse.ArgumentParser:
    """CLI for the ClipLab timeline export."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path, help="path to storyboard.json")
    parser.add_argument("--frames", type=Path, default=None, help="frames dir (default: <board dir>/frames)")
    parser.add_argument(
        "--voice",
        type=Path,
        default=None,
        help="dir of shot_NNN_line_II.wav dialogue audio (default: <board dir>/voice if present)",
    )
    parser.add_argument("--out", type=Path, default=None, help="output dir (default: <board dir>/cliplab)")
    parser.add_argument("--force", action="store_true", help="re-encode clips that already exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Encode per-shot clips and write manifest.json; exit 0 only if all shots exported."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    board = json.loads(args.board.read_text(encoding="utf-8"))
    shots = board.get("shots", [])
    if not shots:
        log.error("board has no shots: %s", args.board)
        return 1
    frames_dir = args.frames or args.board.parent / "frames"
    out_dir = args.out or args.board.parent / "cliplab"
    out_dir.mkdir(parents=True, exist_ok=True)
    voice_dir = args.voice
    if voice_dir is None and (args.board.parent / "voice").is_dir():
        voice_dir = args.board.parent / "voice"
    if voice_dir is not None:
        log.info("dialogue audio: %s", voice_dir)

    ffmpeg = find_ffmpeg()
    log.info("ffmpeg: %s", ffmpeg)

    clip_files: dict[int, Path] = {}
    failures = 0
    for shot in shots:
        number = shot["shot"]
        frame = frames_dir / f"shot_{number:03d}.png"
        if not frame.exists():
            failures += 1
            log.error("shot %d: frame missing (%s) — render the board first", number, frame)
            continue
        clip_path = out_dir / f"shot_{number:03d}.mp4"
        if clip_path.exists() and not args.force:
            log.info("shot %d: clip exists, skipping (use --force)", number)
            clip_files[number] = clip_path
            continue
        duration = shot_duration(shot, voice_dir)
        try:
            render_clip(ffmpeg, frame, clip_path, duration)
            clip_files[number] = clip_path
            log.info("shot %d (%s): %.1fs -> %s", number, shot.get("beat"), duration, clip_path)
        except (OSError, subprocess.CalledProcessError) as exc:
            failures += 1
            log.error("shot %d: encode failed: %s", number, exc)

    manifest = build_manifest(board, clip_files, voice_dir)
    _localize_voice_audio(manifest, out_dir)
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    total = sum(c["sourceDuration"] for c in manifest["clips"])
    log.info(
        "manifest: %d clips, %d caption cues, %d voice lines, %.1fs timeline -> %s",
        len(manifest["clips"]), len(manifest["captions"]["cues"]), len(manifest["sfx"]),
        total, manifest_path,
    )
    return 0 if failures == 0 and manifest["clips"] else 1


if __name__ == "__main__":
    sys.exit(main())
