"""Synthesize a storyboard's dialogue lines via voice-lab (Kokoro TTS).

Reads storyboard.json, parses each dialogue line ('Name: "text"'), synthesizes
it with the speaker's cast voice through voice-lab's CLI, and writes
``shot_NNN_line_II.wav`` files that export_cliplab.py picks up via --voice.

Usage:
    python scripts/voice_dialogue.py runs/<ts>/storyboard.json \
        --cast "Mira=af_sky:1.05" --cast "Rook=bm_george:0.85" \
        --out runs/<ts>/voice
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

log = logging.getLogger("voice_dialogue")

VOICE_LAB = Path("C:/projects/voice-lab")


def parse_cast(specs: list[str]) -> dict[str, tuple[str, float]]:
    """Parse repeated --cast "Name=voice[:speed]" into {name: (voice_id, speed)}."""
    cast: dict[str, tuple[str, float]] = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        voice, _, speed = rest.partition(":")
        if not name or not voice:
            raise ValueError(f"bad --cast spec {spec!r}; expected Name=voice[:speed]")
        cast[name.strip()] = (voice.strip(), float(speed) if speed else 1.0)
    return cast


def parse_line(line: str) -> tuple[str, str]:
    """Split 'Name: "text"' into (speaker, spoken text)."""
    speaker, _, rest = line.partition(":")
    text = rest.strip().strip('"').strip()
    return speaker.strip(), text


def synthesize(voicelab: Path, text: str, voice: str, speed: float, out_path: Path) -> None:
    """One voice-lab CLI call; writes a WAV at out_path."""
    python = voicelab / ".venv" / "Scripts" / "python.exe"
    subprocess.run(
        [
            str(python), str(voicelab / "scripts" / "synthesize.py"),
            "--text", text, "--voice", voice, "--speed", f"{speed:g}",
            "--out", str(out_path),
        ],
        check=True,
        cwd=str(voicelab),
        capture_output=True,
    )


def main(argv: list[str] | None = None) -> int:
    """Synthesize every dialogue line whose speaker has a cast voice."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path, help="path to storyboard.json")
    parser.add_argument("--cast", action="append", default=[], metavar="Name=voice[:speed]", required=True)
    parser.add_argument("--out", type=Path, default=None, help="wav dir (default: <board dir>/voice)")
    parser.add_argument("--voicelab", type=Path, default=VOICE_LAB)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    board = json.loads(args.board.read_text(encoding="utf-8"))
    cast = parse_cast(args.cast)
    # resolve() is load-bearing: the synth subprocess runs with cwd=voice-lab,
    # so a relative --out would land the WAVs inside the voice-lab tree
    out_dir = (args.out or args.board.parent / "voice").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    made = failures = 0
    for shot in board.get("shots", []):
        for i, line in enumerate(shot.get("dialogue", [])):
            speaker, text = parse_line(line)
            if speaker not in cast or not text:
                log.warning("shot %d line %d: no cast voice for %r, skipping", shot["shot"], i, speaker)
                continue
            wav = out_dir / f"shot_{shot['shot']:03d}_line_{i:02d}.wav"
            if wav.exists() and not args.force:
                log.info("shot %d line %d: exists, skipping", shot["shot"], i)
                made += 1
                continue
            voice, speed = cast[speaker]
            try:
                synthesize(args.voicelab, text, voice, speed, wav)
                made += 1
                log.info("shot %d line %d: %s (%s) -> %s", shot["shot"], i, speaker, voice, wav.name)
            except subprocess.CalledProcessError as exc:
                failures += 1
                stderr = (exc.stderr or b"").decode("utf-8", "replace")[-300:]
                log.error("shot %d line %d failed: %s", shot["shot"], i, stderr)
    log.info("voice lines: %d written to %s", made, out_dir)
    return 0 if failures == 0 and made else 1


if __name__ == "__main__":
    sys.exit(main())
