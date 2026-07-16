"""Render a storyboard.json into key frames via a Stable Diffusion HTTP API.

Consumes the contract StoryboardProjector emits and talks to an
AUTOMATIC1111-compatible endpoint (`/sdapi/v1/txt2img`) — locally that is
sd-lab's A1111 backend on 127.0.0.1:7861. One PNG per shot lands in a
``frames/`` directory next to the board, plus a ``frames/index.html`` contact
sheet pairing each frame with its action, dialogue and camera facts.

Deterministic by default: each shot is rendered with ``--seed + shot number``,
so re-rendering an unchanged board reproduces the same frames. Existing frames
are skipped unless ``--force``.

Usage:
    python scripts/render_storyboard.py runs/<ts>/storyboard.json
    python scripts/render_storyboard.py board.json --shots 5-7 --steps 20
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
import urllib.error
import urllib.request
from html import escape
from pathlib import Path

log = logging.getLogger("render_storyboard")

DEFAULT_BASE_URL = "http://127.0.0.1:7861"
DEFAULT_NEGATIVE = (
    "lowres, bad anatomy, bad hands, extra fingers, text, watermark, signature, "
    "jpeg artifacts, worst quality, blurry, deformed"
)


def _post(base_url: str, path: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def parse_lora_specs(specs: list[str]) -> dict[str, tuple[str, float, str]]:
    """Parse repeated --lora "Name=file:weight:trigger" into {name: (file, weight, trigger)}.

    The mapping is applied per shot from its ground-truth `characters` list, so
    a character's LoRA (and trigger token) is only active when that character
    is actually in frame.
    """
    mapping: dict[str, tuple[str, float, str]] = {}
    for spec in specs:
        name, _, rest = spec.partition("=")
        parts = rest.split(":")
        if not name or len(parts) != 3:
            raise ValueError(f"bad --lora spec {spec!r}; expected Name=file:weight:trigger")
        mapping[name.strip()] = (parts[0].strip(), float(parts[1]), parts[2].strip())
    return mapping


def apply_loras(prompt: str, characters: list[str], loras: dict[str, tuple[str, float, str]]) -> str:
    """Prepend <lora:...> tags + trigger tokens for the characters present in the shot."""
    prefixes = []
    for character in characters:
        if character in loras:
            file, weight, trigger = loras[character]
            prefixes.append(f"<lora:{file}:{weight:g}> {trigger}")
    return f"{', '.join(prefixes)}, {prompt}" if prefixes else prompt


def parse_shot_selection(spec: str | None, count: int) -> list[int]:
    """Parse "1,3,5-7" into shot numbers; None means all shots."""
    if not spec:
        return list(range(1, count + 1))
    selected: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            selected.update(range(int(lo), int(hi) + 1))
        elif part:
            selected.add(int(part))
    return sorted(n for n in selected if 1 <= n <= count)


def render_shot(
    base_url: str,
    shot: dict,
    out_path: Path,
    args: argparse.Namespace,
    loras: dict[str, tuple[str, float, str]] | None = None,
) -> None:
    """One txt2img call; writes the first returned image to out_path."""
    prompt = shot["image_prompt"]
    if loras:
        prompt = apply_loras(prompt, shot.get("characters", []), loras)
    payload = {
        "prompt": prompt,
        "negative_prompt": args.negative,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfg_scale": args.cfg,
        "sampler_name": args.sampler,
        "seed": args.seed + shot["shot"],
        "batch_size": 1,
        "n_iter": 1,
    }
    body = _post(base_url, "/sdapi/v1/txt2img", payload, timeout=args.timeout)
    images = body.get("images") or []
    if not images:
        raise RuntimeError(f"no image returned for shot {shot['shot']}")
    out_path.write_bytes(base64.b64decode(images[0].split(",", 1)[-1]))


def write_contact_sheet(board: dict, frames_dir: Path, rendered: dict[int, str]) -> Path:
    """frames/index.html — every frame next to its ground-truth facts."""
    rows = []
    for shot in board["shots"]:
        frame = rendered.get(shot["shot"])
        img = f'<img src="{escape(frame)}" alt="shot {shot["shot"]}">' if frame else "<em>not rendered</em>"
        dialogue = "".join(f"<blockquote>{escape(line)}</blockquote>" for line in shot.get("dialogue", []))
        rows.append(
            f'<section><h2>Shot {shot["shot"]} — {escape(shot["beat"].upper())}'
            f' <small>(ep {shot["episode"]})</small></h2>{img}'
            f'<p><strong>{escape(shot["action"])}</strong></p>{dialogue}'
            f'<p><small>{escape(shot["camera"])}</small></p></section>'
        )
    html = (
        f"<!doctype html><meta charset='utf-8'><title>{escape(board['title'])}</title>"
        "<style>body{font-family:sans-serif;max-width:860px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%;border-radius:8px}section{margin-bottom:2.5rem}"
        "blockquote{margin:.3rem 0;font-style:italic}</style>"
        f"<h1>{escape(board['title'])}</h1>" + "".join(rows)
    )
    path = frames_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    """CLI for rendering a storyboard into frames."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path, help="path to storyboard.json")
    parser.add_argument("--out", type=Path, default=None, help="frames dir (default: <board dir>/frames)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="A1111-compatible API base URL")
    parser.add_argument("--checkpoint", default="DreamShaper_8_pruned", help="checkpoint to activate first ('' to skip)")
    parser.add_argument(
        "--lora",
        action="append",
        default=[],
        metavar="Name=file:weight:trigger",
        help="per-character LoRA, applied only to shots that character is in "
        "(e.g. Mira=mira_oc:0.85:miraoc); repeatable",
    )
    parser.add_argument("--shots", default=None, help="subset like '1,3,5-7' (default: all)")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=432)
    parser.add_argument("--steps", type=int, default=28)
    parser.add_argument("--cfg", type=float, default=7.0)
    parser.add_argument("--sampler", default="DPM++ 2M Karras")
    parser.add_argument("--seed", type=int, default=1234, help="base seed; shot N renders at seed+N")
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--timeout", type=float, default=600.0, help="per-shot HTTP timeout, seconds")
    parser.add_argument("--force", action="store_true", help="re-render frames that already exist")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Render selected shots; exit 0 only if every selected shot has a frame."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    board = json.loads(args.board.read_text(encoding="utf-8"))
    shots = board.get("shots", [])
    if not shots:
        log.error("board has no shots: %s", args.board)
        return 1
    frames_dir = args.out or args.board.parent / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if args.checkpoint:
        log.info("activating checkpoint %s (may take a moment)", args.checkpoint)
        _post(args.base_url, "/sdapi/v1/options", {"sd_model_checkpoint": args.checkpoint}, timeout=300.0)

    loras = parse_lora_specs(args.lora)
    if loras:
        log.info("character LoRAs: %s", {k: v[0] for k, v in loras.items()})
    selection = parse_shot_selection(args.shots, len(shots))
    by_number = {shot["shot"]: shot for shot in shots}
    rendered: dict[int, str] = {}
    failures = 0
    for number in selection:
        shot = by_number[number]
        out_path = frames_dir / f"shot_{number:03d}.png"
        if out_path.exists() and not args.force:
            log.info("shot %d: exists, skipping (use --force to re-render)", number)
            rendered[number] = out_path.name
            continue
        try:
            render_shot(args.base_url, shot, out_path, args, loras)
            rendered[number] = out_path.name
            log.info("shot %d (%s): %s", number, shot["beat"], out_path)
        except (urllib.error.URLError, RuntimeError, OSError) as exc:
            failures += 1
            log.error("shot %d failed: %s", number, exc)

    # include frames from earlier runs so the sheet stays complete on subset renders
    for shot in shots:
        path = frames_dir / f"shot_{shot['shot']:03d}.png"
        if shot["shot"] not in rendered and path.exists():
            rendered[shot["shot"]] = path.name
    sheet = write_contact_sheet(board, frames_dir, rendered)
    log.info("contact sheet: %s (%d/%d frames)", sheet, len(rendered), len(shots))
    return 0 if failures == 0 and rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
