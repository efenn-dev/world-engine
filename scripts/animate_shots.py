"""Animate storyboard frames into video clips via Wan 2.2 (ComfyUI image-to-video).

The I2V architecture is the point: each rendered key frame is the FIRST FRAME
of its shot's video, so all the character-consistency work (LoRAs, regional
prompting) carries over — the frame is the identity anchor, and the video
model only adds motion. Motion is guided by a text prompt built from the
shot's ground-truth action + camera direction.

Per shot: upload frame -> queue a Wan 2.2 TI2V-5B workflow -> poll history ->
download the generated frames -> assemble an H.264 clip with ffmpeg (24 fps).
export_cliplab.py --animated then prefers these clips over still-encodes.

Wan holds about 5s (121 frames) in a single pass, so shots longer than that are
generated as a CHAIN: each pass after the first is anchored on the previous
pass's final frame, and the duplicated anchor frame is dropped when stitching.
Without this a long shot was silently truncated to 5s and the exporter froze
the last frame for the remainder (render.py's tpad=stop_mode=clone).

Requires sd-lab's ComfyUI running on :8188 with the Wan 2.2 5B weights in
D:/sd-lab-models (diffusion_models/, text_encoders/, vae/).

Usage:
    python scripts/animate_shots.py runs/<ts>/storyboard.json \
        --frames runs/<ts>/frames-lora --shots 5
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_storyboard import parse_shot_selection  # noqa: E402
from export_cliplab import find_ffmpeg, shot_duration  # noqa: E402

log = logging.getLogger("animate_shots")

DEFAULT_BASE_URL = "http://127.0.0.1:8188"
FPS = 24
MAX_FRAMES = 121  # Wan 2.2 trains on 5s@24fps clips; one pass past this degrades

#: Shots longer than MAX_FRAMES are generated as a CHAIN of passes: each pass
#: after the first is anchored on the previous pass's final frame. That keeps
#: every individual pass inside the band Wan was trained on (and inside 16GB of
#: VRAM) while still filling the shot's real duration.
#:
#: The cost is drift: pass 2 onward anchors on a Wan-generated frame rather than
#: the crisp SD key frame, so each link is a little softer than the last. Three
#: passes (~15s) is the point where that stops being worth it -- past this the
#: shot is truncated as before and the exporter holds the final frame.
MAX_CHAIN_PASSES = 3

NEGATIVE = (
    "blurry, distorted, deformed faces, extra limbs, text, watermark, subtitles, "
    "jpeg artifacts, static image, freeze frame, worst quality"
)

MODEL_FILES = {
    "unet": "wan2.2_ti2v_5B_fp16.safetensors",
    "clip": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "vae": "wan2.2_vae.safetensors",
}


def _get(base_url: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post_json(base_url: str, path: str, payload: dict, timeout: float = 60.0) -> dict:
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def upload_image(base_url: str, path: Path) -> str:
    """POST /upload/image (multipart) — returns the server-side filename."""
    boundary = uuid.uuid4().hex
    data = path.read_bytes()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{path.name}"\r\n'
        "Content-Type: image/png\r\n\r\n"
    ).encode() + data + (
        f"\r\n--{boundary}\r\n"
        'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue\r\n'
        f"--{boundary}--\r\n"
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/upload/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["name"]


def frame_count(seconds: float) -> int:
    """Wan latent time-compression wants length = 4k+1; cap at MAX_FRAMES."""
    frames = min(round(seconds * FPS), MAX_FRAMES)
    return max(4 * round((frames - 1) / 4) + 1, 9)


def plan_passes(seconds: float) -> list[int]:
    """Frame count for each generation pass needed to cover `seconds`.

    A continuation pass re-renders its anchor as its own first frame, so it
    contributes length-1 NEW frames to the finished clip; the caller drops that
    duplicate when stitching. Returns [n] for anything that fits in one pass,
    so short shots behave exactly as they did before chaining existed.
    """
    # The shortest legal pass is 9 frames, and a continuation pass spends one of
    # them re-rendering its anchor -- so it can never add fewer than 8 frames.
    # Chasing a gap smaller than that overshoots the shot instead of filling it;
    # leave it to the exporter, which holds the final frame for the last <0.34s.
    min_gain = 8

    target = max(round(seconds * FPS), 9)
    passes: list[int] = []
    covered = 0
    while covered < target and len(passes) < MAX_CHAIN_PASSES:
        remaining = target - covered
        if passes and remaining < min_gain:
            break
        # +1 on continuation passes to pay for the duplicated anchor frame
        want = remaining if not passes else remaining + 1
        length = frame_count(want / FPS)
        passes.append(length)
        covered += length if len(passes) == 1 else length - 1
    return passes


def generate_clip(
    base_url: str, ffmpeg: str, key_frame: Path, prompt: str, seconds: float,
    width: int, height: int, steps: int, cfg: float, seed: int, prefix: str,
    timeout: float, out_path: Path,
) -> tuple[int, int]:
    """Generate one shot's clip, chaining passes when it exceeds MAX_FRAMES.

    Returns (frames_written, passes_used).
    """
    passes = plan_passes(seconds)
    anchor = key_frame
    collected: list[Path] = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for i, length in enumerate(passes):
            if i:
                log.info("  pass %d/%d: %d frames, continuing from the previous frame",
                         i + 1, len(passes), length)
            image_name = upload_image(base_url, anchor)
            workflow = build_workflow(image_name, prompt, width, height, length,
                                      seed + i, steps, cfg, prefix=f"{prefix}_p{i}")
            prompt_id = _post_json(base_url, "/prompt", {"prompt": workflow})["prompt_id"]
            outputs = wait_for_prompt(base_url, prompt_id, timeout)

            pass_dir = tmp_root / f"pass{i}"
            pass_dir.mkdir()
            frames = download_frames(base_url, outputs, pass_dir)
            if len(frames) < length * 0.9:
                raise RuntimeError(
                    f"pass {i + 1}: expected ~{length} frames, got {len(frames)}")

            # frames[0] of a continuation pass is a re-render of the anchor
            collected.extend(frames if i == 0 else frames[1:])
            anchor = frames[-1]

        assemble(ffmpeg, collected, out_path)
    return len(collected), len(passes)


def motion_prompt(shot: dict, style: str) -> str:
    """Ground-truth action + camera as the motion guide; style keeps the look."""
    return (
        f"{shot.get('action', '')}. {shot.get('camera', '')}. "
        f"{style}. subtle natural motion, coherent characters, gentle camera movement"
    )


def build_workflow(
    image_name: str, prompt: str, width: int, height: int, length: int,
    seed: int, steps: int, cfg: float, prefix: str,
) -> dict:
    """Wan 2.2 TI2V-5B image-to-video graph (ComfyUI API format), frames via SaveImage."""
    return {
        "37": {"class_type": "UNETLoader",
               "inputs": {"unet_name": MODEL_FILES["unet"], "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader",
               "inputs": {"clip_name": MODEL_FILES["clip"], "type": "wan", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": MODEL_FILES["vae"]}},
        "56": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": prompt}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["38", 0], "text": NEGATIVE}},
        "55": {"class_type": "Wan22ImageToVideoLatent",
               "inputs": {"vae": ["39", 0], "start_image": ["56", 0],
                          "width": width, "height": height, "length": length, "batch_size": 1}},
        "3": {"class_type": "KSampler",
              "inputs": {"model": ["37", 0], "positive": ["6", 0], "negative": ["7", 0],
                         "latent_image": ["55", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": "uni_pc", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
        "60": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def wait_for_prompt(base_url: str, prompt_id: str, timeout: float) -> dict:
    """Poll /history until the prompt completes; returns its outputs.

    Poll failures are survivable and must NOT abort the shot. While the sampler
    is running, ComfyUI's HTTP thread can be starved long enough that /history
    doesn't answer inside the socket timeout -- that says nothing about the
    generation, which is happily proceeding on the GPU. Treat any transient
    error as "not ready yet" and keep polling until the real `timeout` budget
    is spent; only a hard error status from ComfyUI itself ends the wait early.
    """
    start = time.monotonic()
    consecutive_errors = 0
    while time.monotonic() - start < timeout:
        try:
            # generous per-request timeout: a slow answer is still an answer
            history = _get(base_url, f"/history/{prompt_id}", timeout=120.0).get(prompt_id)
            consecutive_errors = 0
        except Exception as exc:  # noqa: BLE001 -- any poll failure is transient
            consecutive_errors += 1
            log.debug("poll %d for %s failed (%s); still waiting",
                      consecutive_errors, prompt_id, exc)
            if consecutive_errors in (5, 20, 60):
                log.warning("ComfyUI has not answered %d consecutive polls (%s) — "
                            "still waiting, generation is probably still running",
                            consecutive_errors, exc)
            time.sleep(5)
            continue

        if history:
            status = history.get("status", {})
            if status.get("status_str") == "error":
                messages = json.dumps(status.get("messages", []))[:400]
                raise RuntimeError(f"ComfyUI prompt failed: {messages}")
            if history.get("outputs"):
                return history["outputs"]
        time.sleep(5)
    raise TimeoutError(f"prompt {prompt_id} did not finish within {timeout:.0f}s")


def download_frames(base_url: str, outputs: dict, dest: Path) -> list[Path]:
    """Fetch every SaveImage frame via /view, in order."""
    frames: list[Path] = []
    for node_output in outputs.values():
        for image in node_output.get("images", []):
            query = urllib.parse.urlencode(
                {"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                 "type": image.get("type", "output")}
            )
            path = dest / image["filename"]
            with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=60) as r:
                path.write_bytes(r.read())
            frames.append(path)
    return sorted(frames)


def assemble(ffmpeg: str, frames: list[Path], out_path: Path) -> None:
    """Frames -> H.264 mp4 (video only; the exporter adds audio + padding)."""
    list_file = frames[0].parent / "frames.txt"
    list_file.write_text(
        "".join(f"file '{f.as_posix()}'\nduration {1 / FPS}\n" for f in frames), encoding="utf-8"
    )
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(list_file),
         "-vf", "format=yuv420p", "-r", str(FPS), "-c:v", "libx264", "-crf", "18",
         "-movflags", "+faststart", str(out_path)],
        check=True,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("board", type=Path)
    parser.add_argument("--frames", type=Path, default=None, help="key-frame dir (default: <board dir>/frames)")
    parser.add_argument("--out", type=Path, default=None, help="clip dir (default: <board dir>/animated)")
    parser.add_argument("--shots", default=None, help="subset like '5' or '1-4,6' (default: all)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    # 720p / 35 steps = Wan 2.2 5B's native quality band. ~11 min/shot warm on a
    # 5070 Ti. For fast iteration drop to --width 736 --height 416 --steps 20
    # (~3 min/shot, draft quality — noticeably softer and shimmerier).
    parser.add_argument("--width", type=int, default=1280, help="multiple of 32 (Wan 5B native 1280)")
    parser.add_argument("--height", type=int, default=704, help="multiple of 32 (Wan 5B native 704)")
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=1234, help="base seed; shot N uses seed+N")
    parser.add_argument("--timeout", type=float, default=3600.0, help="per-shot generation timeout, seconds")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    board = json.loads(args.board.read_text(encoding="utf-8"))
    shots = board.get("shots", [])
    frames_dir = args.frames or args.board.parent / "frames"
    out_dir = (args.out or args.board.parent / "animated").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = find_ffmpeg()

    selection = parse_shot_selection(args.shots, len(shots))
    by_number = {shot["shot"]: shot for shot in shots}
    failures = done = 0
    for number in selection:
        shot = by_number[number]
        key_frame = frames_dir / f"shot_{number:03d}.png"
        clip_path = out_dir / f"shot_{number:03d}.mp4"
        if not key_frame.exists():
            log.warning("shot %d: no key frame at %s, skipping", number, key_frame)
            continue
        if clip_path.exists() and not args.force:
            log.info("shot %d: animated clip exists, skipping (use --force)", number)
            done += 1
            continue

        seconds = shot_duration(shot)
        passes = plan_passes(seconds)
        covered = (passes[0] + sum(n - 1 for n in passes[1:])) / FPS
        prompt = motion_prompt(shot, board.get("style", ""))
        detail = f"{len(passes)} passes" if len(passes) > 1 else "1 pass"
        log.info("shot %d (%s): %d frames (%.1fs of %.1fs, %s) — generating...",
                 number, shot.get("beat"), round(covered * FPS), covered, seconds, detail)
        if covered < seconds - 0.1:
            log.warning("shot %d: %.1fs of %.1fs — capped at %d passes; the exporter "
                        "will hold the final frame for the remaining %.1fs",
                        number, covered, seconds, MAX_CHAIN_PASSES, seconds - covered)
        start = time.monotonic()
        try:
            written, used = generate_clip(
                args.base_url, ffmpeg, key_frame, prompt, seconds,
                args.width, args.height, args.steps, args.cfg,
                args.seed + number, f"wan_shot_{number:03d}", args.timeout, clip_path,
            )
            done += 1
            log.info("shot %d: %s (%d frames, %.1fs, %d pass(es), %.1f min)",
                     number, clip_path, written, written / FPS, used,
                     (time.monotonic() - start) / 60)
        except Exception as exc:  # noqa: BLE001 — per-shot isolation, report and continue
            failures += 1
            log.error("shot %d failed after %.1f min: %s", number, (time.monotonic() - start) / 60, exc)

    log.info("animated %d shot(s), %d failure(s) -> %s", done, failures, out_dir)
    return 0 if failures == 0 and done else 1


if __name__ == "__main__":
    sys.exit(main())
