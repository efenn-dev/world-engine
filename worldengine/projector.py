"""Projectors — pure views from engine output to a medium.

Same truth, many renderings: a log line today, an energy plot, a storyboard
prompt or a ClipLab timeline tomorrow. Projectors never mutate world state,
so adding an output medium never touches the engine.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

from .models import ExperimentResult, Outcome

log = logging.getLogger(__name__)


class Projector:
    """No-op base; override what you need."""

    def on_result(self, result: ExperimentResult) -> None:
        """Called after each completed experiment."""

    def finalize(self, outcome: Outcome) -> None:
        """Called once when the session ends."""


class TextProjector(Projector):
    """Renders results and the conclusion as log lines."""

    def on_result(self, result: ExperimentResult) -> None:
        status = "PASS" if result.passed else "FAIL"
        if math.isnan(result.max_drift_pct):  # non-physics verifier — no drift metric
            log.info(
                "[%s] experiment %d: %s over %g", status, result.experiment, result.integrator, result.duration
            )
        else:
            log.info(
                "[%s] experiment %d: %s dt=%g -> max drift %.4f%%",
                status,
                result.experiment,
                result.integrator,
                result.dt,
                result.max_drift_pct,
            )

    def finalize(self, outcome: Outcome) -> None:
        if outcome.concluded:
            log.info("conclusion after %d rounds: %s", outcome.rounds, outcome.summary)
        else:
            log.info("no conclusion after %d rounds", outcome.rounds)


class PlotProjector(Projector):
    """Writes an energy-over-time PNG for all experiments (needs matplotlib).

    Degrades gracefully: if matplotlib is not installed, logs a hint and skips.
    """

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)

    def finalize(self, outcome: Outcome) -> None:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            log.warning("matplotlib not installed — skipping energy plot (pip install matplotlib)")
            return
        if not outcome.results:
            return
        fig, ax = plt.subplots(figsize=(9, 5))
        first_energy: float | None = None
        for result in outcome.results:
            samples = [s for s in outcome.samples.get(result.experiment, []) if "energy" in s]
            if not samples:
                continue
            if first_energy is None:
                first_energy = samples[0]["energy"]
            ax.plot(
                [s["t"] for s in samples],
                [s["energy"] for s in samples],
                label=f"#{result.experiment} {result.integrator} dt={result.dt:g} "
                f"({'pass' if result.passed else 'fail'})",
            )
        if first_energy is None:  # nothing energy-shaped to plot (e.g. social backend)
            plt.close(fig)
            return
        ax.axhline(first_energy, color="gray", linestyle="--", linewidth=1, label="E0 (truth)")
        ax.set_xlabel("t (s)")
        ax.set_ylabel("total energy (J)")
        ax.set_title("Energy drift by integrator — ground truth conserves E exactly")
        ax.legend(fontsize=8)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / "energy.png"
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        log.info("energy plot written to %s", path)


class TranscriptProjector(Projector):
    """Renders social-backend events as a readable script — the Track-3 seed.

    Same truth, different projection: moves and dialogue from the event stream
    become script lines. Logs them and, if ``out_dir`` is given, writes
    ``transcript.txt``. This is the precursor of StoryboardProjector: replace
    "format a line" with "emit a shot prompt" and the pipeline is the same.
    """

    def __init__(self, out_dir: str | Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir is not None else None

    def finalize(self, outcome: Outcome) -> None:
        lines: list[str] = []
        for result in outcome.results:
            episode: list[str] = []
            for s in outcome.samples.get(result.experiment, []):
                kind = s.get("kind")
                if kind == "agent.moved":
                    episode.append(f"[t{int(s['t'])}] {s['name']}: {s['from']} -> {s['to']}")
                elif kind == "agent.spoke":
                    episode.append(f"[t{int(s['t'])}] {s['name']} ({s['place']}): \"{s['text']}\"")
                elif kind == "agent.confused":
                    episode.append(f"[t{int(s['t'])}] {s['name']} hesitates ({s['reason']})")
            if episode:
                lines.append(f"--- episode {result.experiment} ({result.duration:g} ticks, "
                             f"{'rendezvous' if result.passed else 'no rendezvous'}) ---")
                lines.extend(episode)
        if not lines:
            return
        for line in lines:
            log.info("%s", line)
        if outcome.concluded:
            lines.append(f"--- conclusion: {outcome.summary} ---")
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            path = self.out_dir / "transcript.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            log.info("transcript written to %s", path)


# StoryboardProjector (Track 3: beats → shot prompts) lives in worldengine.storyboard —
# it is a full subsystem (beat segmentation, prompt assembly, LLM enrichment).
