"""Projectors — pure views from engine output to a medium.

Same truth, many renderings: a log line today, an energy plot, a storyboard
prompt or a ClipLab timeline tomorrow. Projectors never mutate world state,
so adding an output medium never touches the engine.
"""

from __future__ import annotations

import logging
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
        for result in outcome.results:
            samples = outcome.samples.get(result.experiment, [])
            if not samples:
                continue
            ax.plot(
                [s["t"] for s in samples],
                [s["energy"] for s in samples],
                label=f"#{result.experiment} {result.integrator} dt={result.dt:g} "
                f"({'pass' if result.passed else 'fail'})",
            )
        first = next(iter(outcome.samples.values()), [])
        if first:
            ax.axhline(first[0]["energy"], color="gray", linestyle="--", linewidth=1, label="E0 (truth)")
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


class StoryboardProjector(Projector):
    """Placeholder for the Track-3 swap: event slices → shot prompts → media.

    The intended shape: read narrative-relevant events (who did what, where,
    verdicts, reversals), group them into beats, emit one shot-prompt per beat
    (character sheet + scene + camera), and hand the batch to an image/video
    pipeline (SDXL → ClipLab). Deliberately unimplemented in this slice.
    """
