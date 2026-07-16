"""Deterministic invariant checks over ground truth — the verifiable reward.

The verifier, not the Director, judges outcomes. It reads only backend samples
(never LLM output), so a pass/fail verdict is always anchored in the engine's
ground truth.
"""

from __future__ import annotations

from typing import Protocol

from .models import Verdict


class Verifier(Protocol):
    """Protocol every verifier implements."""

    def check(self, samples: list[dict[str, float]]) -> Verdict:
        """Judge one experiment's sample stream."""
        ...


class EnergyConservation:
    """Passes iff max relative energy drift across a run stays under tolerance.

    The true (undamped) dynamics conserve energy exactly, so any drift in the
    samples is integrator error — a crisp, unfakeable signal.
    """

    def __init__(self, tolerance_pct: float = 1.0) -> None:
        self.tolerance_pct = tolerance_pct

    def check(self, samples: list[dict[str, float]]) -> Verdict:
        """Compute max |E - E0| / |E0| over the run and compare to tolerance."""
        if len(samples) < 2:
            return Verdict(passed=False, reason="not enough samples to judge", metrics={})
        e0 = samples[0]["energy"]
        if e0 == 0:
            return Verdict(passed=False, reason="reference energy is zero", metrics={})
        drift = max(abs(s["energy"] - e0) / abs(e0) for s in samples) * 100.0
        passed = drift <= self.tolerance_pct
        verb = "within" if passed else "exceeds"
        return Verdict(
            passed=passed,
            reason=f"max energy drift {drift:.4f}% {verb} tolerance {self.tolerance_pct:g}%",
            metrics={"max_drift_pct": drift},
        )
