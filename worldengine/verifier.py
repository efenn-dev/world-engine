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
        samples = [s for s in samples if "energy" in s]  # ignore non-energy diagnostics
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


class Rendezvous:
    """Passes iff every agent occupies the same place on the final tick sample.

    Ground truth for the social backend: co-location is read from the
    backend's ``social.tick`` snapshots, never from what agents *say* — a
    character claiming "I found you!" in a different room does not pass.
    """

    def __init__(self, place: str | None = None) -> None:
        self.place = place  # optionally require a specific meeting place

    def check(self, samples: list[dict[str, float]]) -> Verdict:
        """Judge co-location on the last social.tick snapshot of the run."""
        ticks = [s for s in samples if s.get("kind") == "social.tick"]
        if not ticks:
            return Verdict(passed=False, reason="no tick samples to judge", metrics={})
        locations: dict[str, str] = ticks[-1]["locations"]
        spots = set(locations.values())
        together = len(spots) == 1
        at_target = self.place is None or spots == {self.place}
        passed = together and at_target
        if passed:
            reason = f"all {len(locations)} agents together at {next(iter(spots))!r} after {len(ticks)} ticks"
        elif together:
            reason = f"agents together at {next(iter(spots))!r} but required place is {self.place!r}"
        else:
            spread = ", ".join(f"{name}@{place}" for name, place in sorted(locations.items()))
            reason = f"agents in {len(spots)} different places after {len(ticks)} ticks ({spread})"
        return Verdict(
            passed=passed,
            reason=reason,
            metrics={"ticks": float(len(ticks)), "distinct_places_final": float(len(spots))},
        )
