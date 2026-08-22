"""Blind the Director to integrator *names* so the loop is tested, not recall.

Why this exists. The pendulum benchmark offers `explicit_euler`,
`symplectic_euler` and `rk4`. Any competent model knows symplectic integrators
conserve energy in Hamiltonian systems, so an LLM Director picks the winner on
round 0 with zero results in hand. Measured on Qwen3.5-4B, 8/8 samples went
straight to `symplectic_euler` before running a single experiment. That is
recall, not discovery -- the observe/decide/verify loop never gets exercised,
and the demo cannot fail in a way that would tell you the loop is broken.

Blinding the labels separates the two. Same world, same verifier, same typed
contract; the Director just cannot rank the options by name. Measured with the
same model and blinded labels:

  * round 0, nothing tried      -> picks a baseline (8/8)
  * one option failed twice     -> moves off it (7/8)
  * two options failed          -> picks the remaining untried one (8/8)
  * a different option failed   -> picks an untried one (8/8)

That is evidence-driven exploration rather than positional habit, and it is the
behaviour the engine actually claims. The named benchmark simply cannot show it.

Design. This wraps a *Director*, not a backend, so blinding is a perception
transform and nothing else changes: the world, the event log and the verifier
all keep using real integrator names, and a run stays replayable and honest.
Only the text handed to the model is relabelled.
"""

from __future__ import annotations

import random
from typing import Protocol

from .models import Intent, Observation

__all__ = ["BlindedDirector", "blind_labels"]

#: Opaque, order-free labels. Deliberately not "option_1/2/3" -- ordinal names
#: invite ordinal strategies, which is one of the behaviours worth ruling out.
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


class _Director(Protocol):
    def decide(self, observation: Observation) -> Intent: ...


def blind_labels(names: list[str] | tuple[str, ...], seed: int = 0) -> dict[str, str]:
    """Map real integrator names to opaque labels.

    The mapping is shuffled under ``seed`` so the winner does not sit at a fixed
    position across runs -- otherwise "always pick the second one" would score
    as discovery. Deterministic given the seed, so a run stays reproducible.
    """
    reals = list(names)
    labels = [f"method_{_ALPHABET[i]}" for i in range(len(reals))]
    random.Random(seed).shuffle(labels)
    return dict(zip(reals, labels))


class BlindedDirector:
    """Wrap a Director so it sees opaque integrator labels instead of real names.

    The Observation handed to the inner Director has its
    ``available_integrators``, ``configured`` and ``results`` relabelled; the
    Intent that comes back is translated straight back into real names before
    the engine ever sees it.
    """

    def __init__(self, inner: _Director, seed: int = 0) -> None:
        self._inner = inner
        self._seed = seed
        self._to_blind: dict[str, str] = {}
        self._to_real: dict[str, str] = {}

    def _ensure_mapping(self, names: list[str]) -> None:
        # Built once, from the first observation that carries capabilities, so
        # the labels stay stable for the whole run.
        if not self._to_blind and names:
            self._to_blind = blind_labels(names, self._seed)
            self._to_real = {v: k for k, v in self._to_blind.items()}

    def _blind(self, name: str | None) -> str | None:
        return self._to_blind.get(name, name) if name else name

    def decide(self, observation: Observation) -> Intent:
        self._ensure_mapping(list(observation.available_integrators))

        obs = observation.model_copy(deep=True)
        obs.available_integrators = [
            self._to_blind.get(n, n) for n in obs.available_integrators
        ]
        if obs.configured and "integrator" in obs.configured:
            obs.configured = {
                **obs.configured,
                "integrator": self._blind(obs.configured["integrator"]),
            }
        for result in obs.results:
            real = getattr(result, "integrator", None)
            if real is not None:
                result.integrator = self._blind(real)
        # Notes are free text written by the verifier and can name integrators.
        obs.notes = [self._relabel_text(n) for n in obs.notes]

        intent = self._inner.decide(obs)

        args = dict(intent.args or {})
        if "integrator" in args:
            args["integrator"] = self._to_real.get(args["integrator"], args["integrator"])
        # Free-text the Director writes (notably a conclude summary) can name the
        # labels it was shown. Translate it back so the event log and every run
        # artifact stay readable in real names -- nothing downstream of this
        # wrapper should ever have to know blinding happened.
        if isinstance(args.get("summary"), str):
            args["summary"] = self._restore_text(args["summary"])
        return intent.model_copy(
            update={"args": args, "rationale": self._restore_text(intent.rationale or "")}
        )

    def _relabel_text(self, text: str) -> str:
        # Longest-first so a name that is a prefix of another cannot corrupt it.
        for real in sorted(self._to_blind, key=len, reverse=True):
            text = text.replace(real, self._to_blind[real])
        return text

    def _restore_text(self, text: str) -> str:
        for blind in sorted(self._to_real, key=len, reverse=True):
            text = text.replace(blind, self._to_real[blind])
        return text

    @property
    def mapping(self) -> dict[str, str]:
        """Real name -> blinded label, for the run manifest and for debugging."""
        return dict(self._to_blind)
