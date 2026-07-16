"""Directors — the reasoning layer. They propose typed intents; they never touch state.

`ScriptedDirector` is a deterministic offline stand-in (and the test fixture).
`LLMDirector` calls an OpenAI-compatible endpoint — by default the local
llm-router gateway on 127.0.0.1:4000 — and parses strict-JSON intents. Both
implement the same one-method protocol, so the engine cannot tell them apart.
"""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.request
from typing import Protocol

from pydantic import ValidationError

from .models import Intent, Observation

log = logging.getLogger(__name__)


class Director(Protocol):
    """Protocol every director implements: one decision per world round."""

    def decide(self, observation: Observation) -> Intent:
        """Turn a scoped observation into one typed intent."""
        ...


class DirectorError(RuntimeError):
    """The director failed to produce a valid intent."""


def llm_defaults(
    base_url: str | None = None,
    model: str | None = None,
    route: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str | None, str | None]:
    """Resolve LLM endpoint config from args, falling back to WORLD_ENGINE_LLM_* env vars."""
    return (
        (base_url or os.environ.get("WORLD_ENGINE_LLM_BASE", "http://127.0.0.1:4000/v1")).rstrip("/"),
        model or os.environ.get("WORLD_ENGINE_LLM_MODEL", "auto"),
        route or os.environ.get("WORLD_ENGINE_LLM_ROUTE") or None,
        api_key or os.environ.get("WORLD_ENGINE_LLM_KEY") or None,
    )


def chat_completion(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    route: str | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    temperature: float = 0.2,
) -> str:
    """POST an OpenAI-compatible chat completion; return the reply text.

    Shared by `LLMDirector` and `backends.social.LLMMind` — one HTTP path for
    every LLM in the system.
    """
    payload: dict[str, object] = {"model": model, "messages": messages, "temperature": temperature}
    if route:
        payload["route"] = route
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read().decode("utf-8"))
    return str(body["choices"][0]["message"]["content"])


#: (integrator, dt, rationale) — a hypothesis arc for the pure-Python pendulum backend
PENDULUM_PLAN: tuple[tuple[str, float, str], ...] = (
    ("explicit_euler", 0.02, "baseline: naive integrator at coarse dt"),
    ("explicit_euler", 0.002, "hypothesis: drift is step-size error — refine dt 10x"),
    (
        "symplectic_euler",
        0.005,
        "hypothesis: drift is structural to explicit Euler — switch to a symplectic method",
    ),
    ("rk4", 0.02, "fallback: high-order integrator"),
)


class ScriptedDirector:
    """Deterministic hypothesis arc, reacting to verifier feedback.

    Walks a (integrator, dt, rationale) plan one experiment at a time,
    concludes on the first verifier pass, and concedes if the plan runs out.
    The default plan targets the pure-Python pendulum backend; pass a
    different plan for other backends (see demos/pendulum_energy.py).
    """

    def __init__(
        self,
        run_duration: float = 30.0,
        plan: tuple[tuple[str, float, str], ...] = PENDULUM_PLAN,
    ) -> None:
        self.run_duration = run_duration
        self.plan = plan
        self._pending_run = False

    def decide(self, observation: Observation) -> Intent:
        """Configure the next experiment, run it, or conclude on a pass."""
        results = observation.results
        if results and results[-1].passed:
            best = results[-1]
            failed = ", ".join(f"{r.integrator}@dt={r.dt:g}" for r in results if not r.passed)
            return Intent(
                action="conclude",
                args={
                    "summary": (
                        f"{best.integrator} at dt={best.dt:g} conserves energy "
                        f"(max drift {best.max_drift_pct:.4f}% over {best.duration:g}s)"
                        + (f"; failed along the way: {failed}." if failed else ".")
                    )
                },
                rationale="verifier passed — hypothesis confirmed",
            )
        if self._pending_run:
            self._pending_run = False
            return Intent(
                action="run",
                args={"duration": self.run_duration},
                rationale="run the configured experiment",
            )
        attempt = len(results)
        if attempt >= len(self.plan):
            return Intent(
                action="conclude",
                args={"summary": "no tested configuration conserved energy within tolerance"},
                rationale="plan exhausted",
            )
        integrator, dt, rationale = self.plan[attempt]
        self._pending_run = True
        return Intent(
            action="configure",
            args={"integrator": integrator, "dt": dt},
            rationale=rationale,
        )


_SYSTEM_PROMPT = """You are the Director of a physics world-engine. Your goal is stated in each observation.

You interact ONLY by emitting exactly one JSON intent per turn, chosen from:
  {"action": "configure", "args": {"integrator": "<one of the observation's available_integrators>", "dt": <number, 0 < dt <= 0.1>}, "rationale": "<why>"}
  {"action": "run", "args": {"duration": <seconds, 0 < duration <= 300>}, "rationale": "<why>"}
  {"action": "conclude", "args": {"summary": "<what you learned>"}, "rationale": "<why>"}

Rules:
- configure before the first run; every run restarts from the same initial conditions.
- args.integrator MUST be copied verbatim from the observation's available_integrators list —
  never use an integrator name that is not in that list, even if you know it from physics.
- If available_integrators is empty there is nothing to configure — skip configure and go
  straight to run.
- The observation's "configured" field shows the currently configured experiment (null if none).
  If it is set and there is no result for it yet, send run — do NOT reconfigure the same settings.
- A deterministic verifier (not you) judges energy conservation; read its results and notes, then adapt.
- The moment ANY result has "passed": true, your next intent MUST be conclude (summarize what
  you learned). Do not keep exploring after a pass.
- Respond with ONLY the JSON object — no prose, no code fences, no thinking text."""


class LLMDirector:
    """Director backed by an OpenAI-compatible chat endpoint (default: llm-router :4000).

    Config (constructor args win over env vars):
      - base URL: ``WORLD_ENGINE_LLM_BASE``  (default http://127.0.0.1:4000/v1)
      - model:    ``WORLD_ENGINE_LLM_MODEL`` (default "auto"; the router routes semantically)
      - route:    ``WORLD_ENGINE_LLM_ROUTE`` (optional llm-router force-pin, e.g. local_reason)
      - api key:  ``WORLD_ENGINE_LLM_KEY``   (optional; local routing works keyless)
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        route: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url, self.model, self.route, self.api_key = llm_defaults(
            base_url, model, route, api_key
        )
        self.timeout = timeout

    def decide(self, observation: Observation) -> Intent:
        """Ask the LLM for one intent; retry once with the validation error as feedback."""
        user = observation.model_dump_json(indent=2)
        error_note: str | None = None
        for _ in range(2):
            prompt = user if error_note is None else (
                f"{user}\n\nYour previous reply was invalid: {error_note}\n"
                "Reply with ONLY the JSON intent object."
            )
            content = self._chat(prompt)
            try:
                return Intent.model_validate_json(extract_json(content))
            except (ValidationError, ValueError) as exc:
                error_note = str(exc)
                log.warning("LLM intent parse failed, retrying: %s", exc)
        raise DirectorError(f"LLM failed to produce a valid intent: {error_note}")

    def _chat(self, user_content: str) -> str:
        return chat_completion(
            self.base_url,
            self.model,
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            route=self.route,
            api_key=self.api_key,
            timeout=self.timeout,
        )


def extract_json(text: str) -> str:
    """Pull the outermost JSON object out of an LLM reply (fences, think-blocks, prose)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError(f"no JSON object found in reply: {text[:200]!r}")
    return text[start : end + 1]
