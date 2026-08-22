"""Contract validation, including the configure-livelock guard."""

from __future__ import annotations

import pytest

from worldengine.contract import IntentError, validate_intent
from worldengine.models import Event, ExperimentResult, Intent
from worldengine.world import World

INTEGRATORS = ("explicit_euler", "symplectic_euler", "rk4")
ENTITY = 1


def world_with(config: dict | None) -> World:
    w = World()
    if config is not None:
        w.apply(
            Event(
                kind="experiment.configured",
                entity=ENTITY,
                data={"set_components": {"sim_config": config}},
            )
        )
    return w


def result(integrator: str, dt: float, passed: bool = False) -> ExperimentResult:
    return ExperimentResult(
        experiment=1,
        integrator=integrator,
        dt=dt,
        duration=30.0,
        max_drift_pct=41.8,
        passed=passed,
    )


def configure(integrator: str = "rk4", dt: float = 0.01) -> Intent:
    return Intent(action="configure", args={"integrator": integrator, "dt": dt})


class TestStatelessChecks:
    def test_unknown_integrator_is_rejected(self) -> None:
        with pytest.raises(IntentError, match="unknown integrator"):
            validate_intent(configure("implicitfast"), world_with(None), ENTITY, INTEGRATORS)

    @pytest.mark.parametrize("dt", [0.0, -1.0, 0.5, "fast", True, None])
    def test_bad_dt_is_rejected(self, dt: object) -> None:
        intent = Intent(action="configure", args={"integrator": "rk4", "dt": dt})
        with pytest.raises(IntentError, match="dt must be"):
            validate_intent(intent, world_with(None), ENTITY, INTEGRATORS)

    def test_run_before_configure_is_rejected(self) -> None:
        intent = Intent(action="run", args={"duration": 30.0})
        with pytest.raises(IntentError, match="no experiment configured"):
            validate_intent(intent, world_with(None), ENTITY, INTEGRATORS)

    def test_conclude_requires_a_summary(self) -> None:
        with pytest.raises(IntentError, match="summary"):
            validate_intent(
                Intent(action="conclude", args={"summary": "   "}),
                world_with(None),
                ENTITY,
                INTEGRATORS,
            )


class TestConfigureLivelockGuard:
    """A Director that reconfigures without ever running learns nothing.

    Observed live: a blinded Director burned seven consecutive rounds
    alternating two configure intents and never emitted a single run.
    """

    def test_reconfiguring_an_unrun_experiment_is_rejected(self) -> None:
        world = world_with({"integrator": "symplectic_euler", "dt": 0.01})
        with pytest.raises(IntentError, match="has no result yet"):
            validate_intent(configure("rk4"), world, ENTITY, INTEGRATORS, results=[])

    def test_the_rejection_names_what_to_do_instead(self) -> None:
        world = world_with({"integrator": "symplectic_euler", "dt": 0.01})
        with pytest.raises(IntentError, match="send run"):
            validate_intent(configure("rk4"), world, ENTITY, INTEGRATORS, results=[])

    def test_reconfiguring_after_a_result_is_allowed(self) -> None:
        world = world_with({"integrator": "symplectic_euler", "dt": 0.01})
        validate_intent(
            configure("rk4"),
            world,
            ENTITY,
            INTEGRATORS,
            results=[result("symplectic_euler", 0.01)],
        )

    def test_dt_is_compared_numerically(self) -> None:
        """0.01 and 1e-2 are the same experiment, not two."""
        world = world_with({"integrator": "rk4", "dt": 0.01})
        validate_intent(
            configure("explicit_euler"),
            world,
            ENTITY,
            INTEGRATORS,
            results=[result("rk4", 1e-2)],
        )

    def test_a_result_for_a_different_dt_does_not_count(self) -> None:
        world = world_with({"integrator": "rk4", "dt": 0.01})
        with pytest.raises(IntentError, match="has no result yet"):
            validate_intent(
                configure("explicit_euler"),
                world,
                ENTITY,
                INTEGRATORS,
                results=[result("rk4", 0.02)],
            )

    def test_run_is_always_allowed_while_a_config_is_pending(self) -> None:
        world = world_with({"integrator": "rk4", "dt": 0.01})
        validate_intent(
            Intent(action="run", args={"duration": 30.0}), world, ENTITY, INTEGRATORS, results=[]
        )

    def test_conclude_is_always_allowed_while_a_config_is_pending(self) -> None:
        """Never trap a Director that wants to stop."""
        world = world_with({"integrator": "rk4", "dt": 0.01})
        validate_intent(
            Intent(action="conclude", args={"summary": "done"}),
            world,
            ENTITY,
            INTEGRATORS,
            results=[],
        )

    def test_first_configure_is_allowed(self) -> None:
        validate_intent(configure(), world_with(None), ENTITY, INTEGRATORS, results=[])

    def test_omitting_results_skips_the_guard(self) -> None:
        """Without results we cannot tell a livelock from a legitimate
        reconfigure, so the stateful check is skipped rather than guessed.
        Existing callers keep exactly their old behaviour."""
        world = world_with({"integrator": "rk4", "dt": 0.01})
        validate_intent(configure("explicit_euler"), world, ENTITY, INTEGRATORS)

    def test_empty_results_still_triggers_the_guard(self) -> None:
        """[] means "nothing has run", which is different from "I don't know"."""
        world = world_with({"integrator": "rk4", "dt": 0.01})
        with pytest.raises(IntentError, match="has no result yet"):
            validate_intent(configure("explicit_euler"), world, ENTITY, INTEGRATORS, results=[])


class TestMemoryOptOut:
    """The Director's view is the Observation; injected memory bypasses it."""

    def _headers(self, monkeypatch, env: dict[str, str]) -> dict[str, str]:
        import json as _json
        import urllib.request

        from worldengine import director as d

        for k in ("WORLD_ENGINE_LLM_MEMORY", "WORLD_ENGINE_LLM_CALLER"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)

        captured: dict[str, dict[str, str]] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return _json.dumps({"choices": [{"message": {"content": "{}"}}]}).encode()

        def fake_urlopen(request, timeout=None):
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return FakeResponse()

        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        d.chat_completion("http://x/v1", "m", [{"role": "user", "content": "hi"}])
        return captured["headers"]

    def test_memory_is_disabled_by_default(self, monkeypatch) -> None:
        assert self._headers(monkeypatch, {})["x-memory"] == "0"

    def test_memory_can_be_re_enabled(self, monkeypatch) -> None:
        assert "x-memory" not in self._headers(monkeypatch, {"WORLD_ENGINE_LLM_MEMORY": "1"})

    def test_caller_header_still_works(self, monkeypatch) -> None:
        h = self._headers(monkeypatch, {"WORLD_ENGINE_LLM_CALLER": "world-engine-director"})
        assert h["x-caller"] == "world-engine-director"
        assert h["x-memory"] == "0"
