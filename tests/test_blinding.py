"""Tests for the Director label-blinding wrapper."""

from __future__ import annotations

from worldengine.blinding import BlindedDirector, blind_labels
from worldengine.models import ExperimentResult, Intent, Observation

REAL = ["explicit_euler", "symplectic_euler", "rk4"]


class Spy:
    """Records the Observation it was handed and replays a canned Intent."""

    def __init__(self, intent: Intent) -> None:
        self.intent = intent
        self.seen: Observation | None = None

    def decide(self, observation: Observation) -> Intent:
        self.seen = observation
        return self.intent


def observation(**kw) -> Observation:
    base = {
        "round": 0,
        "goal": "conserve energy",
        "available_integrators": list(REAL),
        "configured": None,
        "results": [],
        "notes": [],
    }
    base.update(kw)
    return Observation(**base)


class TestMapping:
    def test_every_name_gets_a_distinct_label(self) -> None:
        m = blind_labels(REAL)
        assert len(set(m.values())) == len(REAL)
        assert all(v.startswith("method_") for v in m.values())

    def test_mapping_is_deterministic_under_a_seed(self) -> None:
        assert blind_labels(REAL, seed=7) == blind_labels(REAL, seed=7)

    def test_seed_changes_the_permutation(self) -> None:
        """The winner must not sit at a fixed position across seeds."""
        seeds = {tuple(blind_labels(REAL, seed=s).values()) for s in range(12)}
        assert len(seeds) > 1


class TestObservationIsBlinded:
    def test_available_integrators_are_relabelled(self) -> None:
        spy = Spy(Intent(action="run", args={"duration": 30.0}))
        BlindedDirector(spy).decide(observation())
        assert spy.seen is not None
        assert all(n.startswith("method_") for n in spy.seen.available_integrators)
        assert not set(spy.seen.available_integrators) & set(REAL)

    def test_results_and_configured_are_relabelled(self) -> None:
        spy = Spy(Intent(action="run", args={"duration": 30.0}))
        obs = observation(
            configured={"integrator": "explicit_euler", "dt": 0.02},
            results=[
                ExperimentResult(
                    experiment=1,
                    integrator="explicit_euler",
                    dt=0.02,
                    duration=30.0,
                    max_drift_pct=41.8,
                    passed=False,
                )
            ],
        )
        BlindedDirector(spy).decide(obs)
        assert spy.seen is not None
        assert spy.seen.configured["integrator"].startswith("method_")
        assert spy.seen.results[0].integrator.startswith("method_")

    def test_notes_do_not_leak_real_names(self) -> None:
        spy = Spy(Intent(action="run", args={"duration": 30.0}))
        obs = observation(notes=["explicit_euler at dt=0.02 gained 41.8% energy"])
        BlindedDirector(spy).decide(obs)
        assert spy.seen is not None
        assert "explicit_euler" not in spy.seen.notes[0]
        assert "method_" in spy.seen.notes[0]

    def test_the_callers_observation_is_not_mutated(self) -> None:
        spy = Spy(Intent(action="run", args={"duration": 30.0}))
        obs = observation(notes=["explicit_euler drifted"])
        BlindedDirector(spy).decide(obs)
        assert obs.available_integrators == REAL
        assert obs.notes == ["explicit_euler drifted"]


class TestIntentIsRestored:
    def test_integrator_comes_back_in_real_names(self) -> None:
        wrapper = BlindedDirector(Spy(Intent(action="run", args={})))
        wrapper.decide(observation())
        label = wrapper.mapping["symplectic_euler"]

        spy = Spy(Intent(action="configure", args={"integrator": label, "dt": 0.005}))
        inner = BlindedDirector(spy)
        inner._to_blind = dict(wrapper.mapping)
        inner._to_real = {v: k for k, v in wrapper.mapping.items()}
        out = inner.decide(observation())
        assert out.args["integrator"] == "symplectic_euler"

    def test_conclude_summary_is_restored(self) -> None:
        """The event log must stay readable in real names."""
        wrapper = BlindedDirector(Spy(Intent(action="run", args={})))
        wrapper.decide(observation())
        label = wrapper.mapping["rk4"]

        spy = Spy(Intent(action="conclude", args={"summary": f"{label} at dt=0.005 works"}))
        inner = BlindedDirector(spy)
        inner._to_blind = dict(wrapper.mapping)
        inner._to_real = {v: k for k, v in wrapper.mapping.items()}
        out = inner.decide(observation())
        assert "rk4" in out.args["summary"]
        assert "method_" not in out.args["summary"]

    def test_rationale_is_restored(self) -> None:
        wrapper = BlindedDirector(Spy(Intent(action="run", args={})))
        wrapper.decide(observation())
        label = wrapper.mapping["explicit_euler"]

        spy = Spy(Intent(action="run", args={"duration": 30.0}, rationale=f"testing {label}"))
        inner = BlindedDirector(spy)
        inner._to_blind = dict(wrapper.mapping)
        inner._to_real = {v: k for k, v in wrapper.mapping.items()}
        assert "explicit_euler" in inner.decide(observation()).rationale

    def test_unknown_label_passes_through_untouched(self) -> None:
        """A hallucinated name must reach the validator, not be silently rewritten."""
        spy = Spy(Intent(action="configure", args={"integrator": "implicitfast", "dt": 0.005}))
        out = BlindedDirector(spy).decide(observation())
        assert out.args["integrator"] == "implicitfast"


class TestMappingStability:
    def test_labels_are_stable_across_rounds(self) -> None:
        spy = Spy(Intent(action="run", args={"duration": 30.0}))
        wrapper = BlindedDirector(spy)
        wrapper.decide(observation(round=0))
        first = dict(wrapper.mapping)
        wrapper.decide(observation(round=1))
        assert wrapper.mapping == first
