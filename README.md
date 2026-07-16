# world-engine

An **LLM Director looping over deterministic ground truth** — the shared substrate for
three tracks that compose later: physics/chem theory-testing, living worlds
(agents/events/characters), and generative media (storyboard → video) output.

**Core invariant:** pluggable truth (backends) behind a uniform world-state spine; the
LLM only touches that truth through a typed contract (`Observation` in, `Intent` out);
everything deterministic is replayable from the append-only event log.

```
        ┌──────────────────────────────┐
        │      Director  (LLM)         │  reasons · plans · proposes
        └──────┬───────────────▲───────┘
        intents│               │observation (scoped view)
        ┌──────▼───────────────┴───────┐
        │    Validator + Verifier      │  contract · invariants · feedback
        └──────┬───────────────────────┘
        apply → events
┌───────┐┌─────▼────────────────────────┐
│Projec-◄┤   World State · Event Log    │  ECS components + append-only SQLite log
│tor    ││  (fold(events) == state)     │
└───────┘└─────▲────────────────────────┘
               │ advance truth (deterministic, substeps)
   ┌───────────┴─┬──────────────┬─────────────┐
   │  Physics    │  Chemistry   │   Social    │   ← pluggable backends
   │ (pendulum + │  (planned)   │ (LLM agent  │
   │   MuJoCo)   │              │   minds)    │
   └─────────────┴──────────────┴─────────────┘
```

## First vertical slice (implemented)

A deterministic nonlinear pendulum is the truth; the Director hunts for integrator
settings that conserve energy. Explicit Euler pumps energy in (fails), a 10× smaller
step still fails (the drift is structural, not resolution), symplectic Euler passes —
a real discovery arc judged entirely by the verifier, never by the LLM.

## Quickstart

```bash
cd world-engine
pip install -r requirements.txt      # pydantic required; matplotlib optional (plots)

python demos/pendulum_energy.py                  # scripted director, fully offline
python demos/pendulum_energy.py --director llm   # decisions via llm-router (:4000)
python demos/pendulum_energy.py --backend mujoco # MuJoCo as ground truth (same engine)

python demos/rendezvous.py                       # social backend: agents find each other
python demos/rendezvous.py --minds llm           # each agent is an LLM via llm-router

python -m pytest tests -q                        # MuJoCo tests skip if mujoco missing
```

Each run writes `runs/<timestamp>/events.db` (the full replayable session) and
`energy.png` (if matplotlib is installed). Exit code 0 = concluded with a passing
configuration + the replay check held.

### LLM director wiring (llm-router)

| Env var | Default | Meaning |
|---|---|---|
| `WORLD_ENGINE_LLM_BASE` | `http://127.0.0.1:4000/v1` | OpenAI-compatible base URL |
| `WORLD_ENGINE_LLM_MODEL` | `auto` | model name passed through (router routes semantically) |
| `WORLD_ENGINE_LLM_ROUTE` | — | optional llm-router force-pin (e.g. `local_reason`) |
| `WORLD_ENGINE_LLM_KEY` | — | optional bearer token (local routing works keyless) |

CLI equivalents: `--llm-base`, `--llm-model`, `--llm-route`.

## Layout

```
worldengine/
  models.py       Event · Intent · Observation · Verdict · Outcome (the typed contract)
  eventlog.py     append-only SQLite log — replay + branch
  world.py        ECS-lite state; state = fold(events)
  contract.py     intent validation (the LLM's only write path)
  verifier.py     deterministic invariants (EnergyConservation)
  director.py     ScriptedDirector (offline) · LLMDirector (llm-router)
  engine.py       the loop: observe → decide → validate → apply → advance → verify
  projector.py    TextProjector · PlotProjector · StoryboardProjector (stub)
  backends/
    pendulum.py        deterministic ground truth (explicit/symplectic Euler, RK4)
    mujoco_backend.py  MuJoCo behind the same protocol (any MJCF; euler/implicit*/rk4)
    social.py          persona agents: scoped AgentView → Mind → typed AgentAction
                       (SeekerMind/WaiterMind scripted · LLMMind via llm-router)
assets/pendulum.xml   MJCF model for the MuJoCo demo (undamped point-mass pendulum)
demos/pendulum_energy.py    physics hypothesis loop (--backend pendulum|mujoco)
demos/rendezvous.py         social episode loop (--minds scripted|llm)
tests/                      event log · backend physics · MuJoCo · social · engine loop
```

## Design decisions

- **Event sourcing.** State is a fold over an append-only log. Buys replay
  (deterministic re-runs), branching (`EventLog.branch` — fork at seq N, change one
  thing), causality (the log *is* the trace), and a narrative beat-sheet for free.
- **Event granularity.** Backends integrate at fine dt internally and emit coarse
  *sampled* events (~0.1 s), so the log stays queryable instead of drowning in
  substeps. Determinism inside a tick is guaranteed by the backend, not the log.
- **The contract.** The Director never writes components. It emits typed `Intent`s;
  `contract.validate_intent` rejects violations and the reason goes back as an
  observation note. Hallucinations become feedback, never state.
- **Determinism + replay.** Backends are pure and seedless-deterministic; the LLM is
  the only nondeterministic part and its raw decisions are logged (`director.decided`),
  so a full session replays exactly.
- **Cadence decoupling.** One Director decision per round; thousands of deterministic
  substeps inside a single `run` intent. Never call the LLM per physics step.
- **Scoped observation.** The Director sees a small typed digest (`Observation`),
  never the raw world — this is the context-budget wall, treated as a first-class
  perception function. Hard-won corollary: the view must carry every piece of
  state the model is expected to act on (the Director needs `configured`; an
  agent needs its own travel history, or "avoid repeats" is an instruction with
  no data behind it — both found in live runs).
- **Capabilities are backend-owned.** Each backend advertises its `integrators`;
  the contract validates against them and the observation carries them, so the
  Director learns what's possible from the world, not from a hardcoded prompt.
  (Live-verified: an LLM proposing a pendulum-backend integrator to MuJoCo gets a
  contract rejection listing the real options, and recovers.)

## Swap points (roadmap)

1. ~~**Physics for real:** `MuJoCoBackend`~~ — **done.** Any MJCF model behind the same
   reset/advance protocol; the engine, contract, verifier and directors were untouched
   (see `test_full_engine_loop_on_mujoco`).
2. **Chemistry:** RDKit/OpenMM backend; verifier = mass balance / energy.
3. ~~**Track 2 (living world):** `backends/social.py`~~ — **done.** Persona agents with
   room-scoped perception and rolling memory; minds propose typed actions through the
   same propose → validate → apply shape (invalid action → `agent.confused` + memory
   entry, never state). `Rendezvous` verifier judges co-location from `social.tick`
   ground truth, never from what agents *say*. `TranscriptProjector` renders episodes
   as scripts (see `runs/<ts>/transcript.txt`).
4. **Track 3 (media):** implement `projector.StoryboardProjector` — event slices →
   beats → shot prompts → SDXL → ClipLab timeline. `TranscriptProjector` is its
   precursor: replace "format a line" with "emit a shot prompt". Projectors are pure;
   the engine never changes.
