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

### Recall is not discovery — use `--blind`

That arc is the **scripted** Director's. An LLM Director short-circuits it: the
options are named `explicit_euler`, `symplectic_euler` and `rk4`, and any competent
model knows symplectic integrators conserve energy in Hamiltonian systems. Measured
on Qwen3.5-4B, **8/8 samples configured `symplectic_euler` on round 0** with zero
results in hand. The observe/decide/verify loop never ran. A benchmark whose labels
give away the answer cannot tell you the loop works — or that it broke.

`--blind` relabels the integrators to opaque tokens (`method_a/b/c`) for the
Director only; the world, verifier and event log keep real names, so runs stay
replayable. With priors removed the same model behaves the way the engine claims:

| situation | pick | n |
| --- | --- | --- |
| round 0, nothing tried | a baseline | 8/8 |
| one option failed twice | moves off it | 7/8 |
| **two options failed** | **the remaining untried one** | **8/8** |
| a different option failed | an untried one | 8/8 |

The third row is the one that matters: positional habit ("try the next in the
list") would have repeated an already-failed option. It does not. End to end,
the blinded LLM Director configures a baseline, reads a 1629% drift failure from
the verifier, switches, and passes — and it converges on `rk4`, a *second* valid
answer the named benchmark never explores because the model always recites the
one it already knows.

Use `--blind` whenever you want to test the loop rather than the model's memory.

**What blinding found on MuJoCo** (4 integrators, weaker priors — a harder case):

- *Named*, the model's priors are **actively wrong**. It opened with `implicit`
  ("generally known for superior energy conservation compared to explicit methods
  like Euler or RK4"), then `implicitfast` ("a higher-order symplectic method" —
  simply false), burning two rounds before evidence took over and `rk4` passed.
- *Blinded*, it **confabulates physics about meaningless tokens**: "method_c is a
  symplectic integrator often used for Hamiltonian systems." Free text from a
  blinded Director is therefore logged **verbatim** — an earlier version rewrote
  labels back to real names, which disguised invented reasoning as domain
  knowledge. Runs print the label mapping so the log stays decodable.
- It also exposed a real hole in the contract: the Director could **reconfigure
  forever without ever running**, learning nothing. One run burned seven straight
  rounds alternating two `configure` intents. `validate_intent` now rejects a
  reconfigure while an unrun experiment is pending, and that rejection becomes
  feedback like any other — the same run then converged and concluded.

## Quickstart

```bash
cd world-engine
pip install -r requirements.txt      # pydantic required; matplotlib optional (plots)

python demos/pendulum_energy.py                  # scripted director, fully offline
python demos/pendulum_energy.py --director llm   # decisions via llm-router (:4000)
python demos/pendulum_energy.py --director llm --blind   # hide integrator NAMES: tests the loop, not recall
python demos/pendulum_energy.py --backend mujoco # MuJoCo as ground truth (same engine)

python demos/rendezvous.py                       # social backend: agents find each other
python demos/rendezvous.py --minds llm           # each agent is an LLM via llm-router
python demos/rendezvous.py --storyboard llm      # LLM-enriched shot prompts (Track 3)

# render the board into key frames (needs sd-lab's A1111 on :7861)
python scripts/render_storyboard.py runs/<ts>/storyboard.json \
    --lora "Mira=mira_oc:1:miraoc" --lora "Rook=rook_oc:1:rookoc" \
    --regional   # multi-character shots: one region per character

# frames + dialogue -> ClipLab Pro Editor timeline (manifest v2 + per-shot clips)
python scripts/export_cliplab.py runs/<ts>/storyboard.json --frames runs/<ts>/frames-lora

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
| `WORLD_ENGINE_LLM_CALLER` | — | identify this client to the gateway (`x-caller`) |
| `WORLD_ENGINE_LLM_MEMORY` | `0` | `1` lets the gateway inject recalled memory. Off by default: the Director's view is the `Observation`, and injected memory is an undeclared side channel around the contract — and a drifting one, since the store is written continuously |

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
  projector.py    TextProjector · PlotProjector · TranscriptProjector
  storyboard.py   Track 3: beats → Shot prompts (StoryboardProjector · LLMShotEnricher)
  backends/
    pendulum.py        deterministic ground truth (explicit/symplectic Euler, RK4)
    mujoco_backend.py  MuJoCo behind the same protocol (any MJCF; euler/implicit*/rk4)
    social.py          persona agents: scoped AgentView → Mind → typed AgentAction
                       (SeekerMind/WaiterMind scripted · LLMMind via llm-router)
assets/pendulum.xml   MJCF model for the MuJoCo demo (undamped point-mass pendulum)
demos/pendulum_energy.py    physics hypothesis loop (--backend pendulum|mujoco)
demos/rendezvous.py         social episode loop (--minds scripted|llm)
scripts/render_storyboard.py  storyboard.json → frames/*.png + contact sheet (A1111 API)
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
4. ~~**Track 3 (media):** `StoryboardProjector`~~ — **done** (`worldengine/storyboard.py`).
   Deterministic beat segmentation (establishing / search montage / meeting / dialogue /
   card) over the event stream; one image prompt per beat assembled from configured
   character/setting sheets — but WHO/WHAT/WHERE always comes from ground-truth events,
   and a sheet only appears when that character is actually in the shot. Optional
   `LLMShotEnricher` rewrites prompts cinematically (facts pinned; template kept on any
   failure). Outputs `storyboard.json` (ready for an image pipeline: SDXL → ClipLab) and
   `storyboard.md`. `scripts/render_storyboard.py` closes the loop: it feeds the board
   to sd-lab's A1111 API (:7861) and drops deterministic frames (seed = base + shot) plus
   a `frames/index.html` contact sheet next to the board. **Character consistency recipe**
   (both verified live): solo shots use per-character LoRAs (`--lora Name=file:w:trigger`,
   injected only when that character is in the shot's ground-truth cast); multi-character
   shots use `--regional` — one Regional-Prompter *Attention* column per character built
   from the board's character/place sheets. Per-region LoRAs (Latent mode) were tried and
   rejected: identity LoRAs bleed across regions and blend into one figure.
   `scripts/export_cliplab.py` completes the chain: per-shot H.264 clips (beat-scaled
   durations) + a ClipForge **manifest v2** with dialogue as clip-anchored caption cues —
   verified against ClipLab's own validator; import it via the Editor's manifest flow.

## License

MIT. See [LICENSE](LICENSE). Copyright (c) 2026 Emmitt Fennessey.
