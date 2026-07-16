"""world-engine: an LLM Director looping over a deterministic world-state spine.

Core invariant: pluggable truth (backends) behind a uniform world state; the
LLM only touches that truth through a typed contract (Observation in, Intent
out); everything deterministic is replayable from the append-only event log.
"""

__version__ = "0.1.0"
