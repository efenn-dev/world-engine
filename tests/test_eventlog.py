"""Event log: append/read round-trip, fold equivalence, branching."""

from pathlib import Path

from worldengine.eventlog import EventLog
from worldengine.models import Event
from worldengine.world import World, fold


def _events() -> list[Event]:
    return [
        Event(t=0.0, kind="world.seeded", entity=1, data={"set_components": {"pos": {"x": 0.0}}}),
        Event(t=1.0, kind="physics.sampled", entity=1, data={"set_components": {"pos": {"x": 2.5}}, "energy": 4.5}),
        Event(t=1.0, kind="verdict.issued", entity=1, data={"passed": True}),
    ]


def test_append_read_roundtrip(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.db")
    stored = [log.append(e) for e in _events()]
    assert [e.seq for e in stored] == [1, 2, 3]
    back = log.read()
    assert [e.kind for e in back] == ["world.seeded", "physics.sampled", "verdict.issued"]
    assert back[1].data["energy"] == 4.5
    assert log.read(since=2)[0].kind == "verdict.issued"
    assert [e.kind for e in log.read(kinds=("verdict.issued",))] == ["verdict.issued"]
    log.close()


def test_fold_matches_incremental_apply(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.db")
    live = World()
    for event in _events():
        live.apply(log.append(event))
    replayed = fold(log.read())
    assert replayed.get(1, "pos") == live.get(1, "pos") == {"x": 2.5}
    log.close()


def test_branch_copies_prefix(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "events.db")
    log.append_many(_events())
    branch = log.branch(tmp_path / "branch.db", upto_seq=2)
    assert branch.last_seq() == 2
    assert [e.kind for e in branch.read()] == ["world.seeded", "physics.sampled"]
    # diverge the branch; source is untouched
    branch.append(Event(kind="alt.timeline", data={}))
    assert branch.last_seq() == 3
    assert log.last_seq() == 3 and log.read()[-1].kind == "verdict.issued"
    branch.close()
    log.close()
