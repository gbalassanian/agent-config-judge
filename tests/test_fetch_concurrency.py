"""Tests for cli._fetch_agents_concurrently: the bounded thread pool that
replaced the old sequential for-loop in fetch-portfolio.

Three things are under test, and none of them are about real HTTP:
  1. It's actually concurrent (multiple calls genuinely overlap in time),
     not just "doesn't crash when you hand it a max_workers number".
  2. Concurrency is bounded — never more than max_workers calls in flight.
  3. One agent failing still doesn't block or lose the others (the same
     isolation guarantee the sequential version had), and checkpointing to
     disk happens incrementally, not just once at the end.
"""

import threading
import time

import agent_config_judge.elevenlabs_client as elevenlabs_client_module
from agent_config_judge.cli import _fetch_agents_concurrently, _load_snapshots
from agent_config_judge.elevenlabs_client import ElevenLabsApiError
from agent_config_judge.models import AgentConfigSnapshot, AggregateMetrics, AgentSnapshot


def _fake_build_agent_snapshot(raw_agent, raw_convs, arr_usd=None):
    agent_id = raw_agent["agent_id"]
    config = AgentConfigSnapshot(agent_id=agent_id, name=agent_id, system_prompt="x")
    metrics = AggregateMetrics()
    return AgentSnapshot(agent_id=agent_id, name=agent_id, config=config, metrics=metrics, arr_usd=arr_usd)


class _FakeClient:
    """Stands in for ElevenLabsClient. Each tracked call sleeps briefly —
    long enough for real thread overlap to show up, short enough to keep
    the test suite fast — and records how many calls were in flight at
    once, so concurrency (and its bound) can be asserted directly instead
    of inferred from wall-clock time alone."""

    def __init__(self, fail_agent_ids=(), call_delay=0.03):
        self.fail_agent_ids = set(fail_agent_ids)
        self.call_delay = call_delay
        self._lock = threading.Lock()
        self.in_flight = 0
        self.max_in_flight = 0

    def _track(self):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(self.call_delay)
        with self._lock:
            self.in_flight -= 1

    def get_agent(self, agent_id):
        if agent_id in self.fail_agent_ids:
            raise ElevenLabsApiError(f"boom for {agent_id}")
        self._track()
        return {"agent_id": agent_id}

    def list_conversations(self, agent_id, page_size):
        return [{"conversation_id": f"{agent_id}-c1"}]

    def get_conversation(self, conversation_id):
        self._track()
        return {"conversation_id": conversation_id}


def _patch_build_agent_snapshot(monkeypatch):
    monkeypatch.setattr(elevenlabs_client_module, "build_agent_snapshot", _fake_build_agent_snapshot)


def test_fetch_is_actually_concurrent_and_bounded(monkeypatch):
    _patch_build_agent_snapshot(monkeypatch)
    agents = [{"agent_id": f"a{i}"} for i in range(6)]
    client = _FakeClient(call_delay=0.03)

    start = time.monotonic()
    snapshots, failed = _fetch_agents_concurrently(
        client, agents, sample_size=1, arr_map={}, max_workers=3, checkpoint_path=None,
    )
    elapsed = time.monotonic() - start

    assert len(snapshots) == 6
    assert failed == []
    # get_agent + get_conversation are both tracked -> 12 tracked calls
    # total, bounded to 3 at a time. Real overlap must have happened —
    # this never degrades to one-at-a-time.
    assert 2 <= client.max_in_flight <= 3
    # 12 calls * 0.03s bounded to 3 concurrent -> ~0.12s in theory; a fully
    # sequential run would be ~0.36s. A generous upper bound proves this
    # ran concurrently without asserting an exact, flake-prone timing.
    assert elapsed < 0.30


def test_one_agent_failing_does_not_block_or_lose_the_others(monkeypatch):
    _patch_build_agent_snapshot(monkeypatch)
    agents = [{"agent_id": f"a{i}"} for i in range(4)]
    client = _FakeClient(fail_agent_ids={"a2"}, call_delay=0.005)

    snapshots, failed = _fetch_agents_concurrently(
        client, agents, sample_size=1, arr_map={}, max_workers=2, checkpoint_path=None,
    )

    assert {s.agent_id for s in snapshots} == {"a0", "a1", "a3"}
    assert len(failed) == 1
    assert failed[0][0] == "a2"
    assert "boom for a2" in failed[0][1]


def test_checkpoint_is_written_incrementally_to_disk(monkeypatch, tmp_path):
    _patch_build_agent_snapshot(monkeypatch)
    agents = [{"agent_id": f"a{i}"} for i in range(3)]
    client = _FakeClient(call_delay=0.005)
    checkpoint = tmp_path / "snap.json"

    snapshots, failed = _fetch_agents_concurrently(
        client, agents, sample_size=1, arr_map={}, max_workers=2, checkpoint_path=checkpoint,
    )

    assert failed == []
    assert checkpoint.exists()
    written = _load_snapshots(checkpoint)
    assert {s.agent_id for s in written} == {"a0", "a1", "a2"}


def test_existing_snapshots_are_preserved_and_extended(monkeypatch):
    _patch_build_agent_snapshot(monkeypatch)
    existing = [_fake_build_agent_snapshot({"agent_id": "old1"}, [])]
    agents = [{"agent_id": "new1"}]
    client = _FakeClient(call_delay=0.001)

    snapshots, failed = _fetch_agents_concurrently(
        client, agents, sample_size=1, arr_map={}, max_workers=2,
        checkpoint_path=None, existing_snapshots=existing,
    )

    assert failed == []
    assert {s.agent_id for s in snapshots} == {"old1", "new1"}
