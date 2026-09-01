"""Tests for CachedJudgeBackend: skip re-invoking the judge when nothing it
would read has changed.

The load-bearing guarantee here: a cache hit is decided purely by whether
the judge's actual inputs (config + sampled conversations) are unchanged —
never by anything the judge doesn't read (ARR, name changes, cheap-pass
score) — and a cache miss falls through to a real call and updates the
cache, so the next identical call becomes a hit.

The healthy_ttl_days tests below cover the second, separate guarantee: a
cached "healthy" verdict expires on its own after enough time passes with
an unchanged fingerprint (closing the dormant-agent gap — see the module
docstring), while a cached real failure never expires this way, and the
default (None) changes nothing about the behavior above.
"""

from datetime import datetime, timedelta, timezone

from agent_config_judge.judge import AgentConfigSnapshot
from agent_config_judge.judge_cache import CachedJudgeBackend
from agent_config_judge.judge_ensemble import EnsembleJudgeBackend
from agent_config_judge.models import ConversationRecord, ConversationTurn

_ALL_PASS_RAW = {
    "criteria": {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
}


class _CountingBackend:
    """Records how many times the real judge would have been called."""

    def __init__(self, raw=None):
        self.calls = 0
        self.raw = raw if raw is not None else _ALL_PASS_RAW

    def judge(self, config, conversations):
        self.calls += 1
        return self.raw


def _config(agent_id="a1", system_prompt="You are a helpful, bounded support agent."):
    return AgentConfigSnapshot(agent_id=agent_id, name=agent_id, system_prompt=system_prompt)


def _conversations(text="Hello, how can I help?"):
    return (ConversationRecord("c1", "widget", (ConversationTurn("agent", text),)),)


def test_first_call_is_a_miss_and_populates_the_cache(tmp_path):
    inner = _CountingBackend()
    cache_path = tmp_path / "cache.json"
    backend = CachedJudgeBackend(backend=inner, cache_path=cache_path)

    result = backend.judge(_config(), _conversations())

    assert inner.calls == 1
    assert result == _ALL_PASS_RAW
    assert cache_path.exists()


def test_identical_second_call_is_a_cache_hit(tmp_path):
    inner = _CountingBackend()
    backend = CachedJudgeBackend(backend=inner, cache_path=tmp_path / "cache.json")

    backend.judge(_config(), _conversations())
    result2 = backend.judge(_config(), _conversations())

    assert inner.calls == 1  # NOT 2 — the second call must not reach the real backend
    assert result2 == _ALL_PASS_RAW


def test_changed_system_prompt_is_a_cache_miss(tmp_path):
    inner = _CountingBackend()
    backend = CachedJudgeBackend(backend=inner, cache_path=tmp_path / "cache.json")

    backend.judge(_config(system_prompt="Prompt version one, bounded and clear."), _conversations())
    backend.judge(_config(system_prompt="Prompt version two, completely different."), _conversations())

    assert inner.calls == 2


def test_changed_conversations_is_a_cache_miss(tmp_path):
    inner = _CountingBackend()
    backend = CachedJudgeBackend(backend=inner, cache_path=tmp_path / "cache.json")

    backend.judge(_config(), _conversations("First conversation sample."))
    backend.judge(_config(), _conversations("A different, newer conversation sample."))

    assert inner.calls == 2


def test_different_agents_are_cached_independently(tmp_path):
    inner = _CountingBackend()
    backend = CachedJudgeBackend(backend=inner, cache_path=tmp_path / "cache.json")

    backend.judge(_config(agent_id="a1"), _conversations())
    backend.judge(_config(agent_id="a2"), _conversations())
    # Re-running a1 unchanged must still hit, even after a2 was added to the cache.
    backend.judge(_config(agent_id="a1"), _conversations())

    assert inner.calls == 2  # one real call each for a1 and a2, no more


def test_cache_persists_across_separate_backend_instances(tmp_path):
    """A fresh CachedJudgeBackend instance pointed at the same file (e.g. a
    new CLI process the next day) must still see yesterday's cache."""
    cache_path = tmp_path / "cache.json"
    inner1 = _CountingBackend()
    CachedJudgeBackend(backend=inner1, cache_path=cache_path).judge(_config(), _conversations())

    inner2 = _CountingBackend()
    result = CachedJudgeBackend(backend=inner2, cache_path=cache_path).judge(_config(), _conversations())

    assert inner2.calls == 0
    assert result == _ALL_PASS_RAW


# --- healthy_ttl_days ------------------------------------------------------

_ONE_FAIL_RAW = {
    "criteria": {
        cid: (
            {"verdict": "fail", "evidence_config_field": "tools", "cause_code": "handoff_no_transfer_tool"}
            if cid == "human_handoff"
            else {"verdict": "pass", "evidence_config_field": cid}
        )
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
}


def _fixed_clock(dt: datetime):
    return lambda: dt


def test_default_none_ttl_never_expires_even_a_very_old_healthy_entry(tmp_path):
    inner = _CountingBackend()
    cache_path = tmp_path / "cache.json"
    long_ago = datetime.now(timezone.utc) - timedelta(days=10_000)
    CachedJudgeBackend(backend=inner, cache_path=cache_path, clock=_fixed_clock(long_ago)).judge(
        _config(), _conversations()
    )

    # healthy_ttl_days left at its default (None) — same instance's later
    # call, or a fresh one, must still be a hit no matter how old the entry is.
    backend2 = CachedJudgeBackend(backend=inner, cache_path=cache_path)  # clock = real now
    backend2.judge(_config(), _conversations())

    assert inner.calls == 1


def test_healthy_entry_younger_than_ttl_is_still_a_hit(tmp_path):
    inner = _CountingBackend()
    cache_path = tmp_path / "cache.json"
    now = datetime.now(timezone.utc)
    ten_days_ago = now - timedelta(days=10)
    CachedJudgeBackend(backend=inner, cache_path=cache_path, clock=_fixed_clock(ten_days_ago)).judge(
        _config(), _conversations()
    )

    backend2 = CachedJudgeBackend(
        backend=inner, cache_path=cache_path, healthy_ttl_days=30, clock=_fixed_clock(now)
    )
    result = backend2.judge(_config(), _conversations())

    assert inner.calls == 1  # still a hit — only 10 of 30 days elapsed
    assert result == _ALL_PASS_RAW


def test_healthy_entry_older_than_ttl_is_a_miss_and_refreshes(tmp_path):
    inner = _CountingBackend()
    cache_path = tmp_path / "cache.json"
    now = datetime.now(timezone.utc)
    thirty_one_days_ago = now - timedelta(days=31)
    CachedJudgeBackend(backend=inner, cache_path=cache_path, clock=_fixed_clock(thirty_one_days_ago)).judge(
        _config(), _conversations()
    )

    backend2 = CachedJudgeBackend(
        backend=inner, cache_path=cache_path, healthy_ttl_days=30, clock=_fixed_clock(now)
    )
    backend2.judge(_config(), _conversations())

    assert inner.calls == 2  # the stale "healthy" forced a real re-judge

    # And the cache is refreshed: a third call at the same "now" is a hit again.
    backend2.judge(_config(), _conversations())
    assert inner.calls == 2


def test_a_real_failure_never_expires_regardless_of_age(tmp_path):
    inner = _CountingBackend(raw=_ONE_FAIL_RAW)
    cache_path = tmp_path / "cache.json"
    now = datetime.now(timezone.utc)
    very_old = now - timedelta(days=10_000)
    CachedJudgeBackend(backend=inner, cache_path=cache_path, clock=_fixed_clock(very_old)).judge(
        _config(), _conversations()
    )

    backend2 = CachedJudgeBackend(
        backend=inner, cache_path=cache_path, healthy_ttl_days=1, clock=_fixed_clock(now)
    )
    result = backend2.judge(_config(), _conversations())

    assert inner.calls == 1  # a real failure is never treated as stale
    assert result == _ONE_FAIL_RAW


def test_legacy_cache_entry_without_judged_at_is_never_forced_stale(tmp_path):
    """A cache file written before healthy_ttl_days existed has no
    judged_at field at all — must not crash, and must not be treated as
    infinitely stale either."""
    import json

    cache_path = tmp_path / "cache.json"
    inner = _CountingBackend()
    fingerprint_backend = CachedJudgeBackend(backend=inner, cache_path=cache_path)
    from agent_config_judge.judge_cache import _fingerprint
    fp = _fingerprint(_config(), _conversations())
    cache_path.write_text(json.dumps({"a1": {"fingerprint": fp, "raw_judgement": _ALL_PASS_RAW}}))

    backend = CachedJudgeBackend(backend=inner, cache_path=cache_path, healthy_ttl_days=1)
    result = backend.judge(_config(), _conversations())

    assert inner.calls == 0  # still a hit — legacy entry isn't punished for predating this field
    assert result == _ALL_PASS_RAW


def test_cache_records_ensemble_attempt_count(tmp_path):
    inner_calls = {"n": 0}

    class _SeqBackend:
        def judge(self, config, conversations):
            inner_calls["n"] += 1
            return _ALL_PASS_RAW  # always clean -> ensemble uses its full budget

    ensemble = EnsembleJudgeBackend(backend=_SeqBackend(), max_extra_runs=2)
    cache_path = tmp_path / "cache.json"
    CachedJudgeBackend(backend=ensemble, cache_path=cache_path).judge(_config(), _conversations())

    import json
    saved = json.loads(cache_path.read_text())
    assert saved["a1"]["ensemble_attempts"] == 3  # 1 + max_extra_runs, all spent


def test_cache_records_one_attempt_when_no_ensemble_is_involved(tmp_path):
    inner = _CountingBackend()
    cache_path = tmp_path / "cache.json"
    CachedJudgeBackend(backend=inner, cache_path=cache_path).judge(_config(), _conversations())

    import json
    saved = json.loads(cache_path.read_text())
    assert saved["a1"]["ensemble_attempts"] == 1
