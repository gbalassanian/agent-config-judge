"""Tests for CachedJudgeBackend: skip re-invoking the judge when nothing it
would read has changed.

The load-bearing guarantee here: a cache hit is decided purely by whether
the judge's actual inputs (config + sampled conversations) are unchanged —
never by anything the judge doesn't read (ARR, name changes, cheap-pass
score) — and a cache miss falls through to a real call and updates the
cache, so the next identical call becomes a hit.
"""

from agent_config_judge.judge import AgentConfigSnapshot
from agent_config_judge.judge_cache import CachedJudgeBackend
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
