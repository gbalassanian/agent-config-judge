"""Tests for scan_portfolio's per-agent failure isolation.

The load-bearing guarantee here: one agent's judge call raising must never
sink the whole portfolio scan, and a failed agent must never be silently
folded into an existing classification (see FailedTriage's docstring for
why that would corrupt route_unflagged's not_flagged/healthy distinction).
"""

from agent_config_judge.judge import AgentConfigSnapshot
from agent_config_judge.models import AggregateMetrics, AgentSnapshot
from agent_config_judge.pipeline import scan_portfolio


def _flagged_snapshot(agent_id: str) -> AgentSnapshot:
    # A tool error in the sample forces flagged=True regardless of score
    # (should_force_flag) — the simplest reliable way to guarantee this
    # snapshot reaches the judge tier.
    config = AgentConfigSnapshot(agent_id=agent_id, name=agent_id, system_prompt="You are a helpful agent.")
    metrics = AggregateMetrics(n_conversations_sampled=1, n_turns_sampled=1, tool_call_count=1, tool_error_count=1)
    return AgentSnapshot(agent_id=agent_id, name=agent_id, config=config, metrics=metrics)


class _FlakyBackend:
    """Raises for one named agent_id, returns a valid all-pass judgement for
    everyone else — stands in for "the real API call blew up for this one
    agent after retries were exhausted"."""

    def __init__(self, boom_agent_id: str):
        self.boom_agent_id = boom_agent_id

    def judge(self, config: AgentConfigSnapshot, conversations) -> dict:
        if config.agent_id == self.boom_agent_id:
            raise RuntimeError("simulated transient failure, retries exhausted")
        return {
            "criteria": {
                cid: {"verdict": "pass", "evidence_config_field": cid}
                for cid in (
                    "system_prompt", "knowledge_base", "human_handoff", "fallback",
                    "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
                )
            }
        }


def test_one_agent_failing_does_not_sink_the_scan():
    snapshots = [_flagged_snapshot("a1"), _flagged_snapshot("a2"), _flagged_snapshot("a3")]
    backend = _FlakyBackend(boom_agent_id="a2")

    results, failures = scan_portfolio(snapshots, backend)

    assert {r.agent_id for r in results} == {"a1", "a3"}
    assert len(failures) == 1
    assert failures[0].agent_id == "a2"
    assert "simulated transient failure" in failures[0].error


def test_failure_is_never_folded_into_an_existing_classification():
    snapshots = [_flagged_snapshot("a1")]
    backend = _FlakyBackend(boom_agent_id="a1")

    results, failures = scan_portfolio(snapshots, backend)

    assert results == []  # not "healthy", not "not_flagged" — just absent
    assert len(failures) == 1
    assert failures[0].agent_id == "a1"


def test_no_failures_when_nothing_raises():
    snapshots = [_flagged_snapshot("a1"), _flagged_snapshot("a2")]
    backend = _FlakyBackend(boom_agent_id="no-such-agent")

    results, failures = scan_portfolio(snapshots, backend)

    assert len(results) == 2
    assert failures == []
