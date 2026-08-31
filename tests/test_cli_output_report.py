"""Tests that the CLI's --output JSON report actually carries validator_notes
— the load-bearing gap this closes: without it, a fabricated/discarded
citation is invisible outside custom code, which makes the "measure the
discard rate against real usage" plan (see README) impossible to run.
"""

import argparse
import json

from agent_config_judge.cli import cmd_evaluate, cmd_scan
from agent_config_judge.elevenlabs_client import compute_aggregate_metrics
from agent_config_judge.judge import RecordedJudgeBackend
from agent_config_judge.models import (
    AgentConfigSnapshot,
    AgentSnapshot,
    ConversationRecord,
    ConversationTurn,
    agent_snapshot_to_dict,
)


def _snapshot_path(tmp_path, agent_id: str) -> str:
    config = AgentConfigSnapshot(agent_id=agent_id, name="Test Agent", system_prompt="You help customers.")
    conversations = (ConversationRecord("c1", "widget", (
        ConversationTurn("agent", "We accept returns within 14 days."),
    )),)
    snapshot = AgentSnapshot(
        agent_id=agent_id, name="Test Agent", config=config, conversations=conversations,
        metrics=compute_aggregate_metrics(list(conversations)),
    )
    path = tmp_path / "snapshot.json"
    path.write_text(json.dumps([agent_snapshot_to_dict(snapshot)]))
    return str(path)


def _fixture_with_fabricated_quote(tmp_path, agent_id: str) -> str:
    """A recorded judgement whose grounding citation doesn't match the real
    transcript above — the exact case validator_notes exists to surface."""
    all_pass = {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
    all_pass["grounding"] = {
        "verdict": "fail",
        "evidence_quote": "We accept returns within 90 days",  # real transcript says 14
        "cause_code": "grounding_missing_source_attribution",
    }
    fixture_path = tmp_path / "recorded.json"
    fixture_path.write_text(json.dumps({agent_id: {"criteria": all_pass}}))
    return str(fixture_path)


def test_evaluate_output_report_includes_validator_notes_for_a_discarded_quote(tmp_path):
    agent_id = "agent_test1"
    ns = argparse.Namespace(
        agent_id=agent_id,
        snapshot=_snapshot_path(tmp_path, agent_id),
        sample_size=20, arr_usd=None,
        backend="recorded", fixture=_fixture_with_fabricated_quote(tmp_path, agent_id),
        model="claude-sonnet-5", judge_cache=None, force_judge=True,
        output=str(tmp_path / "report.json"),
    )
    cmd_evaluate(ns)
    report = json.loads((tmp_path / "report.json").read_text())
    assert any("discarded as fabricated" in n for n in report["validator_notes"])


def test_scan_output_report_includes_empty_validator_notes_when_nothing_was_discarded(tmp_path):
    """Whether or not the cheap pass sends this agent to the judge, an
    all-pass (or never-judged) result must report an empty list, not crash
    on a None judgement — the fixture below answers all-pass either way, so
    this doesn't depend on cheap_pass's own scoring internals."""
    agent_id = "agent_test2"
    config = AgentConfigSnapshot(agent_id=agent_id, name="Healthy Agent", system_prompt="You help customers with clear scope.")
    snapshot = AgentSnapshot(
        agent_id=agent_id, name="Healthy Agent", config=config, conversations=(),
        metrics=compute_aggregate_metrics([]),
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(json.dumps([agent_snapshot_to_dict(snapshot)]))

    all_pass = {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
    fixture_path = tmp_path / "recorded.json"
    fixture_path.write_text(json.dumps({agent_id: {"criteria": all_pass}}))

    ns = argparse.Namespace(
        snapshot=str(snapshot_path),
        backend="recorded", fixture=str(fixture_path),
        model="claude-sonnet-5", judge_cache=None,
        output=str(tmp_path / "report.json"),
    )
    cmd_scan(ns)
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["agents"][0]["validator_notes"] == []
