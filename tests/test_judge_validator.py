"""Tests for the structural rules judge.py is built around. These are the
load-bearing guarantees of the whole repo — if any of these break, a
systemic agent could silently start routing as an automated nudge."""

from agent_config_judge.judge import validate_judge_output
from agent_config_judge.models import ConversationRecord, ConversationTurn


def _all_pass(overrides: dict) -> dict:
    """A minimal all-pass criteria dict, with `overrides` replacing some keys."""
    base = {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
    base.update(overrides)
    return {"criteria": base}


def test_unevidenced_verdict_is_downgraded_to_unknown():
    raw = _all_pass({"grounding": {"verdict": "fail", "cause_code": "grounding_missing_source_attribution"}})
    j = validate_judge_output(raw, agent_id="a1")
    assert j.criteria["grounding"].verdict == "unknown"
    assert j.classification == "healthy"  # no surviving failures


def test_fabricated_quote_is_discarded():
    convs = (ConversationRecord("c1", "widget", (ConversationTurn("agent", "We accept returns within 14 days."),)),)
    raw = _all_pass({"grounding": {
        "verdict": "fail", "evidence_quote": "We accept returns within 90 days, no questions asked",
        "cause_code": "grounding_missing_source_attribution",
    }})
    j = validate_judge_output(raw, agent_id="a1", conversations=convs)
    assert j.criteria["grounding"].verdict == "unknown"
    assert any("fabricated" in n for n in j.validator_notes)


def test_genuine_quote_survives_verification():
    convs = (ConversationRecord("c1", "widget", (ConversationTurn("agent", "We accept returns within 14 days."),)),)
    raw = _all_pass({"grounding": {
        "verdict": "fail", "evidence_quote": "We accept returns within 14 days.",
        "cause_code": "grounding_missing_source_attribution",
    }})
    j = validate_judge_output(raw, agent_id="a1", conversations=convs)
    assert j.criteria["grounding"].verdict == "fail"
    assert j.classification == "standard"


def test_unknown_cause_code_forces_systemic_even_with_other_mapped_failures():
    raw = _all_pass({
        "knowledge_base": {"verdict": "fail", "evidence_config_field": "knowledge_base_ids", "cause_code": "kb_not_connected"},
        "grounding": {"verdict": "fail", "evidence_config_field": "x", "cause_code": "totally_made_up_cause"},
    })
    j = validate_judge_output(raw, agent_id="a1")
    assert j.classification == "systemic"
    assert j.criteria["knowledge_base"].recipe is not None  # this one WAS mapped
    assert j.criteria["grounding"].recipe is None  # this one wasn't — that's what forces systemic


def test_three_mapped_failures_are_still_standard_not_systemic():
    raw = _all_pass({
        "system_prompt": {"verdict": "fail", "evidence_config_field": "system_prompt", "cause_code": "system_prompt_missing"},
        "knowledge_base": {"verdict": "fail", "evidence_config_field": "knowledge_base_ids", "cause_code": "kb_not_connected"},
        "human_handoff": {"verdict": "fail", "evidence_config_field": "tools", "cause_code": "handoff_no_transfer_tool"},
    })
    j = validate_judge_output(raw, agent_id="a1")
    assert j.classification == "standard"
    assert len(j.failures) == 3


def test_classification_field_in_raw_output_is_ignored():
    raw = _all_pass({})
    raw["classification"] = "systemic"  # the model claiming something the validator must not trust
    j = validate_judge_output(raw, agent_id="a1")
    assert j.classification == "healthy"  # recomputed from (lack of) failures, not read from the model


def test_cause_code_on_wrong_criterion_is_rejected():
    raw = _all_pass({
        # kb_not_connected belongs to knowledge_base, not human_handoff
        "human_handoff": {"verdict": "fail", "evidence_config_field": "tools", "cause_code": "kb_not_connected"},
    })
    j = validate_judge_output(raw, agent_id="a1")
    assert j.criteria["human_handoff"].recipe is None
    assert j.classification == "systemic"


def test_missing_criterion_key_scores_unknown_not_crash():
    raw = {"criteria": {"system_prompt": {"verdict": "pass", "evidence_config_field": "system_prompt"}}}
    j = validate_judge_output(raw, agent_id="a1")
    assert j.criteria["latency"].verdict == "unknown"
    assert j.classification == "healthy"
