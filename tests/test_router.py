"""Tests for router.py: classification x ARR -> exactly one action."""

from agent_config_judge.judge import validate_judge_output
from agent_config_judge.router import ArrTier, RouteAction, route, route_unflagged

HIGH_ARR = 100_000.0
LOW_ARR = 1_000.0


def _judgement(criteria_overrides: dict):
    base = {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
    base.update(criteria_overrides)
    return validate_judge_output({"criteria": base}, agent_id="a1")


def test_healthy_routes_to_no_action():
    j = _judgement({})
    d = route(j, arr_usd=HIGH_ARR)
    assert d.action == RouteAction.NO_ACTION
    assert d.requires_human_approval is False


def test_standard_all_self_serve_routes_to_self_serve_fix():
    j = _judgement({"knowledge_base": {"verdict": "fail", "evidence_config_field": "knowledge_base_ids", "cause_code": "kb_not_connected"}})
    d = route(j, arr_usd=LOW_ARR)
    assert d.action == RouteAction.SELF_SERVE_FIX


def test_standard_with_a_nudge_recipe_routes_to_targeted_nudge():
    j = _judgement({"human_handoff": {
        "verdict": "fail", "evidence_quote": "x", "cause_code": "handoff_tool_unsupported_on_channel",
    }})
    # evidence_quote needs no transcript check here since conversations=None was used
    d = route(j, arr_usd=LOW_ARR)
    assert d.action == RouteAction.TARGETED_NUDGE


def test_systemic_high_arr_escalates_to_engineer():
    j = _judgement({"grounding": {"verdict": "fail", "evidence_config_field": "x", "cause_code": "unmapped_thing"}})
    d = route(j, arr_usd=HIGH_ARR)
    assert d.action == RouteAction.ESCALATE_TO_ENGINEER
    assert d.arr_tier == ArrTier.HIGH
    assert len(d.recipe_gaps) == 1


def test_systemic_low_arr_gets_nearest_guidance_and_logs_a_gap():
    j = _judgement({"grounding": {"verdict": "fail", "evidence_config_field": "x", "cause_code": "unmapped_thing"}})
    d = route(j, arr_usd=LOW_ARR)
    assert d.action == RouteAction.NEAREST_GUIDANCE
    assert d.arr_tier == ArrTier.LOW
    assert len(d.recipe_gaps) == 1
    assert d.recipe_gaps[0].judge_cause_code == "unmapped_thing"


def test_unknown_arr_is_treated_as_low():
    j = _judgement({"grounding": {"verdict": "fail", "evidence_config_field": "x", "cause_code": "unmapped_thing"}})
    d = route(j, arr_usd=None)
    assert d.arr_tier == ArrTier.UNKNOWN
    assert d.action == RouteAction.NEAREST_GUIDANCE  # same branch as low, not high


def test_nothing_touches_a_live_agent_today():
    for action in RouteAction:
        j = _judgement({})
        d = route(j, arr_usd=HIGH_ARR) if action == RouteAction.NO_ACTION else None
    # Direct check against the router's own table rather than constructing
    # every branch: today, every action must resolve to False.
    from agent_config_judge.router import _TOUCHES_LIVE_AGENT
    assert all(v is False for v in _TOUCHES_LIVE_AGENT.values())


def test_route_unflagged_is_distinct_from_judge_confirmed_healthy():
    d = route_unflagged("a1", arr_usd=HIGH_ARR)
    assert d.classification == "not_flagged"
    assert d.classification != "healthy"
    assert d.action == RouteAction.NO_ACTION
