"""Tests for cheap_pass.py: the forced-flag rule and the asymmetric-cost
scoring (unknown counts against, not neutral)."""

from agent_config_judge.cheap_pass import score_agent, should_force_flag
from agent_config_judge.models import AgentConfigSnapshot, AggregateMetrics, ToolConfig


def _config(**overrides) -> AgentConfigSnapshot:
    base = dict(
        agent_id="a1", name="Test Agent",
        system_prompt="You are a support agent for Acme. You only handle billing questions.",
        knowledge_base_ids=("kb1",),
        tools=(),
    )
    base.update(overrides)
    return AgentConfigSnapshot(**base)


def test_forced_flag_fires_on_any_tool_error_regardless_of_score():
    config = _config(tools=(ToolConfig(name="transfer_to_number", tool_type="system", system_tool_type="transfer_to_number"),))
    metrics = AggregateMetrics(n_conversations_sampled=5, channels_seen=("react_sdk",), tool_error_count=1)
    result = score_agent(config, metrics)
    assert result.forced_flag is True
    assert result.flagged is True
    assert should_force_flag(metrics) is True


def test_no_forced_flag_without_a_tool_error():
    metrics = AggregateMetrics(n_conversations_sampled=5, tool_error_count=0)
    assert should_force_flag(metrics) is False


def test_empty_system_prompt_fails_outright():
    config = _config(system_prompt="")
    metrics = AggregateMetrics()
    result = score_agent(config, metrics)
    assert result.criterion("system_prompt").verdict.value == "fail"


def test_missing_knowledge_base_is_unknown_not_pass_or_fail():
    config = _config(knowledge_base_ids=())
    metrics = AggregateMetrics()
    result = score_agent(config, metrics)
    # "no signal" must not read the same as "confirmed healthy"
    assert result.criterion("knowledge_base").verdict.value == "unknown"
    assert result.criterion("knowledge_base").score < 100.0


def test_transfer_to_number_on_phone_only_channels_passes_human_handoff():
    config = _config(tools=(ToolConfig(name="transfer_to_number", tool_type="system", system_tool_type="transfer_to_number"),))
    metrics = AggregateMetrics(channels_seen=("twilio",))
    result = score_agent(config, metrics)
    assert result.criterion("human_handoff").verdict.value == "pass"


def test_transfer_to_number_on_non_telephony_channel_fails_human_handoff():
    """The exact shape of the case-study finding: config alone would pass this."""
    config = _config(tools=(ToolConfig(name="transfer_to_number", tool_type="system", system_tool_type="transfer_to_number"),))
    metrics = AggregateMetrics(channels_seen=("react_sdk",))
    result = score_agent(config, metrics)
    assert result.criterion("human_handoff").verdict.value == "fail"


def test_fallback_always_reports_unknown_never_fabricates_a_proxy():
    config = _config()
    metrics = AggregateMetrics(n_conversations_sampled=10)
    result = score_agent(config, metrics)
    assert result.criterion("fallback").verdict.value == "unknown"
