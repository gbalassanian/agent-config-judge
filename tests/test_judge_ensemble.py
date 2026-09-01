"""Tests for EnsembleJudgeBackend: re-confirm a clean read, never a dirty one.

The load-bearing guarantees:
- A first call that already has a real (validated) failure short-circuits —
  no extra calls, no extra spend, on the majority case that already found
  something to route.
- A first call that's clean triggers more calls, up to the cap, stopping
  the moment any one finds something real.
- "Found something" means "survived the exact same evidence validation
  every live call goes through" — a fail with a fabricated/unverifiable
  citation must NOT short-circuit the extra-confirmation runs, or a
  would-be-discarded claim could fool the ensemble into skipping the
  safety net it exists to provide.
- Merging is per-criterion: whichever attempt first validates a real fail
  for a given criterion contributes its evidence for that criterion; every
  other criterion falls back to the first attempt.
"""

from agent_config_judge.judge import AgentConfigSnapshot, validate_judge_output
from agent_config_judge.judge_ensemble import EnsembleJudgeBackend
from agent_config_judge.models import ConversationRecord, ConversationTurn

_CRITERIA_IDS = (
    "system_prompt", "knowledge_base", "human_handoff", "fallback",
    "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
)


def _all_pass() -> dict:
    return {
        "criteria": {
            cid: {"verdict": "pass", "evidence_config_field": cid} for cid in _CRITERIA_IDS
        }
    }


def _with_fail(cid: str, **fail_fields) -> dict:
    raw = _all_pass()
    raw["criteria"][cid] = {"verdict": "fail", **fail_fields}
    return raw


class _SequenceBackend:
    """Returns one queued raw dict per call, in order; counts total calls."""

    def __init__(self, raws: list[dict]):
        self._raws = list(raws)
        self.calls = 0

    def judge(self, config, conversations):
        self.calls += 1
        return self._raws[self.calls - 1]


def _config(agent_id="a1") -> AgentConfigSnapshot:
    return AgentConfigSnapshot(agent_id=agent_id, name=agent_id, system_prompt="You are a helpful, bounded agent.")


def _conversations(text="We accept returns within 14 days.") -> tuple[ConversationRecord, ...]:
    return (ConversationRecord("c1", "widget", (ConversationTurn("agent", text),)),)


def test_first_call_with_a_real_failure_short_circuits():
    inner = _SequenceBackend([
        _with_fail("human_handoff", evidence_config_field="tools", cause_code="handoff_no_transfer_tool"),
        _all_pass(),  # would prove a bug if this ever got called
    ])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=2)

    raw = backend.judge(_config(), _conversations())

    assert inner.calls == 1
    validated = validate_judge_output(raw, agent_id="a1", conversations=_conversations())
    assert any(c.criterion_id == "human_handoff" and c.verdict == "fail" for c in validated.failures)


def test_clean_first_call_triggers_a_second_that_finds_something():
    inner = _SequenceBackend([
        _all_pass(),
        _with_fail("knowledge_base", evidence_config_field="knowledge_base_ids", cause_code="kb_not_connected"),
        _all_pass(),  # would prove a bug if this ever got called (should stop at attempt 2)
    ])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=2)

    raw = backend.judge(_config(), _conversations())

    assert inner.calls == 2
    validated = validate_judge_output(raw, agent_id="a1", conversations=_conversations())
    assert [c.criterion_id for c in validated.failures] == ["knowledge_base"]


def test_all_attempts_clean_returns_healthy_after_using_the_full_budget():
    inner = _SequenceBackend([_all_pass(), _all_pass(), _all_pass()])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=2)

    raw = backend.judge(_config(), _conversations())

    assert inner.calls == 3  # 1 + max_extra_runs, all spent since nothing was ever found
    validated = validate_judge_output(raw, agent_id="a1", conversations=_conversations())
    assert validated.classification == "healthy"
    assert validated.failures == []


def test_max_extra_runs_zero_behaves_like_no_ensemble():
    inner = _SequenceBackend([_all_pass()])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=0)

    backend.judge(_config(), _conversations())

    assert inner.calls == 1


def test_a_fabricated_citation_does_not_short_circuit_confirmation():
    """The first attempt claims a fail with a quote that isn't actually in
    the transcript — validate_judge_output would downgrade that to
    unknown, so it must NOT count as "found something real" and must not
    stop the ensemble from checking again."""
    inner = _SequenceBackend([
        _with_fail("grounding", evidence_quote="This quote does not appear anywhere in the transcript."),
        _with_fail("grounding", evidence_quote="We accept returns within 14 days.", cause_code="grounding_missing_source_attribution"),
    ])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=2)

    raw = backend.judge(_config(), _conversations("We accept returns within 14 days."))

    assert inner.calls == 2
    validated = validate_judge_output(raw, agent_id="a1", conversations=_conversations("We accept returns within 14 days."))
    assert [c.criterion_id for c in validated.failures] == ["grounding"]


def test_merge_keeps_the_winning_attempts_evidence_for_that_criterion_only():
    """Attempt 2's finding on human_handoff must carry ITS OWN evidence into
    the merged output — never a mix of attempt 1's fields with attempt 2's
    verdict — while every other criterion still comes from attempt 1."""
    inner = _SequenceBackend([
        _all_pass(),
        _with_fail("human_handoff", evidence_config_field="tools", cause_code="handoff_no_transfer_tool"),
    ])
    backend = EnsembleJudgeBackend(backend=inner, max_extra_runs=1)

    raw = backend.judge(_config(), _conversations())

    assert raw["criteria"]["human_handoff"]["evidence_config_field"] == "tools"
    assert raw["criteria"]["human_handoff"]["cause_code"] == "handoff_no_transfer_tool"
    # every other criterion still traces back to attempt 1's all-pass entry
    assert raw["criteria"]["system_prompt"]["verdict"] == "pass"
