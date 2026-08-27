"""Regression guard: the cheap pass must not lose recall, and the required
false-positive traps must actually resolve to healthy through the judge.

This is deliberately not a check on the exact numbers printed in the
README — those are allowed to move as the rubric/catalog evolve. What must
never regress silently is recall (a missed broken agent is invisible to
everything downstream) and the two required traps (a detector that never
reverses its own proxies isn't being tested)."""

from agent_config_judge.cheap_pass import score_agent
from eval.golden_set import load_golden_set
from eval.run_eval import run_all_cases, score_cheap_pass, score_judge_classification


def test_cheap_pass_recall_is_perfect_on_the_golden_set():
    """Every case that SHOULD be flagged must actually get flagged — cheap
    pass's one job. (Precision is allowed to be imperfect; see README.)"""
    for case in load_golden_set():
        if case.should_flag:
            result = score_agent(case.snapshot.config, case.snapshot.metrics)
            assert result.flagged, f"{case.snapshot.agent_id} should have been flagged but wasn't"


def test_required_false_positive_traps_resolve_to_healthy():
    results = run_all_cases(load_golden_set())
    trap_ids = {
        "synthetic_grounding_trap_user_number",
        "synthetic_escalation_trap_ticket_tool",
    }
    seen = {r.case.snapshot.agent_id: r for r in results}
    for trap_id in trap_ids:
        assert trap_id in seen, f"required trap {trap_id} missing from golden set"
        assert seen[trap_id].judge_classification == "healthy", (
            f"{trap_id} is a required false-positive trap and must resolve to healthy"
        )


def test_recorded_run_meets_the_reported_bar():
    """A loose floor, not a pin to today's exact numbers — catches an
    accidental regression without forbidding the numbers from moving as
    the rubric is calibrated."""
    results = run_all_cases(load_golden_set())
    cp = score_cheap_pass(results)
    jc = score_judge_classification(results)
    assert cp["recall"] == 1.0
    assert not jc["exceeds_threshold"]
