"""Tests for _normalize_for_match's expanded normalization — cosmetic
differences a real quote can pick up when the judge reproduces it (curly
quotes, contractions, sentence punctuation) should still match; a
genuinely different claim (a different number, a different fact) must not.

The second half of that sentence is the load-bearing guarantee: this
normalization exists to stop real evidence from being wrongly discarded as
"fabricated," not to make the fabricated-quote check more permissive. Every
test here that checks a *negative* case (numbers that must NOT collide) is
protecting against the exact regression that would turn this into an
accidental fuzzy-match.
"""

from agent_config_judge.judge import _normalize_for_match, validate_judge_output
from agent_config_judge.models import ConversationRecord, ConversationTurn


def _all_pass(overrides: dict) -> dict:
    base = {
        cid: {"verdict": "pass", "evidence_config_field": cid}
        for cid in (
            "system_prompt", "knowledge_base", "human_handoff", "fallback",
            "grounding", "multi_turn", "escalation_health", "sentiment", "latency",
        )
    }
    base.update(overrides)
    return {"criteria": base}


# --- direct unit tests on _normalize_for_match -----------------------------

def test_curly_quotes_match_straight_quotes():
    assert _normalize_for_match("“We’re sorry, we can’t help.”") == _normalize_for_match(
        '"We\'re sorry, we cannot help."'
    )


def test_contraction_matches_its_expansion():
    assert _normalize_for_match("We don't offer refunds.") == _normalize_for_match(
        "We do not offer refunds"
    )
    assert _normalize_for_match("I can't process that.") == _normalize_for_match(
        "I cannot process that"
    )


def test_trailing_and_embedded_punctuation_is_cosmetic():
    assert _normalize_for_match("You have 45 days to return it.") == _normalize_for_match(
        "You have 45 days to return it"
    )
    assert _normalize_for_match("Hello, how can I help?") == _normalize_for_match(
        "Hello how can I help"
    )


def test_decimal_and_thousands_punctuation_is_preserved():
    """The one thing this function must never do: let two different numbers
    collapse into the same normalized text just because a period or comma
    between digits got stripped."""
    assert _normalize_for_match("It costs $45.00 total.") != _normalize_for_match(
        "It costs $4500 total."
    )
    assert _normalize_for_match("We shipped 1,000 units.") != _normalize_for_match(
        "We shipped 1000 units."
    )


def test_a_genuinely_different_number_still_does_not_match():
    assert _normalize_for_match("You have 45 days to return it.") != _normalize_for_match(
        "You have 90 days to return it."
    )


# --- end-to-end through validate_judge_output ------------------------------

def test_quote_reproduced_with_different_contraction_and_punctuation_survives():
    convs = (ConversationRecord("c1", "widget", (
        ConversationTurn("agent", "We don't offer refunds after 14 days."),
    )),)
    raw = _all_pass({"grounding": {
        "verdict": "fail",
        # Same claim, judge reproduced it with "do not" instead of "don't"
        # and dropped the period — exactly the case this was built for.
        "evidence_quote": "We do not offer refunds after 14 days",
        "cause_code": "grounding_missing_source_attribution",
    }})
    j = validate_judge_output(raw, agent_id="a1", conversations=convs)
    assert j.criteria["grounding"].verdict == "fail"
    assert j.classification == "standard"


def test_altered_number_in_citation_is_still_discarded_as_fabricated():
    """The regression this whole module exists to prevent: a citation that
    changes the actual fact must still fail, even after normalization."""
    convs = (ConversationRecord("c1", "widget", (
        ConversationTurn("agent", "We accept returns within 14 days."),
    )),)
    raw = _all_pass({"grounding": {
        "verdict": "fail",
        "evidence_quote": "We accept returns within 90 days",
        "cause_code": "grounding_missing_source_attribution",
    }})
    j = validate_judge_output(raw, agent_id="a1", conversations=convs)
    assert j.criteria["grounding"].verdict == "unknown"
    assert any("fabricated" in n for n in j.validator_notes)
