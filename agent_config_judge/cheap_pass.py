"""Tier 1: the cheap pass.

Scores all nine rubric criteria from config fields and mechanically-derived
aggregate metrics only — no transcript reading, no LLM call. Its only job is
deciding who gets read in depth by the judge (tier 2).

Type signature is the enforcement mechanism, not a comment: score_agent()
takes an AgentConfigSnapshot and an AggregateMetrics, never an AgentSnapshot
or a ConversationRecord. There is no raw transcript text reachable from
here.

Asymmetric cost is the whole design brief: a false positive costs one judge
call; a false negative is an undetected broken agent costing a client. So
every threshold below is set to over-flag, and unknown signal is scored
worse than a pass, never neutral — "we don't know" is not "it's fine".

CALIBRATION STATUS: every threshold and weight in this file is a placeholder.
None of them have been tuned against a labeled portfolio yet — the eval
harness (eval/run_eval.py) measures where they land, and Section 5 of the
README reports the actual recall/precision/FP-rate numbers this file
produces on the golden set today. Recalibrate here first if that FP rate
crosses 30%.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_config_judge.models import AgentConfigSnapshot, AggregateMetrics
from agent_config_judge.rubric import CRITERION_ORDER

# --- placeholder thresholds -------------------------------------------------
# Every constant in this block is unvalidated. They exist so the logic below
# has *something* concrete to compute with; treat the values as "a guess
# that over-flags on purpose," not as a calibrated cutoff.

# Below this many characters we don't trust a system prompt to express a
# bounded role at all, regardless of what it says.
MIN_SYSTEM_PROMPT_CHARS = 40

# Overall score under this line gets flagged for the judge. Set high on
# purpose (see module docstring: over-flag, don't under-flag).
FLAG_SCORE_THRESHOLD = 85

# score assigned per criterion verdict before weighting; "unknown" scores
# worse than "pass" but not as bad as a confirmed "fail" — see module doc.
class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"


_VERDICT_SCORE = {Verdict.PASS: 100.0, Verdict.UNKNOWN: 40.0, Verdict.FAIL: 0.0}

# Latency bands are per-channel because a websocket web-widget round trip
# and a PSTN phone call have different acceptable TTFBs. Placeholder ms
# values — not measured against real channel baselines yet.
LATENCY_BAND_MS_BY_CHANNEL: dict[str, float] = {
    "twilio": 1200.0,
    "sip_trunk": 1200.0,
    "exotel": 1200.0,
    "audiocodes": 1200.0,
    "genesys": 1200.0,
    "genesys_bot_connector": 1200.0,
    "avaya": 1200.0,
    "react_sdk": 2000.0,
    "js_sdk": 2000.0,
    "react_native_sdk": 2000.0,
    "android_sdk": 2000.0,
    "swift_sdk": 2000.0,
    "flutter_sdk": 2000.0,
    "python_sdk": 2000.0,
    "node_js_sdk": 2000.0,
    "widget": 2000.0,
    "whatsapp": 2500.0,
    "twilio_sms": 2500.0,
}
DEFAULT_LATENCY_BAND_MS = 1800.0

# Grounding: fraction of specific-claim turns with no KB doc used, above
# which the criterion fails outright rather than just losing score.
GROUNDING_UNSOURCED_FAIL_RATE = 0.5

# Escalation health: neither of these bands should be true.
ESCALATION_RATE_FLOOR = 0.0  # exactly zero, with any conversations sampled, is suspicious
ESCALATION_RATE_CEILING = 0.6  # escalating most conversations looks like a broken agent, not a helpful one

# Multi-turn: repeat-detector rate above this fails.
REPEAT_QUESTION_FAIL_RATE = 0.2

# Sentiment: negative-turn rate above this fails.
NEGATIVE_SENTIMENT_FAIL_RATE = 0.25

# Latency: over-band rate above this fails.
LATENCY_OVER_BAND_FAIL_RATE = 0.3

# Phone-only transfer tool, but the agent's observed channels are all
# non-telephony: this is the exact config-passes-but-runtime-fails case
# from the case study. Cheap pass gets a *partial* heuristic for it from
# structured channel metadata (not transcript text) — but the guaranteed
# catch is the forced-flag rule below, because this heuristic can miss
# cases the forced flag won't (e.g. the mismatch is subtler than channel
# family, or the sample didn't happen to include that tool's call).
_PHONE_ONLY_CHANNELS = frozenset(
    {"twilio", "exotel", "sip_trunk", "audiocodes", "genesys", "genesys_bot_connector", "avaya"}
)


@dataclass(frozen=True)
class CriterionScore:
    criterion_id: str
    verdict: Verdict
    score: float
    detail: str


@dataclass(frozen=True)
class CheapPassResult:
    agent_id: str
    criterion_scores: tuple[CriterionScore, ...]
    score: float  # 0-100
    flagged: bool
    forced_flag: bool  # True if flagged regardless of score (tool error in sample)
    flag_reasons: tuple[str, ...]

    def criterion(self, criterion_id: str) -> CriterionScore:
        for c in self.criterion_scores:
            if c.criterion_id == criterion_id:
                return c
        raise KeyError(criterion_id)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def _score_system_prompt(config: AgentConfigSnapshot) -> CriterionScore:
    prompt = config.system_prompt or ""
    if len(prompt.strip()) < MIN_SYSTEM_PROMPT_CHARS:
        return CriterionScore(
            "system_prompt", Verdict.FAIL, _VERDICT_SCORE[Verdict.FAIL],
            f"system_prompt is {len(prompt.strip())} chars (< {MIN_SYSTEM_PROMPT_CHARS}); "
            "too short to express a bounded role.",
        )
    # Cheap pass cannot detect self-contradiction — that needs a reader, not
    # a length check. A present, non-trivial prompt is the best this tier
    # can confirm; contradiction detection is explicitly a judge-tier job.
    return CriterionScore(
        "system_prompt", Verdict.PASS, _VERDICT_SCORE[Verdict.PASS],
        f"system_prompt present ({len(prompt.strip())} chars). "
        "NOTE: cheap pass cannot detect self-contradiction; judge tier can.",
    )


def _score_knowledge_base(config: AgentConfigSnapshot) -> CriterionScore:
    n = len(config.knowledge_base_ids)
    if n == 0:
        # We can't tell from config alone whether this agent's job requires
        # a KB, so this is "unknown", not a hard "fail" — but unknown still
        # counts against the score per the module's asymmetric-cost design.
        return CriterionScore(
            "knowledge_base", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN],
            "No knowledge_base sources attached. Unknown whether this agent's "
            "job requires one — scored as unknown, not pass.",
        )
    return CriterionScore(
        "knowledge_base", Verdict.PASS, _VERDICT_SCORE[Verdict.PASS],
        f"{n} knowledge_base source(s) attached.",
    )


def _score_human_handoff(config: AgentConfigSnapshot, metrics: AggregateMetrics) -> CriterionScore:
    has_number_transfer = config.has_tool_type("transfer_to_number")
    has_agent_transfer = config.has_tool_type("transfer_to_agent")

    if not has_number_transfer and not has_agent_transfer:
        return CriterionScore(
            "human_handoff", Verdict.FAIL, _VERDICT_SCORE[Verdict.FAIL],
            "No transfer_to_number or transfer_to_agent tool configured.",
        )

    if has_agent_transfer:
        # Agent-to-agent transfer works on every channel — no runtime
        # channel dependency to check.
        return CriterionScore(
            "human_handoff", Verdict.PASS, _VERDICT_SCORE[Verdict.PASS],
            "transfer_to_agent configured (channel-independent).",
        )

    # Only transfer_to_number is configured: this is exactly the shape of
    # the motivating case. Cross-check against observed channels, which are
    # structured metadata (conversation_initiation_source), not transcript
    # text, so this stays in-bounds for a cheap-pass heuristic.
    channels = set(metrics.channels_seen)
    if channels and channels.issubset(_PHONE_ONLY_CHANNELS):
        return CriterionScore(
            "human_handoff", Verdict.PASS, _VERDICT_SCORE[Verdict.PASS],
            f"transfer_to_number configured; observed channels {sorted(channels)} are telephony.",
        )
    if channels:
        return CriterionScore(
            "human_handoff", Verdict.FAIL, _VERDICT_SCORE[Verdict.FAIL],
            f"transfer_to_number configured, but observed channels {sorted(channels)} "
            "include non-telephony channels where this tool cannot function "
            "(requires Twilio/Exotel/SIP). Config alone would have passed this.",
        )
    # No channel data in the sample at all — can't rule the mismatch in or
    # out from config. Unknown, not pass: this is the gap the forced flag
    # (see should_force_flag) exists to backstop.
    return CriterionScore(
        "human_handoff", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN],
        "transfer_to_number configured, but no channel data in the sample to "
        "confirm it can actually run there.",
    )


def _score_fallback(metrics: AggregateMetrics) -> CriterionScore:
    # No first-class "guessed then escalated" signal exists anywhere in
    # config or aggregate counts — sequencing within a conversation is
    # exactly the kind of thing that needs a reader. Cheap pass has no
    # honest proxy for this criterion at all, so it always reports
    # unknown rather than pretending a weak correlate is a score.
    return CriterionScore(
        "fallback", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN],
        "No cheap-pass proxy exists for guess-then-escalate sequencing; "
        "always unknown at this tier, judged only by the transcript reader.",
    )


def _score_grounding(metrics: AggregateMetrics) -> CriterionScore:
    rate = _rate(metrics.specific_claim_turns_without_kb, metrics.specific_claim_turns)
    if rate is None:
        return CriterionScore(
            "grounding", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN],
            "No specific factual claims observed in the sample to check.",
        )
    verdict = Verdict.FAIL if rate >= GROUNDING_UNSOURCED_FAIL_RATE else Verdict.PASS
    score = _VERDICT_SCORE[verdict] if verdict == Verdict.FAIL else max(0.0, 100.0 * (1 - rate))
    return CriterionScore(
        "grounding", verdict, score,
        f"{metrics.specific_claim_turns_without_kb}/{metrics.specific_claim_turns} "
        f"specific-claim turns had no attributable source ({rate:.0%}) — no KB doc used, "
        "no adjacent tool call, and the claim wasn't an echo of something the user just "
        "said. NOTE: still a proxy — attribution is per TURN, not per individual claim, "
        "so a turn mixing one grounded and one fabricated claim reads as fine either way.",
    )


def _score_multi_turn(metrics: AggregateMetrics) -> CriterionScore:
    rate = _rate(metrics.repeated_question_conversations, metrics.n_conversations_sampled)
    if rate is None:
        return CriterionScore("multi_turn", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN], "No conversations sampled.")
    verdict = Verdict.FAIL if rate > REPEAT_QUESTION_FAIL_RATE else Verdict.PASS
    score = _VERDICT_SCORE[verdict] if verdict == Verdict.FAIL else max(0.0, 100.0 * (1 - rate))
    return CriterionScore(
        "multi_turn", verdict, score,
        f"{metrics.repeated_question_conversations}/{metrics.n_conversations_sampled} "
        f"sampled conversations have a user turn repeating an earlier one verbatim ({rate:.0%}). "
        "NOTE: only catches literal repeats (e.g. re-stating a phone number the ASR missed) — "
        "the more common real failure, the agent re-asking the same thing in different words, "
        "needs to understand that two different phrasings mean the same question, which this "
        "tier deliberately doesn't attempt; that's judged only by the transcript reader.",
    )


def _score_escalation_health(metrics: AggregateMetrics) -> CriterionScore:
    rate = _rate(metrics.conversations_with_escalation, metrics.n_conversations_sampled)
    if rate is None:
        return CriterionScore("escalation_health", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN], "No conversations sampled.")
    if rate <= ESCALATION_RATE_FLOOR:
        return CriterionScore(
            "escalation_health", Verdict.FAIL, _VERDICT_SCORE[Verdict.FAIL],
            f"0/{metrics.n_conversations_sampled} sampled conversations escalated. NOTE: this "
            "metric only counts transfer_to_number/transfer_to_agent tool calls — an agent "
            "that escalates via a ticket-creation tool will show zero here; see golden set "
            "false-positive trap.",
        )
    if rate > ESCALATION_RATE_CEILING:
        return CriterionScore(
            "escalation_health", Verdict.FAIL, _VERDICT_SCORE[Verdict.FAIL],
            f"{metrics.conversations_with_escalation}/{metrics.n_conversations_sampled} "
            f"conversations escalated ({rate:.0%}) — looks like runaway escalation.",
        )
    return CriterionScore(
        "escalation_health", Verdict.PASS, _VERDICT_SCORE[Verdict.PASS],
        f"{metrics.conversations_with_escalation}/{metrics.n_conversations_sampled} "
        f"conversations escalated ({rate:.0%}), within the placeholder band.",
    )


def _score_sentiment(metrics: AggregateMetrics) -> CriterionScore:
    rate = _rate(metrics.negative_sentiment_turns, metrics.agent_turns_sampled)
    if rate is None:
        return CriterionScore("sentiment", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN], "No agent turns sampled.")
    verdict = Verdict.FAIL if rate > NEGATIVE_SENTIMENT_FAIL_RATE else Verdict.PASS
    score = _VERDICT_SCORE[verdict] if verdict == Verdict.FAIL else max(0.0, 100.0 * (1 - rate))
    return CriterionScore(
        "sentiment", verdict, score,
        f"{metrics.negative_sentiment_turns}/{metrics.agent_turns_sampled} agent turns follow a "
        f"frustration-keyword match ({rate:.0%}). NOTE: keyword heuristic can't attribute cause — "
        "a user already angry about something unrelated will also trip this; judge tier decides "
        "whether the agent caused it.",
    )


def _score_latency(metrics: AggregateMetrics) -> CriterionScore:
    rate = _rate(metrics.turns_over_latency_band, metrics.turns_with_latency_data)
    if rate is None:
        return CriterionScore("latency", Verdict.UNKNOWN, _VERDICT_SCORE[Verdict.UNKNOWN], "No TTFB data in the sample.")
    verdict = Verdict.FAIL if rate > LATENCY_OVER_BAND_FAIL_RATE else Verdict.PASS
    score = _VERDICT_SCORE[verdict] if verdict == Verdict.FAIL else max(0.0, 100.0 * (1 - rate))
    return CriterionScore(
        "latency", verdict, score,
        f"{metrics.turns_over_latency_band}/{metrics.turns_with_latency_data} turns exceeded "
        f"their channel's placeholder latency band ({rate:.0%}).",
    )


def should_force_flag(metrics: AggregateMetrics) -> bool:
    """Any tool error in the sample forces a judge read regardless of score.

    This is the direct answer to the case-study finding: a config-approved
    transfer_to_number tool that fails at runtime with "only available for
    ... Twilio, Exotel, or SIP trunking" shows up nowhere in config checks —
    the only trace is tool_results[].is_error in the transcript. Rather than
    trying to out-guess every way a configured tool can fail at runtime, we
    just force every such agent to the judge. Cheap, because it's one
    boolean OR over a count the client already computed; not a proxy that
    can be "close enough" — either the sample recorded an error or it didn't.
    """
    return metrics.tool_error_count > 0


def score_agent(config: AgentConfigSnapshot, metrics: AggregateMetrics) -> CheapPassResult:
    scorers = {
        "system_prompt": lambda: _score_system_prompt(config),
        "knowledge_base": lambda: _score_knowledge_base(config),
        "human_handoff": lambda: _score_human_handoff(config, metrics),
        "fallback": lambda: _score_fallback(metrics),
        "grounding": lambda: _score_grounding(metrics),
        "multi_turn": lambda: _score_multi_turn(metrics),
        "escalation_health": lambda: _score_escalation_health(metrics),
        "sentiment": lambda: _score_sentiment(metrics),
        "latency": lambda: _score_latency(metrics),
    }
    criterion_scores = tuple(scorers[cid]() for cid in CRITERION_ORDER)

    # Equal weighting across all nine criteria — a placeholder simplification.
    # Weighting config-checkable criteria higher (they're more reliable) is
    # an obvious next calibration step; not done here because it needs a
    # labeled portfolio to justify specific weights rather than a guess.
    overall = sum(c.score for c in criterion_scores) / len(criterion_scores)

    forced = should_force_flag(metrics)
    reasons: list[str] = []
    if forced:
        reasons.append(
            f"forced: {metrics.tool_error_count} tool error(s) observed in the sample "
            "(config-passing tool may still fail at runtime)"
        )
    if overall < FLAG_SCORE_THRESHOLD:
        reasons.append(f"score {overall:.1f} < flag threshold {FLAG_SCORE_THRESHOLD}")

    flagged = forced or overall < FLAG_SCORE_THRESHOLD

    return CheapPassResult(
        agent_id=config.agent_id,
        criterion_scores=criterion_scores,
        score=round(overall, 1),
        flagged=flagged,
        forced_flag=forced,
        flag_reasons=tuple(reasons),
    )
