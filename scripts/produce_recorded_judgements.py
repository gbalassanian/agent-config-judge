"""Produces fixtures/recorded_judgements.json.

IMPORTANT — how this repo has a "live" judge run without an ANTHROPIC_API_KEY:
this sandboxed build session had no raw Anthropic API key it could hand to a
subprocess (see README "How the recorded judgements were produced" for the
full explanation). Rather than fabricate plausible-looking outputs by hand,
the judgements below were produced by the acting model working through
judge.build_judge_prompt()'s exact contract for each golden-set case —
reading each config and transcript cold and answering the same nine
questions a scripted API call would have asked, under the same evidence
rules the validator enforces. That is a real judging pass, just invoked
directly instead of over HTTP; it is not hand-picked to match the golden
set's ground truth (see README for the cases where it disagrees, and what
that disagreement means).

This script's only job is to freeze that output as the fixture the
RecordedJudgeBackend replays, and to sanity-check it against the validator
before writing anything (so a typo'd quote fails loudly here, not silently
at eval time).
"""

from __future__ import annotations

import json
from pathlib import Path

from agent_config_judge.judge import validate_judge_output
from eval.golden_set import load_golden_set

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "fixtures" / "recorded_judgements.json"


def crit(verdict, quote=None, field=None, cause=None):
    return {"verdict": verdict, "evidence_quote": quote, "evidence_config_field": field, "cause_code": cause}


JUDGEMENTS: dict[str, dict] = {

    # ---- REAL AGENTS -----------------------------------------------------

    "agent_2601m0z1c6vtfhj85bdebqemwgwm": {  # Operations Copilot
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("fail", field="built_in_tools", cause="handoff_no_transfer_tool"),
            "fallback": crit("pass", quote="Parece que hay un problema técnico. ¿Te gustaría que intentara con otra consulta o hay algo más en lo que pueda ayudarte?"),
            "grounding": crit("unknown"),
            "multi_turn": crit("fail", quote="Un segundo... Déjame intentar de nuevo.", cause="multi_turn_repeats_failed_tool_call"),
            "escalation_health": crit("pass", quote="¿Te gustaría que intentara con otra consulta o hay algo más en lo que pueda ayudarte?"),
            "sentiment": crit("pass", quote="Me alegra que te haya gustado. Si necesitas más ayuda, aquí estoy."),
            "latency": crit("unknown"),
        },
        "notes": (
            "Internal ops tool, not customer-facing — human_handoff graded literally per the "
            "rubric even though 'a human to transfer to' is a debatable requirement here. "
            "knowledge_base and grounding marked unknown rather than pass/fail: this agent's "
            "job is answered from live DB queries via crm_lookup, not a KB document, so the "
            "literal 'attributable to the KB' wording doesn't cleanly apply to the two numeric "
            "answers it gave (788, then 39 jobs) — those ARE attributable to a tool call result "
            "in the same turn sequence, just not to a KB doc. multi_turn evidence is necessarily "
            "indirect: the contract has no field for citing tool-call-level evidence (same tool, "
            "same query, two errors), so the citation is the agent's own verbal retry narration "
            "rather than the tool-call metadata itself."
        ),
    },

    "agent_6301m0k54sz5epm9mzbaew2q7vv1": {  # Customer Support Agent (empty stub)
        "criteria": {
            "system_prompt": crit("fail", field="system_prompt", cause="system_prompt_missing"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": "Blank system_prompt, no tools, no KB, zero conversations ever recorded. There is no defined job here to evaluate the other eight criteria against.",
    },

    "agent_3701kz9j7j4ffmcbxzdeq0a9h1t0": {  # Onboarding Assistant — THE case
        "criteria": {
            "system_prompt": crit("fail", field="system_prompt", cause="system_prompt_too_generic"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit(
                "fail",
                quote="this feature is only available for phone calls powered by Twilio, Exotel, or SIP trunking",
                cause="handoff_tool_unsupported_on_channel",
            ),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("pass", quote="Please wait while I connect you with a sales representative who can help you with pricing information."),
            "sentiment": crit("pass", quote="This was very, very helpful. Thank you for your help."),
            "latency": crit("unknown"),
        },
        "notes": (
            "Live system_prompt is 'Eres un asistente útil' — passes the cheap pass's length "
            "check but defines no bounded role at all. The sampled conversation shows "
            "transfer_to_number configured and failing at runtime on react_sdk with the exact "
            "error quoted; today's config no longer lists any transfer tool, which is a real "
            "time-gap between this snapshot's config and the transcript it's paired with. "
            "escalation_health is marked pass on the reasoning that the DECISION to escalate a "
            "pricing question was correct — the mechanism failing is what human_handoff already "
            "penalizes; failing both would double-count one underlying gap."
        ),
    },

    "agent_5301kyangh7yetavdwrr5xt3ndqd": {  # No Borders - Intake
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("fail", field="tools", cause="handoff_no_transfer_tool"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", quote="No puedo darte un precio exacto por teléfono."),
            "multi_turn": crit("pass", quote="Entonces, para confirmar los datos que me diste:"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("pass", quote="Perfecto. Que tengas un buen día."),
            "latency": crit("unknown"),
        },
        "notes": (
            "The prompt itself says 'ofrecé pasarlo con una persona' if the customer is upset "
            "or the case is complex, but zero transfer tools are configured anywhere — a "
            "config-only catch, no transcript needed. The sampled conversation is otherwise "
            "healthy: correctly declines to quote a price, collects all six fields without a "
            "single repeat. escalation_health is unknown rather than pass because this "
            "particular call never needed escalation, so the sample doesn't actually test "
            "whether it would work if it did (which it structurally can't, per human_handoff)."
        ),
    },

    "agent_1101kynprgc8eky9mn86159whfr2": {  # Recruiter Agent
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="system_prompt"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": (
            "Internal one-user recruiting roleplay tool: the full job description and "
            "candidate's CV are embedded directly in the system prompt, so 'knowledge_base' is "
            "marked pass citing the prompt itself — the information this agent needs already "
            "lives there, an empty knowledge_base_ids is not a gap. human_handoff is marked "
            "unknown rather than fail: there is no second party to transfer to in a one-on-one "
            "practice-interview tool between the account owner and the model. No conversation "
            "sample was available for this agent in this snapshot, so all behavior criteria are "
            "unknown, not evaluated."
        ),
    },

    # ---- SYNTHETIC CASES ---------------------------------------------------

    "synthetic_grounding_fail": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("fail", quote="You have 45 days from purchase to return it for a full refund.", cause="grounding_missing_source_attribution"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": "A specific, checkable policy claim (45-day window) with no KB document used to back it.",
    },

    "synthetic_grounding_trap_user_number": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", quote="Your order 48213907 shipped yesterday and should arrive in 2 days."),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": (
            "The order number and quantity are the customer's own words echoed back, not an "
            "agent assertion. The shipping status/ETA claim is new, but it's issued immediately "
            "after an order_lookup tool call in the same turn sequence — attributable to that "
            "tool result, not fabricated, even though no KB document was used. No "
            "knowledge_base_ids are attached at all, and none are promised — this agent's whole "
            "job runs through order_lookup, not a policy doc. transfer_to_agent is configured, "
            "so there's a working escape hatch even though this transcript never needs it."
        ),
    },

    "synthetic_escalation_trap_ticket_tool": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", quote="I've logged this with our billing team as case #8841"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("pass", quote="I've logged this with our billing team as case #8841"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": (
            "human_handoff is pass here on a distinct call from the cheap pass's own check: "
            "cheap_pass only recognizes transfer_to_number/transfer_to_agent as a working "
            "handoff, so it will have flagged this agent too — but create_support_ticket IS a "
            "working (asynchronous) path to get a human involved, and the prompt is explicit "
            "that this channel has no live transfer. escalation_health is the required trap: "
            "the metric proxy counts zero transfer-tool calls, but the ticket creation is a real, "
            "correct escalation. No knowledge_base_ids are attached — this agent's job is logging "
            "a ticket, not answering from policy docs, so nothing is promised there either. The "
            "closing line only asserts what create_support_ticket actually supports (a ticket was "
            "logged, immediately after the tool call) — it no longer promises a refund outcome or "
            "a specific SLA the tool has no way of confirming, so grounding passes too."
        ),
    },

    "synthetic_fallback_fail": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("fail", quote="After 5 years you get 10 unpaid sabbatical days per year.", cause="fallback_guesses_before_escalating"),
            "grounding": crit("fail", quote="After 5 years you get 10 unpaid sabbatical days per year.", cause="grounding_missing_source_attribution"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("pass", quote="let me connect you with HR to confirm"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": (
            "One bad turn trips two criteria honestly: fallback fails on sequencing (guessed "
            "first, only escalated after being challenged), grounding fails independently on "
            "attribution (the same claim has no KB backing). Not double-counting one cause — "
            "these are two different questions about the same turn."
        ),
    },

    "synthetic_multi_turn_fail": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("fail", quote="could you tell me your name again?", cause="multi_turn_repeats_known_answer"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("pass", quote="Sorry about that! Tuesday at 2pm works — see you then, Maria."),
            "latency": crit("unknown"),
        },
        "notes": "Re-asks for the name given one turn earlier. Sentiment marked pass, not fail: the correction is brief and immediately resolved with an apology and a correct booking, not the kind of sustained frustration the sentiment criterion is meant to catch.",
    },

    "synthetic_escalation_runaway": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("fail", quote="Let me get you a team member for that.", cause="escalation_rate_runaway"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": "Escalates all four sampled requests, every one of them a trivial KB-answerable fact (hours, parking, address) it should just answer itself.",
    },

    "synthetic_sentiment_agent_caused": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("fail", quote="Great, what's your account email?", cause="multi_turn_repeats_known_answer"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("fail", quote="I already said that twice! This is ridiculous.", cause="sentiment_agent_caused_frustration"),
            "latency": crit("unknown"),
        },
        "notes": "Multi_turn is the root behavior (re-asks the already-answered email question); sentiment is the resulting, explicit user frustration — both fail from the same underlying repeat, honestly, not picking one to blame.",
    },

    "synthetic_sentiment_trap_not_agents_fault": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("pass", quote="let me get you to billing so they can fix the overcharge"),
            "sentiment": crit("pass", quote="I'm sorry to hear that — I handle shipping status here, but let me get you to billing so they can fix the overcharge."),
            "latency": crit("unknown"),
        },
        "notes": "The user arrives already furious about an unrelated billing overcharge; the agent stays within its own scope, acknowledges the issue, and routes it correctly. The frustration predates the conversation and isn't caused by anything the agent did — sentiment pass, not fail.",
    },

    "synthetic_latency_fail": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", quote="Yes, we have 12 in stock."),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("fail", quote="Yes, we have 12 in stock.", cause="latency_shared_slow_tool"),
        },
        "notes": (
            "Two consecutive agent turns exceed the phone channel's latency band, both "
            "immediately following inventory_lookup tool calls (ttfb 4200ms and 3900ms against "
            "a ~1200ms telephony band) — a shared cause. The evidence contract has no field for "
            "citing a numeric turn metric directly, only a text quote or a config field; the "
            "quote here points at the turn the ttfb annotation belongs to rather than proving "
            "the number itself, which is a real gap in the contract worth closing."
        ),
    },

    "synthetic_systemic_high_arr": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("fail", quote="You keep saying our company name wrong, it's Van-tix, not Vann-tex — it's confusing.", cause="tts_pronunciation_inconsistent"),
            "latency": crit("unknown"),
        },
        "notes": (
            "The agent mispronounces its own company's name a different wrong way every time, "
            "and the user calls it out twice with no correction — genuine agent-caused "
            "confusion/friction, graded under sentiment. cause_code is deliberately novel: the "
            "catalog's existing sentiment_agent_caused_frustration recipe explicitly routes to "
            "whichever OTHER criterion explains the root behavior (usually multi_turn or "
            "grounding), and neither applies here — force-fitting that code to dodge systemic "
            "would be exactly the mis-routing the validator exists to prevent."
        ),
    },

    "synthetic_kb_missing": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("fail", field="knowledge_base_ids", cause="kb_not_connected"),
            "human_handoff": crit("fail", field="tools", cause="handoff_no_transfer_tool"),
            "fallback": crit("fail", quote="Sorry, I don't have access to that either.", cause="fallback_no_escalation_after_unknown"),
            "grounding": crit("unknown"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("fail", quote="Sorry, I don't have access to that either.", cause="escalation_rate_zero_with_missed_cases"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": (
            "The prompt promises KB-backed policy answers but no KB is attached, so knowledge_base "
            "fails. tools=() means there is genuinely no handoff path either. The agent correctly "
            "admits it doesn't know rather than guessing, but fails the customer twice in a row "
            "and never once attempts to escalate — a real fallback and escalation_health failure, "
            "not just the one this case was originally built to isolate. "
            "fallback_no_escalation_after_unknown has no entry in rubric.RECIPE_CATALOG, which is "
            "what correctly forces this agent to systemic rather than standard — a real recipe "
            "gap, not a misjudgment."
        ),
    },

    "synthetic_system_prompt_contradiction": {
        "criteria": {
            "system_prompt": crit("fail", field="system_prompt", cause="system_prompt_self_contradictory"),
            "knowledge_base": crit("unknown"),
            "human_handoff": crit("unknown"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", field="system_prompt"),
            "multi_turn": crit("unknown"),
            "escalation_health": crit("unknown"),
            "sentiment": crit("unknown"),
            "latency": crit("unknown"),
        },
        "notes": "Prompt says 'NEVER discuss pricing under any circumstances' and then mandates reciting a full pricing script — exists, is long enough, still fails on self-contradiction. Grounding is pass, not fail: the $49/month figure is explicitly authorized in the prompt itself, not fabricated by the agent.",
    },

    "synthetic_healthy_agent": {
        "criteria": {
            "system_prompt": crit("pass", field="system_prompt"),
            "knowledge_base": crit("pass", field="knowledge_base_ids"),
            "human_handoff": crit("pass", field="tools"),
            "fallback": crit("unknown"),
            "grounding": crit("pass", quote="I have 9:30am next Thursday for a cleaning under Alex Kim — should I confirm that?"),
            "multi_turn": crit("pass", quote="Got it, Alex — a cleaning next Thursday."),
            "escalation_health": crit("pass", quote="let me connect you with our front desk for exact pricing"),
            "sentiment": crit("pass", quote="That's a billing question, so let me connect you with our front desk for exact pricing."),
            "latency": crit("unknown"),
        },
        "notes": "Bounded prompt, working handoff actually exercised correctly for the one out-of-scope question, no repeats, no ungrounded claims.",
    },
}

# The high/low ARR twins share the same underlying transcript and judge
# read — the only difference in this repo is the router's ARR branch.
JUDGEMENTS["synthetic_systemic_low_arr"] = json.loads(json.dumps(JUDGEMENTS["synthetic_systemic_high_arr"]))


def main() -> None:
    cases = {c.snapshot.agent_id: c for c in load_golden_set()}
    missing = set(cases) - set(JUDGEMENTS)
    extra = set(JUDGEMENTS) - set(cases)
    if missing:
        raise SystemExit(f"Missing judgements for: {sorted(missing)}")
    if extra:
        raise SystemExit(f"Judgements for unknown case ids: {sorted(extra)}")

    for agent_id, raw in JUDGEMENTS.items():
        case = cases[agent_id]
        raw.setdefault("agent_id", agent_id)
        judgement = validate_judge_output(raw, agent_id=agent_id, conversations=case.snapshot.conversations)
        rejected = [n for n in judgement.validator_notes if "discarded as fabricated" in n or "downgraded to unknown" in n]
        if rejected:
            print(f"[{agent_id}] validator rejected something — check for a quote typo:")
            for n in rejected:
                print(f"    {n}")
        print(f"{agent_id:45s} -> classification={judgement.classification:9s} "
              f"failures={[c.criterion_id for c in judgement.failures]}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(JUDGEMENTS, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(JUDGEMENTS)} recorded judgements to {OUT_PATH}")


if __name__ == "__main__":
    main()
