"""The golden set: labeled cases the eval harness (eval/run_eval.py) scores
the pipeline against.

Two kinds of case, and they're never allowed to blur together:

  - REAL cases (source="real"): the 5 agents in
    fixtures/real_portfolio_snapshot.json, loaded as-is. Ground truth here
    is this repo's own best-effort read of what's actually true about
    each agent, argued in each case's `notes` — including the ambiguous
    calls (e.g. does an internal ops tool need a customer "human
    handoff"?). See README for the full discussion.

  - SYNTHETIC cases (source="synthetic"): hand-built to cover failure
    shapes the 5 real agents didn't happen to exhibit, including the two
    required false-positive traps (grounding on a user-supplied number,
    escalation via an uncounted ticket tool) plus a few more of the same
    kind. Every synthetic case's synthetic_note says exactly what it
    stands in for — AgentSnapshot.__post_init__ enforces that a synthetic
    snapshot can't be constructed without one.

Ground truth per case is a GoldenCase: the AgentSnapshot, the classification
a correct judge should reach, which criteria should come back FAIL and
with what cause_code, and whether the cheap pass should flag it at all
(this is the target for cheap-pass recall/precision, kept separate from
judge classification accuracy — they're different failure surfaces, see
eval/run_eval.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_config_judge.elevenlabs_client import compute_aggregate_metrics
from agent_config_judge.models import (
    AgentConfigSnapshot,
    AgentSnapshot,
    ConversationRecord,
    ConversationTurn,
    ToolCallRecord,
    ToolConfig,
    agent_snapshot_from_dict,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_SNAPSHOT_PATH = REPO_ROOT / "fixtures" / "real_portfolio_snapshot.json"


@dataclass(frozen=True)
class GoldenCase:
    snapshot: AgentSnapshot
    source: str  # "real" | "synthetic"
    should_flag: bool  # target for cheap-pass recall/precision
    expected_classification: str  # "healthy" | "standard" | "systemic" — what a correct judge concludes
    # criterion_id -> expected cause_code (None if the criterion should
    # come back pass/unknown, i.e. not a failure at all)
    expected_failures: dict[str, str] = field(default_factory=dict)
    notes: str = ""


def _turn(role: str, text: str, kb_ids: tuple[str, ...] = (), tool_calls: tuple[ToolCallRecord, ...] = (), ttfb_ms: float | None = None) -> ConversationTurn:
    return ConversationTurn(role=role, text=text, used_static_kb_document_ids=kb_ids, tool_calls=tool_calls, ttfb_ms=ttfb_ms)


def _snapshot(agent_id: str, name: str, system_prompt: str, tools: tuple[ToolConfig, ...],
              kb_ids: tuple[str, ...], conversations: tuple[ConversationRecord, ...],
              arr_usd: float, synthetic_note: str) -> AgentSnapshot:
    config = AgentConfigSnapshot(agent_id=agent_id, name=name, system_prompt=system_prompt,
                                  knowledge_base_ids=kb_ids, tools=tools)
    metrics = compute_aggregate_metrics(list(conversations))
    return AgentSnapshot(agent_id=agent_id, name=name, config=config, metrics=metrics,
                          conversations=conversations, arr_usd=arr_usd,
                          is_synthetic=True, synthetic_note=synthetic_note)


def _load_real_cases() -> list[GoldenCase]:
    with open(REAL_SNAPSHOT_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    snapshots = {d["agent_id"]: agent_snapshot_from_dict(d) for d in raw}

    return [
        GoldenCase(
            snapshot=snapshots["agent_2601m0z1c6vtfhj85bdebqemwgwm"],  # Operations Copilot
            source="real", should_flag=True, expected_classification="standard",
            expected_failures={
                "human_handoff": "handoff_no_transfer_tool",
                "multi_turn": "multi_turn_repeats_failed_tool_call",
            },
            notes=(
                "Internal ops tool for employees, not a customer-facing agent — whether "
                "'human handoff' meaningfully applies here at all is genuinely arguable; "
                "graded per the rubric literally rather than softened for context. The "
                "multi_turn failure is the live crm_lookup 401-retry-verbatim case."
            ),
        ),
        GoldenCase(
            snapshot=snapshots["agent_6301m0k54sz5epm9mzbaew2q7vv1"],  # Customer Support Agent (stub)
            source="real", should_flag=True, expected_classification="standard",
            expected_failures={"system_prompt": "system_prompt_missing"},
            notes="Completely unconfigured stub: blank prompt, no tools, never called.",
        ),
        GoldenCase(
            snapshot=snapshots["agent_3701kz9j7j4ffmcbxzdeq0a9h1t0"],  # Onboarding Assistant
            source="real", should_flag=True, expected_classification="standard",
            expected_failures={"human_handoff": "handoff_tool_unsupported_on_channel"},
            notes=(
                "THE motivating case. The sampled conversation shows transfer_to_number "
                "configured and failing at runtime on react_sdk with the exact case-study "
                "error; today's live config no longer lists any transfer tool at all "
                "(likely removed since). Ground truth here follows the transcript evidence "
                "the judge actually has to cite, not the current config's absence of a tool — "
                "a real config/transcript time-drift this repo did not paper over."
            ),
        ),
        GoldenCase(
            snapshot=snapshots["agent_5301kyangh7yetavdwrr5xt3ndqd"],  # No Borders - Intake
            source="real", should_flag=True, expected_classification="standard",
            expected_failures={"human_handoff": "handoff_no_transfer_tool"},
            notes=(
                "Prompt literally promises 'ofrecé pasarlo con una persona' if the customer "
                "is upset or the case is complex; zero transfer tools configured. Config-only "
                "catch, no transcript needed. The sampled conversation itself is genuinely "
                "healthy (no price given, no repeats, all six fields collected once each)."
            ),
        ),
        GoldenCase(
            snapshot=snapshots["agent_1101kynprgc8eky9mn86159whfr2"],  # Recruiter Agent
            source="real", should_flag=True, expected_classification="healthy",
            expected_failures={},
            notes=(
                "Internal one-user recruiting roleplay tool with the full job description "
                "and candidate CV embedded in the prompt. Cheap pass flags missing "
                "knowledge_base and human_handoff — neither applies to a self-contained "
                "roleplay script for one person's practice interview. A correct judge reads "
                "the prompt's context and does not fail either criterion; this is a REAL "
                "instance of the cheap pass's designed-in over-flagging, distinct from the "
                "two synthetic traps below."
            ),
        ),
    ]


def _synthetic_cases() -> list[GoldenCase]:
    cases: list[GoldenCase] = []

    # S1 — grounding failure: a specific, checkable claim with no KB backing.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_grounding_fail", "Synthetic: Warranty Support Bot",
            "You are a warranty support agent for Acme Appliances. Only answer from the "
            "attached warranty knowledge base; if it's not in the KB, say you don't know.",
            tools=(), kb_ids=("kb_warranty_docs",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, how can I help with your warranty today?"),
                _turn("user", "How many days do I have to return a broken blender?"),
                _turn("agent", "You have 45 days from purchase to return it for a full refund.", kb_ids=()),
            )),),
            arr_usd=20000.0,
            synthetic_note="Grounding failure: specific claim (45-day window) with no KB document used to back it.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"grounding": "grounding_missing_source_attribution"},
        notes="The 45-day figure is stated as fact with no used_static_kb_document_ids — a real grounding failure, not the user-supplied-number trap below.",
    ))

    # S2 — TRAP: grounding proxy flags it, but the number came from the user.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_grounding_trap_user_number", "Synthetic: Order Status Bot",
            "You are an order status agent. Confirm the order number the customer gives you "
            "and look up its status.",
            tools=(), kb_ids=("kb_policies",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi! What's your order number?"),
                _turn("user", "It's 48213907, and I ordered 3 units."),
                _turn("agent", "Got it — order 48213907, 3 units. Let me check the status for you.", kb_ids=()),
                _turn("agent", "Your order 48213907 shipped yesterday and should arrive in 2 days."),
            )),),
            arr_usd=15000.0,
            synthetic_note=(
                "FALSE-POSITIVE TRAP (required): cheap pass's grounding proxy sees a specific-looking "
                "claim (numbers, 'day') with no KB doc used and flags it — but the agent is just "
                "echoing back the order number and quantity the USER supplied, not asserting an "
                "ungrounded fact. Correct grounding verdict is pass."
            ),
        ),
        source="synthetic", should_flag=True, expected_classification="healthy",
        expected_failures={},
        notes="Required trap #1: low/zero attribution caused by user-supplied numbers, not a real grounding gap.",
    ))

    # S3 — TRAP: escalation proxy counts zero because escalation happens via a ticket tool.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_escalation_trap_ticket_tool", "Synthetic: Billing Support Bot",
            "You handle billing questions. If you can't resolve something yourself, create a "
            "support ticket for a human to follow up — do not tell the customer you'll "
            "'transfer' them, since this channel has no live transfer.",
            tools=(ToolConfig(name="create_support_ticket", tool_type="webhook", detail="https://acme.example/tickets"),),
            kb_ids=("kb_billing",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, what's up with your billing?"),
                _turn("user", "I was charged twice for the same invoice and need a refund."),
                _turn("agent", None, tool_calls=(ToolCallRecord("create_support_ticket", False),)),
                _turn("agent", "I've opened ticket #8841 for our billing team — they'll refund the duplicate charge and follow up by email within 1 business day."),
            )),),
            arr_usd=25000.0,
            synthetic_note=(
                "FALSE-POSITIVE TRAP (required): cheap pass's escalation proxy only counts "
                "transfer_to_number/transfer_to_agent tool calls, so this agent shows a 0% "
                "escalation rate and gets flagged as never escalating — but it DOES escalate, "
                "correctly, via a ticket-creation tool the metric doesn't know to count."
            ),
        ),
        source="synthetic", should_flag=True, expected_classification="healthy",
        expected_failures={},
        notes="Required trap #2: zero-escalation-rate false alarm caused by a metric blind spot, not a real escalation-health gap.",
    ))

    # S4 — fallback: guesses first, escalates only after being caught.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_fallback_fail", "Synthetic: HR Policy Bot",
            "You answer employee HR policy questions. If you're not sure, say so and offer to "
            "connect them with HR — never guess.",
            tools=(ToolConfig(name="transfer_to_agent", tool_type="system", system_tool_type="transfer_to_agent"),),
            kb_ids=("kb_hr_policy",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, what HR question can I help with?"),
                _turn("user", "How many unpaid sabbatical days do I get after 5 years?"),
                _turn("agent", "After 5 years you get 10 unpaid sabbatical days per year.", kb_ids=()),
                _turn("user", "Are you sure? That's not in the handbook I have."),
                _turn("agent", "You're right, I'm not actually sure about that — let me connect you with HR to confirm.", tool_calls=(ToolCallRecord("transfer_to_agent", False),)),
            )),),
            arr_usd=30000.0,
            synthetic_note="Fallback failure: agent states a made-up policy confidently, and only admits uncertainty after the user pushes back — escalation came after the guess, not instead of it.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"fallback": "fallback_guesses_before_escalating"},
        notes="Sequencing is the failure: escalating eventually doesn't fix answering confidently first.",
    ))

    # S5 — multi_turn: re-asks a question already answered.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_multi_turn_fail", "Synthetic: Appointment Booking Bot",
            "You book service appointments. Collect the customer's name and preferred date, "
            "then confirm a slot.",
            tools=(), kb_ids=(),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi! What's your name and what date works for you?"),
                _turn("user", "I'm Maria Chen, and next Tuesday would be great."),
                _turn("agent", "Great, Maria — let me check Tuesday's availability."),
                _turn("agent", "Actually, could you tell me your name again?"),
                _turn("user", "...Maria Chen. I just said that."),
                _turn("agent", "Sorry about that! Tuesday at 2pm works — see you then, Maria."),
            )),),
            arr_usd=12000.0,
            synthetic_note="Multi-turn failure: re-asks for the name given one turn earlier, and the user has to call it out.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"multi_turn": "multi_turn_repeats_known_answer"},
        notes="",
    ))

    # S6 — escalation_health: runaway (escalates trivial requests too).
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_escalation_runaway", "Synthetic: Store Hours Bot",
            "You answer basic store questions: hours, location, and simple product availability.",
            tools=(ToolConfig(name="transfer_to_agent", tool_type="system", system_tool_type="transfer_to_agent"),),
            kb_ids=("kb_store_info",),
            conversations=tuple(
                ConversationRecord(f"c{i}", "widget", (
                    _turn("agent", "Hi! How can I help?"),
                    _turn("user", q),
                    _turn("agent", "Let me get you a team member for that.", tool_calls=(ToolCallRecord("transfer_to_agent", False),)),
                ))
                for i, q in enumerate([
                    "What time do you close today?", "Are you open on Sundays?",
                    "Do you have parking?", "What's your address?",
                ])
            ),
            arr_usd=8000.0,
            synthetic_note="Escalation-health failure: escalates every trivial question (store hours, parking) that it should just answer from the KB.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"escalation_health": "escalation_rate_runaway"},
        notes="",
    ))

    # S7 — sentiment: frustration genuinely caused by the agent.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_sentiment_agent_caused", "Synthetic: Password Reset Bot",
            "You help users reset their account password.",
            tools=(), kb_ids=("kb_account_help",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, I can help you reset your password. What's your account email?"),
                _turn("user", "It's maria@example.com"),
                _turn("agent", "Great, what's your account email?"),
                _turn("user", "I already told you, maria@example.com. This isn't working."),
                _turn("agent", "Sorry, what's your account email again?"),
                _turn("user", "I already said that twice! This is ridiculous."),
            )),),
            arr_usd=18000.0,
            synthetic_note="Sentiment failure: the agent itself causes the frustration by repeating the same question after already receiving the answer.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"sentiment": "sentiment_agent_caused_frustration"},
        notes="Root behavior is a multi_turn-shaped repeat, but the observable damage graded here is the sentiment criterion.",
    ))

    # S8 — bonus trap: user arrives already angry about something unrelated; not the agent's fault.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_sentiment_trap_not_agents_fault", "Synthetic: Shipping Status Bot",
            "You answer shipping status questions.",
            tools=(), kb_ids=("kb_shipping",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, how can I help?"),
                _turn("user", "I'm furious, you guys overcharged me $200 on my last invoice and nobody has fixed it."),
                _turn("agent", "I'm sorry to hear that — I handle shipping status here, but let me get you to billing so they can fix the overcharge.", tool_calls=(ToolCallRecord("create_support_ticket", False),)),
                _turn("user", "Fine, thanks."),
            )),),
            arr_usd=15000.0,
            synthetic_note="Bonus false-positive trap reinforcing 'sentiment != agent's fault': user arrives already furious about an unrelated billing issue; the agent handles its own scope well and routes the real issue out.",
        ),
        source="synthetic", should_flag=True, expected_classification="healthy",
        expected_failures={},
        notes="Negative-keyword proxy will trip on 'furious'/'overcharged' — the point is that a correct judge attributes the cause and doesn't fail the agent for it.",
    ))

    # S9 — latency: shared slow tool.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_latency_fail", "Synthetic: Inventory Lookup Bot",
            "You look up product inventory for customers calling in.",
            tools=(ToolConfig(name="inventory_lookup", tool_type="webhook", detail="https://acme.example/inventory"),),
            kb_ids=(),
            conversations=(ConversationRecord("c1", "twilio", (
                _turn("agent", "Thanks for calling Acme, what product are you looking for?"),
                _turn("user", "Do you have the model X42 in stock?"),
                _turn("agent", None, tool_calls=(ToolCallRecord("inventory_lookup", False),)),
                _turn("agent", "Yes, we have 12 in stock.", ttfb_ms=4200.0),
                _turn("user", "Great, what about the X43?"),
                _turn("agent", None, tool_calls=(ToolCallRecord("inventory_lookup", False),)),
                _turn("agent", "We have 3 of those.", ttfb_ms=3900.0),
            )),),
            arr_usd=22000.0,
            synthetic_note="Latency failure: two consecutive agent turns exceed the phone channel's latency band, both following the same inventory_lookup tool call.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"latency": "latency_shared_slow_tool"},
        notes="",
    ))

    # S10a — systemic, high ARR: novel cause with no catalog match.
    novel_convo = (ConversationRecord("c1", "widget", (
        _turn("agent", "Hi, welcome to Vantix Robotics support."),
        _turn("user", "How do I reset my Vantix device?"),
        _turn("agent", "Sure! First, unplug your Vann-tex device for 10 seconds."),
        _turn("user", "You keep saying our company name wrong, it's Van-tix, not Vann-tex — it's confusing."),
        _turn("agent", "Apologies! To reset your Vahn-teeks device, unplug it for 10 seconds."),
    )),)
    novel_note = (
        "Systemic case: the agent's TTS/pronunciation consistently mangles the company's own "
        "name differently every time, visibly confusing the user, who calls it out twice with "
        "no correction. No rubric criterion or catalog recipe cleanly covers 'can't say the "
        "customer's own product name' — deliberately chosen to have no good catalog match, so "
        "a judge naming a real cause here should still end up unmapped -> systemic."
    )
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_systemic_high_arr", "Synthetic: Vantix Support Bot (high ARR)",
            "You are Vantix Robotics' support agent for device resets.",
            tools=(), kb_ids=(), conversations=novel_convo, arr_usd=120000.0,
            synthetic_note=novel_note,
        ),
        source="synthetic", should_flag=True, expected_classification="systemic",
        expected_failures={},  # deliberately empty: no criterion maps cleanly, which is the point
        notes="High-ARR twin of the case below — same failure, routed to escalate_to_engineer instead of nearest_guidance.",
    ))

    # S10b — systemic, low ARR: identical failure, different ARR -> different route.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_systemic_low_arr", "Synthetic: Vantix Support Bot (low ARR)",
            "You are Vantix Robotics' support agent for device resets.",
            tools=(), kb_ids=(), conversations=novel_convo, arr_usd=4000.0,
            synthetic_note=novel_note,
        ),
        source="synthetic", should_flag=True, expected_classification="systemic",
        expected_failures={},
        notes="Low-ARR twin of the case above — tests that identical judge output still routes differently by ARR (nearest_guidance + logged recipe gap, not an engineer).",
    ))

    # S11 — knowledge_base: job needs one, none attached.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_kb_missing", "Synthetic: Returns Policy Bot",
            "You answer questions about our return policy using the knowledge base.",
            tools=(), kb_ids=(),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, what can I help with?"),
                _turn("user", "Can I return an opened item after 20 days?"),
                _turn("agent", "I don't have that information available right now."),
                _turn("user", "What about unopened items?"),
                _turn("agent", "Sorry, I don't have access to that either."),
            )),),
            arr_usd=10000.0,
            synthetic_note="Knowledge-base failure: the prompt promises KB-backed answers but none is attached, so the agent can't answer anything it's supposed to handle.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"knowledge_base": "kb_not_connected"},
        notes="",
    ))

    # S12 — system_prompt: self-contradictory.
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_system_prompt_contradiction", "Synthetic: Pricing Bot (contradictory prompt)",
            "You must NEVER discuss pricing under any circumstances — always redirect pricing "
            "questions to sales. If asked for a price, recite this script exactly: 'Our Pro "
            "plan is $49/month and Enterprise is custom-priced; would you like me to send you "
            "the full price sheet?'",
            tools=(), kb_ids=(),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, how can I help?"),
                _turn("user", "How much does the Pro plan cost?"),
                _turn("agent", "Our Pro plan is $49/month and Enterprise is custom-priced; would you like me to send you the full price sheet?"),
            )),),
            arr_usd=17000.0,
            synthetic_note="System-prompt failure: the prompt exists (not empty, not too short) but directly contradicts itself — 'never discuss pricing' followed by a mandatory pricing script.",
        ),
        source="synthetic", should_flag=True, expected_classification="standard",
        expected_failures={"system_prompt": "system_prompt_self_contradictory"},
        notes="A prompt that exists and is long enough still fails if it's self-contradictory — the cheap pass can't catch this (it only checks length); the judge must actually read it.",
    ))

    # S13 — fully healthy agent, for precision (cheap pass shouldn't need to flag this).
    cases.append(GoldenCase(
        snapshot=_snapshot(
            "synthetic_healthy_agent", "Synthetic: Clinic Scheduling Bot",
            "You schedule appointments for Riverside Dental Clinic. Collect the patient's name, "
            "reason for visit, and preferred date/time; confirm details before booking. If you "
            "don't know something, say so and offer to connect the caller with the front desk. "
            "You only handle scheduling — for billing or clinical questions, escalate.",
            tools=(ToolConfig(name="transfer_to_agent", tool_type="system", system_tool_type="transfer_to_agent"),),
            kb_ids=("kb_clinic_hours",),
            conversations=(ConversationRecord("c1", "widget", (
                _turn("agent", "Hi, thanks for calling Riverside Dental — how can I help?"),
                _turn("user", "I'd like to book a cleaning for next Thursday, I'm Alex Kim."),
                _turn("agent", "Got it, Alex — a cleaning next Thursday. What time works best?"),
                _turn("user", "Morning if possible."),
                _turn("agent", "I have 9:30am next Thursday for a cleaning under Alex Kim — should I confirm that?"),
                _turn("user", "Yes, and one more thing, how much does a cleaning cost?"),
                _turn("agent", "That's a billing question, so let me connect you with our front desk for exact pricing.", tool_calls=(ToolCallRecord("transfer_to_agent", False),)),
            )),),
            arr_usd=35000.0,
            synthetic_note="Fully healthy control case: bounded prompt, working handoff, no repeats, correctly escalates the one out-of-scope question, no ungrounded claims.",
        ),
        source="synthetic", should_flag=False, expected_classification="healthy",
        expected_failures={},
        notes="Precision check: the cheap pass ideally does NOT flag this one, since nothing here is actually wrong.",
    ))

    return cases


def load_golden_set() -> list[GoldenCase]:
    return _load_real_cases() + _synthetic_cases()
