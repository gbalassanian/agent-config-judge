"""The nine-criteria rubric and the recipe catalog.

This module is pure data (plus a few lookup helpers). It has no dependency on
the ElevenLabs client, the cheap pass, or the judge — everything downstream
imports *from* here, nothing feeds back into it. That's deliberate: the
rubric is the one artifact a human should be able to read and sign off on
without reading any code.

Two things live here, and they answer two different questions:

  - CRITERIA answers "what does a healthy agent look like?" — nine axes,
    three checkable from config alone, six that only show up in transcripts.

  - RECIPE_CATALOG answers "which failures do we already know how to fix?"
    This catalog IS the standard/systemic boundary. A failure cause with an
    entry here is "standard" — automatable, either as a self-serve doc link
    or an account-specific nudge. A failure cause with no entry is
    "systemic" by construction, not by judgment call. See judge.py for how
    that boundary gets enforced instead of merely asserted.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Category(str, Enum):
    """Where the signal for this criterion comes from."""

    CONFIG = "config"  # checkable by reading agent config fields directly
    BEHAVIOR = "behavior"  # only observable in transcripts; config can lie


class RecipeTier(str, Enum):
    """How a known failure cause gets fixed, per the router's contract.

    self_serve: the same fix, worded the same way, applies across the whole
    portfolio — a docs link is enough. Batchable.

    nudge: the fix is known, but the message has to name *this account's*
    specific config (which tool, which channel, which KB doc) to be
    actionable. Not batchable as a form letter, but still fully automatable
    to *draft* — a human just approves the send if it touches a live agent.
    """

    SELF_SERVE = "self_serve"
    NUDGE = "nudge"


@dataclass(frozen=True)
class Criterion:
    id: str
    name: str
    category: Category
    description: str
    # Short reminder of the failure mode that motivated including it —
    # keeps the rubric legible on its own, without needing this docstring.
    fail_looks_like: str


# Order matches the case study write-up: three config-checkable criteria
# first, then the six that need a transcript. The judge always evaluates in
# this order so its output is diffable across runs.
CRITERIA: dict[str, Criterion] = {
    "system_prompt": Criterion(
        id="system_prompt",
        name="System prompt",
        category=Category.CONFIG,
        description=(
            "A clear, bounded role with explicit limits on what the agent "
            "does and doesn't handle."
        ),
        fail_looks_like=(
            "Prompt exists but contradicts itself (e.g. told to both "
            "'never discuss pricing' and given a pricing script) — that's "
            "a FAIL, not a pass just because a prompt is present."
        ),
    ),
    "knowledge_base": Criterion(
        id="knowledge_base",
        name="Knowledge base",
        category=Category.CONFIG,
        description="At least one KB source connected, if the job requires one.",
        fail_looks_like=(
            "Agent's prompt promises answers from docs/policies but no KB "
            "is attached, or the attached KB is empty/unused."
        ),
    ),
    "human_handoff": Criterion(
        id="human_handoff",
        name="Human handoff",
        category=Category.CONFIG,
        description=(
            "A transfer path that FUNCTIONS on the channel(s) this agent "
            "actually runs on — not just a tool that's present in config."
        ),
        fail_looks_like=(
            "transfer_to_number configured and passing every config check, "
            "but the agent runs on react_sdk (web) where phone transfer "
            "cannot work — the config is healthy and the escape hatch is "
            "still dead. This is the case that motivated the judge tier: "
            "only a transcript shows the tool call failing at runtime."
        ),
    ),
    "fallback": Criterion(
        id="fallback",
        name="Fallback",
        category=Category.BEHAVIOR,
        description="Says it doesn't know, then escalates — in that order.",
        fail_looks_like=(
            "Agent guesses first and only escalates after the guess falls "
            "apart. Still a fail even if the eventual escalation rate "
            "looks fine in aggregate — the sequencing is the failure."
        ),
    ),
    "grounding": Criterion(
        id="grounding",
        name="Grounding",
        category=Category.BEHAVIOR,
        description=(
            "Specific claims — numbers, limits, formats, policies, prices — "
            "are attributable to the KB. Conversational filler needs no source."
        ),
        fail_looks_like=(
            "A specific factual claim lands in a turn with no KB document "
            "used. NOTE: this alone is a proxy, not a verdict — see the "
            "user-supplied-number false-positive trap in the golden set."
        ),
    ),
    "multi_turn": Criterion(
        id="multi_turn",
        name="Multi-turn coherence",
        category=Category.BEHAVIOR,
        description=(
            "Holds context across turns: doesn't re-ask a question it "
            "already has the answer to, doesn't loop on edge cases."
        ),
        fail_looks_like="Re-asks for info the user already gave two turns ago.",
    ),
    "escalation_health": Criterion(
        id="escalation_health",
        name="Escalation health",
        category=Category.BEHAVIOR,
        description="Neither zero nor runaway — escalates what it should, resolves what it should.",
        fail_looks_like=(
            "Escalation rate near zero AND agent is taking calls that "
            "clearly needed a human. NOTE: a zero rate from the metric "
            "isn't proof by itself — see the ticket-tool false-positive "
            "trap in the golden set, where escalation happens through a "
            "tool the aggregate metric doesn't count."
        ),
    ),
    "sentiment": Criterion(
        id="sentiment",
        name="Sentiment",
        category=Category.BEHAVIOR,
        description="No frustration CAUSED BY the agent.",
        fail_looks_like=(
            "User arrives already angry about an unrelated billing issue "
            "and stays angry — that's not this agent's failure. Fail is "
            "reserved for frustration the agent's own behavior produced "
            "(repetition, wrong answers, dead-end loops)."
        ),
    ),
    "latency": Criterion(
        id="latency",
        name="Latency",
        category=Category.BEHAVIOR,
        description="Within band for the channel; slow turns get a shared-cause note when one exists.",
        fail_looks_like="Turns blow the channel's latency band, especially with a shared cause (e.g. one slow tool called repeatedly).",
    ),
}

CRITERION_ORDER: list[str] = list(CRITERIA.keys())

CONFIG_CHECKABLE_IDS = frozenset(
    cid for cid, c in CRITERIA.items() if c.category == Category.CONFIG
)
BEHAVIOR_ONLY_IDS = frozenset(
    cid for cid, c in CRITERIA.items() if c.category == Category.BEHAVIOR
)


@dataclass(frozen=True)
class Recipe:
    """A known fix for a named failure cause.

    cause_code is the join key between the judge's output and this catalog.
    The judge names a cause; the validator (in judge.py) looks it up here —
    it never trusts the judge's own tier/classification for that cause.
    """

    cause_code: str
    criterion_id: str
    title: str
    fix: str
    tier: RecipeTier
    # Required for self_serve (batchable via a docs link), optional for
    # nudge (the nudge message itself carries the specifics).
    doc_url: str | None = None

    def __post_init__(self) -> None:
        if self.criterion_id not in CRITERIA:
            raise ValueError(f"Recipe {self.cause_code!r} references unknown criterion {self.criterion_id!r}")
        if self.tier == RecipeTier.SELF_SERVE and not self.doc_url:
            raise ValueError(f"Recipe {self.cause_code!r} is self_serve but has no doc_url")


# The standard/systemic frontier. Adding a row here is how the system gets
# cheaper over time: every cause_code added is one more failure mode that
# routes itself instead of landing on an engineer's desk. Keep entries
# narrow and literal (one cause per row) rather than vague buckets — a
# vague bucket invites the judge to stretch a cause to fit it, which is
# exactly the silent systemic-to-standard leak the validator exists to stop.
_RECIPES: list[Recipe] = [
    Recipe(
        cause_code="system_prompt_missing",
        criterion_id="system_prompt",
        title="No system prompt set",
        fix="Add a role-scoped system prompt using the workspace default template.",
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/system-prompt",
    ),
    Recipe(
        cause_code="system_prompt_self_contradictory",
        criterion_id="system_prompt",
        title="Prompt contradicts itself",
        fix=(
            "Quote the two conflicting instructions to the account and ask "
            "which one is intended; this needs a human decision about "
            "product behavior, not a template swap."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="system_prompt_too_generic",
        criterion_id="system_prompt",
        # Found live: a real agent's entire prompt was "Eres un asistente
        # útil" ("You are a helpful assistant") — not empty (so
        # system_prompt_missing doesn't fit), not self-contradictory, just
        # no bounded role, audience, or scope at all. A third, distinct
        # failure shape the catalog didn't have a cause for until this one
        # was found while producing this repo's own recorded judgements.
        title="Prompt exists but defines no bounded role",
        fix=(
            "Replace the generic 'you are a helpful assistant' line with a "
            "specific role, audience, and explicit scope boundary (what "
            "this agent does and does NOT handle) using the workspace "
            "default template."
        ),
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/system-prompt",
    ),
    Recipe(
        cause_code="kb_not_connected",
        criterion_id="knowledge_base",
        title="No knowledge base connected",
        fix="Attach at least one KB source (URL, file, or text) in agent settings.",
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/knowledge-base",
    ),
    Recipe(
        cause_code="kb_connected_but_unused",
        criterion_id="knowledge_base",
        title="KB attached but never retrieved",
        fix=(
            "Point out which document is attached and unused, and confirm "
            "whether RAG/retrieval is actually enabled for it — connecting "
            "a doc doesn't enable retrieval by itself."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="handoff_tool_unsupported_on_channel",
        criterion_id="human_handoff",
        title="Transfer tool doesn't work on the channel this agent runs on",
        fix=(
            "Name the specific mismatch (e.g. transfer_to_number requires "
            "Twilio/Exotel/SIP but this agent runs on react_sdk) and either "
            "swap in transfer_to_agent or move the deployment to a "
            "telephony channel — this is the case that motivated the judge "
            "tier: config validation alone approves this agent."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="handoff_no_transfer_tool",
        criterion_id="human_handoff",
        title="No transfer/handoff tool configured at all",
        fix="Add transfer_to_number or transfer_to_agent with at least one working destination.",
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/tools/system-tools/transfer-to-human",
    ),
    Recipe(
        cause_code="fallback_guesses_before_escalating",
        criterion_id="fallback",
        title="Agent guesses first, escalates only after the guess fails",
        fix=(
            "Add an explicit prompt instruction: on low-confidence "
            "questions, say 'I'm not sure' and escalate immediately rather "
            "than answering speculatively first."
        ),
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/system-prompt",
    ),
    Recipe(
        cause_code="grounding_missing_source_attribution",
        criterion_id="grounding",
        title="Specific claims made with no KB document used in that turn",
        fix=(
            "Enable source_attribution in conversation config so the model "
            "is instructed to only state specifics it can attribute, and "
            "confirm the claimed facts actually live in the KB."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="multi_turn_repeats_known_answer",
        criterion_id="multi_turn",
        title="Re-asks for information already given",
        fix=(
            "Usually a dynamic-variable / context-window issue for this "
            "account's flow; name the exact repeated question and turn "
            "number so engineering can trace why it wasn't carried forward."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="multi_turn_repeats_failed_tool_call",
        criterion_id="multi_turn",
        # Found live: an agent hit a webhook 401, retried the identical
        # query verbatim, got the identical 401, then gave up — a loop on
        # an edge case per the rubric's own definition, distinct from
        # re-asking the user a question. Generalizable enough to be
        # self-serve rather than account-specific.
        title="Retries an identical tool call after it already failed",
        fix=(
            "Add an explicit prompt instruction: on a tool error, either "
            "change something about the retry (different query, different "
            "tool) or tell the user plainly that something failed — never "
            "repeat the exact same call and expect a different result."
        ),
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/tools",
    ),
    Recipe(
        cause_code="escalation_rate_zero_with_missed_cases",
        criterion_id="escalation_health",
        title="Never escalates despite clear-cut cases that needed a human",
        fix=(
            "Add explicit escalation triggers to the prompt for the "
            "specific request types observed failing to escalate."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="escalation_rate_runaway",
        criterion_id="escalation_health",
        title="Escalates almost everything, including easy requests",
        fix="Raise the bar in the prompt for what counts as escalation-worthy; add 1-2 worked examples the model can pattern-match against.",
        tier=RecipeTier.SELF_SERVE,
        doc_url="https://elevenlabs.io/docs/conversational-ai/customization/system-prompt",
    ),
    Recipe(
        cause_code="sentiment_agent_caused_frustration",
        criterion_id="sentiment",
        title="Agent behavior (repetition, wrong answers, dead ends) causes visible frustration",
        fix=(
            "Cite the specific turn(s) that triggered the frustration and "
            "route to whichever other recipe explains the root behavior "
            "(usually multi_turn or grounding) — sentiment is a symptom, "
            "the fix lives on the causing criterion."
        ),
        tier=RecipeTier.NUDGE,
    ),
    Recipe(
        cause_code="latency_shared_slow_tool",
        criterion_id="latency",
        title="Turns blow the channel's latency band with one shared cause",
        fix="Name the specific slow tool/step and its p50 latency; ask the account to raise its timeout or cache the call.",
        tier=RecipeTier.NUDGE,
    ),
]

RECIPE_CATALOG: dict[str, Recipe] = {r.cause_code: r for r in _RECIPES}


def get_recipe(cause_code: str) -> Recipe | None:
    """Look up a recipe by cause code. Returns None for anything not catalogued —
    callers (the judge's validator) treat that None as "this is systemic"."""
    return RECIPE_CATALOG.get(cause_code)


def is_known_cause(cause_code: str) -> bool:
    return cause_code in RECIPE_CATALOG


def recipes_for_criterion(criterion_id: str) -> list[Recipe]:
    return [r for r in _RECIPES if r.criterion_id == criterion_id]
