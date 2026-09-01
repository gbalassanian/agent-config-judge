"""Tier 2: the LLM judge.

Reads the agent's config AND the transcripts the cheap pass flagged, scores
all nine criteria with cited evidence, names a cause for every failure, and
maps that cause onto rubric.RECIPE_CATALOG. This module is built around
three non-negotiable structural rules (see the case study write-up):

  1. Evidence is enforced, not requested. Every pass/fail verdict needs a
     short verbatim quote or a named config field. One that arrives without
     either gets downgraded to "unknown" by the validator — a judge that
     can't show its work doesn't get to route work.

  2. The judge does not own classification. It names a cause; this module
     looks the cause up in rubric.RECIPE_CATALOG and recomputes
     standard/systemic from that mapping every time, ignoring whatever the
     raw output claims about its own classification. A cause_code that
     doesn't exist in the catalog is deleted, and the agent it belongs to
     falls to "systemic" — that's the one failure mode that would silently
     turn a systemic agent into an automated nudge, so it's a hard rule,
     not a warning.

  3. Quantity is not severity. Three failures that all map to known
     recipes are still "standard". One failure with no mapping makes the
     whole agent "systemic", no matter how healthy the other eight
     criteria look.

Two backends implement JudgeBackend: LiveJudgeBackend (real Anthropic API
calls) and RecordedJudgeBackend (replays saved raw outputs from a fixture
file, keyed by agent_id). Both return an unvalidated dict shaped like
JUDGE_OUTPUT CONTRACT below; validate_judge_output() is the only code path
that turns that into a trusted Judgement, and it runs identically over
both backends' output — that's what makes "recorded" a reproducibility
tool rather than a shortcut: changing the rubric or the recipe catalog and
re-running against the SAME saved raw outputs will re-derive a
(potentially different) classification.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from agent_config_judge.models import AgentConfigSnapshot, ConversationRecord
from agent_config_judge.rubric import CRITERION_ORDER, CRITERIA, RECIPE_CATALOG, Recipe, get_recipe

# Hard cap on a quote's length, not a calibration knob: this exists to keep
# "evidence" meaning a pointer into the transcript, not a paste of the
# whole thing. A quote over the cap is truncated (with a validator note),
# never rejected outright — the judge still gets credit for citing
# *something* real.
MAX_QUOTE_CHARS = 240

VALID_VERDICTS = frozenset({"pass", "fail", "unknown"})

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_config(config: AgentConfigSnapshot) -> str:
    lines = [
        f"agent_id: {config.agent_id}",
        f"name: {config.name}",
        f"system_prompt: {config.system_prompt!r}",
        f"knowledge_base_ids: {list(config.knowledge_base_ids)}",
        f"knowledge_base_usage_modes: {[d.usage_mode for d in config.knowledge_base_docs]}",
        f"rag_enabled: {config.rag_enabled}",
        "tools:",
    ]
    if not config.tools:
        lines.append("  (none configured)")
    for t in config.tools:
        kind = t.system_tool_type or t.tool_type
        lines.append(f"  - name={t.name!r} type={t.tool_type} system_tool_type={t.system_tool_type} detail={t.detail!r}")
    return "\n".join(lines)


def _format_conversations(conversations: tuple[ConversationRecord, ...]) -> str:
    if not conversations:
        return "(no conversations in the sample)"
    blocks = []
    for conv in conversations:
        turn_lines = []
        for i, turn in enumerate(conv.turns):
            tool_bits = ""
            if turn.tool_calls:
                tool_bits = " tool_calls=" + ", ".join(
                    f"{tc.tool_name}{'[ERROR]' if tc.is_error else ''}" for tc in turn.tool_calls
                )
            kb_bits = f" used_kb_docs={list(turn.used_static_kb_document_ids)}" if turn.role == "agent" else ""
            ttfb_bits = f" ttfb_ms={turn.ttfb_ms}" if turn.ttfb_ms is not None else ""
            turn_lines.append(f"  [{i}] {turn.role}: {turn.text}{tool_bits}{kb_bits}{ttfb_bits}")
        blocks.append(
            f"conversation {conv.conversation_id} (channel={conv.channel}):\n" + "\n".join(turn_lines)
        )
    return "\n\n".join(blocks)


def _format_catalog() -> str:
    lines = []
    for cause_code, recipe in RECIPE_CATALOG.items():
        lines.append(f"  - {cause_code} (criterion={recipe.criterion_id}): {recipe.title}")
    return "\n".join(lines)


def _format_criteria() -> str:
    lines = []
    for cid in CRITERION_ORDER:
        c = CRITERIA[cid]
        lines.append(f"  - {cid} [{c.category.value}]: {c.description} Fail looks like: {c.fail_looks_like}")
    return "\n".join(lines)


JUDGE_OUTPUT_CONTRACT = """Return ONLY a single JSON object, no prose before or after, shaped exactly like this:

{
  "agent_id": "<the agent_id given above>",
  "criteria": {
    "system_prompt": {"verdict": "pass|fail|unknown", "evidence_quote": "<=240 char verbatim quote from a transcript turn, or null", "evidence_config_field": "<named config field, e.g. tools[0].system_tool_type>, or null", "cause_code": "<catalog cause_code if verdict is fail, else null>"},
    "knowledge_base": { ... same shape ... },
    "human_handoff": { ... same shape ... },
    "fallback": { ... same shape ... },
    "grounding": { ... same shape ... },
    "multi_turn": { ... same shape ... },
    "escalation_health": { ... same shape ... },
    "sentiment": { ... same shape ... },
    "latency": { ... same shape ... }
  },
  "notes": "<free-text summary, optional>"
}

Rules that will be mechanically enforced whether or not you follow them:
- Every criterion key above must be present.
- A "pass" or "fail" verdict with both evidence_quote and evidence_config_field null gets
  downgraded to "unknown" — it will not be trusted or routed.
- evidence_quote must be a short VERBATIM excerpt actually present in a transcript turn above,
  not a paraphrase or summary. Keep it under 240 characters.
- cause_code, when given, should be one of the catalog codes listed above IF one truly matches.
  If no catalog cause fits, you may still fail the criterion, but leave cause_code null or use
  a new short snake_case code that describes the cause precisely — do NOT force-fit a listed
  cause_code that doesn't actually match just to make the agent look "standard". A cause_code
  that isn't in the catalog will make this agent route as systemic (needs a human), which is
  the correct outcome for a genuinely novel failure — misusing a catalog code to dodge that
  is the one mistake that silently mis-routes a systemic agent as automatable.
- You do not decide standard vs. systemic classification. That is computed downstream from
  whether every cause_code you name is in the catalog. Do not include a "classification" field;
  it will be ignored if you do.
"""


def build_judge_prompt(
    config: AgentConfigSnapshot,
    conversations: tuple[ConversationRecord, ...],
) -> str:
    return f"""You are auditing one ElevenLabs conversational voice agent against a nine-criterion \
rubric. You have the agent's config and a sample of its real conversation transcripts. Config \
checks alone are not sufficient — a tool can be fully configured and still fail at runtime in a \
way that is only visible in a transcript (e.g. a transfer_to_number tool that is configured \
correctly but the agent runs on a channel where phone transfer cannot work at all). Read the \
transcripts; do not assume config health implies behavioral health.

RUBRIC (nine criteria, in the order you must answer them):
{_format_criteria()}

KNOWN FAILURE-CAUSE CATALOG (use a code from here when a failure genuinely matches; a failure with
no genuine match should get a novel cause_code or null, not a forced fit):
{_format_catalog()}

AGENT CONFIG:
{_format_config(config)}

CONVERSATION SAMPLE:
{_format_conversations(conversations)}

{JUDGE_OUTPUT_CONTRACT}"""


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class JudgeBackend(Protocol):
    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        """Return a parsed (but not yet validated) dict matching JUDGE_OUTPUT_CONTRACT."""
        ...


class JudgeError(RuntimeError):
    pass


def _extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first top-level JSON object out of a model's free-text response.

    Models asked for "only JSON" sometimes still wrap it in a code fence or a
    sentence. This is deliberately tolerant of that but not of malformed
    JSON itself — a genuinely broken payload raises JudgeError, which the
    caller treats as every criterion being unknown (see run_judge).
    """
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise JudgeError(f"Judge output was not valid JSON: {e}") from e


@dataclass
class LiveJudgeBackend:
    """Calls the real Anthropic API. Requires ANTHROPIC_API_KEY.

    Kept as a thin wrapper: prompt construction and output validation are
    shared with the recorded backend, so the only thing this class owns is
    the actual network call and pulling JSON out of the model's free text.

    Retries: the Anthropic SDK already retries connection errors and
    429/5xx responses with backoff on its own (default max_retries=2) — we
    don't reimplement that here, we just raise the ceiling. A one-off
    interactive call should fail fast after a couple of tries; a portfolio
    scan running unattended over thousands of agents overnight can afford
    to wait longer rather than lose one agent's judgement to a blip that
    would have cleared on the fourth or fifth attempt.
    """

    model: str = "claude-sonnet-5"
    # 4096 was too low: current models (Sonnet 5 included) default to
    # adaptive thinking at "high" effort, and thinking tokens count against
    # this same ceiling as the final JSON answer — on this prompt's length
    # (full transcripts + a 9-criterion rubric to reason through) the model
    # can spend the whole budget thinking and stop before writing any
    # answer at all (see the empty-text check in judge() below, added after
    # exactly that happened on a live call). 16000 leaves generous room for
    # both on prompts this size; revisit if a much longer conversation
    # sample ever gets judged in one call.
    max_tokens: int = 16000
    max_retries: int = 5
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import anthropic
        except ImportError as e:
            raise JudgeError(
                "The 'anthropic' package is required for the live judge backend. "
                "Install it (see requirements.txt) or use --backend recorded."
            ) from e
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise JudgeError(
                "ANTHROPIC_API_KEY is not set. The live judge backend needs it; "
                "use --backend recorded to run against saved judgements instead."
            )
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=self.max_retries)
        return self._client

    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        client = self._get_client()
        prompt = build_judge_prompt(config, conversations)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if getattr(block, "type", None) == "text")
        if not text.strip():
            # Current models (Sonnet 5 included) default to adaptive thinking
            # at "high" effort, and thinking tokens count against the same
            # max_tokens ceiling as the final answer. On a long enough
            # prompt the model can spend the entire budget thinking and stop
            # (stop_reason="max_tokens") before emitting any text block at
            # all — this is not a malformed-JSON case _extract_json_object
            # is meant to catch, so surface it distinctly with the actual
            # stop reason and block types instead of a bare JSONDecodeError
            # that gives no clue what happened.
            block_types = [getattr(b, "type", "?") for b in response.content]
            raise JudgeError(
                f"Judge call returned no text content (stop_reason={response.stop_reason!r}, "
                f"content block types={block_types}). Likely cause: the model spent the entire "
                f"max_tokens budget ({self.max_tokens}) on thinking before writing an answer. "
                f"Raise LiveJudgeBackend.max_tokens (or pass output_config={{'effort': 'medium'}} "
                f"or lower to reduce thinking spend) and retry."
            )
        return _extract_json_object(text)


@dataclass
class RecordedJudgeBackend:
    """Replays saved raw judge outputs from a JSON fixture, keyed by agent_id.

    This is not a mock for convenience — it's how the eval stays
    reproducible when the rubric, the recipe catalog, or the router change:
    the SAME raw outputs get re-validated and re-routed, so you can see
    exactly what a rubric change would have done to real judge output
    without spending another API call.
    """

    fixture_path: str
    _data: dict[str, Any] | None = field(default=None, repr=False)

    def _load(self) -> dict[str, Any]:
        if self._data is None:
            with open(self.fixture_path, encoding="utf-8") as f:
                self._data = json.load(f)
        return self._data

    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        data = self._load()
        if config.agent_id not in data:
            raise JudgeError(
                f"No recorded judgement for agent_id={config.agent_id!r} in {self.fixture_path!r}. "
                "Run with --backend live (needs ANTHROPIC_API_KEY) to generate one, or add it to the fixture."
            )
        return data[config.agent_id]


# ---------------------------------------------------------------------------
# Validation: evidence enforcement + recipe-mapping-owns-classification
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionVerdict:
    criterion_id: str
    verdict: str  # "pass" | "fail" | "unknown" — post-validation
    raw_verdict: str  # what the judge originally said, before any downgrade
    evidence_quote: str | None
    evidence_config_field: str | None
    cause_code: str | None  # the judge's claimed cause (fail only)
    recipe: Recipe | None  # resolved from rubric.RECIPE_CATALOG; None means unmapped

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence_quote) or bool(self.evidence_config_field)

    @property
    def is_unmapped_failure(self) -> bool:
        return self.verdict == "fail" and self.recipe is None


@dataclass(frozen=True)
class Judgement:
    agent_id: str
    criteria: dict[str, CriterionVerdict]
    classification: str  # "healthy" | "standard" | "systemic" — computed, never trusted from the model
    notes: str
    validator_notes: tuple[str, ...]  # what the validator changed and why, for auditability

    @property
    def failures(self) -> list[CriterionVerdict]:
        return [c for c in self.criteria.values() if c.verdict == "fail"]

    @property
    def unmapped_failures(self) -> list[CriterionVerdict]:
        return [c for c in self.criteria.values() if c.is_unmapped_failure]


# Curly/smart quote variants -> straight ASCII, so a citation that survives
# a copy/paste through something that "prettifies" quotes still matches.
_QUOTE_TRANSLATION = str.maketrans({
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
})

# Irregular contractions: the stem itself changes, not just an expansion, so
# these need an exact whole-word swap before the generic suffix rules below
# run — a plain "n't" -> " not" rule would turn "can't" into "ca not", not
# "cannot".
_IRREGULAR_CONTRACTIONS = {
    "can't": "cannot",
    "won't": "will not",
    "shan't": "shall not",
}

# Regular clitic suffixes: the stem is unchanged, just expanded. The target
# picked for each doesn't have to be the "correct" expansion in context
# ("'d" collapses both "had" and "would") — it only has to be applied
# consistently, since both the judge's citation and the real transcript text
# pass through this same function before being compared. Deliberately
# excludes "'s": it's genuinely ambiguous between "is"/"has" and a
# possessive ("Alex's"), and guessing wrong would change what the text
# means instead of just how it's punctuated — exactly the risk this
# function exists to avoid.
_CLITIC_SUFFIXES = (
    ("n't", " not"), ("'re", " are"), ("'ve", " have"),
    ("'ll", " will"), ("'d", " would"), ("'m", " am"),
)


def _normalize_for_match(s: str) -> str:
    """Collapse cosmetic differences a real quote can pick up when the judge
    reproduces it from context, without tolerating anything that changes
    what it actually says.

    Three narrow, hand-picked passes — deliberately not a general similarity
    score: a hand-picked list of "harmless" differences can't quietly grow
    into "close enough" the way a fuzzy-match threshold could, which matters
    here because the failure mode this guards against (a citation that
    changed the actual claim — a different number, a different fact) is
    worse than the one it accepts (rejecting a same-fact citation over
    formatting and downgrading it to "unknown," which never crashes
    anything, just under-reports).

      1. Curly quotes -> straight quotes (typographic only).
      2. Contractions -> one canonical expanded form, so "can't" and
         "cannot" — or a transcript's "don't" reproduced as "do not" —
         compare equal.
      3. Sentence punctuation (,.;:!?) stripped, *except* between two
         digits, where it's part of the number itself ("45.00", "1,000")
         rather than punctuation — stripping it there would let two
         genuinely different numbers collide into the same normalized text,
         which is exactly the kind of false match this function must not
         introduce.
    """
    s = s.translate(_QUOTE_TRANSLATION)
    s = re.sub(r"\s+", " ", s).strip().lower()
    for word, expanded in _IRREGULAR_CONTRACTIONS.items():
        s = re.sub(r"\b" + re.escape(word) + r"\b", expanded, s)
    for suffix, expanded in _CLITIC_SUFFIXES:
        s = re.sub(re.escape(suffix) + r"\b", expanded, s)
    s = re.sub(r"(?<!\d)[,.;:!?]|[,.;:!?](?!\d)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _validate_one_criterion(
    criterion_id: str, raw: Any, validator_notes: list[str], transcript_text: str | None
) -> CriterionVerdict:
    if not isinstance(raw, dict):
        validator_notes.append(f"{criterion_id}: judge output missing or malformed; scored unknown.")
        return CriterionVerdict(criterion_id, "unknown", "unknown", None, None, None, None)

    raw_verdict = str(raw.get("verdict", "unknown")).lower()
    if raw_verdict not in VALID_VERDICTS:
        validator_notes.append(f"{criterion_id}: judge gave unrecognized verdict {raw_verdict!r}; scored unknown.")
        raw_verdict = "unknown"

    quote = raw.get("evidence_quote")
    quote = quote.strip() if isinstance(quote, str) and quote.strip() else None
    if quote and len(quote) > MAX_QUOTE_CHARS:
        validator_notes.append(
            f"{criterion_id}: evidence_quote truncated from {len(quote)} to {MAX_QUOTE_CHARS} chars."
        )
        quote = quote[:MAX_QUOTE_CHARS]

    # Evidence is enforced, not just requested: a quote has to actually
    # appear in the transcript sample the judge was given, not merely be
    # present as a string. This is what stops a judge from citing a
    # plausible-sounding but fabricated line. transcript_text is None only
    # when the caller has no transcript to check against (shouldn't happen
    # via run_judge, which always passes it); in that case the check is
    # skipped rather than failing everything.
    if quote and transcript_text is not None and _normalize_for_match(quote) not in transcript_text:
        validator_notes.append(
            f"{criterion_id}: evidence_quote {quote!r} does not appear verbatim in the transcript "
            "sample provided; discarded as fabricated evidence."
        )
        quote = None

    config_field = raw.get("evidence_config_field")
    config_field = config_field.strip() if isinstance(config_field, str) and config_field.strip() else None

    verdict = raw_verdict
    if verdict in ("pass", "fail") and not quote and not config_field:
        validator_notes.append(
            f"{criterion_id}: verdict {verdict!r} arrived with no evidence quote or config field "
            "that survived verification; downgraded to unknown — an unevidenced verdict is not trusted."
        )
        verdict = "unknown"

    cause_code = raw.get("cause_code")
    cause_code = cause_code.strip() if isinstance(cause_code, str) and cause_code.strip() else None

    recipe: Recipe | None = None
    if verdict == "fail":
        if cause_code is None:
            validator_notes.append(f"{criterion_id}: fail with no cause_code named; treated as unmapped (systemic).")
        else:
            recipe = get_recipe(cause_code)
            if recipe is None:
                validator_notes.append(
                    f"{criterion_id}: cause_code {cause_code!r} is not in the recipe catalog; "
                    "deleted and treated as unmapped (systemic) rather than trusting the judge's own framing."
                )
            elif recipe.criterion_id != criterion_id:
                validator_notes.append(
                    f"{criterion_id}: cause_code {cause_code!r} belongs to criterion "
                    f"{recipe.criterion_id!r}, not {criterion_id!r}; rejected as unmapped (systemic)."
                )
                recipe = None
    elif cause_code is not None:
        # cause_code is only meaningful on a fail; ignore it elsewhere rather
        # than erroring, but note it since it suggests the judge is confused.
        validator_notes.append(f"{criterion_id}: cause_code given on a {verdict!r} verdict; ignored.")
        cause_code = None

    return CriterionVerdict(
        criterion_id=criterion_id,
        verdict=verdict,
        raw_verdict=raw_verdict,
        evidence_quote=quote,
        evidence_config_field=config_field,
        cause_code=cause_code,
        recipe=recipe,
    )


def validate_judge_output(
    raw: dict[str, Any],
    agent_id: str,
    conversations: tuple[ConversationRecord, ...] | None = None,
) -> Judgement:
    """The one function that turns an untrusted judge dict into a routable Judgement.

    Recomputes classification from the recipe mapping every time — never
    reads a "classification" field even if the raw output includes one.

    Passing `conversations` enables the verbatim-quote check (see
    _validate_one_criterion): every evidence_quote must actually occur in
    that transcript sample or it's discarded. Omit it only for validating
    hand-constructed test payloads that have no real transcript behind
    them; run_judge always supplies it.
    """
    validator_notes: list[str] = []
    raw_criteria = raw.get("criteria", {}) if isinstance(raw, dict) else {}
    if not isinstance(raw_criteria, dict):
        raw_criteria = {}

    transcript_text: str | None = None
    if conversations is not None:
        transcript_text = _normalize_for_match(
            " ".join(turn.text for conv in conversations for turn in conv.turns if turn.text)
        )

    criteria: dict[str, CriterionVerdict] = {}
    for cid in CRITERION_ORDER:
        criteria[cid] = _validate_one_criterion(cid, raw_criteria.get(cid), validator_notes, transcript_text)

    failures = [c for c in criteria.values() if c.verdict == "fail"]
    if not failures:
        classification = "healthy"
    elif any(c.recipe is None for c in failures):
        classification = "systemic"
    else:
        classification = "standard"

    notes = raw.get("notes", "") if isinstance(raw, dict) else ""
    if not isinstance(notes, str):
        notes = ""

    reported_agent_id = raw.get("agent_id") if isinstance(raw, dict) else None
    if reported_agent_id and reported_agent_id != agent_id:
        validator_notes.append(
            f"judge's agent_id {reported_agent_id!r} does not match expected {agent_id!r}; ignored, using expected."
        )

    return Judgement(
        agent_id=agent_id,
        criteria=criteria,
        classification=classification,
        notes=notes,
        validator_notes=tuple(validator_notes),
    )


def run_judge(
    backend: JudgeBackend,
    config: AgentConfigSnapshot,
    conversations: tuple[ConversationRecord, ...],
) -> Judgement:
    try:
        raw = backend.judge(config, conversations)
    except JudgeError:
        raise
    if not isinstance(raw, dict):
        raise JudgeError(f"Judge backend for {config.agent_id!r} returned non-dict output: {type(raw)!r}")
    return validate_judge_output(raw, agent_id=config.agent_id, conversations=conversations)
