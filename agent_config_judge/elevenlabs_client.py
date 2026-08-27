"""ElevenLabs Agents API client + snapshot builder.

Two jobs live here, deliberately kept in one module because they share the
same real-API shape knowledge:

  1. ElevenLabsClient: thin REST wrapper (requests + xi-api-key) over the
     four endpoints this project needs: GET /v1/convai/agents,
     /agents/{id}, /conversations?agent_id=, /conversations/{id}. Only
     fields that exist in the real API today are read — nothing here
     depends on a metric ElevenLabs doesn't actually expose.

  2. The snapshot builder: turns raw API JSON into the models.py types the
     rest of the pipeline consumes, and — this is the important part —
     mechanically derives AggregateMetrics from the transcript sample so
     cheap_pass.py never has to look at turn text itself. See models.py's
     docstring for why that split matters.

Everything here was written against real response shapes pulled from a
live ElevenLabs workspace while building this repo (see fixtures/ and the
README) — not from documentation guesses. Two real-data quirks worth
flagging because a naive implementation would get them wrong:

  - Turn-level latency (conversation_turn_metrics.metrics.convai_llm_service_ttfb)
    is reported in SECONDS (e.g. 0.30038...), not milliseconds, despite the
    field name. We convert to ms once here so the rest of the pipeline
    only ever sees ms.

  - A single "tool round" is split across TWO consecutive transcript
    entries: one agent-role entry carrying tool_calls (request) with an
    empty message, and the next agent-role entry carrying tool_results
    (matched by request_id) with is_error, also with an empty message.
    _merge_tool_rounds() below collapses that pair into one
    ConversationTurn so ToolCallRecord can carry name + is_error together,
    which is the shape the rest of this codebase expects.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from agent_config_judge.models import (
    AgentConfigSnapshot,
    AggregateMetrics,
    AgentSnapshot,
    ConversationRecord,
    ConversationTurn,
    ToolCallRecord,
    ToolConfig,
)

ELEVENLABS_API_BASE = "https://api.elevenlabs.io"

# System tool slots the Agents API surfaces under
# conversation_config.agent.prompt.built_in_tools — configured slots are an
# object, unconfigured ones are null. transfer_to_number / transfer_to_agent
# are the only ones the rubric cares about (human_handoff), but we keep the
# rest so a config dump remains complete for debugging.
_BUILT_IN_TOOL_KEYS = (
    "transfer_to_agent",
    "transfer_to_number",
    "end_call",
    "language_detection",
    "skip_turn",
    "play_keypad_touch_tone",
    "voicemail_detection",
)


class ElevenLabsApiError(RuntimeError):
    pass


@dataclass
class ElevenLabsClient:
    """Thin wrapper over the four Agents API endpoints this project uses.

    Auth is the xi-api-key header, per the Agents API — not a Bearer token.
    """

    api_key: str | None = None
    base_url: str = ELEVENLABS_API_BASE
    timeout_secs: float = 30.0
    _session: Any = None

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("ELEVENLABS_API_KEY")
        if not self.api_key:
            raise ElevenLabsApiError(
                "ELEVENLABS_API_KEY not set (env var or api_key argument). "
                "See .env.example."
            )

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        import requests  # imported lazily so the rest of the package has no hard dep

        resp = requests.get(
            f"{self.base_url}{path}",
            headers={"xi-api-key": self.api_key},
            params=params or {},
            timeout=self.timeout_secs,
        )
        if resp.status_code != 200:
            raise ElevenLabsApiError(f"GET {path} -> {resp.status_code}: {resp.text[:500]}")
        return resp.json()

    def list_agents(self, page_size: int = 100) -> list[dict[str, Any]]:
        agents: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/v1/convai/agents", params)
            agents.extend(data.get("agents", []))
            cursor = data.get("next_cursor")
            if not cursor or not data.get("has_more"):
                break
        return agents

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self._get(f"/v1/convai/agents/{agent_id}")

    def list_conversations(self, agent_id: str, page_size: int = 30) -> list[dict[str, Any]]:
        convos: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {"agent_id": agent_id, "page_size": page_size}
            if cursor:
                params["cursor"] = cursor
            data = self._get("/v1/convai/conversations", params)
            convos.extend(data.get("conversations", []))
            cursor = data.get("next_cursor")
            if not cursor or not data.get("has_more"):
                break
        return convos

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        return self._get(f"/v1/convai/conversations/{conversation_id}")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


def parse_agent_config(raw_agent: dict[str, Any]) -> AgentConfigSnapshot:
    """Build an AgentConfigSnapshot from a raw GET /agents/{id} response.

    Defensive .get() chains throughout: this project has seen agents with
    almost every field null or missing (the empty "Customer Support Agent"
    stub in the fixtures is a real example), and a config snapshot is
    exactly the artifact that should never itself crash on a broken agent.
    """
    prompt_block = (
        raw_agent.get("conversation_config", {}).get("agent", {}).get("prompt", {}) or {}
    )

    kb_ids = tuple(
        doc.get("id", "") for doc in (prompt_block.get("knowledge_base") or []) if doc.get("id")
    )

    tools: list[ToolConfig] = []
    seen_names: set[str] = set()

    for t in prompt_block.get("tools") or []:
        name = t.get("name", "")
        tool_type = t.get("type", "unknown")
        system_tool_type = None
        detail = ""
        if tool_type == "system":
            system_tool_type = (t.get("params") or {}).get("system_tool_type")
        elif tool_type == "webhook":
            detail = (t.get("api_schema") or {}).get("url", "")
        tools.append(ToolConfig(name=name, tool_type=tool_type, system_tool_type=system_tool_type, detail=detail))
        seen_names.add(name)

    # built_in_tools is a second, sometimes-redundant source for the
    # transfer/handoff tools specifically — real data shows transfer tools
    # configured there without necessarily appearing in the `tools` list.
    built_in = prompt_block.get("built_in_tools") or {}
    for key in _BUILT_IN_TOOL_KEYS:
        cfg = built_in.get(key)
        if cfg is None:
            continue
        name = cfg.get("name") or key
        if name in seen_names:
            continue
        system_tool_type = (cfg.get("params") or {}).get("system_tool_type", key)
        detail = ""
        if system_tool_type == "transfer_to_number":
            transfers = (cfg.get("params") or {}).get("transfers") or []
            detail = f"{len(transfers)} destination(s)"
        elif system_tool_type == "transfer_to_agent":
            transfers = (cfg.get("params") or {}).get("transfers") or []
            detail = f"{len(transfers)} agent transfer(s)"
        tools.append(ToolConfig(name=name, tool_type="system", system_tool_type=system_tool_type, detail=detail))
        seen_names.add(name)

    return AgentConfigSnapshot(
        agent_id=raw_agent.get("agent_id", ""),
        name=raw_agent.get("name", ""),
        system_prompt=prompt_block.get("prompt", "") or "",
        knowledge_base_ids=kb_ids,
        tools=tuple(tools),
    )


# ---------------------------------------------------------------------------
# Conversation parsing
# ---------------------------------------------------------------------------


def _merge_tool_rounds(raw_turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse a (tool_calls-only turn, tool_results-only turn) pair into one.

    See module docstring: the real API reports a tool round as two separate
    transcript entries. This walks the list once and merges any entry whose
    tool_calls have no matching tool_results yet with the following entry
    that supplies them (matched by request_id, falling back to tool_name
    order if request_id is absent).
    """
    merged: list[dict[str, Any]] = []
    i = 0
    while i < len(raw_turns):
        turn = raw_turns[i]
        calls = turn.get("tool_calls") or []
        results = turn.get("tool_results") or []
        if calls and not results and i + 1 < len(raw_turns):
            nxt = raw_turns[i + 1]
            nxt_results = nxt.get("tool_results") or []
            if nxt_results and not (nxt.get("tool_calls") or []) and not nxt.get("message"):
                combined = dict(turn)
                combined["tool_results"] = nxt_results
                merged.append(combined)
                i += 2
                continue
        merged.append(turn)
        i += 1
    return merged


# Regex proxy for "this turn states a specific, checkable fact" — numbers,
# currency, percentages, or an explicit policy/limit word near a number.
# Deliberately broad (over-flag, per cheap_pass's design brief) rather than
# precise; the judge is what actually confirms grounding.
_SPECIFIC_CLAIM_RE = re.compile(
    r"(\$\s?\d|\d+\s?%|\bUSD\b|\b\d{1,3}(?:[.,]\d{3})+\b|\b\d+\s?(days?|d[ií]as?|hours?|horas?)\b)",
    re.IGNORECASE,
)

# Escalation proxy beyond the transfer_to_number/transfer_to_agent tools:
# a phrase heuristic for "handing off", used only to decide whether a
# conversation counts as "escalated" for the escalation_health proxy. Kept
# separate from tool-name matching so agents that escalate via a custom
# tool (e.g. create_support_ticket) can still be caught if their agent
# message says so explicitly — though the known blind spot (a ticket tool
# with no such phrasing) is exactly the golden-set false-positive trap.
_ESCALATION_PHRASE_RE = re.compile(
    r"(transfer(ring)? you|conectar(te)? con|pasar(te)? con|an? (agent|advisor|rep)|"
    r"un asesor|una persona|con alguien|create.*ticket|open.*ticket|crear.*ticket)",
    re.IGNORECASE,
)

# Frustration-keyword proxy — coarse, English/Spanish, and explicitly not
# cause-attributed (see cheap_pass._score_sentiment note).
_FRUSTRATION_RE = re.compile(
    r"(this (isn'?t|is not) working|i already (told|said)|ya te dije|no (funciona|entiendes)|"
    r"this is (ridiculous|useless)|forget it|olvidalo|estoy hart[oa]|come on|"
    r"i said that|de nuevo\?|again\?)",
    re.IGNORECASE,
)


def parse_conversation(raw_conv: dict[str, Any]) -> ConversationRecord:
    """Build a ConversationRecord from a raw GET /conversations/{id} response."""
    channel = raw_conv.get("metadata", {}).get("conversation_initiation_source", "unknown")
    raw_turns = _merge_tool_rounds(raw_conv.get("transcript", []) or [])

    turns: list[ConversationTurn] = []
    for rt in raw_turns:
        tool_calls = tuple(
            ToolCallRecord(tool_name=tr.get("tool_name", ""), is_error=bool(tr.get("is_error", False)))
            for tr in (rt.get("tool_results") or [])
        )
        ttfb_ms = None
        metrics = ((rt.get("conversation_turn_metrics") or {}).get("metrics") or {})
        ttfb = metrics.get("convai_llm_service_ttfb")
        if isinstance(ttfb, dict) and "elapsed_time" in ttfb:
            ttfb_ms = float(ttfb["elapsed_time"]) * 1000.0  # API reports seconds despite the field name

        turns.append(
            ConversationTurn(
                role=rt.get("role", "unknown"),
                text=rt.get("message") or "",
                used_static_kb_document_ids=tuple(rt.get("used_static_kb_document_ids") or []),
                tool_calls=tool_calls,
                ttfb_ms=ttfb_ms,
            )
        )

    return ConversationRecord(
        conversation_id=raw_conv.get("conversation_id", ""),
        channel=channel,
        turns=tuple(turns),
    )


# ---------------------------------------------------------------------------
# Aggregate metrics: the mechanical (non-LLM) derivation cheap_pass consumes
# ---------------------------------------------------------------------------


def compute_aggregate_metrics(conversations: list[ConversationRecord]) -> AggregateMetrics:
    """Mechanically derive AggregateMetrics from a conversation sample.

    Every count here comes from counting/regex, never from an LLM reading
    the transcript for meaning — that distinction is what lets cheap_pass
    take this as input without violating its own "no LLM, no transcript
    reading" rule. See models.py's docstring for the full rationale.
    """
    n_conversations = len(conversations)
    n_turns = 0
    channels: set[str] = set()
    tool_call_count = 0
    tool_error_count = 0
    specific_claim_turns = 0
    specific_claim_turns_without_kb = 0
    escalation_tool_call_count = 0
    conversations_with_escalation = 0
    repeated_question_conversations = 0
    agent_turns_sampled = 0
    negative_sentiment_turns = 0
    turns_with_latency_data = 0
    turns_over_latency_band = 0

    # Import here (not at module top) to avoid a cheap_pass -> elevenlabs_client
    # -> cheap_pass cycle; cheap_pass owns the latency-band constants.
    from agent_config_judge.cheap_pass import LATENCY_BAND_MS_BY_CHANNEL, DEFAULT_LATENCY_BAND_MS

    for conv in conversations:
        n_turns += len(conv.turns)
        channels.add(conv.channel)
        band = LATENCY_BAND_MS_BY_CHANNEL.get(conv.channel, DEFAULT_LATENCY_BAND_MS)

        conv_escalated = False
        seen_user_texts: list[str] = []

        for turn in conv.turns:
            for tc in turn.tool_calls:
                tool_call_count += 1
                if tc.is_error:
                    tool_error_count += 1
                if any(k in tc.tool_name.lower() for k in ("transfer_to_number", "transfer_to_agent")):
                    escalation_tool_call_count += 1
                    conv_escalated = True

            if turn.role == "agent":
                agent_turns_sampled += 1
                if turn.text and _SPECIFIC_CLAIM_RE.search(turn.text):
                    specific_claim_turns += 1
                    if not turn.used_static_kb_document_ids:
                        specific_claim_turns_without_kb += 1
                if turn.text and _ESCALATION_PHRASE_RE.search(turn.text):
                    conv_escalated = True
                if turn.ttfb_ms is not None:
                    turns_with_latency_data += 1
                    if turn.ttfb_ms > band:
                        turns_over_latency_band += 1

            if turn.role == "user" and turn.text:
                # crude repeat detector: an exact (normalized) repeat of an
                # earlier user turn in the same conversation is a proxy for
                # "had to say it again" — not exhaustive, just cheap.
                normalized = turn.text.strip().lower()
                if normalized in seen_user_texts:
                    repeated_question_conversations += 1
                    seen_user_texts = []  # count each conversation at most once
                else:
                    seen_user_texts.append(normalized)
                if _FRUSTRATION_RE.search(turn.text):
                    negative_sentiment_turns += 1

        if conv_escalated:
            conversations_with_escalation += 1

    return AggregateMetrics(
        n_conversations_sampled=n_conversations,
        n_turns_sampled=n_turns,
        channels_seen=tuple(sorted(channels)),
        tool_call_count=tool_call_count,
        tool_error_count=tool_error_count,
        specific_claim_turns=specific_claim_turns,
        specific_claim_turns_without_kb=specific_claim_turns_without_kb,
        escalation_tool_call_count=escalation_tool_call_count,
        conversations_with_escalation=conversations_with_escalation,
        repeated_question_conversations=repeated_question_conversations,
        agent_turns_sampled=agent_turns_sampled,
        negative_sentiment_turns=negative_sentiment_turns,
        turns_with_latency_data=turns_with_latency_data,
        turns_over_latency_band=turns_over_latency_band,
    )


def build_agent_snapshot(
    raw_agent: dict[str, Any],
    raw_conversations: list[dict[str, Any]],
    arr_usd: float | None = None,
) -> AgentSnapshot:
    """End-to-end: raw agent + raw conversation list -> AgentSnapshot.

    This is what `agentjudge fetch-portfolio` calls per agent, and what the
    fixture-building script used to produce fixtures/real_portfolio_snapshot.json.
    """
    config = parse_agent_config(raw_agent)
    conversations = [parse_conversation(c) for c in raw_conversations]
    metrics = compute_aggregate_metrics(conversations)
    return AgentSnapshot(
        agent_id=config.agent_id,
        name=config.name,
        config=config,
        metrics=metrics,
        conversations=tuple(conversations),
        arr_usd=arr_usd,
    )


def redact_phone_numbers(text: str) -> str:
    """Redact phone-number-shaped substrings (E.164 and loose variants).

    Used when saving any real transcript/config text to a fixture file —
    per the repo's own rule, real data goes in redacted, everything else
    stays as it came.
    """
    # +<country><number>, 8+ consecutive digits optionally grouped with
    # spaces/dashes/dots/parens — deliberately broad, this is a redaction
    # tool, not a validator, so it should err toward over-matching.
    pattern = re.compile(r"(\+?\d[\d\s\-\.\(\)]{6,}\d)")

    def _replace(m: re.Match[str]) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) < 7:
            return m.group(0)
        return "[REDACTED_PHONE]"

    return pattern.sub(_replace, text)
