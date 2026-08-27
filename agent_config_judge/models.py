"""Shared data model for an agent's snapshot: config + mechanically-derived
aggregate metrics + the raw conversation sample.

Why this split exists (it's the load-bearing design decision in this repo):

  - AgentConfigSnapshot: fields that exist in the ElevenLabs Agents API
    config today. No wishlist metrics.

  - AggregateMetrics: numbers *mechanically* derived from a conversation
    sample (regex/counting, not an LLM reading anything). This is what the
    cheap pass is allowed to touch. Deriving these is real work — it's how
    a proxy exists at all for signals ElevenLabs doesn't expose as a
    first-class metric (grounding attribution, escalation health, dead
    ends) — but it's mechanical work, not judgment, so it doesn't count as
    "reading conversations" in the sense the cheap pass is barred from.

  - ConversationRecord / turns: the actual transcript text. Only the judge
    (judge.py) is allowed to read these. cheap_pass.py's function signature
    intentionally takes AgentConfigSnapshot + AggregateMetrics and never an
    AgentSnapshot or ConversationRecord, so a cheap-pass implementation
    that started reading turn text would be a type error, not just a
    style violation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolConfig:
    """One tool attached to the agent, config-side only."""

    name: str
    tool_type: str  # "system" | "webhook" | "client" | "mcp" | "smb" | ...
    # For system tools this is the specific kind, e.g. "transfer_to_number",
    # "transfer_to_agent", "end_call". None for non-system tools.
    system_tool_type: str | None = None
    # Free-form extra detail kept for the judge/router to cite as evidence
    # (e.g. transfer destinations) without over-modeling the whole API shape.
    detail: str = ""


@dataclass(frozen=True)
class KnowledgeBaseDocConfig:
    """One KB document attached to the agent, config-side only.

    usage_mode is ElevenLabs' own field (e.g. "auto") controlling whether
    this doc is always injected into the prompt or only pulled in via RAG
    retrieval at query time. Judge-relevant context for explaining a
    grounding failure — never read by cheap_pass, which only ever counts
    knowledge_base_ids (presence/absence), not how a doc is actually used.
    """

    id: str
    name: str = ""
    usage_mode: str | None = None


@dataclass(frozen=True)
class AgentConfigSnapshot:
    agent_id: str
    name: str
    system_prompt: str
    knowledge_base_ids: tuple[str, ...] = ()
    tools: tuple[ToolConfig, ...] = ()
    # Judge-facing detail beyond the cheap-pass-scored knowledge_base_ids
    # above: per-doc usage_mode, and conversation_config.agent.prompt.rag.
    # enabled — whether RAG retrieval is actually on for this agent, as
    # opposed to a KB attached but always-injected (usage_mode="auto")
    # with rag never queried. cheap_pass never reads either field; it's
    # the judge that needs "is a KB attached" (cheap pass's question)
    # distinguished from "is it actually retrievable/used" (the judge's).
    knowledge_base_docs: tuple[KnowledgeBaseDocConfig, ...] = ()
    rag_enabled: bool | None = None

    def has_tool_type(self, system_tool_type: str) -> bool:
        return any(t.system_tool_type == system_tool_type for t in self.tools)


@dataclass(frozen=True)
class AggregateMetrics:
    """Numbers mechanically counted from a sample of conversations.

    Every *_count / *_turns field is a raw count over the sample, not a
    rate — cheap_pass.py divides, so the sample size travels alongside the
    numbers instead of being baked into a ratio that hides n=3.
    """

    n_conversations_sampled: int = 0
    n_turns_sampled: int = 0
    channels_seen: tuple[str, ...] = ()

    tool_call_count: int = 0
    tool_error_count: int = 0  # tool_results[].is_error across the sample

    # Grounding proxy: turns where the agent made a specific factual claim
    # (number/price/policy/format — see elevenlabs_client's regex) and
    # whether that claim is attributable — to a used_static_kb_document_id,
    # an adjacent tool call, or a number the user supplied a few turns
    # earlier. The field name predates the second and third sources (added
    # after finding real false-positive cases — a tool-grounded answer with
    # no KB doc, and an echoed user-supplied order number) and is kept for
    # compatibility; read it as "without an attributable source", not
    # literally "without a KB doc".
    specific_claim_turns: int = 0
    specific_claim_turns_without_kb: int = 0

    # Escalation proxy: transfer_to_number / transfer_to_agent tool calls,
    # or an agent turn matching an escalation-phrase heuristic.
    escalation_tool_call_count: int = 0
    conversations_with_escalation: int = 0

    # Multi-turn proxy: a later user turn that is a VERBATIM (normalized)
    # repeat of an earlier one in the same conversation — e.g. spelling out
    # a phone number or confirmation code a second time because the ASR
    # missed it the first time. This deliberately does NOT try to catch the
    # more common real failure — the agent re-asking the same thing in
    # different words ("what's your name" vs "could you tell me your name
    # again") — because detecting that needs to understand that two
    # different strings mean the same question, which is a semantic
    # judgment, not a mechanical one. A similarity heuristic confident
    # enough to attempt that would just be a worse, cheaper copy of the
    # judge running inside the tier that isn't supposed to interpret
    # anything; the honest tradeoff is a proxy with real but narrow recall
    # (never wrong when it fires, blind to the paraphrased case) rather
    # than a fuzzy one that can be confidently wrong in either direction.
    repeated_question_conversations: int = 0

    # Sentiment: two sources, real preferred over heuristic.
    #
    # ElevenLabs computes its own per-conversation sentiment judgment
    # (analysis.sentiment_analysis.overall_label) — a real signal, not a
    # heuristic, so cheap_pass prefers it when the sample has it.
    conversations_with_sentiment_label: int = 0
    conversations_negative_sentiment_label: int = 0

    # Fallback proxy for samples with no real sentiment_analysis data
    # (synthetic golden-set cases; an older account/version that hasn't
    # backfilled it): agent turns following a user turn that matched a
    # frustration-keyword heuristic ("this isn't working", "I already told
    # you", profanity list, etc.) — a coarse stand-in, flagged as such in
    # cheap_pass.
    agent_turns_sampled: int = 0
    negative_sentiment_turns: int = 0

    # Latency proxy: convai_llm_service_ttfb per turn vs a channel latency
    # band (placeholder bands — see cheap_pass.LATENCY_BAND_MS_BY_CHANNEL).
    turns_with_latency_data: int = 0
    turns_over_latency_band: int = 0


@dataclass(frozen=True)
class ToolCallRecord:
    tool_name: str
    is_error: bool


@dataclass(frozen=True)
class ConversationTurn:
    role: str  # "user" | "agent"
    text: str
    used_static_kb_document_ids: tuple[str, ...] = ()
    tool_calls: tuple[ToolCallRecord, ...] = ()
    ttfb_ms: float | None = None


@dataclass(frozen=True)
class ConversationRecord:
    conversation_id: str
    channel: str  # metadata.conversation_initiation_source
    turns: tuple[ConversationTurn, ...] = ()
    # analysis.sentiment_analysis.overall_label — ElevenLabs' own real
    # per-conversation sentiment judgment (e.g. "positive"/"neutral"/
    # "negative"). None when the sample has no analysis block at all
    # (synthetic cases; an account/version that doesn't compute it).
    overall_sentiment_label: str | None = None


@dataclass(frozen=True)
class AgentSnapshot:
    """Everything the pipeline knows about one agent.

    is_synthetic / synthetic_note exist because the golden set mixes real
    portfolio agents with hand-built synthetic cases (see eval/golden_set.py)
    — every synthetic case must self-declare so nobody downstream mistakes
    a fabricated transcript for a real customer conversation.
    """

    agent_id: str
    name: str
    config: AgentConfigSnapshot
    metrics: AggregateMetrics
    conversations: tuple[ConversationRecord, ...] = ()
    arr_usd: float | None = None  # None = unknown; router treats unknown as low-ARR
    is_synthetic: bool = False
    synthetic_note: str | None = None

    def __post_init__(self) -> None:
        if self.is_synthetic and not self.synthetic_note:
            raise ValueError(
                f"Synthetic agent snapshot {self.agent_id!r} must carry a synthetic_note "
                "explaining what failure mode it stands in for."
            )


# ---------------------------------------------------------------------------
# JSON (de)serialization — used by the CLI's snapshot files and by the eval
# harness's golden set. Written by hand rather than dataclasses.asdict()
# because reconstruction needs typed nested objects back (tuples of
# ToolConfig, ConversationTurn, etc.), not plain dicts.
# ---------------------------------------------------------------------------


def agent_snapshot_to_dict(snap: AgentSnapshot) -> dict:
    return {
        "agent_id": snap.agent_id,
        "name": snap.name,
        "arr_usd": snap.arr_usd,
        "is_synthetic": snap.is_synthetic,
        "synthetic_note": snap.synthetic_note,
        "config": {
            "agent_id": snap.config.agent_id,
            "name": snap.config.name,
            "system_prompt": snap.config.system_prompt,
            "knowledge_base_ids": list(snap.config.knowledge_base_ids),
            "knowledge_base_docs": [
                {"id": d.id, "name": d.name, "usage_mode": d.usage_mode} for d in snap.config.knowledge_base_docs
            ],
            "rag_enabled": snap.config.rag_enabled,
            "tools": [
                {"name": t.name, "tool_type": t.tool_type, "system_tool_type": t.system_tool_type, "detail": t.detail}
                for t in snap.config.tools
            ],
        },
        "metrics": {
            "n_conversations_sampled": snap.metrics.n_conversations_sampled,
            "n_turns_sampled": snap.metrics.n_turns_sampled,
            "channels_seen": list(snap.metrics.channels_seen),
            "tool_call_count": snap.metrics.tool_call_count,
            "tool_error_count": snap.metrics.tool_error_count,
            "specific_claim_turns": snap.metrics.specific_claim_turns,
            "specific_claim_turns_without_kb": snap.metrics.specific_claim_turns_without_kb,
            "escalation_tool_call_count": snap.metrics.escalation_tool_call_count,
            "conversations_with_escalation": snap.metrics.conversations_with_escalation,
            "repeated_question_conversations": snap.metrics.repeated_question_conversations,
            "conversations_with_sentiment_label": snap.metrics.conversations_with_sentiment_label,
            "conversations_negative_sentiment_label": snap.metrics.conversations_negative_sentiment_label,
            "agent_turns_sampled": snap.metrics.agent_turns_sampled,
            "negative_sentiment_turns": snap.metrics.negative_sentiment_turns,
            "turns_with_latency_data": snap.metrics.turns_with_latency_data,
            "turns_over_latency_band": snap.metrics.turns_over_latency_band,
        },
        "conversations": [
            {
                "conversation_id": c.conversation_id,
                "channel": c.channel,
                "overall_sentiment_label": c.overall_sentiment_label,
                "turns": [
                    {
                        "role": t.role,
                        "text": t.text,
                        "used_static_kb_document_ids": list(t.used_static_kb_document_ids),
                        "tool_calls": [{"tool_name": tc.tool_name, "is_error": tc.is_error} for tc in t.tool_calls],
                        "ttfb_ms": t.ttfb_ms,
                    }
                    for t in c.turns
                ],
            }
            for c in snap.conversations
        ],
    }


def agent_snapshot_from_dict(d: dict) -> AgentSnapshot:
    config_d = d["config"]
    config = AgentConfigSnapshot(
        agent_id=config_d["agent_id"],
        name=config_d["name"],
        system_prompt=config_d["system_prompt"],
        knowledge_base_ids=tuple(config_d.get("knowledge_base_ids", [])),
        knowledge_base_docs=tuple(
            KnowledgeBaseDocConfig(id=doc["id"], name=doc.get("name", ""), usage_mode=doc.get("usage_mode"))
            for doc in config_d.get("knowledge_base_docs", [])
        ),
        rag_enabled=config_d.get("rag_enabled"),
        tools=tuple(
            ToolConfig(name=t["name"], tool_type=t["tool_type"], system_tool_type=t.get("system_tool_type"), detail=t.get("detail", ""))
            for t in config_d.get("tools", [])
        ),
    )
    metrics_d = d["metrics"]
    metrics = AggregateMetrics(
        n_conversations_sampled=metrics_d.get("n_conversations_sampled", 0),
        n_turns_sampled=metrics_d.get("n_turns_sampled", 0),
        channels_seen=tuple(metrics_d.get("channels_seen", [])),
        tool_call_count=metrics_d.get("tool_call_count", 0),
        tool_error_count=metrics_d.get("tool_error_count", 0),
        specific_claim_turns=metrics_d.get("specific_claim_turns", 0),
        specific_claim_turns_without_kb=metrics_d.get("specific_claim_turns_without_kb", 0),
        escalation_tool_call_count=metrics_d.get("escalation_tool_call_count", 0),
        conversations_with_escalation=metrics_d.get("conversations_with_escalation", 0),
        repeated_question_conversations=metrics_d.get("repeated_question_conversations", 0),
        conversations_with_sentiment_label=metrics_d.get("conversations_with_sentiment_label", 0),
        conversations_negative_sentiment_label=metrics_d.get("conversations_negative_sentiment_label", 0),
        agent_turns_sampled=metrics_d.get("agent_turns_sampled", 0),
        negative_sentiment_turns=metrics_d.get("negative_sentiment_turns", 0),
        turns_with_latency_data=metrics_d.get("turns_with_latency_data", 0),
        turns_over_latency_band=metrics_d.get("turns_over_latency_band", 0),
    )
    conversations = tuple(
        ConversationRecord(
            conversation_id=c["conversation_id"],
            channel=c["channel"],
            overall_sentiment_label=c.get("overall_sentiment_label"),
            turns=tuple(
                ConversationTurn(
                    role=t["role"],
                    text=t["text"],
                    used_static_kb_document_ids=tuple(t.get("used_static_kb_document_ids", [])),
                    tool_calls=tuple(ToolCallRecord(tool_name=tc["tool_name"], is_error=tc["is_error"]) for tc in t.get("tool_calls", [])),
                    ttfb_ms=t.get("ttfb_ms"),
                )
                for t in c.get("turns", [])
            ),
        )
        for c in d.get("conversations", [])
    )
    return AgentSnapshot(
        agent_id=d["agent_id"],
        name=d["name"],
        config=config,
        metrics=metrics,
        conversations=conversations,
        arr_usd=d.get("arr_usd"),
        is_synthetic=d.get("is_synthetic", False),
        synthetic_note=d.get("synthetic_note"),
    )
