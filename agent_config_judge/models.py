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
class AgentConfigSnapshot:
    agent_id: str
    name: str
    system_prompt: str
    knowledge_base_ids: tuple[str, ...] = ()
    tools: tuple[ToolConfig, ...] = ()

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
    # whether that turn had any used_static_kb_document_ids.
    specific_claim_turns: int = 0
    specific_claim_turns_without_kb: int = 0

    # Escalation proxy: transfer_to_number / transfer_to_agent tool calls,
    # or an agent turn matching an escalation-phrase heuristic.
    escalation_tool_call_count: int = 0
    conversations_with_escalation: int = 0

    # Multi-turn proxy: a later user turn re-supplying information that
    # matched an earlier turn's extracted slot (crude repeat detector).
    repeated_question_conversations: int = 0

    # Sentiment proxy: agent turns following a user turn that matched a
    # frustration-keyword heuristic ("this isn't working", "I already told
    # you", profanity list, etc.) — a coarse stand-in for real sentiment
    # analysis, flagged as such in cheap_pass.
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
