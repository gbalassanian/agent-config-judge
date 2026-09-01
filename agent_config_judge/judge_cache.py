"""Skip re-invoking the judge when nothing it would read has changed.

The judge is the expensive tier — a real LLM call, real money — and its
output only depends on two things: the agent's config, and the sampled
conversations (see judge.py's `build_judge_prompt`). If neither has
changed since the last time this exact agent was judged, re-running it
produces the identical answer for the identical cost. This is exactly the
"re-scanning should be incremental" idea from README's "Path to scale"
point 4, implemented at the smallest scale that idea actually needs: a
cache file remembering, per agent_id, a fingerprint of what the judge last
saw and the raw output it produced — not a database, not a service, just
enough memory to stop paying twice for the same answer.

Deliberately built as a `JudgeBackend` wrapper, not as logic inside
`pipeline.py` or `run_judge`: `JudgeBackend.judge()` already returns an
unvalidated raw dict — `RecordedJudgeBackend` replays one from a fixture
file instead of calling a live model, and `validate_judge_output()`
re-validates identically no matter where the dict came from. A cache hit
here is exactly that same pattern (a previously-produced raw dict standing
in for a freshly-called one), so it slots in without `pipeline.py` (which
its own docstring says "owns no logic of its own") or `run_judge` needing
to know a cache exists at all — wrap any real backend in a
`CachedJudgeBackend` and pass that to `run_judge` like any other backend.

The fingerprint is a hash of `judge.py`'s own `_format_config` +
`_format_conversations` output — the exact text the judge prompt is built
from — rather than a hand-picked list of "fields that matter" maintained
here separately. That choice is deliberate: a hand-picked list can quietly
drift out of sync with what `build_judge_prompt` actually reads (someone
adds a field to the prompt and forgets to add it here); reusing the same
formatting functions makes that drift structurally impossible. One
consequence worth naming: this means ARR, the agent's display name outside
`config.name`, and the cheap-pass score can never invalidate the cache,
because none of them are part of what gets hashed — not an oversight, the
same "type signature is the enforcement mechanism" principle cheap_pass.py
uses (the judge never reads ARR either; see router.py's docstring).

A fingerprint match means "nothing changed" — but "nothing changed" can
also mean "this agent has had zero new conversations in months," and a
fingerprint has no opinion on how long that's been true. A "healthy"
verdict is the one outcome nothing downstream ever double-checks (see
judge_ensemble.py's docstring for the full reasoning: a flagged agent's
failures reach a human via the router regardless of a second miss; a
"healthy" agent reaches no one) — so a dormant agent that got a wrong
"healthy" once, purely by bad luck on a single judge call, could stay
wrong indefinitely: no new conversation ever arrives to change the
fingerprint and force a re-check. `healthy_ttl_days` exists for exactly
that gap: once a cached "healthy" verdict is older than this many days,
it's treated as a miss and re-judged, even with an unchanged fingerprint.
A cached verdict with a real failure never expires this way — that agent
already has a human looking at it via the router, so re-confirming it on
a timer buys little and costs a real call every time. Off by default
(`healthy_ttl_days=None`), same as every other opt-in knob here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from agent_config_judge.judge import _format_config, _format_conversations, validate_judge_output
from agent_config_judge.models import AgentConfigSnapshot, ConversationRecord


def _fingerprint(config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> str:
    text = _format_config(config) + "\n---\n" + _format_conversations(conversations)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class CachedJudgeBackend:
    """Wraps another JudgeBackend; skips calling it when this agent_id's
    config + conversation sample fingerprint matches the last cached run.

    Read-then-write the whole cache file on every call, not once per scan —
    the same O(n) per-call / O(n^2) total tradeoff `_fetch_agents_concurrently`
    already accepts for its checkpoint file, for the same reason: it's fine
    at the portfolio sizes this targets, and it means a crash mid-scan loses
    nothing already cached. NOT thread-safe as written — fine today because
    the judge tier is deliberately still sequential (see README's "Path to
    scale"); if that ever changes, this needs the same lock the fetch side's
    checkpoint write already uses.
    """

    backend: Any  # JudgeBackend — Any to avoid importing the Protocol just for a type hint
    cache_path: Path
    healthy_ttl_days: int | None = None
    # Injectable so tests can simulate "30 days later" without waiting 30
    # days — never override this outside a test.
    clock: Callable[[], datetime] = field(default=lambda: datetime.now(timezone.utc), repr=False)
    _cache: dict[str, dict[str, Any]] | None = field(default=None, repr=False)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        with open(self.cache_path, encoding="utf-8") as f:
            return json.load(f)

    def _save(self, cache: dict[str, dict[str, Any]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

    def _is_stale_healthy(
        self, entry: dict[str, Any], config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]
    ) -> bool:
        if self.healthy_ttl_days is None:
            return False
        judged_at_str = entry.get("judged_at")
        if judged_at_str is None:
            # A cache entry written before this field existed — don't force
            # an expiry on data that never recorded when it was judged.
            return False
        validated = validate_judge_output(entry["raw_judgement"], agent_id=config.agent_id, conversations=conversations)
        if validated.failures:
            return False  # a real failure never expires this way — see module docstring
        age_days = (self.clock() - datetime.fromisoformat(judged_at_str)).total_seconds() / 86400.0
        return age_days >= self.healthy_ttl_days

    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        fingerprint = _fingerprint(config, conversations)
        cache = self._load()
        entry = cache.get(config.agent_id)
        if (
            entry is not None
            and entry.get("fingerprint") == fingerprint
            and not self._is_stale_healthy(entry, config, conversations)
        ):
            return entry["raw_judgement"]

        raw = self.backend.judge(config, conversations)
        # How many live calls actually produced this raw output — 1 unless
        # the wrapped backend is (or wraps) an EnsembleJudgeBackend, which
        # tracks it per agent_id. Not used to vary healthy_ttl_days today
        # (see judge_ensemble.py / README calibration backlog on why not
        # yet) — recorded so that calibration has real numbers to work from
        # once there's enough history to look at.
        ensemble_attempts = getattr(self.backend, "last_attempt_counts", {}).get(config.agent_id, 1)
        cache[config.agent_id] = {
            "fingerprint": fingerprint,
            "raw_judgement": raw,
            "judged_at": self.clock().isoformat(),
            "ensemble_attempts": ensemble_attempts,
        }
        self._save(cache)
        return raw
