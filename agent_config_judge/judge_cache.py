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
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_config_judge.judge import _format_config, _format_conversations
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

    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        fingerprint = _fingerprint(config, conversations)
        cache = self._load()
        entry = cache.get(config.agent_id)
        if entry is not None and entry.get("fingerprint") == fingerprint:
            return entry["raw_judgement"]

        raw = self.backend.judge(config, conversations)
        cache[config.agent_id] = {"fingerprint": fingerprint, "raw_judgement": raw}
        self._save(cache)
        return raw
