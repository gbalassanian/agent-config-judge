"""Re-confirm a clean judge read before trusting it — never re-check a dirty one.

Motivated by something observed running the live backend for real, not a
theoretical worry: the same agent, the exact same config and conversation
sample, judged four separate times, came back "healthy" once and found a
real (evidence-validated, not fabricated) failure the other three times.
One in four single-call misses on a criterion that isn't even a subtle
one. That number is a single anecdote, not a calibrated rate — see the
class docstring below and README's calibration backlog — but it's real
data, not a guess, and it points at a specific, fixable gap: a "healthy"
verdict gets exactly one chance to be wrong, and nothing downstream ever
asks again.

That asymmetry is the whole design here. An agent the judge flags with a
real failure already reaches a human via the router (self_serve_fix,
targeted_nudge, escalate) regardless of whether that one call missed a
second failure too — the stakes of an extra miss on an already-flagged
agent are low, there's already a human in the loop. An agent the judge
calls "healthy" reaches no one. That is the single outcome worth spending
extra live calls to protect, so this backend spends them precisely there:
never on an agent the first call already found something wrong with, and
increasingly rarely as a "still clean" agent survives more confirmation
attempts.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from agent_config_judge.judge import validate_judge_output
from agent_config_judge.models import AgentConfigSnapshot, ConversationRecord
from agent_config_judge.rubric import CRITERION_ORDER


@dataclass
class EnsembleJudgeBackend:
    """Wraps another JudgeBackend; only re-runs it when the read so far is clean.

    Calls the wrapped backend once. If that call, once validated the same
    way run_judge always validates it, already has a real failure anywhere,
    returns it unchanged — one confirmed problem is enough to act on, and
    calling again wouldn't change the fact that there's already something
    to route. Only when a call comes back with zero validated failures
    does this call the backend again, up to `max_extra_runs` more times,
    stopping the moment any attempt reports a real one. If every attempt
    (up to 1 + max_extra_runs total) comes back clean, "healthy" is what
    gets returned — now backed by that many independent confirmations
    instead of one.

    max_extra_runs=2 is today's pick, not a validated number: it comes from
    a single observed case (see module docstring) suggesting a rough ~25%
    per-call miss rate, and the diminishing-returns arithmetic that follows
    from it — going 1->2 confirmations cuts the odds every attempt misses
    from ~25% to ~6%, 2->3 cuts it further to ~2%, past which extra calls
    buy very little for their linear cost. That arithmetic assumes each
    call is an independent roll, which nothing here actually verifies —
    it could just as easily be a reproducible blind spot on a specific
    evidence shape, which more calls would never fix. Treat this constant
    the same as every other unvalidated number in README's calibration
    backlog: a placeholder pending the real fix — a proper repeatability
    eval (run the judge N times per golden-set case, live, and measure the
    actual agreement rate) instead of one afternoon's anecdote.

    Merging: for each of the 9 criteria, independently, the FIRST attempt
    (in call order) whose validated verdict for that criterion is "fail"
    contributes its raw (unvalidated) entry for that criterion to the
    merged output; every other criterion falls back to the first attempt's
    entry, since pass vs. unknown never affects classification or routing
    (only fail does — see judge.py's classify-by-recipe-mapping). Each
    attempt is validated with the exact same validate_judge_output() every
    single live call already goes through — this never invents a lighter-
    weight check, so a fabricated or altered citation in ANY attempt is
    caught and discarded exactly as it would be outside the ensemble;
    "found a fail" here always means "found a fail that survived real
    evidence validation," never a raw, unvalidated claim.

    One known simplification, left as one deliberately rather than fixed
    silently: if two different attempts both validate a fail on the SAME
    criterion but name two different cause_codes, only the first one found
    is kept — CriterionVerdict has room for exactly one cause_code, not a
    set of candidates. A real second cause on the same criterion from a
    later attempt is silently dropped rather than surfaced twice.
    """

    backend: Any  # JudgeBackend — Any to avoid importing the Protocol just for a type hint
    max_extra_runs: int = 2
    # Per-agent attempt count from the most recent judge() call, keyed by
    # agent_id — read by CachedJudgeBackend (duck-typed, optional) so a
    # cached "healthy" can record how many attempts actually backed it,
    # without JudgeBackend's return contract needing a metadata field just
    # for this. See judge_cache.py's docstring.
    last_attempt_counts: dict[str, int] = field(default_factory=dict)

    def judge(self, config: AgentConfigSnapshot, conversations: tuple[ConversationRecord, ...]) -> dict[str, Any]:
        budget = 1 + max(self.max_extra_runs, 0)
        attempts: list[dict[str, Any]] = []
        found = False
        for _ in range(budget):
            raw = self.backend.judge(config, conversations)
            attempts.append(raw)
            validated = validate_judge_output(raw, agent_id=config.agent_id, conversations=conversations)
            if validated.failures:
                found = True
                break
        self.last_attempt_counts[config.agent_id] = len(attempts)

        # Visible on purpose: the whole point of this backend is a decision
        # ("how many times did we actually have to ask before trusting this
        # read?") that a cost line on an Anthropic invoice can't answer by
        # itself — see the conversation that led to this. stderr, not
        # stdout, so it never gets mixed into --output's JSON or into a
        # test capturing normal print() output.
        outcome = "found a real failure" if found else "still clean after full budget"
        print(
            f"  [ensemble] {config.agent_id}: used {len(attempts)}/{budget} attempt(s) ({outcome})",
            file=sys.stderr,
        )

        return _merge(attempts, config, conversations)


def _merge(
    attempts: list[dict[str, Any]],
    config: AgentConfigSnapshot,
    conversations: tuple[ConversationRecord, ...],
) -> dict[str, Any]:
    validated_attempts = [
        validate_judge_output(raw, agent_id=config.agent_id, conversations=conversations) for raw in attempts
    ]
    merged_criteria: dict[str, Any] = {}
    for cid in CRITERION_ORDER:
        chosen = None
        for raw, validated in zip(attempts, validated_attempts):
            if validated.criteria[cid].verdict == "fail":
                chosen = raw.get("criteria", {}).get(cid) if isinstance(raw, dict) else None
                break
        if chosen is None:
            first_raw = attempts[0] if isinstance(attempts[0], dict) else {}
            chosen = first_raw.get("criteria", {}).get(cid)
        merged_criteria[cid] = chosen

    last_raw = attempts[-1] if isinstance(attempts[-1], dict) else {}
    return {
        "agent_id": config.agent_id,
        "criteria": merged_criteria,
        "notes": last_raw.get("notes", "") if isinstance(last_raw, dict) else "",
    }
