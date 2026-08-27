"""Orchestration: cheap pass -> (maybe) judge -> router, per agent and over
a whole portfolio.

This module owns no logic of its own — it just wires the three tiers
together in the order the case study demands: score everyone cheaply,
read in depth only the ones that got flagged (or forced), then route by
classification x ARR. Keeping the wiring in one place (rather than
letting the CLI call each tier ad hoc) is what makes route_unflagged()'s
distinction — "not_flagged" vs. judge-confirmed "healthy" — impossible to
skip by accident.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_config_judge.cheap_pass import CheapPassResult, score_agent
from agent_config_judge.judge import Judgement, JudgeBackend, run_judge
from agent_config_judge.models import AgentSnapshot
from agent_config_judge.router import RoutingDecision, route, route_unflagged


@dataclass(frozen=True)
class TriageResult:
    agent_id: str
    name: str
    cheap_pass: CheapPassResult
    judgement: Judgement | None  # None iff the cheap pass never flagged this agent
    routing: RoutingDecision


def triage_agent(snapshot: AgentSnapshot, judge_backend: JudgeBackend) -> TriageResult:
    cheap_result = score_agent(snapshot.config, snapshot.metrics)

    if cheap_result.flagged:
        judgement = run_judge(judge_backend, snapshot.config, snapshot.conversations)
        routing = route(judgement, snapshot.arr_usd)
    else:
        judgement = None
        routing = route_unflagged(snapshot.agent_id, snapshot.arr_usd)

    return TriageResult(
        agent_id=snapshot.agent_id,
        name=snapshot.name,
        cheap_pass=cheap_result,
        judgement=judgement,
        routing=routing,
    )


def scan_portfolio(snapshots: list[AgentSnapshot], judge_backend: JudgeBackend) -> list[TriageResult]:
    return [triage_agent(s, judge_backend) for s in snapshots]


@dataclass(frozen=True)
class TriageSummary:
    total_agents: int
    flagged_count: int
    judged_count: int  # == flagged_count today; kept separate in case a future tier changes that
    classification_counts: dict[str, int]  # "not_flagged" | "healthy" | "standard" | "systemic"
    action_counts: dict[str, int]
    requires_approval_count: int
    recipe_gap_count: int


def summarize(results: list[TriageResult]) -> TriageSummary:
    classification_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    requires_approval = 0
    recipe_gaps = 0
    judged = 0

    for r in results:
        classification_counts[r.routing.classification] = classification_counts.get(r.routing.classification, 0) + 1
        action_counts[r.routing.action.value] = action_counts.get(r.routing.action.value, 0) + 1
        if r.routing.requires_human_approval:
            requires_approval += 1
        recipe_gaps += len(r.routing.recipe_gaps)
        if r.judgement is not None:
            judged += 1

    return TriageSummary(
        total_agents=len(results),
        flagged_count=sum(1 for r in results if r.cheap_pass.flagged),
        judged_count=judged,
        classification_counts=classification_counts,
        action_counts=action_counts,
        requires_approval_count=requires_approval,
        recipe_gap_count=recipe_gaps,
    )
