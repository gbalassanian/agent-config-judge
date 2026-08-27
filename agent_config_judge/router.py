"""Router: judge classification × account ARR -> exactly one action.

This is the only module in the pipeline that reads ARR. Neither the cheap
pass nor the judge should ever see it — mixing revenue into detection would
mean a high-ARR account's broken agent scores healthier, which is precisely
backwards. ARR belongs at the routing decision, once, here.

Four outcomes:

  - healthy            -> no_action
  - standard            -> self_serve_fix (all failures batchable via a docs
                           link) or targeted_nudge (at least one failure
                           needs an account-specific message)
  - systemic + high ARR -> escalate_to_engineer
  - systemic + low ARR  -> nearest_guidance (closest self-serve/nudge
                           material that exists, even though it doesn't
                           fully fix an unmapped cause) + a logged
                           RecipeGap, because the causes that recur on this
                           branch are exactly the backlog for the next
                           rubric.py catalog entry.

requires_human_approval is mechanical, not a per-branch judgment call: it's
true only for an action whose RouteAction.touches_live_agent is True. Every
action this router currently emits is a drafted artifact for a human or
customer to act on (a docs link, a nudge message, a ticket) — none of them
write to a live agent's config — so every action here has
touches_live_agent=False and the flag is currently always False. That's
intentional, not an oversight: the day this system gains an action that
auto-applies a fix to a live agent, it sets touches_live_agent=True on that
RouteAction and the approval gate turns on automatically, with no call site
needing to remember to add the check.

CALIBRATION STATUS: ARR_HIGH_THRESHOLD_USD is a placeholder guess, not a
number pulled from real account economics — see README limitations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from agent_config_judge.judge import Judgement, CriterionVerdict
from agent_config_judge.rubric import Recipe

# Placeholder: real threshold needs input from whoever owns account
# economics, not a guess baked into a detector. See README limitations.
ARR_HIGH_THRESHOLD_USD = 50_000.0


class ArrTier(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"  # unknown ARR is treated as low — see models.AgentSnapshot docstring


class RouteAction(str, Enum):
    NO_ACTION = "no_action"
    SELF_SERVE_FIX = "self_serve_fix"
    TARGETED_NUDGE = "targeted_nudge"
    ESCALATE_TO_ENGINEER = "escalate_to_engineer"
    NEAREST_GUIDANCE = "nearest_guidance"


# Whether each action, if executed, writes directly to a customer's live
# agent. None do today — see module docstring — but this table is the
# single place that would need to change if one ever did, rather than a
# scattered set of `if action == ...: requires_approval = True` checks.
_TOUCHES_LIVE_AGENT: dict[RouteAction, bool] = {
    RouteAction.NO_ACTION: False,
    RouteAction.SELF_SERVE_FIX: False,
    RouteAction.TARGETED_NUDGE: False,
    RouteAction.ESCALATE_TO_ENGINEER: False,
    RouteAction.NEAREST_GUIDANCE: False,
}


@dataclass(frozen=True)
class RecipeGap:
    """One occurrence of a failure with no catalog recipe.

    These are the raw material for growing rubric.RECIPE_CATALOG: cause
    codes that show up here repeatedly, worded similarly, are a strong
    signal for the next recipe to add. Logged for every systemic failure
    regardless of which systemic branch it took (engineer vs. nearest
    guidance) — the goal is a complete backlog, not just the low-ARR slice
    the case study text called out.
    """

    agent_id: str
    criterion_id: str
    judge_cause_code: str | None
    evidence_quote: str | None
    evidence_config_field: str | None


@dataclass(frozen=True)
class RoutingDecision:
    agent_id: str
    classification: str  # from Judgement, passed through unchanged
    arr_tier: ArrTier
    action: RouteAction
    requires_human_approval: bool
    detail: str
    recipes_applied: tuple[Recipe, ...] = ()
    recipe_gaps: tuple[RecipeGap, ...] = ()


def _arr_tier(arr_usd: float | None) -> ArrTier:
    if arr_usd is None:
        return ArrTier.UNKNOWN
    return ArrTier.HIGH if arr_usd >= ARR_HIGH_THRESHOLD_USD else ArrTier.LOW


def _gaps_for(judgement: Judgement) -> tuple[RecipeGap, ...]:
    return tuple(
        RecipeGap(
            agent_id=judgement.agent_id,
            criterion_id=c.criterion_id,
            judge_cause_code=c.cause_code,
            evidence_quote=c.evidence_quote,
            evidence_config_field=c.evidence_config_field,
        )
        for c in judgement.unmapped_failures
    )


def route(judgement: Judgement, arr_usd: float | None) -> RoutingDecision:
    tier = _arr_tier(arr_usd)

    if judgement.classification == "healthy":
        action = RouteAction.NO_ACTION
        detail = "All criteria pass or unknown-with-no-failure; nothing to route."
        return RoutingDecision(
            agent_id=judgement.agent_id, classification=judgement.classification, arr_tier=tier,
            action=action, requires_human_approval=_TOUCHES_LIVE_AGENT[action], detail=detail,
        )

    if judgement.classification == "standard":
        recipes = tuple(c.recipe for c in judgement.failures if c.recipe is not None)
        all_self_serve = all(r.tier.value == "self_serve" for r in recipes)
        action = RouteAction.SELF_SERVE_FIX if all_self_serve else RouteAction.TARGETED_NUDGE
        detail = (
            f"{len(recipes)} failure(s), all mapped to known recipes "
            f"({', '.join(r.cause_code for r in recipes)}); "
            + ("all self-serve, batchable via docs links." if all_self_serve
               else "at least one recipe needs an account-specific nudge.")
        )
        return RoutingDecision(
            agent_id=judgement.agent_id, classification=judgement.classification, arr_tier=tier,
            action=action, requires_human_approval=_TOUCHES_LIVE_AGENT[action], detail=detail,
            recipes_applied=recipes,
        )

    # classification == "systemic"
    gaps = _gaps_for(judgement)
    if tier == ArrTier.HIGH:
        action = RouteAction.ESCALATE_TO_ENGINEER
        detail = (
            f"High-ARR account with {len(gaps)} unmapped failure(s) "
            f"({', '.join(g.criterion_id for g in gaps)}); needs an engineer, not automation."
        )
    else:
        action = RouteAction.NEAREST_GUIDANCE
        recipes = tuple(c.recipe for c in judgement.failures if c.recipe is not None)
        detail = (
            f"Low/unknown-ARR account with {len(gaps)} unmapped failure(s) "
            f"({', '.join(g.criterion_id for g in gaps)}); sending nearest available guidance "
            + (f"({len(recipes)} mapped failure(s) also present) " if recipes else "")
            + "and logging the gap(s) as candidates for the next recipe."
        )

    return RoutingDecision(
        agent_id=judgement.agent_id, classification=judgement.classification, arr_tier=tier,
        action=action, requires_human_approval=_TOUCHES_LIVE_AGENT[action], detail=detail,
        recipes_applied=tuple(c.recipe for c in judgement.failures if c.recipe is not None),
        recipe_gaps=gaps,
    )
