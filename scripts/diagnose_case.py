"""Diagnostic tool: run the live judge on ONE golden-set case by name and
print its full per-criterion verdict — evidence quotes, cause_codes,
validator notes — not just the classification eval/run_eval.py reports.

Why this exists: run_eval.py's CaseResult only keeps criterion_id ->
cause_code, enough to score accuracy but not enough to see WHY a case
came back wrong. When a --backend live run disagrees with a golden case's
expected_classification, this is the next step — read what the judge
actually said, criterion by criterion, before deciding whether that's a
prompt fix or a rubric-calibration item. It costs one real Anthropic API
call per case named, not a re-run of the whole 19-case set.

Usage:
    PYTHONPATH=. python3 scripts/diagnose_case.py <case_name> [<case_name> ...]

    PYTHONPATH=. python3 scripts/diagnose_case.py \\
        agent_1101kynprgc8eky9mn86159whfr2 \\
        synthetic_grounding_trap_user_number \\
        synthetic_escalation_trap_ticket_tool

A "case name" is a golden case's agent_id for source="real" cases (the
raw ElevenLabs agent_id, e.g. agent_1101kynprgc8eky9mn86159whfr2) or its
synthetic_note-bearing agent_id for source="synthetic" cases (e.g.
synthetic_grounding_trap_user_number) — exactly what run_eval.py's
per-case report prints in its first column.

Needs ANTHROPIC_API_KEY. Each name given is one real, billed API call.
"""

from __future__ import annotations

import sys

from agent_config_judge.judge import LiveJudgeBackend, run_judge
from eval.golden_set import GoldenCase, load_golden_set


def find_case(cases: list[GoldenCase], name: str) -> GoldenCase | None:
    for c in cases:
        if c.snapshot.agent_id == name:
            return c
    return None


def diagnose(case: GoldenCase, backend: LiveJudgeBackend) -> None:
    judgement = run_judge(backend, case.snapshot.config, case.snapshot.conversations)

    print("=" * 78)
    print(f"{case.snapshot.agent_id}  (source={case.source})")
    print(f"  expected_classification={case.expected_classification!r}  "
          f"got={judgement.classification!r}  "
          f"{'OK' if judgement.classification == case.expected_classification else 'WRONG'}")
    if case.notes:
        print(f"  ground-truth notes: {case.notes}")
    if getattr(case.snapshot, "synthetic_note", None):
        print(f"  synthetic_note: {case.snapshot.synthetic_note}")

    print("\n  --- per-criterion ---")
    for criterion_id, cv in judgement.criteria.items():
        expected_cause = case.expected_failures.get(criterion_id)  # None if not expected to fail
        expected_to_fail = criterion_id in case.expected_failures
        flag = "  "
        if expected_to_fail and cv.verdict != "fail":
            flag = "MISSED FAIL (expected)"
        elif not expected_to_fail and cv.verdict == "fail":
            flag = "UNEXPECTED FAIL"
        print(f"  [{criterion_id:20s}] verdict={cv.verdict:8s}(raw={cv.raw_verdict:8s}) "
              f"cause_code={cv.cause_code!r:45s} expected_cause={expected_cause!r} {flag}")
        if cv.evidence_quote:
            print(f"      evidence_quote: {cv.evidence_quote!r}")
        if cv.evidence_config_field:
            print(f"      evidence_config_field: {cv.evidence_config_field!r}")

    if judgement.notes:
        print(f"\n  judge notes: {judgement.notes}")
    if judgement.validator_notes:
        print("\n  validator notes (what the validator changed and why):")
        for n in judgement.validator_notes:
            print(f"    - {n}")
    print("=" * 78)


def main() -> None:
    names = sys.argv[1:]
    if not names:
        print(__doc__)
        sys.exit(1)

    cases = load_golden_set()
    backend = LiveJudgeBackend()
    for name in names:
        case = find_case(cases, name)
        if case is None:
            available = ", ".join(sorted(c.snapshot.agent_id for c in cases))
            print(f"error: no golden case named {name!r}. Available: {available}", file=sys.stderr)
            sys.exit(1)
        diagnose(case, backend)


if __name__ == "__main__":
    main()
