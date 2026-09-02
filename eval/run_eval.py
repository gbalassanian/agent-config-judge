"""The eval harness. Run: PYTHONPATH=. python3 eval/run_eval.py

Defaults to the recorded backend (no API key needed, reproducible run to
run) — pass `--backend live` for a real `ANTHROPIC_API_KEY` call per case
instead. Recorded is not a shortcut standing in for live: see README's "How
the recorded judgements were produced" for exactly why a `--backend live`
run is the one number here that isn't self-graded, and "Eval results" for
what changed the first time this actually ran against a live judge.

Reports three separate scorecards because the three tiers fail in
different directions and blending them into one number would hide that:

  - cheap pass: recall FIRST (a broken agent that never gets flagged is
    invisible to the whole system), precision second (over-flagging just
    costs a judge call, which is the design's whole point).
  - judge: classification accuracy, plus precision/recall on the specific
    failing criteria it names (not just "did it flag something").
  - judge false-positive rate: of the agents that are actually healthy,
    what fraction does the judge wrongly call standard/systemic? This is
    the number checked against the 30% recalibration threshold — it's a
    judge-tier number, not a cheap-pass number, because over-flagging is
    the cheap pass's JOB; falsely accusing a healthy agent is the judge's
    failure mode to avoid.

Every number below comes from the 19-case golden set in eval/golden_set.py
— 5 real portfolio agents, 14 synthetic. That is a small n. This script
does not round that off or hide it; the printed report says so, and
README repeats it before quoting any of these numbers.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_config_judge.cheap_pass import score_agent
from agent_config_judge.judge import LiveJudgeBackend, RecordedJudgeBackend, run_judge
from agent_config_judge.rubric import CRITERION_ORDER
from eval.golden_set import GoldenCase, load_golden_set

REPO_ROOT = Path(__file__).resolve().parent.parent
RECORDED_JUDGEMENTS_PATH = REPO_ROOT / "fixtures" / "recorded_judgements.json"

# The threshold from the task brief: above this, the rubric needs
# recalibrating rather than the judge being trusted as-is.
JUDGE_FALSE_POSITIVE_THRESHOLD = 0.30


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


@dataclass
class CaseResult:
    case: GoldenCase
    cheap_flagged: bool
    judge_classification: str
    judge_failures: dict[str, str | None]  # criterion_id -> cause_code (None if unmapped)


def run_all_cases(cases: list[GoldenCase], backend: Any = None) -> list[CaseResult]:
    if backend is None:
        backend = RecordedJudgeBackend(fixture_path=str(RECORDED_JUDGEMENTS_PATH))
    results = []
    for case in cases:
        cp = score_agent(case.snapshot.config, case.snapshot.metrics)
        judgement = run_judge(backend, case.snapshot.config, case.snapshot.conversations)
        failures = {c.criterion_id: c.cause_code for c in judgement.failures}
        results.append(CaseResult(case=case, cheap_flagged=cp.flagged,
                                   judge_classification=judgement.classification,
                                   judge_failures=failures))
    return results


def score_cheap_pass(results: list[CaseResult]) -> dict:
    tp = fp = tn = fn = 0
    for r in results:
        want, got = r.case.should_flag, r.cheap_flagged
        if want and got:
            tp += 1
        elif want and not got:
            fn += 1
        elif not want and got:
            fp += 1
        else:
            tn += 1
    return {
        "n": len(results), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
    }


def score_judge_classification(results: list[CaseResult]) -> dict:
    correct = sum(1 for r in results if r.judge_classification == r.case.expected_classification)
    healthy_cases = [r for r in results if r.case.expected_classification == "healthy"]
    healthy_fp = sum(1 for r in healthy_cases if r.judge_classification != "healthy")
    return {
        "n": len(results),
        "accuracy": _rate(correct, len(results)),
        "n_healthy_ground_truth": len(healthy_cases),
        "false_positive_rate": _rate(healthy_fp, len(healthy_cases)),
        "false_positive_count": healthy_fp,
        "exceeds_threshold": (
            _rate(healthy_fp, len(healthy_cases)) is not None
            and _rate(healthy_fp, len(healthy_cases)) > JUDGE_FALSE_POSITIVE_THRESHOLD
        ),
    }


def score_judge_failures(results: list[CaseResult]) -> dict:
    tp = fp = fn = 0
    cause_matches = cause_checkable = 0
    for r in results:
        expected = r.case.expected_failures
        actual = r.judge_failures
        for cid in CRITERION_ORDER:
            if cid in expected and cid in actual:
                tp += 1
                if expected[cid] is not None:  # a real (mapped) cause expected — check it matches
                    cause_checkable += 1
                    if actual[cid] == expected[cid]:
                        cause_matches += 1
            elif cid in expected and cid not in actual:
                fn += 1
            elif cid not in expected and cid in actual:
                fp += 1
    return {
        "criterion_level_tp": tp, "criterion_level_fp": fp, "criterion_level_fn": fn,
        "recall": _rate(tp, tp + fn),
        "precision": _rate(tp, tp + fp),
        "cause_code_accuracy_given_correct_criterion": _rate(cause_matches, cause_checkable),
        "cause_code_checkable_n": cause_checkable,
    }


def print_report(results: list[CaseResult]) -> None:
    n_real = sum(1 for r in results if r.case.source == "real")
    n_synth = sum(1 for r in results if r.case.source == "synthetic")

    print("=" * 78)
    print(f"EVAL: {len(results)} golden-set cases ({n_real} real portfolio agents, {n_synth} synthetic).")
    print("n is small — see README for what a larger sample would need to say more.")
    print("=" * 78)

    print("\n--- per-case ---")
    for r in results:
        c = r.case
        flag_ok = "ok" if r.cheap_flagged == c.should_flag else "MISS"
        class_ok = "ok" if r.judge_classification == c.expected_classification else "WRONG"
        print(f"  [{r.case.source:9s}] {c.snapshot.agent_id:42s} "
              f"flag(want={c.should_flag!s:5s} got={r.cheap_flagged!s:5s} {flag_ok:4s}) "
              f"class(want={c.expected_classification:9s} got={r.judge_classification:9s} {class_ok})")

    cp_scores = score_cheap_pass(results)
    print("\n--- cheap pass (recall first, precision second — over-flagging is by design) ---")
    print(f"  n={cp_scores['n']}  TP={cp_scores['tp']} FP={cp_scores['fp']} TN={cp_scores['tn']} FN={cp_scores['fn']}")
    print(f"  recall:    {cp_scores['recall']:.0%}" if cp_scores["recall"] is not None else "  recall: n/a")
    print(f"  precision: {cp_scores['precision']:.0%}" if cp_scores["precision"] is not None else "  precision: n/a")

    jc_scores = score_judge_classification(results)
    print("\n--- judge: classification accuracy + false-positive rate on healthy agents ---")
    print(f"  n={jc_scores['n']}  accuracy={jc_scores['accuracy']:.0%}")
    print(f"  of {jc_scores['n_healthy_ground_truth']} agents that are actually healthy, "
          f"judge wrongly flagged {jc_scores['false_positive_count']} "
          f"({jc_scores['false_positive_rate']:.0%}) as standard/systemic")
    verdict = "EXCEEDS" if jc_scores["exceeds_threshold"] else "within"
    print(f"  {verdict} the {JUDGE_FALSE_POSITIVE_THRESHOLD:.0%} recalibration threshold")

    jf_scores = score_judge_failures(results)
    print("\n--- judge: precision/recall on the specific failing criteria it names ---")
    print(f"  criterion-level TP={jf_scores['criterion_level_tp']} "
          f"FP={jf_scores['criterion_level_fp']} FN={jf_scores['criterion_level_fn']}")
    print(f"  recall:    {jf_scores['recall']:.0%}" if jf_scores["recall"] is not None else "  recall: n/a")
    print(f"  precision: {jf_scores['precision']:.0%}" if jf_scores["precision"] is not None else "  precision: n/a")
    if jf_scores["cause_code_checkable_n"]:
        print(f"  of {jf_scores['cause_code_checkable_n']} correctly-identified mapped failures, "
              f"cause_code matched ground truth in "
              f"{jf_scores['cause_code_accuracy_given_correct_criterion']:.0%}")

    print("\n" + "=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=["recorded", "live"], default="recorded",
        help="recorded (default): replay fixtures/recorded_judgements.json, no API key needed. "
             "live: a real ANTHROPIC_API_KEY call per case (19 calls total) — the one number in "
             "this report that isn't self-graded; see README's 'Eval results'.",
    )
    parser.add_argument("--model", default="claude-sonnet-5", help="Model id (--backend live only).")
    args = parser.parse_args()

    if args.backend == "live":
        backend: Any = LiveJudgeBackend(model=args.model)
        print(f"--backend live: {len(load_golden_set())} real Anthropic API calls follow (model={args.model}).")
    else:
        backend = RecordedJudgeBackend(fixture_path=str(RECORDED_JUDGEMENTS_PATH))

    results = run_all_cases(load_golden_set(), backend=backend)
    print_report(results)


if __name__ == "__main__":
    main()
