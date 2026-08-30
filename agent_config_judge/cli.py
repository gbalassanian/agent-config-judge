"""Command-line entry points.

    agentjudge fetch-portfolio --out snapshot.json      # needs ELEVENLABS_API_KEY
    agentjudge scan --snapshot snapshot.json --backend live       # needs ANTHROPIC_API_KEY
    agentjudge scan --snapshot snapshot.json --backend recorded --fixture judgements.json
    agentjudge demo                                      # zero API keys — bundled fixtures

`demo` is the one command guaranteed to run with no keys at all: it scans
the real (redacted) portfolio snapshot this repo ships in fixtures/,
against the recorded judge outputs also shipped there. See README "Running
without API keys".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent_config_judge.judge import JudgeBackend, LiveJudgeBackend, RecordedJudgeBackend
from agent_config_judge.models import AgentSnapshot, agent_snapshot_from_dict, agent_snapshot_to_dict
from agent_config_judge.pipeline import TriageResult, scan_portfolio, summarize
from agent_config_judge.rubric import get_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "fixtures" / "real_portfolio_snapshot.json"
DEFAULT_RECORDED_JUDGEMENTS_PATH = REPO_ROOT / "fixtures" / "recorded_judgements.json"


def _load_snapshots(path: Path) -> list[AgentSnapshot]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [agent_snapshot_from_dict(d) for d in data]


def _save_snapshots(snapshots: list[AgentSnapshot], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([agent_snapshot_to_dict(s) for s in snapshots], f, indent=2, ensure_ascii=False)


def _print_triage_line(result: TriageResult) -> None:
    cp = result.cheap_pass
    flag_str = "FORCED" if cp.forced_flag else ("flagged" if cp.flagged else "-")
    print(
        f"  {result.agent_id:32s} {result.name[:28]:28s} "
        f"cheap={cp.score:5.1f} [{flag_str:7s}] -> {result.routing.classification:11s} "
        f"action={result.routing.action.value:18s} approval={result.routing.requires_human_approval}"
    )
    if result.judgement is not None:
        for c in result.judgement.failures:
            recipe_str = c.recipe.cause_code if c.recipe else "UNMAPPED"
            print(f"      - {c.criterion_id}: {recipe_str}")
    for gap in result.routing.recipe_gaps:
        print(f"      ! recipe gap: {gap.criterion_id} cause={gap.judge_cause_code!r}")


def _build_judge_backend(args: argparse.Namespace) -> JudgeBackend:
    if args.backend == "live":
        return LiveJudgeBackend(model=args.model)
    fixture = Path(args.fixture) if args.fixture else DEFAULT_RECORDED_JUDGEMENTS_PATH
    if not fixture.exists():
        print(f"error: recorded judgements fixture not found at {fixture}", file=sys.stderr)
        sys.exit(1)
    return RecordedJudgeBackend(fixture_path=str(fixture))


def cmd_scan(args: argparse.Namespace) -> None:
    snapshot_path = Path(args.snapshot)
    snapshots = _load_snapshots(snapshot_path)
    backend = _build_judge_backend(args)

    results, failures = scan_portfolio(snapshots, backend)
    summary = summarize(results)

    print(f"Scanned {summary.total_agents} agent(s); "
          f"{summary.flagged_count} flagged by the cheap pass -> judged.\n")
    for r in results:
        _print_triage_line(r)

    print("\n--- triage summary ---")
    print(f"classifications: {summary.classification_counts}")
    print(f"actions:         {summary.action_counts}")
    print(f"needs approval:  {summary.requires_approval_count}")
    print(f"recipe gaps:     {summary.recipe_gap_count}")
    if failures:
        # These agents are NOT counted anywhere above — they never reached
        # a classification at all, which is a different (and worse) state
        # than "healthy" or "not_flagged". Surfaced loudly, on purpose.
        print(f"\n{len(failures)} agent(s) FAILED to triage (no classification, needs a re-run):")
        for f in failures:
            print(f"  ! {f.agent_id:32s} {f.name[:28]:28s} {f.error}")

    if args.output:
        out = {
            "summary": {
                "total_agents": summary.total_agents,
                "flagged_count": summary.flagged_count,
                "judged_count": summary.judged_count,
                "classification_counts": summary.classification_counts,
                "action_counts": summary.action_counts,
                "requires_approval_count": summary.requires_approval_count,
                "recipe_gap_count": summary.recipe_gap_count,
                "failed_count": len(failures),
            },
            "agents": [
                {
                    "agent_id": r.agent_id,
                    "name": r.name,
                    "cheap_pass_score": r.cheap_pass.score,
                    "flagged": r.cheap_pass.flagged,
                    "forced_flag": r.cheap_pass.forced_flag,
                    "classification": r.routing.classification,
                    "action": r.routing.action.value,
                    "requires_human_approval": r.routing.requires_human_approval,
                    "detail": r.routing.detail,
                }
                for r in results
            ],
            "failed_agents": [
                {"agent_id": f.agent_id, "name": f.name, "error": f.error}
                for f in failures
            ],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote report to {args.output}")


def cmd_fetch_portfolio(args: argparse.Namespace) -> None:
    from agent_config_judge.elevenlabs_client import ElevenLabsApiError, ElevenLabsClient, build_agent_snapshot

    arr_map: dict[str, float] = {}
    if args.arr_file:
        arr_map = json.loads(Path(args.arr_file).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    snapshots: list[AgentSnapshot] = []
    already_fetched: set[str] = set()
    if args.resume and out_path.exists():
        snapshots = _load_snapshots(out_path)
        already_fetched = {s.agent_id for s in snapshots}
        print(f"--resume: {len(snapshots)} agent(s) already in {out_path}, will skip those.")

    client = ElevenLabsClient()
    agents = client.list_agents()
    print(f"Found {len(agents)} agent(s) in the workspace.")

    # One agent being unreachable (even after ElevenLabsClient's own
    # retries are exhausted — see elevenlabs_client.py) must not sink the
    # whole fetch. At real scale some non-zero number of agents WILL fail
    # for reasons that have nothing to do with the other N-1, so a failure
    # here is recorded and reported, never silently dropped and never fatal
    # to the run.
    failed: list[tuple[str, str]] = []
    for a in agents:
        agent_id = a["agent_id"]
        if agent_id in already_fetched:
            continue
        try:
            raw_agent = client.get_agent(agent_id)
            raw_convs_meta = client.list_conversations(agent_id, page_size=args.sample_size)[: args.sample_size]
            raw_convs = [client.get_conversation(c["conversation_id"]) for c in raw_convs_meta]
            snap = build_agent_snapshot(raw_agent, raw_convs, arr_usd=arr_map.get(agent_id))
        except ElevenLabsApiError as e:
            print(f"  ! FAILED {agent_id} ({a.get('name', '')}): {e}")
            failed.append((agent_id, str(e)))
            continue
        snapshots.append(snap)
        print(f"  fetched {agent_id} ({a.get('name', '')}): {len(raw_convs)} conversation(s) sampled")
        # Write after every agent, not just once at the end: a crash (or a
        # Ctrl-C) partway through loses nothing already fetched, and
        # --resume picks up exactly where it left off. This rewrites the
        # whole file each time — O(n) per agent, O(n^2) total — which is
        # fine for the portfolio sizes this PoC targets but is exactly the
        # kind of thing a real 10k-agent production version would replace
        # with an append-only format instead of a full JSON array rewrite.
        _save_snapshots(snapshots, out_path)

    print(f"\nWrote {len(snapshots)} agent snapshot(s) to {out_path}")
    if failed:
        print(f"\n{len(failed)} agent(s) FAILED and were skipped (not written):")
        for agent_id, err in failed:
            print(f"  - {agent_id}: {err}")
        print("Re-run with --resume to retry just the missing/failed agents.")
    print("\nNOTE: raw text is NOT redacted by this command. Run scripts/redact_snapshot.py "
          "before committing any fetched snapshot to a repo.")


def cmd_demo(args: argparse.Namespace) -> None:
    args.snapshot = str(DEFAULT_SNAPSHOT_PATH)
    args.backend = "recorded"
    args.fixture = str(DEFAULT_RECORDED_JUDGEMENTS_PATH)
    args.model = "unused"
    args.output = args.output or None
    cmd_scan(args)


def cmd_show_recipe(args: argparse.Namespace) -> None:
    recipe = get_recipe(args.cause_code)
    if recipe is None:
        print(f"'{args.cause_code}' is not in the catalog -> any agent citing it is systemic.")
        return
    print(json.dumps(
        {
            "cause_code": recipe.cause_code, "criterion_id": recipe.criterion_id,
            "title": recipe.title, "fix": recipe.fix, "tier": recipe.tier.value, "doc_url": recipe.doc_url,
        },
        indent=2,
    ))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agentjudge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="Cheap pass + judge + router over a snapshot file.")
    p_scan.add_argument("--snapshot", required=True, help="Path to a portfolio snapshot JSON file.")
    p_scan.add_argument("--backend", choices=["live", "recorded"], default="recorded")
    p_scan.add_argument("--fixture", help="Recorded-judgements JSON file (backend=recorded only).")
    p_scan.add_argument("--model", default="claude-sonnet-5", help="Model id (backend=live only).")
    p_scan.add_argument("--output", help="Write a JSON triage report here.")
    p_scan.set_defaults(func=cmd_scan)

    p_fetch = sub.add_parser("fetch-portfolio", help="Pull real agents+conversations from ElevenLabs (needs ELEVENLABS_API_KEY).")
    p_fetch.add_argument("--out", required=True, help="Where to write the snapshot JSON.")
    p_fetch.add_argument("--sample-size", type=int, default=20, help="Max conversations sampled per agent.")
    p_fetch.add_argument("--arr-file", help="Optional JSON file mapping agent_id -> arr_usd.")
    p_fetch.add_argument("--resume", action="store_true",
                          help="Skip agent_ids already present in --out (from a prior partial/failed run).")
    p_fetch.set_defaults(func=cmd_fetch_portfolio)

    p_demo = sub.add_parser("demo", help="Run the full pipeline on the bundled fixtures. No API keys needed.")
    p_demo.add_argument("--output", help="Write a JSON triage report here.")
    p_demo.set_defaults(func=cmd_demo)

    p_recipe = sub.add_parser("show-recipe", help="Look up (or fail to find) a cause_code in the catalog.")
    p_recipe.add_argument("cause_code")
    p_recipe.set_defaults(func=cmd_show_recipe)

    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
