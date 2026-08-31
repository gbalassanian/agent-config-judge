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
import dataclasses
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from agent_config_judge.cheap_pass import score_agent
from agent_config_judge.judge import JudgeBackend, LiveJudgeBackend, RecordedJudgeBackend, run_judge
from agent_config_judge.models import AgentSnapshot, agent_snapshot_from_dict, agent_snapshot_to_dict
from agent_config_judge.pipeline import TriageResult, scan_portfolio, summarize, triage_agent
from agent_config_judge.router import route
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
        backend: JudgeBackend = LiveJudgeBackend(model=args.model)
    else:
        fixture = Path(args.fixture) if args.fixture else DEFAULT_RECORDED_JUDGEMENTS_PATH
        if not fixture.exists():
            print(f"error: recorded judgements fixture not found at {fixture}", file=sys.stderr)
            sys.exit(1)
        backend = RecordedJudgeBackend(fixture_path=str(fixture))

    if args.judge_cache:
        from agent_config_judge.judge_cache import CachedJudgeBackend
        backend = CachedJudgeBackend(backend=backend, cache_path=Path(args.judge_cache))

    return backend


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


def cmd_evaluate(args: argparse.Namespace) -> None:
    """On-demand triage of exactly one agent — the FDE/Deployment-Strategist
    entry point, distinct from `scan`'s "prepare a whole portfolio file
    first" flow. Reuses triage_agent() for the default path unchanged (so a
    given agent gets the identical verdict here as it would inside a full
    scan), and only bypasses it — narrowly, visibly — for --force-judge,
    which is an explicit human override of the cheap-pass gate, not a
    second code path that quietly duplicates the pipeline's logic.
    """
    if args.snapshot:
        snapshots = _load_snapshots(Path(args.snapshot))
        matches = [s for s in snapshots if s.agent_id == args.agent_id]
        if not matches:
            print(f"error: agent_id {args.agent_id!r} not found in {args.snapshot}", file=sys.stderr)
            sys.exit(1)
        snapshot = matches[0]
    else:
        from agent_config_judge.elevenlabs_client import ElevenLabsClient, build_agent_snapshot

        client = ElevenLabsClient()
        print(f"Fetching {args.agent_id} live from ElevenLabs...")
        raw_agent = client.get_agent(args.agent_id)
        raw_convs_meta = client.list_conversations(args.agent_id, page_size=args.sample_size)[: args.sample_size]
        raw_convs = [client.get_conversation(c["conversation_id"]) for c in raw_convs_meta]
        snapshot = build_agent_snapshot(raw_agent, raw_convs, arr_usd=args.arr_usd)
        print(f"  fetched: {len(raw_convs)} conversation(s) sampled")

    backend = _build_judge_backend(args)
    result = triage_agent(snapshot, backend)

    if args.force_judge and result.judgement is None:
        # The cheap pass didn't flag this agent, but a human explicitly
        # asked for the deep read anyway — a deliberate, visible override
        # of the gate, not a second scoring path. triage_agent() itself
        # stays untouched; this is the one place --force-judge is allowed
        # to reach past it, and only when there was no judge read already.
        print("  --force-judge: cheap pass did not flag this agent, but running the judge anyway.")
        judgement = run_judge(backend, snapshot.config, snapshot.conversations)
        routing = route(judgement, snapshot.arr_usd)
        result = dataclasses.replace(result, judgement=judgement, routing=routing)

    print()
    _print_triage_line(result)

    if args.output:
        out = {
            "agent_id": result.agent_id,
            "name": result.name,
            "cheap_pass_score": result.cheap_pass.score,
            "flagged": result.cheap_pass.flagged,
            "forced_flag": result.cheap_pass.forced_flag,
            "classification": result.routing.classification,
            "action": result.routing.action.value,
            "requires_human_approval": result.routing.requires_human_approval,
            "detail": result.routing.detail,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote report to {args.output}")


def _fetch_agents_concurrently(
    client: Any,
    agents: list[dict[str, Any]],
    sample_size: int,
    arr_map: dict[str, float],
    max_workers: int,
    checkpoint_path: Path | None,
    existing_snapshots: list[AgentSnapshot] | None = None,
) -> tuple[list[AgentSnapshot], list[tuple[str, str]]]:
    """Fetches + builds a snapshot for every agent in `agents`, bounded to
    at most `max_workers` requests in flight at once.

    This only parallelizes the ElevenLabs side (plain GETs, no cost per
    call) — the judge tier stays exactly as sequential as before, on
    purpose: paralleling paid LLM calls is a separate decision with a real
    cost/budget dimension to it, out of scope here.

    One agent being unreachable (even after ElevenLabsClient's own retries
    are exhausted — see elevenlabs_client.py) must not sink the whole
    fetch, concurrent or not. At real scale some non-zero number of agents
    WILL fail for reasons that have nothing to do with the other N-1, so a
    failure here is recorded and reported, never silently dropped and
    never fatal to the run — same guarantee the old sequential loop had,
    just now under a thread pool instead of a for loop.

    Every completed snapshot is appended and (if checkpoint_path is given)
    persisted under one lock, so a crash mid-run — worker thread or not —
    still loses nothing already fetched, and the on-disk file is never a
    half-written / torn JSON array from two threads writing at once.
    """
    from agent_config_judge.elevenlabs_client import ElevenLabsApiError, build_agent_snapshot

    snapshots: list[AgentSnapshot] = list(existing_snapshots or [])
    failed: list[tuple[str, str]] = []
    lock = threading.Lock()

    def _fetch_one(agent: dict[str, Any]) -> AgentSnapshot:
        agent_id = agent["agent_id"]
        raw_agent = client.get_agent(agent_id)
        raw_convs_meta = client.list_conversations(agent_id, page_size=sample_size)[:sample_size]
        raw_convs = [client.get_conversation(c["conversation_id"]) for c in raw_convs_meta]
        return build_agent_snapshot(raw_agent, raw_convs, arr_usd=arr_map.get(agent_id))

    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        future_to_agent = {pool.submit(_fetch_one, a): a for a in agents}
        for future in as_completed(future_to_agent):
            agent = future_to_agent[future]
            agent_id = agent["agent_id"]
            try:
                snap = future.result()
            except ElevenLabsApiError as e:
                print(f"  ! FAILED {agent_id} ({agent.get('name', '')}): {e}")
                failed.append((agent_id, str(e)))
                continue
            with lock:
                snapshots.append(snap)
                # Checkpointing to disk under the same lock that guards the
                # in-memory list — this rewrites the whole file each time
                # (O(n) per agent, O(n^2) total), which is fine for the
                # portfolio sizes this PoC targets but is exactly the kind
                # of thing a real 10k-agent production version would
                # replace with an append-only format instead of a full JSON
                # array rewrite.
                if checkpoint_path is not None:
                    _save_snapshots(snapshots, checkpoint_path)
            print(f"  fetched {agent_id} ({agent.get('name', '')}): {len(snap.conversations)} conversation(s) sampled")

    return snapshots, failed


def cmd_fetch_portfolio(args: argparse.Namespace) -> None:
    from agent_config_judge.elevenlabs_client import ElevenLabsClient

    arr_map: dict[str, float] = {}
    if args.arr_file:
        arr_map = json.loads(Path(args.arr_file).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    existing_snapshots: list[AgentSnapshot] = []
    already_fetched: set[str] = set()
    if args.resume and out_path.exists():
        existing_snapshots = _load_snapshots(out_path)
        already_fetched = {s.agent_id for s in existing_snapshots}
        print(f"--resume: {len(existing_snapshots)} agent(s) already in {out_path}, will skip those.")

    client = ElevenLabsClient()
    agents = client.list_agents()
    print(f"Found {len(agents)} agent(s) in the workspace.")

    to_fetch = [a for a in agents if a["agent_id"] not in already_fetched]
    if to_fetch:
        print(f"Fetching {len(to_fetch)} agent(s), up to {args.max_workers} concurrently...")
        snapshots, failed = _fetch_agents_concurrently(
            client, to_fetch, args.sample_size, arr_map, args.max_workers,
            checkpoint_path=out_path, existing_snapshots=existing_snapshots,
        )
    else:
        print("Nothing left to fetch.")
        snapshots, failed = existing_snapshots, []

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
    args.judge_cache = None
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
    p_scan.add_argument(
        "--judge-cache",
        help="Path to a judge-result cache file. When set, an agent whose config and sampled "
             "conversations are byte-identical to the last cached run for its agent_id reuses "
             "that cached judgement instead of calling the judge again — see judge_cache.py. "
             "Off by default: omit this flag to always call the judge fresh, exactly today's "
             "behavior.",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_eval = sub.add_parser(
        "evaluate",
        help="Triage exactly one agent on demand — fetch it live (or pick it out of an existing "
             "snapshot file) and triage it in one step, no portfolio file to prepare first. The "
             "FDE/Deployment-Strategist entry point, as opposed to scan's whole-portfolio flow.",
    )
    p_eval.add_argument("--agent-id", required=True, help="The agent to evaluate.")
    p_eval.add_argument(
        "--snapshot",
        help="Pick --agent-id out of this existing snapshot JSON instead of fetching live from "
             "ElevenLabs (no ELEVENLABS_API_KEY needed this way — useful for testing/demos).",
    )
    p_eval.add_argument("--sample-size", type=int, default=20, help="Max conversations sampled (live fetch only).")
    p_eval.add_argument("--arr-usd", type=float, help="This agent's ARR, for routing (live fetch only).")
    p_eval.add_argument("--backend", choices=["live", "recorded"], default="recorded")
    p_eval.add_argument("--fixture", help="Recorded-judgements JSON file (backend=recorded only).")
    p_eval.add_argument("--model", default="claude-sonnet-5", help="Model id (backend=live only).")
    p_eval.add_argument("--judge-cache", help="Same judge-result cache as scan — see judge_cache.py.")
    p_eval.add_argument(
        "--force-judge", action="store_true",
        help="Run the judge even if the cheap pass didn't flag this agent — a deliberate, visible "
             "override for a human who wants the deep read regardless. Off by default: the normal "
             "cheap-pass gate applies here exactly like it does in a full scan.",
    )
    p_eval.add_argument("--output", help="Write a JSON report here too.")
    p_eval.set_defaults(func=cmd_evaluate)

    p_fetch = sub.add_parser("fetch-portfolio", help="Pull real agents+conversations from ElevenLabs (needs ELEVENLABS_API_KEY).")
    p_fetch.add_argument("--out", required=True, help="Where to write the snapshot JSON.")
    p_fetch.add_argument("--sample-size", type=int, default=20, help="Max conversations sampled per agent.")
    p_fetch.add_argument("--arr-file", help="Optional JSON file mapping agent_id -> arr_usd.")
    p_fetch.add_argument("--resume", action="store_true",
                          help="Skip agent_ids already present in --out (from a prior partial/failed run).")
    p_fetch.add_argument("--max-workers", type=int, default=5,
                          help="Max concurrent in-flight ElevenLabs requests (default 5). This is a "
                               "guess, not a number pulled from ElevenLabs' real rate limits — tune it "
                               "against the actual API before relying on it at volume. Judge/Anthropic "
                               "calls are unaffected by this flag; they still run one at a time.")
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
