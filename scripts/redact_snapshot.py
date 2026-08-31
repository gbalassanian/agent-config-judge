#!/usr/bin/env python3
"""Redact a fetched portfolio snapshot before committing it anywhere.

`agentjudge fetch-portfolio` pulls real system prompts, real tool endpoint
URLs, and real conversation transcripts straight from a live ElevenLabs
workspace. None of that is this repo's to publish — it's the customer's (or
your own business's) content, not example data. This script exists because
that gap was real: `fetch-portfolio`'s own printed note has told people to
run "scripts/redact_snapshot.py" since before this file existed.

What gets redacted, and why each one is handled differently:

  - `config.system_prompt`: replaced with filler text of the EXACT SAME
    LENGTH. Length matters here, not content — cheap_pass.py's
    system_prompt check (MIN_SYSTEM_PROMPT_CHARS) is a length threshold,
    so silently shortening or lengthening a redacted prompt would change
    what the snapshot demonstrates about the pipeline without anyone
    noticing. Content is discarded; the one number the scorer actually
    reads is preserved exactly.
  - `config.tools[].detail`: replaced with a fixed placeholder, no length
    constraint. Nothing in cheap_pass or the judge scores on this field's
    content (human_handoff keys off `system_tool_type`, not `detail`) —
    it exists here purely as a citable string, and today that string is
    often a live webhook URL. Safe to blank outright.
  - `conversations[].turns[].text`: replaced with a fixed placeholder by
    default — this is the most privacy-sensitive part (real transcript
    content) and nothing about cheap_pass's *aggregate* metrics
    (turn counts, tool_calls, ttfb_ms, sentiment labels) depends on the
    text itself, only on structure that's computed once at fetch time
    and stored separately in `metrics`. See --keep-conversations below
    for the one real reason to skip this.

TRADEOFF, stated plainly: redacting turn text means the output file can no
longer be paired with a recorded_judgements.json that cites real evidence
quotes from these transcripts — validate_judge_output's evidence-quote
check searches for the judge's quoted text verbatim inside the
conversation, and a redacted transcript won't contain it. Pass
--keep-conversations for a snapshot you intend to use that way (e.g.
extending this repo's own golden set) — understand that doing so leaves
real conversation text in the output.

What is NOT touched, because none of it is prompt/transcript content:
tool types, KB attachment (ids/usage_mode — names are left as-is; rename
by hand if a KB doc name itself is sensitive), channels, tool_calls/
is_error, ttfb_ms, sentiment labels, and every count in `metrics` — a
redacted snapshot still exercises the exact same scoring code paths a
real one would.

Usage:
    python3 scripts/redact_snapshot.py --in snapshot.json --out snapshot.redacted.json
    python3 scripts/redact_snapshot.py --in snapshot.json --out snapshot.redacted.json --keep-conversations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

_FILLER = "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
_FREE_TEXT_PLACEHOLDER = "[REDACTED]"


def _redact_preserving_length(text: str) -> str:
    """Same character count, none of the real content — see module docstring
    on why length (not content) is what needs to survive here."""
    if not text:
        return text
    repeats = len(text) // len(_FILLER) + 1
    return (_FILLER * repeats)[: len(text)]


def redact_agents(agents: list[dict], redact_conversations: bool) -> tuple[list[dict], dict[str, int]]:
    """Redacts in place (agents is mutated AND returned) and reports counts
    of what was touched, so the CLI can print a real summary rather than
    just "done"."""
    counts = {"system_prompts": 0, "tool_details": 0, "conversation_turns": 0}
    for agent in agents:
        config = agent.get("config", {})
        prompt = config.get("system_prompt", "")
        if prompt:
            config["system_prompt"] = _redact_preserving_length(prompt)
            counts["system_prompts"] += 1
        for tool in config.get("tools", []):
            if tool.get("detail"):
                tool["detail"] = _FREE_TEXT_PLACEHOLDER
                counts["tool_details"] += 1
        if redact_conversations:
            for conv in agent.get("conversations", []):
                for turn in conv.get("turns", []):
                    if turn.get("text"):
                        turn["text"] = _FREE_TEXT_PLACEHOLDER
                        counts["conversation_turns"] += 1
    return agents, counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--in", dest="in_path", required=True, help="Snapshot JSON to redact.")
    parser.add_argument("--out", dest="out_path", required=True, help="Where to write the redacted copy.")
    parser.add_argument(
        "--keep-conversations", action="store_true",
        help="Skip transcript redaction. Only for a snapshot you intend to pair with a "
             "recorded_judgements.json that cites real evidence quotes from these same "
             "transcripts — understand this leaves real conversation text in the output.",
    )
    args = parser.parse_args()

    with open(args.in_path, encoding="utf-8") as f:
        agents = json.load(f)

    redacted, counts = redact_agents(agents, redact_conversations=not args.keep_conversations)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2, ensure_ascii=False)

    print(f"Redacted {len(agents)} agent(s): "
          f"{counts['system_prompts']} system prompt(s), {counts['tool_details']} tool detail field(s)"
          + (f", {counts['conversation_turns']} conversation turn(s)" if not args.keep_conversations else ""))
    if args.keep_conversations:
        print("--keep-conversations: conversation transcript text was left UNREDACTED. "
              "Only commit this if you've checked it doesn't contain anything real.")
    else:
        print("Conversation text redacted — this file can no longer be paired with a "
              "recorded_judgements.json that cites real evidence quotes from it.")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
