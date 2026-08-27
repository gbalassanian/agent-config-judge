"""Calibration tool for cheap_pass.LATENCY_BAND_MS_BY_CHANNEL — NOT a fix.

Every band in that table is a placeholder guess (see its comment in
cheap_pass.py): "not measured against real channel baselines yet." The
honest way to replace a guess with a real number is to compute it from
enough real observed TTFB, not to swap one guess for a different-looking
one — so this script only ever reads a snapshot file and reports what the
data supports; it never writes to cheap_pass.py itself. Reviewing and
editing LATENCY_BAND_MS_BY_CHANNEL by hand, using this output plus
judgment about what changed between samples, stays a human decision.

Usage:
    PYTHONPATH=. python3 scripts/calibrate_latency_bands.py fixtures/real_portfolio_snapshot.json

Deliberately conservative about small samples: below MIN_SAMPLES_PER_CHANNEL
observations for a channel, it reports "insufficient data" instead of a
number — a percentile computed from a handful of turns is not a baseline,
it's noise wearing a baseline's clothes, and printing it as a suggestion
would be exactly the kind of confidently-wrong heuristic this project
tries not to build anywhere else.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from agent_config_judge.cheap_pass import DEFAULT_LATENCY_BAND_MS, LATENCY_BAND_MS_BY_CHANNEL
from agent_config_judge.models import agent_snapshot_from_dict

MIN_SAMPLES_PER_CHANNEL = 30


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        raise ValueError("empty sample")
    idx = min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def collect_ttfb_by_channel(snapshot_path: Path) -> dict[str, list[float]]:
    raw = json.loads(snapshot_path.read_text())
    agents = raw if isinstance(raw, list) else raw.get("agents", [])
    by_channel: dict[str, list[float]] = defaultdict(list)
    for agent_d in agents:
        snap = agent_snapshot_from_dict(agent_d)
        for conv in snap.conversations:
            for turn in conv.turns:
                if turn.ttfb_ms is not None:
                    by_channel[conv.channel].append(turn.ttfb_ms)
    return by_channel


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <snapshot.json>", file=sys.stderr)
        raise SystemExit(2)

    by_channel = collect_ttfb_by_channel(Path(sys.argv[1]))
    all_channels = sorted(set(by_channel) | set(LATENCY_BAND_MS_BY_CHANNEL))

    print(f"{'channel':<24}{'n':>6}{'p50 (ms)':>12}{'p90 (ms)':>12}{'current band':>16}   verdict")
    for channel in all_channels:
        vals = sorted(by_channel.get(channel, []))
        current = LATENCY_BAND_MS_BY_CHANNEL.get(channel, DEFAULT_LATENCY_BAND_MS)
        if len(vals) < MIN_SAMPLES_PER_CHANNEL:
            verdict = f"insufficient data (n<{MIN_SAMPLES_PER_CHANNEL}) — keep placeholder"
            p50 = p90 = float("nan")
        else:
            p50, p90 = _percentile(vals, 50), _percentile(vals, 90)
            verdict = "review: consider band ~= observed p90" if abs(p90 - current) > 0.15 * current else "placeholder roughly matches observed p90"
        print(f"{channel:<24}{len(vals):>6}{p50:>12.0f}{p90:>12.0f}{current:>16.0f}   {verdict}")

    print(
        "\nNo channel here is auto-applied to cheap_pass.py. Edit "
        "LATENCY_BAND_MS_BY_CHANNEL by hand once a channel clears the "
        "sample-size bar consistently across more than one snapshot."
    )


if __name__ == "__main__":
    main()
