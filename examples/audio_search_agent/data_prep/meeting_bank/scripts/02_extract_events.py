#!/usr/bin/env python3
"""Phase 2: Extract structured events from topic windows using LLM.

Reads: processed/windows.jsonl
Writes: processed/events.jsonl

For each window, prompts the LLM to identify discrete events (motions, votes,
presentations, public comments, etc.) with time bounds and speaker info.
Deduplicates events across overlapping windows.

Usage:
    python scripts/02_extract_events.py \
        --output-dir /path/to/output \
        --vllm-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

from shared import (
    AsyncLLMClient,
    fmt_time,
    read_jsonl,
    snap_to_3s,
    write_jsonl,
)

SYSTEM_PROMPT = (
    "You are an expert at analyzing city council meeting transcripts. "
    "You identify discrete events (motions, votes, presentations, public comments, etc.) "
    "with precise time boundaries. Always respond with valid JSON and nothing else."
)

EVENT_TYPES = [
    "agenda_transition",
    "presentation",
    "public_comment",
    "motion",
    "vote",
    "vote_result",
    "proclamation",
    "question",
    "response",
    "discussion",
    "adjournment",
    "other",
]

EXTRACT_PROMPT = """\
Given this transcript window from a city council meeting ({start} to {end}):

{formatted_turns}

Extract all discrete events. For each event, output:
- event_type: one of {event_types}
- label: short canonical name (e.g. "Resolution 14 — airport contract")
- speakers: list of speaker IDs (integers) involved
- start_sec: start time in seconds
- end_sec: end time in seconds
- summary: 1-2 sentence description

Return a JSON array. If no clear events, return [].
Example: [{{"event_type": "motion", "label": "Resolution 14", "speakers": [1, 3], "start_sec": 120.0, "end_sec": 180.0, "summary": "Council member moved to approve Resolution 14."}}]"""


def format_window_turns(window: dict) -> str:
    """Format a window's text with speaker/time annotations."""
    # The window text is a flat concatenation — we'll present it as-is with time context
    return window["text"]


def _events_overlap(a: dict, b: dict, label_threshold: float = 0.6) -> bool:
    """Check if two events are duplicates (overlapping time + similar label)."""
    # Time overlap check
    time_overlap = min(a["end_sec"], b["end_sec"]) - max(a["start_sec"], b["start_sec"])
    if time_overlap <= 0:
        return False

    # Label similarity (simple word overlap)
    words_a = set(a["label"].lower().split())
    words_b = set(b["label"].lower().split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    union = len(words_a | words_b)
    similarity = overlap / union if union > 0 else 0

    return similarity >= label_threshold and a["event_type"] == b["event_type"]


def deduplicate_events(events: list[dict]) -> list[dict]:
    """Merge duplicate events from overlapping windows."""
    if not events:
        return []

    events = sorted(events, key=lambda e: e["start_sec"])
    merged = [events[0]]

    for ev in events[1:]:
        found_dup = False
        for i, existing in enumerate(merged):
            if _events_overlap(existing, ev):
                # Merge: keep union of time bounds, combine speakers
                merged[i] = {
                    **existing,
                    "start_sec": min(existing["start_sec"], ev["start_sec"]),
                    "end_sec": max(existing["end_sec"], ev["end_sec"]),
                    "speakers": sorted(
                        set(existing["speakers"]) | set(ev["speakers"])
                    ),
                }
                found_dup = True
                break
        if not found_dup:
            merged.append(ev)

    return merged


async def extract_events_for_window(
    llm: AsyncLLMClient, window: dict
) -> list[dict]:
    """Extract events from a single window."""
    prompt = EXTRACT_PROMPT.format(
        start=fmt_time(window["start_sec"]),
        end=fmt_time(window["end_sec"]),
        formatted_turns=format_window_turns(window),
        event_types=EVENT_TYPES,
    )

    result = await llm.generate(
        prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=1024,
    )

    if result is None:
        return []

    # Handle both list result and dict with events key
    if isinstance(result, dict):
        result = result.get("events", [])
    if not isinstance(result, list):
        return []

    events = []
    for ev in result:
        if not isinstance(ev, dict):
            continue
        if not all(k in ev for k in ("event_type", "label", "start_sec", "end_sec")):
            continue

        # Validate and clean
        event_type = ev["event_type"]
        if event_type not in EVENT_TYPES:
            event_type = "other"

        try:
            start_sec, end_sec = snap_to_3s(float(ev["start_sec"]), float(ev["end_sec"]))
        except (ValueError, TypeError):
            continue

        # Clamp to window bounds (LLM sometimes hallucinates times)
        start_sec = max(start_sec, window["start_sec"])
        end_sec = min(end_sec, window["end_sec"])
        if end_sec <= start_sec:
            continue

        speakers = ev.get("speakers", [])
        if not isinstance(speakers, list):
            speakers = []
        speakers = [int(s) for s in speakers if isinstance(s, (int, float))]

        events.append(
            {
                "meeting_id": window["meeting_id"],
                "event_type": event_type,
                "label": str(ev.get("label", ""))[:200],
                "speakers": speakers,
                "start_sec": round(start_sec, 3),
                "end_sec": round(end_sec, 3),
                "summary": str(ev.get("summary", ""))[:500],
            }
        )

    return events


async def async_main(args: argparse.Namespace) -> None:
    windows = read_jsonl(args.output_dir / "processed" / "windows.jsonl")
    if not windows:
        print("No windows found. Run 01_preprocess.py first.")
        return

    print(f"Loaded {len(windows)} windows")

    # Check for existing output (resumability)
    events_path = args.output_dir / "processed" / "events.jsonl"
    existing_meetings: set[str] = set()
    existing_events: list[dict] = []
    if events_path.is_file():
        existing_events = read_jsonl(events_path)
        existing_meetings = {e["meeting_id"] for e in existing_events}
        print(f"Resuming: {len(existing_meetings)} meetings already processed")

    # Filter to unprocessed windows
    windows = [w for w in windows if w["meeting_id"] not in existing_meetings]
    if not windows:
        print("All windows already processed.")
        return

    print(f"Processing {len(windows)} remaining windows...")

    llm = AsyncLLMClient(
        base_url=args.vllm_url,
        model=args.model,
        concurrency=args.concurrency,
        system_prompt=SYSTEM_PROMPT,
    )

    # Group windows by meeting for deduplication
    windows_by_meeting: dict[str, list[dict]] = defaultdict(list)
    for w in windows:
        windows_by_meeting[w["meeting_id"]].append(w)

    all_events = list(existing_events)
    meetings_done = 0

    for meeting_id, meeting_windows in windows_by_meeting.items():
        if llm.server_down.is_set():
            print("\n[WARN] Server down — stopping early.")
            break

        # Process all windows for this meeting concurrently
        tasks = [extract_events_for_window(llm, w) for w in meeting_windows]
        results = await asyncio.gather(*tasks)

        # Flatten and deduplicate
        raw_events = [ev for batch in results for ev in batch]
        deduped = deduplicate_events(raw_events)

        # Assign event IDs
        for i, ev in enumerate(deduped):
            ev["event_id"] = f"{meeting_id}:e{i}"

        all_events.extend(deduped)
        meetings_done += 1

        if meetings_done % 20 == 0:
            print(
                f"  [{meetings_done}/{len(windows_by_meeting)}] "
                f"{meeting_id}: {len(deduped)} events from {len(meeting_windows)} windows"
            )
            # Checkpoint
            write_jsonl(events_path, all_events)

    write_jsonl(events_path, all_events)
    llm.print_stats()

    n_meetings = len({e["meeting_id"] for e in all_events})
    print(f"\nDone: {n_meetings} meetings, {len(all_events)} events total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2: Extract events")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/nvme/bffw/lmaben/long_speech/meeting_bank_v2"),
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
