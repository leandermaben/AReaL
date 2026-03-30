#!/usr/bin/env python3
"""Phase 3: Enrich speaker IDs with names and roles using LLM.

Reads: processed/turns.jsonl, processed/events.jsonl
Writes: processed/speakers.jsonl

For each meeting, collects sample turns per speaker and asks the LLM to infer
names and roles from self-introductions and context.

Usage:
    python scripts/03_enrich_speakers.py \
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
    read_jsonl,
    truncate_to_budget,
    write_jsonl,
)

SYSTEM_PROMPT = (
    "You are an expert at identifying speakers in city council meetings. "
    "You infer names and roles from transcript context. "
    "Always respond with valid JSON and nothing else."
)

VALID_ROLES = [
    "chair",
    "councilmember",
    "staff",
    "presenter",
    "public_commenter",
    "unknown",
]

ENRICH_PROMPT = """\
Below are transcript excerpts from different speakers in a city council meeting.
For each speaker ID, infer:
- name: the speaker's name (if identifiable from self-introduction or being addressed by others). Use "Unknown" if not identifiable.
- role: one of {roles}

{speaker_excerpts}

Return a JSON array:
[{{"speaker_id": 0, "name": "Jane Smith", "role": "chair"}}, ...]"""

MAX_TURNS_PER_SPEAKER = 10
MAX_TOKENS_PER_TURN = 100  # ~100 tokens per turn excerpt


def format_speaker_excerpts(
    turns_by_speaker: dict[int, list[dict]],
) -> str:
    """Format sample turns per speaker for the prompt."""
    parts = []
    for speaker_id in sorted(turns_by_speaker):
        turns = turns_by_speaker[speaker_id][:MAX_TURNS_PER_SPEAKER]
        excerpts = []
        for t in turns:
            text = truncate_to_budget(t["text"], MAX_TOKENS_PER_TURN)
            excerpts.append(f'  "{text}"')
        parts.append(f"Speaker {speaker_id}:\n" + "\n".join(excerpts))
    return "\n\n".join(parts)


async def enrich_meeting_speakers(
    llm: AsyncLLMClient,
    meeting_id: str,
    turns_by_speaker: dict[int, list[dict]],
) -> list[dict]:
    """Enrich speakers for one meeting."""
    prompt = ENRICH_PROMPT.format(
        roles=VALID_ROLES,
        speaker_excerpts=format_speaker_excerpts(turns_by_speaker),
    )

    result = await llm.generate(
        prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=512,
        temperature=0.3,
    )

    if result is None:
        return []

    if isinstance(result, dict):
        result = result.get("speakers", [])
    if not isinstance(result, list):
        return []

    speakers = []
    for entry in result:
        if not isinstance(entry, dict):
            continue
        speaker_id = entry.get("speaker_id")
        if speaker_id is None:
            continue

        role = entry.get("role", "unknown")
        if role not in VALID_ROLES:
            role = "unknown"

        speakers.append(
            {
                "meeting_id": meeting_id,
                "speaker_id": int(speaker_id),
                "name": str(entry.get("name", "Unknown"))[:100],
                "role": role,
            }
        )

    # Add any speakers we missed (those the LLM didn't return)
    returned_ids = {s["speaker_id"] for s in speakers}
    for speaker_id in turns_by_speaker:
        if speaker_id not in returned_ids:
            speakers.append(
                {
                    "meeting_id": meeting_id,
                    "speaker_id": speaker_id,
                    "name": "Unknown",
                    "role": "unknown",
                }
            )

    return speakers


async def async_main(args: argparse.Namespace) -> None:
    turns = read_jsonl(args.output_dir / "processed" / "turns.jsonl")
    if not turns:
        print("No turns found. Run 01_preprocess.py first.")
        return

    print(f"Loaded {len(turns)} turns")

    # Check for existing output
    speakers_path = args.output_dir / "processed" / "speakers.jsonl"
    existing_meetings: set[str] = set()
    existing_speakers: list[dict] = []
    if speakers_path.is_file():
        existing_speakers = read_jsonl(speakers_path)
        existing_meetings = {s["meeting_id"] for s in existing_speakers}
        print(f"Resuming: {len(existing_meetings)} meetings already processed")

    # Group turns by meeting and speaker
    meetings: dict[str, dict[int, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for t in turns:
        mid = t["meeting_id"]
        if mid not in existing_meetings:
            meetings[mid][t["speaker_id"]].append(t)

    if not meetings:
        print("All meetings already processed.")
        return

    print(f"Processing {len(meetings)} meetings...")

    llm = AsyncLLMClient(
        base_url=args.vllm_url,
        model=args.model,
        concurrency=args.concurrency,
        system_prompt=SYSTEM_PROMPT,
    )

    # Process meetings concurrently in batches
    all_speakers = list(existing_speakers)
    meeting_items = list(meetings.items())
    batch_size = min(args.concurrency, 32)

    for batch_start in range(0, len(meeting_items), batch_size):
        if llm.server_down.is_set():
            print("\n[WARN] Server down — stopping early.")
            break

        batch = meeting_items[batch_start : batch_start + batch_size]
        tasks = [
            enrich_meeting_speakers(llm, mid, spk_turns)
            for mid, spk_turns in batch
        ]
        results = await asyncio.gather(*tasks)

        for speakers_list in results:
            all_speakers.extend(speakers_list)

        done = min(batch_start + batch_size, len(meeting_items))
        print(f"  [{done}/{len(meeting_items)}] meetings processed")

        # Checkpoint
        write_jsonl(speakers_path, all_speakers)

    write_jsonl(speakers_path, all_speakers)
    llm.print_stats()

    n_meetings = len({s["meeting_id"] for s in all_speakers})
    print(f"\nDone: {n_meetings} meetings, {len(all_speakers)} speaker entries")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: Enrich speakers")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_v2"),
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--concurrency", type=int, default=64)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
