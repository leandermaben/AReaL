#!/usr/bin/env python3
"""Phase 1: Preprocess raw transcripts into segments, turns, and topic windows.

No LLM needed. Reads raw transcript JSONs, produces:
  - processed/segments.jsonl  — one row per segment
  - processed/turns.jsonl     — merged consecutive same-speaker segments
  - processed/windows.jsonl   — sliding topic windows (~2-5 min, 50% overlap)

Usage:
    python scripts/01_preprocess.py \
        --data-root /path/to/raw \
        --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
from pathlib import Path

from shared import (
    US_TO_S,
    count_tokens,
    discover_pairs,
    load_raw_segments,
    seg_bounds_s,
    seg_text,
    truncate_to_budget,
    write_jsonl,
)

# ── Window parameters ─────────────────────────────────────────────────────────

WINDOW_TARGET_SEC = 180  # target 3 minutes per window
WINDOW_MIN_SEC = 120  # minimum 2 minutes
WINDOW_MAX_SEC = 300  # maximum 5 minutes
WINDOW_OVERLAP_FRAC = 0.5  # 50% overlap
WINDOW_TOKEN_BUDGET = 2000  # max tokens per window text


def build_segments(
    raw_segs: list[dict], meeting_id: str, city: str
) -> list[dict]:
    """Convert raw transcript segments to structured rows."""
    segments = []
    for idx, seg in enumerate(raw_segs):
        start_s, end_s = seg_bounds_s(seg)
        text = seg_text(seg)
        if not text:
            continue
        segments.append(
            {
                "meeting_id": meeting_id,
                "seg_idx": idx,
                "speaker_id": seg.get("speaker", -1),
                "start_sec": round(start_s, 3),
                "end_sec": round(end_s, 3),
                "text": text,
                "word_count": len(text.split()),
                "city": city,
            }
        )
    return segments


def build_turns(segments: list[dict], meeting_id: str) -> list[dict]:
    """Merge consecutive segments from the same speaker into turns."""
    if not segments:
        return []

    turns = []
    current_speaker = segments[0]["speaker_id"]
    current_segs = [segments[0]]

    for seg in segments[1:]:
        if seg["speaker_id"] == current_speaker:
            current_segs.append(seg)
        else:
            turns.append(_make_turn(current_segs, len(turns), meeting_id))
            current_speaker = seg["speaker_id"]
            current_segs = [seg]

    turns.append(_make_turn(current_segs, len(turns), meeting_id))
    return turns


def _make_turn(segs: list[dict], turn_idx: int, meeting_id: str) -> dict:
    text = " ".join(s["text"] for s in segs)
    return {
        "meeting_id": meeting_id,
        "turn_idx": turn_idx,
        "speaker_id": segs[0]["speaker_id"],
        "start_sec": segs[0]["start_sec"],
        "end_sec": segs[-1]["end_sec"],
        "text": text,
        "seg_indices": [s["seg_idx"] for s in segs],
    }


def build_windows(
    segments: list[dict], meeting_id: str
) -> list[dict]:
    """Build sliding topic windows with overlap and token budgeting."""
    if not segments:
        return []

    total_dur = segments[-1]["end_sec"] - segments[0]["start_sec"]
    if total_dur < WINDOW_MIN_SEC:
        # Entire meeting fits in one window
        text = " ".join(s["text"] for s in segments)
        text = truncate_to_budget(text, WINDOW_TOKEN_BUDGET)
        return [
            {
                "meeting_id": meeting_id,
                "window_idx": 0,
                "start_sec": segments[0]["start_sec"],
                "end_sec": segments[-1]["end_sec"],
                "text": text,
                "seg_indices": [s["seg_idx"] for s in segments],
                "token_count": count_tokens(text),
            }
        ]

    # Sliding window with overlap
    step_sec = WINDOW_TARGET_SEC * (1 - WINDOW_OVERLAP_FRAC)
    windows = []
    window_start = segments[0]["start_sec"]
    meeting_end = segments[-1]["end_sec"]

    while window_start < meeting_end:
        window_end = window_start + WINDOW_TARGET_SEC

        # Collect segments in this window
        win_segs = [
            s
            for s in segments
            if s["start_sec"] < window_end and s["end_sec"] > window_start
        ]
        if not win_segs:
            window_start += step_sec
            continue

        text = " ".join(s["text"] for s in win_segs)
        text = truncate_to_budget(text, WINDOW_TOKEN_BUDGET)

        windows.append(
            {
                "meeting_id": meeting_id,
                "window_idx": len(windows),
                "start_sec": round(max(window_start, win_segs[0]["start_sec"]), 3),
                "end_sec": round(min(window_end, win_segs[-1]["end_sec"]), 3),
                "text": text,
                "seg_indices": [s["seg_idx"] for s in win_segs],
                "token_count": count_tokens(text),
            }
        )

        window_start += step_sec

    return windows


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Preprocess transcripts")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/work/hdd/bbjs/lmaben/speech/long_speech/temp_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/nvme/bffw/lmaben/long_speech/meeting_bank_v2"),
    )
    parser.add_argument("--cities", nargs="+", metavar="CITY")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    pairs = discover_pairs(args.data_root, args.cities)
    if not pairs:
        print(f"No paired files found under {args.data_root}")
        return
    if args.limit:
        pairs = pairs[: args.limit]

    print(f"Processing {len(pairs)} audio+transcript pairs...")

    all_segments: list[dict] = []
    all_turns: list[dict] = []
    all_windows: list[dict] = []

    for i, (mp3, transcript, city) in enumerate(pairs):
        meeting_id = f"{city}/{mp3.stem}"
        raw_segs = load_raw_segments(transcript)
        if not raw_segs:
            print(f"  [{i+1}/{len(pairs)}] {meeting_id} — empty transcript, skipping")
            continue

        segments = build_segments(raw_segs, meeting_id, city)
        turns = build_turns(segments, meeting_id)
        windows = build_windows(segments, meeting_id)

        all_segments.extend(segments)
        all_turns.extend(turns)
        all_windows.extend(windows)

        if (i + 1) % 50 == 0 or i == len(pairs) - 1:
            print(
                f"  [{i+1}/{len(pairs)}] {meeting_id}: "
                f"{len(segments)} segs, {len(turns)} turns, {len(windows)} windows"
            )

    processed_dir = args.output_dir / "processed"
    write_jsonl(processed_dir / "segments.jsonl", all_segments)
    write_jsonl(processed_dir / "turns.jsonl", all_turns)
    write_jsonl(processed_dir / "windows.jsonl", all_windows)

    n_meetings = len({s["meeting_id"] for s in all_segments})
    print(
        f"\nDone: {n_meetings} meetings, "
        f"{len(all_segments)} segments, "
        f"{len(all_turns)} turns, "
        f"{len(all_windows)} windows"
    )
    print(f"Output: {processed_dir}")


if __name__ == "__main__":
    main()
