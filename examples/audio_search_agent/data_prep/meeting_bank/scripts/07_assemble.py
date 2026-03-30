#!/usr/bin/env python3
"""Phase 7: Assemble final output (audio_N.wav + audio_N.json + manifest.json).

Reads: raw audio (MP3), qa/qa_verified.jsonl, processed/speakers.jsonl,
       processed/segments.jsonl
Writes: train/val/test directories with audio_N.wav + audio_N.json, manifest.json

Output format is identical to v1 so the training pipeline needs zero changes.

Usage:
    python scripts/07_assemble.py \
        --data-root /path/to/raw \
        --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from shared import discover_pairs, read_jsonl, seg_bounds_s, load_raw_segments


def convert_mp3_to_wav(mp3: Path, wav: Path) -> bool:
    """Convert MP3 to 16kHz mono WAV."""
    wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-i", str(mp3), "-ar", "16000", "-ac", "1", str(wav)]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  [ffmpeg error] {r.stderr.decode()[-300:]}", file=sys.stderr)
    return r.returncode == 0


def assign_splits(
    meeting_ids: list[str],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> dict[str, str]:
    """Assign train/val/test splits by meeting ID."""
    rng = random.Random(seed)
    ids = list(meeting_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    result = {}
    for i, mid in enumerate(ids):
        if i < n_train:
            result[mid] = "train"
        elif i < n_train + n_val:
            result[mid] = "val"
        else:
            result[mid] = "test"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: Assemble final output")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/work/hdd/bbjs/lmaben/speech/long_speech/temp_data"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_v2"),
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip MP3→WAV conversion (if WAVs already exist).",
    )
    args = parser.parse_args()

    # Load verified questions
    qa_path = args.output_dir / "qa" / "qa_verified.jsonl"
    questions = read_jsonl(qa_path)
    if not questions:
        print("No verified questions found. Run 06_verify_and_filter.py first.")
        return

    # Load speaker info
    speakers_raw = read_jsonl(args.output_dir / "processed" / "speakers.jsonl")
    speakers_map: dict[str, dict[int, dict]] = defaultdict(dict)
    for s in speakers_raw:
        speakers_map[s["meeting_id"]][s["speaker_id"]] = s

    # Load segments for duration info
    segments = read_jsonl(args.output_dir / "processed" / "segments.jsonl")
    meeting_durations: dict[str, float] = {}
    meeting_n_speakers: dict[str, int] = {}
    for s in segments:
        mid = s["meeting_id"]
        meeting_durations[mid] = max(meeting_durations.get(mid, 0), s["end_sec"])
        if mid not in meeting_n_speakers:
            meeting_n_speakers[mid] = 0
    # Count unique speakers per meeting
    spk_sets: dict[str, set[int]] = defaultdict(set)
    for s in segments:
        spk_sets[s["meeting_id"]].add(s["speaker_id"])
    meeting_n_speakers = {mid: len(spks) for mid, spks in spk_sets.items()}

    # Group questions by meeting
    qs_by_meeting: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        qs_by_meeting[q["meeting_id"]].append(q)

    meeting_ids = sorted(qs_by_meeting.keys())
    print(f"Loaded {len(questions)} verified questions across {len(meeting_ids)} meetings")

    # Discover audio files to find MP3 paths
    pairs = discover_pairs(args.data_root)
    mp3_map: dict[str, Path] = {}
    for mp3, transcript, city in pairs:
        meeting_id = f"{city}/{mp3.stem}"
        mp3_map[meeting_id] = mp3

    # Assign splits
    split_map = assign_splits(meeting_ids, args.train_frac, args.val_frac, args.seed)
    for s in ("train", "val", "test"):
        n = sum(1 for m in meeting_ids if split_map.get(m) == s)
        print(f"  {s}: {n} meetings")

    # Assemble
    manifest = []
    split_counters: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for meeting_id in meeting_ids:
        split = split_map[meeting_id]
        idx = split_counters[split]
        split_counters[split] += 1

        split_dir = args.output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        wav_path = split_dir / f"audio_{idx}.wav"
        json_path = split_dir / f"audio_{idx}.json"
        audio_id = f"meetingbank_{split}_{idx}"

        # Convert audio
        mp3 = mp3_map.get(meeting_id)
        if mp3 and not args.skip_audio:
            if not wav_path.exists():
                ok = convert_mp3_to_wav(mp3, wav_path)
                if not ok:
                    print(f"  [WARN] Failed to convert {mp3.name}")

        # Build per-audio JSON (same schema as v1)
        meeting_qs = qs_by_meeting[meeting_id]
        # Reassign question IDs to match output format
        for qi, q in enumerate(meeting_qs):
            q["question_id"] = f"{audio_id}:q{qi}"
            # Remove internal fields not needed in final output
            q.pop("source_events", None)

        city = meeting_id.split("/")[0] if "/" in meeting_id else "unknown"

        payload = {
            "audio_id": audio_id,
            "source_city": city,
            "source_file": mp3.name if mp3 else meeting_id,
            "split": split,
            "duration_seconds": round(meeting_durations.get(meeting_id, 0), 2),
            "num_speakers": meeting_n_speakers.get(meeting_id, 0),
            "questions": meeting_qs,
        }
        json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

        n_single = sum(1 for q in meeting_qs if q["tag"] == "single_event")
        n_multi = sum(1 for q in meeting_qs if q["tag"] == "multi_hop")
        n_spk = sum(1 for q in meeting_qs if q["tag"] == "speaker_count")

        manifest.append(
            {
                "audio_id": audio_id,
                "split": split,
                "source_city": city,
                "source_file": mp3.name if mp3 else meeting_id,
                "duration_seconds": round(meeting_durations.get(meeting_id, 0), 2),
                "num_questions": len(meeting_qs),
                "num_single_event": n_single,
                "num_multi_hop": n_multi,
                "num_speaker_count": n_spk,
            }
        )

    # Write manifest
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    total_q = sum(e["num_questions"] for e in manifest)
    print(f"\nDone: {len(manifest)} audios, {total_q} total questions")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")

    for s in ("train", "val", "test"):
        n_files = sum(1 for e in manifest if e["split"] == s)
        n_qs = sum(e["num_questions"] for e in manifest if e["split"] == s)
        print(f"  {s}: {n_files} files, {n_qs} questions")


if __name__ == "__main__":
    main()
