#!/usr/bin/env python3
"""Phase 5: Generate questions from events and graph edges.

Reads: processed/events.jsonl, processed/event_edges.jsonl,
       processed/speakers.jsonl, processed/segments.jsonl
Writes: qa/qa_candidates.jsonl

Three question types:
1. Single-event MCQ (~40%) — LLM generates from one event's transcript
2. Multi-hop MCQ (~40%) — LLM generates from two linked events
3. Speaker count MCQ (~20%) — deterministic from segment speaker IDs

Usage:
    python scripts/05_generate_questions.py \
        --output-dir /path/to/output \
        --vllm-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import asyncio
import random
from collections import defaultdict
from pathlib import Path

from shared import (
    AsyncLLMClient,
    count_tokens,
    fmt_time,
    read_jsonl,
    snap_to_3s,
    truncate_to_budget,
    write_jsonl,
)

SYSTEM_PROMPT = (
    "You are a question generation expert for long-form audio recordings of city council "
    "meetings. You generate accurate multiple-choice questions based on transcript excerpts. "
    "Always respond with valid JSON and nothing else — no markdown, no prose."
)

SINGLE_EVENT_PROMPT = """\
Below is a transcript excerpt from a city council meeting.

Event: {event_type} — "{label}"
Time: {start} to {end}
Speakers: {speaker_names}
Transcript:
{text}

Generate one multiple-choice question (A/B/C/D) that:
- Is answerable ONLY from this transcript
- Has one clearly correct answer
- Has 3 plausible but wrong distractors
- Tests comprehension of specific details (names, numbers, decisions), not vague summaries

Return ONLY this JSON:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "comment": "brief explanation"}}"""

MULTI_HOP_PROMPT = """\
Below are two related transcript excerpts from a city council meeting.

Event 1: {type1} — "{label1}" ({start1} to {end1})
Speakers: {speakers1}
{text1}

Event 2: {type2} — "{label2}" ({start2} to {end2})
Speakers: {speakers2}
{text2}

Relationship: {edge_type}

Generate one multiple-choice question that REQUIRES information from BOTH events.
The reader must first extract a fact from one event, then use it to answer about the other.

Return ONLY this JSON:
{{"question": "...", "options": {{"A": "...", "B": "...", "C": "...", "D": "..."}}, "answer": "A", "is_multi_hop": true, "comment": "explain the reasoning chain across both events"}}"""

# Speaker count question phrasings
SPEAKER_COUNT_PHRASINGS = [
    "How many distinct speakers appear in this meeting?",
    "How many different people spoke during this meeting?",
    "Count the number of unique speakers in this recording.",
    "How many individuals can be heard speaking in this meeting?",
    "What is the total number of distinct speakers in this meeting?",
    "How many separate speakers participated in this meeting?",
]


def _get_event_text(event: dict, segments: list[dict]) -> str:
    """Get transcript text for an event's time range from segments."""
    meeting_segs = [
        s for s in segments if s["meeting_id"] == event["meeting_id"]
    ]
    in_range = [
        s
        for s in meeting_segs
        if s["start_sec"] < event["end_sec"] and s["end_sec"] > event["start_sec"]
    ]
    if not in_range:
        return event.get("summary", "")
    return " ".join(s["text"] for s in in_range)


def _get_speaker_names(
    event: dict, speakers_map: dict[str, dict[int, dict]]
) -> str:
    """Get formatted speaker names for an event."""
    meeting_speakers = speakers_map.get(event["meeting_id"], {})
    names = []
    for sid in event.get("speakers", []):
        info = meeting_speakers.get(sid)
        if info and info["name"] != "Unknown":
            names.append(f"{info['name']} ({info['role']})")
        else:
            names.append(f"Speaker {sid}")
    return ", ".join(names) if names else "Unknown"


async def generate_single_event_question(
    llm: AsyncLLMClient,
    event: dict,
    segments: list[dict],
    speakers_map: dict[str, dict[int, dict]],
    question_id: str,
) -> dict | None:
    """Generate a single-event MCQ."""
    text = _get_event_text(event, segments)
    if not text or count_tokens(text) < 20:
        return None

    text = truncate_to_budget(text, 3000)
    speaker_names = _get_speaker_names(event, speakers_map)

    prompt = SINGLE_EVENT_PROMPT.format(
        event_type=event["event_type"],
        label=event["label"],
        start=fmt_time(event["start_sec"]),
        end=fmt_time(event["end_sec"]),
        speaker_names=speaker_names,
        text=text,
    )

    result = await llm.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800)
    if result is None or not isinstance(result, dict):
        return None
    if not all(k in result for k in ("question", "options", "answer")):
        return None
    if result.get("answer") not in ("A", "B", "C", "D"):
        return None

    return {
        "question_id": question_id,
        "meeting_id": event["meeting_id"],
        "question": result["question"],
        "options": result["options"],
        "answer": result["answer"],
        "gold_spans": [
            {
                "start_time": round(event["start_sec"], 3),
                "end_time": round(event["end_sec"], 3),
            }
        ],
        "tag": "single_event",
        "multi_hop": False,
        "comment": result.get("comment", ""),
        "source_events": [event["event_id"]],
    }


async def generate_multi_hop_question(
    llm: AsyncLLMClient,
    edge: dict,
    events_map: dict[str, dict],
    segments: list[dict],
    speakers_map: dict[str, dict[int, dict]],
    question_id: str,
) -> dict | None:
    """Generate a multi-hop MCQ from two linked events."""
    ev_a = events_map.get(edge["src_event_id"])
    ev_b = events_map.get(edge["dst_event_id"])
    if not ev_a or not ev_b:
        return None

    text_a = _get_event_text(ev_a, segments)
    text_b = _get_event_text(ev_b, segments)
    if not text_a or not text_b:
        return None

    # Budget: 1500 tokens each, truncate longer one if needed
    tok_a, tok_b = count_tokens(text_a), count_tokens(text_b)
    if tok_a + tok_b > 3000:
        if tok_a > tok_b:
            text_a = truncate_to_budget(text_a, 3000 - min(tok_b, 1500))
        else:
            text_b = truncate_to_budget(text_b, 3000 - min(tok_a, 1500))

    prompt = MULTI_HOP_PROMPT.format(
        type1=ev_a["event_type"],
        label1=ev_a["label"],
        start1=fmt_time(ev_a["start_sec"]),
        end1=fmt_time(ev_a["end_sec"]),
        speakers1=_get_speaker_names(ev_a, speakers_map),
        text1=text_a,
        type2=ev_b["event_type"],
        label2=ev_b["label"],
        start2=fmt_time(ev_b["start_sec"]),
        end2=fmt_time(ev_b["end_sec"]),
        speakers2=_get_speaker_names(ev_b, speakers_map),
        text2=text_b,
        edge_type=edge["edge_type"],
    )

    result = await llm.generate(prompt, system_prompt=SYSTEM_PROMPT, max_tokens=800)
    if result is None or not isinstance(result, dict):
        return None
    if not all(k in result for k in ("question", "options", "answer")):
        return None
    if result.get("answer") not in ("A", "B", "C", "D"):
        return None

    # Gold spans = union of both events
    spans = [
        {"start_time": round(ev_a["start_sec"], 3), "end_time": round(ev_a["end_sec"], 3)},
        {"start_time": round(ev_b["start_sec"], 3), "end_time": round(ev_b["end_sec"], 3)},
    ]

    return {
        "question_id": question_id,
        "meeting_id": ev_a["meeting_id"],
        "question": result["question"],
        "options": result["options"],
        "answer": result["answer"],
        "gold_spans": spans,
        "tag": "multi_hop",
        "multi_hop": True,
        "comment": result.get("comment", ""),
        "source_events": [ev_a["event_id"], ev_b["event_id"]],
    }


def make_speaker_count_questions(
    segments: list[dict],
    meeting_id: str,
    rng: random.Random,
    base_q_idx: int,
) -> list[dict]:
    """Generate deterministic speaker count MCQs for a meeting."""
    meeting_segs = [s for s in segments if s["meeting_id"] == meeting_id]
    if not meeting_segs:
        return []

    unique_speakers = {s["speaker_id"] for s in meeting_segs}
    correct = len(unique_speakers)
    if correct < 2:
        return []

    # Full meeting bounds
    start_sec = meeting_segs[0]["start_sec"]
    end_sec = meeting_segs[-1]["end_sec"]
    start_sec, end_sec = snap_to_3s(start_sec, end_sec)

    # Generate distractors: 3 nearby integers
    candidates = [v for v in range(max(1, correct - 3), correct + 4) if v != correct]
    rng.shuffle(candidates)
    distractors = candidates[:3]

    pool = [correct] + distractors
    rng.shuffle(pool)
    letters = ["A", "B", "C", "D"]
    options = {l: str(v) for l, v in zip(letters, pool)}
    answer = letters[pool.index(correct)]

    question_text = rng.choice(SPEAKER_COUNT_PHRASINGS)

    return [
        {
            "question_id": f"{meeting_id}:q{base_q_idx}",
            "meeting_id": meeting_id,
            "question": question_text,
            "options": options,
            "answer": answer,
            "gold_spans": [
                {"start_time": round(start_sec, 3), "end_time": round(end_sec, 3)}
            ],
            "tag": "speaker_count",
            "multi_hop": False,
            "comment": f"Correct count: {correct} speakers",
            "source_events": [],
        }
    ]


async def async_main(args: argparse.Namespace) -> None:
    # Load all inputs
    events = read_jsonl(args.output_dir / "processed" / "events.jsonl")
    edges = read_jsonl(args.output_dir / "processed" / "event_edges.jsonl")
    speakers_raw = read_jsonl(args.output_dir / "processed" / "speakers.jsonl")
    segments = read_jsonl(args.output_dir / "processed" / "segments.jsonl")

    if not events:
        print("No events found. Run 02_extract_events.py first.")
        return

    print(
        f"Loaded: {len(events)} events, {len(edges)} edges, "
        f"{len(speakers_raw)} speaker entries, {len(segments)} segments"
    )

    # Build lookup maps
    events_map = {e["event_id"]: e for e in events}
    speakers_map: dict[str, dict[int, dict]] = defaultdict(dict)
    for s in speakers_raw:
        speakers_map[s["meeting_id"]][s["speaker_id"]] = s

    # Check for existing output
    qa_path = args.output_dir / "qa" / "qa_candidates.jsonl"
    existing_questions: list[dict] = []
    existing_meetings: set[str] = set()
    if qa_path.is_file():
        existing_questions = read_jsonl(qa_path)
        existing_meetings = {q["meeting_id"] for q in existing_questions}
        print(f"Resuming: {len(existing_meetings)} meetings already have questions")

    # Get meetings to process
    all_meeting_ids = sorted({e["meeting_id"] for e in events})
    meeting_ids = [m for m in all_meeting_ids if m not in existing_meetings]
    if not meeting_ids:
        print("All meetings already have questions.")
        return

    print(f"Generating questions for {len(meeting_ids)} meetings...")

    llm = AsyncLLMClient(
        base_url=args.vllm_url,
        model=args.model,
        concurrency=args.concurrency,
        system_prompt=SYSTEM_PROMPT,
    )

    rng = random.Random(args.seed)
    all_questions = list(existing_questions)
    q_counter = len(existing_questions)

    for m_idx, meeting_id in enumerate(meeting_ids):
        if llm.server_down.is_set():
            print("\n[WARN] Server down — stopping early.")
            break

        meeting_events = [e for e in events if e["meeting_id"] == meeting_id]
        meeting_edges = [e for e in edges if e["meeting_id"] == meeting_id]

        tasks = []

        # 1. Single-event questions
        for ev in meeting_events:
            qid = f"{meeting_id}:q{q_counter}"
            q_counter += 1
            tasks.append(
                (
                    "single",
                    generate_single_event_question(
                        llm, ev, segments, speakers_map, qid
                    ),
                )
            )

        # 2. Multi-hop questions (from edges, preferring non-temporal edges)
        # Prioritize structured edges
        priority_edges = [e for e in meeting_edges if e["edge_type"] != "temporal"]
        temporal_edges = [e for e in meeting_edges if e["edge_type"] == "temporal"]
        selected_edges = priority_edges + temporal_edges[:5]

        for edge in selected_edges:
            qid = f"{meeting_id}:q{q_counter}"
            q_counter += 1
            tasks.append(
                (
                    "multi_hop",
                    generate_multi_hop_question(
                        llm, edge, events_map, segments, speakers_map, qid
                    ),
                )
            )

        # Run LLM tasks concurrently
        if tasks:
            coros = [t[1] for t in tasks]
            results = await asyncio.gather(*coros)
            for result in results:
                if result is not None:
                    all_questions.append(result)

        # 3. Speaker count (deterministic, no LLM)
        speaker_qs = make_speaker_count_questions(
            segments, meeting_id, rng, q_counter
        )
        q_counter += len(speaker_qs)
        all_questions.extend(speaker_qs)

        if (m_idx + 1) % 20 == 0 or m_idx == len(meeting_ids) - 1:
            n_new = len(all_questions) - len(existing_questions)
            print(f"  [{m_idx+1}/{len(meeting_ids)}] {meeting_id}: {n_new} new questions total")
            write_jsonl(qa_path, all_questions)

    write_jsonl(qa_path, all_questions)
    llm.print_stats()

    # Stats
    tag_counts: dict[str, int] = defaultdict(int)
    for q in all_questions:
        tag_counts[q["tag"]] += 1

    print(f"\nDone: {len(all_questions)} total questions")
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: Generate questions")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_v2"),
    )
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
