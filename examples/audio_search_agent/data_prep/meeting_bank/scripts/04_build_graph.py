#!/usr/bin/env python3
"""Phase 4: Build event relationship graph (no LLM).

Reads: processed/events.jsonl
Writes: processed/event_edges.jsonl

Heuristically links events within the same meeting:
- motion → vote: motion followed by vote with similar label
- question → response: question followed by response from different speaker
- presentation → discussion: presentation followed by discussion on same topic
- same_topic: events with overlapping labels (fuzzy match)
- temporal: consecutive events within a time gap

Usage:
    python scripts/04_build_graph.py --output-dir /path/to/output
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

from shared import read_jsonl, write_jsonl

# Maximum time gap (seconds) between events to consider them related
MAX_TEMPORAL_GAP = 300  # 5 minutes


def _word_set(label: str) -> set[str]:
    """Normalize a label to a set of lowercase words, dropping short stopwords."""
    stopwords = {"the", "a", "an", "of", "to", "and", "in", "on", "for", "by", "is"}
    return {w for w in label.lower().split() if len(w) > 1 and w not in stopwords}


def _label_similarity(a: str, b: str) -> float:
    """Jaccard similarity between label word sets."""
    wa, wb = _word_set(a), _word_set(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _speakers_differ(ev_a: dict, ev_b: dict) -> bool:
    """Check if two events have different speaker sets."""
    sa = set(ev_a.get("speakers", []))
    sb = set(ev_b.get("speakers", []))
    if not sa or not sb:
        return True  # assume different if unknown
    return sa != sb


def build_edges(events: list[dict]) -> list[dict]:
    """Build edges between events in the same meeting."""
    # Group by meeting
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for ev in events:
        by_meeting[ev["meeting_id"]].append(ev)

    edges = []

    for meeting_id, meeting_events in by_meeting.items():
        # Sort by start time
        meeting_events.sort(key=lambda e: e["start_sec"])

        for i, ev_a in enumerate(meeting_events):
            for j in range(i + 1, len(meeting_events)):
                ev_b = meeting_events[j]

                # Skip if too far apart
                gap = ev_b["start_sec"] - ev_a["end_sec"]
                if gap > MAX_TEMPORAL_GAP:
                    break  # sorted, so all further events are even farther

                edge_type = _classify_edge(ev_a, ev_b)
                if edge_type:
                    edges.append(
                        {
                            "src_event_id": ev_a["event_id"],
                            "dst_event_id": ev_b["event_id"],
                            "meeting_id": meeting_id,
                            "edge_type": edge_type,
                        }
                    )

    return edges


def _classify_edge(ev_a: dict, ev_b: dict) -> str | None:
    """Classify the relationship between two events, or None if no edge."""
    type_a = ev_a["event_type"]
    type_b = ev_b["event_type"]
    sim = _label_similarity(ev_a["label"], ev_b["label"])

    # motion → vote / vote_result
    if type_a == "motion" and type_b in ("vote", "vote_result"):
        if sim >= 0.3:
            return "motion_to_vote"

    # question → response
    if type_a == "question" and type_b == "response":
        if _speakers_differ(ev_a, ev_b):
            return "question_to_response"

    # presentation → discussion
    if type_a == "presentation" and type_b == "discussion":
        if sim >= 0.3:
            return "presentation_to_discussion"

    # same_topic (high label similarity, different events)
    if sim >= 0.5 and type_a != type_b:
        return "same_topic"

    # General temporal adjacency (only if close and no stronger edge)
    gap = ev_b["start_sec"] - ev_a["end_sec"]
    if 0 <= gap <= 60:  # within 1 minute
        return "temporal"

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: Build event graph")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/work/nvme/bffw/lmaben/long_speech/meeting_bank_v2"),
    )
    args = parser.parse_args()

    events = read_jsonl(args.output_dir / "processed" / "events.jsonl")
    if not events:
        print("No events found. Run 02_extract_events.py first.")
        return

    print(f"Loaded {len(events)} events")
    edges = build_edges(events)

    # Stats
    edge_types: dict[str, int] = defaultdict(int)
    for e in edges:
        edge_types[e["edge_type"]] += 1

    edges_path = args.output_dir / "processed" / "event_edges.jsonl"
    write_jsonl(edges_path, edges)

    print(f"\nDone: {len(edges)} edges")
    for etype, count in sorted(edge_types.items(), key=lambda x: -x[1]):
        print(f"  {etype}: {count}")
    print(f"Output: {edges_path}")


if __name__ == "__main__":
    main()
