"""CLAP retrieval recall sanity check.

For a sample of audio files and their questions, runs CLAP text search with the
question text as query and checks how many gold spans are "hit" at various top-k
levels. A gold span counts as hit if any retrieved segment overlaps with it.

Usage:
    python -m examples.audio_search_agent.sanity_checks.clap.test_clap_recall \
        [--data-root /work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared] \
        [--cache-dir /work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank] \
        [--num-audios 3] \
        [--max-questions-per-audio 5]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.audio_search_agent.v1.preprocessing.clap_indexer import (
    CLAPConfig,
    CLAPIndexer,
)

DATA_ROOT = "/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared"
CACHE_DIR = "/work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank"
TOP_K_LEVELS = [5, 10, 30, 50]


def segments_overlap(seg_start: float, seg_end: float, gold_start: float, gold_end: float) -> bool:
    """Check if two time intervals overlap."""
    return seg_start < gold_end and seg_end > gold_start


def evaluate_question(
    indexer: CLAPIndexer,
    audio_id: str,
    question: dict,
    top_k_levels: list[int],
) -> dict:
    """Evaluate recall for a single question at multiple top-k levels.

    Uses the question text as the CLAP query.
    A gold span is "hit" if any retrieved segment overlaps with it.

    Returns:
        dict with keys: question_id, question_text, num_gold_spans,
                        recall@k for each k in top_k_levels
    """
    query = question["question"]
    gold_spans = question["gold_spans"]
    max_k = max(top_k_levels)

    results = indexer.search_by_text(query_text=query, audio_id=audio_id, k=max_k)

    recall_at_k = {}
    for k in top_k_levels:
        top_results = results[:k]
        hits = 0
        for gs in gold_spans:
            gs_start, gs_end = gs["start_time"], gs["end_time"]
            for r in top_results:
                if segments_overlap(r["start"], r["end"], gs_start, gs_end):
                    hits += 1
                    break
        recall_at_k[k] = hits / len(gold_spans) if gold_spans else 0.0

    return {
        "question_id": question.get("question_id", "unknown"),
        "question_text": query[:100],
        "num_gold_spans": len(gold_spans),
        "recall": recall_at_k,
    }


def main():
    parser = argparse.ArgumentParser(description="CLAP retrieval recall sanity check")
    parser.add_argument("--data-root", default=DATA_ROOT)
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument("--num-audios", type=int, default=3,
                        help="Number of val audios to test")
    parser.add_argument("--max-questions-per-audio", type=int, default=5,
                        help="Max questions to evaluate per audio")
    parser.add_argument("--device", default="cuda",
                        help="Device for CLAP model (cuda or cpu)")
    args = parser.parse_args()

    config = CLAPConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    indexer = CLAPIndexer(config)

    val_dir = Path(args.data_root) / "val"
    json_files = sorted(val_dir.glob("audio_*.json"))[:args.num_audios]

    all_results = []
    aggregate_recall = {k: [] for k in TOP_K_LEVELS}

    for json_path in json_files:
        meta = json.loads(json_path.read_text())
        audio_id = meta["audio_id"]
        questions = meta["questions"][:args.max_questions_per_audio]

        print(f"\n{'='*70}")
        print(f"Audio: {audio_id}  ({meta.get('duration_seconds', '?')}s, "
              f"{meta.get('source_city', '?')})")
        print(f"Evaluating {len(questions)} questions")
        print(f"{'='*70}")

        for q in questions:
            result = evaluate_question(indexer, audio_id, q, TOP_K_LEVELS)
            all_results.append(result)
            for k in TOP_K_LEVELS:
                aggregate_recall[k].append(result["recall"][k])

            recalls_str = "  ".join(
                f"R@{k}: {result['recall'][k]:.2f}" for k in TOP_K_LEVELS
            )
            print(f"  [{result['question_id']}] {recalls_str}  "
                  f"(gold_spans={result['num_gold_spans']})")
            print(f"    Q: {result['question_text']}")

    # Summary
    print(f"\n{'='*70}")
    print("AGGREGATE RECALL")
    print(f"{'='*70}")
    print(f"Total questions evaluated: {len(all_results)}")
    for k in TOP_K_LEVELS:
        vals = aggregate_recall[k]
        mean = np.mean(vals)
        print(f"  Recall@{k:>2}: {mean:.4f}  (min={min(vals):.2f}, max={max(vals):.2f})")

    # Save results
    out_path = Path(__file__).parent / "recall_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "top_k_levels": TOP_K_LEVELS,
            "num_questions": len(all_results),
            "aggregate_recall": {str(k): float(np.mean(v)) for k, v in aggregate_recall.items()},
            "per_question": all_results,
        }, f, indent=2)
    print(f"\nDetailed results saved to {out_path}")


if __name__ == "__main__":
    main()
