#!/usr/bin/env python3
"""Phase 6: Verify and filter generated questions using LLM.

Reads: qa/qa_candidates.jsonl, processed/segments.jsonl
Writes: qa/qa_verified.jsonl

For each candidate question, asks the LLM to verify:
- Is the answer correct?
- Is it answerable from the gold spans only?
- Are all spans necessary?
- Is multi-hop confirmed (if claimed)?

Usage:
    python scripts/06_verify_and_filter.py \
        --output-dir /path/to/output \
        --vllm-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from shared import (
    AsyncLLMClient,
    read_jsonl,
    truncate_to_budget,
    write_jsonl,
)

SYSTEM_PROMPT = (
    "You are a quality assurance expert for educational questions. "
    "You verify that questions are answerable, correct, and unambiguous. "
    "Always respond with valid JSON and nothing else."
)

VERIFY_PROMPT = """\
Question: {question}
Options: {opts}
Claimed answer: {answer}

Supporting transcript:
{gold_span_text}

Verify:
1. Is the claimed answer correct?
2. Is the question answerable from ONLY the supporting transcript?
3. Are ALL supporting spans necessary (removing any one makes it unanswerable)?
4. If multi-hop: does answering truly require chaining across spans?
5. Is the question unambiguous?

Return ONLY this JSON:
{{"correct": true, "answerable": true, "all_spans_necessary": true, "multi_hop_confirmed": false, "unambiguous": true, "issues": ""}}"""


def _get_gold_span_text(question: dict, segments: list[dict]) -> str:
    """Get transcript text for a question's gold spans."""
    meeting_segs = [
        s for s in segments if s["meeting_id"] == question["meeting_id"]
    ]
    parts = []
    for span in question.get("gold_spans", []):
        in_range = [
            s
            for s in meeting_segs
            if s["start_sec"] < span["end_time"] and s["end_sec"] > span["start_time"]
        ]
        text = " ".join(s["text"] for s in in_range)
        if text:
            parts.append(text)
    full = "\n\n---\n\n".join(parts)
    return truncate_to_budget(full, 2500)


async def verify_question(
    llm: AsyncLLMClient,
    question: dict,
    segments: list[dict],
) -> dict | None:
    """Verify a single question. Returns the question with verification, or None."""
    # Speaker count questions are deterministic — skip verification
    if question["tag"] == "speaker_count":
        question["verification"] = {
            "correct": True,
            "answerable": True,
            "all_spans_necessary": True,
            "multi_hop_confirmed": False,
            "unambiguous": True,
            "issues": "",
        }
        return question

    gold_text = _get_gold_span_text(question, segments)
    if not gold_text:
        return None

    opts_str = " | ".join(
        f"{k}: {v}" for k, v in question.get("options", {}).items()
    )

    prompt = VERIFY_PROMPT.format(
        question=question["question"],
        opts=opts_str,
        answer=question["answer"],
        gold_span_text=gold_text,
    )

    result = await llm.generate(
        prompt,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=256,
        temperature=0.2,
    )

    if result is None or not isinstance(result, dict):
        return None

    question["verification"] = result

    # Filter logic
    correct = result.get("correct", False)
    answerable = result.get("answerable", False)
    unambiguous = result.get("unambiguous", False)

    if not (correct and answerable and unambiguous):
        return None

    # For multi-hop, also require confirmation
    if question.get("multi_hop") and not result.get("multi_hop_confirmed", False):
        return None

    return question


async def async_main(args: argparse.Namespace) -> None:
    candidates = read_jsonl(args.output_dir / "qa" / "qa_candidates.jsonl")
    segments = read_jsonl(args.output_dir / "processed" / "segments.jsonl")

    if not candidates:
        print("No candidate questions found. Run 05_generate_questions.py first.")
        return

    print(f"Loaded {len(candidates)} candidate questions")

    # Check for existing verified output
    verified_path = args.output_dir / "qa" / "qa_verified.jsonl"
    existing_ids: set[str] = set()
    existing_verified: list[dict] = []
    if verified_path.is_file():
        existing_verified = read_jsonl(verified_path)
        existing_ids = {q["question_id"] for q in existing_verified}
        print(f"Resuming: {len(existing_ids)} questions already verified")

    # Filter to unverified
    to_verify = [q for q in candidates if q["question_id"] not in existing_ids]
    if not to_verify:
        print("All questions already verified.")
        return

    print(f"Verifying {len(to_verify)} questions...")

    llm = AsyncLLMClient(
        base_url=args.vllm_url,
        model=args.model,
        concurrency=args.concurrency,
        system_prompt=SYSTEM_PROMPT,
    )

    verified = list(existing_verified)
    batch_size = min(args.concurrency, 128)

    for batch_start in range(0, len(to_verify), batch_size):
        if llm.server_down.is_set():
            print("\n[WARN] Server down — stopping early.")
            break

        batch = to_verify[batch_start : batch_start + batch_size]
        tasks = [verify_question(llm, q, segments) for q in batch]
        results = await asyncio.gather(*tasks)

        for result in results:
            if result is not None:
                verified.append(result)

        done = min(batch_start + batch_size, len(to_verify))
        passed = len(verified) - len(existing_verified)
        print(f"  [{done}/{len(to_verify)}] verified, {passed} passed so far")

        # Checkpoint every 10 batches
        if (batch_start // batch_size) % 10 == 9:
            write_jsonl(verified_path, verified)

    write_jsonl(verified_path, verified)
    llm.print_stats()

    # Stats
    n_new = len(verified) - len(existing_verified)
    n_processed = len(to_verify)
    pass_rate = n_new / n_processed * 100 if n_processed > 0 else 0

    print(f"\nDone: {n_new}/{n_processed} passed verification ({pass_rate:.1f}%)")
    print(f"Total verified: {len(verified)}")

    tag_counts: dict[str, int] = {}
    for q in verified:
        tag = q["tag"]
        tag_counts[tag] = tag_counts.get(tag, 0) + 1
    for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1]):
        print(f"  {tag}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: Verify and filter questions"
    )
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
