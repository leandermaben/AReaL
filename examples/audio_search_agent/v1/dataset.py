"""Dataset loader for MeetingBank audio QA (RL training).

Scans JSON files from MeetingBank and builds a HuggingFace Dataset with
filtered questions:
  - ALL factual questions where gold_spans total <= 30 min and all_spans_necessary
  - Non-factual questions (speaker_count, pseudo_question) capped at 10% of mix

Each sample exposes:
    audio_id   : str   -- e.g. "meetingbank_val_0"
    messages   : list  -- [{"role": "user", "content": <question>}]
    answer     : str   -- ground-truth answer (letter A-D or free text)
    gold_spans : list  -- [{start_time, end_time, text}, ...]
    tag        : str   -- question type tag
    wav_path   : str   -- path to the audio wav file
    duration   : float -- total audio duration in seconds
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from datasets import Dataset

from areal.utils.logging import getLogger

logger = getLogger("MeetingBankDataset")

DEFAULT_DATA_ROOT = "/work/nvme/bffw/lmaben/long_speech/meeting_bank_v2"
MAX_GOLD_SPAN_DURATION = 1800.0  # 30 minutes
MAX_NONFACTUAL_RATIO = 0.1  # non-factual questions at most 10% of total


def _total_gold_span_duration(question: dict) -> float:
    """Sum the duration of all gold spans for a question."""
    return sum(
        s["end_time"] - s["start_time"]
        for s in question.get("gold_spans", [])
    )


# Tags considered "factual" (LLM-generated, verified)
FACTUAL_TAGS = {"factual", "single_event", "multi_hop"}


def _is_factual_eligible(question: dict) -> bool:
    """Check if a factual question passes the filtering criteria."""
    if _total_gold_span_duration(question) > MAX_GOLD_SPAN_DURATION:
        return False
    v = question.get("verification")
    if v is None:
        return True  # v2 verified questions already passed filtering
    return v.get("all_spans_necessary", True)


def get_meetingbank_dataset(
    data_root: str = DEFAULT_DATA_ROOT,
    split: str = "val",
    max_nonfactual_ratio: float = MAX_NONFACTUAL_RATIO,
    max_factual: int = 0,
    max_nonfactual: int = 0,
    seed: int = 42,
) -> Dataset:
    """Load filtered MeetingBank QA dataset.

    Parameters
    ----------
    data_root : path to meeting_bank_prepared directory
    split : which subdirectory to scan (val, train, test)
    max_nonfactual_ratio : max proportion of non-factual questions
    max_factual : if > 0, cap factual questions to this number
    max_nonfactual : if > 0, override nonfactual cap (ignores max_nonfactual_ratio)
    seed : random seed for sampling
    """
    split_dir = Path(data_root) / split
    json_files = sorted(split_dir.glob("audio_*.json"))

    if not json_files:
        raise RuntimeError(f"No JSON files found in {split_dir}")

    factual_samples = []
    nonfactual_samples = []

    for json_path in json_files:
        meta = json.loads(json_path.read_text())
        audio_id = meta["audio_id"]
        duration = meta.get("duration_seconds", 0.0)
        wav_path = str(json_path.with_suffix(".wav"))

        for q in meta.get("questions", []):
            tag = q.get("tag", "unknown")
            answer = q.get("answer")
            question_text = q.get("question", "")
            if not question_text or answer is None:
                continue

            # Include MCQ options in the question text
            options = q.get("options", {})
            if options:
                option_lines = []
                for letter in sorted(options):
                    if letter in {"A", "B", "C", "D"}:
                        option_lines.append(f"{letter}. {options[letter]}")
                if option_lines:
                    question_text = question_text + "\n\n" + "\n".join(option_lines)

            sample = {
                "audio_id": audio_id,
                "messages": [{"role": "user", "content": question_text}],
                "answer": str(answer),
                "gold_spans": q.get("gold_spans", []),
                "tag": tag,
                "wav_path": wav_path,
                "duration": duration,
            }

            # Skip speaker_count questions
            if tag == "speaker_count":
                continue

            if tag in FACTUAL_TAGS:
                if _is_factual_eligible(q):
                    factual_samples.append(sample)
            else:
                nonfactual_samples.append(sample)

    rng = random.Random(seed)

    # Cap factual if requested
    if max_factual > 0 and len(factual_samples) > max_factual:
        factual_samples = rng.sample(factual_samples, max_factual)

    # Cap non-factual: use explicit cap if given, else ratio-based
    if max_nonfactual > 0:
        nf_cap = max_nonfactual
    else:
        # nf / (f + nf) <= ratio  =>  nf <= f * ratio / (1 - ratio)
        nf_cap = int(len(factual_samples) * max_nonfactual_ratio / (1 - max_nonfactual_ratio))
    if len(nonfactual_samples) > nf_cap:
        nonfactual_samples = rng.sample(nonfactual_samples, nf_cap)

    all_samples = factual_samples + nonfactual_samples
    rng.shuffle(all_samples)

    logger.info(
        f"MeetingBank dataset ({split}): {len(factual_samples)} factual + "
        f"{len(nonfactual_samples)} non-factual = {len(all_samples)} samples"
    )
    return Dataset.from_list(all_samples)
