"""
Dataset loader for LongformAudio QA tasks.

Scans JSON files from the LongformAudio directories and builds a HuggingFace
Dataset of (audio_id, question, answer) triples.  Only records whose audio_id
already exists in the PostgreSQL database are included so the agent can
actually query them during rollout.

Each sample exposes:
    audio_id  : str   -- e.g. "qa-partII_10min_audio_0"
    messages  : list  -- [{"role": "user", "content": <question>}]
    answer    : str   -- ground-truth answer (letter A-D or free text)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from datasets import Dataset

from areal.utils.logging import getLogger

logger = getLogger("AudioQADataset")

LONGFORM_ROOT = Path("/data/user_data/msomeki/30_shared/LongformAudio")

# Default QA subdirectories (10-min and 30-min; expand as needed)
DEFAULT_QA_DIRS = [
    "qa-partI_10min",
    "qa-partII_10min",
    "qa-partI_30min",
    "qa-partII_30min",
]


def _iter_qa_samples(dirs: list[Path]):
    """Yield (audio_id, question, answer) for every QA entry in dirs."""
    for d in dirs:
        if not d.exists():
            logger.warning(f"QA directory not found, skipping: {d}")
            continue
        for json_path in sorted(
            d.glob("audio_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        ):
            audio_id = f"{d.name}_{json_path.stem}"
            try:
                with open(json_path) as f:
                    meta = json.load(f)
            except Exception:
                logger.warning(f"Failed to read {json_path}, skipping")
                continue

            questions = meta.get("questions") or [
                {"question": meta.get("question"), "answer": meta.get("answer")}
            ]
            for q in questions:
                question = q.get("question") or ""
                answer = q.get("answer")
                if not question or answer is None:
                    continue
                yield audio_id, question, str(answer)


def _ingested_audio_ids(postgres_host: str, postgres_port: int) -> set[str]:
    """Return the set of audio_ids that have been ingested into the DB."""
    try:
        import psycopg2

        conn = psycopg2.connect(
            dbname=os.environ.get("POSTGRES_DB", "espnet_db"),
            user=os.environ.get("POSTGRES_USER", "espnet_user"),
            password=os.environ.get("POSTGRES_PASSWORD", "espnet_password"),
            host=postgres_host,
            port=postgres_port,
        )
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT audio_id FROM transcription")
            ids = {row[0] for row in cur.fetchall()}
        conn.close()
        return ids
    except Exception as exc:
        logger.warning(f"Could not query DB for ingested audio_ids: {exc}. Including all samples.")
        return set()


def get_audio_qa_dataset(
    data_dirs: list[str] | None = None,
    split: str = "train",
    train_ratio: float = 0.9,
    postgres_host: str = "localhost",
    postgres_port: int = 5433,
    filter_to_ingested: bool = True,
) -> Dataset:
    """Load the audio QA dataset from LongformAudio JSON files.

    Parameters
    ----------
    data_dirs:
        Explicit list of directory paths to scan.  Defaults to the
        DEFAULT_QA_DIRS subdirectories under LONGFORM_ROOT.
    split:
        "train" or "test" (or any other value treated as test).
    train_ratio:
        Fraction of samples to assign to the training split.
    postgres_host / postgres_port:
        Database connection info; used to filter to ingested audio_ids.
    filter_to_ingested:
        If True (default), drop samples whose audio_id is not yet in the DB.
    """
    if data_dirs is None:
        dirs = [LONGFORM_ROOT / d for d in DEFAULT_QA_DIRS]
    else:
        dirs = [Path(d) for d in data_dirs]

    ingested = _ingested_audio_ids(postgres_host, postgres_port) if filter_to_ingested else set()

    samples = []
    for audio_id, question, answer in _iter_qa_samples(dirs):
        if filter_to_ingested and ingested and audio_id not in ingested:
            continue
        samples.append(
            {
                "audio_id": audio_id,
                "messages": [{"role": "user", "content": question}],
                "answer": answer,
            }
        )

    if not samples:
        raise RuntimeError(
            "No samples found.  Check data_dirs and ensure audio files are ingested."
        )

    n_train = int(len(samples) * train_ratio)
    if split == "train":
        subset = samples[:n_train]
    else:
        subset = samples[n_train:]

    logger.info(f"Loaded {len(subset)} '{split}' samples from {len(dirs)} directories.")
    return Dataset.from_list(subset)
