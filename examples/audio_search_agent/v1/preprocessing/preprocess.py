"""CLI entry point for the CLAP preprocessing pipeline.

Modes
-----
embed   -- Compute CLAP embeddings for a shard of audio files and cache them.
           Safe to run multiple workers in parallel (each touches disjoint .npz files).
index   -- Build per-audio FAISS indexes from all cached .npz files.
           Run single-process after all embed workers finish.
all     -- embed then index in a single process (useful for single-GPU runs / debugging).

Examples
--------
# Single GPU, full dataset:
python -m examples.audio_search_agent.v1.preprocessing.preprocess \\
    --mode all \\
    --data-root /work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared \\
    --cache-dir  /work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank

# Multi-GPU embedding (run concurrently, one per GPU):
CUDA_VISIBLE_DEVICES=0 python -m examples.audio_search_agent.v1.preprocessing.preprocess \\
    --mode embed --worker-id 0 --num-workers 2 ...
CUDA_VISIBLE_DEVICES=1 python -m examples.audio_search_agent.v1.preprocessing.preprocess \\
    --mode embed --worker-id 1 --num-workers 2 ...

# Then build the FAISS index (CPU, single process):
python -m examples.audio_search_agent.v1.preprocessing.preprocess \\
    --mode index ...
"""

import argparse
import sys

from areal.utils.logging import getLogger

from .clap_indexer import CLAPConfig, CLAPIndexer

logger = getLogger("CLAPIndexer")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="CLAP preprocessing pipeline for audio search agent (v1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── required paths ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--data-root",
        default="/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared",
        help="Root directory of the prepared dataset (contains manifest.json + split dirs).",
    )
    parser.add_argument(
        "--cache-dir",
        default="/work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank",
        help="Directory where per-audio embeddings and the FAISS index are stored.",
    )

    # ── mode ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--mode",
        choices=["embed", "index", "all"],
        default="all",
        help=(
            "embed: compute + cache CLAP embeddings only. "
            "index: build FAISS from cached embeddings only. "
            "all: embed then index (single-process)."
        ),
    )

    # ── model / segmentation ───────────────────────────────────────────────────
    parser.add_argument(
        "--model-id",
        default="laion/larger_clap_general",
        help="HuggingFace model ID for CLAP.",
    )
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=9.0,
        help="Duration of each audio chunk in seconds.",
    )
    parser.add_argument(
        "--chunk-stride",
        type=float,
        default=3.0,
        help="Stride between consecutive chunks in seconds (< chunk_duration → overlap).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of audio chunks per CLAP forward pass.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device string (cuda / cpu).",
    )

    # ── dataset scope ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to process.",
    )

    # ── multi-worker sharding ──────────────────────────────────────────────────
    parser.add_argument(
        "--worker-id",
        type=int,
        default=0,
        help="This worker's index (0-based).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Total number of parallel workers (for sharding the file list).",
    )

    # ── misc ───────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if a cache file already exists.",
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    config = CLAPConfig(
        data_root=args.data_root,
        cache_dir=args.cache_dir,
        model_id=args.model_id,
        chunk_duration=args.chunk_duration,
        chunk_stride=args.chunk_stride,
        batch_size=args.batch_size,
        device=args.device,
        splits=args.splits,
    )

    indexer = CLAPIndexer(config)

    if args.mode in ("embed", "all"):
        logger.info(
            f"Starting embedding phase  "
            f"[worker {args.worker_id}/{args.num_workers}]  "
            f"splits={args.splits}"
        )
        newly_embedded = indexer.run_embedding_phase(
            worker_id=args.worker_id,
            num_workers=args.num_workers,
            force=args.force,
        )
        logger.info(f"Embedding phase done. Newly embedded: {newly_embedded}")

    if args.mode in ("index", "all"):
        if args.mode == "index" and args.num_workers > 1 and args.worker_id != 0:
            # Only worker 0 builds the index to avoid races
            logger.info(
                f"Worker {args.worker_id}: skipping FAISS build "
                "(only worker 0 builds the index)."
            )
            return

        logger.info("Starting per-audio FAISS index build...")
        built = indexer.build_faiss_indexes()
        logger.info(
            f"FAISS build complete. "
            f"Newly built: {built}  "
            f"Cache dir: {config.cache_dir}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
