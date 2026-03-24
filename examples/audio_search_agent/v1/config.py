"""Config dataclass for the v1 audio search agent training experiment.

Kept in a separate file with minimal imports so the RPC server worker
subprocess can import and reconstruct it without pulling in PPOTrainer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AudioSearchV1Config(GRPOConfig):
    """GRPO configuration for the v1 audio search agent."""

    # Agent
    max_search_turns: int = field(
        default=12, metadata={"help": "Maximum tool-use turns per episode."},
    )
    step_limit: int = field(
        default=12,
        metadata={"help": "Exact turn count required for non-zero reward."},
    )

    # Data
    meetingbank_data_root: str = field(
        default="/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared",
        metadata={"help": "Path to meeting_bank_prepared directory."},
    )
    meetingbank_split: str = field(
        default="val", metadata={"help": "Which split to use (val/train/test)."},
    )
    max_nonfactual_ratio: float = field(
        default=0.1,
        metadata={"help": "Max proportion of non-factual questions in the mix."},
    )

    # Validation set caps (to limit eval compute in early runs)
    val_max_factual: int = field(
        default=0,
        metadata={"help": "If > 0, cap val factual questions to this number."},
    )
    val_max_nonfactual: int = field(
        default=0,
        metadata={"help": "If > 0, cap val non-factual questions."},
    )

    # CLAP
    clap_cache_dir: str = field(
        default="/work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank",
        metadata={"help": "Path to pre-built CLAP FAISS index cache."},
    )
    clap_device: str = field(
        default="cpu", metadata={"help": "Device for CLAP model (cpu/cuda)."},
    )

    # Omni probe (optional external vLLM server)
    omni_url: str = field(
        default="", metadata={"help": "Omni vLLM server URL. Empty = disabled."},
    )
    omni_model: str = field(
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        metadata={"help": "Omni model name."},
    )
    omni_slice_tmpdir: str = field(
        default="",
        metadata={"help": "Shared tmpdir for Omni audio slices. Empty = system tmp."},
    )
    iou_threshold: float = field(
        default=0.5,
        metadata={"help": "Minimum IoU to count a predicted span as matching a gold span."},
    )
