"""Config dataclass for the audio search agent training experiment.

Kept in a separate file with minimal imports so the RPC server worker
subprocess can import and reconstruct it during configuration without
pulling in PPOTrainer or other heavy dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class AudioSearchGRPOConfig(GRPOConfig):
    """GRPO configuration for the audio search agent."""

    max_search_turns: int = field(
        default=5,
        metadata={"help": "Maximum SQL search steps per rollout episode."},
    )
    postgres_host: str = field(default="localhost", metadata={"help": "PostgreSQL host."})
    postgres_port: int = field(default=5433, metadata={"help": "PostgreSQL port."})
    postgres_db: str = field(default="espnet_db", metadata={"help": "PostgreSQL database name."})
    postgres_user: str = field(default="espnet_user", metadata={"help": "PostgreSQL user."})
    postgres_password: str = field(
        default="espnet_password", metadata={"help": "PostgreSQL password."}
    )
    data_dirs: list[str] = field(
        default_factory=list,
        metadata={
            "help": (
                "List of LongformAudio QA directories to load. "
                "Defaults to qa-partI_10min, qa-partII_10min, "
                "qa-partI_30min, qa-partII_30min under LONGFORM_ROOT."
            )
        },
    )
    train_ratio: float = field(
        default=0.9, metadata={"help": "Fraction of data used for training."}
    )
    filter_to_ingested: bool = field(
        default=True,
        metadata={"help": "Skip samples whose audio_id is not yet in the DB."},
    )
