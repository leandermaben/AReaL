"""Configuration dataclasses for werewolf RL training."""

from dataclasses import dataclass, field

from areal.api.cli_args import GRPOConfig


@dataclass
class WerewolfEnvConfig:
    """Environment configuration for the werewolf game.

    Attributes:
        max_game_days: Maximum number of day/night cycles before game ends.
        timeout: Maximum seconds per game episode before timeout.
    """

    max_game_days: int = field(
        default=6,
        metadata={"help": "Maximum number of day/night cycles per game."},
    )
    timeout: float = field(
        default=600.0,
        metadata={"help": "Maximum seconds per game episode."},
    )


@dataclass
class WerewolfGRPOConfig(GRPOConfig):
    """GRPO configuration for werewolf RL training."""

    econfig: WerewolfEnvConfig = field(default_factory=WerewolfEnvConfig)

    trajectory_log_freq: int = field(
        default=10,
        metadata={"help": "Save a trajectory every N episodes. 0 = disabled."},
    )
    trajectory_log_dir: str = field(
        default="",
        metadata={"help": "Directory for trajectory logs. Empty = 'trajectories' in cwd."},
    )
