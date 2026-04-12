"""GRPO training script for werewolf RL.

Usage:
    python -m examples.agent_safety.rl.train \
        --config examples/agent_safety/rl/config_2gpu.yaml

The workflow is a WerewolfWorkflow (not a RolloutWorkflow), so AReaL
wraps it automatically with OpenAIProxyWorkflow using the rollout.openai
configuration from the YAML.
"""

import json
import sys
import time

from datasets import Dataset

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config
from areal.utils.logging import getLogger

from .configs import WerewolfGRPOConfig

logger = getLogger("WerewolfTrain")


def load_werewolf_dataset(path: str) -> Dataset:
    """Load a werewolf game config dataset from a JSONL file.

    Each row must contain: game_id, agents, messages, num_wolves, num_villagers.
    """
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))

    # Duplicate if fewer than 128 items for efficient batching
    if len(rows) < 128:
        original = rows.copy()
        while len(rows) < 128:
            rows.extend(original)

    dataset = Dataset.from_list(rows)
    logger.info(f"Loaded {len(dataset)} game configs from {path}")
    return dataset


def main(args):
    config, _ = load_expr_config(args, WerewolfGRPOConfig)
    econfig = config.econfig

    # Add timestamp to trial_name so each run gets unique log/checkpoint dirs
    ts = time.strftime("%Y%m%d_%H%M%S")
    config.trial_name = f"{config.trial_name}_{ts}"

    # Set wandb run name with timestamp if not already set
    if not config.stats_logger.wandb.name:
        config.stats_logger.wandb.name = config.trial_name

    # Log key config values
    logger.info(
        f"adv_norm: mean_level={config.actor.adv_norm.mean_level}, "
        f"std_level={config.actor.adv_norm.std_level}, "
        f"group_size={config.actor.adv_norm.group_size}"
    )
    logger.info(
        f"n_samples={config.gconfig.n_samples}, kl_ctl={config.actor.kl_ctl}, "
        f"max_game_days={econfig.max_game_days}, timeout={econfig.timeout}"
    )

    # Load datasets from JSONL files
    train_dataset = load_werewolf_dataset(config.train_dataset.path)
    valid_dataset = load_werewolf_dataset(config.valid_dataset.path)

    # Build workflow kwargs
    workflow_kwargs = dict(
        max_game_days=econfig.max_game_days,
        timeout=econfig.timeout,
        trajectory_log_freq=config.trajectory_log_freq,
        trajectory_log_dir=config.trajectory_log_dir or None,
        trial_name=config.trial_name,
    )

    with PPOTrainer(
        config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
    ) as trainer:
        trainer.train(
            workflow="examples.agent_safety.rl.werewolf_workflow.WerewolfWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.agent_safety.rl.werewolf_workflow.WerewolfWorkflow",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
