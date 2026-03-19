"""
GRPO training script for the AudioSearchAgent.

Usage:
    python -m examples.audio_search_agent.train \\
        --config examples/audio_search_agent/config_2gpu.yaml

The workflow is an AudioSearchAgent (not a RolloutWorkflow), so AReaL
wraps it automatically with OpenAIProxyWorkflow using the rollout.openai
configuration from the YAML.

Environment variables consumed by the agent's DB connection:
    POSTGRES_HOST     (default: localhost)
    POSTGRES_PORT     (default: 5433)
    POSTGRES_DB       (default: espnet_db)
    POSTGRES_USER     (default: espnet_user)
    POSTGRES_PASSWORD (default: espnet_password)
"""
from __future__ import annotations

import os
import sys

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config

# Config class lives in a separate lightweight module so the RPC server
# subprocess can import and reconstruct it without pulling in PPOTrainer.
from examples.audio_search_agent.config import AudioSearchGRPOConfig

# Ensure DB port default is 5433 (can be overridden by env var)
os.environ.setdefault("POSTGRES_PORT", "5433")


def main(args):
    config, _ = load_expr_config(args, AudioSearchGRPOConfig)

    # Propagate DB config to environment so the agent picks it up
    os.environ["POSTGRES_HOST"] = config.postgres_host
    os.environ["POSTGRES_PORT"] = str(config.postgres_port)
    os.environ["POSTGRES_DB"] = config.postgres_db
    os.environ["POSTGRES_USER"] = config.postgres_user
    os.environ["POSTGRES_PASSWORD"] = config.postgres_password

    from examples.audio_search_agent.dataset import get_audio_qa_dataset

    train_dataset = get_audio_qa_dataset(
        data_dirs=config.data_dirs or None,
        split="train",
        train_ratio=config.train_ratio,
        postgres_host=config.postgres_host,
        postgres_port=config.postgres_port,
        filter_to_ingested=config.filter_to_ingested,
    )
    valid_dataset = get_audio_qa_dataset(
        data_dirs=config.data_dirs or None,
        split="test",
        train_ratio=config.train_ratio,
        postgres_host=config.postgres_host,
        postgres_port=config.postgres_port,
        filter_to_ingested=config.filter_to_ingested,
    )

    workflow_kwargs = dict(
        max_search_turns=config.max_search_turns,
        postgres_host=config.postgres_host,
        postgres_port=config.postgres_port,
        postgres_db=config.postgres_db,
        postgres_user=config.postgres_user,
        postgres_password=config.postgres_password,
    )

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="examples.audio_search_agent.agent.AudioSearchAgent",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.audio_search_agent.agent.AudioSearchAgent",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
