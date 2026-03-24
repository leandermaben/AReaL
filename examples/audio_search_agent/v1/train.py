"""GRPO training script for the v1 AudioSearchAgent.

Usage:
    python -m examples.audio_search_agent.v1.train \
        --config examples/audio_search_agent/v1/config_2gpu.yaml

The workflow is an AudioSearchWorkflow (not a RolloutWorkflow), so AReaL
wraps it automatically with OpenAIProxyWorkflow using the rollout.openai
configuration from the YAML.
"""
from __future__ import annotations

import sys
import time

from areal import PPOTrainer
from areal.api.cli_args import load_expr_config

from .config import AudioSearchV1Config


def main(args):
    config, _ = load_expr_config(args, AudioSearchV1Config)

    # Set wandb run name with timestamp if not already set
    if not config.stats_logger.wandb.name:
        ts = time.strftime("%Y%m%d_%H%M%S")
        config.stats_logger.wandb.name = f"{config.trial_name}_{ts}"

    from .dataset import get_meetingbank_dataset

    # Train: full filtered set (no caps)
    train_dataset = get_meetingbank_dataset(
        data_root=config.meetingbank_data_root,
        split=config.meetingbank_split,
        max_nonfactual_ratio=config.max_nonfactual_ratio,
    )
    # Val: optionally capped to reduce eval compute
    valid_dataset = get_meetingbank_dataset(
        data_root=config.meetingbank_data_root,
        split=config.meetingbank_split,
        max_nonfactual_ratio=config.max_nonfactual_ratio,
        max_factual=config.val_max_factual,
        max_nonfactual=config.val_max_nonfactual,
    )

    workflow_kwargs = dict(
        max_search_turns=config.max_search_turns,
        step_limit=config.step_limit,
        clap_cache_dir=config.clap_cache_dir,
        data_root=config.meetingbank_data_root,
        clap_device=config.clap_device,
        omni_url=config.omni_url or None,
        omni_model=config.omni_model,
        omni_slice_tmpdir=config.omni_slice_tmpdir or None,
        aux_weight=config.aux_weight,
        answer_weight=config.answer_weight,
    )

    with PPOTrainer(config, train_dataset=train_dataset, valid_dataset=valid_dataset) as trainer:
        trainer.train(
            workflow="examples.audio_search_agent.v1.workflow.AudioSearchWorkflow",
            workflow_kwargs=workflow_kwargs,
            eval_workflow="examples.audio_search_agent.v1.workflow.AudioSearchWorkflow",
            eval_workflow_kwargs=workflow_kwargs,
        )


if __name__ == "__main__":
    main(sys.argv[1:])
