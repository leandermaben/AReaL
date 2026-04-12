"""Werewolf RL workflow for AReaL.

Runs a werewolf game where one wolf uses the trainable model (via
AReaL's OpenAI proxy) and all other agents use fixed external LLMs.
Returns {response_id: reward} for RL credit assignment.
"""

import asyncio
import json
import os
import random
import time
import logging

from openai import AsyncOpenAI

from examples.agent_safety.agent.agent import AsyncAgent
from examples.agent_safety.scenarios.werewolf.config import WerewolfConfig
from examples.agent_safety.scenarios.werewolf.game import WerewolfGame
from examples.agent_safety.scenarios.werewolf.roles import Role, Team

logger = logging.getLogger(__name__)

# Map string role names (from dataset) to Role enum
ROLE_MAP = {
    "werewolf": Role.WEREWOLF,
    "villager": Role.VILLAGER,
    "seer": Role.SEER,
    "witch": Role.WITCH,
    "guard": Role.GUARD,
}


class WerewolfWorkflow:
    """AReaL inline workflow: run a werewolf game, return rewards.

    The OpenAIProxyWorkflow calls ``await workflow.run(data, **extra_kwargs)``
    where extra_kwargs contains the proxy connection info for the trainable
    model. The workflow:

    1. Builds agents from the dataset row's ``agents`` list.
       - The trainable wolf gets an AsyncOpenAI client pointing at the proxy.
       - All other agents get direct AsyncOpenAI clients.
    2. Constructs a WerewolfGame with pre-assigned roles (no shuffle).
    3. Runs the game to completion.
    4. Returns ``{response_id: reward}`` for all trainable-wolf LLM calls.
    """

    def __init__(
        self,
        max_game_days: int = 6,
        timeout: float = 600.0,
        trajectory_log_freq: int = 10,
        trajectory_log_dir: str = "",
        trial_name: str = "",
        **kwargs,
    ):
        self.max_game_days = max_game_days
        self.timeout = timeout
        self.trajectory_log_freq = trajectory_log_freq
        self.trajectory_log_dir = trajectory_log_dir
        self.trial_name = trial_name

    def _should_save_trajectory(self) -> bool:
        if self.trajectory_log_freq <= 0:
            return False
        return random.random() < 1.0 / self.trajectory_log_freq

    def _save_trajectory(
        self,
        game_id: int,
        result,
        agents: dict[str, AsyncAgent],
        trainable_agent_name: str,
        reward: float,
        is_eval: bool,
    ) -> None:
        base_dir = self.trajectory_log_dir or os.environ.get(
            "TRAJECTORY_LOG_DIR", "trajectories"
        )
        log_dir = os.path.join(base_dir, self.trial_name) if self.trial_name else base_dir
        os.makedirs(log_dir, exist_ok=True)
        prefix = "eval" if is_eval else "train"
        path = os.path.join(log_dir, f"{prefix}_trajectories.jsonl")

        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "game_id": game_id,
            "outcome": result.outcome,
            "winner": result.winner.value if hasattr(result.winner, "value") else str(result.winner),
            "day_count": result.day_count,
            "reward": reward,
            "trainable_agent": trainable_agent_name,
            "game_log": result.log,
            "trainable_trajectory": agents[trainable_agent_name].get_trajectory(),
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info("Game %d: saved trajectory to %s", game_id, path)

    async def run(self, data: dict, **extra_kwargs) -> dict[str, float]:
        """Run one werewolf game and return rewards keyed by response ID."""
        # AReaL proxy connection for the trainable model
        proxy_base_url = extra_kwargs.get("base_url")
        proxy_http_client = extra_kwargs.get("http_client")
        proxy_api_key = extra_kwargs.get("api_key")

        agent_specs = data["agents"]
        game_id = data["game_id"]

        # Count roles from agent specs
        num_wolves = sum(1 for a in agent_specs if a["role"] == "werewolf")
        num_villagers_total = len(agent_specs) - num_wolves

        # Determine which specials are present
        roles_present = {a["role"] for a in agent_specs}
        include_seer = "seer" in roles_present
        include_witch = "witch" in roles_present
        include_guard = "guard" in roles_present
        num_plain_villagers = num_villagers_total - sum([
            include_seer, include_witch, include_guard
        ])

        config = WerewolfConfig(
            num_werewolves=num_wolves,
            num_villagers=num_plain_villagers,
            include_seer=include_seer,
            include_witch=include_witch,
            include_guard=include_guard,
            use_sheriff=False,
            max_game_days=self.max_game_days,
            use_random_names=False,
        )

        # Build agents dict and track the trainable wolf
        agents = {}
        trainable_agent_name = None

        for spec in agent_specs:
            name = spec["name"]

            if spec.get("trainable"):
                # Trainable wolf: route through AReaL proxy
                client = AsyncOpenAI(
                    base_url=proxy_base_url,
                    api_key=proxy_api_key,
                    http_client=proxy_http_client,
                    max_retries=0,
                )
                agents[name] = AsyncAgent(
                    client=client,
                    model="trainable",  # model name doesn't matter for proxy
                    system_prompt="",   # set by game._setup_players
                    agent_id=name,
                    temperature=1.0,
                )
                trainable_agent_name = name
            else:
                # Fixed opponent: direct to external API
                extra_body = spec.get("extra_body", {})
                client = AsyncOpenAI(
                    base_url=spec["base_url"],
                    api_key=spec["api_key"],
                )
                agents[name] = AsyncAgent(
                    client=client,
                    model=spec["model"],
                    system_prompt="",
                    agent_id=name,
                    temperature=0.7,
                    extra_body=extra_body if extra_body else None,
                )

        # Build game with pre-assigned roles (override _setup_players)
        game = WerewolfGame(config, seed=game_id)
        _patch_fixed_roles(game, agent_specs)

        # Run the game with timeout
        try:
            result = await asyncio.wait_for(
                game.run(agents), timeout=self.timeout
            )
        except TimeoutError:
            logger.error(
                "Game %d exceeded %ds timeout. Discarding trajectory.",
                game_id, self.timeout,
            )
            raise

        # Compute reward: +1 if wolves win, -1 if they lose
        wolves_won = result.winner == Team.WEREWOLF
        reward = 1.0 if wolves_won else 0

        # Count trainable agent turns
        trainable_agent = agents[trainable_agent_name]
        num_trainable_turns = len(trainable_agent.response_ids)
        total_players = len(agent_specs)

        logger.info(
            "Game %d: %s (day %d, trainable=%s, turns=%d, reward=%.1f)",
            game_id, result.outcome, result.day_count,
            trainable_agent_name, num_trainable_turns, reward,
        )

        # Log metrics to wandb via stats_tracker
        try:
            from areal import workflow_context
            from areal.utils import stats_tracker
            tracker = stats_tracker.get(workflow_context.stat_scope())
            tracker.scalar(
                reward=reward,
                wolves_won=float(wolves_won),
                day_count=float(result.day_count),
                num_trainable_turns=float(num_trainable_turns),
                num_wolves=float(num_wolves),
                num_villagers=float(num_villagers_total),
                total_players=float(total_players),
            )
        except Exception:
            pass

        # Probabilistically save trajectory
        if self._should_save_trajectory():
            try:
                is_eval = False
                try:
                    from areal import workflow_context
                    is_eval = workflow_context.get().is_eval
                except Exception:
                    pass
                self._save_trajectory(
                    game_id=game_id,
                    result=result,
                    agents=agents,
                    trainable_agent_name=trainable_agent_name,
                    reward=reward,
                    is_eval=is_eval,
                )
            except Exception as exc:
                logger.warning("Failed to save trajectory: %s", exc)

        # Collect response IDs from the trainable wolf agent
        if not trainable_agent.response_ids:
            logger.warning("Game %d: trainable agent had no response IDs", game_id)
            return {}

        return {rid: reward for rid in trainable_agent.response_ids}


def _patch_fixed_roles(game: WerewolfGame, agent_specs: list[dict]) -> None:
    """Override _setup_players to use pre-assigned roles from the dataset."""
    from examples.agent_safety.scenarios.werewolf.game import PlayerState
    from examples.agent_safety.scenarios.werewolf.prompts import build_system_prompt

    def fixed_setup():
        agent_ids = list(game.agents.keys())
        game.player_order = agent_ids

        # Map names to specs for role lookup
        spec_by_name = {s["name"]: s for s in agent_specs}

        for pid in agent_ids:
            spec = spec_by_name[pid]
            role = ROLE_MAP[spec["role"]]
            game.players[pid] = PlayerState(agent_id=pid, role=role)
            game.agents[pid].system_prompt = build_system_prompt(
                role, player_name=pid
            )

        for pid, ps in game.players.items():
            game._log(f"[Setup] {pid} is assigned role: {ps.role.value}")

    game._setup_players = fixed_setup
