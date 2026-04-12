"""Integration test: run a small werewolf game with a real LLM.

Uses gpt-4.1-mini via the CMU AI gateway. Marked slow since it makes
real API calls. Can also be run as a standalone script for debugging.
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pytest
from openai import AsyncOpenAI

from examples.agent_safety.agent.agent import AsyncAgent
from examples.agent_safety.scenarios.werewolf.config import WerewolfConfig
from examples.agent_safety.scenarios.werewolf.game import WerewolfGame

# Load LLM config
LLM_DIR = Path(__file__).parent.parent / "llm"
HOSTED_LLMS = json.loads((LLM_DIR / "hosted_llms.json").read_text())["models"]
HOSTED_KEYS = json.loads((LLM_DIR / "hosted_llm_keys.json").read_text())

# Use gpt-4.1-mini for testing (cheap and fast)
GPT_MINI = next(m for m in HOSTED_LLMS if m["model"] == "gpt-4.1-mini")
API_KEY = HOSTED_KEYS.get(GPT_MINI["api_key"], GPT_MINI["api_key"])
BASE_URL = GPT_MINI["base_url"]
MODEL = GPT_MINI["model"]

# Player names for a small game
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve"]


def create_game_and_agents(
    base_url: str = BASE_URL,
    api_key: str = API_KEY,
    model: str = MODEL,
) -> tuple[WerewolfGame, dict[str, AsyncAgent]]:
    """Create a small 5-player game with real LLM agents."""
    config = WerewolfConfig(
        num_werewolves=2,
        num_villagers=1,
        include_seer=True,
        include_witch=True,
        include_guard=False,
        use_sheriff=False,  # Skip sheriff to reduce API calls
        max_game_days=5,
        use_random_names=False,
    )

    game = WerewolfGame(config, seed=42)

    # Create agents — system prompts are set by the game during setup
    # based on the shuffled role assignment.
    agents = {}
    for name in NAMES:
        client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        agents[name] = AsyncAgent(
            client=client,
            model=model,
            system_prompt="",  # overwritten by game._setup_players()
            agent_id=name,
            temperature=0.7,
        )

    return game, agents


def save_game_logs(
    result,
    agents: dict[str, AsyncAgent],
    output_dir: Path | None = None,
) -> Path:
    """Save game log and per-agent trajectories to files.

    Args:
        result: GameResult from the completed game.
        agents: Dict of agent_id -> AsyncAgent with conversation history.
        output_dir: Directory to write logs into. Defaults to
            examples/agent_safety/logs/<timestamp>/

    Returns:
        Path to the output directory.
    """
    if output_dir is None:
        logs_root = Path(__file__).parent.parent / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = logs_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- game_summary.txt ---
    summary_lines = [
        f"GAME OVER: {result.outcome}",
        f"Days played: {result.day_count}",
        f"Alive at end: {', '.join(result.alive_at_end)}",
        "",
        "Role assignments:",
    ]
    for pid, role in result.player_roles.items():
        alive = "alive" if pid in result.alive_at_end else "dead"
        summary_lines.append(f"  {pid}: {role} ({alive})")
    summary_lines.append("")
    summary_lines.append(f"Game log ({len(result.game_log)} entries):")
    for entry in result.game_log:
        summary_lines.append(f"  {entry}")
    (output_dir / "game_summary.txt").write_text("\n".join(summary_lines) + "\n")

    # --- game_log.json ---
    game_log_data = {
        "outcome": result.outcome,
        "winner": result.winner.value if result.winner else None,
        "day_count": result.day_count,
        "alive_at_end": result.alive_at_end,
        "player_roles": {
            k: (v.value if hasattr(v, "value") else v)
            for k, v in result.player_roles.items()
        },
        "game_log": result.game_log,
    }
    (output_dir / "game_log.json").write_text(
        json.dumps(game_log_data, indent=2) + "\n"
    )

    # --- per-agent trajectory files ---
    traj_dir = output_dir / "trajectories"
    traj_dir.mkdir(exist_ok=True)
    for name, agent in agents.items():
        traj = agent.get_trajectory()
        (traj_dir / f"{name}.json").write_text(json.dumps(traj, indent=2) + "\n")

    return output_dir


async def run_game() -> None:
    """Run a full game and save results to log files."""
    game, agents = create_game_and_agents()
    result = await game.run(agents)

    output_dir = save_game_logs(result, agents)
    print(f"Game logs saved to: {output_dir}")

    # Also print summary to stdout
    print(f"\n{'='*60}")
    print(f"GAME OVER: {result.outcome}")
    print(f"Days played: {result.day_count}")
    print(f"Alive at end: {', '.join(result.alive_at_end)}")
    print(f"\nRole assignments:")
    for pid, role in result.player_roles.items():
        alive = "alive" if pid in result.alive_at_end else "dead"
        print(f"  {pid}: {role} ({alive})")


@pytest.mark.slow
@pytest.mark.asyncio
async def test_full_game_completes():
    """Run a small game and verify it completes with a valid result."""
    game, agents = create_game_and_agents()
    result = await game.run(agents)

    # Game should complete
    assert result is not None
    assert result.outcome in ("Werewolves win", "Villagers win", "Draw (max days reached)")
    assert result.day_count >= 1

    # All agents should have non-empty trajectories
    for name, agent in agents.items():
        traj = agent.get_trajectory()
        assert len(traj) >= 2, f"{name} has too few messages"

    # Role assignments should be complete
    assert set(result.player_roles.keys()) == set(NAMES)

    # Game log should have entries
    assert len(result.game_log) > 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run_game())
