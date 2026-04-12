"""Benchmark: Qwen3-4B as one werewolf vs various opponent models.

One werewolf (Alice) is always powered by the local Qwen3-4B vllm server.
All other players (including the second werewolf) use the opponent model.
Runs 4 games per opponent model and reports wolf win rates.

Usage:
    python -m examples.agent_safety.tests.benchmark_wolf
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from examples.agent_safety.agent.agent import AsyncAgent
from examples.agent_safety.scenarios.werewolf.config import WerewolfConfig
from examples.agent_safety.scenarios.werewolf.game import WerewolfGame

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ── Config ──────────────────────────────────────────────────────────

LLM_DIR = Path(__file__).parent.parent / "llm"
HOSTED_LLMS = json.loads((LLM_DIR / "hosted_llms.json").read_text())["models"]
HOSTED_KEYS = json.loads((LLM_DIR / "hosted_llm_keys.json").read_text())

# Trainable wolf: Qwen3-4B on local vllm
QWEN_BASE_URL = "http://gpua088:8000/v1/"
QWEN_API_KEY = "not-needed"
QWEN_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# The trainable wolf is always the first player (Alice).
# The game shuffles roles, but we fix Alice as a wolf by overriding
# _setup_players. We do this by assigning roles deterministically.
NAMES = ["Alice", "Bob", "Charlie", "Diana", "Eve"]

# 5 opponent models of varying size/ability
OPPONENT_MODELS = [
    "gpt-4.1-mini",
    "gemini-2.5-flash",
    "claude-haiku-4-5-20251001-v1:0",
    "llama3-2-11b-instruct",
    "gpt-5.4-nano",
]

GAMES_PER_OPPONENT = 4


def _get_model_config(model_name: str) -> dict:
    """Look up a model in hosted_llms.json."""
    for m in HOSTED_LLMS:
        if m["model"] == model_name:
            return m
    raise ValueError(f"Model {model_name!r} not found in hosted_llms.json")


def create_benchmark_game(
    opponent_model: str,
    seed: int,
) -> tuple[WerewolfGame, dict[str, AsyncAgent]]:
    """Create a game where Alice is always the Qwen wolf.

    Config: 2 wolves, 1 seer, 1 witch, 1 villager (no guard, no sheriff).
    Alice is forced to be a werewolf. The remaining roles are shuffled
    among the other 4 players.
    """
    config = WerewolfConfig(
        num_werewolves=2,
        num_villagers=1,
        include_seer=True,
        include_witch=True,
        include_guard=False,
        use_sheriff=False,
        max_game_days=5,
        use_random_names=False,
    )

    game = WerewolfGame(config, seed=seed)

    # Look up opponent model config
    opp = _get_model_config(opponent_model)
    opp_api_key = HOSTED_KEYS.get(opp["api_key"], opp["api_key"])

    agents = {}

    # Alice: Qwen3-4B (trainable wolf)
    agents["Alice"] = AsyncAgent(
        client=AsyncOpenAI(base_url=QWEN_BASE_URL, api_key=QWEN_API_KEY),
        model=QWEN_MODEL,
        system_prompt="",
        agent_id="Alice",
        temperature=0.7,
    )

    # Everyone else: opponent model
    for name in NAMES[1:]:
        agents[name] = AsyncAgent(
            client=AsyncOpenAI(base_url=opp["base_url"], api_key=opp_api_key),
            model=opponent_model,
            system_prompt="",
            agent_id=name,
            temperature=0.7,
        )

    return game, agents


def patch_game_for_fixed_wolf(game: WerewolfGame) -> None:
    """Monkey-patch _setup_players so Alice is always a werewolf.

    The remaining roles (1 werewolf, 1 seer, 1 witch, 1 villager)
    are shuffled among the other 4 players.
    """
    from examples.agent_safety.scenarios.werewolf.prompts import build_system_prompt
    from examples.agent_safety.scenarios.werewolf.roles import Role

    original_setup = game._setup_players

    def fixed_setup():
        from dataclasses import field as _  # noqa: F811

        roles_pool = game.config.build_roles()
        # Remove one werewolf for Alice
        roles_pool.remove(Role.WEREWOLF)
        game.rng.shuffle(roles_pool)

        agent_ids = list(game.agents.keys())
        game.player_order = agent_ids

        # Alice is always werewolf
        from examples.agent_safety.scenarios.werewolf.game import PlayerState

        game.players["Alice"] = PlayerState(agent_id="Alice", role=Role.WEREWOLF)
        game.agents["Alice"].system_prompt = build_system_prompt(
            Role.WEREWOLF, player_name="Alice"
        )

        # Assign remaining roles to other players
        others = [pid for pid in agent_ids if pid != "Alice"]
        for pid, role in zip(others, roles_pool):
            game.players[pid] = PlayerState(agent_id=pid, role=role)
            game.agents[pid].system_prompt = build_system_prompt(
                role, player_name=pid
            )

        for pid, ps in game.players.items():
            game._log(f"[Setup] {pid} is assigned role: {ps.role.value}")

    game._setup_players = fixed_setup


def save_benchmark_log(
    all_results: list[dict],
    output_dir: Path,
) -> None:
    """Save benchmark summary to a JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(all_results, indent=2) + "\n"
    )


async def run_benchmark() -> None:
    logs_root = Path(__file__).parent.parent / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    benchmark_dir = logs_root / f"benchmark_{timestamp}"

    all_results = []

    print(f"{'='*70}")
    print(f"Benchmark: Qwen3-4B (Alice=wolf) vs opponent models")
    print(f"Games per opponent: {GAMES_PER_OPPONENT}")
    print(f"{'='*70}\n")

    for opp_model in OPPONENT_MODELS:
        wolf_wins = 0
        villager_wins = 0
        draws = 0
        game_details = []

        print(f"--- Opponent: {opp_model} ---")

        for game_idx in range(GAMES_PER_OPPONENT):
            seed = hash((opp_model, game_idx)) % (2**31)
            game, agents = create_benchmark_game(opp_model, seed=seed)
            patch_game_for_fixed_wolf(game)

            try:
                result = await game.run(agents)

                # Save individual game logs
                game_dir = benchmark_dir / opp_model / f"game_{game_idx}"
                from examples.agent_safety.tests.test_full_game import save_game_logs

                save_game_logs(result, agents, output_dir=game_dir)

                outcome = result.outcome
                if "Werewolves win" in outcome:
                    wolf_wins += 1
                elif "Villagers win" in outcome:
                    villager_wins += 1
                else:
                    draws += 1

                # Check if Alice (Qwen wolf) survived
                alice_alive = "Alice" in result.alive_at_end
                roles = result.player_roles

                detail = {
                    "game": game_idx,
                    "outcome": outcome,
                    "days": result.day_count,
                    "alice_survived": alice_alive,
                    "roles": roles,
                }
                game_details.append(detail)

                status = "W" if "Werewolves" in outcome else "V" if "Villagers" in outcome else "D"
                print(
                    f"  Game {game_idx+1}: {status} "
                    f"(day {result.day_count}, "
                    f"Alice {'alive' if alice_alive else 'dead'})"
                )

            except Exception as e:
                logger.error(f"Game {game_idx} vs {opp_model} failed: {e}")
                game_details.append({
                    "game": game_idx,
                    "outcome": "ERROR",
                    "error": str(e)[:200],
                })
                print(f"  Game {game_idx+1}: ERROR - {str(e)[:80]}")

        total = wolf_wins + villager_wins + draws
        wolf_rate = wolf_wins / total * 100 if total > 0 else 0

        model_result = {
            "opponent": opp_model,
            "wolf_wins": wolf_wins,
            "villager_wins": villager_wins,
            "draws": draws,
            "wolf_win_rate": round(wolf_rate, 1),
            "games": game_details,
        }
        all_results.append(model_result)
        print(
            f"  Result: Wolves {wolf_wins}/{total} ({wolf_rate:.0f}%) | "
            f"Villagers {villager_wins}/{total}\n"
        )

    # Save full results
    save_benchmark_log(all_results, benchmark_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Opponent Model':<40s} {'Wolf W':>6s} {'Vill W':>6s} {'Draw':>5s} {'Wolf %':>7s}")
    print(f"{'-'*70}")
    for r in all_results:
        print(
            f"{r['opponent']:<40s} "
            f"{r['wolf_wins']:>6d} "
            f"{r['villager_wins']:>6d} "
            f"{r['draws']:>5d} "
            f"{r['wolf_win_rate']:>6.1f}%"
        )
    print(f"{'='*70}")
    print(f"\nLogs saved to: {benchmark_dir}")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
