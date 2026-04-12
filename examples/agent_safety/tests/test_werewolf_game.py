"""Unit tests for werewolf game logic with mock agents (no LLM calls)."""

import json
from collections import Counter
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from examples.agent_safety.scenarios.werewolf.config import WerewolfConfig
from examples.agent_safety.scenarios.werewolf.game import (
    GameResult,
    NightResult,
    PlayerState,
    WerewolfGame,
)
from examples.agent_safety.scenarios.werewolf.roles import Role, Team
from examples.agent_safety.scenarios.werewolf.prompts import build_system_prompt
from examples.agent_safety.scenarios.werewolf.tools import (
    guard_protect_tool,
    seer_check_tool,
    vote_tool,
    werewolf_kill_tool,
)


# ── Config Tests ─────────────────────────────────────────────────────


class TestWerewolfConfig:
    def test_build_roles_default(self):
        config = WerewolfConfig()
        roles = config.build_roles()
        role_counts = Counter(roles)
        assert role_counts[Role.WEREWOLF] == 2
        assert role_counts[Role.VILLAGER] == 3
        assert role_counts[Role.SEER] == 1
        assert role_counts[Role.WITCH] == 1
        assert role_counts[Role.GUARD] == 1
        assert config.total_players == 8

    def test_build_roles_no_specials(self):
        config = WerewolfConfig(
            include_seer=False, include_witch=False, include_guard=False
        )
        roles = config.build_roles()
        role_counts = Counter(roles)
        assert role_counts[Role.SEER] == 0
        assert config.total_players == 5

    def test_build_roles_custom(self):
        config = WerewolfConfig(num_werewolves=3, num_villagers=5)
        roles = config.build_roles()
        role_counts = Counter(roles)
        assert role_counts[Role.WEREWOLF] == 3
        assert role_counts[Role.VILLAGER] == 5


# ── Prompt Tests ─────────────────────────────────────────────────────


class TestPrompts:
    def test_system_prompt_contains_rules(self):
        prompt = build_system_prompt(Role.VILLAGER)
        assert "GAME RULES" in prompt
        assert "WIN CONDITIONS" in prompt

    def test_system_prompt_contains_role(self):
        for role in Role:
            prompt = build_system_prompt(role)
            assert role.value.upper() in prompt.upper()


# ── Tool Tests ───────────────────────────────────────────────────────


class TestTools:
    def test_werewolf_kill_tool_format(self):
        tools = werewolf_kill_tool(["Alice", "Bob"])
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "werewolf_kill"
        params = tools[0]["function"]["parameters"]["properties"]
        assert "target" in params
        assert "reasoning" in params

    def test_guard_protect_tool_mentions_restriction(self):
        tools = guard_protect_tool(["Alice", "Bob"], "Alice")
        desc = tools[0]["function"]["description"]
        assert "Alice" in desc

    def test_vote_tool_lists_players(self):
        tools = vote_tool(["X", "Y", "Z"])
        desc = tools[0]["function"]["parameters"]["properties"]["target"]["description"]
        assert "X" in desc and "Y" in desc and "Z" in desc


# ── Game State Tests ─────────────────────────────────────────────────


class TestGameState:
    def _make_game(self, seed=42):
        config = WerewolfConfig(
            num_werewolves=2,
            num_villagers=2,
            include_seer=True,
            include_witch=False,
            include_guard=False,
            use_sheriff=False,
        )
        return WerewolfGame(config, seed=seed)

    def test_alive_players(self):
        game = self._make_game()
        # Manually set up players
        game.player_order = ["A", "B", "C", "D", "E"]
        game.players = {
            "A": PlayerState("A", Role.WEREWOLF, alive=True),
            "B": PlayerState("B", Role.WEREWOLF, alive=True),
            "C": PlayerState("C", Role.SEER, alive=True),
            "D": PlayerState("D", Role.VILLAGER, alive=False),
            "E": PlayerState("E", Role.VILLAGER, alive=True),
        }
        assert game.alive_players == ["A", "B", "C", "E"]
        assert game.alive_wolves == ["A", "B"]
        assert game.alive_villager_team == ["C", "E"]

    def test_check_termination_wolves_win(self):
        game = self._make_game()
        game.player_order = ["A", "B", "C"]
        game.players = {
            "A": PlayerState("A", Role.WEREWOLF, alive=True),
            "B": PlayerState("B", Role.WEREWOLF, alive=True),
            "C": PlayerState("C", Role.VILLAGER, alive=True),
        }
        result = game.check_termination()
        assert result is not None
        assert result.winner == Team.WEREWOLF

    def test_check_termination_villagers_win(self):
        game = self._make_game()
        game.player_order = ["A", "B", "C"]
        game.players = {
            "A": PlayerState("A", Role.WEREWOLF, alive=False),
            "B": PlayerState("B", Role.VILLAGER, alive=True),
            "C": PlayerState("C", Role.SEER, alive=True),
        }
        result = game.check_termination()
        assert result is not None
        assert result.winner == Team.VILLAGER

    def test_check_termination_game_continues(self):
        game = self._make_game()
        game.player_order = ["A", "B", "C"]
        game.players = {
            "A": PlayerState("A", Role.WEREWOLF, alive=True),
            "B": PlayerState("B", Role.VILLAGER, alive=True),
            "C": PlayerState("C", Role.SEER, alive=True),
        }
        assert game.check_termination() is None

    def test_validate_target(self):
        game = self._make_game()
        assert game._validate_target("Alice", ["Alice", "Bob"]) == "Alice"
        assert game._validate_target("alice", ["Alice", "Bob"]) == "Alice"
        assert game._validate_target("skip", ["Alice", "Bob"]) is None
        assert game._validate_target("none", ["Alice", "Bob"]) is None
        assert game._validate_target("Charlie", ["Alice", "Bob"]) is None

    def test_wolf_consensus_top_choice(self):
        game = self._make_game()
        # 2 out of 3 agree — unique top choice
        choices = {"W1": "Alice", "W2": "Alice", "W3": "Bob"}
        assert game._check_wolf_consensus(choices) == "Alice"

    def test_wolf_consensus_single_vote(self):
        game = self._make_game()
        # Only one wolf, unique top choice
        choices = {"W1": "Alice"}
        assert game._check_wolf_consensus(choices) == "Alice"

    def test_wolf_consensus_three_way_tie(self):
        game = self._make_game()
        choices = {"W1": "Alice", "W2": "Bob", "W3": "Charlie"}
        assert game._check_wolf_consensus(choices) is None

    def test_wolf_consensus_two_way_tie(self):
        game = self._make_game()
        choices = {"W1": "Alice", "W2": "Bob"}
        assert game._check_wolf_consensus(choices) is None

    def test_wolf_consensus_all_skip(self):
        game = self._make_game()
        choices = {"W1": None, "W2": None}
        assert game._check_wolf_consensus(choices) is None

    def test_eliminate_player(self):
        game = self._make_game()
        game.player_order = ["A"]
        game.players = {"A": PlayerState("A", Role.VILLAGER, alive=True)}
        game._eliminate_player("A", "test")
        assert not game.players["A"].alive
        assert any("eliminated" in log for log in game.game_log)
