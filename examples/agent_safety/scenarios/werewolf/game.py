"""Werewolf game engine.

Orchestrates the full werewolf game loop: night phases (guard, wolves, seer,
witch) and day phases (announcements, sheriff election, speeches, voting).

Each agent receives observations as user messages and responds via tool calls.
The game engine manages state, validates actions, and determines outcomes.
"""

import logging
import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from ...agent.agent import AsyncAgent
from .config import WerewolfConfig
from .prompts import build_system_prompt
from .roles import ROLE_TEAM, Role, Team
from .tools import (
    badge_flow_tool,
    decide_speech_order_tool,
    guard_protect_tool,
    last_words_tool,
    run_for_sheriff_tool,
    seer_check_tool,
    sheriff_speech_tool,
    speech_tool,
    vote_for_sheriff_tool,
    vote_tool,
    werewolf_discuss_tool,
    werewolf_kill_tool,
    witch_action_tool,
)

logger = logging.getLogger(__name__)


@dataclass
class PlayerState:
    """Per-player game state."""

    agent_id: str
    role: Role
    alive: bool = True
    # Guard tracking
    guard_last_protected: str | None = None
    # Witch tracking
    has_antidote: bool = True
    has_poison: bool = True
    # Seer tracking
    check_history: dict[int, dict[str, str]] = field(default_factory=dict)


@dataclass
class NightResult:
    """Resolved results of a night phase."""

    wolf_target: str | None = None
    guard_target: str | None = None
    seer_target: str | None = None
    seer_result: str | None = None  # "werewolf" or "not a werewolf"
    witch_used_antidote: bool = False
    witch_poison_target: str | None = None
    deaths: list[str] = field(default_factory=list)


@dataclass
class GameResult:
    """Final result of a completed game."""

    winner: Team
    outcome: str  # "Werewolves win" or "Villagers win"
    day_count: int
    game_log: list[str]
    player_roles: dict[str, str]  # agent_id -> role name
    alive_at_end: list[str]


class WerewolfGame:
    """Orchestrates a complete werewolf game."""

    def __init__(self, config: WerewolfConfig, seed: int | None = None):
        self.config = config
        self.rng = random.Random(seed)
        self.players: dict[str, PlayerState] = {}
        self.agents: dict[str, AsyncAgent] = {}
        self.player_order: list[str] = []  # ordered list of agent IDs
        self.sheriff: str | None = None
        self.day_count: int = 0
        self.game_log: list[str] = []

    def _log(self, msg: str) -> None:
        self.game_log.append(msg)
        logger.info(msg)

    @property
    def alive_players(self) -> list[str]:
        return [pid for pid in self.player_order if self.players[pid].alive]

    @property
    def alive_wolves(self) -> list[str]:
        return [
            pid
            for pid in self.alive_players
            if self.players[pid].role == Role.WEREWOLF
        ]

    @property
    def alive_villager_team(self) -> list[str]:
        return [
            pid
            for pid in self.alive_players
            if ROLE_TEAM[self.players[pid].role] == Team.VILLAGER
        ]

    def _get_player_by_role(self, role: Role) -> str | None:
        """Get alive player with given role, or None."""
        for pid in self.alive_players:
            if self.players[pid].role == role:
                return pid
        return None

    def _validate_target(self, target: str, valid: list[str]) -> str | None:
        """Validate a target is in the valid list. Returns None if invalid."""
        target = target.strip()
        if target.lower() in ("skip", "none", "abstain", "null"):
            return None
        # Fuzzy match: case-insensitive
        for v in valid:
            if v.lower() == target.lower():
                return v
        # Try partial match
        for v in valid:
            if target.lower() in v.lower() or v.lower() in target.lower():
                return v
        logger.warning("Invalid target '%s', valid: %s", target, valid)
        return None

    def check_termination(self) -> GameResult | None:
        """Check if the game has ended."""
        wolves = self.alive_wolves
        villagers = self.alive_villager_team

        if not wolves:
            return GameResult(
                winner=Team.VILLAGER,
                outcome="Villagers win",
                day_count=self.day_count,
                game_log=self.game_log,
                player_roles={pid: ps.role.value for pid, ps in self.players.items()},
                alive_at_end=self.alive_players,
            )

        if len(wolves) >= len(villagers):
            return GameResult(
                winner=Team.WEREWOLF,
                outcome="Werewolves win",
                day_count=self.day_count,
                game_log=self.game_log,
                player_roles={pid: ps.role.value for pid, ps in self.players.items()},
                alive_at_end=self.alive_players,
            )

        return None

    def _eliminate_player(self, pid: str, reason: str) -> None:
        """Mark a player as dead."""
        self.players[pid].alive = False
        self._log(f"{pid} has been eliminated ({reason}).")

    # ── Night Phase ────────────��───────────────��─────────────────────

    async def _guard_action(self) -> str | None:
        """Guard chooses a player to protect. Returns protected player ID."""
        guard_id = self._get_player_by_role(Role.GUARD)
        if not guard_id:
            return None

        ps = self.players[guard_id]
        last = ps.guard_last_protected
        valid_targets = [p for p in self.alive_players if p != last]

        obs_parts = [
            f"Night {self.day_count}. As the Guard, choose a player to protect.",
            f"Alive players: {', '.join(self.alive_players)}.",
        ]
        if last:
            obs_parts.append(
                f"You protected {last} last night and cannot protect them again."
            )

        action = await self.agents[guard_id].act(
            "\n".join(obs_parts),
            guard_protect_tool(valid_targets, last),
        )

        target = self._validate_target(action.get("target", ""), valid_targets)
        if target:
            ps.guard_last_protected = target
            self._log(f"[Night {self.day_count}] Guard protects {target}.")
        else:
            ps.guard_last_protected = None
            self._log(f"[Night {self.day_count}] Guard failed to protect anyone.")
        return target

    async def _werewolf_action(self) -> str | None:
        """Werewolves discuss and vote on a kill target. Returns target or None."""
        wolves = self.alive_wolves
        if not wolves:
            return None

        non_wolf_alive = [p for p in self.alive_players if p not in wolves]
        all_targets = self.alive_players  # wolves can target anyone (including self-knife)

        # Round 1: individual proposals
        wolf_choices: dict[str, str | None] = {}
        wolf_messages: dict[str, str] = {}
        for wid in wolves:
            others = [w for w in wolves if w != wid]
            obs = (
                f"Night {self.day_count}. You are a werewolf.\n"
                f"Your fellow werewolves: {', '.join(others) if others else 'none'}.\n"
                f"Alive players: {', '.join(self.alive_players)}.\n"
                f"Choose a player to target for elimination tonight. "
                f"Include a message to your fellow werewolves."
            )
            action = await self.agents[wid].act(obs, werewolf_kill_tool(all_targets))
            target = self._validate_target(action.get("target", ""), all_targets)
            wolf_choices[wid] = target
            wolf_messages[wid] = action.get("message", "")

        # Check consensus
        consensus = self._check_wolf_consensus(wolf_choices)
        if consensus:
            self._log(
                f"[Night {self.day_count}] Werewolves agree to target {consensus}."
            )
            return consensus

        # Discussion rounds
        for rnd in range(2, self.config.max_wolf_discussion_rounds + 1):
            # Build summary of previous round's choices and messages
            round_summary_parts = []
            for w in wolves:
                t = wolf_choices.get(w, "skip") or "skip"
                msg = wolf_messages.get(w, "")
                if msg:
                    round_summary_parts.append(f"{w} chose {t} and said: \"{msg}\"")
                else:
                    round_summary_parts.append(f"{w} chose {t}")
            round_summary = "\n".join(round_summary_parts)

            remaining = self.config.max_wolf_discussion_rounds - rnd + 1

            wolf_choices = {}
            wolf_messages = {}
            for wid in wolves:
                obs = (
                    f"Werewolf discussion round {rnd}. "
                    f"Rounds remaining: {remaining}.\n"
                    f"Previous round:\n{round_summary}\n"
                    f"You must try to reach consensus. "
                    f"If no consensus after all rounds, no one is killed."
                )
                action = await self.agents[wid].act(
                    obs, werewolf_discuss_tool(all_targets)
                )
                target = self._validate_target(action.get("target", ""), all_targets)
                wolf_choices[wid] = target
                wolf_messages[wid] = action.get("message", "")

            consensus = self._check_wolf_consensus(wolf_choices)
            if consensus:
                self._log(
                    f"[Night {self.day_count}] Werewolves reach consensus "
                    f"on {consensus} (round {rnd})."
                )
                return consensus

        self._log(
            f"[Night {self.day_count}] Werewolves failed to reach consensus. No kill."
        )
        return None

    def _check_wolf_consensus(
        self, choices: dict[str, str | None]
    ) -> str | None:
        """Check if werewolves agreed on a target (top choice, no tie)."""
        targets = [t for t in choices.values() if t is not None]
        if not targets:
            return None
        counts = Counter(targets)
        most_common = counts.most_common(2)
        top_target, top_count = most_common[0]
        # Accept if there's a unique top choice (no tie)
        if len(most_common) == 1 or top_count > most_common[1][1]:
            return top_target
        return None

    async def _seer_action(self) -> tuple[str | None, str | None]:
        """Seer checks a player. Returns (target, result)."""
        seer_id = self._get_player_by_role(Role.SEER)
        if not seer_id:
            return None, None

        ps = self.players[seer_id]
        valid = [p for p in self.alive_players if p != seer_id]

        history_lines = []
        for night, info in sorted(ps.check_history.items()):
            history_lines.append(
                f"Night {night}: Checked {info['player']} → {info['result']}"
            )
        history_str = "\n".join(history_lines) if history_lines else "No checks yet."

        obs = (
            f"Night {self.day_count}. As the Seer, choose a player to check.\n"
            f"Alive players (excluding yourself): {', '.join(valid)}.\n"
            f"Your check history:\n{history_str}"
        )
        action = await self.agents[seer_id].act(obs, seer_check_tool(valid))
        target = self._validate_target(action.get("target", ""), valid)

        if target:
            is_wolf = self.players[target].role == Role.WEREWOLF
            result = "werewolf" if is_wolf else "not a werewolf"
            ps.check_history[self.day_count] = {"player": target, "result": result}

            # Inform seer of result via observation (no action needed)
            self.agents[seer_id].add_observation(
                f"Seer check result: {target} is {result}."
            )
            self._log(
                f"[Night {self.day_count}] Seer checks {target}: {result}."
            )
            return target, result

        self._log(f"[Night {self.day_count}] Seer failed to check anyone.")
        return None, None

    async def _witch_action(self, wolf_target: str | None) -> NightResult:
        """Witch decides on antidote/poison. Returns updated NightResult partial."""
        witch_id = self._get_player_by_role(Role.WITCH)
        result = NightResult(wolf_target=wolf_target)

        if not witch_id:
            return result

        ps = self.players[witch_id]
        killed_name = wolf_target if wolf_target else None
        alive_for_poison = [p for p in self.alive_players if p != witch_id]

        obs_parts = [
            f"Night {self.day_count}. As the Witch, decide on your potions.",
        ]
        if killed_name:
            obs_parts.append(f"Tonight, {killed_name} was killed by werewolves.")
        else:
            obs_parts.append("No one was killed by werewolves tonight.")
        if ps.has_antidote:
            obs_parts.append("You have 1 antidote remaining.")
        else:
            obs_parts.append("You have no antidote remaining.")
        if ps.has_poison:
            obs_parts.append("You have 1 poison remaining.")
        else:
            obs_parts.append("You have no poison remaining.")
        obs_parts.append(f"Alive players: {', '.join(self.alive_players)}.")

        action = await self.agents[witch_id].act(
            "\n".join(obs_parts),
            witch_action_tool(alive_for_poison, killed_name, ps.has_antidote, ps.has_poison),
        )

        use_antidote = action.get("use_antidote", False)
        use_poison = action.get("use_poison", False)
        poison_target_raw = action.get("poison_target", "none")

        # Cannot use both
        if use_antidote and use_poison:
            use_poison = False
            logger.warning("Witch tried to use both potions; ignoring poison.")

        if use_antidote and ps.has_antidote and killed_name:
            result.witch_used_antidote = True
            ps.has_antidote = False
            self._log(
                f"[Night {self.day_count}] Witch uses antidote to save {killed_name}."
            )

        if use_poison and ps.has_poison:
            ptarget = self._validate_target(poison_target_raw, alive_for_poison)
            if ptarget:
                result.witch_poison_target = ptarget
                ps.has_poison = False
                self._log(
                    f"[Night {self.day_count}] Witch uses poison on {ptarget}."
                )

        return result

    async def _night_phase(self) -> NightResult:
        """Execute the full night phase."""
        self._log(f"=== Night {self.day_count} ===")

        # 1. Guard protects
        guard_target = await self._guard_action()

        # 2. Werewolves choose target
        wolf_target = await self._werewolf_action()

        # 3. Seer checks
        seer_target, seer_result = await self._seer_action()

        # 4. Witch decides
        night_result = await self._witch_action(wolf_target)
        night_result.guard_target = guard_target
        night_result.seer_target = seer_target
        night_result.seer_result = seer_result

        # 5. Resolve deaths
        deaths = []

        # Wolf kill (unless guarded or saved by witch)
        if wolf_target:
            protected = wolf_target == guard_target
            saved = night_result.witch_used_antidote
            if not protected and not saved:
                deaths.append(wolf_target)
            elif protected:
                self._log(
                    f"[Night {self.day_count}] Guard successfully protected "
                    f"{wolf_target} from werewolf attack!"
                )
            elif saved:
                self._log(
                    f"[Night {self.day_count}] Witch saved {wolf_target} "
                    f"with antidote!"
                )

        # Witch poison (guard cannot protect against poison)
        if night_result.witch_poison_target:
            if night_result.witch_poison_target not in deaths:
                deaths.append(night_result.witch_poison_target)

        night_result.deaths = deaths
        for d in deaths:
            self._eliminate_player(d, "killed at night")

        return night_result

    # ── Day Phase ─────────────────────────────���──────────────────────

    async def _announce_deaths(self, night_result: NightResult) -> None:
        """Announce night deaths and collect last words."""
        if not night_result.deaths:
            msg = (
                f"Day {self.day_count} begins. "
                f"It was a peaceful night — no one died."
            )
            self._log(msg)
            for pid in self.alive_players:
                self.agents[pid].add_observation(msg)
            return

        dead_names = ", ".join(night_result.deaths)
        msg = (
            f"Day {self.day_count} begins. "
            f"Last night, {dead_names} {'were' if len(night_result.deaths) > 1 else 'was'} "
            f"eliminated."
        )
        self._log(msg)

        # Broadcast death announcement
        for pid in self.alive_players:
            self.agents[pid].add_observation(msg)

        # Collect last words from each dead player
        for dead_id in night_result.deaths:
            if dead_id not in self.agents:
                continue
            obs = (
                f"You have been eliminated. "
                f"Give your last words to help your team."
            )
            action = await self.agents[dead_id].act(obs, last_words_tool())
            content = action.get("content", "(silence)")
            last_words_msg = f"{dead_id}'s last words: \"{content}\""
            self._log(last_words_msg)
            for pid in self.alive_players:
                self.agents[pid].add_observation(last_words_msg)

    async def _sheriff_election(self) -> None:
        """Run the sheriff election on Day 1."""
        if not self.config.use_sheriff:
            return

        self._log("Sheriff election begins.")

        # Phase 1: Who runs?
        candidates = []
        for pid in self.alive_players:
            obs = (
                f"Sheriff election! The sheriff has 1.5x vote weight and "
                f"decides speaking order. Do you want to run for sheriff?"
            )
            action = await self.agents[pid].act(obs, run_for_sheriff_tool())
            if action.get("run", False):
                candidates.append(pid)
                self._log(f"{pid} decides to run for sheriff.")

        if not candidates:
            self._log("No one ran for sheriff. No sheriff this game.")
            return

        if len(candidates) == 1:
            self.sheriff = candidates[0]
            msg = f"{self.sheriff} is elected sheriff unopposed."
            self._log(msg)
            for pid in self.alive_players:
                self.agents[pid].add_observation(msg)
            return

        # Phase 2: Campaign speeches
        speeches = {}
        for i, cid in enumerate(candidates):
            prev_speeches = "\n".join(
                f"{s}: \"{speeches[s]}\"" for s in candidates[:i] if s in speeches
            )
            obs = (
                f"Sheriff election speeches. You are candidate {i + 1} of "
                f"{len(candidates)}.\n"
                f"Candidates: {', '.join(candidates)}.\n"
            )
            if prev_speeches:
                obs += f"Previous speeches:\n{prev_speeches}\n"
            obs += "Give your campaign speech."

            action = await self.agents[cid].act(obs, sheriff_speech_tool())

            if not action.get("continue_running", True):
                self._log(f"{cid} withdraws from the sheriff election.")
                candidates.remove(cid)
                continue

            speech_content = action.get("speech_content", "(no speech)")
            speeches[cid] = speech_content
            speech_msg = f"{cid}'s sheriff campaign speech: \"{speech_content}\""
            self._log(speech_msg)
            for pid in self.alive_players:
                if pid != cid:
                    self.agents[pid].add_observation(speech_msg)

        if not candidates:
            self._log("All candidates withdrew. No sheriff this game.")
            return

        if len(candidates) == 1:
            self.sheriff = candidates[0]
            msg = f"{self.sheriff} is elected sheriff (others withdrew)."
            self._log(msg)
            for pid in self.alive_players:
                self.agents[pid].add_observation(msg)
            return

        # Phase 3: Voting
        votes: dict[str, str | None] = {}
        for pid in self.alive_players:
            obs = (
                f"Vote for sheriff. Candidates: {', '.join(candidates)}.\n"
                f"Cast your vote."
            )
            action = await self.agents[pid].act(
                obs, vote_for_sheriff_tool(candidates)
            )
            target = self._validate_target(
                action.get("target", "abstain"), candidates
            )
            votes[pid] = target

        # Tally
        vote_counts: Counter[str] = Counter()
        for voter, target in votes.items():
            if target:
                vote_counts[target] += 1

        if not vote_counts:
            self._log("All votes were abstentions. No sheriff elected.")
            return

        top_count = vote_counts.most_common(1)[0][1]
        top_candidates = [c for c, cnt in vote_counts.items() if cnt == top_count]

        if len(top_candidates) == 1:
            self.sheriff = top_candidates[0]
        else:
            # Tie-breaker: random
            self.sheriff = self.rng.choice(top_candidates)

        msg = f"{self.sheriff} is elected sheriff!"
        self._log(msg)
        for pid in self.alive_players:
            self.agents[pid].add_observation(msg)

    async def _speeches(self) -> dict[str, str]:
        """All alive players give speeches. Returns {player: speech_content}."""
        # Determine speaking order
        order = list(self.alive_players)

        # If sheriff exists and is alive, they could decide order
        # For simplicity, sheriff speaks last and order is player_order
        if self.sheriff and self.sheriff in order:
            order.remove(self.sheriff)
            order.append(self.sheriff)

        speeches: dict[str, str] = {}
        for i, pid in enumerate(order):
            prev = "\n".join(
                f"{s}: \"{speeches[s]}\"" for s in list(speeches.keys())
            )
            obs = (
                f"Day {self.day_count} discussion. You are speaker "
                f"{i + 1} of {len(order)}.\n"
                f"Alive players: {', '.join(self.alive_players)}.\n"
            )
            if self.sheriff:
                obs += f"Current sheriff: {self.sheriff}.\n"
            if prev:
                obs += f"Previous speeches:\n{prev}\n"
            obs += "It's your turn to speak."

            action = await self.agents[pid].act(obs, speech_tool())
            content = action.get("content", "(silence)")
            speeches[pid] = content

            speech_msg = f"{pid} says: \"{content}\""
            self._log(speech_msg)
            # Broadcast to others
            for other in self.alive_players:
                if other != pid:
                    self.agents[other].add_observation(speech_msg)

        return speeches

    async def _vote(self) -> str | None:
        """All alive players vote to banish. Returns banished player or None."""
        votes: dict[str, str | None] = {}
        valid = self.alive_players

        for pid in self.alive_players:
            obs = (
                f"Voting phase. Vote to banish one player, or abstain.\n"
                f"Alive players: {', '.join(valid)}."
            )
            action = await self.agents[pid].act(obs, vote_tool(valid))
            target = self._validate_target(action.get("target", "abstain"), valid)
            votes[pid] = target
            self._log(f"{pid} votes for: {target or 'abstain'}")

        # Tally with sheriff weight
        vote_counts: dict[str, float] = {}
        for voter, target in votes.items():
            if target:
                weight = 1.5 if voter == self.sheriff else 1.0
                vote_counts[target] = vote_counts.get(target, 0) + weight

        if not vote_counts:
            self._log("All votes were abstentions. No one is banished.")
            return None

        max_votes = max(vote_counts.values())
        top = [p for p, v in vote_counts.items() if v == max_votes]

        if len(top) > 1:
            self._log(
                f"Vote tied between {', '.join(top)}. No one is banished."
            )
            return None

        banished = top[0]
        self._log(f"{banished} is banished by vote ({max_votes:.1f} votes).")
        return banished

    async def _handle_banishment(self, banished: str) -> None:
        """Handle a player being banished: last words, badge flow, elimination."""
        # Last words
        obs = "You have been voted out. Give your last words."
        action = await self.agents[banished].act(obs, last_words_tool())
        content = action.get("content", "(silence)")
        msg = f"{banished}'s last words: \"{content}\""
        self._log(msg)
        for pid in self.alive_players:
            if pid != banished:
                self.agents[pid].add_observation(msg)

        # Badge flow if sheriff
        if banished == self.sheriff:
            alive_others = [p for p in self.alive_players if p != banished]
            if alive_others:
                obs = (
                    f"As the sheriff being eliminated, decide what to do "
                    f"with the badge. You can pass it to another player "
                    f"or destroy it."
                )
                action = await self.agents[banished].act(
                    obs, badge_flow_tool(alive_others)
                )
                if action.get("pass_badge", False):
                    target = self._validate_target(
                        action.get("target", ""), alive_others
                    )
                    if target:
                        self.sheriff = target
                        msg = f"The sheriff badge is passed to {target}."
                    else:
                        self.sheriff = None
                        msg = "The sheriff badge is destroyed (invalid target)."
                else:
                    self.sheriff = None
                    msg = "The sheriff badge is destroyed."
                self._log(msg)
                for pid in self.alive_players:
                    if pid != banished:
                        self.agents[pid].add_observation(msg)

        self._eliminate_player(banished, "banished by vote")

    async def _day_phase(self, night_result: NightResult) -> None:
        """Execute the full day phase."""
        self._log(f"=== Day {self.day_count} ===")

        # 1. Announce deaths
        await self._announce_deaths(night_result)

        # Check termination after deaths
        if self.check_termination():
            return

        # 2. Sheriff election (Day 1 only)
        if self.day_count == 1:
            await self._sheriff_election()

        # 3. Speeches
        await self._speeches()

        # 4. Vote
        banished = await self._vote()

        # 5. Handle banishment
        if banished:
            await self._handle_banishment(banished)

    # ── Main Game Loop ───────────────────────────────────────────────

    async def run(self, agents: dict[str, AsyncAgent]) -> GameResult:
        """Run the full game to completion.

        Args:
            agents: Mapping of agent_id -> AsyncAgent for each player.
                    Must have an entry for every player in config.

        Returns:
            GameResult with winner, outcome, and game log.
        """
        self.agents = agents
        self._setup_players()

        self._log("Werewolf game begins!")
        self._log(f"Players: {', '.join(self.player_order)}")

        while self.day_count < self.config.max_game_days:
            self.day_count += 1

            # Night phase
            night_result = await self._night_phase()

            # Check termination after night
            result = self.check_termination()
            if result:
                self._log(f"Game over after Night {self.day_count}: {result.outcome}")
                return result

            # Day phase
            await self._day_phase(night_result)

            # Check termination after day
            result = self.check_termination()
            if result:
                self._log(f"Game over after Day {self.day_count}: {result.outcome}")
                return result

        # Safety: max days reached
        self._log(f"Game reached max day limit ({self.config.max_game_days}).")
        return GameResult(
            winner=Team.VILLAGER,
            outcome="Draw (max days reached)",
            day_count=self.day_count,
            game_log=self.game_log,
            player_roles={pid: ps.role.value for pid, ps in self.players.items()},
            alive_at_end=self.alive_players,
        )

    def _setup_players(self) -> None:
        """Initialize player states from the agents dict."""
        roles = self.config.build_roles()
        self.rng.shuffle(roles)

        agent_ids = list(self.agents.keys())
        if len(agent_ids) != len(roles):
            raise ValueError(
                f"Expected {len(roles)} agents, got {len(agent_ids)}"
            )

        self.player_order = agent_ids
        for pid, role in zip(agent_ids, roles):
            self.players[pid] = PlayerState(agent_id=pid, role=role)
            # Set each agent's system prompt to match their assigned role and name
            self.agents[pid].system_prompt = build_system_prompt(role, player_name=pid)

        # Log role assignments (private — not shown to agents)
        for pid, ps in self.players.items():
            self._log(f"[Setup] {pid} is assigned role: {ps.role.value}")
