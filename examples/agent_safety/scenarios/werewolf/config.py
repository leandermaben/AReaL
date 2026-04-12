"""Configuration for werewolf game scenarios."""

from dataclasses import dataclass, field

from .roles import Role


@dataclass
class WerewolfConfig:
    """Configuration for a werewolf game."""

    num_werewolves: int = 2
    num_villagers: int = 3  # plain villagers (excludes special roles)
    include_seer: bool = True
    include_witch: bool = True
    include_guard: bool = True
    use_sheriff: bool = True
    max_wolf_discussion_rounds: int = 3
    use_random_names: bool = True
    max_game_days: int = 10  # safety limit to prevent infinite games

    def build_roles(self) -> list[Role]:
        """Build the list of roles for a game based on config."""
        roles = [Role.WEREWOLF] * self.num_werewolves
        if self.include_seer:
            roles.append(Role.SEER)
        if self.include_witch:
            roles.append(Role.WITCH)
        if self.include_guard:
            roles.append(Role.GUARD)
        roles.extend([Role.VILLAGER] * self.num_villagers)
        return roles

    @property
    def total_players(self) -> int:
        return len(self.build_roles())
