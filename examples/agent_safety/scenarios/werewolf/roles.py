"""Werewolf game role definitions."""

from enum import Enum


class Role(Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"
    SEER = "seer"
    WITCH = "witch"
    GUARD = "guard"


class Team(Enum):
    WEREWOLF = "werewolf"
    VILLAGER = "villager"


ROLE_TEAM = {
    Role.WEREWOLF: Team.WEREWOLF,
    Role.VILLAGER: Team.VILLAGER,
    Role.SEER: Team.VILLAGER,
    Role.WITCH: Team.VILLAGER,
    Role.GUARD: Team.VILLAGER,
}

ROLE_HAS_NIGHT_ACTION = {
    Role.WEREWOLF: True,
    Role.VILLAGER: False,
    Role.SEER: True,
    Role.WITCH: True,
    Role.GUARD: True,
}
