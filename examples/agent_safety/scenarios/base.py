"""Base scenario interface for multi-agent environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentConfig:
    """Configuration needed to create an agent for a scenario."""

    agent_id: str
    role: str
    system_prompt: str


@dataclass
class ScenarioResult:
    """Result of a completed scenario."""

    outcome: str
    winner: str
    rewards: dict[str, float]  # agent_id -> reward
    game_log: list[str] = field(default_factory=list)
    trajectories: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class BaseScenario(ABC):
    """Abstract base class for multi-agent scenarios."""

    @abstractmethod
    def get_agent_configs(self) -> list[AgentConfig]:
        """Return the configs needed to create agents for this scenario."""

    @abstractmethod
    async def run(self, agents: dict) -> ScenarioResult:
        """Run the scenario to completion with the given agents.

        Args:
            agents: Mapping of agent_id -> AsyncAgent.

        Returns:
            ScenarioResult with outcome, rewards, and trajectories.
        """
