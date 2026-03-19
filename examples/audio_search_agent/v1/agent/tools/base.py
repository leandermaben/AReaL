"""Minimal base class for agent tools.

Keeps things simple: each tool declares its name, description, parameter schema,
and an execute() method. That's it — no registry decorators, no metaclasses.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class for all agent tools.

    Subclasses must define:
        NAME:        tool name used in LLM function calls
        DESCRIPTION: one-paragraph description shown to the LLM
        PARAMETERS:  JSON Schema dict for the tool's arguments

    And implement:
        execute(**kwargs) -> dict   (returns structured results)

    For async tools (e.g. external API calls), override execute_async() instead.
    The default execute_async() delegates to execute().
    """

    NAME: str
    DESCRIPTION: str
    PARAMETERS: dict  # JSON Schema (type: object, properties: {...}, required: [...])

    @classmethod
    def schema(cls) -> dict:
        """Return an OpenAI-compatible function/tool schema."""
        return {
            "type": "function",
            "function": {
                "name": cls.NAME,
                "description": cls.DESCRIPTION,
                "parameters": cls.PARAMETERS,
            },
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict:
        """Run the tool and return structured results.

        Returns a dict with at minimum:
            status: "ok" | "error"
        Plus tool-specific fields.
        """
        ...

    async def execute_async(self, **kwargs: Any) -> dict:
        """Async variant. Override for tools that need async I/O.

        Default implementation runs sync execute() in a thread-pool executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: self.execute(**kwargs))
