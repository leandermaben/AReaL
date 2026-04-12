"""Generic async LLM agent with tool calling and multi-turn conversation."""

import json
import logging
from typing import Any

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AsyncAgent:
    """A scenario-agnostic LLM agent that maintains conversation history.

    The agent receives observations as user messages and responds via tool calls.
    It tracks response IDs for future RL reward mapping. The client can be any
    AsyncOpenAI instance -- during RL training, AReaL passes a proxy client.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        agent_id: str,
        temperature: float = 1.0,
        extra_body: dict[str, Any] | None = None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.agent_id = agent_id
        self.temperature = temperature
        self.extra_body = extra_body or {}

        self.messages: list[dict[str, Any]] = []
        self.response_ids: list[str] = []

    async def act(
        self,
        observation: str,
        tools: list[dict[str, Any]],
        retries: int = 3,
    ) -> dict[str, Any]:
        """Send an observation and get back a parsed tool-call action.

        Args:
            observation: Text describing what happened in the environment.
            tools: OpenAI function-calling tool definitions for this phase.
            retries: Number of retry attempts on API failure.

        Returns:
            Parsed arguments dict from the first tool call in the response.
        """
        self.messages.append({"role": "user", "content": observation})

        full_messages = [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]

        last_error = None
        for attempt in range(retries):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=full_messages,
                    tools=tools,
                    tool_choice="required",
                    temperature=self.temperature,
                )
                if self.extra_body:
                    kwargs["extra_body"] = self.extra_body
                response = await self.client.chat.completions.create(**kwargs)
                break
            except Exception as e:
                last_error = e
                logger.warning(
                    "Agent %s API call failed (attempt %d/%d): %s",
                    self.agent_id,
                    attempt + 1,
                    retries,
                    e,
                )
        else:
            raise RuntimeError(
                f"Agent {self.agent_id} failed after {retries} retries: {last_error}"
            )

        choice = response.choices[0]
        tool_calls = choice.message.tool_calls

        if not tool_calls:
            # Fallback: model responded with text instead of a tool call
            text = choice.message.content or ""
            logger.warning(
                "Agent %s returned text instead of tool call: %s",
                self.agent_id,
                text[:200],
            )
            self.messages.append({"role": "assistant", "content": text})
            return {"_text_response": text, "_tool_name": None}

        # Track response ID for RL reward mapping
        if response.id:
            self.response_ids.append(response.id)

        # Use the first tool call
        tc = tool_calls[0]
        tool_name = tc.function.name

        try:
            args = json.loads(tc.function.arguments)
        except json.JSONDecodeError:
            args = {"_raw": tc.function.arguments}

        # Append assistant message with tool call to conversation history.
        # Always include "content" key (even when empty) to preserve the
        # prefix property needed for KV cache reuse in AReaL concat mode.
        self.messages.append(
            {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": tc.function.arguments,
                        },
                    }
                ],
            }
        )

        # OpenAI API requires a tool response after each tool_call
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": "Action received.",
            }
        )

        args["_tool_name"] = tool_name
        return args

    def add_observation(self, observation: str) -> None:
        """Add an observation without requesting an action.

        Useful for broadcasting information (e.g., another player's speech)
        that the agent should see but doesn't need to respond to immediately.
        """
        self.messages.append({"role": "user", "content": observation})

    def get_trajectory(self) -> list[dict[str, Any]]:
        """Return the full conversation trajectory including system prompt."""
        return [
            {"role": "system", "content": self.system_prompt},
            *self.messages,
        ]

    def reset(self) -> None:
        """Clear conversation history and response IDs."""
        self.messages.clear()
        self.response_ids.clear()
