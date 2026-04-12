"""Unit tests for AsyncAgent with mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from examples.agent_safety.agent.agent import AsyncAgent


def _make_mock_response(tool_name: str, args: dict, response_id: str = "resp-1"):
    """Create a mock OpenAI chat completion response with a tool call."""
    tc = MagicMock()
    tc.id = "call-1"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    choice = MagicMock()
    choice.message.tool_calls = [tc]
    choice.message.content = None

    response = MagicMock()
    response.id = response_id
    response.choices = [choice]
    return response


def _make_text_response(text: str, response_id: str = "resp-2"):
    """Create a mock response where the model returned text instead of tool call."""
    choice = MagicMock()
    choice.message.tool_calls = None
    choice.message.content = text

    response = MagicMock()
    response.id = response_id
    response.choices = [choice]
    return response


@pytest.fixture
def mock_client():
    client = AsyncMock()
    return client


@pytest.fixture
def agent(mock_client):
    return AsyncAgent(
        client=mock_client,
        model="test-model",
        system_prompt="You are a test agent.",
        agent_id="test-agent",
    )


DUMMY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "test_action",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string"},
                },
                "required": ["target"],
            },
        },
    }
]


@pytest.mark.asyncio
async def test_act_returns_parsed_args(agent, mock_client):
    """act() should return parsed tool call arguments."""
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "test_action", {"target": "Alice"}
    )

    result = await agent.act("Choose a target.", DUMMY_TOOLS)

    assert result["target"] == "Alice"
    assert result["_tool_name"] == "test_action"


@pytest.mark.asyncio
async def test_act_builds_messages_correctly(agent, mock_client):
    """act() should pass system + history + new observation to the API."""
    mock_client.chat.completions.create.return_value = _make_mock_response(
        "test_action", {"target": "Bob"}
    )

    await agent.act("Observation 1", DUMMY_TOOLS)

    call_args = mock_client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a test agent."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Observation 1"


@pytest.mark.asyncio
async def test_act_appends_to_history(agent, mock_client):
    """Successive act() calls should accumulate message history."""
    mock_client.chat.completions.create.side_effect = [
        _make_mock_response("test_action", {"target": "A"}, "resp-1"),
        _make_mock_response("test_action", {"target": "B"}, "resp-2"),
    ]

    await agent.act("Turn 1", DUMMY_TOOLS)
    await agent.act("Turn 2", DUMMY_TOOLS)

    # 2 user messages + 2 assistant messages + 2 tool responses
    assert len(agent.messages) == 6
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[1]["role"] == "assistant"
    assert agent.messages[2]["role"] == "tool"
    assert agent.messages[3]["role"] == "user"
    assert agent.messages[4]["role"] == "assistant"
    assert agent.messages[5]["role"] == "tool"


@pytest.mark.asyncio
async def test_act_tracks_response_ids(agent, mock_client):
    """act() should track response IDs for RL reward mapping."""
    mock_client.chat.completions.create.side_effect = [
        _make_mock_response("test_action", {"target": "A"}, "resp-abc"),
        _make_mock_response("test_action", {"target": "B"}, "resp-def"),
    ]

    await agent.act("Turn 1", DUMMY_TOOLS)
    await agent.act("Turn 2", DUMMY_TOOLS)

    assert agent.response_ids == ["resp-abc", "resp-def"]


@pytest.mark.asyncio
async def test_act_handles_text_response(agent, mock_client):
    """act() should handle models that return text instead of tool calls."""
    mock_client.chat.completions.create.return_value = _make_text_response(
        "I think Alice is suspicious."
    )

    result = await agent.act("What do you think?", DUMMY_TOOLS)

    assert result["_tool_name"] is None
    assert "I think Alice" in result["_text_response"]


@pytest.mark.asyncio
async def test_act_retries_on_failure(agent, mock_client):
    """act() should retry on API errors."""
    mock_client.chat.completions.create.side_effect = [
        RuntimeError("API error"),
        _make_mock_response("test_action", {"target": "C"}),
    ]

    result = await agent.act("Try again", DUMMY_TOOLS, retries=3)
    assert result["target"] == "C"


@pytest.mark.asyncio
async def test_act_raises_after_max_retries(agent, mock_client):
    """act() should raise after exhausting retries."""
    mock_client.chat.completions.create.side_effect = RuntimeError("fail")

    with pytest.raises(RuntimeError, match="failed after 2 retries"):
        await agent.act("Will fail", DUMMY_TOOLS, retries=2)


def test_add_observation(agent):
    """add_observation() should append a user message without action."""
    agent.add_observation("Something happened.")
    assert len(agent.messages) == 1
    assert agent.messages[0]["role"] == "user"
    assert agent.messages[0]["content"] == "Something happened."


def test_get_trajectory(agent):
    """get_trajectory() should include system prompt and messages."""
    agent.add_observation("obs1")
    traj = agent.get_trajectory()
    assert traj[0]["role"] == "system"
    assert traj[1]["role"] == "user"
    assert len(traj) == 2


def test_reset(agent):
    """reset() should clear messages and response IDs."""
    agent.messages.append({"role": "user", "content": "test"})
    agent.response_ids.append("resp-1")

    agent.reset()

    assert agent.messages == []
    assert agent.response_ids == []
