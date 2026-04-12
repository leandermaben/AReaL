"""Test which hosted LLMs support tool calling (function calling).

Sends a simple tool-call request to each model and reports which ones
return a valid tool call response.

Usage:
    python -m examples.agent_safety.llm.test_tool_call
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).parent
MODELS = json.loads((HERE / "hosted_llms.json").read_text())["models"]
KEYS = json.loads((HERE / "hosted_llm_keys.json").read_text())

MSG = [{"role": "user", "content": "What is the weather in Pittsburgh?"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The city name.",
                    },
                },
                "required": ["city"],
            },
        },
    }
]


async def test_tool_call(entry: dict) -> tuple[str, bool, str]:
    """Test a single model for tool-call support."""
    model = entry["model"]
    api_key = KEYS.get(entry["api_key"], entry["api_key"])
    client = AsyncOpenAI(base_url=entry["base_url"], api_key=api_key)
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=MSG,
            tools=TOOLS,
            tool_choice="required",
            max_tokens=100,
        )
        choice = resp.choices[0]
        tool_calls = choice.message.tool_calls
        if tool_calls and len(tool_calls) > 0:
            tc = tool_calls[0]
            args = json.loads(tc.function.arguments)
            return (
                model,
                True,
                f"tool:{tc.function.name}({args}) ({time.time() - t0:.1f}s)",
            )
        # Model responded but without a tool call
        text = (choice.message.content or "")[:60]
        return model, False, f"no tool_call, text: {text!r} ({time.time() - t0:.1f}s)"
    except Exception as e:
        err = str(e).split("\n")[0][:120]
        return model, False, f"{err} ({time.time() - t0:.1f}s)"


def main():
    n = len(MODELS)
    print(f"=== Tool Call Test — {n} models in parallel ===\n")

    t0 = time.time()

    async def run_all():
        return await asyncio.gather(*[test_tool_call(e) for e in MODELS])

    results = asyncio.run(run_all())

    ok = fail = 0
    working = []
    for model, success, msg in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status:4s}] {model:45s} {msg}")
        if success:
            ok += 1
            working.append(model)
        else:
            fail += 1

    print(f"\n{ok}/{n} models support tool calling. "
          f"(total wall time: {time.time() - t0:.1f}s)\n")

    if working:
        print("Models with working tool call support:")
        for m in working:
            print(f"  - {m}")

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
