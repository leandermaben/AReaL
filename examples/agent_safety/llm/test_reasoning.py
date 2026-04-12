"""Test reasoning disable for hosted LLMs that support it.

For each reasoning model, verifies that:
1. Tool calls work with reasoning disabled
2. The extra_body params from hosted_llms.json are correct

Usage:
    python -m examples.agent_safety.llm.test_reasoning
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


async def test_reasoning_disable(entry: dict) -> tuple[str, bool, str]:
    """Test tool call with reasoning disabled for a single model."""
    model = entry["model"]
    api_key = KEYS.get(entry["api_key"], entry["api_key"])
    client = AsyncOpenAI(base_url=entry["base_url"], api_key=api_key)
    disable_params = entry.get("reasoning_disable")

    t0 = time.time()
    try:
        kwargs = dict(
            model=model,
            messages=[{"role": "user", "content": "What is the weather in Pittsburgh?"}],
            tools=TOOLS,
            tool_choice="required",
            max_tokens=100,
        )
        if isinstance(disable_params, dict):
            kwargs["extra_body"] = disable_params

        resp = await client.chat.completions.create(**kwargs)
        tc = resp.choices[0].message.tool_calls
        if tc and len(tc) > 0:
            args = json.loads(tc[0].function.arguments)
            return (
                model,
                True,
                f"tool:{tc[0].function.name}({args}) ({time.time() - t0:.1f}s)",
            )
        text = (resp.choices[0].message.content or "")[:60]
        return model, False, f"no tool_call, text: {text!r} ({time.time() - t0:.1f}s)"
    except Exception as e:
        err = str(e).split("\n")[0][:120]
        return model, False, f"{err} ({time.time() - t0:.1f}s)"


def main():
    reasoning_models = [m for m in MODELS if m.get("reasoning", False)]
    n = len(reasoning_models)

    print(f"=== Reasoning Disable Test — {n} reasoning models ===\n")
    print(f"{'Model':<45s} {'Disable Method':<30s} {'Status':<6s} Result")
    print("-" * 120)

    t0 = time.time()

    async def run_all():
        return await asyncio.gather(*[test_reasoning_disable(e) for e in reasoning_models])

    results = asyncio.run(run_all())

    ok = fail = 0
    for entry, (model, success, msg) in zip(reasoning_models, results):
        status = "OK" if success else "FAIL"
        disable = entry.get("reasoning_disable", "N/A")
        if isinstance(disable, dict):
            disable_str = json.dumps(disable)
        else:
            disable_str = str(disable)
        print(f"  {model:<43s} {disable_str:<28s} [{status:4s}] {msg}")
        if success:
            ok += 1
        else:
            fail += 1

    print(f"\n{ok}/{n} reasoning models work with reasoning disabled. "
          f"(total wall time: {time.time() - t0:.1f}s)\n")

    # Also show non-reasoning models for reference
    non_reasoning = [m for m in MODELS if not m.get("reasoning", False)]
    print(f"Non-reasoning models ({len(non_reasoning)}):")
    for m in non_reasoning:
        print(f"  - {m['model']}")

    sys.exit(0 if fail == 0 else 1)


if __name__ == "__main__":
    main()
