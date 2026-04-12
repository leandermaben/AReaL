"""Ping each hosted LLM to verify it responds (sync and async)."""

import asyncio
import json
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI, OpenAI

HERE = Path(__file__).parent
MODELS = json.loads((HERE / "hosted_llms.json").read_text())["models"]
KEYS = json.loads((HERE / "hosted_llm_keys.json").read_text())

MSG = [{"role": "user", "content": "Say OK"}]
KWARGS = {"max_tokens": 3}


def test_model_sync(entry: dict) -> tuple[str, bool, str]:
    model = entry["model"]
    api_key = KEYS.get(entry["api_key"], entry["api_key"])
    client = OpenAI(base_url=entry["base_url"], api_key=api_key)
    t0 = time.time()
    try:
        resp = client.chat.completions.create(model=model, messages=MSG, **KWARGS)
        text = resp.choices[0].message.content.strip()
        return model, True, f"{text} ({time.time() - t0:.1f}s)"
    except Exception as e:
        err = str(e).split("\n")[0][:80]
        return model, False, f"{err} ({time.time() - t0:.1f}s)"


async def test_model_async(entry: dict) -> tuple[str, bool, str]:
    model = entry["model"]
    api_key = KEYS.get(entry["api_key"], entry["api_key"])
    client = AsyncOpenAI(base_url=entry["base_url"], api_key=api_key)
    t0 = time.time()
    try:
        resp = await client.chat.completions.create(model=model, messages=MSG, **KWARGS)
        text = resp.choices[0].message.content.strip()
        return model, True, f"{text} ({time.time() - t0:.1f}s)"
    except Exception as e:
        err = str(e).split("\n")[0][:80]
        return model, False, f"{err} ({time.time() - t0:.1f}s)"


def print_results(results: list[tuple[str, bool, str]]) -> tuple[int, int]:
    ok = fail = 0
    for model, success, msg in results:
        status = "OK" if success else "FAIL"
        print(f"  [{status:4s}] {model:45s} {msg}")
        if success:
            ok += 1
        else:
            fail += 1
    return ok, fail


def main():
    n = len(MODELS)

    # --- Sync ---
    # print(f"=== Sync (OpenAI) — {n} models ===\n")
    # sync_results = [test_model_sync(e) for e in MODELS]
    # ok1, fail1 = print_results(sync_results)
    # print(f"\n{ok1}/{n} passed.\n")

    # --- Async ---
    print(f"=== Async (AsyncOpenAI) — {n} models in parallel ===\n")
    t0 = time.time()
    async def run_all():
        return await asyncio.gather(*[test_model_async(e) for e in MODELS])

    async_results = asyncio.run(run_all())
    print_results(async_results)
    ok2, fail2 = print_results(async_results)
    print(f"\n{ok2}/{n} passed. (total wall time: {time.time() - t0:.1f}s)\n")

    sys.exit(0 if (fail2) == 0 else 1)


if __name__ == "__main__":
    main()
