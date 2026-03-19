"""Async agent loop for the audio search agent.

The agent receives a question about a long audio recording and iteratively
uses tools (CLAP search, Omni probe, submit) to find and verify relevant
snippets before submitting a final answer.

This module is self-contained and can be run independently for testing,
or imported into an AReaL workflow later.

Standalone usage:
    python -m examples.audio_search_agent.v1.agent.agent_loop \
        --llm-url http://localhost:8000/v1 \
        --llm-model Qwen/Qwen3-4B-Instruct-2507 \
        --clap-cache-dir /work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank \
        --data-root /work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared \
        --audio-id meetingbank_test_0 \
        --question "What was discussed about the city budget?"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessage

from areal.utils.logging import getLogger

from .tools.base import Tool

logger = getLogger("AgentLoop")

# ──────────────────────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an audio search agent. Your task is to find the relevant audio snippets \
that answer a question about a long audio recording.

You have access to tools for searching and analyzing the audio. A typical workflow is:
1. Use clap_search to find segments that semantically match aspects of the question.
2. Use omni_probe to deeply analyze the most promising segments — get transcripts, \
   speaker info, and verify relevance. Note: omni_probe accepts segments up to 150 \
   seconds (2.5 minutes) per call.
3. Once you have identified all relevant snippets, use submit to deliver your final answer.

IMPORTANT RULES:
- You have exactly {max_turns} turns. You MUST call submit before your turns run out.
- Do NOT call more than 5 tools in a single turn.
- When you call submit, do NOT call any other tools in the same turn. All other tool \
  calls in a turn with submit will be ignored.
- Plan your search strategy to finish within the turn budget.

Strategy tips:
- Start broad: search for key topics/sounds mentioned in the question or options.
- It is often a good idea to search based on the options provided in the question.
- Narrow down: probe the top CLAP hits to verify they actually contain the answer.
- Be thorough: the answer may span multiple non-contiguous segments.
- Include reasoning: when you submit, explain why each snippet is relevant.
- Don't submit until you're confident — but don't wait too long either.

The audio_id for this episode is: {audio_id}
The recording is approximately {duration} long."""


# ──────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────
class AudioSearchAgent:
    """Async agent that searches long audio recordings via tool use.

    Attributes:
        tools:       dict mapping tool name → Tool instance
        tool_schemas: list of OpenAI function schemas for the LLM
        messages:    full conversation history (populated during run_episode)
    """

    def __init__(
        self,
        llm_url: str,
        llm_model: str,
        tools: list[Tool],
        max_turns: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        api_key: str = "dummy",
    ):
        self._client = AsyncOpenAI(base_url=llm_url, api_key=api_key)
        self._llm_model = llm_model
        self.tools: dict[str, Tool] = {t.NAME: t for t in tools}
        self.tool_schemas: list[dict] = [t.schema() for t in tools]
        self._max_turns = max_turns
        self._max_tokens = max_tokens
        self._temperature = temperature
        self.messages: list[dict] = []

    async def run_episode(
        self,
        question: str,
        audio_id: str,
        duration_str: str = "unknown duration",
    ) -> dict[str, Any]:
        """Run one full episode: question → tool loop → submission.

        Args:
            question:     The question to answer about the audio.
            audio_id:     Identifier for the audio recording.
            duration_str: Human-readable duration (e.g. "80 minutes") for the prompt.

        Returns:
            dict with keys:
                status:     "submitted" | "max_turns_reached" | "error"
                audio_id:   the audio_id
                question:   the question
                turns:      number of LLM turns used
                submission: the submit tool result (if submitted), else None
                messages:   full conversation history
        """
        system = SYSTEM_PROMPT.format(
            audio_id=audio_id, duration=duration_str, max_turns=self._max_turns,
        )
        self.messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ]

        logger.info(
            f"Episode start: audio_id={audio_id}, question='{question[:80]}...'"
        )

        submission = None

        for turn in range(self._max_turns):
            # ── LLM call ──────────────────────────────────────
            try:
                response = await self._client.chat.completions.create(
                    model=self._llm_model,
                    messages=self.messages,
                    tools=self.tool_schemas,
                    tool_choice="auto",
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
            except Exception as exc:
                logger.error(f"LLM call failed on turn {turn}: {exc}")
                return self._result(
                    "error", audio_id, question, turn, submission,
                    error=f"LLM call failed: {exc}",
                )

            msg = response.choices[0].message

            # Append assistant message to history
            self.messages.append(_message_to_dict(msg))

            # ── No tool calls → LLM chose to respond with text ─
            if not msg.tool_calls:
                logger.info(
                    f"Turn {turn}: LLM responded with text (no tool call): "
                    f"'{(msg.content or '')[:100]}'"
                )
                # If the LLM just talks without calling tools, nudge it
                if turn < self._max_turns - 1:
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "Please use the available tools to search the audio "
                            "and submit your findings. Call clap_search to find "
                            "relevant segments, omni_probe to analyze them, "
                            "and submit when ready."
                        ),
                    })
                continue

            # ── Execute tool calls concurrently ───────────────
            # Parse all tool calls first
            parsed_calls = []
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                    logger.warning(
                        f"Turn {turn}: malformed tool args for {name}: "
                        f"{tool_call.function.arguments[:200]}"
                    )
                logger.info(f"Turn {turn}: tool_call={name}, args={_truncate(args)}")
                parsed_calls.append((tool_call, name, args))

            # If submit is in this batch, only execute submit; skip others
            submit_idx = next(
                (i for i, (_, n, _) in enumerate(parsed_calls) if n == "submit"),
                None,
            )
            if submit_idx is not None:
                if len(parsed_calls) > 1:
                    logger.info(
                        f"Turn {turn}: submit called with {len(parsed_calls) - 1} "
                        f"other tool(s) — ignoring non-submit calls"
                    )
                calls_to_run = [parsed_calls[submit_idx]]
                skipped = [pc for i, pc in enumerate(parsed_calls) if i != submit_idx]
            else:
                calls_to_run = parsed_calls
                skipped = []

            # Run tool calls concurrently via asyncio.gather
            async def _run_tool(name: str, args: dict) -> dict:
                tool = self.tools.get(name)
                if tool is None:
                    return {"status": "error", "error": f"Unknown tool: {name}"}
                try:
                    return await tool.execute_async(**args)
                except Exception as exc:
                    logger.error(f"Tool {name} raised: {exc}")
                    return {"status": "error", "error": str(exc)}

            results = await asyncio.gather(
                *[_run_tool(name, args) for _, name, args in calls_to_run]
            )

            # Append skipped tool results as ignored
            for tool_call, name, _ in skipped:
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps({
                        "status": "ignored",
                        "error": "Ignored: submit was called in the same turn.",
                    }),
                })

            # Append executed results in order and check for submit
            for (tool_call, name, _), result in zip(calls_to_run, results):
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                })
                logger.info(
                    f"Turn {turn}: tool_result={name}, status={result.get('status')}"
                )

                if name == "submit" and result.get("status") == "ok":
                    submission = result
                    logger.info(
                        f"Episode done: {result.get('num_snippets', 0)} snippets "
                        f"submitted after {turn + 1} turns"
                    )
                    return self._result(
                        "submitted", audio_id, question, turn + 1, submission,
                    )

        # Exhausted max turns without a successful submission
        logger.warning(
            f"Episode ended: max turns ({self._max_turns}) reached without submission"
        )
        return self._result(
            "max_turns_reached", audio_id, question, self._max_turns, submission,
        )

    def _result(
        self,
        status: str,
        audio_id: str,
        question: str,
        turns: int,
        submission: dict | None,
        error: str = "",
    ) -> dict[str, Any]:
        result = {
            "status": status,
            "audio_id": audio_id,
            "question": question,
            "turns": turns,
            "submission": submission,
            "messages": self.messages,
        }
        if error:
            result["error"] = error
        return result


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def _message_to_dict(msg: ChatCompletionMessage) -> dict:
    """Convert an OpenAI ChatCompletionMessage to a serialisable dict."""
    d: dict[str, Any] = {"role": "assistant"}
    if msg.content:
        d["content"] = msg.content
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in msg.tool_calls
        ]
    return d


def _truncate(obj: Any, max_len: int = 200) -> str:
    """Truncate a repr for logging."""
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= max_len else s[:max_len] + "..."


# ──────────────────────────────────────────────────────────────
# Standalone CLI for testing
# ──────────────────────────────────────────────────────────────
def _build_tools(args) -> list[Tool]:
    """Instantiate tools from CLI args."""
    from ..preprocessing.clap_indexer import CLAPConfig, CLAPIndexer
    from .tools.clap_search import CLAPSearchTool
    from .tools.omni_probe import OmniProbeTool
    from .tools.submit import SubmitTool

    # CLAP search
    clap_config = CLAPConfig(
        data_root=args.data_root,
        cache_dir=args.clap_cache_dir,
        device=args.clap_device,
    )
    clap_tool = CLAPSearchTool(
        indexer=CLAPIndexer(clap_config),
        audio_id=args.audio_id,
    )

    # Submit
    submit_tool = SubmitTool(audio_id=args.audio_id)

    tools: list[Tool] = [clap_tool, submit_tool]

    # Omni probe (optional — needs a running server)
    if args.omni_url:
        # Resolve wav_path from data_root + audio_id
        wav_path = _resolve_wav_path(args.data_root, args.audio_id)
        if wav_path:
            omni_tool = OmniProbeTool(
                omni_url=args.omni_url,
                model=args.omni_model,
                wav_path=wav_path,
                audio_id=args.audio_id,
            )
            tools.append(omni_tool)
        else:
            logger.warning(f"Could not resolve wav_path for {args.audio_id}, skipping omni_probe")

    return tools


def _resolve_wav_path(data_root: str, audio_id: str) -> str | None:
    """Resolve audio_id → wav file path using the manifest."""
    from pathlib import Path

    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        if entry["audio_id"] == audio_id:
            split = entry["split"]
            # audio_id format: meetingbank_{split}_{idx}
            idx = audio_id.rsplit("_", 1)[-1]
            wav = Path(data_root) / split / f"audio_{idx}.wav"
            return str(wav) if wav.exists() else None
    return None


def _get_duration_str(data_root: str, audio_id: str) -> str:
    """Get human-readable duration from manifest."""
    from pathlib import Path

    manifest_path = Path(data_root) / "manifest.json"
    if not manifest_path.exists():
        return "unknown duration"
    manifest = json.loads(manifest_path.read_text())
    for entry in manifest:
        if entry["audio_id"] == audio_id:
            dur = entry.get("duration_seconds", 0)
            mins = int(dur // 60)
            return f"{mins} minutes"
    return "unknown duration"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Run the audio search agent on a single question (for testing).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # LLM
    p.add_argument("--llm-url", required=True, help="vLLM server URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--llm-model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.7)

    # CLAP
    p.add_argument("--clap-cache-dir", required=True, help="Path to CLAP index cache")
    p.add_argument("--clap-device", default="cpu", help="Device for CLAP model (cpu/cuda)")

    # Omni (optional)
    p.add_argument("--omni-url", default=None, help="Omni vLLM server URL (skip if not available)")
    p.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")

    # Data
    p.add_argument("--data-root", required=True, help="Path to meeting_bank_prepared")
    p.add_argument("--audio-id", required=True, help="Audio ID to search within")
    p.add_argument("--question", required=True, help="Question to answer")

    return p.parse_args(argv)


async def async_main(argv=None):
    args = parse_args(argv)

    tools = _build_tools(args)
    logger.info(f"Tools loaded: {[t.NAME for t in tools]}")

    agent = AudioSearchAgent(
        llm_url=args.llm_url,
        llm_model=args.llm_model,
        tools=tools,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    duration_str = _get_duration_str(args.data_root, args.audio_id)

    result = await agent.run_episode(
        question=args.question,
        audio_id=args.audio_id,
        duration_str=duration_str,
    )

    # Print result (without full message history for readability)
    output = {k: v for k, v in result.items() if k != "messages"}
    print("\n" + "=" * 70)
    print("EPISODE RESULT")
    print("=" * 70)
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Print conversation summary
    print("\n" + "=" * 70)
    print("CONVERSATION SUMMARY")
    print("=" * 70)
    for msg in result["messages"]:
        role = msg["role"]
        if role == "system":
            print(f"[system] (prompt)")
        elif role == "user":
            print(f"[user] {msg.get('content', '')[:120]}")
        elif role == "assistant":
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            if tool_calls:
                for tc in tool_calls:
                    fn = tc["function"]
                    print(f"[assistant] → {fn['name']}({fn['arguments'][:100]})")
            elif content:
                print(f"[assistant] {content[:120]}")
        elif role == "tool":
            parsed = json.loads(msg["content"])
            status = parsed.get("status", "?")
            # Compact summary per tool type
            if "results" in parsed:
                n = len(parsed["results"])
                print(f"[tool] ← status={status}, {n} results")
            elif "num_snippets" in parsed:
                print(f"[tool] ← status={status}, {parsed['num_snippets']} snippets submitted")
            else:
                print(f"[tool] ← status={status}")


def main(argv=None):
    asyncio.run(async_main(argv))


if __name__ == "__main__":
    main(sys.argv[1:])
