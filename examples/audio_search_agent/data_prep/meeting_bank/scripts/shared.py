"""Shared utilities for MeetingBank data prep v2.

Provides:
- AsyncLLMClient: async OpenAI wrapper with concurrency, retry, server-down detection
- Token counting and context budgeting
- JSON parsing, time formatting, 3s grid snapping
- Transcript loading helpers
"""
from __future__ import annotations

import asyncio
import json
import math
import re
import sys
from pathlib import Path

import openai
from openai import AsyncOpenAI

# ── Constants ─────────────────────────────────────────────────────────────────

US_TO_S = 1e-6  # microseconds → seconds

# Approximate tokens per character for English text (conservative).
# tiktoken would be more accurate but adds a dependency; this is good enough
# for budgeting since we leave large margins.
_CHARS_PER_TOKEN = 4.0


# ── Token counting & truncation ──────────────────────────────────────────────


def count_tokens(text: str) -> int:
    """Approximate token count for context budgeting."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def truncate_to_budget(text: str, max_tokens: int) -> str:
    """Truncate text to fit within a token budget, preserving sentence boundaries."""
    if count_tokens(text) <= max_tokens:
        return text
    max_chars = int(max_tokens * _CHARS_PER_TOKEN)
    truncated = text[:max_chars]
    # Try to end at a sentence boundary
    last_period = truncated.rfind(". ")
    if last_period > max_chars * 0.5:
        truncated = truncated[: last_period + 1]
    return truncated


# ── Time utilities ────────────────────────────────────────────────────────────


def fmt_time(seconds: float) -> str:
    """Human-readable time string."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h > 0:
        return f"{h}h{m:02d}m{s:05.2f}s"
    return f"{m}m{s:05.2f}s"


def snap_to_3s(start_s: float, end_s: float) -> tuple[float, float]:
    """Snap start/end to 3-second grid (floor start, ceil end)."""
    return math.floor(start_s / 3) * 3, math.ceil(end_s / 3) * 3


# ── JSON parsing ─────────────────────────────────────────────────────────────


def parse_json_response(raw: str) -> dict | list | None:
    """Strip markdown fences and parse JSON from LLM output."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ── Transcript loading ────────────────────────────────────────────────────────


def seg_bounds_s(seg: dict) -> tuple[float, float]:
    """Return (start_sec, end_sec) for a raw transcript segment."""
    start = seg["offset"] * US_TO_S
    return start, start + seg["duration"] * US_TO_S


def seg_text(seg: dict) -> str:
    """Extract text from a raw transcript segment."""
    nbest = seg.get("nbest")
    return nbest[0]["text"].strip() if nbest else ""


def load_raw_segments(transcript_path: Path) -> list[dict]:
    """Load raw segments from a transcript JSON file."""
    data = json.loads(transcript_path.read_text())
    return data.get("segments", [])


# ── JSONL I/O ─────────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    """Read all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write records to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, record: dict) -> None:
    """Append a single record to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Async LLM Client ─────────────────────────────────────────────────────────


class AsyncLLMClient:
    """Async OpenAI client with concurrency control, retry, and server-down detection.

    Usage:
        client = AsyncLLMClient("http://localhost:8000/v1", "model-name")
        result = await client.generate("Your prompt here")
        # result is parsed JSON (dict/list) or None on failure
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        concurrency: int = 64,
        max_retries: int = 3,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system_prompt: str | None = None,
    ):
        self.client = AsyncOpenAI(base_url=base_url, api_key="dummy")
        self.model = model
        self.sem = asyncio.Semaphore(concurrency)
        self.max_retries = max_retries
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.server_down = asyncio.Event()

        # Stats
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

    async def generate(
        self,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
    ) -> dict | list | None:
        """Send a prompt and return parsed JSON, or None on failure."""
        if self.server_down.is_set():
            return None

        messages = []
        sys_prompt = system_prompt or self.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        messages.append({"role": "user", "content": user_prompt})

        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        for attempt in range(self.max_retries):
            if self.server_down.is_set():
                return None
            try:
                self.total_calls += 1
                async with self.sem:
                    resp = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=max_tok,
                        temperature=temp,
                    )
                content = resp.choices[0].message.content or ""
                parsed = parse_json_response(content)
                if parsed is not None:
                    self.successful_calls += 1
                    return parsed
                # JSON parse failed — retry
                if attempt == self.max_retries - 1:
                    self.failed_calls += 1
                    print(
                        f"  [LLM] JSON parse failed after {self.max_retries} attempts",
                        file=sys.stderr,
                    )
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                print(f"\n[FATAL] vLLM server unreachable: {exc}", file=sys.stderr)
                self.server_down.set()
                self.failed_calls += 1
                return None
            except openai.APIStatusError as exc:
                if exc.status_code in (502, 503, 504):
                    print(
                        f"\n[FATAL] vLLM server returned {exc.status_code}",
                        file=sys.stderr,
                    )
                    self.server_down.set()
                    self.failed_calls += 1
                    return None
                if attempt == self.max_retries - 1:
                    self.failed_calls += 1
                    print(f"  [LLM] API error: {exc}", file=sys.stderr)
                else:
                    await asyncio.sleep(1 + attempt)
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    self.failed_calls += 1
                    print(f"  [LLM] unexpected error: {exc}", file=sys.stderr)
                else:
                    await asyncio.sleep(1 + attempt)

        return None

    def print_stats(self) -> None:
        """Print LLM call statistics."""
        print(
            f"LLM stats: {self.total_calls} total, "
            f"{self.successful_calls} successful, {self.failed_calls} failed"
        )


# ── Discovery ─────────────────────────────────────────────────────────────────


def discover_pairs(
    data_root: Path, cities: list[str] | None = None
) -> list[tuple[Path, Path, str]]:
    """Find all (mp3, transcript, city) triplets under data_root."""
    city_filter = {c.lower() for c in cities} if cities else None
    pairs = []
    for city_dir in sorted(data_root.iterdir()):
        if not city_dir.is_dir():
            continue
        if city_filter and city_dir.name.lower() not in city_filter:
            continue
        mp3_dir = city_dir / "mp3"
        tr_dir = city_dir / "transcripts"
        if not mp3_dir.is_dir() or not tr_dir.is_dir():
            continue
        for mp3 in sorted(mp3_dir.glob("*.mp3")):
            tr = tr_dir / f"{mp3.name}.transcript.json"
            if tr.is_file():
                pairs.append((mp3, tr, city_dir.name))
    return pairs
