"""
AudioSearchAgent for AReaL GRPO training.

This agent is an "agent-like workflow" (not a RolloutWorkflow subclass) so
AReaL automatically wraps it with OpenAIProxyWorkflow when it is passed as
the `workflow` argument to PPOTrainer.train().

The agent:
  1. Runs a SQL-based search loop against the PostgreSQL audio database.
  2. Calls `submit_snippets` to end search once enough context is collected.
  3. Generates a final answer from the retrieved snippets.
  4. Returns a float reward (exact-match for MCQ, token-F1 for open-ended).

All LLM calls go through the AReaL proxy (via base_url / api_key injected
by OpenAIProxyWorkflow), so every token generated is tracked for training.
DB queries use psycopg2 directly in a thread-pool executor.
"""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any

import psycopg2
import psycopg2.extras

from areal.utils.logging import getLogger

from .reward import audio_qa_reward

logger = getLogger("AudioSearchAgent")

# ---------------------------------------------------------------------------
# DB schema description injected into the system prompt
# ---------------------------------------------------------------------------

_SCHEMA = """\
Database tables (PostgreSQL). Always filter by audio_id = '<audio_id>'.

transcription(id, audio_id, text TEXT, start_time FLOAT, end_time FLOAT, text_tsv tsvector)
  -- speech-to-text segments
  -- full-text search: text_tsv @@ plainto_tsquery('english', '<word>')

speaker_emotion(id, audio_id, speaker TEXT, emotion_vector JSONB, start_time FLOAT, end_time FLOAT)
  -- speaker diarization + emotion per segment
  -- emotion_vector JSON dict: {"happy": 0.87, "neutral": 0.10, ...}
  -- to get dominant emotion: use jsonb_each(emotion_vector) ORDER BY value DESC

non_semantics_data(id, audio_id, labels JSONB, start_time FLOAT, end_time FLOAT)
  -- audio events detected by BEATs model
  -- labels JSON array: [{"label": "Speech", "score": 0.95}, ...]
  -- to search by label: labels @> '[{"label":"<value>"}]'
"""

# ---------------------------------------------------------------------------
# Tool definitions (identical to the inference-time SearchAgent)
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_database",
            "description": (
                "Execute a SQL SELECT query against the audio database. "
                "Always filter by audio_id. Return only relevant columns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A valid PostgreSQL SELECT statement.",
                    }
                },
                "required": ["sql"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_snippets",
            "description": (
                "Submit the snippets you have collected as relevant evidence. "
                "Call this once you have gathered enough context to answer the question. "
                "Each snippet must have: source, start_time, end_time, content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "snippets": {
                        "type": "array",
                        "description": "List of relevant snippets.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "string",
                                    "enum": [
                                        "transcription",
                                        "speaker_emotion",
                                        "audio_events",
                                    ],
                                },
                                "start_time": {"type": "number"},
                                "end_time": {"type": "number"},
                                "content": {"type": "string"},
                            },
                            "required": ["source", "start_time", "end_time", "content"],
                        },
                    }
                },
                "required": ["snippets"],
            },
        },
    },
]


def _format_snippets(snippets: list[dict]) -> str:
    if not snippets:
        return "(no relevant snippets found)"
    return "\n".join(
        f"[{s['source']} {s['start_time']:.1f}s\u2013{s['end_time']:.1f}s] {s['content']}"
        for s in snippets
    )


class AudioSearchAgent:
    """Search-and-answer agent for audio QA, compatible with AReaL proxy workflow.

    AReaL detects that this class is not a RolloutWorkflow and automatically
    wraps it with OpenAIProxyWorkflow, injecting `base_url` and `api_key`
    (pointing to the proxy server) into every call to `self.run()`.
    """

    def __init__(
        self,
        max_search_turns: int = 5,
        postgres_host: str = "localhost",
        postgres_port: int = 5433,
        postgres_db: str = "espnet_db",
        postgres_user: str = "espnet_user",
        postgres_password: str = "espnet_password",
    ):
        self.max_search_turns = max_search_turns
        self._db_kwargs = dict(
            host=postgres_host,
            port=postgres_port,
            dbname=postgres_db,
            user=postgres_user,
            password=postgres_password,
        )

    # ------------------------------------------------------------------
    # DB helpers (synchronous; called via run_in_executor)
    # ------------------------------------------------------------------

    def _run_sql(self, sql: str) -> tuple[str, int]:
        """Execute a SQL query. Returns (formatted_result, n_rows)."""
        sql = sql.strip()
        if not sql.upper().lstrip().startswith("SELECT"):
            return "Error: only SELECT queries are allowed.", 0
        try:
            conn = psycopg2.connect(**self._db_kwargs)
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(sql)
                rows = cur.fetchmany(50)
            conn.close()
        except Exception as exc:
            return f"SQL error: {exc}", 0

        n_rows = len(rows)
        logger.debug(f"query_database → {n_rows} row(s)")

        if not rows:
            return "No rows returned.", 0

        cols = list(rows[0].keys())
        lines = [" | ".join(cols), " | ".join("---" for _ in cols)]
        for row in rows:
            lines.append(" | ".join(str(row[c]) for c in cols))
        return "\n".join(lines), n_rows

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _system_prompt(self, audio_id: str) -> str:
        schema = _SCHEMA.replace("<audio_id>", audio_id)
        return (
            "You are a database search agent. Your sole job is to find relevant "
            "snippets from an audio database to help answer a question.\n\n"
            "Tools:\n"
            "- query_database: write a SQL SELECT query; the tool executes it\n"
            "- submit_snippets: submit relevant snippets and end your search\n\n"
            "Guidelines:\n"
            "- Always filter by audio_id in WHERE clause.\n"
            "- Use ILIKE for keyword search on text.\n"
            "- Add LIMIT 20 to every query.\n"
            "- Narrow iteratively: start specific, broaden only if needed.\n"
            "- Never fetch all rows without a filter.\n\n"
            f"Schema:\n{schema}"
        )

    # ------------------------------------------------------------------
    # Core agent loop
    # ------------------------------------------------------------------

    async def _search_loop(
        self,
        client: Any,
        audio_id: str,
        question: str,
    ) -> tuple[list[dict], dict]:
        """Run the SQL search loop. Returns (snippets, stats)."""
        messages: list[dict] = [
            {"role": "system", "content": self._system_prompt(audio_id)},
            {"role": "user", "content": question},
        ]
        snippets: list[dict] | None = None
        loop = asyncio.get_event_loop()
        stats = {"n_queries": 0, "n_rows_total": 0, "n_empty_queries": 0}

        for _ in range(self.max_search_turns):
            response = await client.chat.completions.create(
                model="default",
                messages=messages,
                tools=_TOOLS,
                tool_choice="required",
                max_tokens=1024,
            )
            msg = response.choices[0].message
            messages.append(msg.model_dump(exclude_unset=True))

            if not msg.tool_calls:
                break

            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)

                if name == "query_database":
                    result, n_rows = await loop.run_in_executor(
                        None, self._run_sql, args["sql"]
                    )
                    stats["n_queries"] += 1
                    stats["n_rows_total"] += n_rows
                    if n_rows == 0:
                        stats["n_empty_queries"] += 1
                elif name == "submit_snippets":
                    snippets = args["snippets"]
                    result = f"Submitted {len(snippets)} snippet(s)."
                else:
                    result = f"Unknown tool: {name}"

                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": result}
                )

            if snippets is not None:
                break

        return snippets or [], stats

    async def _generate_answer(
        self,
        client: Any,
        question: str,
        snippets: list[dict],
    ) -> str:
        """Generate final answer from retrieved snippets."""
        context = _format_snippets(snippets)
        prompt = (
            f"The following snippets were retrieved from an audio database:\n\n"
            f"{context}\n\n"
            f"Based on these snippets, answer the question concisely.\n\n"
            f"Question: {question}"
        )
        response = await client.chat.completions.create(
            model="default",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Entry point called by OpenAIProxyWorkflow
    # ------------------------------------------------------------------

    async def run(
        self,
        data: dict,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client=None,
        **kwargs,
    ) -> float:
        """Run the full search-and-answer pipeline; return scalar reward.

        AReaL's OpenAIProxyWorkflow calls this method with `base_url` and
        `api_key` pointing to the proxy server.  All LLM calls made through
        `client` are intercepted and tracked for GRPO training.

        In `subproc` mode, OpenAIProxyWorkflow sets OPENAI_BASE_URL and
        OPENAI_API_KEY environment variables instead of passing kwargs, so
        both parameters have environment-variable fallbacks.

        Returns
        -------
        float
            Reward attached to the last (answer-generation) completion.
            Earlier (search) completions receive discounted rewards via
            OpenAIProxyWorkflow's `turn_discount` setting.
        """
        from openai import AsyncOpenAI

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        client_kwargs: dict = {"base_url": resolved_base_url, "api_key": resolved_api_key}
        if http_client is not None:
            client_kwargs["http_client"] = http_client
        client = AsyncOpenAI(**client_kwargs)

        audio_id: str = data["audio_id"]
        question: str = data["messages"][-1]["content"]
        answer: str = data["answer"]

        snippets, stats = await self._search_loop(client, audio_id, question)
        final_answer = await self._generate_answer(client, question, snippets)
        reward = audio_qa_reward(final_answer, answer)

        logger.info(
            f"[{audio_id}] queries={stats['n_queries']} "
            f"rows_total={stats['n_rows_total']} "
            f"empty_queries={stats['n_empty_queries']} "
            f"snippets_submitted={len(snippets)} "
            f"reward={reward:.3f}"
        )
        return reward
