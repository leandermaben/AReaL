"""CLAP-based soft audio search tool.

Wraps the CLAPIndexer to provide text-to-audio retrieval scoped to the
current episode's audio_id. The agent sends a natural-language query
describing the audio content it's looking for, and gets back the top-k
matching segments with timestamps and similarity scores.

Example tool call from the LLM:
    {
        "name": "clap_search",
        "arguments": {
            "query": "applause after a vote",
            "top_k": 5
        }
    }
"""

from __future__ import annotations

import json
from typing import Any

from areal.utils.logging import getLogger

from ...preprocessing.clap_indexer import CLAPConfig, CLAPIndexer
from .base import Tool

logger = getLogger("CLAPSearchTool")


class CLAPSearchTool(Tool):
    """Text-to-audio segment retrieval using CLAP embeddings + FAISS."""

    NAME = "clap_search"
    DESCRIPTION = (
        "Search the current audio recording for segments matching a text description. "
        "Uses CLAP (Contrastive Language-Audio Pretraining) embeddings to find audio "
        "segments that semantically match your query. Returns the top-k segments with "
        "start/end timestamps (in seconds) and similarity scores.\n\n"
        "Good queries describe audio content: speech topics, sounds, emotions, events.\n"
        "Examples: 'discussion about the city budget', 'applause', 'heated argument "
        "between speakers', 'someone reading a proclamation'."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Natural-language description of the audio content to search for."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": "Number of segments to return (default: 10, max: 50).",
                "default": 10,
            },
        },
        "required": ["query"],
    }

    def __init__(self, indexer: CLAPIndexer, audio_id: str):
        """Create a CLAP search tool bound to one audio recording.

        Args:
            indexer:  A CLAPIndexer with pre-built FAISS indexes available.
            audio_id: The audio_id to search within (set per episode).
        """
        self._indexer = indexer
        self._audio_id = audio_id

    @classmethod
    def from_config(cls, config: CLAPConfig, audio_id: str) -> CLAPSearchTool:
        """Convenience constructor from a CLAPConfig."""
        indexer = CLAPIndexer(config)
        return cls(indexer=indexer, audio_id=audio_id)

    @property
    def audio_id(self) -> str:
        return self._audio_id

    @audio_id.setter
    def audio_id(self, value: str) -> None:
        """Switch to a different audio_id (e.g. between episodes)."""
        self._audio_id = value

    def execute(self, **kwargs: Any) -> dict:
        """Run CLAP text-to-audio search.

        Args (via kwargs):
            query: Text description of desired audio content.
            top_k: Number of results (default 10).

        Returns:
            dict with keys:
                status:   "ok" | "error"
                query:    the query string
                audio_id: which audio was searched
                results:  list of {start, end, score} dicts (empty on error)
                error:    error message (only if status == "error")
        """
        query = kwargs["query"]
        top_k = min(kwargs.get("top_k", 10), 50)

        logger.info(f"CLAP search: audio_id={self._audio_id}, query='{query}', top_k={top_k}")

        try:
            results = self._indexer.search_by_text(
                query_text=query,
                audio_id=self._audio_id,
                k=top_k,
            )
        except FileNotFoundError:
            return {
                "status": "error",
                "query": query,
                "audio_id": self._audio_id,
                "results": [],
                "error": f"No CLAP index found for audio_id '{self._audio_id}'.",
            }

        return {
            "status": "ok",
            "query": query,
            "audio_id": self._audio_id,
            "results": [
                {"start": r["start"], "end": r["end"], "score": round(r["score"], 4)}
                for r in results
            ],
        }
