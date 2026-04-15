"""Keyword search over ASR transcript segments.

Provides grep-style keyword matching against the ASR transcript of the
current audio recording. Returns matching segments with timestamps, speaker
IDs, and surrounding context.

The transcript is loaded from the raw MeetingBank Azure Cognitive Services
JSON files. Segments are indexed by their text content for fast keyword lookup.

Example tool call from the LLM:
    {
        "name": "transcript_search",
        "arguments": {
            "query": "budget amendment",
            "top_k": 20
        }
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from areal.utils.logging import getLogger

from .base import Tool

logger = getLogger("TranscriptSearch")

# Azure Cognitive Services uses 100-nanosecond ticks.
TICKS_TO_SECONDS = 1e-7


class TranscriptSearchTool(Tool):
    """Keyword search over ASR transcript segments with timestamps."""

    NAME = "transcript_search"
    DESCRIPTION = (
        "Search the transcript of the current audio recording for segments "
        "containing specific keywords or phrases. Returns matching segments "
        "with start/end timestamps (seconds), speaker ID, and text.\n\n"
        "This is a text-based grep-style search — use it when looking for "
        "specific words, names, topics, or phrases mentioned in the audio. "
        "Combine with clap_search (for acoustic/non-speech content) and "
        "omni_probe (for deep verification).\n\n"
        "Tips:\n"
        "- Use short, specific keywords for best results (e.g. 'budget' not "
        "'what was said about the city budget').\n"
        "- Search for names, numbers, or distinctive phrases.\n"
        "- Use multiple single-keyword searches rather than long phrases.\n"
        "- Results include surrounding context segments for temporal grounding."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keyword or phrase to search for in the transcript. "
                    "Case-insensitive. Supports multiple words (all must appear "
                    "in the same segment)."
                ),
            },
            "top_k": {
                "type": "integer",
                "description": (
                    "Maximum number of matching segments to return "
                    "(default: 20, max: 50)."
                ),
                "default": 20,
            },
            "context": {
                "type": "integer",
                "description": (
                    "Number of surrounding segments to include before and after "
                    "each match for context (default: 1)."
                ),
                "default": 1,
            },
        },
        "required": ["query"],
    }

    def __init__(
        self,
        transcript_path: str,
        audio_id: str,
    ):
        """Create a transcript search tool for one audio recording.

        Args:
            transcript_path: Path to the Azure ASR transcript JSON file.
            audio_id:        The audio_id for this episode.
        """
        self._audio_id = audio_id
        self._segments: list[dict] = []
        self._load_transcript(transcript_path)

    def _load_transcript(self, path: str) -> None:
        """Parse Azure Cognitive Services transcript JSON into segments."""
        p = Path(path)
        if not p.exists():
            logger.warning(f"Transcript file not found: {path}")
            return

        raw = json.loads(p.read_text())
        asr_segments = raw.get("segments", [])

        for seg in asr_segments:
            offset_s = seg["offset"] * TICKS_TO_SECONDS
            duration_s = seg["duration"] * TICKS_TO_SECONDS
            text = seg["nbest"][0]["text"] if seg.get("nbest") else ""
            speaker = seg.get("speaker", -1)

            self._segments.append({
                "start_time": round(offset_s, 2),
                "end_time": round(offset_s + duration_s, 2),
                "speaker": speaker,
                "text": text,
            })

        # Sort by start time
        self._segments.sort(key=lambda s: s["start_time"])
        logger.info(
            f"Loaded transcript for {self._audio_id}: "
            f"{len(self._segments)} segments"
        )

    def execute(self, **kwargs: Any) -> dict:
        query: str = kwargs.get("query", "")
        top_k: int = min(kwargs.get("top_k", 20), 50)
        context: int = min(kwargs.get("context", 1), 3)

        if not query.strip():
            return {
                "status": "error",
                "error": "Empty query",
            }

        if not self._segments:
            return {
                "status": "ok",
                "audio_id": self._audio_id,
                "query": query,
                "num_results": 0,
                "results": [],
                "note": "No transcript available for this audio.",
            }

        logger.info(
            f"Transcript search: audio_id={self._audio_id}, "
            f"query='{query}', top_k={top_k}"
        )

        # Case-insensitive keyword matching
        keywords = query.lower().split()
        matches: list[int] = []

        for idx, seg in enumerate(self._segments):
            text_lower = seg["text"].lower()
            if all(kw in text_lower for kw in keywords):
                matches.append(idx)

        # Build results with context
        seen_indices: set[int] = set()
        results: list[dict] = []

        for match_idx in matches[:top_k]:
            # Determine context window
            ctx_start = max(0, match_idx - context)
            ctx_end = min(len(self._segments), match_idx + context + 1)

            # Merge overlapping context windows
            window_indices = list(range(ctx_start, ctx_end))
            new_indices = [i for i in window_indices if i not in seen_indices]
            if not new_indices and match_idx in seen_indices:
                continue  # Already fully covered by a previous match's context
            seen_indices.update(window_indices)

            # Build context segments
            context_segs = []
            for i in window_indices:
                seg = self._segments[i]
                context_segs.append({
                    "start_time": seg["start_time"],
                    "end_time": seg["end_time"],
                    "speaker": seg["speaker"],
                    "text": seg["text"],
                    "is_match": i == match_idx,
                })

            results.append({
                "match_start_time": self._segments[match_idx]["start_time"],
                "match_end_time": self._segments[match_idx]["end_time"],
                "match_text": self._segments[match_idx]["text"],
                "speaker": self._segments[match_idx]["speaker"],
                "context": context_segs,
            })

        return {
            "status": "ok",
            "audio_id": self._audio_id,
            "query": query,
            "num_results": len(results),
            "total_matches": len(matches),
            "results": results,
        }
