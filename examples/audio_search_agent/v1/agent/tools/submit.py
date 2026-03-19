"""Submit tool — terminal action that records the agent's final snippet selection.

The agent calls this once it has gathered enough evidence to answer the question.
Each snippet has start/end times, a reason for relevance, and optional transcript
and additional info from earlier tool calls.

Actual evaluation (precision/recall) happens in the reward function, not here.
"""

from __future__ import annotations

from typing import Any

from areal.utils.logging import getLogger

from .base import Tool

logger = getLogger("SubmitTool")


class SubmitTool(Tool):
    """Submit the final list of relevant audio snippets."""

    NAME = "submit"
    DESCRIPTION = (
        "Submit your final answer: a list of audio snippets that are relevant to the "
        "question. Call this once you have gathered enough evidence. Each snippet must "
        "have start_time and end_time (in seconds) and a reason explaining why it is "
        "relevant. Optionally include transcript text and any additional info gathered "
        "from other tools."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "snippets": {
                "type": "array",
                "description": "List of relevant audio snippets to submit.",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "number",
                            "description": "Start time of the snippet in seconds.",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "End time of the snippet in seconds.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Why this snippet is relevant to the question.",
                        },
                        "transcript": {
                            "type": "string",
                            "description": "Transcript of the snippet (if available from probing).",
                        },
                        "additional_info": {
                            "type": "string",
                            "description": (
                                "Any extra information about the snippet "
                                "(events, speaker info, etc.)."
                            ),
                        },
                    },
                    "required": ["start_time", "end_time", "reason"],
                },
            },
        },
        "required": ["snippets"],
    }

    def __init__(self, audio_id: str):
        self._audio_id = audio_id
        self._submission: list[dict] | None = None

    @property
    def audio_id(self) -> str:
        return self._audio_id

    @audio_id.setter
    def audio_id(self, value: str) -> None:
        self._audio_id = value
        self._submission = None  # reset on new episode

    @property
    def last_submission(self) -> list[dict] | None:
        """The most recent submission (for reward computation)."""
        return self._submission

    def execute(self, **kwargs: Any) -> dict:
        """Validate and record the submitted snippets.

        Returns:
            dict with status, audio_id, num_snippets, and the validated snippets.
        """
        snippets = kwargs.get("snippets", [])

        if not snippets:
            return {
                "status": "error",
                "audio_id": self._audio_id,
                "error": "No snippets provided. Submit at least one snippet.",
                "num_snippets": 0,
                "snippets": [],
            }

        validated = []
        errors = []
        for i, s in enumerate(snippets):
            start = s.get("start_time")
            end = s.get("end_time")
            reason = s.get("reason", "")

            if start is None or end is None:
                errors.append(f"Snippet {i}: missing start_time or end_time.")
                continue
            if end <= start:
                errors.append(f"Snippet {i}: end_time ({end}) must be > start_time ({start}).")
                continue
            if not reason:
                errors.append(f"Snippet {i}: reason is required.")
                continue

            validated.append({
                "start_time": float(start),
                "end_time": float(end),
                "reason": reason,
                "transcript": s.get("transcript", ""),
                "additional_info": s.get("additional_info", ""),
            })

        if errors:
            logger.warning(f"Submit validation errors for {self._audio_id}: {errors}")

        self._submission = validated

        logger.info(
            f"Submission for {self._audio_id}: {len(validated)} snippets "
            f"({len(errors)} rejected)"
        )

        result = {
            "status": "ok" if validated else "error",
            "audio_id": self._audio_id,
            "num_snippets": len(validated),
            "snippets": validated,
        }
        if errors:
            result["validation_errors"] = errors
        return result
