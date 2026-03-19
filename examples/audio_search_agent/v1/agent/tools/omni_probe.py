"""Omni probe tool — multimodal audio understanding via Qwen3-Omni.

Sends a slice of audio to Qwen3-Omni with a question and gets back structured
analysis: answer, transcript, detected events, speaker info, and additional notes.

The Qwen3-Omni model is served via vLLM with an OpenAI-compatible API.
Audio segments are saved as temp files on shared storage so the vLLM server
can read them directly — avoids base64 bloat over HTTP.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import torchaudio
from openai import AsyncOpenAI

from areal.utils.logging import getLogger

from .base import Tool

logger = getLogger("OmniProbeTool")

# Maximum audio slice duration (seconds) to prevent overly expensive calls.
MAX_SLICE_DURATION = 150.0

# Directory for temp audio slices. Must be on shared storage visible to the
# vLLM server. Falls back to system temp if not set.
_SLICE_TMPDIR = os.environ.get("OMNI_SLICE_TMPDIR", None)

_SYSTEM_PROMPT = """\
You are an expert audio analyst. You will receive an audio clip and a question about it.
Listen carefully and respond with ONLY a JSON object (no markdown, no prose) with these fields:

{
  "answer": "Direct answer to the question",
  "events": ["list", "of", "detected", "audio", "events"],
  "speakers": [
    {
      "id": "Speaker 1",
      "characteristics": "Brief description: gender, tone, accent, speaking style"
    }
  ],
  "num_speakers": 1,
  "transcript": "Verbatim transcript of the audio clip",
}

Be precise with the transcript. For events, include things like: applause, laughter,
background noise, music, silence, crosstalk, gavel, etc. For speakers, assign
sequential IDs (Speaker 1, Speaker 2, ...) and describe distinguishing characteristics."""


def _slice_audio_to_file(
    wav_path: str,
    start_time: float,
    end_time: float,
    tmpdir: str | None = None,
) -> str:
    """Slice a wav file and save the segment to a temp file.

    Returns the path to the temp file. The caller is responsible for cleanup.
    """
    waveform, sr = torchaudio.load(wav_path)  # (C, T)

    start_sample = max(0, min(int(start_time * sr), waveform.shape[1]))
    end_sample = max(start_sample, min(int(end_time * sr), waveform.shape[1]))
    segment = waveform[:, start_sample:end_sample]

    fd, tmp_path = tempfile.mkstemp(suffix=".wav", dir=tmpdir)
    os.close(fd)
    torchaudio.save(tmp_path, segment, sr, format="wav")
    return tmp_path


def _parse_json_response(raw: str) -> dict | None:
    """Try to parse JSON from an LLM response, stripping markdown fences."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


class OmniProbeTool(Tool):
    """Probe an audio segment with Qwen3-Omni for multimodal understanding."""

    NAME = "omni_probe"
    DESCRIPTION = (
        "Send a segment of the current audio to the Omni model for detailed analysis. "
        "Provide start_time and end_time (in seconds) to select the segment, and a "
        "question to ask about it. Returns the answer, a transcript of the segment, "
        "detected audio events, speaker information, and any additional noteworthy "
        "observations.\n\n"
        "Use this after clap_search to deeply analyze promising segments. "
        "Keep segments short (under 2 minutes) for best results.\n\n"
        "Returns: answer, transcript, detected audio events, and speaker information."
    )
    PARAMETERS = {
        "type": "object",
        "properties": {
            "start_time": {
                "type": "number",
                "description": "Start time of the audio segment in seconds.",
            },
            "end_time": {
                "type": "number",
                "description": "End time of the audio segment in seconds.",
            },
            "question": {
                "type": "string",
                "description": "Question to ask about this audio segment.",
            },
        },
        "required": ["start_time", "end_time", "question"],
    }

    def __init__(
        self,
        omni_url: str,
        model: str,
        wav_path: str,
        audio_id: str,
        api_key: str = "dummy",
        max_retries: int = 2,
        temperature: float = 0.2,
        slice_tmpdir: str | None = _SLICE_TMPDIR,
    ):
        """Create an Omni probe tool bound to one audio recording.

        Args:
            omni_url:    Base URL of the vLLM server (e.g. "http://localhost:8000/v1").
            model:       Model name (e.g. "Qwen/Qwen3-Omni-30B-A3B-Instruct").
            wav_path:    Path to the audio file for this episode.
            audio_id:    Audio ID for logging/tracking.
            api_key:     API key (default "dummy" for local vLLM).
            max_retries: Retry count on API failures.
            temperature: Sampling temperature for the Omni model.
            slice_tmpdir: Directory for temp audio slices. Should be on shared
                         storage if vLLM runs on a different node. Defaults to
                         $OMNI_SLICE_TMPDIR or system temp.
        """
        self._client = AsyncOpenAI(base_url=omni_url, api_key=api_key)
        self._model = model
        self._wav_path = wav_path
        self._audio_id = audio_id
        self._max_retries = max_retries
        self._temperature = temperature
        self._slice_tmpdir = slice_tmpdir

    @property
    def audio_id(self) -> str:
        return self._audio_id

    def set_episode(self, audio_id: str, wav_path: str) -> None:
        """Switch to a new audio recording (between episodes)."""
        self._audio_id = audio_id
        self._wav_path = wav_path

    def execute(self, **kwargs: Any) -> dict:
        """Sync wrapper — runs the async call in an event loop."""
        try:
            asyncio.get_running_loop()
            raise RuntimeError(
                "OmniProbeTool.execute() called from async context. "
                "Use execute_async() instead."
            )
        except RuntimeError as e:
            if "no current event loop" in str(e) or "no running event loop" in str(e):
                return asyncio.run(self.execute_async(**kwargs))
            raise

    async def execute_async(self, **kwargs: Any) -> dict:
        """Send an audio segment to Qwen3-Omni and return structured analysis.

        Slices the audio to a temp file on shared storage and passes the file
        path to the model, avoiding base64 serialisation overhead.

        Returns:
            dict with keys:
                status:          "ok" | "error"
                audio_id:        which audio was probed
                start_time:      segment start (seconds)
                end_time:        segment end (seconds)
                question:        the question asked
                answer:          model's answer to the question
                events:          list of detected audio events
                speakers:        list of {id, characteristics} dicts
                num_speakers:    number of speakers detected
                transcript:      verbatim transcript of the segment
                error:           error message (only if status == "error")
        """
        start_time = kwargs["start_time"]
        end_time = kwargs["end_time"]
        question = kwargs["question"]

        logger.info(
            f"Omni probe: audio_id={self._audio_id}, "
            f"segment={start_time:.1f}–{end_time:.1f}s, question='{question}'"
        )

        # Validate
        if end_time <= start_time:
            return self._error(start_time, end_time, question, "end_time must be > start_time")

        duration = end_time - start_time
        if duration > MAX_SLICE_DURATION:
            return self._error(
                start_time, end_time, question,
                f"Segment too long ({duration:.0f}s). Maximum is {MAX_SLICE_DURATION:.0f}s.",
            )

        # Slice audio to a temp file
        tmp_path: str | None = None
        try:
            tmp_path = _slice_audio_to_file(
                self._wav_path, start_time, end_time, self._slice_tmpdir,
            )
        except Exception as exc:
            return self._error(start_time, end_time, question, f"Audio slicing failed: {exc}")

        try:
            return await self._call_omni(tmp_path, start_time, end_time, question)
        finally:
            # Clean up temp file
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def _call_omni(
        self, audio_path: str, start_time: float, end_time: float, question: str,
    ) -> dict:
        """Make the actual API call with the sliced audio file path."""
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": f"file://{audio_path}"},
                    },
                    {"type": "text", "text": question},
                ],
            },
        ]

        for attempt in range(self._max_retries + 1):
            try:
                resp = await self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=self._temperature,
                )
                raw = resp.choices[0].message.content or ""
                parsed = _parse_json_response(raw)

                if parsed is None:
                    logger.warning(
                        f"Omni probe: JSON parse failed (attempt {attempt+1}), "
                        f"raw={raw[:200]}"
                    )
                    if attempt < self._max_retries:
                        continue
                    # Return raw text as answer on final attempt
                    parsed = {
                        "answer": raw,
                        "events": [],
                        "speakers": [],
                        "num_speakers": 0,
                        "transcript": "",
                    }

                return {
                    "status": "ok",
                    "audio_id": self._audio_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "question": question,
                    "answer": parsed.get("answer", ""),
                    "events": parsed.get("events", []),
                    "speakers": parsed.get("speakers", []),
                    "num_speakers": parsed.get("num_speakers", 0),
                    "transcript": parsed.get("transcript", ""),
                }

            except Exception as exc:
                logger.warning(
                    f"Omni probe API error (attempt {attempt+1}/{self._max_retries+1}): {exc}"
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(1 + attempt)
                else:
                    return self._error(
                        start_time, end_time, question, f"API call failed: {exc}"
                    )

        return self._error(start_time, end_time, question, "Exhausted retries")

    def _error(
        self, start_time: float, end_time: float, question: str, error: str
    ) -> dict:
        """Build a standardised error response."""
        return {
            "status": "error",
            "audio_id": self._audio_id,
            "start_time": start_time,
            "end_time": end_time,
            "question": question,
            "answer": "",
            "events": [],
            "speakers": [],
            "num_speakers": 0,
            "transcript": "",
            "error": error,
        }
