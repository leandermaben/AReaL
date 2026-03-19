"""
Sanity check: clip 30s of audio, save it, and query Qwen3-Omni via AsyncOpenAI.
"""

import asyncio
import base64
import wave

from openai import AsyncOpenAI

# --- Config ---
SRC_AUDIO = "/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared/val/audio_0.wav"
CLIP_OUT = "/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared/val/audio_0_clip_30s.wav"
CLIP_SECONDS = 30
API_BASE = "http://gpub001:8901/v1"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
QUESTION = "What are the speakers discussing in this audio clip? Provide a brief summary."


def clip_wav(src: str, dst: str, seconds: int) -> str:
    """Clip the first `seconds` of a WAV file and save to `dst`."""
    with wave.open(src, "r") as r:
        sr = r.getframerate()
        ch = r.getnchannels()
        sw = r.getsampwidth()
        n_frames = min(sr * seconds, r.getnframes())
        frames = r.readframes(n_frames)

    with wave.open(dst, "w") as w:
        w.setnchannels(ch)
        w.setsampwidth(sw)
        w.setframerate(sr)
        w.writeframes(frames)

    duration = n_frames / sr
    print(f"Saved {duration:.1f}s clip to {dst}  (sr={sr}, ch={ch})")
    return dst


def audio_to_data_uri(path: str) -> str:
    """Read a WAV file and return a base64 data URI."""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:audio/wav;base64,{b64}"


async def query_model(audio_path: str) -> str:
    client = AsyncOpenAI(base_url=API_BASE, api_key="EMPTY")
    data_uri = audio_to_data_uri(audio_path)
    print(f"Audio base64 payload: {len(data_uri)} chars")

    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio_url",
                        "audio_url": {"url": data_uri},
                    },
                    {
                        "type": "text",
                        "text": QUESTION,
                    },
                ],
            }
        ],
        max_tokens=512,
    )
    return resp.choices[0].message.content


async def main():
    # 1. Clip audio
    clip_path = clip_wav(SRC_AUDIO, CLIP_OUT, CLIP_SECONDS)

    # 2. Query model
    print(f"\nSending {clip_path} to {MODEL} at {API_BASE} ...")
    answer = await query_model(clip_path)
    print(f"\n--- Model Response ---\n{answer}\n")


if __name__ == "__main__":
    asyncio.run(main())
