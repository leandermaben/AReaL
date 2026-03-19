"""Omni probe tool sanity check.

Slices a ~60s segment from a val audio, sends it to Qwen3-Omni via the
OmniProbeTool, and prints the structured response (answer, transcript,
speakers, events).

Usage:
    python -m examples.audio_search_agent.sanity_checks.omni.test_omni_probe \
        [--omni-url http://gpuc05:8901/v1] \
        [--audio-idx 0] \
        [--start-time 0] \
        [--duration 60]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.audio_search_agent.v1.agent.tools.omni_probe import OmniProbeTool

DATA_ROOT = "/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"


def main():
    parser = argparse.ArgumentParser(description="Omni probe tool sanity check")
    parser.add_argument("--omni-url", default="http://gpuc05:8901/v1",
                        help="vLLM server base URL")
    parser.add_argument("--audio-idx", type=int, default=0,
                        help="Index of val audio file (audio_N.wav)")
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="Start time in seconds for the probe segment")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Duration of the segment in seconds")
    parser.add_argument("--question", type=str,
                        default="What is being discussed in this audio segment? "
                                "Provide a detailed summary.",
                        help="Question to ask about the segment")
    parser.add_argument("--slice-tmpdir", type=str,
                        default="/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared/val",
                        help="Temp dir for sliced audio (must be on shared storage "
                             "visible to the vLLM server)")
    args = parser.parse_args()

    val_dir = Path(DATA_ROOT) / "val"
    wav_path = val_dir / f"audio_{args.audio_idx}.wav"
    json_path = val_dir / f"audio_{args.audio_idx}.json"

    if not wav_path.exists():
        print(f"ERROR: {wav_path} not found")
        sys.exit(1)

    meta = json.loads(json_path.read_text()) if json_path.exists() else {}
    audio_id = meta.get("audio_id", f"audio_{args.audio_idx}")
    duration = meta.get("duration_seconds", 0)

    end_time = args.start_time + args.duration
    if duration > 0:
        end_time = min(end_time, duration)

    print(f"Audio: {audio_id}  ({wav_path})")
    print(f"Total duration: {duration:.1f}s")
    print(f"Probing segment: {args.start_time:.1f}s – {end_time:.1f}s "
          f"({end_time - args.start_time:.1f}s)")
    print(f"Question: {args.question}")
    print(f"Server: {args.omni_url}")
    print(f"Slice tmpdir: {args.slice_tmpdir}")
    print()

    tool = OmniProbeTool(
        omni_url=args.omni_url,
        model=MODEL,
        wav_path=str(wav_path),
        audio_id=audio_id,
        slice_tmpdir=args.slice_tmpdir,
    )

    print("Calling omni_probe ...")
    result = tool.execute(
        start_time=args.start_time,
        end_time=end_time,
        question=args.question,
    )

    print(f"\nStatus: {result['status']}")
    if result["status"] == "error":
        print(f"Error: {result.get('error', 'unknown')}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print("ANSWER")
    print(f"{'='*70}")
    print(result.get("answer", "(none)"))

    print(f"\n{'='*70}")
    print("TRANSCRIPT")
    print(f"{'='*70}")
    print(result.get("transcript", "(none)"))

    print(f"\n{'='*70}")
    print(f"SPEAKERS ({result.get('num_speakers', 0)})")
    print(f"{'='*70}")
    for s in result.get("speakers", []):
        print(f"  {s.get('id', '?')}: {s.get('characteristics', '?')}")

    print(f"\n{'='*70}")
    print("EVENTS")
    print(f"{'='*70}")
    for e in result.get("events", []):
        print(f"  - {e}")

    if result.get("additional_info"):
        print(f"\n{'='*70}")
        print("ADDITIONAL INFO")
        print(f"{'='*70}")
        print(result["additional_info"])

    # Save result
    out_path = Path(__file__).parent / f"probe_result_audio{args.audio_idx}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFull result saved to {out_path}")


if __name__ == "__main__":
    main()
