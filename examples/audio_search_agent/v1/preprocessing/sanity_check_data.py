"""Sanity check on the MeetingBank prepared dataset.

Analyzes question distribution, verification status, tag breakdown,
gold span coverage (by question type), and manifest consistency.

Generation context (prepare_meetingbank.py):
  - factual:        LLM-generated MCQ from 1-7 sampled spans (single/window/chunk).
                    Gold spans are the sampled spans themselves. Span types:
                      single = 1 segment, window = 2-5 merged segments,
                      chunk = segments filling 3/6/9/12/15s target.
                    All snapped to 3s boundaries.
  - speaker_count:  Template-based, single gold span = random window [30s, min(1800s, 0.9*dur)].
  - pseudo_question: Template-based (emotion/audio_event/events), single gold span = 5-20s.

Usage:
    python -m examples.audio_search_agent.v1.preprocessing.sanity_check_data \
        --data-root /work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared
"""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def _pct(n: int, total: int) -> str:
    return f"{n/total*100:.1f}%" if total else "N/A"


def _dist_summary(vals: list[float], unit: str = "s") -> str:
    if not vals:
        return "(no data)"
    return (
        f"min={min(vals):.1f}{unit}  p25={sorted(vals)[len(vals)//4]:.1f}{unit}  "
        f"median={statistics.median(vals):.1f}{unit}  p75={sorted(vals)[3*len(vals)//4]:.1f}{unit}  "
        f"max={max(vals):.1f}{unit}  mean={statistics.mean(vals):.1f}{unit}  "
        f"stdev={statistics.stdev(vals):.1f}{unit}" if len(vals) > 1 else
        f"single value: {vals[0]:.1f}{unit}"
    )


def main():
    parser = argparse.ArgumentParser(description="Sanity check MeetingBank prepared data.")
    parser.add_argument(
        "--data-root",
        default="/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared",
    )
    args = parser.parse_args()
    root = Path(args.data_root)

    # ── Load manifest ──────────────────────────────────────────
    manifest = json.loads((root / "manifest.json").read_text())
    manifest_by_id = {e["audio_id"]: e for e in manifest}

    print("=" * 70)
    print("MANIFEST SUMMARY")
    print("=" * 70)
    print(f"Total audio files: {len(manifest)}")

    splits = Counter(e["split"] for e in manifest)
    print(f"By split: {dict(splits)}")

    durations = [e["duration_seconds"] for e in manifest]
    print(
        f"Duration: min={min(durations):.0f}s  max={max(durations):.0f}s  "
        f"mean={statistics.mean(durations):.0f}s  median={statistics.median(durations):.0f}s"
    )

    print("\nManifest declared question counts:")
    for key in ["num_questions", "num_factual", "num_speaker_count", "num_pseudo"]:
        vals = [e.get(key, 0) for e in manifest]
        print(
            f"  {key:25s}: total={sum(vals):5d}  "
            f"min={min(vals):2d}  max={max(vals):2d}  mean={statistics.mean(vals):.1f}"
        )

    # ── Scan all JSON files ────────────────────────────────────
    print("\n" + "=" * 70)
    print("ACTUAL JSON FILE ANALYSIS")
    print("=" * 70)

    total_questions = 0
    tag_counts = Counter()
    verification_statuses = Counter()
    multi_hop_counts = Counter()

    # Per-question-type gold span analysis
    # key = tag, value = list of per-question dicts
    per_tag_span_count: dict[str, list[int]] = defaultdict(list)
    per_tag_total_dur: dict[str, list[float]] = defaultdict(list)
    per_tag_individual_span_dur: dict[str, list[float]] = defaultdict(list)

    per_audio_tag_counts = []
    manifest_mismatches = []

    # Also collect audio durations from JSONs for context
    audio_durations: dict[str, float] = {}

    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob("audio_*.json")):
            data = json.loads(jf.read_text())
            audio_id = data["audio_id"]
            audio_dur = data.get("duration_seconds", 0.0)
            audio_durations[audio_id] = audio_dur
            qs = data.get("questions", [])
            total_questions += len(qs)

            audio_tags = Counter()
            for q in qs:
                tag = q.get("tag", "missing_tag")
                tag_counts[tag] += 1
                audio_tags[tag] += 1

                multi_hop_counts[q.get("multi_hop", "missing")] += 1

                # Verification
                v = q.get("verification")
                if v is None:
                    verification_statuses["no_verification_field"] += 1
                elif v.get("issues"):
                    verification_statuses["has_issues"] += 1
                elif v.get("all_spans_necessary") is True:
                    verification_statuses["verified_clean"] += 1
                else:
                    verification_statuses["other"] += 1

                # Gold spans — per-type analysis
                spans = q.get("gold_spans", [])
                per_tag_span_count[tag].append(len(spans))
                total_dur = sum(
                    s.get("end_time", 0) - s.get("start_time", 0) for s in spans
                )
                per_tag_total_dur[tag].append(total_dur)
                for s in spans:
                    dur = s.get("end_time", 0) - s.get("start_time", 0)
                    per_tag_individual_span_dur[tag].append(dur)

            per_audio_tag_counts.append(audio_tags)

            # Check manifest consistency (note: manifest uses "num_pseudo"
            # but actual tag is "pseudo_question")
            m = manifest_by_id.get(audio_id)
            if m:
                declared = {
                    "total": m.get("num_questions", 0),
                    "factual": m.get("num_factual", 0),
                    "speaker_count": m.get("num_speaker_count", 0),
                    "pseudo": m.get("num_pseudo", 0),
                }
                actual = {
                    "total": len(qs),
                    "factual": audio_tags.get("factual", 0),
                    "speaker_count": audio_tags.get("speaker_count", 0),
                    "pseudo": audio_tags.get("pseudo_question", 0),
                }
                if declared != actual:
                    manifest_mismatches.append(
                        {"audio_id": audio_id, "declared": declared, "actual": actual}
                    )

    print(f"Total questions across all files: {total_questions}")

    # ── Tag distribution ───────────────────────────────────────
    print(f"\nQuestion tags (type):")
    for tag, count in tag_counts.most_common():
        print(f"  {tag:25s}: {count:5d}  ({_pct(count, total_questions)})")

    print(f"\nPer-audio tag counts:")
    for tag in sorted(tag_counts.keys()):
        vals = [a.get(tag, 0) for a in per_audio_tag_counts]
        print(
            f"  {tag:25s}: min={min(vals):2d}  max={max(vals):2d}  "
            f"mean={statistics.mean(vals):.1f}  median={statistics.median(vals):.0f}  "
            f"stdev={statistics.stdev(vals):.1f}"
        )

    # ── Multi-hop ──────────────────────────────────────────────
    print(f"\nMulti-hop breakdown:")
    for mh, count in multi_hop_counts.most_common():
        print(f"  {str(mh):10s}: {count:5d}  ({_pct(count, total_questions)})")

    # ── Verification ───────────────────────────────────────────
    print(f"\nVerification status:")
    for vs, count in verification_statuses.most_common():
        print(f"  {vs:30s}: {count:5d}  ({_pct(count, total_questions)})")

    issue_examples = []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob("audio_*.json")):
            data = json.loads(jf.read_text())
            for q in data.get("questions", []):
                v = q.get("verification", {})
                if v and v.get("issues"):
                    issue_examples.append(
                        {
                            "question_id": q.get("question_id"),
                            "tag": q.get("tag"),
                            "issues": v["issues"],
                        }
                    )
            if len(issue_examples) >= 10:
                break
        if len(issue_examples) >= 10:
            break

    if issue_examples:
        print(f"\nSample verification issues (up to 10):")
        for ex in issue_examples[:10]:
            print(f"  [{ex['question_id']}] ({ex['tag']}): {ex['issues'][:120]}")

    # ── Gold span duration BY QUESTION TYPE ────────────────────
    print(f"\n{'=' * 70}")
    print("GOLD SPAN DURATION BY QUESTION TYPE")
    print("=" * 70)
    print(
        "\nContext from prepare_meetingbank.py:"
        "\n  factual:        1-7 spans sampled (single seg / 2-5 seg window / 3-15s chunk), snapped to 3s"
        "\n  speaker_count:  1 span, window = U[30s, min(1800s, 0.9*audio_dur)], snapped to 3s"
        "\n  pseudo_question: 1 span, duration = U[5s, 20s], snapped to 3s"
    )

    for tag in ["factual", "speaker_count", "pseudo_question"]:
        n = tag_counts.get(tag, 0)
        if n == 0:
            continue

        print(f"\n--- {tag} ({n} questions) ---")

        # Spans per question
        sc = per_tag_span_count[tag]
        print(f"  Spans per question: {_dist_summary(sc, '')}")
        print(f"    Distribution: {Counter(sc).most_common(10)}")

        # Total gold span duration per question (sum of all spans)
        td = per_tag_total_dur[tag]
        print(f"  Total gold span duration per question:")
        print(f"    {_dist_summary(td)}")

        # Bucket the total durations for readability
        buckets = [0, 10, 30, 60, 120, 300, 600, 1200, 1800, float("inf")]
        bucket_labels = [
            "0-10s", "10-30s", "30-60s", "1-2m", "2-5m",
            "5-10m", "10-20m", "20-30m", "30m+"
        ]
        bucket_counts = [0] * len(bucket_labels)
        for d in td:
            for bi in range(len(bucket_labels)):
                if buckets[bi] <= d < buckets[bi + 1]:
                    bucket_counts[bi] += 1
                    break
        print(f"    Duration buckets:")
        for label, bc in zip(bucket_labels, bucket_counts):
            if bc > 0:
                bar = "█" * max(1, bc * 40 // n)
                print(f"      {label:8s}: {bc:5d} ({_pct(bc, n):>6s}) {bar}")

        # Individual span durations (each span separately)
        isd = per_tag_individual_span_dur[tag]
        if isd:
            print(f"  Individual span durations ({len(isd)} spans total):")
            print(f"    {_dist_summary(isd)}")

        # For speaker_count: ratio of span duration to audio duration
        if tag == "speaker_count":
            ratios = []
            # Re-scan to get per-question audio_id for ratio calculation
            for split in ["train", "val", "test"]:
                split_dir = root / split
                if not split_dir.exists():
                    continue
                for jf in sorted(split_dir.glob("audio_*.json")):
                    data = json.loads(jf.read_text())
                    aid = data["audio_id"]
                    adur = audio_durations.get(aid, 0)
                    if adur <= 0:
                        continue
                    for q in data.get("questions", []):
                        if q.get("tag") != "speaker_count":
                            continue
                        spans = q.get("gold_spans", [])
                        sdur = sum(
                            s.get("end_time", 0) - s.get("start_time", 0)
                            for s in spans
                        )
                        ratios.append(sdur / adur)
            if ratios:
                print(f"  Span duration / audio duration ratio:")
                print(f"    {_dist_summary(ratios, '')}")

    # ── Factual: multi-hop vs single-hop span analysis ─────────
    print(f"\n{'=' * 70}")
    print("FACTUAL: MULTI-HOP vs SINGLE-HOP")
    print("=" * 70)
    mh_durs, sh_durs = [], []
    mh_spans, sh_spans = [], []
    for split in ["train", "val", "test"]:
        split_dir = root / split
        if not split_dir.exists():
            continue
        for jf in sorted(split_dir.glob("audio_*.json")):
            data = json.loads(jf.read_text())
            for q in data.get("questions", []):
                if q.get("tag") != "factual":
                    continue
                spans = q.get("gold_spans", [])
                tdur = sum(s.get("end_time", 0) - s.get("start_time", 0) for s in spans)
                if q.get("multi_hop"):
                    mh_durs.append(tdur)
                    mh_spans.append(len(spans))
                else:
                    sh_durs.append(tdur)
                    sh_spans.append(len(spans))

    print(f"\n  Single-hop ({len(sh_durs)} questions):")
    print(f"    Spans per question: {_dist_summary(sh_spans, '')}")
    print(f"    Total duration:     {_dist_summary(sh_durs)}")
    print(f"\n  Multi-hop ({len(mh_durs)} questions):")
    print(f"    Spans per question: {_dist_summary(mh_spans, '')}")
    print(f"    Total duration:     {_dist_summary(mh_durs)}")

    # ── Manifest consistency ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("MANIFEST vs ACTUAL CONSISTENCY")
    print("=" * 70)
    print(f"Mismatches: {len(manifest_mismatches)} / {len(manifest)}")
    if manifest_mismatches:
        print(f"\nFirst 5 mismatches:")
        for mm in manifest_mismatches[:5]:
            print(f"  {mm['audio_id']}:")
            print(f"    declared: {mm['declared']}")
            print(f"    actual:   {mm['actual']}")

    # ── Summary ────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    verified_clean = verification_statuses.get("verified_clean", 0)
    has_issues = verification_statuses.get("has_issues", 0)
    no_field = verification_statuses.get("no_verification_field", 0)
    print(f"  Total questions:     {total_questions}")
    print(f"  Verified clean:      {verified_clean} ({_pct(verified_clean, total_questions)})")
    print(f"  Has issues:          {has_issues} ({_pct(has_issues, total_questions)})")
    print(f"  No verification:     {no_field} ({_pct(no_field, total_questions)})")
    print(f"  Manifest mismatches: {len(manifest_mismatches)}/{len(manifest)} audio files")
    print(
        f"\n  NOTE: manifest uses 'num_pseudo' but actual JSON tag is 'pseudo_question'."
        f"\n  If all mismatches are pseudo-only, this is a naming inconsistency, not data loss."
    )


if __name__ == "__main__":
    main()
