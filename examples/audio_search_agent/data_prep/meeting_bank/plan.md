# MeetingBank Data Prep v2 — Implementation Plan

## Problems with v1

1. **Random span sampling** — questions are built from random segments, not meaningful
   events. Gold spans are often noisy/irrelevant.
2. **No context length management** — prompts can blow up when segments are long.
3. **Weak multi-hop** — claimed multi-hop questions often answerable from a single span.
4. **No structured events** — questions don't reflect the natural structure of meetings
   (motions, votes, presentations, public comments).
5. **Speaker info underused** — speaker IDs exist but aren't enriched with names/roles.

## Design Principles

- **Each script is one phase** — idempotent, reads from previous phase's output, writes
  its own. Can re-run any phase independently.
- **All LLM calls go through a shared async client** with concurrency control, retry,
  token counting, and context length budgeting.
- **Intermediate artifacts are JSONL/Parquet** — inspectable, resumable.
- **Final output is the same `audio_N.json` format** as v1 so the training pipeline
  needs zero changes.

## Inputs

- `{data_root}/{city}/mp3/*.mp3` — raw audio
- `{data_root}/{city}/transcripts/*.mp3.transcript.json` — time-aligned segments with
  word-level timings and speaker IDs

281 total files across 5 cities (Boston, Denver, KingCounty, LongBeach, Seattle).

## Output

Same format as v1 (`audio_N.wav` + `audio_N.json` + `manifest.json`) so dataset loader
and training pipeline are unchanged. But with much higher quality questions.

---

## Phase 1: Preprocess (no LLM)

**Script**: `01_preprocess.py`
**Input**: Raw transcript JSONs
**Output**: `processed/segments.jsonl` — one row per segment

For each meeting:
1. Parse transcript JSON, convert microseconds to seconds.
2. Emit one row per segment:
   ```json
   {"meeting_id": "Denver/xxx", "seg_idx": 0, "speaker_id": 1,
    "start_sec": 12.3, "end_sec": 15.6, "text": "...",
    "word_count": 42, "city": "Denver"}
   ```
3. Build **speaker turns** — merge consecutive segments from the same speaker into turns:
   ```json
   {"meeting_id": "...", "turn_idx": 0, "speaker_id": 1,
    "start_sec": 12.3, "end_sec": 45.6, "text": "...", "seg_indices": [0,1,2]}
   ```
   Output: `processed/turns.jsonl`
4. Build **topic windows** — sliding windows of ~2-5 minutes with 50% overlap, for
   feeding to the LLM in later phases. Each window records which segments and turns it
   contains. Cap text to ~2000 tokens per window.
   Output: `processed/windows.jsonl`

**Context length handling**: Windows are built to a token budget (~2000 tokens) so
downstream LLM prompts never exceed limits.

---

## Phase 2: Event Extraction (LLM)

**Script**: `02_extract_events.py`
**Input**: `processed/windows.jsonl`
**Output**: `processed/events.jsonl`

For each topic window, prompt the LLM to extract structured events:

```
Given this transcript window from a city council meeting:
[Speaker 1, 12:30-14:45]: "..."
[Speaker 2, 14:45-15:30]: "..."

Extract all discrete events. For each event, output:
- event_type: one of [agenda_transition, presentation, public_comment, motion,
  vote, vote_result, proclamation, question, response, discussion, adjournment, other]
- label: short canonical name (e.g. "Resolution 14 — airport contract")
- speakers: list of speaker IDs involved
- start_sec, end_sec: time bounds
- summary: 1-2 sentence description

Return JSON array. If no events, return [].
```

**Context length**: Each window is pre-budgeted to ~2000 tokens. With system prompt +
output budget, total stays under 4096 tokens per call.

Post-processing:
- Snap event times to 3-second grid.
- Deduplicate events that span overlapping windows (same label + overlapping time →
  merge, keep union of time bounds).
- Assign globally unique `event_id` per meeting.

---

## Phase 3: Speaker Enrichment (LLM)

**Script**: `03_enrich_speakers.py`
**Input**: `processed/turns.jsonl`, `processed/events.jsonl`
**Output**: `processed/speakers.jsonl`

For each meeting, collect the first ~10 turns per speaker ID. Prompt:

```
Below are transcript excerpts from different speakers in a city council meeting.
For each speaker ID, infer:
- name (if identifiable from self-introduction or being addressed)
- role: one of [chair, councilmember, staff, presenter, public_commenter, unknown]

Speaker 0:
"Good afternoon, I'm Council President Lorena Gonzalez..."

Speaker 1:
"Thank you Madam President. I'd like to move that we..."

Return JSON: [{"speaker_id": 0, "name": "Lorena Gonzalez", "role": "chair"}, ...]
```

**Context length**: ~10 turns × ~100 tokens each = ~1000 tokens. Fits easily.

---

## Phase 4: Event Graph (no LLM)

**Script**: `04_build_graph.py`
**Input**: `processed/events.jsonl`
**Output**: `processed/event_edges.jsonl`

Build edges heuristically:
- **temporal**: `follows(A, B)` if A.end < B.start and they're in the same meeting
- **same_topic**: events with overlapping labels (fuzzy string match on canonical label)
- **motion→vote**: motion event followed by vote event with matching label
- **question→response**: question event followed by response from different speaker
- **presentation→discussion**: presentation followed by discussion on same topic

Each edge: `{"src_event_id": ..., "dst_event_id": ..., "edge_type": "motion_to_vote"}`

---

## Phase 5: Question Generation (LLM)

**Script**: `05_generate_questions.py`
**Input**: `processed/events.jsonl`, `processed/event_edges.jsonl`,
`processed/speakers.jsonl`
**Output**: `qa/qa_candidates.jsonl`

### Strategy: Generate from event structures, not random spans

**Single-event questions** (~40% of questions):
- Pick one event. Feed its supporting transcript to the LLM.
- Ask it to generate an MCQ about that event.
- Gold spans = the event's time bounds.

**Multi-hop questions via graph patterns** (~40% of questions):
- Pick an edge (e.g. motion→vote). Feed both events' transcript.
- Prompt: "Generate a question that requires information from BOTH events to answer.
  The reader must first learn X from event A, then use that to answer about event B."
- Gold spans = union of both events' time bounds.

Concrete patterns:
| Pattern | Example |
|---|---|
| motion + vote_result | "What was the outcome of the motion regarding [topic]?" |
| question + response | "How did [speaker] respond to the question about [topic]?" |
| person + contribution | "What did the person who [introduced X] say about [Y]?" |
| presentation + later_action | "What action did the council take after the presentation on [X]?" |

**Template questions** (~20% of questions):
- **Speaker count**: Count unique `speaker_id`s per meeting from `segments.jsonl`.
  Generate MCQ: "How many different speakers participated in this meeting?"
  with the correct count as answer and 3 nearby integers as distractors.
  Gold spans = full audio duration (entire meeting).
  Answer is deterministic — no LLM needed.

### Context length management

For each LLM call:
1. Compute token count of event transcript(s) using tiktoken or similar.
2. If total > 3000 tokens, truncate the longest event's text to fit.
3. System prompt (~200 tokens) + event text (~3000) + output budget (~800) = ~4000 total.
4. Log a warning if truncation happens.

### Prompt template (single-event)

```
Below is a transcript excerpt from a city council meeting.

Event: {event_type} — "{label}"
Time: {start} to {end}
Speakers: {speaker_names}
Transcript:
{text}

Generate one multiple-choice question (A/B/C/D) that:
- Is answerable ONLY from this transcript
- Has one clearly correct answer
- Has 3 plausible but wrong distractors
- Tests comprehension of specific details (names, numbers, decisions), not vague summaries

Return ONLY this JSON:
{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer": "A", "comment": "brief explanation"}
```

### Prompt template (multi-hop)

```
Below are two related transcript excerpts from a city council meeting.

Event 1: {type1} — "{label1}" ({start1} to {end1})
Speakers: {speakers1}
{text1}

Event 2: {type2} — "{label2}" ({start2} to {end2})
Speakers: {speakers2}
{text2}

Relationship: {edge_type}

Generate one multiple-choice question that REQUIRES information from BOTH events.
The reader must first extract a fact from one event, then use it to answer about the other.

Return ONLY this JSON:
{"question": "...", "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
 "answer": "A", "is_multi_hop": true,
 "comment": "explain the reasoning chain across both events"}
```

---

## Phase 6: Verification & Filtering (LLM)

**Script**: `06_verify_and_filter.py`
**Input**: `qa/qa_candidates.jsonl`, `processed/events.jsonl`
**Output**: `qa/qa_verified.jsonl`

For each candidate question, run a verification prompt:

```
Question: {question}
Options: A: ... | B: ... | C: ... | D: ...
Claimed answer: {answer}

Supporting transcript:
{gold_span_text}

Verify:
1. Is the claimed answer correct?
2. Is the question answerable from ONLY the supporting transcript?
3. Are ALL supporting spans necessary (removing any one makes it unanswerable)?
4. If multi-hop: does answering truly require chaining across spans?
5. Is the question unambiguous?

Return ONLY: {"correct": true, "answerable": true, "all_spans_necessary": true,
"multi_hop_confirmed": false, "unambiguous": true, "issues": ""}
```

**Filter**: Keep only questions where `correct AND answerable AND unambiguous`.
For multi-hop: also require `multi_hop_confirmed`.

---

## Phase 7: Assembly (no LLM)

**Script**: `07_assemble.py`
**Input**: Raw audio, `qa/qa_verified.jsonl`, `processed/speakers.jsonl`
**Output**: Final `audio_N.wav` + `audio_N.json` + `manifest.json`

1. Assign train/val/test splits (80/10/10 by audio file).
2. Convert MP3→WAV (16kHz mono).
3. Assemble per-audio JSON in the same schema as v1.
4. Write manifest.

---

## Directory Layout

```
examples/audio_search_agent/data_prep/meeting_bank/
  plan.md                    # this file
  scripts/
    shared.py                # async LLM client, token counting, retry, helpers
    01_preprocess.py
    02_extract_events.py
    03_enrich_speakers.py
    04_build_graph.py
    05_generate_questions.py
    06_verify_and_filter.py
    07_assemble.py
  prompts/                   # prompt templates as text files (optional)

{output_dir}/
  processed/
    segments.jsonl
    turns.jsonl
    windows.jsonl
    events.jsonl
    event_edges.jsonl
    speakers.jsonl
  qa/
    qa_candidates.jsonl
    qa_verified.jsonl
  train/  val/  test/        # final output (same as v1)
  manifest.json
```

---

## Shared Utilities (`shared.py`)

```python
# Key components:
# - AsyncLLMClient: wraps AsyncOpenAI with concurrency semaphore, retry, server-down detection
# - count_tokens(text) -> int: fast token counting for context budgeting
# - truncate_to_budget(text, max_tokens) -> str: smart truncation preserving sentence boundaries
# - snap_to_3s(start, end) -> (float, float): snap to 3-second grid
# - parse_json_response(raw) -> dict | None: strip markdown fences, parse JSON
# - load_segments(transcript_path) -> list[dict]: parse raw transcript
# - fmt_time(seconds) -> str: human-readable time
```

---

## Execution Order

```bash
# All scripts assume a vLLM server is running for LLM phases

# Phase 1: No LLM
python scripts/01_preprocess.py --data-root /path/to/raw --output-dir /path/to/output

# Phase 2: LLM
python scripts/02_extract_events.py --output-dir /path/to/output --vllm-url http://localhost:8000/v1

# Phase 3: LLM
python scripts/03_enrich_speakers.py --output-dir /path/to/output --vllm-url http://localhost:8000/v1

# Phase 4: No LLM
python scripts/04_build_graph.py --output-dir /path/to/output

# Phase 5: LLM
python scripts/05_generate_questions.py --output-dir /path/to/output --vllm-url http://localhost:8000/v1

# Phase 6: LLM
python scripts/06_verify_and_filter.py --output-dir /path/to/output --vllm-url http://localhost:8000/v1

# Phase 7: No LLM
python scripts/07_assemble.py --data-root /path/to/raw --output-dir /path/to/output
```

Each script is resumable — checks for existing output and skips already-processed items.

---

## Context Length Budget

| Phase | Input tokens | Output tokens | Total | Strategy |
|---|---|---|---|---|
| Event extraction | ~2000 (window) | ~500 | ~2700 | Windows pre-built to budget |
| Speaker enrichment | ~1000 (10 turns) | ~200 | ~1400 | Cap turns per speaker |
| Question gen (single) | ~1500 (1 event) | ~800 | ~2500 | Truncate event text if needed |
| Question gen (multi-hop) | ~2500 (2 events) | ~800 | ~3500 | Truncate longer event |
| Verification | ~2000 (spans + Q) | ~200 | ~2400 | Already bounded by gen phase |

All well within 4K-8K context. Safe for Qwen3-4B (32K context) with large margin.

---

## Estimated Scale

- 281 audio files
- ~50 topic windows per audio → ~14K event extraction calls
- ~8 speakers per audio → ~281 speaker enrichment calls
- ~30 questions per audio (20 LLM + 10 template) → ~5600 question gen calls
- ~5600 verification calls
- **Total: ~26K LLM calls** at ~3K tokens each ≈ ~80M tokens

With 64 concurrent requests to a local vLLM server, ~2-4 hours total.
