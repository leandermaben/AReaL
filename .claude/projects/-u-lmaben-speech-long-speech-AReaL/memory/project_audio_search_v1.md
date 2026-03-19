---
name: project_audio_search_v1
description: Audio search agent v1 — CLAP preprocessing pipeline status, architecture, and planned research forks
type: project
---

## Overview
RL-trained agent that searches long audio recordings to answer questions.
Pipeline: Audio → Search Agent (Qwen3-4B) uses tools to find relevant snippets → Answering system.

## Phase 1 (CLAP preprocessing) — implemented as of 2026-03-10
- Code at: `examples/audio_search_agent/v1/preprocessing/`
- `clap_indexer.py`: CLAPIndexer class — segments audio (9s chunks, 3s stride), embeds with `laion/larger_clap_general`, builds per-audio FAISS IndexFlatIP (cosine sim on L2-normed 512-dim vectors)
- `preprocess.py`: CLI with `--mode embed|index|all`, multi-GPU sharding via `--worker-id`/`--num-workers`
- `launch/prepare_clap_index.sbatch`: 2× A100 40GB, 2 parallel embed workers then single FAISS build
- Per-audio indexes: search is scoped to a specific `audio_id` (no cross-audio bleed)
- Cache: `/work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank/` (embeddings/*.npz, faiss/*.index + *.jsonl)
- Data: `/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared/` — 273 audio files (train:217, val:26, test:30), up to 80min each, 16kHz wav + json sidecar

## Planned tools for the agent
1. CLAP retriever (Phase 1 — done)
2. Transcript Probe (ASR on selected audio span)
3. Omni Probe (multimodal model on selected span + query)
4. Non-model tools: VAD, silence/loudness peaks, change points, repeat detector, temporal expand, window sampler, cluster summary
5. PostgreSQL-backed ASR+Emotion+Diarization+Events (separate research fork)
6. Submit tool (final snippet list with start/end times)

## Research forks (planned)
- **Fork 1**: Agent with full preprocessing (PostgreSQL with multiple models) — more tools available
- **Fork 2**: Agent with only CLAP soft matching + probes — fewer tools, less preprocessing
- These will become v1.1 and v1.2

## RL
- Primary reward: precision + recall on retrieved snippets
- Secondary rewards: turn count, tool costs, snippet length costs, DB row costs

## v0 reference
- `examples/audio_search_agent/desc_v0.md` — PostgreSQL-based SQL search agent, Qwen3-4B, AReaL GRPO training
- v0 used `AudioSearchAgent`, `AudioQADataset` loggers (already registered)
