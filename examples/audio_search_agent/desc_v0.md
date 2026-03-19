# Audio Search Agent — GRPO Training

RL training pipeline inside AReaL that trains **Qwen3-4B** to answer questions
about long audio by querying a PostgreSQL database (transcriptions, speaker
emotions, audio events).

---

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Loads QA JSONs from LongformAudio dirs, filters to DB-ingested audio IDs |
| `reward.py` | MCQ exact-match (A–D) + token-F1 for open-ended answers |
| `agent.py` | Async search agent: SQL tool-call loop → final answer → reward |
| `config.py` | Lightweight `AudioSearchGRPOConfig` dataclass (separated to avoid RPC import issues) |
| `train.py` | Entry point: builds datasets, calls `PPOTrainer.train()` |
| `config_2gpu.yaml` | 1-node 2× A6000: 1 vLLM GPU + 1 actor GPU |
| `config_8gpu.yaml` | 1-node 8× A6000: 2 vLLM GPUs (d2) + 6 actor GPUs (d6, FSDP data-parallel) |
| `ingest_and_train_on_qa.sbatch` | Slurm job: ingest 10-min + 30-min QA → launch 8-GPU training |

**One AReaL core change:** `areal/utils/logging.py` — color entries for
`AudioQADataset` (light_green) and `AudioSearchAgent` (light_purple).

---

## Pre-processing

Training requires two offline pre-processing steps before the first run.

### 1. Audio ingestion (`ingest_all.py` / `ingest_parallel.sh`)

Runs in the **LongAudioUnderstandingSystem** repo against the raw `.wav` files
under `/data/user_data/msomeki/30_shared/LongformAudio/`. For each audio file
three models are applied and results written to PostgreSQL (port 5433):

| Table | Model | Content |
|---|---|---|
| `transcription` | ASR (Whisper-style) | Text segments with `start_time`, `end_time`; full-text search index (`text_tsv`) |
| `speaker_emotion` | Diarization + emotion classifier | Per-speaker emotion probability vector (`{"happy": 0.87, ...}`) per segment |
| `non_semantics_data` | BEATs audio event detector | Detected event labels + confidence scores per segment |

All rows are keyed by `audio_id` (e.g. `qa-partII_10min_audio_3`).
The parallel launcher (`ingest_parallel.sh --num-gpus 8`) shards the file list
across 8 GPU workers; each worker processes a disjoint subset
(`--num-workers N --worker-id i`).

### 2. Dataset preparation (`dataset.py`)

At training startup, `get_audio_qa_dataset()`:

1. Scans the four QA JSON dirs (`qa-partI/II_10min`, `qa-partI/II_30min`) for
   `audio_*.json` files, each containing a `questions` list with `question` and
   `answer` fields.
2. Queries the DB for the set of `audio_id`s already in `transcription`
   (`filter_to_ingested=True`) and drops any sample not yet ingested.
3. Formats each surviving sample as
   `{"audio_id": ..., "messages": [{"role": "user", "content": <question>}], "answer": ...}`.
4. Splits 90 / 10 into train / validation by position.

The resulting HuggingFace `Dataset` is passed directly to `PPOTrainer`.

---

## Architecture

```
Slurm job
├── PostgreSQL on port 5433 (started from PGDATA on /data)
├── Step 1: ingest_parallel.sh --num-gpus 8 --dirs qa-partI/II_10min/30min
│     └── 8× ingest_all.py workers (LongAudioUnderstandingSystem venv)
└── Step 2: AReaL GRPO training
      ├── GPU 0-1: vLLM (2 instances, rollout)
      └── GPU 2-7: FSDP actor + colocated ref (6-way data parallel)
            └── OpenAIProxyWorkflow wraps AudioSearchAgent
                  └── each episode: SQL search loop → answer → reward
```

Each LLM tool-call step is an **independent training sample**
(`export_style: individual`), with earlier search steps receiving discounted
reward (`turn_discount: 0.9`).

---

## Key Config Values (8-GPU)

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3-4B-Instruct-2507` |
| `max_model_len` | 12288 |
| `max_tokens_per_mb` | 8192 |
| `max_new_tokens` | 1024 |
| `max_search_turns` | 4 |
| `n_samples` | 4 |
| Checkpoints | `/data/user_data/lmaben/areal/audio_search/` |

---

## Launch

```bash
# 2-GPU (dev/debug)
POSTGRES_PORT=5433 python -m examples.audio_search_agent.train \
    --config examples/audio_search_agent/config_2gpu.yaml

# 8-GPU (full run)
POSTGRES_PORT=5433 python -m examples.audio_search_agent.train \
    --config examples/audio_search_agent/config_8gpu.yaml

# Slurm (ingest + train)
sbatch examples/audio_search_agent/ingest_and_train_on_qa.sbatch
```

---

## Porting to a New Cluster

Update the following before running:

- **`cmd:` in `scheduling_spec`** — full path to the new cluster's venv Python:
  ```yaml
  cmd: /path/to/areal/.venv/bin/python -m areal.infra.rpc.rpc_server
  ```
- **`cluster.fileroot`** — checkpoint output directory
- **`cluster.name_resolve.nfs_record_root`** — NFS-accessible path for worker discovery
- **`postgres_host` / `postgres_port`** in configs
- **`PGDATA` path** in sbatch (`PG_DATA=...`)
- **`HF_TOKEN`**, **`WANDB_API_KEY`** in sbatch
- **`NCCL_SOCKET_IFNAME`** in `env_vars` if the default network interface differs
- **`source .venv/bin/activate`** before the training command in sbatch (ensures
  all child processes — vLLM server, RPC workers — use the venv Python, not the
  system Python)

---

## Bugs Fixed

| Error | Fix |
|---|---|
| `Unknown scheduler type: None` | `scheduler.type: local` |
| `'dict' has no attribute 'seed'` (RPC deserialization) | Moved config class to lightweight `config.py` |
| `Unsupported tool_call_parser: qwen3` | Changed to `qwen25` |
| `ffd_allocate: Values larger than capacity` | Raised `max_tokens_per_mb` to match longest sequence |
| OOM at `lm_head` during `ppo_update` | Lowered `max_model_len` and `max_tokens_per_mb` to 8192 |
| `SRE module mismatch` (system Python in subprocesses) | `source .venv/bin/activate` in sbatch + full venv path in `cmd` |
| NCCL timeout on multi-GPU data-parallel | `NCCL_P2P_DISABLE: "1"` in actor `env_vars` |
