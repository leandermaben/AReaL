# CLAUDE.md - AReaL

## WHAT: Project Overview

AReaL is a distributed RL training framework for LLM alignment via reinforcement
learning.

**Tech Stack**: Python 3.12+ | PyTorch | FSDP2/Megatron | SGLang/vLLM

**Core Directories**:

- `areal/` - Core package
  - `api/` - Config dataclasses, workflow/engine contracts
  - `engine/` - FSDP2, Megatron, SGLang/vLLM adapters
    - `fsdp_utils/` - FSDP2-specific utilities (checkpoint, grad, optimizer, parallel)
    - `megatron_utils/` - Megatron/FP8 utilities (checkpoint, pipeline, quantization)
    - `core/` - Engine-shared utilities (distributed, lock, model, offload)
  - `infra/` - Infrastructure (launcher, scheduler, RPC)
    - `utils/` - Infrastructure utilities (launcher, proc, http, concurrent, slurm, ray)
  - `workflow/` - RolloutWorkflow implementations
  - `reward/` - Reward functions
  - `dataset/` - Dataset loaders
  - `utils/` - Cross-cutting utilities (logging, data, checkpoints, network, RL
    functional)
- `examples/` - Training scripts and configs
- `docs/` - Jupyter Book source

## WHY: Purpose

- Enable efficient RL training for LLM alignment at scale
- Async rollout + distributed training for high throughput
- Modular design: workflows, engines, rewards, and datasets are independently extensible

## HOW: Core Commands

```bash
# Check environment
python --version              # Requires 3.12+
uv --version                  # Install: https://docs.astral.sh/uv/

# Sync dependencies
uv sync --extra cuda          # CUDA + SGLang inference (default)
uv sync --group dev           # Include dev/test packages
uv run python3 areal/tools/validate_installation.py  # Validate installation

# Pre-commit hooks
pre-commit install --install-hooks  # Set up hooks (run once)
pre-commit run --all-files    # Format and lint

# Run tests
# First check GPU availability (many tests require GPU)
python -c "import torch; print('GPU available:', torch.cuda.is_available())"
uv run pytest tests/test_<topic>.py

# Generate CLI docs
uv run python docs/generate_cli_docs.py

# Build docs (canonical, release-aligned)
./docs/build_all.sh
# Do NOT use `jupyter-book build docs/en|docs/zh` directly for final preview/release,
# because it skips AReaL-specific static setup and output packaging.
```

## Boundaries

### Constraints

- Designed for distributed GPU clusters; assume containerized execution
- Integration tests require multi-node hardware; explain skips when unavailable
- Secrets and endpoints are managed outside the repo

### Always Do

- Read relevant files before modifying code
- Run `pre-commit run --all-files` before committing
- Follow existing code patterns in the same module
- Add tests for new functionality

### Ask First

- Modifying config structures in `areal/api/cli_args.py`
- Adding new dependencies
- Changing launcher or scheduler logic
- Deleting or renaming public APIs
- Running GPU/distributed tests (check GPU first:
  `python -c "import torch; print('GPU available:', torch.cuda.is_available())"`)

### Never Do

- Hardcode secrets, paths, or endpoints
- Skip pre-commit hooks
- Guess cluster configs or rebuild CUDA/driver stacks
- Use wildcard imports (`from x import *`)

## Progressive Disclosure: Detailed Guides

| Task                   | Reference                                                     |
| ---------------------- | ------------------------------------------------------------- |
| Add Workflow           | `docs/customization/agent.md`, `areal/workflow/multi_turn.py` |
| Add Dataset            | `docs/customization/`, `areal/dataset/gsm8k.py`               |
| Add Reward             | `areal/api/reward_api.py`, `areal/reward/geometry3k.py`       |
| Add Archon Model       | `areal/experimental/models/archon/qwen2/`, `qwen3/`           |
| Algorithm Details      | `docs/algorithms/*.md`                                        |
| Quickstart             | `docs/tutorial/quickstart.md`                                 |
| Architecture Deep Dive | `docs/tutorial/gsm8k_grpo.md`                                 |
| CLI Reference          | `docs/cli_reference.md`                                       |

## Git Workflow

- **Commits**: Conventional Commits (`feat:`, `fix:`, `docs:`), ~72 chars subject,
  imperative voice, reasoning in body
- **Squash**: Squash WIP commits before opening PR
- **PR requirements**: Run pre-commit, document test coverage, note hardware limitations

## Audio Search Agent — Key Decisions & Status

This section captures hard-won debugging context for `examples/audio_search_agent/`.

### Data Prep v2 (MeetingBank)

- **Pipeline**: 7 scripts in `examples/audio_search_agent/data_prep/meeting_bank/scripts/`
  (shared.py, 01-07). All phases completed 2026-03-31.
- **Output**: default dir `/work/nvme/bffw/lmaben/long_speech/meeting_bank_v2/`
  (override with `--output-dir`). Raw data at
  `/work/hdd/bbjs/lmaben/speech/long_speech/temp_data/`.
- **Stats**: 278 meetings, 50K events, 5,442 verified questions (5,164 single_event +
  278 speaker_count). Speaker count skipped in training.
- **Multi-hop gap**: 2,769 multi-hop candidates all filtered by `multi_hop_confirmed`
  check in `06_verify_and_filter.py:129`. To recover, relax that filter and re-run Phase 6.
- **Qwen3 think tags**: `parse_json_response` in `shared.py` strips `<think>...</think>`
  blocks — required for Qwen3 models.

### Training Config

- **Advantage norm** (Dr.GRPO): `mean_level: group`, `std_level: null`,
  `group_size: ${gconfig.n_samples}`. group_size MUST match n_samples or all advantages
  become 0 (default group_size=1 means each sample's mean = itself).
- **Concat mode**: `chat_template_type: concat`, `export_style: concat`. Enables KV cache
  reuse across multi-turn agent conversations. Safe with Qwen3 think tokens.
  Requires assistant messages to always include `"content"` key (even if empty string).
- **Tool call parser**: `qwen25` — not `qwen3` (doesn't exist). Same `<tool_call>` XML
  format used by both Qwen2.5 and Qwen3.
- **Reward design**:
  - Window F1 (primary) + auxiliary (submit called at step limit) + answer reward
  - Non-MCQ answer: always 0.5 (neutral, free-form answers are dummies)
  - MCQ answer: 0/1 match. `_is_mcq()` checks gold answer in {A,B,C,D}
- **Trajectory saving**: Probabilistic `random.random() < 1/freq` — stateless, works
  despite workflow re-instantiation on every `submit()` call
  (`_resolve_workflow` in `areal/infra/remote_inf_engine.py` creates new instance each time).

### Known Pitfalls

- **Don't use threading/shared state** in workflow code — AReaL has complex async
  processes and workflows are re-instantiated per call.
- **CLAP indexer device bug**: `torch.load(map_location="cpu")` + `assign=True` puts
  model back on CPU. Fixed to use `map_location=self.config.device`.
- **2-GPU eval proxy**: may OOM on eval-rollout fork. Disable eval on 2-GPU debug configs.
- **Dataset loader** (`v1/dataset.py`): recognizes tags `factual`, `single_event`,
  `multi_hop` as factual-eligible. Speaker count questions are skipped.

## Extended Configuration

See `.claude/agents/`, `.claude/skills/`, `.claude/commands/`, and `.claude/rules/` for
specialized instructions.

### Agents

| Agent                       | Purpose                                   | Activation Trigger                                                  |
| --------------------------- | ----------------------------------------- | ------------------------------------------------------------------- |
| `planner`                   | Implementation planning                   | Before multi-file changes, new features, or architectural decisions |
| `simple-code-reviewer`      | Quick code quality checks                 | After code changes, before committing                               |
| `code-verifier`             | Formatting/linting/tests                  | After code changes, before committing                               |
| `fsdp-engine-expert`        | FSDPEngine implementation                 | FSDPEngine code changes or questions                                |
| `archon-engine-expert`      | ArchonEngine implementation               | ArchonEngine code changes or questions                              |
| `megatron-engine-expert`    | MegatronEngine implementation             | MegatronEngine code changes or questions                            |
| `algorithm-expert`          | RL algorithms                             | GRPO/PPO/DAPO questions                                             |
| `launcher-scheduler-expert` | Cluster launching and resource scheduling | Launcher/scheduler code changes or configuration questions          |

**Stage-by-Stage Agent Guidance**:

1. **Planning Stage** (Before coding): Use `planner` for architecture design and
   implementation planning
1. **Code Formatting & Linting** (After coding): Use `code-verifier` to automatically
   run formatting, linting, and tests, catching syntax errors and style issues quickly
1. **Code Quality Check** (After formatting): Use `simple-code-reviewer` for quick code
   quality checks, focusing on logic issues and code smells

### Skills (Guided Development Workflows)

Skills provide step-by-step guides for common development tasks:

- `/add-dataset` - Dataset loader creation guide
- `/add-workflow` - Workflow implementation guide
- `/add-reward` - Reward function guide
- `/add-archon-model` - Archon engine model architecture guide
- `/debug-distributed` - Distributed debugging guide
- `/add-unit-tests` - Test development guide (NEW)

### Commands (User-invoked Actions)

Commands perform specific actions when invoked:

- `/create-pr` - Rebase, squash commits, and create/update PR with intelligent messages
- `/gen-commit-msg` - Generate commit messages from staged changes
- `/review-pr` - Intelligent PR code review with dynamic agent allocation
- `/translate-doc-zh` - Translate English documentation to Chinese

### Rules (Code Quality Standards)

Project-wide standards enforced across all code changes:

- `api-config.md` - Configuration dataclass design patterns
- `code-style.md` - Coding conventions beyond pre-commit hooks
- `distributed.md` - Distributed training patterns and constraints
- `testing.md` - Testing strategy and coverage requirements
