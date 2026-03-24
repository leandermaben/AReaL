"""AReaL-compatible workflow for the v1 AudioSearchAgent.

This is an "agent-like workflow" (not a RolloutWorkflow subclass) so AReaL
automatically wraps it with OpenAIProxyWorkflow when passed as the `workflow`
argument to PPOTrainer.train().

The workflow:
  1. Instantiates tools (CLAP search, Omni probe, submit) for the episode's audio.
  2. Runs the agent loop with the AReaL proxy client.
  3. Computes reward (window F1 + aux + answer).
  4. Logs custom metrics (turns, spans, reward components) to wandb via stats_tracker.
  5. Returns the scalar reward.
"""
from __future__ import annotations

import json
import os
import threading
from typing import Any

from areal.utils import stats_tracker
from areal.utils.logging import getLogger

from .reward import compute_reward

logger = getLogger("AudioSearchWorkflow")


class AudioSearchWorkflow:
    """Search-and-answer workflow for audio QA, compatible with AReaL proxy.

    AReaL detects that this class is not a RolloutWorkflow and wraps it with
    OpenAIProxyWorkflow, injecting `base_url` and `api_key` into `self.run()`.
    """

    def __init__(
        self,
        max_search_turns: int = 12,
        step_limit: int = 12,
        clap_cache_dir: str = "/work/hdd/bbjs/lmaben/speech/long_speech/clap_index/meetingbank",
        data_root: str = "/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared",
        clap_device: str = "cpu",
        omni_url: str | None = None,
        omni_model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        omni_slice_tmpdir: str | None = None,
        aux_weight: float = 0.15,
        answer_weight: float = 0.25,
        trajectory_log_freq: int = 10,
        trajectory_log_dir: str | None = None,
        trial_name: str = "",
    ):
        self.max_search_turns = max_search_turns
        self.step_limit = step_limit
        self.clap_cache_dir = clap_cache_dir
        self.data_root = data_root
        self.clap_device = clap_device
        self.omni_url = omni_url
        self.omni_model = omni_model
        self.omni_slice_tmpdir = omni_slice_tmpdir
        self.aux_weight = aux_weight
        self.answer_weight = answer_weight
        self.trajectory_log_freq = trajectory_log_freq
        self.trajectory_log_dir = trajectory_log_dir
        self.trial_name = trial_name

        # Lazy-loaded CLAP indexer (shared across episodes)
        self._clap_indexer = None

        # Episode counter for trajectory logging (thread-safe)
        self._episode_counter = 0
        self._counter_lock = threading.Lock()

    def _next_episode_id(self) -> int:
        """Return the next episode number (thread-safe)."""
        with self._counter_lock:
            self._episode_counter += 1
            return self._episode_counter

    def _save_trajectory(
        self, episode_id: int, audio_id: str, question: str,
        messages: list[dict], rewards: dict, gold_spans: list[dict],
        gold_answer: str, predicted_spans: list[dict], predicted_answer: str,
        is_eval: bool,
    ) -> None:
        """Save a trajectory to a JSONL file for debugging."""
        base_dir = self.trajectory_log_dir or os.environ.get(
            "TRAJECTORY_LOG_DIR", "trajectories"
        )
        log_dir = os.path.join(base_dir, self.trial_name) if self.trial_name else base_dir
        os.makedirs(log_dir, exist_ok=True)
        prefix = "eval" if is_eval else "train"
        path = os.path.join(log_dir, f"{prefix}_trajectories.jsonl")

        record = {
            "episode_id": episode_id,
            "audio_id": audio_id,
            "question": question,
            "messages": messages,
            "predicted_spans": predicted_spans,
            "predicted_answer": predicted_answer,
            "gold_spans": gold_spans,
            "gold_answer": gold_answer,
            "rewards": rewards,
        }
        with open(path, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        logger.info(f"[{audio_id}] Saved trajectory (episode {episode_id}) to {path}")

    def _get_clap_indexer(self):
        if self._clap_indexer is None:
            from .preprocessing.clap_indexer import CLAPConfig, CLAPIndexer
            config = CLAPConfig(
                data_root=self.data_root,
                cache_dir=self.clap_cache_dir,
                device=self.clap_device,
            )
            self._clap_indexer = CLAPIndexer(config)
        return self._clap_indexer

    def _build_tools(self, audio_id: str, wav_path: str):
        from .agent.tools.clap_search import CLAPSearchTool
        from .agent.tools.omni_probe import OmniProbeTool
        from .agent.tools.submit import SubmitTool

        tools = [
            CLAPSearchTool(indexer=self._get_clap_indexer(), audio_id=audio_id),
            SubmitTool(audio_id=audio_id),
        ]

        if self.omni_url:
            # Support comma-separated URLs for load balancing
            if isinstance(self.omni_url, str):
                urls = [u.strip() for u in self.omni_url.split(",") if u.strip()]
            else:
                urls = self.omni_url
            tools.append(OmniProbeTool(
                omni_url=urls if len(urls) > 1 else urls[0],
                model=self.omni_model,
                wav_path=wav_path,
                audio_id=audio_id,
                slice_tmpdir=self.omni_slice_tmpdir,
            ))

        return tools

    async def run(
        self,
        data: dict,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        http_client=None,
        **kwargs,
    ) -> float:
        """Run the full search-and-answer pipeline; return scalar reward."""
        from .agent.agent_loop import AudioSearchAgent

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        audio_id: str = data["audio_id"]
        question: str = data["messages"][-1]["content"]
        gold_spans: list[dict] = data.get("gold_spans", [])
        gold_answer: str = data.get("answer", "")
        wav_path: str = data.get("wav_path", "")
        duration: float = data.get("duration", 0.0)
        duration_str = f"{int(duration // 60)} minutes" if duration > 0 else "unknown duration"

        tools = self._build_tools(audio_id, wav_path)

        agent = AudioSearchAgent(
            llm_url=resolved_base_url,
            llm_model="default",
            tools=tools,
            max_turns=self.max_search_turns,
            api_key=resolved_api_key,
        )

        result = await agent.run_episode(
            question=question,
            audio_id=audio_id,
            duration_str=duration_str,
        )

        # Extract submitted spans and predicted answer
        submission = result.get("submission")
        predicted_spans = []
        predicted_answer = ""
        if submission and submission.get("status") == "ok":
            predicted_spans = submission.get("snippets", [])
            predicted_answer = submission.get("answer", "")

        n_turns = result.get("turns", 0)
        status = result.get("status", "error")
        did_submit = 1.0 if status == "submitted" else 0.0

        # Count total tool calls from messages
        messages = result.get("messages", [])
        num_tool_calls = sum(
            len(m.get("tool_calls", [])) for m in messages if m.get("role") == "assistant"
        )

        # Compute reward components
        rewards = compute_reward(
            predicted_spans=predicted_spans,
            gold_spans=gold_spans,
            predicted_answer=predicted_answer,
            gold_answer=gold_answer,
            n_turns=n_turns,
            step_limit=self.step_limit,
            aux_weight=self.aux_weight,
            answer_weight=self.answer_weight,
        )

        # Predicted duration: sum of all predicted span durations
        predicted_duration = sum(
            max(0.0, s.get("end_time", 0.0) - s.get("start_time", 0.0))
            for s in predicted_spans
        )

        # Log custom metrics to wandb via stats_tracker
        try:
            from areal import workflow_context
            tracker = stats_tracker.get(workflow_context.stat_scope())
            tracker.scalar(
                reward=rewards["total"],
                reward_f1=rewards["f1"],
                reward_precision=rewards["precision"],
                reward_recall=rewards["recall"],
                reward_aux=rewards["aux"],
                reward_answer=rewards["answer"],
                num_turns=float(n_turns),
                did_submit=did_submit,
                num_predicted_spans=float(len(predicted_spans)),
                num_gold_spans=float(len(gold_spans)),
                num_tool_calls=float(num_tool_calls),
                predicted_duration=predicted_duration,
            )
        except Exception:
            # stats_tracker may not be available outside AReaL training
            pass

        logger.info(
            f"[{audio_id}] status={status} turns={n_turns}/{self.step_limit} "
            f"submitted={len(predicted_spans)} gold={len(gold_spans)} "
            f"f1={rewards['f1']:.3f} p={rewards['precision']:.3f} r={rewards['recall']:.3f} "
            f"aux={rewards['aux']:.3f} answer={rewards['answer']:.3f} total={rewards['total']:.3f}"
        )

        # Save trajectory every N episodes
        episode_id = self._next_episode_id()
        if self.trajectory_log_freq > 0 and episode_id % self.trajectory_log_freq == 0:
            try:
                is_eval = False
                try:
                    from areal import workflow_context
                    is_eval = workflow_context.get().is_eval
                except Exception:
                    pass
                self._save_trajectory(
                    episode_id=episode_id,
                    audio_id=audio_id,
                    question=question,
                    messages=messages,
                    rewards=rewards,
                    gold_spans=gold_spans,
                    gold_answer=gold_answer,
                    predicted_spans=predicted_spans,
                    predicted_answer=predicted_answer,
                    is_eval=is_eval,
                )
            except Exception as exc:
                logger.warning(f"Failed to save trajectory: {exc}")

        return rewards["total"]
