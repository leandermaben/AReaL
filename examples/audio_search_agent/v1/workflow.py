"""AReaL-compatible workflow for the v1 AudioSearchAgent.

This is an "agent-like workflow" (not a RolloutWorkflow subclass) so AReaL
automatically wraps it with OpenAIProxyWorkflow when passed as the `workflow`
argument to PPOTrainer.train().

The workflow:
  1. Instantiates tools (CLAP search, Omni probe, submit) for the episode's audio.
  2. Runs the agent loop with the AReaL proxy client.
  3. Computes span-level F1 reward, gated on step count.
  4. Logs custom metrics (turns, spans, F1) to wandb via stats_tracker.
  5. Returns the scalar reward.
"""
from __future__ import annotations

import os
from typing import Any

from areal.utils import stats_tracker
from areal.utils.logging import getLogger

from .reward import span_f1, step_gated_f1_reward

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
    ):
        self.max_search_turns = max_search_turns
        self.step_limit = step_limit
        self.clap_cache_dir = clap_cache_dir
        self.data_root = data_root
        self.clap_device = clap_device
        self.omni_url = omni_url
        self.omni_model = omni_model
        self.omni_slice_tmpdir = omni_slice_tmpdir

        # Lazy-loaded CLAP indexer (shared across episodes)
        self._clap_indexer = None

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
            tools.append(OmniProbeTool(
                omni_url=self.omni_url,
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
        """Run the full search-and-answer pipeline; return scalar reward.

        Returns
        -------
        float
            Reward: span F1 gated on using exactly step_limit turns.
        """
        from .agent.agent_loop import AudioSearchAgent

        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY")

        audio_id: str = data["audio_id"]
        question: str = data["messages"][-1]["content"]
        gold_spans: list[dict] = data.get("gold_spans", [])
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

        # Extract submitted spans
        submission = result.get("submission")
        predicted_spans = []
        if submission and submission.get("status") == "ok":
            predicted_spans = submission.get("snippets", [])

        n_turns = result.get("turns", 0)
        status = result.get("status", "error")
        did_submit = 1.0 if status == "submitted" else 0.0

        # Compute raw F1 (un-gated, for logging) and gated reward
        raw_f1 = span_f1(predicted_spans, gold_spans)
        reward = step_gated_f1_reward(
            predicted_spans=predicted_spans,
            gold_spans=gold_spans,
            n_turns=n_turns,
            step_limit=self.step_limit,
        )

        # Log custom metrics to wandb via stats_tracker
        try:
            from areal import workflow_context
            tracker = stats_tracker.get(workflow_context.stat_scope())
            tracker.scalar(
                reward=reward,
                raw_span_f1=raw_f1,
                num_turns=float(n_turns),
                did_submit=did_submit,
                num_predicted_spans=float(len(predicted_spans)),
                num_gold_spans=float(len(gold_spans)),
            )
        except Exception:
            # stats_tracker may not be available outside AReaL training
            pass

        logger.info(
            f"[{audio_id}] status={status} turns={n_turns}/{self.step_limit} "
            f"submitted={len(predicted_spans)} gold={len(gold_spans)} "
            f"raw_f1={raw_f1:.3f} reward={reward:.3f}"
        )
        return reward
