"""Reward functions for audio search agent RL training.

Three reward components:
  1. Window-based F1: Flatten gold and predicted spans into unions of 3-second
     aligned windows, then compute precision/recall/F1 over those windows.
  2. Auxiliary reward: +aux_weight if n_turns == step_limit and agent submitted
     non-zero snippets.
  3. Answer reward: +answer_weight if the predicted answer exactly matches the
     ground-truth answer.

Total reward = f1 + aux + answer  (0.0 if n_turns > step_limit).
"""
from __future__ import annotations

WINDOW_SIZE = 3.0  # seconds — aligned to 3-second grid


def _spans_to_windows(spans: list[dict], window_size: float = WINDOW_SIZE) -> set[int]:
    """Flatten a list of {start_time, end_time} spans into a set of window indices.

    Each window index `i` represents the interval [i*window_size, (i+1)*window_size).
    A window is included if any part of a span overlaps with it.
    """
    windows = set()
    for s in spans:
        start = s["start_time"]
        end = s["end_time"]
        # First window that overlaps with [start, end)
        first_win = int(start // window_size)
        # Last window that overlaps (exclusive boundary)
        last_win = int((end - 1e-9) // window_size) if end > 0 else first_win
        for w in range(first_win, last_win + 1):
            windows.add(w)
    return windows


def window_f1(
    predicted_spans: list[dict],
    gold_spans: list[dict],
) -> float:
    """Compute F1 over 3-second aligned windows.

    Flattens both predicted and gold spans into sets of window indices,
    then computes precision, recall, and F1 over those sets.

    Returns:
        F1 score in [0, 1].
    """
    if not gold_spans:
        return 1.0 if not predicted_spans else 0.0
    if not predicted_spans:
        return 0.0

    pred_windows = _spans_to_windows(predicted_spans)
    gold_windows = _spans_to_windows(gold_spans)

    if not pred_windows or not gold_windows:
        return 0.0

    intersection = pred_windows & gold_windows
    precision = len(intersection) / len(pred_windows)
    recall = len(intersection) / len(gold_windows)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def answer_reward(predicted_answer: str, gold_answer: str) -> float:
    """Return 1.0 if predicted answer exactly matches gold answer, else 0.0.

    Comparison is case-insensitive and stripped of whitespace.
    """
    if not predicted_answer or not gold_answer:
        return 0.0
    return 1.0 if predicted_answer.strip().lower() == gold_answer.strip().lower() else 0.0


def compute_reward(
    predicted_spans: list[dict],
    gold_spans: list[dict],
    predicted_answer: str,
    gold_answer: str,
    n_turns: int,
    step_limit: int = 12,
    aux_weight: float = 0.15,
    answer_weight: float = 0.2,
) -> dict[str, float]:
    """Compute the total reward and its components.

    Returns:
        dict with keys: total, f1, aux, answer
    """
    if n_turns > step_limit:
        return {"total": 0.0, "f1": 0.0, "aux": 0.0, "answer": 0.0}

    f1 = window_f1(predicted_spans, gold_spans)
    aux = aux_weight if (n_turns == step_limit and predicted_spans) else 0.0
    ans = answer_weight * answer_reward(predicted_answer, gold_answer)

    return {
        "total": f1 + aux + ans,
        "f1": f1,
        "aux": aux,
        "answer": ans,
    }
