"""Reward functions for audio search agent RL training.

Span-level F1 reward with step-gating: the agent only gets a non-zero reward
if it terminates in exactly `step_limit` turns.

Span overlap F1:
  - Each submitted snippet is checked against gold spans for temporal overlap.
  - Precision = |gold spans hit by predictions| / |predictions|
  - Recall    = |gold spans hit by predictions| / |gold spans|
  - F1 = 2 * P * R / (P + R)

A gold span counts as "hit" if any submitted snippet has IoU >= 0.5 with it.
A submitted snippet counts as "hitting" if it has IoU >= 0.5 with any gold span.
"""
from __future__ import annotations


def span_iou(
    pred_start: float, pred_end: float,
    gold_start: float, gold_end: float,
) -> float:
    """Compute IoU (Intersection over Union) of two time intervals."""
    intersection = max(0.0, min(pred_end, gold_end) - max(pred_start, gold_start))
    union = (pred_end - pred_start) + (gold_end - gold_start) - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def spans_match(
    pred_start: float, pred_end: float,
    gold_start: float, gold_end: float,
    iou_threshold: float = 0.5,
) -> bool:
    """Check if two time intervals match (IoU >= threshold)."""
    return span_iou(pred_start, pred_end, gold_start, gold_end) >= iou_threshold


def span_f1(
    predicted_spans: list[dict],
    gold_spans: list[dict],
    iou_threshold: float = 0.5,
) -> float:
    """Compute span-level F1 between predicted and gold spans.

    A span pair counts as a "hit" only if their IoU >= iou_threshold.

    Args:
        predicted_spans: list of {start_time, end_time, ...}
        gold_spans: list of {start_time, end_time, ...}
        iou_threshold: minimum IoU to count as a match (default 0.5)

    Returns:
        F1 score in [0, 1].
    """
    if not gold_spans:
        return 1.0 if not predicted_spans else 0.0
    if not predicted_spans:
        return 0.0

    # Recall: fraction of gold spans hit by at least one prediction
    gold_hits = 0
    for gs in gold_spans:
        for ps in predicted_spans:
            if spans_match(ps["start_time"], ps["end_time"],
                           gs["start_time"], gs["end_time"],
                           iou_threshold):
                gold_hits += 1
                break
    recall = gold_hits / len(gold_spans)

    # Precision: fraction of predictions that hit at least one gold span
    pred_hits = 0
    for ps in predicted_spans:
        for gs in gold_spans:
            if spans_match(ps["start_time"], ps["end_time"],
                           gs["start_time"], gs["end_time"],
                           iou_threshold):
                pred_hits += 1
                break
    precision = pred_hits / len(predicted_spans)

    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def step_gated_f1_reward(
    predicted_spans: list[dict],
    gold_spans: list[dict],
    n_turns: int,
    step_limit: int = 12,
    iou_threshold: float = 0.5,
    aux_weight: float = 0.3,
) -> float:
    """Compute span F1 reward, gated on the agent using exactly step_limit turns.

    Returns:
        span_f1 + aux_weight if n_turns == step_limit and submitted non-zero snippets,
        else 0.0.
    """
    if n_turns > step_limit:
        return 0.0
    f1 = span_f1(predicted_spans, gold_spans, iou_threshold=iou_threshold)
    aux = aux_weight if (n_turns == step_limit and predicted_spans) else 0.0
    return f1 + aux
