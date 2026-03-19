"""
Reward functions for audio QA tasks.

For multiple-choice questions (ground-truth is a single letter A-D):
    reward = 1.0 if the predicted letter matches, else 0.0

For open-ended questions:
    reward = token-level F1 between predicted and ground-truth text
"""
from __future__ import annotations

import re


def extract_mcq_answer(text: str) -> str | None:
    """Extract a single letter answer (A-D) from model output.

    Tries JSON format first: {"answer": "A"}, then bare letter.
    """
    # JSON format: {"answer": "X"}
    m = re.search(r'\{[^}]*"answer"\s*:\s*"([A-Da-d])"', text)
    if m:
        return m.group(1).upper()
    # Bare letter at word boundary
    m = re.search(r'\b([A-Da-d])\b', text.strip())
    if m:
        return m.group(1).upper()
    return None


def token_f1(pred: str, gold: str) -> float:
    """Token-level F1 score (bag-of-words overlap)."""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    if not pred_tokens:
        return 0.0
    common = pred_tokens & gold_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def audio_qa_reward(predicted: str, gold: str) -> float:
    """Compute reward for a single (predicted, gold) answer pair.

    Multiple-choice questions are identified when gold is a single letter A-D.
    All other questions use token-level F1.
    """
    gold = gold.strip()
    predicted = predicted.strip()

    if len(gold) == 1 and gold.upper() in "ABCD":
        pred_letter = extract_mcq_answer(predicted)
        return 1.0 if pred_letter == gold.upper() else 0.0

    return token_f1(predicted, gold)
