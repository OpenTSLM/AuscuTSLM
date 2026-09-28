# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Closed-ended (Yes/No) evaluation utilities for CaReSound.

Pipeline (5 steps):
  1. Filter to binary questions  — is_binary_yn_question (starts with do/does/is/are/…)
  2. Filter to valid gold labels — parse_yes_no on gold (skip if gold isn't yes/no)
  3. Parse model predictions     — parse_yes_no on pred (returns 1/0/-1)
  4. Score                       — unparseable preds count as WRONG
  5. Compute metrics             — accuracy, F1, sensitivity, specificity (sklearn)

Functions:
  extract_question_from_post_prompt(post_prompt) → str
  is_binary_yn_question(question)                → bool   (Step 1)
  parse_yes_no(text)                             → int    (Steps 2-3: 1=yes, 0=no, -1=unparseable)
  compute_closed_ended_accuracy(golds, preds, questions) → dict  (Steps 4-5)
"""

import re

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# EOS / special tokens to strip before parsing
_EOS_TOKENS = ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]


# ─── Closed-ended (Yes/No) Metrics ───────────────────────────────────────────
#
# Binary Yes/No closed-ended accuracy:
#   Parse the model's free-text output for Yes/No and compare to ground truth.
# ---------------------------------------------------------------------------

def extract_question_from_post_prompt(post_prompt: str) -> str:
    """
    Extract the question text from a post_prompt string.

    Post-prompts typically have format "Question: <text>\nAnswer:"
    This function extracts just the <text> part.
    """
    if "Question:" in post_prompt:
        # Extract text between "Question:" and "Answer:"
        parts = post_prompt.split("Question:")
        if len(parts) > 1:
            q_part = parts[1].split("Answer:")[0].strip()
            return q_part
    return post_prompt.strip()


def is_binary_yn_question(question: str) -> bool:
    """
    Check if a question is a binary (Yes/No) question.

    Binary questions are characterized by auxiliary verb inversion at the start.
    Examples:
    - "Does this audio indicate the presence of cough?"     → True
    - "Are there any wheezes present?"                      → True
    - "Were crackles detected?"                             → True
    - "What is the diagnosis?"                              → False
    - "Which valve has the murmur?"                         → False

    This is used to filter closed-ended metrics to only actual Yes/No questions.
    """
    q_lower = question.lower().strip()

    # Common auxiliary verbs at the start of polar questions
    binary_starters = [
        "do ", "does ", "did ",
        "is ", "are ", "was ", "were ",
        "can ", "could ", "will ", "would ", "should ",
        "has ", "have ", "had ",
    ]

    return any(q_lower.startswith(s) for s in binary_starters)


def parse_yes_no(text: str) -> int:
    """
    Parse a Yes/No answer from free-form model output.

    Uses word-boundary matching so that clinical terms that happen to start
    with "no" or "yes" (e.g. "Normal", "notably", "yesterday") are NOT
    misclassified.

    Returns:
        1  if the response indicates Yes
        0  if the response indicates No
       -1  if unparseable (counted as wrong in accuracy)
    """
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    text = text.strip().lower()

    # Word-boundary match at start of string (most reliable).
    # re.match anchors at the start; \b ensures "yes"/"no" are complete words,
    # so "yesterday" won't match "yes" and "normal"/"notably" won't match "no".
    if re.match(r"yes\b", text):
        return 1
    if re.match(r"no\b", text):
        return 0

    # Fallback: look for "yes"/"no" as exact standalone words in a short
    # response (≤ 20 words).  "not" is intentionally excluded: it appears in
    # many non-No contexts ("I cannot tell", "not sure", etc.).
    words = text.split()
    if len(words) <= 20:
        if "yes" in words:
            return 1
        if "no" in words:
            return 0

    return -1


def compute_closed_ended_accuracy(gold_answers, pred_answers, questions=None):
    """
    Closed-ended (Yes/No) accuracy.

    Designed for evaluating binary closed-ended questions where:
    - Questions ask about the presence/absence of a specific concept
      (e.g., "Does this audio indicate the presence of cough?")
    - Gold labels are binary: "Yes" (concept present) or "No" (concept absent),
      derived from dataset annotations

    Two-stage filtering:
    1. Filter by question type: only evaluate binary (Yes/No) questions
       - Questions must start with do/does/is/are/was/were/etc. (polar question)
       - "What...", "Which...", "At which..." questions are skipped
    2. Parse gold answers for Yes/No
       - Only samples where gold answer clearly starts with Yes/No are evaluated
       - Open-ended answers are skipped

    Model predictions are parsed for Yes/No using `parse_yes_no`:
    - Unparseable predictions count as wrong

    Args:
        gold_answers: List of gold answer strings
        pred_answers: List of predicted answer strings
        questions: Optional list of question strings (or post_prompts).
                   If provided, only binary (Yes/No) questions will be evaluated.

    Returns:
        dict with keys:
        - accuracy: float, correct / evaluated
        - f1_macro: float, macro-averaged F1 score
        - sensitivity: float, recall for positive class (Yes)
        - specificity: float, recall for negative class (No)
        - correct: int, number of correct predictions
        - evaluated: int, total Yes/No samples evaluated
        - skipped_not_binary_question: int, samples where question is not binary
        - skipped_not_yes_no: int, samples where gold answer is not Yes/No
        - pred_unparseable: int, predictions that couldn't be parsed as Yes/No
        - label_counts: dict with gold_yes, gold_no, pred_yes, pred_no counts
    """
    yes_no_pred = [parse_yes_no(p) for p in pred_answers]
    yes_no_gold = [parse_yes_no(g) for g in gold_answers]

    # Extract question text if post_prompts were passed
    if questions is not None:
        questions_clean = [extract_question_from_post_prompt(q) for q in questions]
        is_binary = [is_binary_yn_question(q) for q in questions_clean]
    else:
        is_binary = [True] * len(gold_answers)

    # Collect valid samples (binary Q + Yes/No gold answer)
    valid_golds = []
    valid_preds = []
    skipped_not_binary_q = 0
    skipped_not_yes_no = 0
    pred_unparseable = 0

    for g, p, is_bin in zip(yes_no_gold, yes_no_pred, is_binary):
        if not is_bin:
            skipped_not_binary_q += 1
            continue
        if g == -1:
            skipped_not_yes_no += 1
            continue

        # For unparseable predictions, treat as wrong (predict opposite of gold)
        if p == -1:
            pred_unparseable += 1
            p = 1 - g  # Force wrong prediction

        valid_golds.append(g)
        valid_preds.append(p)

    # Compute metrics
    evaluated = len(valid_golds)
    if evaluated == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "correct": 0,
            "evaluated": 0,
            "skipped_not_binary_question": skipped_not_binary_q,
            "skipped_not_yes_no": skipped_not_yes_no,
            "skipped_cannot_derive_label": skipped_not_yes_no,  # Alias
            "pred_unparseable": pred_unparseable,
            "label_counts": {"gold_yes": 0, "gold_no": 0, "pred_yes": 0, "pred_no": 0},
        }

    # Convert to numpy arrays for sklearn
    y_true = np.array(valid_golds)
    y_pred = np.array(valid_preds)

    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity (recall for positive class): TP / (TP + FN)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Specificity (recall for negative class): TN / (TN + FP)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Count labels
    label_counts = {
        "gold_yes": int(np.sum(y_true == 1)),
        "gold_no": int(np.sum(y_true == 0)),
        "pred_yes": int(np.sum(y_pred == 1)),
        "pred_no": int(np.sum(y_pred == 0)),
    }

    correct = int(np.sum(y_true == y_pred))

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "correct": correct,
        "evaluated": evaluated,
        "skipped_not_binary_question": skipped_not_binary_q,
        "skipped_not_yes_no": skipped_not_yes_no,
        "skipped_cannot_derive_label": skipped_not_yes_no,  # Alias for backward compatibility
        "pred_unparseable": pred_unparseable,
        "label_counts": label_counts,
    }
