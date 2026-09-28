#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Fix Qwen_regenerate.jsonl before computing metrics:

1. Re-label dataset: Qwen uses a generic pre_prompt ("Medical auscultation audio")
   so dataset is determined from patient_id:
     - purely numeric patient_id (e.g. "158", "178") → ICBHI
     - alphanumeric patient_id (e.g. "BP22", "DP13", "EP100") → KAUH

2. Clean generated_clean: Qwen's chat-style model appends chatbot phrases like
   "If you have any other questions, feel free to ask."
   These are truncated using additional markers beyond the ones in clean_generated().

3. Recompute exact_match, norm_match, contains_match from the cleaned text.

Usage:
    python scripts/comparison/fix_qwen_regenerate.py \
        --input  qwen_eval_results/Qwen_regenerate.jsonl \
        --output qwen_eval_results/Qwen_regenerate_fixed.jsonl
"""

import argparse
import json
import re
import sys
import os
from tqdm import tqdm

# ── Helpers (mirrors evaluate_audio_model.py) ─────────────────────────────────

def clean_generated(text: str) -> str:
    """Clean model output: truncate at follow-up prompts and EOS tokens."""
    # Standard markers from evaluate_audio_model.py
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    # Qwen-specific chatbot coda patterns
    for marker in [
        "\nIf you have any other",
        " If you have any other",
        "\nFeel free",
        " feel free to",
        "\nHuman:",
        "\nAssistant:",
    ]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    # EOS tokens
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|im_end|>"]:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]

    return text.strip()


def normalise(text: str) -> str:
    """Lower-case, strip whitespace/punctuation for fuzzy matching."""
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_first_sentence(text: str) -> str:
    """Take only the first sentence (up to period or newline)."""
    for sep in [".", "\n"]:
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def relabel_dataset(patient_id: str) -> str:
    """
    Re-detect dataset from patient_id.

    Qwen evaluation scripts used a generic pre_prompt, so dataset name
    was not embedded. We recover it from patient_id patterns:
      - Purely numeric (e.g. '101', '158')  → ICBHI
      - Alphanumeric with letter prefix     → KAUH  (e.g. 'BP22', 'DP13')
    Other datasets (SPRSound, ZCHSound, CirCor) have their own distinct
    patient_id formats and are already labelled correctly.
    """
    if patient_id.isdigit():
        return "ICBHI"
    return "KAUH"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input JSONL (dirty)")
    parser.add_argument("--output", required=True, help="Output JSONL (fixed)")
    args = parser.parse_args()

    results = [json.loads(l) for l in open(args.input, encoding="utf-8")]
    print(f"Loaded {len(results)} samples from {args.input}")

    # Dataset distribution before
    from collections import Counter
    before = Counter(r["dataset"] for r in results)
    print(f"Dataset labels BEFORE fix: {dict(sorted(before.items()))}")

    fixed = 0
    cleaned = 0

    for r in tqdm(results, desc="Fixing"):
        # ── 1. Re-label dataset ────────────────────────────────────────────────
        ds = r.get("dataset", "unknown")
        if ds == "KAUH":
            pid = str(r.get("patient_id", ""))
            new_ds = relabel_dataset(pid)
            if new_ds != ds:
                r["dataset"] = new_ds
                fixed += 1

        # ── 2. Re-clean generated_clean ───────────────────────────────────────
        old_clean = r.get("generated_clean", "")
        new_clean = clean_generated(old_clean)
        if new_clean != old_clean:
            r["generated_clean"] = new_clean
            cleaned += 1

        # ── 3. Recompute match metrics ─────────────────────────────────────────
        gold = r.get("gold", "")
        pred = r["generated_clean"]

        exact_match = (pred.strip() == gold.strip())
        norm_match = (normalise(pred) == normalise(gold))
        first_sent_match = (normalise(extract_first_sentence(pred)) == normalise(gold))
        contains_match = (normalise(gold) in normalise(pred))

        r["exact_match"]   = exact_match
        r["norm_match"]    = norm_match or first_sent_match
        r["contains_match"] = contains_match

    after = Counter(r["dataset"] for r in results)
    print(f"Dataset labels AFTER  fix: {dict(sorted(after.items()))}")
    print(f"Re-labelled {fixed} samples (KAUH→ICBHI)")
    print(f"Re-cleaned  {cleaned} samples (chatbot coda removed)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved fixed JSONL → {args.output}")


if __name__ == "__main__":
    main()
