#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Fix Qwen JSONL by applying clean_generated to generated_clean,
then recomputing exact_match / norm_match / contains_match.

Run this BEFORE compute_metrics_from_jsonl.py.

Usage:
    python scripts/comparison/fix_qwen_jsonl.py --jsonl qwen_eval_results/qwen_scored_predictions.jsonl
"""

import os
import re
import json
import argparse
from tqdm import tqdm


def clean_generated(text: str) -> str:
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>"]:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def normalise(text: str) -> str:
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_first_sentence(text: str) -> str:
    for sep in [".", "\n"]:
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx]
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()

    results = [json.loads(l) for l in open(args.jsonl)]
    print(f"Loaded {len(results)} samples from {args.jsonl}")

    changed = 0
    for r in tqdm(results, desc="Cleaning", unit="sample"):
        old_clean = r["generated_clean"]
        new_clean = clean_generated(old_clean)

        if new_clean != old_clean:
            r["generated_clean"] = new_clean
            changed += 1

        gold = r["gold"]
        pred = new_clean

        exact_match = pred.strip() == gold.strip()
        norm_match = (normalise(pred) == normalise(gold)) or \
                     (normalise(extract_first_sentence(pred)) == normalise(gold))
        contains_match = normalise(gold) in normalise(pred)

        r["exact_match"] = exact_match
        r["norm_match"] = norm_match
        r["contains_match"] = contains_match

    print(f"\nCleaned generated_clean in {changed} / {len(results)} samples")
    exact_after = sum(1 for r in results if r["exact_match"])
    norm_after  = sum(1 for r in results if r["norm_match"])
    print(f"exact_match after cleaning: {exact_after} / {len(results)} ({100*exact_after/len(results):.2f}%)")
    print(f"norm_match  after cleaning: {norm_after}  / {len(results)} ({100*norm_after/len(results):.2f}%)")

    print(f"\nSaving to {args.jsonl} ...")
    with open(args.jsonl, "w", encoding="utf-8") as f:
        for r in tqdm(results, desc="Writing", unit="sample"):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Done.")
    print("\nNow run:")
    print(f"  python scripts/evaluation/compute_metrics_from_jsonl.py --jsonl {args.jsonl}")


if __name__ == "__main__":
    main()
