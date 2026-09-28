#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Compute all evaluation metrics from an existing test_predictions.jsonl file.
Skips generation — only computes ROUGE-L, METEOR, BERTScore, and closed-ended metrics.

Usage:
    CUDA_VISIBLE_DEVICES=1 python scripts/evaluation/compute_metrics_from_jsonl.py \
        --jsonl audio_eval_results/mel_lat64/test_predictions.jsonl
"""

import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch

# Add scripts/ to path for label_utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from label_utils import compute_closed_ended_accuracy, parse_yes_no

from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor

for _pkg in ("wordnet", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg == "wordnet" else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


def main():
    parser = argparse.ArgumentParser(description="Compute metrics from existing JSONL predictions")
    parser.add_argument("--jsonl", type=str, required=True, help="Path to test_predictions.jsonl")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load results
    results = []
    with open(args.jsonl) as f:
        for line in f:
            results.append(json.loads(line))
    print(f"Loaded {len(results)} predictions from {args.jsonl}")

    all_preds = [r["generated_clean"] for r in results]
    all_golds = [r["gold"] for r in results]

    # ── ROUGE & METEOR ────────────────────────────────────────────────────────
    print("\nComputing ROUGE & METEOR scores...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = []
    meteor_scores = []
    per_dataset_rougeL = defaultdict(list)
    per_dataset_meteor = defaultdict(list)

    for r in results:
        scores = scorer.score(r["gold"], r["generated_clean"])
        rL = scores["rougeL"].fmeasure
        rougeL_scores.append(rL)
        r["rougeL"] = rL
        per_dataset_rougeL[r["dataset"]].append(rL)

        ref_tokens = nltk.word_tokenize(r["gold"])
        hyp_tokens = nltk.word_tokenize(r["generated_clean"])
        m = nltk_meteor([ref_tokens], hyp_tokens)
        meteor_scores.append(m)
        r["meteor"] = m
        per_dataset_meteor[r["dataset"]].append(m)

    avg_rougeL = sum(rougeL_scores) / max(len(rougeL_scores), 1)
    avg_meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    # ── BERTScore ─────────────────────────────────────────────────────────────
    print("Computing BERTScore (this may take a moment)...")
    max_bert_len = 512
    preds_trunc = [p[:max_bert_len] if p and p.strip() else "[empty]" for p in all_preds]
    golds_trunc = [g[:max_bert_len] if g and g.strip() else "[empty]" for g in all_golds]
    P, R, F1 = bert_score_fn(
        preds_trunc, golds_trunc,
        lang="en",
        model_type="roberta-large",
        device=args.device,
        batch_size=32,
        verbose=False,
    )
    bert_f1_list = F1.tolist()
    bert_p_list = P.tolist()
    bert_r_list = R.tolist()
    avg_bert_f1 = sum(bert_f1_list) / max(len(bert_f1_list), 1)
    avg_bert_p = sum(bert_p_list) / max(len(bert_p_list), 1)
    avg_bert_r = sum(bert_r_list) / max(len(bert_r_list), 1)

    per_dataset_bert = defaultdict(list)
    for r, bf1 in zip(results, bert_f1_list):
        r["bert_f1"] = bf1
        per_dataset_bert[r["dataset"]].append(bf1)

    # ── Closed-ended (Yes/No) ─────────────────────────────────────────────────
    print("Computing closed-ended (Yes/No) accuracy...")
    careqa_closed = compute_closed_ended_accuracy(
        all_golds, all_preds,
        questions=[r["post_prompt"] for r in results],
    )

    for r in results:
        r["careqa_gold_yn"] = parse_yes_no(r["gold"])
        r["careqa_pred_yn"] = parse_yes_no(r["generated_clean"])

    # ── Re-write JSONL with all metrics ───────────────────────────────────────
    with open(args.jsonl, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Aggregate stats ───────────────────────────────────────────────────────
    exact_correct = sum(1 for r in results if r["exact_match"])
    norm_correct = sum(1 for r in results if r["norm_match"])
    contains_correct = sum(1 for r in results if r["contains_match"])
    n = len(results)

    # ── Per-dataset breakdown ─────────────────────────────────────────────────
    per_dataset_exact = defaultdict(list)
    per_dataset_norm = defaultdict(list)
    for r in results:
        per_dataset_exact[r["dataset"]].append(int(r["exact_match"]))
        per_dataset_norm[r["dataset"]].append(int(r["norm_match"]))

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = {
        "total_samples": n,
        "exact_match_accuracy": exact_correct / n,
        "normalised_accuracy": norm_correct / n,
        "contains_match_accuracy": contains_correct / n,
        "avg_rougeL": avg_rougeL,
        "avg_meteor": avg_meteor,
        "avg_bert_f1": avg_bert_f1,
        "avg_bert_precision": avg_bert_p,
        "avg_bert_recall": avg_bert_r,
        "closed_ended": careqa_closed,
        "per_dataset": {},
    }
    for ds in sorted(set(r["dataset"] for r in results)):
        summary["per_dataset"][ds] = {
            "count": len(per_dataset_exact[ds]),
            "exact_acc": np.mean(per_dataset_exact[ds]),
            "norm_acc": np.mean(per_dataset_norm[ds]),
            "rougeL": np.mean(per_dataset_rougeL.get(ds, [0])),
            "meteor": np.mean(per_dataset_meteor.get(ds, [0])),
            "bert_f1": np.mean(per_dataset_bert.get(ds, [0])),
        }

    output_dir = os.path.dirname(args.jsonl)
    summary_path = os.path.join(output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples:          {n}")
    print(f"  Exact match accuracy:   {exact_correct/n:.4f}  ({exact_correct}/{n})")
    print(f"  Normalised accuracy:    {norm_correct/n:.4f}  ({norm_correct}/{n})")
    print(f"  Contains-match acc:     {contains_correct/n:.4f}  ({contains_correct}/{n})")
    print(f"  ROUGE-L (avg):          {avg_rougeL:.4f}")
    print(f"  METEOR (avg):           {avg_meteor:.4f}")
    print(f"  BERTScore F1 (avg):     {avg_bert_f1:.4f}")
    print(f"  BERTScore Prec (avg):   {avg_bert_p:.4f}")
    print(f"  BERTScore Rec (avg):    {avg_bert_r:.4f}")
    print()
    print("  Closed-ended (Yes/No binary):")
    print(f"    Accuracy:             {careqa_closed.get('accuracy', 0):.4f}")
    print(f"    F1 (macro):           {careqa_closed.get('f1_macro', 0):.4f}")
    print(f"    F1 (weighted):        {careqa_closed.get('f1_weighted', 0):.4f}")
    print(f"    Sensitivity:          {careqa_closed.get('sensitivity', 0):.4f}")
    print(f"    Specificity:          {careqa_closed.get('specificity', 0):.4f}")
    print(f"    Evaluated:            {careqa_closed.get('evaluated', 0)}")
    print(f"    Skipped (non-binary): {careqa_closed.get('skipped_not_binary_question', 0)}")
    print()

    print("  Per-dataset breakdown:")
    for ds, info in sorted(summary["per_dataset"].items()):
        print(f"    {ds:20s}  n={info['count']:4d}  exact={info['exact_acc']:.3f}  norm={info['norm_acc']:.3f}  "
              f"ROUGE-L={info['rougeL']:.3f}  METEOR={info['meteor']:.3f}  BERT={info['bert_f1']:.3f}")

    print(f"\n  Summary saved: {summary_path}")
    print(f"  Updated JSONL: {args.jsonl}")


if __name__ == "__main__":
    main()
