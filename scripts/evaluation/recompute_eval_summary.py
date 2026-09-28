#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Recompute eval summary from a raw predictions JSONL file.

Each line in the JSONL must have at least:
  - generated_clean  (model output after cleaning)
  - gold             (reference answer)
  - dataset          (e.g. ICBHI, CirCor, SPRSound, ZCHSound, KAUH)
  - careqa_gold_yn   (1=Yes, 0=No, -1=not a yes/no question)

All per-sample metrics (exact_match, rougeL, bert_f1, etc.) are
RECOMPUTED from scratch — stored values are ignored.

Usage:
    python scripts/evaluation/recompute_eval_summary.py \\
        --predictions path/to/test_predictions.jsonl \\
        --output      path/to/eval_summary.json \\
        [--model_name  "roberta-large"]   # BERTScore model (default: roberta-large)
        [--batch_size  64]                # BERTScore batch size
        [--no_bertscore]                  # Skip BERTScore (fast, CPU-friendly)

Example:
    python scripts/evaluation/recompute_eval_summary.py \\
        --predictions comparison/careaqa_eval_results/careqa_test_predictions.jsonl \\
        --output      comparison/careaqa_eval_results/eval_summary.json
"""

import argparse
import json
import re
import time
from collections import defaultdict

import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer as rs_module
from sklearn.metrics import f1_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


# ── Text utilities ─────────────────────────────────────────────────────────────

def normalise(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_first_sentence(s: str) -> str:
    m = re.match(r"([^.!?]+[.!?])", s.strip())
    return m.group(1).strip() if m else s.strip()


def parse_yn(text: str) -> int:
    """Return 1 (Yes), 0 (No), or -1 (unparseable)."""
    t = re.sub(r"[^\w\s]", " ", text.lower().strip())
    for w in t.split()[:5]:
        if w == "yes":
            return 1
        if w == "no":
            return 0
    return -1


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recompute eval summary from predictions JSONL")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSONL file")
    parser.add_argument("--output", required=True, help="Path to write updated eval_summary.json")
    parser.add_argument("--model_name", default="roberta-large",
                        help="HuggingFace model for BERTScore (default: roberta-large)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for BERTScore (default: 64)")
    parser.add_argument("--no_bertscore", action="store_true",
                        help="Skip BERTScore computation (faster, sets bert_f1=null)")
    args = parser.parse_args()

    t_start = time.time()

    # ── Load predictions ───────────────────────────────────────────────────────
    print(f"Loading predictions from {args.predictions}...")
    records = []
    with open(args.predictions, encoding="utf-8") as f:
        raw = f.read().strip()
    # Support both jsonl (one object per line) and pretty-printed concatenated objects
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        # Pretty-printed: join objects separated by "}\n{"
        records = json.loads("[" + raw.replace("}\n{", "},\n{") + "]")
    print(f"  Loaded {len(records)} records")

    preds = [r["generated_clean"] for r in records]
    golds = [r["gold"] for r in records]
    total = len(records)

    # ── String metrics ─────────────────────────────────────────────────────────
    print("Computing string metrics (exact / normalised / contains)...")
    rouge_scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)

    exact = norm = contains = 0
    rougeL_scores = []
    meteor_scores_list = []

    # Per-dataset accumulators
    per_ds = defaultdict(lambda: {
        "total": 0, "exact": 0, "norm": 0, "contains": 0,
        "rougeL": 0.0, "meteor": 0.0, "bert": 0.0,
        "yn_total": 0, "yn_correct": 0, "yn_parseable": 0,
        "yn_unparseable": 0, "gold_yes": 0, "gold_no": 0,
        "pred_yes": 0, "pred_no": 0,
    })

    # Yes/No accumulators
    yn_gold_all, yn_pred_all = [], []
    yn_correct = yn_evaluated = yn_unparseable = yn_skipped = 0
    yn_pred_yes = yn_pred_no = yn_gold_yes = yn_gold_no = 0

    for i, (r, pred, gold) in enumerate(zip(records, preds, golds)):
        if i % 1000 == 0:
            print(f"  {i}/{total}...")

        ds = r.get("dataset", "unknown")

        # Exact / normalised / contains
        em = pred.strip() == gold.strip()
        nm = (normalise(pred) == normalise(gold) or
              normalise(extract_first_sentence(pred)) == normalise(gold))
        cm = normalise(gold) in normalise(pred)

        exact += int(em)
        norm += int(nm)
        contains += int(cm)

        # ROUGE-L
        rl = rouge_scorer.score(gold, pred)["rougeL"].fmeasure
        rougeL_scores.append(rl)

        # METEOR
        met = meteor_score([gold.split()], pred.split())
        meteor_scores_list.append(met)

        # Per-dataset string metrics
        per_ds[ds]["total"] += 1
        per_ds[ds]["exact"] += int(em)
        per_ds[ds]["norm"] += int(nm)
        per_ds[ds]["contains"] += int(cm)
        per_ds[ds]["rougeL"] += rl
        per_ds[ds]["meteor"] += met

        # Yes/No — recompute from generated text
        g_yn = r.get("careqa_gold_yn", -1)
        p_yn = parse_yn(pred)

        if g_yn not in (0, 1):
            yn_skipped += 1
        else:
            if p_yn == -1:
                yn_unparseable += 1
                per_ds[ds]["yn_unparseable"] += 1
            else:
                yn_evaluated += 1
                yn_gold_all.append(g_yn)
                yn_pred_all.append(p_yn)
                yn_correct += int(p_yn == g_yn)
                yn_pred_yes += int(p_yn == 1)
                yn_pred_no += int(p_yn == 0)
                yn_gold_yes += int(g_yn == 1)
                yn_gold_no += int(g_yn == 0)
                per_ds[ds]["yn_total"] += 1
                per_ds[ds]["yn_correct"] += int(p_yn == g_yn)
                per_ds[ds]["yn_parseable"] += 1
                per_ds[ds]["gold_yes"] += int(g_yn == 1)
                per_ds[ds]["gold_no"] += int(g_yn == 0)
                per_ds[ds]["pred_yes"] += int(p_yn == 1)
                per_ds[ds]["pred_no"] += int(p_yn == 0)

    avg_rougeL = sum(rougeL_scores) / total
    avg_meteor = sum(meteor_scores_list) / total

    # ── BERTScore ──────────────────────────────────────────────────────────────
    avg_bert_p = avg_bert_r = avg_bert_f1 = None
    per_ds_bert = {}

    if not args.no_bertscore:
        print(f"\nComputing BERTScore (model={args.model_name}, batch={args.batch_size})...")
        try:
            from bert_score import score as bert_score_fn
            P, R, F1 = bert_score_fn(
                preds, golds,
                lang="en",
                model_type=args.model_name,
                batch_size=args.batch_size,
                verbose=True,
            )
            avg_bert_p  = P.mean().item()
            avg_bert_r  = R.mean().item()
            avg_bert_f1 = F1.mean().item()

            # Per-dataset BERTScore
            bert_f1_list = F1.tolist()
            ds_list = [r.get("dataset", "unknown") for r in records]
            ds_bert = defaultdict(list)
            for ds, bf in zip(ds_list, bert_f1_list):
                ds_bert[ds].append(bf)
            per_ds_bert = {ds: sum(v) / len(v) for ds, v in ds_bert.items()}

            print(f"BERTScore: P={avg_bert_p:.4f}  R={avg_bert_r:.4f}  F1={avg_bert_f1:.4f}")
        except ImportError:
            print("  WARNING: bert_score not installed. Run: pip install bert-score")
            print("  Skipping BERTScore.")
    else:
        print("\nSkipping BERTScore (--no_bertscore).")

    # ── Yes/No summary ─────────────────────────────────────────────────────────
    yn_acc = yn_correct / max(yn_evaluated, 1)
    f1_mac = f1_wt = sens = spec = 0.0
    if yn_gold_all:
        f1_mac = f1_score(yn_gold_all, yn_pred_all, average="macro",    zero_division=0)
        f1_wt  = f1_score(yn_gold_all, yn_pred_all, average="weighted", zero_division=0)
        sens   = sum(p == 1 and g == 1 for p, g in zip(yn_pred_all, yn_gold_all)) / max(sum(g == 1 for g in yn_gold_all), 1)
        spec   = sum(p == 0 and g == 0 for p, g in zip(yn_pred_all, yn_gold_all)) / max(sum(g == 0 for g in yn_gold_all), 1)

    # ── Build output ───────────────────────────────────────────────────────────
    elapsed = time.time() - t_start

    # Preserve original metadata fields if present
    try:
        original = json.load(open(args.output))
        meta_keys = ["checkpoint", "encoder", "llm_id", "model_id", "model_type",
                     "max_audio_length", "max_audio_seconds", "max_new_tokens",
                     "elapsed_seconds", "fine_tuned", "num_shots"]
        metadata = {k: original[k] for k in meta_keys if k in original}
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    summary = {
        **metadata,
        "total_samples": total,
        "recomputed": True,
        "open_ended_accuracy":    exact   / total * 100,
        "normalised_accuracy":    norm    / total * 100,
        "contains_match_accuracy":contains / total * 100,
        "rougeL_f1":              avg_rougeL,
        "meteor":                 avg_meteor,
        "bert_score_f1":          avg_bert_f1,
        "bert_score_precision":   avg_bert_p,
        "bert_score_recall":      avg_bert_r,
        "closed_ended_yes_no": {
            "accuracy":                  yn_acc,
            "f1_macro":                  f1_mac,
            "f1_weighted":               f1_wt,
            "sensitivity":               sens,
            "specificity":               spec,
            "correct":                   yn_correct,
            "evaluated":                 yn_evaluated,
            "skipped_not_binary_question": yn_skipped,
            "pred_unparseable":          yn_unparseable,
            "label_counts": {
                "gold_yes": yn_gold_yes, "gold_no": yn_gold_no,
                "pred_yes": yn_pred_yes, "pred_no": yn_pred_no,
            },
        },
        "elapsed_recompute_seconds": elapsed,
        "per_dataset": {},
    }

    for ds in sorted(per_ds):
        s = per_ds[ds]
        n = s["total"]
        yn = None
        if s["yn_total"] > 0:
            yn = {
                "total_binary_qs": s["yn_total"],
                "correct":         s["yn_correct"],
                "parseable":       s["yn_parseable"],
                "unparseable":     s["yn_unparseable"],
                "accuracy_all":    s["yn_correct"] / s["yn_total"] * 100,
                "gold_yes":        s["gold_yes"],
                "gold_no":         s["gold_no"],
            }
        summary["per_dataset"][ds] = {
            "total":          n,
            "open_ended_acc": s["exact"]    / n * 100,
            "norm_acc":       s["norm"]     / n * 100,
            "contains_acc":   s["contains"] / n * 100,
            "rougeL_f1":      s["rougeL"]   / n,
            "meteor":         s["meteor"]   / n,
            "bert_f1":        per_ds_bert.get(ds),
            "yes_no":         yn,
        }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMPUTED RESULTS")
    print("=" * 60)
    print(f"  Total samples    : {total}")
    print(f"  Exact match      : {exact}/{total} = {exact/total*100:.2f}%")
    print(f"  Normalised       : {norm}/{total} = {norm/total*100:.2f}%")
    print(f"  Contains         : {contains}/{total} = {contains/total*100:.2f}%")
    print(f"  ROUGE-L          : {avg_rougeL:.4f}")
    print(f"  METEOR           : {avg_meteor:.4f}")
    if avg_bert_f1 is not None:
        print(f"  BERTScore F1     : {avg_bert_f1:.4f}  (P={avg_bert_p:.4f}, R={avg_bert_r:.4f})")
    else:
        print(f"  BERTScore        : not computed (use without --no_bertscore)")
    print(f"  Yes/No accuracy  : {yn_correct}/{yn_evaluated} = {yn_acc*100:.2f}%")
    print(f"  F1 macro         : {f1_mac:.4f}")
    print(f"  Sensitivity      : {sens:.4f}  Specificity: {spec:.4f}")
    print(f"  Unparseable Y/N  : {yn_unparseable}")
    print()
    print(f"  {'Dataset':<12} {'n':>5} {'Exact':>7} {'Norm':>7} {'RougeL':>7} {'METEOR':>7} {'BERT':>7}")
    print("  " + "-" * 58)
    for ds, s in sorted(summary["per_dataset"].items()):
        bert_str = f"{s['bert_f1']:.4f}" if s["bert_f1"] is not None else "  null"
        print(f"  {ds:<12} {s['total']:>5} {s['open_ended_acc']:>6.1f}% {s['norm_acc']:>6.1f}% "
              f"{s['rougeL_f1']:>7.4f} {s['meteor']:>7.4f} {bert_str:>7}")
    print()
    print(f"  Elapsed: {elapsed:.1f}s")

    # ── Write output ───────────────────────────────────────────────────────────
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
