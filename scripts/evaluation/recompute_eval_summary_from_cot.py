#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Recompute an eval_summary.json-style report from CoT predictions.

Expected JSONL fields per row (defaults, configurable via CLI):
  - generated_answer  (prediction; can also read generated_clean/generated_raw)
  - gold_short        (reference short answer)
  - dataset           (e.g. ICBHI, CirCor, SPRSound, ZCHSound, KAUH)
  - question          (question text, often contains "Question:" and "Rationale:")

This mirrors the non-CoT summary format from scripts/evaluation/evaluate_audio_model.py,
but uses CoT-specific columns:
  pred = generated_answer
  gold = gold_short

Usage:
  python3 scripts/evaluation/recompute_eval_summary_from_cot.py \
      --predictions audio_eval_results_v5/tokenizer_lat64_stage2_cot_proper/test_predictions_cot.jsonl \
      --output      audio_eval_results_v5/tokenizer_lat64_stage2_cot_proper/eval_summary.json

  # Parse "Answer: ..." out of CoT text before scoring
  python3 scripts/evaluation/recompute_eval_summary_from_cot.py \
      --predictions ... --output ... --extract_answer

  # Fast mode (skip BERTScore)
  python3 scripts/evaluation/recompute_eval_summary_from_cot.py --predictions ... --output ... --no_bertscore
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict

import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer as rs_module

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from label_utils import compute_closed_ended_accuracy, is_binary_yn_question, parse_yes_no


_EOS_TOKENS = ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]


def normalise(text: str) -> str:
    """Lower-case and strip punctuation/extra whitespace for fuzzy matching."""
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_generated(text: str) -> str:
    """
    Truncate spill-over generation so scoring matches evaluate_audio_model.py.
    """
    text = text or ""
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    for tok in _EOS_TOKENS:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def clean_gold(text: str) -> str:
    """Remove EOS markers from reference answer."""
    text = text or ""
    for tok in _EOS_TOKENS:
        text = text.replace(tok, "")
    return text.strip()


def extract_first_sentence(text: str) -> str:
    """Return first sentence-like chunk to match short gold answers."""
    for sep in [".", "\n"]:
        idx = text.find(sep)
        if idx != -1:
            return text[:idx].strip()
    return text.strip()


def extract_answer_from_cot_output(text: str) -> str:
    """
    Extract final answer from CoT text.

    Preferred format is "... Answer: <final answer>".
    If no marker exists, fallback to first sentence of cleaned output.
    """
    cleaned = clean_generated(text)
    if not cleaned:
        return ""

    m = re.search(r"answer\s*:\s*(.+)", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if m:
        answer = m.group(1).strip()
        # Keep only the first line after "Answer:" to avoid trailing rationale.
        answer = answer.split("\n", 1)[0].strip()
        return clean_generated(answer)

    return extract_first_sentence(cleaned)


def extract_cot_question(question: str) -> str:
    """Extract plain question text from CoT prompt fragments."""
    q = (question or "").strip()
    if "Question:" in q:
        q = q.split("Question:", 1)[1]
    for marker in ("Rationale:", "Answer:"):
        if marker in q:
            q = q.split(marker, 1)[0]
    return q.strip()


def load_jsonl(path: str):
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()

    if not raw:
        return []

    try:
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        # Fallback for concatenated pretty-printed objects
        return json.loads("[" + raw.replace("}\n{", "},\n{") + "]")


def _tokenize_for_meteor(text: str):
    """Tokenize with NLTK when available; fallback to whitespace split."""
    text = text or ""
    try:
        return nltk.word_tokenize(text)
    except LookupError:
        return text.split()


def _load_existing_metadata(path: str):
    """Keep useful metadata fields when recomputing an existing summary."""
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    except json.JSONDecodeError:
        return {}

    keep = [
        "checkpoint",
        "encoder",
        "llm_id",
        "model_id",
        "model_type",
        "cot_data_dir",
        "max_audio_length",
        "max_audio_seconds",
        "max_new_tokens",
        "fine_tuned",
        "num_shots",
    ]
    return {k: existing[k] for k in keep if k in existing}


def main():
    parser = argparse.ArgumentParser(description="Recompute eval_summary.json from CoT predictions JSONL")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions_cot.jsonl")
    parser.add_argument(
        "--output",
        default=None,
        help="Output eval_summary.json path (default: sibling eval_summary.json next to --predictions)",
    )
    parser.add_argument("--model_name", default="roberta-large", help="BERTScore model (default: roberta-large)")
    parser.add_argument("--batch_size", type=int, default=64, help="BERTScore batch size")
    parser.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore computation")
    parser.add_argument(
        "--pred_field",
        default="generated_answer",
        help="Primary prediction field in JSONL (default: generated_answer)",
    )
    parser.add_argument(
        "--pred_fallback_fields",
        default="generated_clean,generated_raw",
        help="Comma-separated fallback prediction fields (default: generated_clean,generated_raw)",
    )
    parser.add_argument(
        "--gold_field",
        default="gold_short",
        help="Gold answer field in JSONL (default: gold_short)",
    )
    parser.add_argument(
        "--question_field",
        default="question",
        help="Question field in JSONL used for Yes/No filtering (default: question)",
    )
    parser.add_argument(
        "--extract_answer",
        action="store_true",
        help="Extract final answer from CoT output (parse 'Answer:' else first sentence)",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(os.path.dirname(os.path.abspath(args.predictions)), "eval_summary.json")

    t_start = time.time()

    print(f"Loading {args.predictions}...")
    records = load_jsonl(args.predictions)
    if not records:
        raise ValueError(f"No records found in {args.predictions}")

    total = len(records)
    print(f"  Loaded {total} records")
    pred_fields = [args.pred_field] + [
        f.strip() for f in args.pred_fallback_fields.split(",") if f.strip() and f.strip() != args.pred_field
    ]
    print(f"  Prediction fields: {pred_fields} (extract_answer={args.extract_answer})")

    rouge_scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)

    exact = norm = contains = 0
    rougeL_scores = []
    meteor_scores = []

    all_preds = []
    all_golds = []
    all_questions = []
    all_datasets = []

    per_ds = defaultdict(
        lambda: {
            "total": 0,
            "exact": 0,
            "norm": 0,
            "contains": 0,
            "rougeL": [],
            "meteor": [],
            "bert": [],
        }
    )

    missing_pred = 0
    used_fallback = 0

    for i, row in enumerate(records):
        if i % 1000 == 0:
            print(f"  {i}/{total}...")

        pred_raw = ""
        pred_field_used = None
        for pf in pred_fields:
            value = str(row.get(pf, "") or "").strip()
            if value:
                pred_raw = value
                pred_field_used = pf
                break
        if pred_field_used is None:
            missing_pred += 1
        elif pred_field_used != args.pred_field:
            used_fallback += 1

        pred = extract_answer_from_cot_output(pred_raw) if args.extract_answer else clean_generated(pred_raw)

        gold = clean_gold(str(row.get(args.gold_field, "") or ""))
        if not gold and args.gold_field != "gold":
            gold = clean_gold(str(row.get("gold", "") or ""))
        ds = str(row.get("dataset", "unknown") or "unknown")
        q_clean = extract_cot_question(
            str(row.get(args.question_field, "") or row.get("post_prompt", "") or "")
        )

        em = pred.strip() == gold.strip()
        nm = normalise(pred) == normalise(gold) or normalise(extract_first_sentence(pred)) == normalise(gold)
        cm = normalise(gold) in normalise(pred)

        rl = rouge_scorer.score(gold, pred)["rougeL"].fmeasure

        ref_tokens = _tokenize_for_meteor(gold) if gold else [""]
        hyp_tokens = _tokenize_for_meteor(pred) if pred else [""]
        met = nltk_meteor([ref_tokens], hyp_tokens)

        exact += int(em)
        norm += int(nm)
        contains += int(cm)
        rougeL_scores.append(rl)
        meteor_scores.append(met)

        all_preds.append(pred)
        all_golds.append(gold)
        all_questions.append(q_clean)
        all_datasets.append(ds)

        stats = per_ds[ds]
        stats["total"] += 1
        stats["exact"] += int(em)
        stats["norm"] += int(nm)
        stats["contains"] += int(cm)
        stats["rougeL"].append(rl)
        stats["meteor"].append(met)

    if missing_pred:
        print(f"  WARNING: {missing_pred} rows had no prediction text in {pred_fields}.")
    if used_fallback:
        print(f"  INFO: used fallback prediction field on {used_fallback} rows.")

    avg_rougeL = sum(rougeL_scores) / total
    avg_meteor = sum(meteor_scores) / total

    avg_bert_p = avg_bert_r = avg_bert_f1 = None
    if not args.no_bertscore:
        try:
            print(f"Computing BERTScore (model={args.model_name}, batch={args.batch_size})...")
            from bert_score import score as bert_score_fn

            max_bert_len = 512
            preds_trunc = [p[:max_bert_len] if p and p.strip() else "[empty]" for p in all_preds]
            golds_trunc = [g[:max_bert_len] if g and g.strip() else "[empty]" for g in all_golds]

            P, R, F1 = bert_score_fn(
                preds_trunc,
                golds_trunc,
                lang="en",
                model_type=args.model_name,
                batch_size=args.batch_size,
                verbose=True,
            )

            bert_p_list = P.tolist()
            bert_r_list = R.tolist()
            bert_f1_list = F1.tolist()
            avg_bert_p = sum(bert_p_list) / len(bert_p_list)
            avg_bert_r = sum(bert_r_list) / len(bert_r_list)
            avg_bert_f1 = sum(bert_f1_list) / len(bert_f1_list)

            for ds, bf1 in zip(all_datasets, bert_f1_list):
                per_ds[ds]["bert"].append(bf1)
        except Exception as exc:
            print(f"  WARNING: BERTScore failed ({exc}).")
            print("  Continuing with bert_score_* = null.")
    else:
        print("Skipping BERTScore (--no_bertscore).")

    print("Computing closed-ended (Yes/No) metrics...")
    closed_ended = compute_closed_ended_accuracy(all_golds, all_preds, questions=all_questions)

    per_dataset_yn = defaultdict(
        lambda: {
            "total_binary_qs": 0,
            "correct": 0,
            "parseable": 0,
            "unparseable": 0,
            "gold_yes": 0,
            "gold_no": 0,
        }
    )

    for ds, q, gold, pred in zip(all_datasets, all_questions, all_golds, all_preds):
        if not is_binary_yn_question(q):
            continue

        gold_yn = parse_yes_no(gold)
        if gold_yn == -1:
            continue

        pred_yn = parse_yes_no(pred)
        entry = per_dataset_yn[ds]
        entry["total_binary_qs"] += 1
        if gold_yn == 1:
            entry["gold_yes"] += 1
        else:
            entry["gold_no"] += 1

        if pred_yn == -1:
            entry["unparseable"] += 1
        else:
            entry["parseable"] += 1
            if pred_yn == gold_yn:
                entry["correct"] += 1

    elapsed = time.time() - t_start

    metadata = _load_existing_metadata(output_path)
    summary = {
        **metadata,
        "source_predictions": args.predictions,
        "total_samples": total,
        "recomputed": True,
        "open_ended_accuracy": exact / total * 100,
        "normalised_accuracy": norm / total * 100,
        "contains_match_accuracy": contains / total * 100,
        "bert_score_f1": avg_bert_f1,
        "bert_score_precision": avg_bert_p,
        "bert_score_recall": avg_bert_r,
        "rougeL_f1": avg_rougeL,
        "meteor": avg_meteor,
        "closed_ended_yes_no": closed_ended,
        "elapsed_recompute_seconds": elapsed,
        "per_dataset": {},
    }

    for ds in sorted(per_ds.keys()):
        s = per_ds[ds]
        n = s["total"]
        yn = None
        if per_dataset_yn[ds]["total_binary_qs"] > 0:
            d = per_dataset_yn[ds]
            yn = {
                "total_binary_qs": d["total_binary_qs"],
                "correct": d["correct"],
                "parseable": d["parseable"],
                "unparseable": d["unparseable"],
                "accuracy_all": d["correct"] / max(d["total_binary_qs"], 1) * 100,
                "accuracy_parseable": d["correct"] / max(d["parseable"], 1) * 100,
                "parseable_rate": d["parseable"] / max(d["total_binary_qs"], 1) * 100,
                "gold_yes": d["gold_yes"],
                "gold_no": d["gold_no"],
            }

        summary["per_dataset"][ds] = {
            "total": n,
            "open_ended_acc": s["exact"] / n * 100,
            "norm_acc": s["norm"] / n * 100,
            "contains_match_acc": s["contains"] / n * 100,
            "rougeL_f1": sum(s["rougeL"]) / max(len(s["rougeL"]), 1),
            "meteor": sum(s["meteor"]) / max(len(s["meteor"]), 1),
            "bert_f1": (sum(s["bert"]) / len(s["bert"])) if s["bert"] else None,
            "yes_no": yn,
        }

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RECOMPUTED EVAL SUMMARY FROM COT PREDICTIONS")
    print("=" * 60)
    print(f"  Total samples    : {total}")
    print(f"  Open-ended       : {exact}/{total} = {exact/total*100:.2f}%")
    print(f"  Normalised       : {norm}/{total} = {norm/total*100:.2f}%")
    print(f"  Contains-match   : {contains}/{total} = {contains/total*100:.2f}%")
    print(f"  ROUGE-L          : {avg_rougeL:.4f}")
    print(f"  METEOR           : {avg_meteor:.4f}")
    if avg_bert_f1 is not None:
        print(f"  BERTScore F1     : {avg_bert_f1:.4f} (P={avg_bert_p:.4f}, R={avg_bert_r:.4f})")
    else:
        print("  BERTScore        : null")

    ce = closed_ended
    print(f"  Yes/No accuracy  : {ce['correct']}/{ce['evaluated']} = {ce['accuracy']*100:.2f}%")
    print(f"  Saved summary    : {output_path}")
    print(f"  Elapsed          : {elapsed:.1f}s")


if __name__ == "__main__":
    main()
