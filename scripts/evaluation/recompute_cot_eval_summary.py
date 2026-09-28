#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Recompute CoT eval summary from an existing test_predictions_cot.jsonl file.

Fields used from the JSONL (written by evaluate_audio_cot.py):
  - generated_rationale  (reasoning text extracted from model output)
  - generated_answer     (final answer extracted from model output)
  - gold_cot             (full gold CoT text, used to extract gold rationale)
  - gold_short           (short gold answer)
  - dataset              (CirCor, ICBHI, SPRSound, ZCHSound, KAUH)
  - question             (truncated question text, used for yes/no detection)
  - has_answer_marker    (bool — already computed by evaluate_audio_cot.py)
  - has_rationale        (bool — already computed by evaluate_audio_cot.py)
  - rationale_word_count (int  — already computed by evaluate_audio_cot.py)

Usage:
    python scripts/evaluation/recompute_cot_eval_summary.py \\
        --predictions audio_eval_results_v5/tokenizer_lat64_stage2_cot_proper/test_predictions_cot.jsonl \\
        --output      audio_eval_results_v5/tokenizer_lat64_stage2_cot_proper/cot_eval_summary.json

    # Skip BERTScore (fast, CPU-friendly):
    python scripts/evaluation/recompute_cot_eval_summary.py \\
        --predictions ... --output ... --no_bertscore
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict

import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor
from rouge_score import rouge_scorer as rs_module
from sklearn.metrics import f1_score

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


# ── Helpers (mirror evaluate_audio_cot.py) ────────────────────────────────────

def normalise(text: str) -> str:
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_cot_components(text: str) -> dict:
    """Extract rationale and answer from 'Rationale... Answer: X' format."""
    match = re.search(r'Answer\s*:\s*(.+?)(?:\.|$)', text, re.IGNORECASE | re.DOTALL)
    if match:
        answer_start = match.start()
        rationale = text[:answer_start].strip()
        answer = match.group(1).strip()
        answer = answer.split('.')[0].split('\n')[0].strip()
        return {"rationale": rationale, "answer": answer,
                "has_answer_marker": True, "rationale_word_count": len(rationale.split())}
    return {"rationale": "", "answer": text.strip(),
            "has_answer_marker": False, "rationale_word_count": 0}


def parse_yn(text: str) -> int:
    """Return 1=Yes, 0=No, -1=unparseable."""
    t = re.sub(r"[^\w\s]", " ", text.lower().strip())
    for w in t.split()[:5]:
        if w == "yes":
            return 1
        if w == "no":
            return 0
    return -1


def is_binary_yn_question(question: str) -> bool:
    q = question.lower().strip()
    starters = ("is ", "are ", "was ", "were ", "does ", "do ", "did ",
                "can ", "could ", "has ", "have ", "had ", "will ", "would ", "should ")
    return any(q.startswith(s) for s in starters)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Recompute CoT eval summary from predictions JSONL")
    parser.add_argument("--predictions", required=True, help="Path to test_predictions_cot.jsonl")
    parser.add_argument("--output", required=True, help="Path to write cot_eval_summary.json")
    parser.add_argument("--model_name", default="roberta-large", help="BERTScore model (default: roberta-large)")
    parser.add_argument("--batch_size", type=int, default=32, help="BERTScore batch size")
    parser.add_argument("--no_bertscore", action="store_true", help="Skip BERTScore computation")
    args = parser.parse_args()

    t_start = time.time()

    # ── Load JSONL ─────────────────────────────────────────────────────────────
    print(f"Loading {args.predictions}...")
    with open(args.predictions, encoding="utf-8") as f:
        raw = f.read().strip()
    try:
        records = [json.loads(line) for line in raw.splitlines() if line.strip()]
    except json.JSONDecodeError:
        records = json.loads("[" + raw.replace("}\n{", "},\n{") + "]")
    print(f"  Loaded {len(records)} records")
    total = len(records)

    # ── Per-sample metrics ─────────────────────────────────────────────────────
    rouge_scorer = rs_module.RougeScorer(["rougeL"], use_stemmer=True)

    exact = norm = 0
    answer_extracted = has_rationale_count = total_rat_words = 0

    per_ds = defaultdict(lambda: {
        "total": 0, "exact": 0, "answer_extracted": 0, "has_rationale": 0,
        "rat_words": [], "rougeL": [], "meteor": [], "bert": [],
    })

    rougeL_list, meteor_list = [], []
    all_pred_full, all_gold_full = [], []

    # Closed-ended yes/no
    yn_gold_all, yn_pred_all = [], []
    yn_correct = yn_evaluated = yn_unparseable = yn_skipped = 0
    yn_gold_yes = yn_gold_no = yn_pred_yes = yn_pred_no = 0

    print("Computing string metrics...")
    for i, r in enumerate(records):
        if i % 1000 == 0:
            print(f"  {i}/{total}...")

        ds = r.get("dataset", "unknown")
        pred_answer = r.get("generated_answer", "")
        gold_answer = r.get("gold_short", "")

        # Answer accuracy
        em = normalise(pred_answer) == normalise(gold_answer)
        exact += int(em)
        norm  += int(em)

        # CoT format (use pre-computed fields when available)
        has_ans   = r.get("has_answer_marker", False)
        has_rat   = r.get("has_rationale", False)
        rat_words = r.get("rationale_word_count", 0)
        answer_extracted     += int(has_ans)
        has_rationale_count  += int(has_rat)
        total_rat_words      += rat_words

        # ROUGE-L on answer
        rl = rouge_scorer.score(gold_answer, pred_answer)["rougeL"].fmeasure
        rougeL_list.append(rl)

        # METEOR on answer
        ref_tok = nltk.word_tokenize(gold_answer) if gold_answer else [""]
        hyp_tok = nltk.word_tokenize(pred_answer) if pred_answer else [""]
        met = nltk_meteor([ref_tok], hyp_tok)
        meteor_list.append(met)

        # Per-dataset accumulation
        per_ds[ds]["total"] += 1
        per_ds[ds]["exact"] += int(em)
        per_ds[ds]["answer_extracted"] += int(has_ans)
        per_ds[ds]["has_rationale"] += int(has_rat)
        per_ds[ds]["rat_words"].append(rat_words)
        per_ds[ds]["rougeL"].append(rl)
        per_ds[ds]["meteor"].append(met)

        all_pred_full.append(pred_answer)
        all_gold_full.append(gold_answer)

        # Closed-ended yes/no
        question = r.get("question", "")
        g_yn = parse_yn(gold_answer)
        p_yn = parse_yn(pred_answer)
        if g_yn == -1 or not is_binary_yn_question(question):
            yn_skipped += 1
        elif p_yn == -1:
            yn_unparseable += 1
        else:
            yn_evaluated += 1
            yn_gold_all.append(g_yn)
            yn_pred_all.append(p_yn)
            yn_correct  += int(p_yn == g_yn)
            yn_gold_yes += int(g_yn == 1)
            yn_gold_no  += int(g_yn == 0)
            yn_pred_yes += int(p_yn == 1)
            yn_pred_no  += int(p_yn == 0)

    avg_rougeL = sum(rougeL_list) / total
    avg_meteor = sum(meteor_list) / total

    # ── BERTScore ──────────────────────────────────────────────────────────────
    avg_bert_f1 = None
    bert_f1_list = [None] * total

    if not args.no_bertscore:
        try:
            print(f"\nComputing BERTScore (model={args.model_name}, batch={args.batch_size})...")
            from bert_score import score as bert_score_fn
            max_bert_len = 512
            preds_trunc = [p[:max_bert_len] if p and p.strip() else "[empty]" for p in all_pred_full]
            golds_trunc = [g[:max_bert_len] if g and g.strip() else "[empty]" for g in all_gold_full]
            P, R, F1 = bert_score_fn(
                preds_trunc, golds_trunc,
                lang="en",
                model_type=args.model_name,
                batch_size=args.batch_size,
                verbose=True,
            )
            avg_bert_f1 = F1.mean().item()
            bert_f1_list = F1.tolist()
            print(f"BERTScore F1: {avg_bert_f1:.4f}")
        except Exception as e:
            print(f"  WARNING: BERTScore failed ({e}). Try: pip install -U bert-score")
    else:
        print("Skipping BERTScore (--no_bertscore).")

    for r, bf1 in zip(records, bert_f1_list):
        if bf1 is not None:
            per_ds[r.get("dataset", "unknown")]["bert"].append(bf1)

    # ── Yes/No summary ─────────────────────────────────────────────────────────
    yn_acc = yn_correct / max(yn_evaluated, 1)
    f1_mac = f1_wt = sens = spec = 0.0
    if yn_gold_all:
        f1_mac = f1_score(yn_gold_all, yn_pred_all, average="macro",    zero_division=0)
        f1_wt  = f1_score(yn_gold_all, yn_pred_all, average="weighted", zero_division=0)
        sens   = sum(p == 1 and g == 1 for p, g in zip(yn_pred_all, yn_gold_all)) / max(sum(g == 1 for g in yn_gold_all), 1)
        spec   = sum(p == 0 and g == 0 for p, g in zip(yn_pred_all, yn_gold_all)) / max(sum(g == 0 for g in yn_gold_all), 1)

    elapsed = time.time() - t_start

    # ── Load original metadata ─────────────────────────────────────────────────
    try:
        original = json.load(open(args.output))
        meta_keys = ["checkpoint", "encoder", "llm_id", "cot_data_dir",
                     "max_audio_length", "max_audio_seconds", "max_new_tokens", "elapsed_seconds"]
        metadata = {k: original[k] for k in meta_keys if k in original}
    except (FileNotFoundError, json.JSONDecodeError):
        metadata = {}

    # ── Build summary ──────────────────────────────────────────────────────────
    summary = {
        **metadata,
        "total_samples": total,
        "recomputed": True,
        "answer_accuracy":          exact / total * 100,
        "normalised_accuracy":      norm  / total * 100,
        "answer_extraction_rate":   answer_extracted    / total * 100,
        "rationale_presence_rate":  has_rationale_count / total * 100,
        "avg_rationale_words":      total_rat_words / total,
        "rougeL_f1":                avg_rougeL,
        "meteor":                   avg_meteor,
        "bert_score_f1":            avg_bert_f1,
        "careqa_closed_ended": {
            "accuracy":     yn_acc,
            "f1_macro":     f1_mac,
            "f1_weighted":  f1_wt,
            "sensitivity":  sens,
            "specificity":  spec,
            "correct":      yn_correct,
            "evaluated":    yn_evaluated,
            "skipped":      yn_skipped,
            "pred_unparseable": yn_unparseable,
            "label_counts": {
                "gold_yes": yn_gold_yes, "gold_no": yn_gold_no,
                "pred_yes": yn_pred_yes, "pred_no": yn_pred_no,
            },
        },
        "elapsed_recompute_seconds": elapsed,
        "per_dataset": {},
    }

    for ds in sorted(per_ds.keys()):
        s = per_ds[ds]
        summary["per_dataset"][ds] = {
            "total":                  s["total"],
            "answer_accuracy":        s["exact"]            / s["total"] * 100,
            "answer_extraction_rate": s["answer_extracted"] / s["total"] * 100,
            "rationale_presence_rate":s["has_rationale"]    / s["total"] * 100,
            "avg_rationale_words":    sum(s["rat_words"])   / max(len(s["rat_words"]), 1),
            "rougeL_f1":              sum(s["rougeL"]) / max(len(s["rougeL"]), 1),
            "meteor":                 sum(s["meteor"]) / max(len(s["meteor"]), 1),
            "bert_f1":                (sum(s["bert"]) / len(s["bert"])) if s["bert"] else None,
        }

    # ── Print ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RECOMPUTED CoT RESULTS")
    print("=" * 60)
    print(f"  Total samples        : {total}")
    print(f"  Answer accuracy      : {exact}/{total} = {exact/total*100:.2f}%")
    print(f"  Answer extraction    : {answer_extracted}/{total} = {answer_extracted/total*100:.2f}%")
    print(f"  Rationale presence   : {has_rationale_count}/{total} = {has_rationale_count/total*100:.2f}%")
    print(f"  Avg rationale words  : {total_rat_words/total:.1f}")
    print(f"  ROUGE-L              : {avg_rougeL:.4f}")
    print(f"  METEOR               : {avg_meteor:.4f}")
    print(f"  BERTScore F1         : {f'{avg_bert_f1:.4f}' if avg_bert_f1 is not None else 'not computed'}")
    print(f"  Yes/No accuracy      : {yn_correct}/{yn_evaluated} = {yn_acc*100:.2f}%")
    print(f"  F1 macro             : {f1_mac:.4f}")
    print(f"  Sensitivity          : {sens:.4f}  Specificity: {spec:.4f}")
    print()
    print(f"  {'Dataset':<12} {'n':>5} {'Acc%':>7} {'AnsExt%':>8} {'RougeL':>7} {'BERT':>7}")
    print("  " + "-" * 52)
    for ds, s in sorted(summary["per_dataset"].items()):
        b = f"{s['bert_f1']:.4f}" if s["bert_f1"] is not None else "  null"
        print(f"  {ds:<12} {s['total']:>5} {s['answer_accuracy']:>6.1f}% "
              f"{s['answer_extraction_rate']:>7.1f}% {s['rougeL_f1']:>7.4f} {b:>7}")
    print()

    # ── Write output ───────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved to {args.output}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
