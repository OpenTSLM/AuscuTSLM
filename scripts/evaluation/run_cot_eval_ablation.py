#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Run CoT evaluation ablations for:
  1) generation length (--max_new_tokens)
  2) answer extraction policy at metric time

This script orchestrates:
  - scripts/evaluation/evaluate_audio_cot.py
  - scripts/evaluation/recompute_eval_summary_from_cot.py

Example:
  python3 scripts/evaluation/run_cot_eval_ablation.py \
    --checkpoint audio_checkpoints_v5/tokenizer_lat64/tokenizer_stage2_audio_cot_proper/best_model.pt \
    --encoder tokenizer \
    --llm_id meta-llama/Llama-3.2-1B \
    --audio_dir data/caresound_audio/audio_merged \
    --cot_data_dir data/caresound_cot \
    --output_root audio_eval_results_v5/ablation_stage2_cot_proper \
    --max_new_tokens_list 128,192,256,384,512
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


def _run(cmd: List[str], dry_run: bool = False) -> None:
    print("\n$ " + " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_summary(path: Path) -> Dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _fmt(x, pct: bool = False) -> str:
    if x is None:
        return "null"
    if pct:
        return f"{x:.2f}"
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _print_table(rows: List[Dict]) -> None:
    if not rows:
        print("\nNo rows to print.")
        return

    headers = [
        "max_new_tokens",
        "extraction",
        "open_acc%",
        "norm_acc%",
        "contains%",
        "rougeL",
        "meteor",
        "bert_f1",
        "yn_acc%",
        "f1_macro",
        "f1_weighted",
        "sensitivity",
        "specificity",
        "pred_unparseable",
    ]
    lines = [headers]
    for r in rows:
        lines.append(
            [
                str(r["max_new_tokens"]),
                r["extraction"],
                _fmt(r["open_ended_accuracy"], pct=True),
                _fmt(r["normalised_accuracy"], pct=True),
                _fmt(r["contains_match_accuracy"], pct=True),
                _fmt(r["rougeL_f1"]),
                _fmt(r["meteor"]),
                _fmt(r["bert_score_f1"]),
                _fmt(r["yn_accuracy"] * 100 if r["yn_accuracy"] is not None else None, pct=True),
                _fmt(r["yn_f1_macro"]),
                _fmt(r["yn_f1_weighted"]),
                _fmt(r["yn_sensitivity"]),
                _fmt(r["yn_specificity"]),
                _fmt(r["yn_pred_unparseable"]),
            ]
        )

    widths = [max(len(row[i]) for row in lines) for i in range(len(headers))]
    print()
    print(" | ".join(h.ljust(widths[i]) for i, h in enumerate(lines[0])))
    print("-+-".join("-" * w for w in widths))
    for row in lines[1:]:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CoT eval ablation (max_new_tokens x extraction policy).")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint passed to evaluate_audio_cot.py")
    parser.add_argument("--encoder", default="tokenizer", choices=["tokenizer", "mel", "wav2vec2", "whisper", "clap"])
    parser.add_argument("--llm_id", default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--audio_dir", default="./data/caresound_audio/audio_merged")
    parser.add_argument("--cot_data_dir", default="data/caresound_cot")
    parser.add_argument("--device", default=None, help="cuda/mps/cpu; default lets evaluate script decide")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional quick subset")
    parser.add_argument("--max_new_tokens_list", default="128,192,256,384,512")
    parser.add_argument("--output_root", required=True, help="Root directory for ablation outputs")
    parser.add_argument("--no_bertscore", action="store_true", help="Pass --no_bertscore when recomputing summaries")
    parser.add_argument("--skip_eval_if_exists", action="store_true", help="Reuse existing test_predictions_cot.jsonl")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    token_values = []
    for tok in args.max_new_tokens_list.split(","):
        tok = tok.strip()
        if tok:
            token_values.append(int(tok))
    if not token_values:
        raise ValueError("No valid max_new_tokens values provided.")

    all_rows: List[Dict] = []

    for max_tokens in token_values:
        run_dir = output_root / f"max_new_tokens_{max_tokens}"
        run_dir.mkdir(parents=True, exist_ok=True)

        pred_jsonl = run_dir / "test_predictions_cot.jsonl"
        eval_cmd = [
            sys.executable,
            str(repo_root / "scripts" / "evaluation" / "evaluate_audio_cot.py"),
            "--checkpoint",
            args.checkpoint,
            "--encoder",
            args.encoder,
            "--llm_id",
            args.llm_id,
            "--audio_dir",
            args.audio_dir,
            "--cot_data_dir",
            args.cot_data_dir,
            "--batch_size",
            str(args.batch_size),
            "--max_new_tokens",
            str(max_tokens),
            "--output_dir",
            str(run_dir),
        ]
        if args.device:
            eval_cmd += ["--device", args.device]
        if args.max_samples is not None:
            eval_cmd += ["--max_samples", str(args.max_samples)]

        if not (args.skip_eval_if_exists and pred_jsonl.exists()):
            _run(eval_cmd, dry_run=args.dry_run)
        else:
            print(f"\n[skip] Reusing existing predictions: {pred_jsonl}")

        recompute_base = [
            sys.executable,
            str(repo_root / "scripts" / "evaluation" / "recompute_eval_summary_from_cot.py"),
            "--predictions",
            str(pred_jsonl),
        ]
        if args.no_bertscore:
            recompute_base.append("--no_bertscore")

        no_extract_summary = run_dir / "eval_summary_no_extract.json"
        extract_summary = run_dir / "eval_summary_extract_answer.json"

        _run(recompute_base + ["--output", str(no_extract_summary)], dry_run=args.dry_run)
        _run(recompute_base + ["--output", str(extract_summary), "--extract_answer"], dry_run=args.dry_run)

        if args.dry_run:
            continue

        for label, summary_path in [
            ("no_extract", no_extract_summary),
            ("extract_answer", extract_summary),
        ]:
            s = _load_summary(summary_path)
            yn = s.get("closed_ended_yes_no", {}) or {}
            all_rows.append(
                {
                    "max_new_tokens": max_tokens,
                    "extraction": label,
                    "open_ended_accuracy": s.get("open_ended_accuracy"),
                    "normalised_accuracy": s.get("normalised_accuracy"),
                    "contains_match_accuracy": s.get("contains_match_accuracy"),
                    "rougeL_f1": s.get("rougeL_f1"),
                    "meteor": s.get("meteor"),
                    "bert_score_f1": s.get("bert_score_f1"),
                    "yn_accuracy": yn.get("accuracy"),
                    "yn_f1_macro": yn.get("f1_macro"),
                    "yn_f1_weighted": yn.get("f1_weighted"),
                    "yn_sensitivity": yn.get("sensitivity"),
                    "yn_specificity": yn.get("specificity"),
                    "yn_pred_unparseable": yn.get("pred_unparseable"),
                }
            )

    if args.dry_run:
        print("\nDry run complete.")
        return

    summary_rows_path = output_root / "ablation_rows.json"
    with summary_rows_path.open("w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print(f"\nSaved ablation rows to: {summary_rows_path}")
    _print_table(all_rows)


if __name__ == "__main__":
    main()
