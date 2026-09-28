<!-- SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md) -->
<!-- SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project. -->
<!-- SPDX-License-Identifier: MIT -->

# Scripts

Organized by purpose. `label_utils.py` stays at the top level because it's a
shared dependency of several scripts across `evaluation/` and `comparison/`.

## training/
- `audio_curriculum_learning.py` — trains AudioFlamingo on CaReSound. Supports
  `--encoder {tokenizer,mel,wav2vec2,whisper,clap}` and `--max_audio_length`
  (in samples, e.g. 160000/320000/480000 = 10s/20s/30s at 16kHz) for the
  encoder/context-length ablations reported in the paper.

## evaluation/
- `evaluate_audio_model.py` — runs a trained checkpoint on the CaReSound test
  set (open-ended + binary metrics). This produced the paper's Table 2 numbers.
- `evaluate_audio_cot.py` — same, but for checkpoints trained on Chain-of-Thought
  data (rationale + answer metrics).
- `recompute_eval_summary.py` / `recompute_cot_eval_summary.py` /
  `recompute_eval_summary_from_cot.py` — recompute an `eval_summary.json` from
  an existing predictions JSONL without re-running generation (e.g. after a
  metric-computation bugfix).
- `compute_metrics_from_jsonl.py` — generic metrics computation from a
  predictions JSONL (used for baseline outputs too).
- `run_cot_eval_ablation.py` — sweeps `evaluate_audio_cot.py` over multiple
  `--max_new_tokens` values.
- `test_audio_encoders.py` — standalone encoder sanity check (`--quick_test`
  needs no LLM; without it, builds the full AudioFlamingo model for one encoder).
- `test_audio_cot.py` / `quick_test_audio_cot.sh` — smoke tests for the CoT
  generation setup (a handful of samples) before running full generation.

## data_prep/
- `audio_data_preparation.py` — download/organize/merge/verify CaReSound audio.
- `construct_circor_qa.py` — builds the CirCor-specific QA subset.
- `create_patient_splits.py` — generates the patient-disjoint train/val/test
  split (`data/caresound_patient_splits.json`, tracked in git — this is the
  exact split behind the paper's numbers).
- `generate_audio_cot.py` — generates Chain-of-Thought rationales via GPT-4o.

## comparison/
Code behind the baseline rows in Table 2 (Audio-Flamingo3, Qwen2-Audio,
Qwen2.5-Omni, CaReAQA) — keep in sync with `comparison/` and `qwen_eval_results/`
at the repo root.
- `af3_benchmark_eval.py` — Audio-Flamingo3 zero-shot baseline.
- `qwen_audio_benchmark_eval.py` — Qwen2-Audio / Qwen2.5-Omni zero-shot baseline.
- `fix_qwen_jsonl.py` / `fix_qwen_regenerate.py` — one-time cleanup passes for
  Qwen's raw generated text before metrics are computed.

## utils/
- `check_openai_balance.py` — checks OpenAI API credit balance (used before/after
  CoT generation runs, since those call GPT-4o).
