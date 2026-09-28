# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
NVIDIA Audio Flamingo 3 (AF3) Benchmark Evaluation on CaReSound

Evaluates NVIDIA's Audio Flamingo 3 (AF3-7B) as a zero-shot benchmark against
the OpenTSLM model on the CaReSound cardiac & respiratory sound QA dataset.

AF3 is a 7B-parameter Large Audio Language Model (LALM) with:
  - AF-Whisper audio encoder (Whisper large-v3 based)
  - On-demand thinking (Chain-of-Thought reasoning via PEFT adapter)
  - Multi-turn, multi-audio chat support
  - Long-context audio support (up to 10 minutes, 30s windows)

This script runs AF3 in **zero-shot** mode (no fine-tuning) on the CaReSound
test set, making it directly comparable to your fine-tuned OpenTSLM model.

Evaluation modes:
  1. Standard (zero-shot):  Direct question answering
  2. Thinking (CoT):        AF-Think PEFT adapter for Chain-of-Thought reasoning
  3. Few-shot:              With in-context examples before the test question

Metrics computed:
  Open-ended (generative):
    - Exact-match accuracy, normalised accuracy, contains-match accuracy
    - BERTScore (F1, Precision, Recall)
    - ROUGE-1/2/L F1
    - METEOR

  Closed-ended (Yes/No) binary metrics:
    - Accuracy, F1 (macro/weighted), Sensitivity, Specificity
    - Per-dataset breakdown (ICBHI, CirCor, SPRSound, ZCHSound, KAUH)

Prerequisites:
    # AF3 requires the latest transformers (with AudioFlamingo3 support):
    pip install --upgrade git+https://github.com/huggingface/transformers accelerate
    pip install bert-score rouge-score nltk peft

    # Model: nvidia/audio-flamingo-3-hf  (non-commercial research license)

Usage:
    # Standard zero-shot evaluation:
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/af3_benchmark_eval.py \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir af3_eval_results

    # With thinking (CoT reasoning via AF-Think adapter):
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/af3_benchmark_eval.py \\
        --mode thinking \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir af3_eval_results_thinking

    # Few-shot (3 examples, text-only — no few-shot audio):
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/af3_benchmark_eval.py \\
        --mode few_shot --num_shots 3 \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir af3_eval_results_fewshot

    # Quick sanity check (50 samples):
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/af3_benchmark_eval.py \\
        --max_samples 50 \\
        --audio_dir ./data/caresound_audio/audio_merged
"""

import os
import sys
import json
import re
import glob
import time
import argparse
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import torch

# Shared closed-ended label extraction (same as baseline & OpenTSLM evals)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from label_utils import (
    compute_closed_ended_accuracy, is_binary_yn_question,
)

# Semantic / n-gram metrics
from bert_score import score as bert_score_fn
from rouge_score import rouge_scorer
import nltk
from nltk.translate.meteor_score import meteor_score as nltk_meteor

# Ensure NLTK data required by METEOR is available
for _pkg in ("wordnet", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg == "wordnet" else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ─── Audio File Discovery ─────────────────────────────────────────────────────

def find_audio_for_patient(
    patient_id: str,
    dataset_name: str,
    audio_dir: str,
) -> List[str]:
    """
    Find audio file paths for a given patient_id in the audio directory.
    Returns absolute paths (AF3 processor needs file paths, not arrays).

    Supports:
      - Flat merged directory:  audio_dir/{patient_id}.wav
      - CirCor valve files:    audio_dir/{patient_id}_AV.wav, _MV.wav, etc.
      - Dataset subdirectory:  audio_dir/{dataset_name}/{patient_id}.wav
    """
    extensions = [".wav", ".flac", ".mp3", ".ogg"]
    found = []

    # Strategy 1: Flat merged directory  {patient_id}*.ext
    for ext in extensions:
        exact = os.path.join(audio_dir, f"{patient_id}{ext}")
        if os.path.exists(exact):
            found.append(os.path.abspath(exact))
        for p in glob.glob(os.path.join(audio_dir, f"{patient_id}_*{ext}")):
            absp = os.path.abspath(p)
            if absp not in found:
                found.append(absp)

    if found:
        return sorted(found)

    # Strategy 2: Dataset subdirectory  {dataset_name}/{patient_id}*.ext
    subdir = os.path.join(audio_dir, dataset_name)
    if os.path.isdir(subdir):
        for ext in extensions:
            exact = os.path.join(subdir, f"{patient_id}{ext}")
            if os.path.exists(exact):
                found.append(os.path.abspath(exact))
            for p in glob.glob(os.path.join(subdir, f"{patient_id}_*{ext}")):
                absp = os.path.abspath(p)
                if absp not in found:
                    found.append(absp)

    return sorted(found)


# ─── Text Utilities ────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lower-case, strip whitespace / punctuation / special tokens for fuzzy matching."""
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>", "<|im_end|>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_generated(text: str) -> str:
    """
    Clean up AF3 generated text:
    - Remove <think>...</think> blocks (thinking mode)
    - Truncate at follow-up question markers
    - Remove EOS-like tokens
    """
    # Remove thinking blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Truncate at follow-up patterns
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:", "\nHuman:", "\nUser:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    # Remove EOS-like tokens
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|im_end|>"]:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]

    return text.strip()


def extract_thinking(text: str) -> str:
    """Extract the content inside <think>...</think> tags (AF3 CoT reasoning)."""
    match = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    return match.group(1).strip() if match else ""


def clean_gold(text: str) -> str:
    """Remove EOS tokens from gold answer."""
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    return text.strip()


# ─── AF3 Model Loading ────────────────────────────────────────────────────────

def load_af3_model(model_id: str, device: str = "cuda", dtype=torch.bfloat16, think_mode: bool = False):
    """
    Load NVIDIA Audio Flamingo 3 from HuggingFace Transformers.

    Uses AudioFlamingo3ForConditionalGeneration + AutoProcessor.
    For think mode, also loads the AF-Think PEFT (LoRA) adapter.

    Args:
        model_id:    HuggingFace model ID (default: "nvidia/audio-flamingo-3-hf")
        device:      torch device string or "auto" for device_map
        dtype:       model dtype (bfloat16 recommended)
        think_mode:  if True, load the AF-Think PEFT adapter for CoT reasoning

    Returns:
        (model, processor) tuple
    """
    print(f"Loading AF3 model: {model_id}")
    print(f"  Device: {device}, dtype: {dtype}, think_mode: {think_mode}")

    try:
        from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor
    except ImportError:
        print("\n✗ AudioFlamingo3ForConditionalGeneration not found in transformers.")
        print("  AF3 requires the latest transformers. Install with:")
        print("    pip install --upgrade git+https://github.com/huggingface/transformers accelerate")
        sys.exit(1)

    # Load processor (handles audio + text)
    processor = AutoProcessor.from_pretrained(model_id)
    print("  ✓ Loaded AutoProcessor")

    # Load model
    device_map = device if device == "auto" else {"": device}
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e9:.1f}B parameters)")

    # Load AF-Think PEFT adapter for thinking mode
    if think_mode:
        try:
            from huggingface_hub import snapshot_download
            from peft import PeftModel

            print("  Loading AF-Think PEFT adapter for CoT reasoning...")
            local_dir = snapshot_download(model_id)

            # Load non-LoRA trainables
            non_lora_path = os.path.join(local_dir, "think", "non_lora_trainables.bin")
            if os.path.exists(non_lora_path):
                non_lora_trainables = torch.load(non_lora_path, map_location="cpu")
                model.load_state_dict(non_lora_trainables, strict=False)
                print("  ✓ Loaded non-LoRA trainables")

            # Load LoRA adapter
            model = PeftModel.from_pretrained(model, local_dir, subfolder="think")
            print("  ✓ AF-Think PEFT adapter loaded")

        except ImportError:
            print("  ⚠ peft not installed. Install with: pip install peft")
            print("  Falling back to standard mode (no thinking).")
        except Exception as e:
            print(f"  ⚠ Failed to load AF-Think adapter: {e}")
            print("  Falling back to standard mode (no thinking).")

    model.eval()
    return model, processor


# ─── Conversation Building ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert medical audio analysis AI. You listen to cardiac and "
    "respiratory auscultation recordings and provide accurate diagnostic assessments. "
    "Answer concisely and accurately based on what you hear in the audio."
)


def build_conversation(
    audio_paths: List[str],
    question: str,
    pre_prompt: str,
    mode: str = "standard",
    few_shot_examples: Optional[List[Dict]] = None,
) -> List[Dict]:
    """
    Build an AF3 conversation in the HuggingFace chat format.

    AF3 expects:
      {"role": "user", "content": [
          {"type": "text", "text": "..."},
          {"type": "audio", "path": "/path/to/audio.wav"},
      ]}

    Args:
        audio_paths:        List of absolute paths to audio files
        question:           The post_prompt question text
        pre_prompt:         Context / instruction text
        mode:               "standard", "thinking", or "few_shot"
        few_shot_examples:  Text-only examples (no audio for few-shot demos)

    Returns:
        List of message dicts for processor.apply_chat_template()
    """
    messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]}]

    # Few-shot text-only examples (no audio — avoids loading extra audio files)
    if mode == "few_shot" and few_shot_examples:
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": f"{ex['pre_prompt']}\n{ex['question']}"},
                ],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["answer"]}],
            })

    # Main user message with audio
    user_content = []
    # Text instruction
    instruction = f"{pre_prompt}\n\n{question}"
    if mode == "thinking":
        instruction += "\n\nPlease think and reason about the input audio before you respond."
    user_content.append({"type": "text", "text": instruction})

    # Audio files
    for audio_path in audio_paths:
        user_content.append({"type": "audio", "path": audio_path})

    messages.append({"role": "user", "content": user_content})

    return messages


# ─── Few-shot Example Selection ───────────────────────────────────────────────

def select_few_shot_examples(
    train_metadata,
    num_shots: int = 3,
    seed: int = 42,
) -> List[Dict]:
    """
    Select diverse few-shot examples from the training set (text-only, no audio).
    """
    rng = np.random.RandomState(seed)

    # Randomly select examples
    indices = rng.choice(len(train_metadata), size=min(num_shots, len(train_metadata)), replace=False)

    selected = []
    for idx in indices:
        row = train_metadata[idx]
        selected.append({
            "pre_prompt": f"Listen to the following {row.get('dataset', 'medical')} auscultation audio.",
            "question": f"Question: {row['question']}\nAnswer:",
            "answer": row["answer"],
        })

    return selected


# ─── Main Evaluation ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate NVIDIA Audio Flamingo 3 (AF3) on CaReSound test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard zero-shot:
  python scripts/comparison/af3_benchmark_eval.py --audio_dir ./data/caresound_audio/audio_merged

  # With thinking (CoT):
  python scripts/comparison/af3_benchmark_eval.py --mode thinking --audio_dir ./data/caresound_audio/audio_merged

  # Few-shot (3 examples):
  python scripts/comparison/af3_benchmark_eval.py --mode few_shot --num_shots 3 --audio_dir ./data/caresound_audio/audio_merged

  # Quick test:
  python scripts/comparison/af3_benchmark_eval.py --max_samples 50 --audio_dir ./data/caresound_audio/audio_merged
        """,
    )

    # Model arguments
    parser.add_argument(
        "--model_id", type=str, default="nvidia/audio-flamingo-3-hf",
        help="HuggingFace model ID for AF3 (default: nvidia/audio-flamingo-3-hf)",
    )
    parser.add_argument(
        "--mode", type=str, default="standard",
        choices=["standard", "thinking", "few_shot"],
        help="Evaluation mode: standard (zero-shot), thinking (CoT via AF-Think), few_shot",
    )
    parser.add_argument(
        "--num_shots", type=int, default=3,
        help="Number of few-shot examples (only used with --mode few_shot)",
    )

    # Data arguments
    parser.add_argument(
        "--audio_dir", type=str, default="./data/caresound_audio/audio_merged",
        help="Directory containing CaReSound audio files",
    )

    # Evaluation arguments
    parser.add_argument(
        "--max_new_tokens", type=int, default=256,
        help="Max tokens to generate per sample (default: 256; auto-raised to 1024 for thinking mode)",
    )
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick check)")
    parser.add_argument("--output_dir", type=str, default="af3_eval_results",
                        help="Directory to save evaluation results")

    # Hardware arguments
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument(
        "--device_map", type=str, default=None,
        help="Device map for model parallelism (e.g. 'auto' for multi-GPU). "
             "Overrides --device if set.",
    )
    parser.add_argument(
        "--flash_attn", action="store_true",
        help="Use Flash Attention 2 (requires flash-attn package)",
    )

    args = parser.parse_args()

    # Adjust max_new_tokens for thinking mode (needs more for CoT)
    if args.mode == "thinking" and args.max_new_tokens == 256:
        args.max_new_tokens = 1024
        print("Note: Increased max_new_tokens to 1024 for thinking mode")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Print config ──────────────────────────────────────────────────────────
    print("=" * 70)
    print("NVIDIA AUDIO FLAMINGO 3 (AF3) - CARESOUND BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"  Model         : {args.model_id}")
    print(f"  Mode          : {args.mode}")
    if args.mode == "few_shot":
        print(f"  Num shots     : {args.num_shots}")
    print(f"  Audio dir     : {args.audio_dir}")
    print(f"  Device        : {args.device_map or args.device}")
    print(f"  Max tokens    : {args.max_new_tokens}")
    print(f"  Output dir    : {args.output_dir}")
    print()

    # ── Load AF3 model ────────────────────────────────────────────────────────
    device_arg = args.device_map if args.device_map else args.device
    model, processor = load_af3_model(
        args.model_id,
        device=device_arg,
        think_mode=(args.mode == "thinking"),
    )

    # ── Load CaReSound dataset (patient-split aware) ────────────────────────
    print("\nLoading CaReSound dataset from HuggingFace...")
    from datasets import load_dataset

    hf_dataset = load_dataset("tsnngw/CaReSound")

    # Use patient-level split if available (same test set as OpenTSLM)
    patient_splits_path = "data/caresound_patient_splits.json"
    if os.path.exists(patient_splits_path):
        import json as _json
        with open(patient_splits_path) as _f:
            _split_data = _json.load(_f)
        _split_map = _split_data["splits"]

        # Pool all HF rows, filter by patient assignment
        all_rows = list(hf_dataset["train"]) + list(hf_dataset["test"])
        test_data = [r for r in all_rows if _split_map.get(r["patient_id"]) == "test"]
        train_data = [r for r in all_rows if _split_map.get(r["patient_id"]) == "train"]
        print(f"  Using patient-disjoint split ({patient_splits_path})")
    else:
        test_data = list(hf_dataset["test"])
        train_data = list(hf_dataset["train"])
        print("  ⚠ No patient split file — using legacy HF train/test split")

    total = len(test_data) if args.max_samples is None else min(len(test_data), args.max_samples)
    print(f"  Test samples  : {total}")
    print(f"  Train samples : {len(train_data)} (for few-shot pool)")

    # ── Select few-shot examples ──────────────────────────────────────────────
    few_shot_examples = None
    if args.mode == "few_shot":
        few_shot_examples = select_few_shot_examples(train_data, args.num_shots)
        print(f"\n  Few-shot examples selected:")
        for i, ex in enumerate(few_shot_examples):
            print(f"    {i+1}. {ex['answer'][:60]}...")

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []
    exact_correct = 0
    norm_correct = 0
    contains_correct_total = 0
    per_dataset_stats = defaultdict(lambda: {"total": 0, "exact": 0, "norm": 0, "contains": 0})
    samples_done = 0
    skipped = 0

    results_jsonl = os.path.join(args.output_dir, "af3_predictions.jsonl")
    print(f"\n  Results will be saved to: {results_jsonl}")
    print()

    t_start = time.time()

    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for idx in range(total):
            row = test_data[idx]
            patient_id = row["patient_id"]
            dataset_name = row.get("dataset", "unknown")
            question = row["question"]
            gold_answer = row["answer"]

            # Find audio file paths
            audio_paths = find_audio_for_patient(patient_id, dataset_name, args.audio_dir)

            if not audio_paths:
                skipped += 1
                if skipped <= 5:
                    print(f"  ⚠ No audio found for patient {patient_id} ({dataset_name}), skipping")
                elif skipped == 6:
                    print(f"  ⚠ (suppressing further 'no audio' warnings...)")
                continue

            # AF3 processor requires 1:1 text-to-audio mapping.
            # Use only the first audio file per patient (CaReSound questions
            # are per-patient so a single recording is sufficient).
            audio_paths = audio_paths[:1]

            # Build prompts
            pre_prompt = (
                f"Listen to the following {dataset_name} auscultation audio "
                f"and answer the diagnostic question."
            )
            post_prompt = f"Question: {question}\nAnswer:"

            # Build conversation and generate
            try:
                conversation = build_conversation(
                    audio_paths=audio_paths,
                    question=post_prompt,
                    pre_prompt=pre_prompt,
                    mode=args.mode,
                    few_shot_examples=few_shot_examples,
                )

                # Use AF3's processor.apply_chat_template() to prepare inputs
                inputs = processor.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                ).to(model.device)

                # Cast float tensors to model dtype (processor returns fp32,
                # model is bfloat16 → dtype mismatch in audio encoder)
                model_dtype = next(model.parameters()).dtype
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        inputs[k] = v.to(model_dtype)

                input_len = inputs["input_ids"].shape[-1]

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,  # greedy for reproducibility
                    )

                # Decode only the newly generated tokens
                generated_ids = output_ids[:, input_len:]
                pred_raw = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            except Exception as e:
                print(f"  ⚠ Generation error for sample {idx} (patient {patient_id}): {e}")
                pred_raw = "[ERROR]"

            # Extract thinking and clean prediction
            thinking = extract_thinking(pred_raw) if args.mode == "thinking" else ""
            pred_clean = clean_generated(pred_raw)
            gold = clean_gold(gold_answer)

            # Compute matches
            exact_match = (pred_clean.strip() == gold.strip())
            norm_match = (normalise(pred_clean) == normalise(gold))
            contains_match = (normalise(gold) in normalise(pred_clean))

            exact_correct += int(exact_match)
            norm_correct += int(norm_match)
            contains_correct_total += int(contains_match)

            # Per-dataset stats
            per_dataset_stats[dataset_name]["total"] += 1
            per_dataset_stats[dataset_name]["exact"] += int(exact_match)
            per_dataset_stats[dataset_name]["norm"] += int(norm_match)
            per_dataset_stats[dataset_name]["contains"] += int(contains_match)

            row_result = {
                "idx": samples_done,
                "patient_id": patient_id,
                "dataset": dataset_name,
                "pre_prompt": pre_prompt,
                "post_prompt": post_prompt,
                "generated_raw": pred_raw,
                "generated_clean": pred_clean,
                "thinking": thinking,
                "gold": gold,
                "exact_match": exact_match,
                "norm_match": norm_match,
                "contains_match": contains_match,
                "num_audio_files": len(audio_paths),
            }
            results.append(row_result)
            f_out.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            samples_done += 1

            # Progress logging
            if samples_done % 50 == 0:
                elapsed_so_far = time.time() - t_start
                rate = elapsed_so_far / samples_done
                eta = rate * (total - samples_done)
                print(
                    f"  [{samples_done}/{total}] "
                    f"exact_acc={exact_correct/samples_done*100:.1f}% "
                    f"norm_acc={norm_correct/samples_done*100:.1f}% "
                    f"({rate:.1f}s/sample, ETA: {eta/60:.0f}min)"
                )
                torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    if samples_done == 0:
        print("\n✗ No samples were evaluated. Check your audio_dir path.")
        sys.exit(1)

    # ── Compute BERTScore, ROUGE, METEOR ──────────────────────────────────────
    all_preds = [r["generated_clean"] for r in results]
    all_golds = [r["gold"] for r in results]

    # ROUGE & METEOR
    print("\nComputing ROUGE & METEOR scores...")
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
    meteor_scores = []
    per_dataset_rougeL = defaultdict(list)
    per_dataset_meteor = defaultdict(list)

    for r in results:
        scores = scorer.score(r["gold"], r["generated_clean"])
        r1 = scores["rouge1"].fmeasure
        r2 = scores["rouge2"].fmeasure
        rL = scores["rougeL"].fmeasure
        rouge1_scores.append(r1)
        rouge2_scores.append(r2)
        rougeL_scores.append(rL)
        r["rouge1"] = r1
        r["rouge2"] = r2
        r["rougeL"] = rL
        per_dataset_rougeL[r["dataset"]].append(rL)

        ref_tokens = nltk.word_tokenize(r["gold"])
        hyp_tokens = nltk.word_tokenize(r["generated_clean"])
        m = nltk_meteor([ref_tokens], hyp_tokens)
        meteor_scores.append(m)
        r["meteor"] = m
        per_dataset_meteor[r["dataset"]].append(m)

    avg_rouge1 = sum(rouge1_scores) / max(len(rouge1_scores), 1)
    avg_rouge2 = sum(rouge2_scores) / max(len(rouge2_scores), 1)
    avg_rougeL = sum(rougeL_scores) / max(len(rougeL_scores), 1)
    avg_meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    # BERTScore
    print("Computing BERTScore (this may take a moment)...")
    max_bert_len = 512
    preds_trunc = [p[:max_bert_len] if p else " " for p in all_preds]
    golds_trunc = [g[:max_bert_len] if g else " " for g in all_golds]

    bert_device = args.device
    P, R, F1 = bert_score_fn(
        preds_trunc, golds_trunc,
        lang="en",
        model_type="roberta-large",
        device=bert_device,
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

    # --- Closed-ended accuracy (Yes/No parsing) ---
    print("Computing closed-ended (Yes/No) accuracy...")
    all_post_prompts = [r["post_prompt"] for r in results]
    closed_ended = compute_closed_ended_accuracy(all_golds, all_preds, questions=all_post_prompts)

    # Store per-result Yes/No flags
    from label_utils import parse_yes_no
    for r in results:
        r["closed_ended_gold_yn"] = parse_yes_no(r["gold"])
        r["closed_ended_pred_yn"] = parse_yes_no(r["generated_clean"])
        r["closed_ended_is_binary_q"] = is_binary_yn_question(r["post_prompt"])

    # Re-write JSONL with all metric columns
    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    exact_acc = exact_correct / max(samples_done, 1) * 100
    norm_acc = norm_correct / max(samples_done, 1) * 100
    contains_acc = contains_correct_total / max(samples_done, 1) * 100

    print()
    print("=" * 70)
    print(f"AF3 EVALUATION RESULTS  (mode: {args.mode})")
    print("=" * 70)
    print(f"  Model                     : {args.model_id}")
    print(f"  Mode                      : {args.mode}")
    print(f"  Total samples evaluated   : {samples_done}")
    print(f"  Skipped (no audio)        : {skipped}")
    print(f"  Time                      : {elapsed:.1f}s ({elapsed/max(samples_done,1):.2f}s/sample)")
    print()
    print("  ── Open-ended (generative) metrics ──")
    print(f"    Exact-match accuracy      : {exact_correct}/{samples_done} = {exact_acc:.2f}%")
    print(f"    Normalised accuracy       : {norm_correct}/{samples_done} = {norm_acc:.2f}%")
    print(f"    Contains-match accuracy   : {contains_correct_total}/{samples_done} = {contains_acc:.2f}%")
    print()
    print(f"    BERTScore F1              : {avg_bert_f1:.4f}  (P={avg_bert_p:.4f}, R={avg_bert_r:.4f})")
    print(f"    ROUGE-1 F1                : {avg_rouge1:.4f}")
    print(f"    ROUGE-2 F1                : {avg_rouge2:.4f}")
    print(f"    ROUGE-L F1                : {avg_rougeL:.4f}")
    print(f"    METEOR                    : {avg_meteor:.4f}")
    print()
    print("  ── Closed-ended (Yes/No) binary metrics ──")
    print(f"    Accuracy                  : {closed_ended['accuracy']:.4f}")
    print(f"    F1 (macro)                : {closed_ended['f1_macro']:.4f}")
    print(f"    F1 (weighted)             : {closed_ended['f1_weighted']:.4f}")
    print(f"    Sensitivity (Recall-Yes)  : {closed_ended['sensitivity']:.4f}")
    print(f"    Specificity (Recall-No)   : {closed_ended['specificity']:.4f}")
    print(f"    Evaluated                 : {closed_ended['evaluated']} binary yes/no samples")
    print(f"    Skipped (non-binary Q)    : {closed_ended['skipped_not_binary_question']}")
    print(f"    Skipped (non-yes/no gold) : {closed_ended["skipped_not_yes_no"]}")
    print(f"    Pred unparseable          : {closed_ended['pred_unparseable']}")
    print(f"    Label distribution        : {closed_ended['label_counts']['gold_yes']} Yes ({closed_ended['label_counts']['gold_yes']/max(closed_ended['evaluated'],1)*100:.1f}%), "
          f"{closed_ended['label_counts']['gold_no']} No ({closed_ended['label_counts']['gold_no']/max(closed_ended['evaluated'],1)*100:.1f}%)")
    print()
    print("  ── Per-dataset breakdown (open-ended) ──")
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        ds_exact = stats["exact"] / max(stats["total"], 1) * 100
        ds_contains = stats["contains"] / max(stats["total"], 1) * 100
        ds_rougeL = sum(per_dataset_rougeL.get(ds_name, [0])) / max(len(per_dataset_rougeL.get(ds_name, [1])), 1)
        ds_meteor = sum(per_dataset_meteor.get(ds_name, [0])) / max(len(per_dataset_meteor.get(ds_name, [1])), 1)
        ds_bert = sum(per_dataset_bert.get(ds_name, [0])) / max(len(per_dataset_bert.get(ds_name, [1])), 1)
        print(
            f"    {ds_name:12s}: n={stats['total']:4d}, exact={ds_exact:5.1f}%, "
            f"ROUGE-L={ds_rougeL:.3f}, METEOR={ds_meteor:.3f}, BERTScore={ds_bert:.3f}"
        )

    # ── Save summary JSON ─────────────────────────────────────────────────────
    summary = {
        "model_id": args.model_id,
        "model_type": "NVIDIA Audio Flamingo 3 (AF3-7B)",
        "mode": args.mode,
        "num_shots": args.num_shots if args.mode == "few_shot" else 0,
        "fine_tuned": False,
        "total_samples": samples_done,
        "skipped_no_audio": skipped,
        "open_ended_accuracy": exact_acc,
        "normalised_accuracy": norm_acc,
        "contains_match_accuracy": contains_acc,
        "bert_score_f1": avg_bert_f1,
        "bert_score_precision": avg_bert_p,
        "bert_score_recall": avg_bert_r,
        "rouge1_f1": avg_rouge1,
        "rouge2_f1": avg_rouge2,
        "rougeL_f1": avg_rougeL,
        "meteor": avg_meteor,
        "careqa_closed_ended": closed_ended,
        "elapsed_seconds": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "per_dataset": {
            ds: {
                "total": per_dataset_stats[ds]["total"],
                "open_ended_acc": per_dataset_stats[ds]["exact"] / max(per_dataset_stats[ds]["total"], 1) * 100,
                "norm_acc": per_dataset_stats[ds]["norm"] / max(per_dataset_stats[ds]["total"], 1) * 100,
                "contains_acc": per_dataset_stats[ds]["contains"] / max(per_dataset_stats[ds]["total"], 1) * 100,
                "rougeL_f1": sum(per_dataset_rougeL.get(ds, [0])) / max(len(per_dataset_rougeL.get(ds, [1])), 1),
                "meteor": sum(per_dataset_meteor.get(ds, [0])) / max(len(per_dataset_meteor.get(ds, [1])), 1),
                "bert_f1": sum(per_dataset_bert.get(ds, [0])) / max(len(per_dataset_bert.get(ds, [1])), 1),
            }
            for ds in sorted(per_dataset_stats.keys())
        },
    }

    summary_path = os.path.join(args.output_dir, "af3_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  Predictions : {results_jsonl}")
    print(f"  Summary     : {summary_path}")

    # ── Show sample predictions ───────────────────────────────────────────────
    print()
    print("-" * 70)
    print("SAMPLE PREDICTIONS (first 5)")
    print("-" * 70)
    for r in results[:5]:
        print(f"  [{r['dataset']}] Patient: {r['patient_id']}")
        print(f"  Q: {r['post_prompt'][:100]}")
        print(f"  Gold:      {r['gold'][:100]}")
        print(f"  Predicted: {r['generated_clean'][:100]}")
        if r.get("thinking"):
            print(f"  Thinking:  {r['thinking'][:150]}...")
        print(f"  Yes/No: gold={r.get('closed_ended_gold_yn','?')}, pred={r.get('closed_ended_pred_yn','?')}")
        print(f"  Match: exact={r['exact_match']}, ROUGE-L={r.get('rougeL', 0):.3f}, "
              f"BERTScore={r.get('bert_f1', 0):.3f}")
        print()

    print("=" * 70)
    print("COMPARISON GUIDE")
    print("=" * 70)
    print("  To compare AF3 vs OpenTSLM, run both:")
    print()
    print("  1. OpenTSLM (after training completes):")
    print("     python scripts/evaluation/evaluate_audio_model.py \\")
    print("         --checkpoint audio_checkpoints/.../best_model.pt \\")
    print("         --encoder mel --audio_dir ./data/caresound_audio/audio_merged")
    print()
    print("  3. AF3 (this script):")
    print(f"     python scripts/comparison/af3_benchmark_eval.py --mode {args.mode}")
    print()
    print("  Then compare af3_eval_summary.json vs eval_summary.json files.")
    print("=" * 70)


if __name__ == "__main__":
    main()
