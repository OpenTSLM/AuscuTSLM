# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Qwen Audio Benchmark Evaluation on CaReSound

Evaluates Qwen audio foundation models as zero-shot baselines against the
OpenTSLM model on the CaReSound cardiac & respiratory sound QA dataset.

Supported models:
  - Qwen/Qwen2-Audio-7B-Instruct   (Qwen2-Audio, instruction-tuned)
  - Qwen/Qwen2.5-Omni-7B           (Qwen2.5-Omni multimodal)
  - Any Qwen2-Audio or Qwen2.5-Omni variant on HuggingFace

Model families and APIs:
  qwen2-audio:   Qwen2AudioForConditionalGeneration + AutoProcessor
                 Audio loaded as numpy array via librosa.
  qwen2.5-omni:  Qwen2_5OmniModel + Qwen2_5OmniProcessor
                 Audio loaded from file path.

Prerequisites:
    pip install transformers accelerate librosa soundfile
    pip install bert-score rouge-score nltk

Usage:
    # Qwen2-Audio zero-shot:
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/qwen_audio_benchmark_eval.py \\
        --model_id Qwen/Qwen2-Audio-7B-Instruct \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir qwen_eval_results/qwen2_audio

    # Qwen2.5-Omni zero-shot:
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/qwen_audio_benchmark_eval.py \\
        --model_id Qwen/Qwen2.5-Omni-7B \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir qwen_eval_results/qwen25_omni

    # Few-shot (3 text-only examples):
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/qwen_audio_benchmark_eval.py \\
        --model_id Qwen/Qwen2-Audio-7B-Instruct \\
        --mode few_shot --num_shots 3 \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --output_dir qwen_eval_results/qwen2_audio_fewshot

    # Quick sanity check (50 samples):
    CUDA_VISIBLE_DEVICES=0 python scripts/comparison/qwen_audio_benchmark_eval.py \\
        --model_id Qwen/Qwen2-Audio-7B-Instruct \\
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
from typing import Dict, List, Optional, Tuple

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

for _pkg in ("wordnet", "punkt_tab"):
    try:
        nltk.data.find(f"corpora/{_pkg}" if _pkg == "wordnet" else f"tokenizers/{_pkg}")
    except LookupError:
        nltk.download(_pkg, quiet=True)


# ─── Audio File Discovery ──────────────────────────────────────────────────────

def find_audio_for_patient(patient_id: str, dataset_name: str, audio_dir: str) -> List[str]:
    """Find audio file path(s) for a given patient in the merged audio directory."""
    extensions = [".wav", ".flac", ".mp3", ".ogg"]
    found = []

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

    # Fallback: dataset subdirectory
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
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>", "<|im_end|>", "<|endoftext|>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_generated(text: str) -> str:
    """Truncate at follow-up question patterns and strip EOS tokens."""
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:"]:
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>"]:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def clean_gold(text: str) -> str:
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    return text.strip()


# ─── Model Family Detection ────────────────────────────────────────────────────

def detect_model_family(model_id: str) -> str:
    """
    Returns 'qwen2_audio' or 'qwen2_5_omni' based on model_id.
    Raises ValueError for unrecognised IDs.
    """
    mid = model_id.lower()
    if "omni" in mid:
        return "qwen2_5_omni"
    if "qwen2-audio" in mid or "qwen2audio" in mid:
        return "qwen2_audio"
    # Default to qwen2_audio for unknown Qwen audio variants
    print(f"  ⚠ Could not auto-detect model family from '{model_id}'. Defaulting to qwen2_audio.")
    return "qwen2_audio"


# ─── Model Loading ─────────────────────────────────────────────────────────────

def load_qwen2_audio(model_id: str, device: str, dtype) -> Tuple:
    """Load Qwen2-Audio model and processor."""
    try:
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
    except ImportError:
        print("✗ Qwen2AudioForConditionalGeneration not found.")
        print("  Install with: pip install --upgrade transformers accelerate")
        sys.exit(1)

    print(f"  Loading AutoProcessor...")
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"  Loading Qwen2AudioForConditionalGeneration...")
    device_map = "auto" if device == "auto" else {"": device}
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Qwen2-Audio loaded ({n_params:.1f}B params)")
    return model, processor


def load_qwen2_5_omni(model_id: str, device: str, dtype) -> Tuple:
    """Load Qwen2.5-Omni model and processor."""
    try:
        from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
    except ImportError:
        # Older transformers may not have Qwen2_5OmniModel; fall back to Auto classes
        print("  ⚠ Qwen2_5OmniModel not found, trying AutoModelForCausalLM...")
        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
            processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            device_map = "auto" if device == "auto" else {"": device}
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            model.eval()
            n_params = sum(p.numel() for p in model.parameters()) / 1e9
            print(f"  ✓ Qwen2.5-Omni loaded via AutoModel ({n_params:.1f}B params)")
            return model, processor
        except Exception as e:
            print(f"✗ Failed to load Qwen2.5-Omni: {e}")
            print("  Install with: pip install --upgrade transformers accelerate")
            sys.exit(1)

    print(f"  Loading Qwen2_5OmniProcessor...")
    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)

    print(f"  Loading Qwen2_5OmniModel...")
    device_map = "auto" if device == "auto" else {"": device}
    model = Qwen2_5OmniModel.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device_map,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  ✓ Qwen2.5-Omni loaded ({n_params:.1f}B params)")
    return model, processor


def load_model(model_id: str, device: str, dtype, model_family: str) -> Tuple:
    print(f"Loading Qwen model: {model_id}")
    print(f"  Family: {model_family}, Device: {device}, dtype: {dtype}")
    if model_family == "qwen2_5_omni":
        return load_qwen2_5_omni(model_id, device, dtype)
    else:
        return load_qwen2_audio(model_id, device, dtype)


# ─── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio_array(audio_path: str, target_sr: int = 16000) -> np.ndarray:
    """Load audio file as a float32 numpy array at the target sample rate."""
    try:
        import librosa
        audio, _ = librosa.load(audio_path, sr=target_sr, mono=True)
        return audio
    except ImportError:
        try:
            import soundfile as sf
            import resampy
            audio, sr = sf.read(audio_path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != target_sr:
                audio = resampy.resample(audio, sr, target_sr)
            return audio.astype(np.float32)
        except ImportError:
            print("✗ Neither librosa nor soundfile+resampy found.")
            print("  Install with: pip install librosa")
            sys.exit(1)


# ─── Conversation Building ─────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert medical audio analysis AI. You listen to cardiac and "
    "respiratory auscultation recordings and provide accurate diagnostic assessments. "
    "Answer concisely and accurately based on what you hear in the audio."
)


def build_qwen2_audio_inputs(
    audio_path: str,
    question: str,
    pre_prompt: str,
    processor,
    device: str,
    few_shot_examples: Optional[List[Dict]] = None,
):
    """
    Build inputs for Qwen2-Audio-7B-Instruct.

    The processor expects audio as numpy arrays and the conversation uses
    {"type": "audio", "audio_url": path} placeholders; the processor resolves
    the actual waveform via its feature extractor.
    """
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    # Text-only few-shot examples (no audio)
    if few_shot_examples:
        for ex in few_shot_examples:
            conversation.append({
                "role": "user",
                "content": f"{ex['pre_prompt']}\n{ex['question']}",
            })
            conversation.append({
                "role": "assistant",
                "content": ex["answer"],
            })

    # Main user message with audio
    conversation.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio_url": audio_path},
            {"type": "text", "text": f"{pre_prompt}\n\n{question}"},
        ],
    })

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )

    # Load audio as numpy array
    sr = processor.feature_extractor.sampling_rate
    audio_array = load_audio_array(audio_path, target_sr=sr)

    inputs = processor(
        text=text,
        audios=[audio_array],
        return_tensors="pt",
        padding=True,
        sampling_rate=sr,
    )
    return {k: v.to(device) for k, v in inputs.items()}


def build_qwen2_5_omni_inputs(
    audio_path: str,
    question: str,
    pre_prompt: str,
    processor,
    device: str,
    few_shot_examples: Optional[List[Dict]] = None,
):
    """
    Build inputs for Qwen2.5-Omni.

    Uses {"type": "audio", "audio": path} content items; the processor
    handles audio loading internally.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    ]

    if few_shot_examples:
        for ex in few_shot_examples:
            messages.append({
                "role": "user",
                "content": [{"type": "text", "text": f"{ex['pre_prompt']}\n{ex['question']}"}],
            })
            messages.append({
                "role": "assistant",
                "content": [{"type": "text", "text": ex["answer"]}],
            })

    messages.append({
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_path},
            {"type": "text", "text": f"{pre_prompt}\n\n{question}"},
        ],
    })

    text = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    sr = getattr(processor.feature_extractor, "sampling_rate", 16000)
    audio_array = load_audio_array(audio_path, target_sr=sr)

    inputs = processor(
        text=text,
        audio=audio_array,
        return_tensors="pt",
        padding=True,
        sampling_rate=sr,
    )
    return {k: v.to(device) for k, v in inputs.items()}


# ─── Few-shot Example Selection ───────────────────────────────────────────────

def select_few_shot_examples(train_metadata, num_shots: int = 3, seed: int = 42) -> List[Dict]:
    rng = np.random.RandomState(seed)
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
        description="Evaluate Qwen Audio models on CaReSound test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/comparison/qwen_audio_benchmark_eval.py \\
      --model_id Qwen/Qwen2-Audio-7B-Instruct \\
      --audio_dir ./data/caresound_audio/audio_merged

  python scripts/comparison/qwen_audio_benchmark_eval.py \\
      --model_id Qwen/Qwen2.5-Omni-7B \\
      --audio_dir ./data/caresound_audio/audio_merged \\
      --output_dir qwen_eval_results/qwen25_omni

  python scripts/comparison/qwen_audio_benchmark_eval.py \\
      --model_id Qwen/Qwen2-Audio-7B-Instruct --max_samples 50 \\
      --audio_dir ./data/caresound_audio/audio_merged
        """,
    )

    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-Audio-7B-Instruct",
                        help="HuggingFace model ID (default: Qwen/Qwen2-Audio-7B-Instruct)")
    parser.add_argument("--model_family", type=str, default=None,
                        choices=["qwen2_audio", "qwen2_5_omni"],
                        help="Model family (auto-detected from model_id if not set)")
    parser.add_argument("--mode", type=str, default="standard",
                        choices=["standard", "few_shot"],
                        help="Evaluation mode: standard (zero-shot) or few_shot")
    parser.add_argument("--num_shots", type=int, default=3,
                        help="Number of few-shot text-only examples (only with --mode few_shot)")
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_merged",
                        help="Directory containing CaReSound audio files")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per sample (default: 128)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick sanity check)")
    parser.add_argument("--output_dir", type=str, default="qwen_eval_results",
                        help="Directory to save evaluation results")

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)
    parser.add_argument("--device_map", type=str, default=None,
                        help="Set 'auto' for multi-GPU. Overrides --device.")
    parser.add_argument("--fp16", action="store_true",
                        help="Use float16 instead of bfloat16")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = torch.float16 if args.fp16 else torch.bfloat16
    device = args.device_map if args.device_map else args.device
    model_family = args.model_family or detect_model_family(args.model_id)

    print("=" * 70)
    print("QWEN AUDIO — CARESOUND BENCHMARK EVALUATION")
    print("=" * 70)
    print(f"  Model         : {args.model_id}")
    print(f"  Family        : {model_family}")
    print(f"  Mode          : {args.mode}")
    if args.mode == "few_shot":
        print(f"  Num shots     : {args.num_shots}")
    print(f"  Audio dir     : {args.audio_dir}")
    print(f"  Device        : {device}")
    print(f"  dtype         : {dtype}")
    print(f"  Max tokens    : {args.max_new_tokens}")
    print(f"  Output dir    : {args.output_dir}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    model, processor = load_model(args.model_id, device, dtype, model_family)

    # ── Load CaReSound dataset (patient-split aware) ──────────────────────────
    print("\nLoading CaReSound dataset from HuggingFace...")
    from datasets import load_dataset

    hf_dataset = load_dataset("tsnngw/CaReSound")

    patient_splits_path = "data/caresound_patient_splits.json"
    if os.path.exists(patient_splits_path):
        with open(patient_splits_path) as f:
            _split_data = json.load(f)
        _split_map = _split_data["splits"]
        all_rows = list(hf_dataset["train"]) + list(hf_dataset["test"])
        test_data = [r for r in all_rows if _split_map.get(r["patient_id"]) == "test"]
        train_data = [r for r in all_rows if _split_map.get(r["patient_id"]) == "train"]
        print(f"  Using patient-disjoint split ({patient_splits_path})")
    else:
        test_data = list(hf_dataset["test"])
        train_data = list(hf_dataset["train"])
        print("  ⚠ No patient split file — using legacy HF train/test split (data leakage risk)")

    total = len(test_data) if args.max_samples is None else min(len(test_data), args.max_samples)
    print(f"  Test samples  : {total}")
    print(f"  Train samples : {len(train_data)} (few-shot pool)")

    few_shot_examples = None
    if args.mode == "few_shot":
        few_shot_examples = select_few_shot_examples(train_data, args.num_shots)
        print(f"\n  Few-shot examples:")
        for i, ex in enumerate(few_shot_examples):
            print(f"    {i+1}. {ex['answer'][:60]}...")

    # ── Inference ─────────────────────────────────────────────────────────────
    results = []
    exact_correct = 0
    norm_correct = 0
    contains_correct_total = 0
    per_dataset_stats = defaultdict(lambda: {"total": 0, "exact": 0, "norm": 0, "contains": 0})
    samples_done = 0
    skipped = 0

    results_jsonl = os.path.join(args.output_dir, "qwen_predictions.jsonl")
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

            audio_paths = find_audio_for_patient(patient_id, dataset_name, args.audio_dir)
            if not audio_paths:
                skipped += 1
                if skipped <= 5:
                    print(f"  ⚠ No audio found for patient {patient_id} ({dataset_name}), skipping")
                elif skipped == 6:
                    print("  ⚠ (suppressing further 'no audio' warnings...)")
                continue

            audio_path = audio_paths[0]  # use first file per patient
            pre_prompt = (
                f"Listen to the following {dataset_name} auscultation audio "
                f"and answer the diagnostic question."
            )
            post_prompt = f"Question: {question}\nAnswer:"

            try:
                if model_family == "qwen2_5_omni":
                    inputs = build_qwen2_5_omni_inputs(
                        audio_path, post_prompt, pre_prompt,
                        processor, device, few_shot_examples,
                    )
                else:
                    inputs = build_qwen2_audio_inputs(
                        audio_path, post_prompt, pre_prompt,
                        processor, device, few_shot_examples,
                    )

                # Cast float tensors to model dtype
                model_dtype = next(model.parameters()).dtype
                for k, v in inputs.items():
                    if isinstance(v, torch.Tensor) and v.is_floating_point():
                        inputs[k] = v.to(model_dtype)

                input_len = inputs["input_ids"].shape[-1]

                with torch.no_grad():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )

                generated_ids = output_ids[:, input_len:]
                pred_raw = processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )[0]

            except Exception as e:
                print(f"  ⚠ Generation error for sample {idx} (patient {patient_id}): {e}")
                pred_raw = "[ERROR]"

            pred_clean = clean_generated(pred_raw)
            gold = clean_gold(gold_answer)

            exact_match = (pred_clean.strip() == gold.strip())
            norm_match = (normalise(pred_clean) == normalise(gold))
            contains_match = (normalise(gold) in normalise(pred_clean))

            exact_correct += int(exact_match)
            norm_correct += int(norm_match)
            contains_correct_total += int(contains_match)

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
                "gold": gold,
                "exact_match": exact_match,
                "norm_match": norm_match,
                "contains_match": contains_match,
            }
            results.append(row_result)
            f_out.write(json.dumps(row_result, ensure_ascii=False) + "\n")
            samples_done += 1

            if samples_done % 50 == 0:
                elapsed_so_far = time.time() - t_start
                rate = elapsed_so_far / samples_done
                eta = rate * (total - samples_done)
                print(
                    f"  [{samples_done}/{total}] "
                    f"exact={exact_correct/samples_done*100:.1f}% "
                    f"norm={norm_correct/samples_done*100:.1f}% "
                    f"({rate:.1f}s/sample, ETA {eta/60:.0f}min)"
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    if samples_done == 0:
        print("\n✗ No samples evaluated. Check --audio_dir path.")
        sys.exit(1)

    # ── Metrics ───────────────────────────────────────────────────────────────
    all_preds = [r["generated_clean"] for r in results]
    all_golds = [r["gold"] for r in results]

    print("\nComputing ROUGE & METEOR scores...")
    _scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge1_scores, rouge2_scores, rougeL_scores, meteor_scores = [], [], [], []
    per_dataset_rougeL = defaultdict(list)
    per_dataset_meteor = defaultdict(list)

    for r in results:
        scores = _scorer.score(r["gold"], r["generated_clean"])
        r["rouge1"] = scores["rouge1"].fmeasure
        r["rouge2"] = scores["rouge2"].fmeasure
        r["rougeL"] = scores["rougeL"].fmeasure
        rouge1_scores.append(r["rouge1"])
        rouge2_scores.append(r["rouge2"])
        rougeL_scores.append(r["rougeL"])
        per_dataset_rougeL[r["dataset"]].append(r["rougeL"])

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

    print("Computing BERTScore...")
    max_bert_len = 512
    preds_trunc = [p[:max_bert_len] if p else " " for p in all_preds]
    golds_trunc = [g[:max_bert_len] if g else " " for g in all_golds]
    bert_device = args.device if not args.device_map else "cuda"
    P, R, F1 = bert_score_fn(
        preds_trunc, golds_trunc,
        lang="en", model_type="roberta-large",
        device=bert_device, batch_size=32, verbose=False,
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

    print("Computing closed-ended (Yes/No) accuracy...")
    all_post_prompts = [r["post_prompt"] for r in results]
    closed_ended = compute_closed_ended_accuracy(all_golds, all_preds, questions=all_post_prompts)

    from label_utils import parse_yes_no
    for r in results:
        r["closed_ended_gold_yn"] = parse_yes_no(r["gold"])
        r["closed_ended_pred_yn"] = parse_yes_no(r["generated_clean"])

    # Rewrite JSONL with all metrics
    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    exact_acc = exact_correct / max(samples_done, 1) * 100
    norm_acc = norm_correct / max(samples_done, 1) * 100
    contains_acc = contains_correct_total / max(samples_done, 1) * 100

    print()
    print("=" * 70)
    print("QWEN AUDIO EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Model                     : {args.model_id}")
    print(f"  Mode                      : {args.mode}")
    print(f"  Total samples evaluated   : {samples_done}")
    print(f"  Skipped (no audio)        : {skipped}")
    print(f"  Time                      : {elapsed:.1f}s ({elapsed/max(samples_done,1):.2f}s/sample)")
    print()
    print("  ── Open-ended metrics ──")
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
    print("  ── Closed-ended (Yes/No) metrics ──")
    print(f"    Accuracy                  : {closed_ended['accuracy']:.4f}")
    print(f"    F1 (macro)                : {closed_ended['f1_macro']:.4f}")
    print(f"    F1 (weighted)             : {closed_ended['f1_weighted']:.4f}")
    print(f"    Sensitivity               : {closed_ended['sensitivity']:.4f}")
    print(f"    Specificity               : {closed_ended['specificity']:.4f}")
    print(f"    Evaluated                 : {closed_ended['evaluated']} binary yes/no samples")
    print(f"    Pred unparseable          : {closed_ended['pred_unparseable']}")
    print()
    print("  ── Per-dataset breakdown ──")
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        ds_exact = stats["exact"] / max(stats["total"], 1) * 100
        ds_rougeL = sum(per_dataset_rougeL.get(ds_name, [0])) / max(len(per_dataset_rougeL.get(ds_name, [1])), 1)
        ds_meteor = sum(per_dataset_meteor.get(ds_name, [0])) / max(len(per_dataset_meteor.get(ds_name, [1])), 1)
        ds_bert = sum(per_dataset_bert.get(ds_name, [0])) / max(len(per_dataset_bert.get(ds_name, [1])), 1)
        print(
            f"    {ds_name:12s}: n={stats['total']:4d}, exact={ds_exact:5.1f}%, "
            f"ROUGE-L={ds_rougeL:.3f}, METEOR={ds_meteor:.3f}, BERTScore={ds_bert:.3f}"
        )

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "model_id": args.model_id,
        "model_family": model_family,
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
        "closed_ended_yes_no": closed_ended,
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

    summary_path = os.path.join(args.output_dir, "qwen_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  Predictions : {results_jsonl}")
    print(f"  Summary     : {summary_path}")

    print()
    print("-" * 70)
    print("SAMPLE PREDICTIONS (first 5)")
    print("-" * 70)
    for r in results[:5]:
        print(f"  [{r['dataset']}] Patient: {r['patient_id']}")
        print(f"  Q: {r['post_prompt'][:100]}")
        print(f"  Gold     : {r['gold'][:100]}")
        print(f"  Generated: {r['generated_clean'][:100]}")
        print(f"  Exact={r['exact_match']}, ROUGE-L={r.get('rougeL',0):.3f}, "
              f"METEOR={r.get('meteor',0):.3f}, BERTScore={r.get('bert_f1',0):.3f}")
        print()


if __name__ == "__main__":
    main()
