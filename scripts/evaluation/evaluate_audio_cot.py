# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluate a trained AudioFlamingo model with Chain-of-Thought on the CaReSound test set.

Loads a checkpoint trained on CoT data, runs generation on every test sample, and computes:

  Standard metrics (from evaluate_audio_model.py):
  - Open-ended accuracy (exact match between generated and gold answer)
  - Normalised accuracy (case-insensitive, stripped)
  - BERTScore, ROUGE-L, METEOR
  - Closed-ended classification metrics (Accuracy, F1, Sensitivity, Specificity)

  CoT-specific metrics:
  - Rationale presence (whether reasoning text exists before "Answer:")
  - Format compliance (follows "Rationale... Answer: [label]" format)
  - Rationale length (average word count)
  - Answer extraction success rate

Results are saved to a JSONL file + a summary JSON.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_audio_cot.py \\
        --checkpoint audio_checkpoints/mel_stage2_audio_cot/best_model.pt \\
        --encoder mel \\
        --llm_id meta-llama/Llama-3.2-1B \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --cot_data_dir data/caresound_cot \\
        --max_new_tokens 512 \\
        --batch_size 1
"""

import os
import sys
import json
import argparse
import re
import time
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Closed-ended Yes/No accuracy
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

# Import audio encoders
from opentslm.model.encoder.RawAudioTokenizer import RawAudioTokenizer
from opentslm.model.encoder.MelSpectrogramTokenizer import MelSpectrogramTokenizer
from opentslm.model.encoder.Wav2Vec2Encoder import Wav2Vec2Encoder
from opentslm.model.encoder.WhisperEncoder import WhisperEncoder
from opentslm.model.encoder.CLAPEncoder import CLAPEncoder

# Import model components
from opentslm.model.projector.MLPProjector import MLPProjector
from opentslm.model.llm.AudioFlamingo import AudioFlamingo

# Import datasets
from opentslm.time_series_datasets.audio.CareSoundCoTDataset import CareSoundCoTDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

# Import config
from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.audio_model_config import AUDIO_EMBED_DIM, AUDIO_PATCH_SIZE


# ─── Helpers ──────────────────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lower-case, strip whitespace / punctuation / EOS tokens for fuzzy matching."""
    # Remove common EOS / special tokens
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)          # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()      # collapse whitespace
    return text


def clean_generated(text: str) -> str:
    """
    Clean up generated text:
    - The model often continues generating after the answer (e.g. starts a new
      'Question: ...' block).  We truncate at that boundary.
    - Remove leading/trailing whitespace.
    """
    # Truncate at follow-up question / repeated prompt patterns
    for marker in ["Question:", "question:", "\nQ:"]:
        idx = text.find(marker)
        if idx > 0:  # only if there's actual content before the marker
            text = text[:idx]
    # Also truncate at EOS-like tokens that may appear mid-text
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>"]:
        idx = text.find(tok)
        if idx >= 0:
            text = text[:idx]
    return text.strip()


def clean_gold(text: str) -> str:
    """Remove EOS tokens from gold answer."""
    for tok in ["<|end_of_text|>", "</s>", "<|eot_id|>", "<eos>"]:
        text = text.replace(tok, "")
    return text.strip()


def extract_cot_components(text: str) -> dict:
    """
    Extract rationale and answer from CoT-formatted text.

    Expected format: "[rationale paragraph] Answer: [label]"

    Returns:
        dict with keys:
        - rationale: reasoning text before "Answer:"
        - answer: extracted answer after "Answer:"
        - has_answer_marker: bool, whether "Answer:" was found
        - rationale_word_count: number of words in rationale
    """
    # Look for "Answer:" marker (case-insensitive)
    match = re.search(r'Answer\s*:\s*(.+?)(?:\.|$)', text, re.IGNORECASE | re.DOTALL)

    if match:
        # Found "Answer:" marker
        answer_start = match.start()
        rationale = text[:answer_start].strip()
        answer = match.group(1).strip()

        # Clean up answer (take only first word/phrase, stop at punctuation)
        answer = answer.split('.')[0].split('\n')[0].strip()

        return {
            "rationale": rationale,
            "answer": answer,
            "has_answer_marker": True,
            "rationale_word_count": len(rationale.split()),
        }
    else:
        # No "Answer:" marker found - entire text is rationale
        return {
            "rationale": text,
            "answer": "",
            "has_answer_marker": False,
            "rationale_word_count": len(text.split()),
        }


def count_reasoning_keywords(text: str) -> dict:
    """
    Count reasoning-related keywords in rationale to assess quality.

    Returns:
        dict with keyword category counts
    """
    text_lower = text.lower()

    # Reasoning indicators
    causal = len(re.findall(r'\b(because|since|therefore|thus|hence|consequently)\b', text_lower))
    temporal = len(re.findall(r'\b(first|second|then|next|finally|during|while)\b', text_lower))
    analysis = len(re.findall(r'\b(shows|reveals|indicates|suggests|demonstrates|consistent|pattern)\b', text_lower))
    uncertainty = len(re.findall(r'\b(may|might|could|possibly|likely|appears)\b', text_lower))
    medical = len(re.findall(r'\b(respiratory|cardiac|auscultation|diagnosis|abnormal|normal|frequency|breath)\b', text_lower))

    return {
        "causal_words": causal,
        "temporal_words": temporal,
        "analysis_words": analysis,
        "uncertainty_words": uncertainty,
        "medical_terms": medical,
        "total_reasoning_keywords": causal + temporal + analysis + uncertainty,
    }


# ─── Model creation (mirrors audio_curriculum_learning.py) ───────────────────

def create_encoder(encoder_type: str, embed_dim: int = 256, patch_size: int = 640, freeze: bool = True):
    if encoder_type == "tokenizer":
        return RawAudioTokenizer(output_dim=embed_dim, patch_size=patch_size, dropout=0.0, max_patches=768)
    elif encoder_type == "mel":
        return MelSpectrogramTokenizer(output_dim=embed_dim, sample_rate=16000, n_mels=64, dropout=0.0, max_time_frames=1000)
    elif encoder_type == "wav2vec2":
        return Wav2Vec2Encoder(output_dim=embed_dim, model_name="facebook/wav2vec2-base", freeze_encoder=freeze)
    elif encoder_type == "whisper":
        return WhisperEncoder(output_dim=embed_dim, model_name="openai/whisper-base", freeze_encoder=freeze)
    elif encoder_type == "clap":
        return CLAPEncoder(output_dim=embed_dim, model_name="laion/clap-htsat-unfused", freeze_encoder=freeze)
    else:
        raise ValueError(f"Unknown encoder: {encoder_type}")


def create_audio_model(
    encoder_type, llm_id, device, embed_dim=256, patch_size=640,
    freeze_encoder=True, num_latents=64,
    cross_attn_every_n_layers=4, lora_rank=0, lora_alpha=16.0, lora_dropout=0.0,
):
    encoder = create_encoder(encoder_type, embed_dim, patch_size, freeze=freeze_encoder)
    encoder_output_dim = encoder.get_output_dim()
    projector = MLPProjector(input_dim=encoder_output_dim, output_dim=ENCODER_OUTPUT_DIM, device=device)
    model = AudioFlamingo(
        encoder=encoder, projector=projector, device=device,
        llm_id=llm_id, vis_dim=ENCODER_OUTPUT_DIM, freeze_encoder=freeze_encoder,
        num_latents=num_latents,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    return model


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained AudioFlamingo with CoT on CaReSound test set")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--encoder", type=str, default="mel",
                        choices=["tokenizer", "mel", "wav2vec2", "whisper", "clap"])
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_merged")
    parser.add_argument("--cot_data_dir", type=str, default="data/caresound_cot",
                        help="Directory containing CoT CSV files")
    parser.add_argument("--embed_dim", type=int, default=AUDIO_EMBED_DIM)
    parser.add_argument("--patch_size", type=int, default=AUDIO_PATCH_SIZE)
    parser.add_argument("--max_audio_length", type=int, default=None,
                        help="Maximum audio length in samples (default: 480000 = 30s at 16kHz). "
                             "Must match training value for fair evaluation.")
    parser.add_argument("--num_latents", type=int, default=64,
                        help="Number of Perceiver latent tokens (must match training value)")
    parser.add_argument("--cross_attn_every_n", type=int, default=1,
                        help="Cross-attention every N layers (must match training value)")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank for cross-attention (must match training value, 0=no LoRA)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling (must match training value)")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout (can be 0 at eval time)")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (1 recommended for generation)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max tokens to generate per sample (512 for CoT, 128 for short answers)")
    parser.add_argument("--answer_only_prompt", action="store_true",
                        help="Override post_prompt to 'Question: ...\\nAnswer:' during generation. "
                             "Useful for fair short-answer comparison and parseability checks.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Enable sampling during generation (default: greedy decoding).")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (used when --do_sample is set).")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="Nucleus sampling top-p (used when --do_sample is set).")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Beam search width (default: 1 = no beam search).")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="Repetition penalty for generation (default: 1.0).")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick check)")
    parser.add_argument("--output_dir", type=str, default="audio_cot_eval_results",
                        help="Directory to save evaluation results")

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("AUDIO CoT MODEL EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint   : {args.checkpoint}")
    print(f"  Encoder      : {args.encoder}")
    print(f"  LLM          : {args.llm_id}")
    print(f"  Device       : {args.device}")
    print(f"  Max tokens   : {args.max_new_tokens}")
    print(f"  Answer-only  : {args.answer_only_prompt}")
    print(f"  Decoding     : {'sampling' if args.do_sample else 'greedy'} "
          f"(temp={args.temperature}, top_p={args.top_p}, beams={args.num_beams}, rep_pen={args.repetition_penalty})")
    print(f"  CoT data dir : {args.cot_data_dir}")
    print()

    print("Creating model...")
    model = create_audio_model(
        encoder_type=args.encoder,
        llm_id=args.llm_id,
        device=args.device,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        freeze_encoder=True,   # doesn't matter for eval
        num_latents=args.num_latents,
        cross_attn_every_n_layers=args.cross_attn_every_n,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,  # no dropout at eval time
    )

    # ── Load checkpoint ───────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)

    # The training script saves model.state_dict() directly (not wrapped in a dict)
    if isinstance(checkpoint, dict) and ("llm" in checkpoint or "model_state" in checkpoint):
        # Saved via model.store_to_file()
        model.load_from_file(args.checkpoint)
    else:
        # Saved via torch.save(model.state_dict(), ...)
        # Use strict=False to tolerate duplicate keys from model./llm. prefixed submodules
        # that may have been saved alongside the main model weights.
        missing, unexpected = model.load_state_dict(checkpoint, strict=False)
        if missing:
            print(f"⚠️  Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"ℹ️  Ignoring {len(unexpected)} unexpected checkpoint keys (e.g. duplicate model./llm. prefixed entries)")
    print("✓ Checkpoint loaded")

    model.to(args.device)
    model.eval()

    # ── Load test dataset ─────────────────────────────────────────────────────
    eos_token = model.tokenizer.eos_token
    print("Loading CaReSound CoT test set...")
    test_ds = CareSoundCoTDataset(
        split="test",
        EOS_TOKEN=eos_token,
        audio_dir=args.audio_dir,
        cot_data_dir=args.cot_data_dir,
        max_audio_length=args.max_audio_length,
    )
    total = len(test_ds) if args.max_samples is None else min(len(test_ds), args.max_samples)
    print(f"  Test samples: {total}")
    if args.max_audio_length:
        print(f"  Max audio length: {args.max_audio_length} samples ({args.max_audio_length/16000:.1f}s at 16kHz)")

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: extend_time_series_to_match_patch_size_and_aggregate(b),
        drop_last=False,
    )

    # ── Run inference ─────────────────────────────────────────────────────────
    results = []
    exact_correct = 0
    norm_correct = 0
    answer_extracted = 0
    has_rationale = 0
    total_rationale_words = 0
    per_dataset_stats = defaultdict(lambda: {
        "total": 0, "exact": 0, "norm": 0, "answer_extracted": 0, "has_rationale": 0, "rationale_words": []
    })
    samples_done = 0

    results_jsonl = os.path.join(args.output_dir, "test_predictions_cot.jsonl")
    print(f"  Results will be saved to: {results_jsonl}")
    print()

    t_start = time.time()

    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", total=min(len(test_loader), (total + args.batch_size - 1) // args.batch_size)):
                if samples_done >= total:
                    break
                try:
                    gen_batch = batch
                    if args.answer_only_prompt:
                        gen_batch = []
                        for sample in batch:
                            sample_copy = dict(sample)
                            q_full = sample_copy.get("post_prompt", "")
                            q_text = q_full
                            if "Question:" in q_text:
                                q_text = q_text.split("Question:", 1)[1]
                            for marker in ("Rationale:", "Answer:"):
                                if marker in q_text:
                                    q_text = q_text.split(marker, 1)[0]
                            q_text = q_text.strip()
                            sample_copy["post_prompt"] = f"Question: {q_text}\nAnswer:"
                            gen_batch.append(sample_copy)

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(args.device == "cuda")):
                        gen_kwargs = {
                            "do_sample": args.do_sample,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "num_beams": args.num_beams,
                            "repetition_penalty": args.repetition_penalty,
                        }
                        # OpenFlamingo wrappers may not accept all HF generation kwargs.
                        # Drop unsupported kwargs dynamically and retry.
                        while True:
                            try:
                                predictions = model.generate(
                                    gen_batch,
                                    max_new_tokens=args.max_new_tokens,
                                    **gen_kwargs,
                                )
                                break
                            except TypeError as te:
                                msg = str(te)
                                m = re.search(r"unexpected keyword argument '([^']+)'", msg)
                                if not m:
                                    raise
                                bad_kw = m.group(1)
                                if bad_kw in gen_kwargs:
                                    print(f"  ⚠ Dropping unsupported generate arg: {bad_kw}")
                                    gen_kwargs.pop(bad_kw, None)
                                    continue
                                raise
                except Exception as e:
                    print(f"  ⚠ Generation error: {e}")
                    predictions = ["[ERROR]"] * len(batch)

                for sample, pred in zip(gen_batch, predictions):
                    if samples_done >= total:
                        break

                    gold_raw = sample.get("answer", "")
                    original_answer = sample.get("original_answer", "")
                    pre_prompt = sample.get("pre_prompt", "")
                    post_prompt = sample.get("post_prompt", "")

                    # Clean up generated text
                    pred_clean = clean_generated(pred)
                    gold = clean_gold(gold_raw)

                    # Extract CoT components from generated text
                    pred_cot = extract_cot_components(pred_clean)
                    gold_cot = extract_cot_components(gold)

                    # Extract final answer from generated text
                    pred_answer = pred_cot["answer"] if pred_cot["has_answer_marker"] else pred_clean

                    # Compare against original short answer
                    exact_match = (normalise(pred_answer) == normalise(original_answer))
                    norm_match = (normalise(pred_answer) == normalise(original_answer))

                    # CoT-specific metrics
                    has_answer = pred_cot["has_answer_marker"]
                    has_reasoning = pred_cot["rationale_word_count"] > 10  # At least 10 words of reasoning

                    # Count reasoning keywords
                    reasoning_keywords = count_reasoning_keywords(pred_cot["rationale"])

                    exact_correct += int(exact_match)
                    norm_correct += int(norm_match)
                    answer_extracted += int(has_answer)
                    has_rationale += int(has_reasoning)
                    total_rationale_words += pred_cot["rationale_word_count"]

                    # Per-dataset breakdown
                    ds_name = sample.get("dataset", "unknown")
                    per_dataset_stats[ds_name]["total"] += 1
                    per_dataset_stats[ds_name]["exact"] += int(exact_match)
                    per_dataset_stats[ds_name]["norm"] += int(norm_match)
                    per_dataset_stats[ds_name]["answer_extracted"] += int(has_answer)
                    per_dataset_stats[ds_name]["has_rationale"] += int(has_reasoning)
                    per_dataset_stats[ds_name]["rationale_words"].append(pred_cot["rationale_word_count"])

                    row = {
                        "idx": samples_done,
                        "patient_id": sample.get("patient_id", ""),
                        "dataset": ds_name,
                        "question": post_prompt[:200] if post_prompt else "",
                        "generated_raw": pred,
                        "generated_clean": pred_clean,
                        "generated_rationale": pred_cot["rationale"],
                        "generated_answer": pred_answer,
                        "gold_cot": gold,
                        "gold_short": original_answer,
                        "exact_match": exact_match,
                        "norm_match": norm_match,
                        "has_answer_marker": has_answer,
                        "has_rationale": has_reasoning,
                        "rationale_word_count": pred_cot["rationale_word_count"],
                        **reasoning_keywords,
                    }
                    results.append(row)
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    samples_done += 1

                # Periodically clear GPU cache
                if samples_done % 50 == 0:
                    torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Compute BERTScore, ROUGE, METEOR ──────────────────────────────────────
    all_pred_rationales = [r["generated_rationale"] for r in results]
    all_gold_rationales = [extract_cot_components(r["gold_cot"])["rationale"] for r in results]
    all_pred_answers = [r["generated_answer"] for r in results]
    all_gold_answers = [r["gold_short"] for r in results]

    # --- ROUGE & METEOR on rationales ---
    print("\nComputing ROUGE & METEOR scores on rationales...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_rationale = []
    meteor_rationale = []
    per_dataset_rougeL = defaultdict(list)
    per_dataset_meteor = defaultdict(list)

    for r, gold_rat in zip(results, all_gold_rationales):
        # ROUGE-L on rationale
        scores = scorer.score(gold_rat, r["generated_rationale"])
        rL = scores["rougeL"].fmeasure
        rougeL_rationale.append(rL)
        r["rougeL_rationale"] = rL
        per_dataset_rougeL[r["dataset"]].append(rL)

        # METEOR on rationale
        ref_tokens = nltk.word_tokenize(gold_rat) if gold_rat else [""]
        hyp_tokens = nltk.word_tokenize(r["generated_rationale"]) if r["generated_rationale"] else [""]
        m = nltk_meteor([ref_tokens], hyp_tokens)
        meteor_rationale.append(m)
        r["meteor_rationale"] = m
        per_dataset_meteor[r["dataset"]].append(m)

    avg_rougeL_rat = sum(rougeL_rationale) / max(len(rougeL_rationale), 1)
    avg_meteor_rat = sum(meteor_rationale) / max(len(meteor_rationale), 1)

    # --- BERTScore on rationales ---
    avg_bert_f1_rat = None
    bert_f1_rat_list = [None] * len(results)
    try:
        print("Computing BERTScore on rationales (this may take a moment)...")
        max_bert_len = 512
        preds_trunc = [p[:max_bert_len] if p else " " for p in all_pred_rationales]
        golds_trunc = [g[:max_bert_len] if g else " " for g in all_gold_rationales]
        P_rat, R_rat, F1_rat = bert_score_fn(
            preds_trunc, golds_trunc,
            lang="en",
            model_type="roberta-large",
            device=args.device,
            batch_size=32,
            verbose=False,
        )
        bert_f1_rat_list = F1_rat.tolist()
        avg_bert_f1_rat = sum(bert_f1_rat_list) / max(len(bert_f1_rat_list), 1)
    except Exception as _bert_err:
        print(f"  WARNING: BERTScore failed ({_bert_err}).")
        print("  Try: pip install -U bert-score")

    per_dataset_bert = defaultdict(list)
    for r, bf1 in zip(results, bert_f1_rat_list):
        r["bert_f1_rationale"] = bf1
        if bf1 is not None:
            per_dataset_bert[r["dataset"]].append(bf1)

    # --- Closed-ended accuracy (Yes/No parsing) ---
    print("Computing closed-ended (Yes/No) accuracy...")
    all_questions = [r["question"] for r in results]
    closed_ended = compute_closed_ended_accuracy(all_gold_answers, all_pred_answers, questions=all_questions)

    from label_utils import parse_yes_no
    for r in results:
        r["closed_ended_gold_yn"] = parse_yes_no(r["gold_short"])
        r["closed_ended_pred_yn"] = parse_yes_no(r["generated_answer"])
        r["closed_ended_is_binary_q"] = is_binary_yn_question(r["question"])

    # Re-write JSONL with all metric columns added
    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    exact_acc = exact_correct / max(samples_done, 1) * 100
    norm_acc = norm_correct / max(samples_done, 1) * 100
    answer_extraction_rate = answer_extracted / max(samples_done, 1) * 100
    rationale_presence_rate = has_rationale / max(samples_done, 1) * 100
    avg_rationale_words = total_rationale_words / max(samples_done, 1)

    print()
    print("=" * 60)
    print("CoT EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples evaluated : {samples_done}")
    print(f"  Time                    : {elapsed:.1f}s ({elapsed/max(samples_done,1):.2f}s/sample)")
    print()
    print("  Answer accuracy metrics:")
    print(f"    Exact match accuracy    : {exact_correct}/{samples_done} = {exact_acc:.2f}%")
    print(f"    Normalised accuracy     : {norm_correct}/{samples_done} = {norm_acc:.2f}%")
    print()
    print("  CoT format metrics:")
    print(f"    Answer extraction rate  : {answer_extracted}/{samples_done} = {answer_extraction_rate:.2f}%")
    print(f"    Rationale presence rate : {has_rationale}/{samples_done} = {rationale_presence_rate:.2f}%")
    print(f"    Avg rationale length    : {avg_rationale_words:.1f} words")
    print()
    print("  Rationale quality (semantic similarity with gold CoT):")
    bert_rat_str = f"{avg_bert_f1_rat:.4f}" if avg_bert_f1_rat is not None else "not computed"
    print(f"    BERTScore F1            : {bert_rat_str}")
    print(f"    ROUGE-L F1              : {avg_rougeL_rat:.4f}")
    print(f"    METEOR                  : {avg_meteor_rat:.4f}")
    print()
    print("  Closed-ended (Yes/No) binary metrics:")
    print(f"    Accuracy                : {closed_ended['accuracy']:.4f}")
    print(f"    F1 (macro)              : {closed_ended['f1_macro']:.4f}")
    print(f"    F1 (weighted)           : {closed_ended['f1_weighted']:.4f}")
    print(f"    Sensitivity             : {closed_ended['sensitivity']:.4f}")
    print(f"    Specificity             : {closed_ended['specificity']:.4f}")
    print(f"    Evaluated: {closed_ended['evaluated']} yes/no samples "
          f"({closed_ended['skipped_not_binary_question']} non-binary Q skipped, "
          f"{closed_ended['skipped_not_yes_no']} non-yes/no gold skipped, "
          f"{closed_ended['pred_unparseable']} pred unparseable)")
    print(f"    Gold: {closed_ended['label_counts']['gold_yes']} Yes / {closed_ended['label_counts']['gold_no']} No  |  "
          f"Pred: {closed_ended['label_counts']['pred_yes']} Yes / {closed_ended['label_counts']['pred_no']} No")
    print()
    print("  Per-dataset breakdown:")
    for ds_name in sorted(per_dataset_stats.keys()):
        stats = per_dataset_stats[ds_name]
        ds_exact = stats["exact"] / max(stats["total"], 1) * 100
        ds_answer_ext = stats["answer_extracted"] / max(stats["total"], 1) * 100
        ds_has_rat = stats["has_rationale"] / max(stats["total"], 1) * 100
        ds_avg_words = sum(stats["rationale_words"]) / max(len(stats["rationale_words"]), 1)
        ds_rougeL = sum(per_dataset_rougeL[ds_name]) / max(len(per_dataset_rougeL[ds_name]), 1)
        ds_bert = (sum(per_dataset_bert[ds_name]) / len(per_dataset_bert[ds_name])) if per_dataset_bert[ds_name] else None
        bert_ds_str = f"{ds_bert:.3f}" if ds_bert is not None else "null"
        print(f"    {ds_name:12s}: n={stats['total']:4d}, acc={ds_exact:5.1f}%, "
              f"ans_ext={ds_answer_ext:5.1f}%, rat={ds_has_rat:5.1f}%, "
              f"words={ds_avg_words:4.0f}, ROUGE={ds_rougeL:.3f}, BERT={bert_ds_str}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        "llm_id": args.llm_id,
        "cot_data_dir": args.cot_data_dir,
        "max_audio_length": args.max_audio_length,
        "max_audio_seconds": args.max_audio_length / 16000 if args.max_audio_length else None,
        "max_new_tokens": args.max_new_tokens,
        "total_samples": samples_done,
        "answer_accuracy": exact_acc,
        "normalised_accuracy": norm_acc,
        "answer_extraction_rate": answer_extraction_rate,
        "rationale_presence_rate": rationale_presence_rate,
        "avg_rationale_words": avg_rationale_words,
        "rationale_bert_f1": avg_bert_f1_rat,
        "rationale_rougeL": avg_rougeL_rat,
        "rationale_meteor": avg_meteor_rat,
        "careqa_closed_ended": closed_ended,
        "elapsed_seconds": elapsed,
        "per_dataset": {},
    }

    for ds in sorted(per_dataset_stats.keys()):
        summary["per_dataset"][ds] = {
            "total": per_dataset_stats[ds]["total"],
            "answer_accuracy": per_dataset_stats[ds]["exact"] / max(per_dataset_stats[ds]["total"], 1) * 100,
            "answer_extraction_rate": per_dataset_stats[ds]["answer_extracted"] / max(per_dataset_stats[ds]["total"], 1) * 100,
            "rationale_presence_rate": per_dataset_stats[ds]["has_rationale"] / max(per_dataset_stats[ds]["total"], 1) * 100,
            "avg_rationale_words": sum(per_dataset_stats[ds]["rationale_words"]) / max(len(per_dataset_stats[ds]["rationale_words"]), 1),
            "rougeL_rationale": sum(per_dataset_rougeL[ds]) / max(len(per_dataset_rougeL[ds]), 1),
            "bert_f1_rationale": (sum(per_dataset_bert[ds]) / len(per_dataset_bert[ds])) if per_dataset_bert[ds] else None,
        }

    summary_path = os.path.join(args.output_dir, "cot_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  Predictions : {results_jsonl}")
    print(f"  Summary     : {summary_path}")

    # ── Show a few examples ───────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("SAMPLE PREDICTIONS (first 3)")
    print("-" * 60)
    for r in results[:3]:
        print(f"  Q: {r['question'][:80]}...")
        print(f"  Gold answer: {r['gold_short']}")
        print(f"  Generated answer: {r['generated_answer']}")
        print(f"  Generated rationale (first 150 chars): {r['generated_rationale'][:150]}...")
        bert_sample = r.get('bert_f1_rationale')
        bert_sample_str = f"{bert_sample:.3f}" if bert_sample is not None else "null"
        print(f"  Metrics: match={r['exact_match']}, has_answer={r['has_answer_marker']}, "
              f"words={r['rationale_word_count']}, BERT={bert_sample_str}")
        print()


if __name__ == "__main__":
    main()
