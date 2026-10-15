# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Evaluate a trained AudioFlamingo model on the CaReSound test set.

This file:
  Helpers:  normalise, clean_generated, clean_gold, extract_first_sentence
  Model:    create_encoder, create_audio_model
  main() pipeline:
    1. Load model + checkpoint
    2. Inference → per-sample predictions
    3. Open-ended metrics (exact/norm/contains match, ROUGE-L, METEOR, BERTScore)
    4. Closed-ended Yes/No accuracy (overall + per-dataset)  — see label_utils.py
    5. Save test_predictions.jsonl + eval_summary.json

label_utils.py (imported):
  Yes/No evaluation via 5-step pipeline:
    1. Filter to binary questions   (is_binary_yn_question)
    2. Filter to valid gold labels  (parse_yes_no on gold)
    3. Parse model predictions      (parse_yes_no on pred → 1/0/-1)
    4. Score — unparseable = WRONG
    5. Compute accuracy, F1, sensitivity, specificity (sklearn)

Metrics:
  Open-ended:  exact_match, norm_match, contains_match, BERTScore, ROUGE-L, METEOR
  Closed-ended: accuracy_all (unparseable=wrong), accuracy_parseable, parseable_rate,
                F1 macro/weighted, sensitivity, specificity
  Per-dataset: all above per dataset, saved in eval_summary.json → per_dataset

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/evaluation/evaluate_audio_model.py \\
        --checkpoint audio_checkpoints/mel_stage1_caresound_qa/best_model.pt \\
        --encoder mel \\
        --llm_id meta-llama/Llama-3.2-1B \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --max_new_tokens 128 \\
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
from label_utils import compute_closed_ended_accuracy

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
from opentslm.time_series_datasets.audio.CareSoundDataset import CareSoundDataset
from opentslm.time_series_datasets.audio.CirCorQAEvalDataset import CirCorQAEvalDataset
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
    for marker in ["Question:", "question:", "\nQ:", "\nAnswer:"]:
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


def extract_first_sentence(text: str) -> str:
    """Take only the first sentence (up to period / newline) for comparison."""
    for sep in [".", "\n"]:
        idx = text.find(sep)
        if idx != -1:
            text = text[:idx]
    return text.strip()


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


def create_audio_model(encoder_type, llm_id, device, embed_dim=256, patch_size=640,
                       freeze_encoder=True, num_latents=64,
                       cross_attn_every_n_layers=4, lora_rank=0,
                       lora_alpha=16.0, lora_dropout=0.0):
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
    parser = argparse.ArgumentParser(description="Evaluate trained AudioFlamingo on CaReSound test set")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--encoder", type=str, default="mel",
                        choices=["tokenizer", "mel", "wav2vec2", "whisper", "clap"])
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_merged")
    parser.add_argument("--dataset", type=str, default="caresound",
                        choices=["caresound", "circor"],
                        help="Which dataset to evaluate on")
    parser.add_argument("--circor_qa_jsonl", type=str, default="data/circor_qa/circor_qa.jsonl",
                        help="Path to circor_qa.jsonl (used when --dataset circor)")
    parser.add_argument("--circor_audio_dir", type=str, default="data/circor_qa/audio",
                        help="Path to CirCor audio dir (used when --dataset circor)")
    parser.add_argument("--circor_tiers", type=str, default=None,
                        help="Comma-separated tiers to evaluate, e.g. Detection,Localization "
                             "(default: all tiers)")
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
                        help="LoRA rank for cross-attention (must match training value, 0=disabled)")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling (must match training value)")

    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for evaluation (1 recommended for generation)")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate per sample")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of test samples (for quick check)")
    parser.add_argument("--output_dir", type=str, default="audio_eval_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--gpu_mem_fraction", type=float, default=None,
                        help="Optional CUDA per-process memory cap in [0, 1]. "
                             "Example: --gpu_mem_fraction 0.8")
    parser.add_argument("--gpu_mem_device", type=int, default=0,
                        help="CUDA device index used for --gpu_mem_fraction "
                             "(default: 0). Example: --gpu_mem_device 3")

    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device)

    args = parser.parse_args()

    # Optional per-process CUDA memory cap (helps avoid contention with other users/jobs)
    if args.gpu_mem_fraction is not None:
        if not (0.0 < args.gpu_mem_fraction <= 1.0):
            raise ValueError("--gpu_mem_fraction must be in the range (0, 1].")
        if not torch.cuda.is_available():
            print("⚠ --gpu_mem_fraction was provided but CUDA is not available; skipping.")
        else:
            torch.cuda.set_per_process_memory_fraction(
                args.gpu_mem_fraction,
                device=args.gpu_mem_device,
            )
            print(
                f"✓ CUDA per-process memory fraction set to {args.gpu_mem_fraction:.3f} "
                f"on device index {args.gpu_mem_device}"
            )

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Build model ───────────────────────────────────────────────────────────
    print("=" * 60)
    print("AUDIO MODEL EVALUATION")
    print("=" * 60)
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Encoder    : {args.encoder}")
    print(f"  LLM        : {args.llm_id}")
    print(f"  Device     : {args.device}")
    print(f"  Max tokens : {args.max_new_tokens}")
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
        model.load_state_dict(checkpoint)
    print("✓ Checkpoint loaded")

    model.to(args.device)
    model.eval()

    # ── Load test dataset ─────────────────────────────────────────────────────
    eos_token = model.tokenizer.eos_token
    if args.dataset == "circor":
        tiers = [t.strip() for t in args.circor_tiers.split(",")] if args.circor_tiers else None
        print(f"Loading CirCor QA eval set from {args.circor_qa_jsonl} ...")
        test_ds = CirCorQAEvalDataset(
            split="test",
            EOS_TOKEN=eos_token,
            qa_jsonl_path=args.circor_qa_jsonl,
            audio_dir=args.circor_audio_dir,
            tiers=tiers,
        )
    else:
        print("Loading CaReSound test set...")
        test_ds = CareSoundDataset(
            split="test", EOS_TOKEN=eos_token, audio_dir=args.audio_dir,
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
    per_dataset_stats = defaultdict(lambda: {"total": 0, "exact": 0, "norm": 0})
    samples_done = 0

    results_jsonl = os.path.join(args.output_dir, "test_predictions.jsonl")
    print(f"  Results will be saved to: {results_jsonl}")
    print()

    t_start = time.time()

    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", total=min(len(test_loader), (total + args.batch_size - 1) // args.batch_size)):
                if samples_done >= total:
                    break
                try:
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(args.device == "cuda")):
                        predictions = model.generate(batch, max_new_tokens=args.max_new_tokens)
                except Exception as e:
                    print(f"  ⚠ Generation error: {e}")
                    predictions = ["[ERROR]"] * len(batch)

                for sample, pred in zip(batch, predictions):
                    if samples_done >= total:
                        break

                    gold_raw = sample.get("answer", "")
                    pre_prompt = sample.get("pre_prompt", "")
                    post_prompt = sample.get("post_prompt", "")

                    # Clean up both sides
                    pred_clean = clean_generated(pred)
                    gold = clean_gold(gold_raw)

                    # Exact match (after cleaning)
                    exact_match = (pred_clean.strip() == gold.strip())
                    # Normalised match (case-insensitive, punctuation removed)
                    norm_match = (normalise(pred_clean) == normalise(gold))
                    # First-sentence match (gold is often a single sentence)
                    first_sent_match = (normalise(extract_first_sentence(pred_clean)) == normalise(gold))
                    # Also check if gold is contained in prediction
                    contains_match = (normalise(gold) in normalise(pred_clean))

                    exact_correct += int(exact_match)
                    norm_correct += int(norm_match or first_sent_match)
                    contains_correct = int(contains_match)

                    # Per-dataset breakdown
                    ds_name = "unknown"
                    for known_ds in ["ICBHI", "CirCor", "SPRSound", "ZCHSound", "KAUH"]:
                        if known_ds.lower() in pre_prompt.lower():
                            ds_name = known_ds
                            break
                    per_dataset_stats[ds_name]["total"] += 1
                    per_dataset_stats[ds_name]["exact"] += int(exact_match)
                    per_dataset_stats[ds_name]["norm"] += int(norm_match or first_sent_match)
                    per_dataset_stats[ds_name]["contains"] = per_dataset_stats[ds_name].get("contains", 0) + contains_correct

                    row = {
                        "idx": samples_done,
                        "pre_prompt": pre_prompt,
                        "post_prompt": post_prompt,
                        "generated_raw": pred,
                        "generated_clean": pred_clean,
                        "gold": gold,
                        "exact_match": exact_match,
                        "norm_match": norm_match or first_sent_match,
                        "contains_match": contains_match,
                        "dataset": ds_name,
                    }
                    results.append(row)
                    f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    samples_done += 1

                # Periodically clear GPU cache
                if samples_done % 50 == 0:
                    torch.cuda.empty_cache()

    elapsed = time.time() - t_start

    # ── Compute BERTScore & ROUGE ─────────────────────────────────────────────
    all_preds = [r["generated_clean"] for r in results]
    all_golds = [r["gold"] for r in results]

    # --- ROUGE & METEOR ---
    print("\nComputing ROUGE & METEOR scores...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL_scores = []
    meteor_scores = []
    per_dataset_rougeL = defaultdict(list)
    per_dataset_meteor = defaultdict(list)

    for r in results:
        # ROUGE-L
        scores = scorer.score(r["gold"], r["generated_clean"])
        rL = scores["rougeL"].fmeasure
        rougeL_scores.append(rL)
        r["rougeL"] = rL
        per_dataset_rougeL[r["dataset"]].append(rL)

        # METEOR (expects tokenised word lists)
        ref_tokens = nltk.word_tokenize(r["gold"])
        hyp_tokens = nltk.word_tokenize(r["generated_clean"])
        m = nltk_meteor([ref_tokens], hyp_tokens)
        meteor_scores.append(m)
        r["meteor"] = m
        per_dataset_meteor[r["dataset"]].append(m)

    avg_rougeL = sum(rougeL_scores) / max(len(rougeL_scores), 1)
    avg_meteor = sum(meteor_scores) / max(len(meteor_scores), 1)

    # --- BERTScore ---
    print("Computing BERTScore (this may take a moment)...")
    # Truncate texts to avoid tokenizer overflow issues
    max_bert_len = 512  # characters — keeps within model's token limit
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

    # --- Closed-ended accuracy (Yes/No parsing) ---
    print("Computing closed-ended (Yes/No) accuracy...")
    careqa_closed = compute_closed_ended_accuracy(
        all_golds, all_preds,
        questions=[r["post_prompt"] for r in results],
    )

    # Store per-result CareAQA Yes/No flags
    from label_utils import parse_yes_no, is_binary_yn_question, extract_question_from_post_prompt
    for r in results:
        r["careqa_gold_yn"] = parse_yes_no(r["gold"])
        r["careqa_pred_yn"] = parse_yes_no(r["generated_clean"])

    # Per-dataset closed-ended (Yes/No) accuracy (including unparseable as wrong)
    per_dataset_yn = defaultdict(lambda: {
        "total_binary_qs": 0, "parseable": 0, "correct": 0,
        "unparseable": 0, "gold_yes": 0, "gold_no": 0,
    })
    for r in results:
        q_text = extract_question_from_post_prompt(r["post_prompt"])
        if not is_binary_yn_question(q_text):
            continue
        gold_yn = r["careqa_gold_yn"]
        pred_yn = r["careqa_pred_yn"]
        if gold_yn == -1:
            continue  # gold is not yes/no → skip

        ds = r["dataset"]
        d = per_dataset_yn[ds]
        d["total_binary_qs"] += 1
        if gold_yn == 1:
            d["gold_yes"] += 1
        else:
            d["gold_no"] += 1
        if pred_yn == -1:
            d["unparseable"] += 1
        else:
            d["parseable"] += 1
            if pred_yn == gold_yn:
                d["correct"] += 1

    # Re-write JSONL with all metric columns added
    with open(results_jsonl, "w", encoding="utf-8") as f_out:
        for r in results:
            f_out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    exact_acc = exact_correct / max(samples_done, 1) * 100
    norm_acc = norm_correct / max(samples_done, 1) * 100
    total_contains = sum(s.get("contains", 0) for s in per_dataset_stats.values())
    contains_acc = total_contains / max(samples_done, 1) * 100

    print()
    print("=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Total samples evaluated : {samples_done}")
    print(f"  Time                    : {elapsed:.1f}s ({elapsed/max(samples_done,1):.2f}s/sample)")
    print()
    print("  Accuracy metrics:")
    print(f"    Open-ended accuracy     : {exact_correct}/{samples_done} = {exact_acc:.2f}%")
    print(f"    Normalised accuracy     : {norm_correct}/{samples_done} = {norm_acc:.2f}%")
    print(f"    Contains-match accuracy : {total_contains}/{samples_done} = {contains_acc:.2f}%")
    print()
    print("  Semantic / n-gram metrics:")
    print(f"    BERTScore F1            : {avg_bert_f1:.4f}  (P={avg_bert_p:.4f}, R={avg_bert_r:.4f})")
    print(f"    ROUGE-L F1              : {avg_rougeL:.4f}")
    print(f"    METEOR                  : {avg_meteor:.4f}")
    print()
    print("  Closed-ended (Yes/No) binary metrics:")
    print(f"    Accuracy                : {careqa_closed['accuracy']:.4f}")
    print(f"    F1 (macro)              : {careqa_closed['f1_macro']:.4f}")
    print(f"    F1 (weighted)           : {careqa_closed['f1_weighted']:.4f}")
    print(f"    Sensitivity             : {careqa_closed['sensitivity']:.4f}")
    print(f"    Specificity             : {careqa_closed['specificity']:.4f}")
    print(f"    Evaluated: {careqa_closed['correct']}/{careqa_closed['evaluated']} yes/no samples "
          f"({careqa_closed.get('skipped_not_binary_question', 0)} non-binary Q skipped, "
          f"{careqa_closed['skipped_not_yes_no']} non-yes/no gold skipped, "
          f"{careqa_closed['pred_unparseable']} pred unparseable)")
    yn = careqa_closed['label_counts']
    print(f"    Gold: {yn['gold_yes']} Yes / {yn['gold_no']} No  |  "
          f"Pred: {yn['pred_yes']} Yes / {yn['pred_no']} No")
    print()
    print("  Per-dataset breakdown (open-ended):")
    for ds_name in sorted(set(list(per_dataset_stats.keys()) + list(per_dataset_rougeL.keys()))):
        stats = per_dataset_stats[ds_name]
        ds_exact = stats["exact"] / max(stats["total"], 1) * 100
        ds_norm = stats["norm"] / max(stats["total"], 1) * 100
        ds_contains = stats.get("contains", 0) / max(stats["total"], 1) * 100
        ds_rougeL = sum(per_dataset_rougeL[ds_name]) / max(len(per_dataset_rougeL[ds_name]), 1)
        ds_meteor = sum(per_dataset_meteor[ds_name]) / max(len(per_dataset_meteor[ds_name]), 1)
        ds_bert = sum(per_dataset_bert[ds_name]) / max(len(per_dataset_bert[ds_name]), 1)
        print(f"    {ds_name:12s}: n={stats['total']:4d}, open_ended={ds_exact:5.1f}%, ROUGE-L={ds_rougeL:.3f}, METEOR={ds_meteor:.3f}, BERTScore={ds_bert:.3f}")

    print()
    print("  Per-dataset breakdown (Yes/No closed-ended):")
    for ds_name in sorted(per_dataset_yn.keys()):
        yn = per_dataset_yn[ds_name]
        total_q = yn["total_binary_qs"]
        if total_q == 0:
            continue
        acc_all = yn["correct"] / total_q * 100  # includes unparseable as wrong
        parseable_rate = yn["parseable"] / total_q * 100
        acc_parseable = yn["correct"] / max(yn["parseable"], 1) * 100
        print(f"    {ds_name:12s}: binary_qs={total_q:4d}, "
              f"acc(all)={acc_all:5.1f}%, "
              f"acc(parseable)={acc_parseable:5.1f}%, "
              f"parseable={yn['parseable']}/{total_q} ({parseable_rate:.0f}%), "
              f"unparseable={yn['unparseable']}")

    # ── Save summary ──────────────────────────────────────────────────────────
    summary = {
        "checkpoint": args.checkpoint,
        "encoder": args.encoder,
        "llm_id": args.llm_id,
        "max_audio_length": args.max_audio_length,
        "max_audio_seconds": args.max_audio_length / 16000 if args.max_audio_length else None,
        "total_samples": samples_done,
        "open_ended_accuracy": exact_acc,
        "normalised_accuracy": norm_acc,
        "contains_match_accuracy": contains_acc,
        "bert_score_f1": avg_bert_f1,
        "bert_score_precision": avg_bert_p,
        "bert_score_recall": avg_bert_r,
        "rougeL_f1": avg_rougeL,
        "meteor": avg_meteor,
        "closed_ended_yes_no": careqa_closed,
        "elapsed_seconds": elapsed,
        "max_new_tokens": args.max_new_tokens,
        "per_dataset": {
            ds: {
                "total": per_dataset_stats[ds]["total"],
                "open_ended_acc": per_dataset_stats[ds]["exact"] / max(per_dataset_stats[ds]["total"], 1) * 100,
                "norm_acc": per_dataset_stats[ds]["norm"] / max(per_dataset_stats[ds]["total"], 1) * 100,
                "contains_match_acc": per_dataset_stats[ds].get("contains", 0) / max(per_dataset_stats[ds]["total"], 1) * 100,
                "rougeL_f1": sum(per_dataset_rougeL[ds]) / max(len(per_dataset_rougeL[ds]), 1),
                "meteor": sum(per_dataset_meteor[ds]) / max(len(per_dataset_meteor[ds]), 1),
                "bert_f1": sum(per_dataset_bert[ds]) / max(len(per_dataset_bert[ds]), 1),
                "yes_no": {
                    "total_binary_qs": per_dataset_yn[ds]["total_binary_qs"],
                    "correct": per_dataset_yn[ds]["correct"],
                    "parseable": per_dataset_yn[ds]["parseable"],
                    "unparseable": per_dataset_yn[ds]["unparseable"],
                    "accuracy_all": per_dataset_yn[ds]["correct"] / max(per_dataset_yn[ds]["total_binary_qs"], 1) * 100,
                    "accuracy_parseable": per_dataset_yn[ds]["correct"] / max(per_dataset_yn[ds]["parseable"], 1) * 100,
                    "parseable_rate": per_dataset_yn[ds]["parseable"] / max(per_dataset_yn[ds]["total_binary_qs"], 1) * 100,
                    "gold_yes": per_dataset_yn[ds]["gold_yes"],
                    "gold_no": per_dataset_yn[ds]["gold_no"],
                } if per_dataset_yn[ds]["total_binary_qs"] > 0 else None,
            }
            for ds in sorted(per_dataset_stats.keys())
        },
    }

    summary_path = os.path.join(args.output_dir, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"  Predictions : {results_jsonl}")
    print(f"  Summary     : {summary_path}")

    # ── Show a few examples ───────────────────────────────────────────────────
    print()
    print("-" * 60)
    print("SAMPLE PREDICTIONS (first 5)")
    print("-" * 60)
    for r in results[:5]:
        print(f"  Q: {r['post_prompt'][:100]}")
        print(f"  Gold:      {r['gold'][:100]}")
        print(f"  Generated: {r['generated_clean'][:100]}")
        print(f"  Match: exact={r['exact_match']}, ROUGE-L={r['rougeL']:.3f}, METEOR={r['meteor']:.3f}, BERTScore={r['bert_f1']:.3f}")
        print()


if __name__ == "__main__":
    main()
