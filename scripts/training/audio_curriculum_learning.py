# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Audio Curriculum Learning Script

Train an audio language model using curriculum learning with AudioFlamingo.

Encoder options:
- tokenizer: RawAudioTokenizer (light, trainable, RECOMMENDED)
- mel: MelSpectrogramTokenizer (light, trainable)
- wav2vec2: Wav2Vec2Encoder (pretrained, frozen)
- whisper: WhisperEncoder (pretrained, frozen)
- clap: CLAPEncoder (LAION CLAP HTSAT, pretrained, frozen)

Training stages:
1. Medical Audio QA (CaReSound) - Cardiac and respiratory sound diagnosis


Usage:
    # Train with RawAudioTokenizer (recommended)
    python scripts/training/audio_curriculum_learning.py \\
        --encoder tokenizer \\
        --audio_dir ./data/caresound_audio/audio_merged \\
        --llm_id meta-llama/Llama-3.2-1B

    # Train with Wav2Vec2 (pretrained)
    python scripts/training/audio_curriculum_learning.py \\
        --encoder wav2vec2 \\
        --audio_dir ./data/caresound_audio/audio_merged

Requirements:
    - Install: pip install -e .
    - Prepare audio: python scripts/data_prep/audio_data_preparation.py --merge_sprsound
"""

import os
import sys
import argparse
import logging
import traceback
import datetime
import time
from typing import List

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # Use filesystem instead of /dev/shm (avoids shm exhaustion)
from threading import Thread
from queue import Queue
from torch.optim import AdamW
from collections import Counter
from torch.utils.data import DataLoader, WeightedRandomSampler, ConcatDataset
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm


class PrefetchLoader:
    """Thread-based prefetch wrapper for DataLoader.

    Loads the next batch in a background thread while the GPU processes the
    current one.  Uses NO multiprocessing and NO /dev/shm — works even when
    shared memory is tiny (e.g. 64 MB Docker default).

    Because the heavy work (reading audio from disk) releases the Python GIL,
    and CUDA kernels also release the GIL, real overlap happens automatically.
    """

    def __init__(self, loader, prefetch: int = 2):
        self.loader = loader
        self.prefetch = prefetch

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        queue: Queue = Queue(maxsize=self.prefetch)

        def _producer():
            for batch in self.loader:
                queue.put(batch)
            queue.put(None)  # sentinel

        thread = Thread(target=_producer, daemon=True)
        thread.start()

        while True:
            item = queue.get()
            if item is None:
                break
            yield item

        thread.join()

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
from opentslm.time_series_datasets.audio.CareSoundCoTDataset import CareSoundCoTDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate

# Import audio configuration
from opentslm.audio_model_config import (
    AUDIO_BATCH_SIZE,
    AUDIO_NUM_EPOCHS,
    AUDIO_EARLY_STOP_PAT,
    AUDIO_LR_ENCODER,
    AUDIO_LR_PROJECTOR,
    AUDIO_LR_OTHER,
    AUDIO_WEIGHT_DECAY,
    AUDIO_GRAD_CLIP_NORM,
    AUDIO_WARMUP_FRAC,
    AUDIO_PATCH_SIZE,
    AUDIO_EMBED_DIM,
    AUDIO_ENCODER_MODEL,
    AUDIO_FREEZE_ENCODER,
    AUDIO_DROPOUT,
    AUDIO_CROSS_ATTN_EVERY_N,
    AUDIO_LORA_RANK,
    # Stage 2: CoT fine-tuning
    AUDIO_COT_LR,
    AUDIO_COT_NUM_EPOCHS,
    AUDIO_COT_EARLY_STOP_PAT,
    AUDIO_COT_LABEL_SMOOTHING,
    AUDIO_COT_DROPOUT,
)

# Global model configuration (required to set Flamingo vis_dim)
from opentslm.model_config import ENCODER_OUTPUT_DIM


# ─── Logging setup ──────────────────────────────────────────────────────────────

def setup_logging(log_file: str = None, encoder_name: str = "unknown"):
    """Setup logging to both console and a file so logs persist after crashes."""
    if log_file is None:
        os.makedirs("audio_logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"audio_logs/train_{encoder_name}_{timestamp}.log"

    # Create logger
    logger = logging.getLogger("audio_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — captures everything, even if process is killed
    fh = logging.FileHandler(log_file, mode="w")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
    logger.addHandler(fh)

    # Console handler
    # Use stderr for console logs to reduce broken-pipe risk when stdout is piped.
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_file}")
    return logger, log_file


def should_disable_tqdm() -> bool:
    """Disable tqdm only when explicitly requested via env var.

    tqdm works in both TTY (in-place bar) and non-TTY (one line per update)
    mode, so we no longer suppress it based on isatty().
    Set AUDIO_TQDM_DISABLE=1 to suppress entirely.
    """
    env_val = os.getenv("AUDIO_TQDM_DISABLE", "").strip().lower()
    return env_val in {"1", "true", "yes", "on"}


def log_gpu_memory(logger, tag: str = ""):
    """Log current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.debug(
            f"GPU [{tag}] allocated={allocated:.2f}GB, reserved={reserved:.2f}GB, "
            f"peak={max_allocated:.2f}GB, total={total:.1f}GB"
        )
        return allocated, reserved, max_allocated, total
    return 0, 0, 0, 0


def log_system_memory(logger, tag: str = ""):
    """Log system RAM usage."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.debug(
            f"RAM [{tag}] used={mem.used / (1024**3):.1f}GB / {mem.total / (1024**3):.1f}GB "
            f"({mem.percent}%)"
        )
    except ImportError:
        pass  # psutil not installed, skip


# ─── Audio curriculum stages ────────────────────────────────────────────────────

AUDIO_CURRICULUM_STAGES = [
    "stage1_caresound_qa",
    "stage2_audio_cot",
    "stage3_audio_captioning",
    "stage4_speech_recognition",
]


def get_audio_datasets(
    stage: str,
    eos_token: str,
    audio_dir: str = None,
    max_audio_length: int = None,
    stage2_mix_qa: bool = False,
):
    """
    Get datasets for a specific audio curriculum stage.

    Args:
        stage: Stage name (e.g., "stage1_caresound_qa", "stage2_audio_cot")
        eos_token: End-of-sequence token from tokenizer
        audio_dir: Directory containing audio files (required for CaReSound)
        max_audio_length: Maximum audio length in samples (default: 480000 = 30s at 16kHz)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    if stage == "stage1_caresound_qa":
        if audio_dir is None:
            raise ValueError(
                "audio_dir is required for CaReSound dataset.\n"
                "Use: --audio_dir ./data/caresound_audio/audio_merged\n"
                "First prepare data: python scripts/data_prep/audio_data_preparation.py --merge_sprsound"
            )
        train_ds = CareSoundDataset(split="train", EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)
        val_ds = CareSoundDataset(split="val", EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)
        test_ds = CareSoundDataset(split="test", EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)

    elif stage == "stage2_audio_cot":
        if audio_dir is None:
            raise ValueError(
                "audio_dir is required for CaReSound CoT dataset.\n"
                "Use: --audio_dir ./data/caresound_audio/audio_merged\n"
                "First generate CoT: python scripts/data_prep/generate_audio_cot.py"
            )
        cot_train_ds = CareSoundCoTDataset(
            split="train",
            EOS_TOKEN=eos_token,
            audio_dir=audio_dir,
            cot_data_dir="data/caresound_cot",
            max_audio_length=max_audio_length
        )
        val_ds = CareSoundCoTDataset(
            split="val",
            EOS_TOKEN=eos_token,
            audio_dir=audio_dir,
            cot_data_dir="data/caresound_cot",
            max_audio_length=max_audio_length
        )
        test_ds = CareSoundCoTDataset(
            split="test",
            EOS_TOKEN=eos_token,
            audio_dir=audio_dir,
            cot_data_dir="data/caresound_cot",
            max_audio_length=max_audio_length
        )
        if stage2_mix_qa:
            qa_train_ds = CareSoundDataset(
                split="train",
                EOS_TOKEN=eos_token,
                audio_dir=audio_dir,
                max_audio_length=max_audio_length,
            )
            train_ds = ConcatDataset([qa_train_ds, cot_train_ds])
        else:
            train_ds = cot_train_ds

    elif stage == "joint_qa_cot":
        if audio_dir is None:
            raise ValueError(
                "audio_dir is required for joint QA+CoT dataset.\n"
                "Use: --audio_dir ./data/caresound_audio/audio_merged"
            )
        qa_train = CareSoundDataset(split="train", EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)
        qa_val   = CareSoundDataset(split="val",   EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)
        qa_test  = CareSoundDataset(split="test",  EOS_TOKEN=eos_token, audio_dir=audio_dir, max_audio_length=max_audio_length)
        cot_train = CareSoundCoTDataset(split="train", EOS_TOKEN=eos_token, audio_dir=audio_dir, cot_data_dir="data/caresound_cot", max_audio_length=max_audio_length)
        cot_val   = CareSoundCoTDataset(split="val",   EOS_TOKEN=eos_token, audio_dir=audio_dir, cot_data_dir="data/caresound_cot", max_audio_length=max_audio_length)
        cot_test  = CareSoundCoTDataset(split="test",  EOS_TOKEN=eos_token, audio_dir=audio_dir, cot_data_dir="data/caresound_cot", max_audio_length=max_audio_length)
        train_ds = ConcatDataset([qa_train, cot_train])
        val_ds   = ConcatDataset([qa_val,   cot_val])
        test_ds  = ConcatDataset([qa_test,  cot_test])

    else:
        raise ValueError(
            f"Unknown stage: {stage}. "
            "Available: stage1_caresound_qa, stage2_audio_cot, joint_qa_cot, stage3_audio_captioning, stage4_speech_recognition"
        )
    return train_ds, val_ds, test_ds


# ─── Encoder & Model creation ───────────────────────────────────────────────────

def create_encoder(encoder_type: str, embed_dim: int = 256, patch_size: int = 640, freeze: bool = True):
    """
    Create an audio encoder based on the specified type.

    Args:
        encoder_type: Type of encoder ('tokenizer', 'mel', 'wav2vec2', 'whisper', 'clap')
        embed_dim: Output embedding dimension (for trainable encoders)
        patch_size: Patch size for RawAudioTokenizer
        freeze: Whether to freeze pretrained encoders

    Returns:
        Encoder instance
    """
    if encoder_type == "tokenizer":
        print(f"Creating RawAudioTokenizer (trainable, embed_dim={embed_dim})")
        encoder = RawAudioTokenizer(
            output_dim=embed_dim,
            patch_size=patch_size,
            dropout=0.1,
            max_patches=768,  # 30s@16kHz = 750 patches + buffer
        )
    elif encoder_type == "mel":
        print(f"Creating MelSpectrogramTokenizer (trainable, embed_dim={embed_dim})")
        encoder = MelSpectrogramTokenizer(
            output_dim=embed_dim,
            sample_rate=16000,
            n_mels=64,
            dropout=0.1,
            max_time_frames=1000,
        )
    elif encoder_type == "wav2vec2":
        print(f"Creating Wav2Vec2Encoder (pretrained, frozen={freeze})")
        encoder = Wav2Vec2Encoder(
            output_dim=embed_dim,
            model_name="facebook/wav2vec2-base",
            freeze_encoder=freeze,
        )
    elif encoder_type == "whisper":
        print(f"Creating WhisperEncoder (pretrained, frozen={freeze})")
        encoder = WhisperEncoder(
            output_dim=embed_dim,
            model_name="openai/whisper-base",
            freeze_encoder=freeze,
        )
    elif encoder_type == "clap":
        print(f"Creating CLAPEncoder (pretrained HTSAT, frozen={freeze})")
        encoder = CLAPEncoder(
            output_dim=embed_dim,
            model_name="laion/clap-htsat-unfused",
            freeze_encoder=freeze,
            max_length_s=30.0,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}. Choose from: tokenizer, mel, wav2vec2, whisper, clap")

    return encoder


def create_audio_model(
    encoder_type: str,
    llm_id: str,
    device: str = "cuda",
    embed_dim: int = 256,
    patch_size: int = 640,
    freeze_encoder: bool = True,
    num_latents: int = 64,
    gradient_checkpointing: bool = False,
    freeze_lm_embeddings: bool = True,
    cross_attn_every_n_layers: int = 4,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
):
    """
    Create an audio language model with specified encoder using Flamingo architecture.

    Architecture:
        Audio -> Encoder -> Projector -> Perceiver -> Cross-Attention with LLM -> Output

    Args:
        encoder_type: Type of encoder ('tokenizer', 'mel', 'wav2vec2', 'whisper', 'clap')
        llm_id: HuggingFace model ID for the LLM
        device: Device to load model on
        embed_dim: Embedding dimension for encoder output
        patch_size: Patch size for RawAudioTokenizer
        freeze_encoder: Whether to freeze encoder (for pretrained encoders like wav2vec2, whisper, clap)

    Returns:
        Initialized AudioFlamingo model
    """
    print()
    print("=" * 60)
    print("Creating Audio Language Model")
    print("=" * 60)
    print()

    # 1. Create encoder
    encoder = create_encoder(
        encoder_type=encoder_type,
        embed_dim=embed_dim,
        patch_size=patch_size,
        freeze=freeze_encoder,
    )

    # Get actual output dimension from encoder
    encoder_output_dim = encoder.get_output_dim()

    # 2. Create projector: map encoder output dim -> Flamingo visual dim
    print(f"Creating MLPProjector: {encoder_output_dim} -> {ENCODER_OUTPUT_DIM}")
    projector = MLPProjector(
        input_dim=encoder_output_dim,
        output_dim=ENCODER_OUTPUT_DIM,
        device=device,
    )

    # 3. Create AudioFlamingo model (handles everything: LLM, perceiver, cross-attention)
    print(f"Creating AudioFlamingo with LLM: {llm_id}")
    model = AudioFlamingo(
        encoder=encoder,
        projector=projector,
        device=device,
        llm_id=llm_id,
        vis_dim=ENCODER_OUTPUT_DIM,
        freeze_encoder=freeze_encoder,
        freeze_lm_embeddings=freeze_lm_embeddings,
        num_latents=num_latents,
        gradient_checkpointing=gradient_checkpointing,
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )

    return model


# ─── Training loop ───────────────────────────────────────────────────────────────

def train_stage(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    logger,
    patch_size: int = 640,
    num_epochs: int = 10,
    early_stop_patience: int = 3,
    checkpoint_dir: str = "checkpoints",
    grad_clip_norm: float = 1.0,
    grad_accum_steps: int = 1,
    dataset_weight_map: dict = None,
    label_smoothing: float = 0.0,
    answer_tail_weight: float = 1.0,
):
    """
    Train model for one curriculum stage.

    Args:
        model: Audio language model
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        device: Device to train on
        logger: Logger instance
        patch_size: Patch size for audio preprocessing
        num_epochs: Number of epochs
        early_stop_patience: Early stopping patience
        checkpoint_dir: Directory to save checkpoints
        grad_clip_norm: Gradient clipping norm
        grad_accum_steps: Number of mini-batches to accumulate before an
            optimizer step.  Effective batch size = batch_size × grad_accum_steps.
        dataset_weight_map: Optional dict mapping dataset source name to a
            float loss weight (e.g. ``{'ICBHI': 0.31, 'KAUH': 6.48, ...}``).
            When provided, ``compute_loss`` receives per-sample weights
            derived from this mapping.
        label_smoothing: Label smoothing factor (0.0 = off, 0.1 = recommended
            for CoT stage).  Passed to ``compute_loss``.
        answer_tail_weight: Multiplicative loss weight for tokens in the
            final answer span (after "Answer:" marker) for CoT targets.
            Use >1.0 to emphasize concise final answer learning.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0

    log_gpu_memory(logger, "before_training")
    log_system_memory(logger, "before_training")

    # AMP autocast for mixed-precision training (saves ~30-50% GPU memory).
    # Use bfloat16 (same range as float32, no GradScaler needed) since the
    # LLM already runs in bfloat16.
    use_amp = torch.cuda.is_available()

    disable_tqdm = should_disable_tqdm()
    if disable_tqdm:
        logger.info("Progress bars disabled (AUDIO_TQDM_DISABLE=1).")

    # Log progress to file every N batches (independent of tqdm)
    LOG_PROGRESS_EVERY = int(os.getenv("AUDIO_LOG_PROGRESS_EVERY", "50"))

    for epoch in range(num_epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        num_batches = 0

        logger.info(f"--- Epoch {epoch+1}/{num_epochs} [Train] ---")
        log_gpu_memory(logger, f"epoch_{epoch+1}_start")

        epoch_t0 = time.time()  # epoch wall-clock start
        # Zero gradients once at the start of each accumulation window
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs} [Train]",
            disable=disable_tqdm,
        )
        for batch_idx, batch in enumerate(pbar):
            try:
                # Log memory every 100 batches for diagnosis
                if batch_idx % 100 == 0:
                    log_gpu_memory(logger, f"batch_{batch_idx}")
                    log_system_memory(logger, f"batch_{batch_idx}")

                # Log batch info on first batch
                if batch_idx == 0:
                    if "time_series" in batch and isinstance(batch["time_series"], list):
                        shapes = [t.shape for t in batch["time_series"]]
                        logger.debug(f"First batch time_series shapes: {shapes}")
                    elif "time_series" in batch:
                        logger.debug(f"First batch time_series shape: {batch['time_series'].shape}")

                # Convert numpy arrays back to torch tensors (when workers
                # returned numpy to avoid /dev/shm — see _collate_as_numpy).
                for elem in batch:
                    ts = elem.get("time_series")
                    if isinstance(ts, np.ndarray):
                        elem["time_series"] = torch.from_numpy(ts)

                # Build per-sample loss weights from dataset_weight_map
                sample_weights = None
                if dataset_weight_map is not None:
                    w = [
                        dataset_weight_map.get(
                            elem.get("dataset_source", "unknown"), 1.0
                        )
                        for elem in batch
                    ]
                    sample_weights = torch.tensor(w, dtype=torch.float32)

                # Forward + backward with bfloat16 autocast (no GradScaler needed)
                # Scale loss by accumulation steps so the averaged gradient is correct
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    loss = model.compute_loss(
                        batch,
                        sample_weights=sample_weights,
                        label_smoothing=label_smoothing,
                        answer_tail_weight=answer_tail_weight,
                    )

                scaled_loss = loss / grad_accum_steps
                scaled_loss.backward()

                logger.debug(f"Batch {batch_idx}: loss={loss.item():.4f}")

                train_loss += loss.item()
                num_batches += 1

                # Optimizer step every grad_accum_steps or at the last batch
                is_accum_step = (batch_idx + 1) % grad_accum_steps == 0
                is_last_batch = (batch_idx + 1) == len(train_loader)
                if is_accum_step or is_last_batch:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                # Release memory from variable-size batches every 50 steps
                # to prevent PyTorch's caching allocator from hoarding blocks
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

                # Update progress bar with loss and GPU memory
                postfix = {"loss": f"{loss.item():.4f}"}
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)
                    postfix["gpu_gb"] = f"{gpu_mem_gb:.1f}"
                if not disable_tqdm:
                    try:
                        pbar.set_postfix(postfix)
                    except BrokenPipeError:
                        disable_tqdm = True
                        pbar.disable = True
                        logger.warning(
                            "Progress output pipe closed; disabling tqdm for stability."
                        )

                # Periodic logger progress with ETA (always written to log file)
                if (batch_idx + 1) % LOG_PROGRESS_EVERY == 0:
                    elapsed = time.time() - epoch_t0
                    batches_done = batch_idx + 1
                    total_batches = len(train_loader)
                    avg_s = elapsed / batches_done
                    eta_s = avg_s * (total_batches - batches_done)
                    avg_loss = train_loss / num_batches
                    gpu_str = f", gpu={gpu_mem_gb:.1f}GB" if torch.cuda.is_available() else ""
                    logger.info(
                        f"  [{batches_done}/{total_batches}] "
                        f"loss={avg_loss:.4f} "
                        f"({avg_s:.1f}s/batch, ETA {eta_s/60:.1f}min{gpu_str})"
                    )

            except torch.cuda.OutOfMemoryError:
                logger.error(f"CUDA OOM at batch {batch_idx}!")
                log_gpu_memory(logger, "OOM")
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)  # Reset accumulated grads
                logger.info("Cleared CUDA cache, skipping batch")
                continue
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                logger.debug(traceback.format_exc())
                continue

        avg_train_loss = train_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1} train: avg_loss={avg_train_loss:.4f}, batches={num_batches}")
        log_gpu_memory(logger, f"epoch_{epoch+1}_after_train")

        # ── Validation ───────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(
                tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{num_epochs} [Val]",
                    disable=disable_tqdm,
                )
            ):
                try:
                    for elem in batch:
                        ts = elem.get("time_series")
                        if isinstance(ts, np.ndarray):
                            elem["time_series"] = torch.from_numpy(ts)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                        loss = model.compute_loss(
                            batch,
                            label_smoothing=label_smoothing,
                            answer_tail_weight=answer_tail_weight,
                        )
                    val_loss += loss.item()
                    val_batches += 1
                except Exception as e:
                    logger.error(f"Error in validation batch {batch_idx}: {e}")
                    continue
        torch.cuda.empty_cache()  # Free validation activations before next train epoch

        avg_val_loss = val_loss / max(val_batches, 1)

        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )
        log_gpu_memory(logger, f"epoch_{epoch+1}_end")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pt"))
            logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience: {patience_counter}/{early_stop_patience}")

        # Early stopping
        if patience_counter >= early_stop_patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    return best_val_loss


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Audio Curriculum Learning with Flamingo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with RawAudioTokenizer (recommended, light, trainable)
    python scripts/training/audio_curriculum_learning.py --encoder tokenizer --audio_dir ./data/caresound_audio/audio_merged

    # Train with MelSpectrogramTokenizer
    python scripts/training/audio_curriculum_learning.py --encoder mel --audio_dir ./data/caresound_audio/audio_merged

    # Train with Wav2Vec2 (pretrained, frozen)
    python scripts/training/audio_curriculum_learning.py --encoder wav2vec2 --audio_dir ./data/caresound_audio/audio_merged

    # Train with Whisper (pretrained, frozen)
    python scripts/training/audio_curriculum_learning.py --encoder whisper --audio_dir ./data/caresound_audio/audio_merged

    # Train with CLAP HTSAT (pretrained, frozen, audio-language aligned)
    python scripts/training/audio_curriculum_learning.py --encoder clap --audio_dir ./data/caresound_audio/audio_merged
"""
    )

    # Encoder selection
    parser.add_argument("--encoder", type=str, default="tokenizer",
                        choices=["tokenizer", "mel", "wav2vec2", "whisper", "clap"],
                        help="Encoder type (default: tokenizer - RECOMMENDED)")

    # Model settings
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                        help="HuggingFace LLM model ID")
    parser.add_argument("--embed_dim", type=int, default=256,
                        help="Encoder output embedding dimension (default: 256)")
    parser.add_argument("--patch_size", type=int, default=640,
                        help="Patch size for RawAudioTokenizer (default: 640 = 40ms at 16kHz)")
    parser.add_argument("--num_latents", type=int, default=64,
                        help="Number of Perceiver Resampler latent tokens (default: 64). "
                             "Higher values reduce compression for longer audio.")
    parser.add_argument("--unfreeze_encoder", action="store_true",
                        help="Force unfreeze pretrained encoder (wav2vec2/whisper/clap) for fine-tuning. "
                             "Note: mel/tokenizer encoders are always unfrozen automatically.")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Force freeze encoder even for mel/tokenizer (not recommended).")
    parser.add_argument("--unfreeze_lm_embeddings", action="store_true",
                        help="Unfreeze LLM input embeddings (frozen by default to reduce overfitting)")
    parser.add_argument("--cross_attn_every_n", type=int, default=1,
                        help="Insert cross-attention every N LLM layers (default: 1). "
                             "Lower = more cross-attention blocks, higher = fewer.")
    parser.add_argument("--lora_rank", type=int, default=0,
                        help="LoRA rank for cross-attention layers (default: 0 = disabled). "
                             "0 = disable LoRA (full fine-tuning). 8 or 16 recommended.")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA alpha scaling (default: 16.0)")
    parser.add_argument("--lora_dropout", type=float, default=None,
                        help="LoRA dropout (default: same as AUDIO_DROPOUT from config)")

    # Data settings
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_merged",
                        help="Directory containing audio files")
    parser.add_argument("--stages", type=str, nargs="+", default=["stage1_caresound_qa"],
                        help="Curriculum stages to train (default: only CaReSound)")
    parser.add_argument("--max_audio_length", type=int, default=None,
                        help="Maximum audio length in samples (default: 480000 = 30s at 16kHz). "
                             "Use 80000 for 5s, 160000 for 10s, 320000 for 20s")

    # Training settings
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size (default: from config)")
    parser.add_argument("--num_epochs", type=int, default=None,
                        help="Number of epochs (default: from config)")
    parser.add_argument("--grad_accum_steps", type=int, default=1,
                        help="Gradient accumulation steps (default: 1 = no accumulation). "
                             "Effective batch = batch_size × grad_accum_steps.")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers for parallel audio loading (default: 2). "
                             "Set to 0 for single-process loading. Keep low if /dev/shm is small.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce activation memory (~60%% less) "
                             "at the cost of ~30%% slower per step. Allows larger batch sizes.")
    parser.add_argument("--gpu_mem_fraction", type=float, default=None,
                        help="Optional CUDA per-process memory cap in [0, 1]. "
                             "Example: --gpu_mem_fraction 0.8")
    parser.add_argument("--gpu_mem_device", type=int, default=0,
                        help="CUDA device index used for --gpu_mem_fraction "
                             "(default: 0). Example: --gpu_mem_device 3")
    parser.add_argument("--balanced_sampling", action="store_true",
                        help="Use dataset-balanced sampling (WeightedRandomSampler) so each source "
                             "dataset (ICBHI, CirCor, ...) is sampled with equal probability per "
                             "epoch, counteracting the ICBHI dominance in the training set.")
    parser.add_argument("--loss_weighting", action="store_true",
                        help="Apply per-sample inverse-frequency loss weighting based on dataset "
                             "source. Samples from under-represented datasets (e.g. KAUH) get "
                             "higher loss weights. Can be combined with --balanced_sampling.")

    # CoT (Stage 2) settings
    parser.add_argument("--cot_lr", type=float, default=None,
                        help=f"Learning rate for CoT Stage 2 (default: {AUDIO_COT_LR}). "
                             "Lower than Stage 1 since alignment is stable.")
    parser.add_argument("--label_smoothing", type=float, default=None,
                        help=f"Label smoothing for CoT loss (default: {AUDIO_COT_LABEL_SMOOTHING} in stage2, 0.0 in stage1). "
                             "Softens one-hot targets — useful for CoT since multiple rationales are valid.")
    parser.add_argument("--answer_tail_weight", type=float, default=None,
                        help="Extra loss weight for tokens in the final answer tail after 'Answer:' in CoT targets. "
                             "Default: 1.0 (off). Recommended starting point for stage2: 2.0-4.0.")
    parser.add_argument("--stage2_mix_qa", action="store_true",
                        help="When training stage2_audio_cot, mix in QA train samples "
                             "(ConcatDataset[QA train, CoT train]) to preserve concise answer behavior.")

    # Auto-detect best available device: CUDA GPU > Apple MPS > CPU
    if torch.cuda.is_available():
        default_device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        default_device = "mps"
    else:
        default_device = "cpu"
    parser.add_argument("--device", type=str, default=default_device,
                        help="Device to train on (auto-detected: cuda > mps > cpu)")
    parser.add_argument("--checkpoint_root", type=str, default="audio_checkpoints",
                        help="Root directory for checkpoints")
    parser.add_argument("--resume_checkpoint", type=str, default=None,
                        help="Path to a checkpoint to load before training starts. "
                             "Use to resume from a previous stage, e.g. load stage1 checkpoint "
                             "then train stage2: --resume_checkpoint audio_checkpoints/mel_stage1_caresound_qa/best_model.pt "
                             "--stages stage2_audio_cot")
    parser.add_argument("--log_file", type=str, default=None,
                        help="Log file path (default: auto-generated in audio_logs/)")

    args = parser.parse_args()

    # ─── Setup logging ───────────────────────────────────────────────────────
    logger, log_file = setup_logging(
        log_file=args.log_file,
        encoder_name=args.encoder,
    )

    logger.info("=" * 60)
    logger.info("AUDIO CURRICULUM LEARNING")
    logger.info("=" * 60)
    logger.info(f"Encoder:    {args.encoder}")
    logger.info(f"LLM:        {args.llm_id}")
    logger.info(f"Audio dir:  {args.audio_dir}")
    logger.info(f"Stages:     {args.stages}")
    logger.info(f"Device:     {args.device}")
    logger.info(f"Log file:   {log_file}")
    if args.resume_checkpoint:
        logger.info(f"Resume from: {args.resume_checkpoint}")
    if args.max_audio_length:
        duration_sec = args.max_audio_length / 16000
        logger.info(f"Max audio:  {args.max_audio_length} samples ({duration_sec:.1f}s at 16kHz)")
    if args.num_latents != 64:
        logger.info(f"Perceiver:  {args.num_latents} latent tokens (default: 64)")
    if args.balanced_sampling:
        logger.info("Balanced sampling: ENABLED (equal dataset-source probability)")
    if args.loss_weighting:
        logger.info("Loss weighting:    ENABLED (inverse-frequency per dataset source)")
    if args.stage2_mix_qa:
        logger.info("Stage2 QA mix:    ENABLED")
    logger.info(
        f"Cross-attn every: "
        f"{args.cross_attn_every_n if args.cross_attn_every_n is not None else AUDIO_CROSS_ATTN_EVERY_N} layers"
    )
    _lora_r = args.lora_rank if args.lora_rank is not None else AUDIO_LORA_RANK
    logger.info(f"LoRA rank:        {_lora_r}" + (" (disabled)" if _lora_r == 0 else ""))
    logger.info(f"LR encoder:       {AUDIO_LR_ENCODER}")
    logger.info(f"LR other:         {AUDIO_LR_OTHER}")
    logger.info(f"Dropout:          {AUDIO_DROPOUT}")

    # Optional per-process CUDA memory cap (helps avoid contention with other users/jobs)
    if args.gpu_mem_fraction is not None:
        if not (0.0 < args.gpu_mem_fraction <= 1.0):
            raise ValueError("--gpu_mem_fraction must be in the range (0, 1].")
        if not torch.cuda.is_available():
            logger.warning("--gpu_mem_fraction was provided but CUDA is not available; skipping.")
        else:
            torch.cuda.set_per_process_memory_fraction(
                args.gpu_mem_fraction,
                device=args.gpu_mem_device,
            )
            logger.info(
                f"CUDA per-process memory fraction set to {args.gpu_mem_fraction:.3f} "
                f"on device index {args.gpu_mem_device}"
            )

    # GPU diagnostics
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_free = torch.cuda.mem_get_info()[0] / (1024**3)
        logger.info(f"GPU: {gpu_name} | {gpu_total:.1f}GB total | {gpu_free:.1f}GB free")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple MPS available")
    else:
        logger.warning("No GPU detected! Training will be on CPU (very slow)")
    log_system_memory(logger, "startup")

    try:
        # ── Create model ─────────────────────────────────────────────────────
        logger.info("Creating model...")
        # Auto-detect encoder freeze: pretrained encoders (wav2vec2, whisper, clap)
        # are frozen by default; lightweight encoders (mel, tokenizer) are trainable.
        PRETRAINED_ENCODERS = {"wav2vec2", "whisper", "clap"}
        if args.freeze_encoder:
            freeze_enc = True
        elif args.unfreeze_encoder:
            freeze_enc = False
        else:
            # Auto: freeze pretrained, unfreeze mel/tokenizer
            freeze_enc = args.encoder in PRETRAINED_ENCODERS
        logger.info(f"Encoder '{args.encoder}' → freeze_encoder={freeze_enc} "
                     f"({'pretrained' if args.encoder in PRETRAINED_ENCODERS else 'lightweight'})")
        # Resolve config values (CLI overrides > config defaults)
        cross_attn_every_n = args.cross_attn_every_n if args.cross_attn_every_n is not None else AUDIO_CROSS_ATTN_EVERY_N
        lora_rank = args.lora_rank if args.lora_rank is not None else AUDIO_LORA_RANK
        lora_dropout = args.lora_dropout if args.lora_dropout is not None else AUDIO_DROPOUT

        model = create_audio_model(
            encoder_type=args.encoder,
            llm_id=args.llm_id,
            device=args.device,
            embed_dim=args.embed_dim,
            patch_size=args.patch_size,
            freeze_encoder=freeze_enc,
            num_latents=args.num_latents,
            gradient_checkpointing=args.gradient_checkpointing,
            freeze_lm_embeddings=not args.unfreeze_lm_embeddings,
            cross_attn_every_n_layers=cross_attn_every_n,
            lora_rank=lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Ensure ALL model components are on the device
        model.to(args.device)
        logger.info(f"Model moved to {args.device}")
        log_gpu_memory(logger, "after_model_load")
        log_system_memory(logger, "after_model_load")

        # ── Resume from checkpoint (e.g. load stage1 before training stage2) ──
        if args.resume_checkpoint:
            logger.info(f"Loading resume checkpoint: {args.resume_checkpoint}")
            ckpt = torch.load(args.resume_checkpoint, map_location=args.device)
            if isinstance(ckpt, dict) and ("llm" in ckpt or "model_state" in ckpt):
                model.load_from_file(args.resume_checkpoint)
            else:
                model.load_state_dict(ckpt)
            logger.info("✓ Resume checkpoint loaded successfully")
            log_gpu_memory(logger, "after_resume_checkpoint")

        # Log model size
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
        logger.info(f"Model: {total_params:,} params ({trainable_params:,} trainable), ~{model_size_mb:.0f}MB")

        # Get EOS token from model
        eos_token = model.tokenizer.eos_token

        # Get training parameters (use args or defaults from config)
        batch_size = args.batch_size if args.batch_size else AUDIO_BATCH_SIZE
        num_epochs = args.num_epochs if args.num_epochs else AUDIO_NUM_EPOCHS

        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Num epochs: {num_epochs}")
        if args.grad_accum_steps > 1:
            logger.info(f"Grad accumulation: {args.grad_accum_steps} steps "
                        f"(effective batch size: {batch_size * args.grad_accum_steps})")
        logger.info(f"DataLoader workers: {args.num_workers}")
        if args.gradient_checkpointing:
            logger.info("Gradient checkpointing: ENABLED (saves ~60% activation memory)")

        # ── Train each stage ─────────────────────────────────────────────────
        for stage_idx, stage in enumerate(args.stages):
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"Stage {stage_idx + 1}/{len(args.stages)}: {stage}")
            logger.info("=" * 60)

            # Determine if this is a CoT stage (Stage 2)
            # joint_qa_cot trains both together without freezing the encoder
            is_cot_stage = "cot" in stage.lower() and stage != "joint_qa_cot"

            # ── Reconfigure model for CoT if entering Stage 2 ─────────────
            if is_cot_stage:
                logger.info("🔄 Reconfiguring model for CoT (Stage 2)...")
                logger.info("   Keeping same trainable set as Stage 1 (encoder/perceiver/projector/cross-attn)")
                model.reconfigure_for_cot(cot_dropout=AUDIO_DROPOUT)
                logger.info(f"   CoT dropout: {AUDIO_DROPOUT}")

            # Get datasets
            try:
                train_ds, val_ds, test_ds = get_audio_datasets(
                    stage,
                    eos_token,
                    args.audio_dir,
                    args.max_audio_length,
                    stage2_mix_qa=args.stage2_mix_qa,
                )
            except Exception as e:
                logger.error(f"Error loading dataset for {stage}: {e}")
                logger.debug(traceback.format_exc())
                continue

            if train_ds is None:
                logger.info(f"Skipping {stage} - dataset not available")
                continue

            logger.info(f"Train samples: {len(train_ds)}")
            logger.info(f"Val samples: {len(val_ds)}")

            # Create data loaders – parallel workers overlap audio I/O with GPU compute
            nw = args.num_workers

            # When using multiprocessing workers, torch tensors are shared via
            # /dev/shm which may be tiny (e.g. 64 MB in Docker).  Returning
            # numpy arrays from the collate avoids shm entirely — numpy arrays
            # are pickled through regular pipes.  We convert back to torch in
            # the training loop (see _batch_numpy_to_torch helper below).
            def _collate_as_numpy(b):
                batch = extend_time_series_to_match_patch_size_and_aggregate(b)
                if nw > 0:
                    for elem in batch:
                        ts = elem.get("time_series")
                        if isinstance(ts, torch.Tensor):
                            elem["time_series"] = ts.numpy()
                return batch

            # ── Balanced sampling & loss weighting ────────────────────────
            train_sampler = None
            dataset_weight_map = None  # dataset_source → loss weight
            ds_labels = None  # initialise before conditional blocks

            if args.balanced_sampling or args.loss_weighting:
                # Retrieve dataset source labels (lightweight, no audio I/O)
                if hasattr(train_ds, "get_dataset_labels"):
                    ds_labels = train_ds.get_dataset_labels()
                else:
                    logger.warning(
                        "Dataset does not support get_dataset_labels(); "
                        "balanced sampling / loss weighting disabled."
                    )

            if ds_labels is not None and args.balanced_sampling:
                ds_counts = Counter(ds_labels)
                total = len(ds_labels)
                n_datasets = len(ds_counts)
                # Weight per sample = 1 / (K * count_of_its_dataset)
                # so each dataset contributes equally to the epoch
                sample_weights_list = [
                    1.0 / (n_datasets * ds_counts[lbl]) for lbl in ds_labels
                ]
                train_sampler = WeightedRandomSampler(
                    weights=sample_weights_list,
                    num_samples=total,  # same epoch length
                    replacement=True,
                )
                logger.info(f"Balanced sampler: {dict(ds_counts)}")
                for src, cnt in sorted(ds_counts.items()):
                    pct = cnt / total * 100
                    logger.info(f"  {src}: {cnt} ({pct:.1f}%) → sampling weight "
                                f"{1.0 / (n_datasets * cnt):.6f}")

            if ds_labels is not None and args.loss_weighting:
                # Inverse-frequency weights normalised so mean weight = 1
                ds_counts = Counter(ds_labels)
                total = len(ds_labels)
                raw_weights = {
                    src: total / cnt for src, cnt in ds_counts.items()
                }
                mean_w = sum(raw_weights.values()) / len(raw_weights)
                dataset_weight_map = {
                    src: w / mean_w for src, w in raw_weights.items()
                }
                logger.info("Loss weights (normalised inverse-frequency):")
                for src, w in sorted(dataset_weight_map.items()):
                    logger.info(f"  {src}: {w:.4f}")

            # ── DataLoaders ───────────────────────────────────────────────
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=(train_sampler is None),  # mutually exclusive with sampler
                sampler=train_sampler,
                collate_fn=_collate_as_numpy,
                drop_last=False,
                num_workers=nw,
                persistent_workers=nw > 0,
                prefetch_factor=2 if nw > 0 else None,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=_collate_as_numpy,
                drop_last=False,
                num_workers=nw,
                persistent_workers=nw > 0,
                prefetch_factor=2 if nw > 0 else None,
            )

            # When num_workers=0 (no multiprocessing), wrap with thread-based
            # prefetcher so audio I/O overlaps with GPU compute — no /dev/shm needed.
            if nw == 0:
                train_loader = PrefetchLoader(train_loader, prefetch=2)
                val_loader = PrefetchLoader(val_loader, prefetch=2)

            # ── Optimizer: stage-specific configuration ───────────────────
            if is_cot_stage:
                # Stage 2 (CoT): same LRs as Stage 1.
                cot_lr = AUDIO_LR_OTHER
                enc_lr = AUDIO_LR_ENCODER

                named_params = list(model.named_parameters())
                trainable = [(n, p) for n, p in named_params if p.requires_grad]

                encoder_params = []
                xattn_params_wd = []
                other_params_no_wd = []

                for name, p in trainable:
                    if "vision_encoder.encoder" in name or "audio_encoder" in name:
                        encoder_params.append(p)
                    elif "gated_cross_attn" in name and "lora" not in name.lower():
                        xattn_params_wd.append(p)
                    else:
                        other_params_no_wd.append(p)

                param_groups = []
                if encoder_params:
                    param_groups.append({
                        "params": encoder_params,
                        "lr": enc_lr,
                        "weight_decay": 0.0,
                        "label": "encoder",
                    })
                if xattn_params_wd:
                    param_groups.append({
                        "params": xattn_params_wd,
                        "lr": cot_lr,
                        "weight_decay": AUDIO_WEIGHT_DECAY,
                        "label": "cross-attn",
                    })
                if other_params_no_wd:
                    param_groups.append({
                        "params": other_params_no_wd,
                        "lr": cot_lr,
                        "weight_decay": 0.0,
                        "label": "other",
                    })
                logger.info(f"CoT optimizer: lr={cot_lr:.1e} (encoder: {enc_lr:.1e}), wd={AUDIO_WEIGHT_DECAY} (on cross-attn)")
            else:
                # Stage 1 (Alignment): differential learning rates
                #   - Encoder params: AUDIO_LR_ENCODER (1e-5)
                #   - Other trainable params (perceiver, cross-attn, projector): AUDIO_LR_OTHER (3e-5)
                named_params = list(model.named_parameters())
                trainable = [(n, p) for n, p in named_params if p.requires_grad]

                # Group params by component for differential LR
                encoder_params = []
                xattn_params_wd = []     # cross-attention with weight decay
                other_params_no_wd = []  # perceiver, projector, LoRA, etc.

                for name, p in trainable:
                    if "vision_encoder.encoder" in name or "audio_encoder" in name:
                        encoder_params.append(p)
                    elif "gated_cross_attn" in name and "lora" not in name.lower():
                        # Original cross-attention weights (if not LoRA mode)
                        xattn_params_wd.append(p)
                    else:
                        other_params_no_wd.append(p)

                param_groups = []
                if encoder_params:
                    param_groups.append({
                        "params": encoder_params,
                        "lr": AUDIO_LR_ENCODER,
                        "weight_decay": 0.0,
                        "label": "encoder",
                    })
                if xattn_params_wd:
                    param_groups.append({
                        "params": xattn_params_wd,
                        "lr": AUDIO_LR_OTHER,
                        "weight_decay": AUDIO_WEIGHT_DECAY,
                        "label": "cross-attn",
                    })
                if other_params_no_wd:
                    param_groups.append({
                        "params": other_params_no_wd,
                        "lr": AUDIO_LR_OTHER,
                        "weight_decay": 0.0,
                        "label": "other",
                    })

            optimizer = AdamW(param_groups)

            total_trainable = sum(sum(p.numel() for p in g["params"]) for g in param_groups)
            logger.info(f"Trainable: {total_trainable:,} params")
            for g in param_groups:
                g_total = sum(p.numel() for p in g["params"])
                logger.info(f"  {g.get('label', '?'):12s}: {g_total:>12,} params, lr={g['lr']:.1e}, wd={g['weight_decay']}")

            # Resolve stage-specific hyperparameters
            stage_num_epochs = num_epochs
            stage_patience = AUDIO_EARLY_STOP_PAT
            stage_label_smoothing = 0.0
            stage_answer_tail_weight = 1.0

            if is_cot_stage:
                # Use same epochs and patience as Stage 1; only label smoothing differs.
                stage_label_smoothing = (
                    args.label_smoothing if args.label_smoothing is not None
                    else AUDIO_COT_LABEL_SMOOTHING
                )
                stage_answer_tail_weight = (
                    args.answer_tail_weight if args.answer_tail_weight is not None
                    else 1.0
                )
                logger.info(f"CoT hyperparameters: epochs={stage_num_epochs}, "
                            f"patience={stage_patience}, label_smoothing={stage_label_smoothing}, "
                            f"answer_tail_weight={stage_answer_tail_weight}")

            # Scheduler counts *optimizer updates*, not forward passes.
            # With gradient accumulation the number of updates is reduced.
            batches_per_epoch = len(train_loader)
            updates_per_epoch = (batches_per_epoch + args.grad_accum_steps - 1) // args.grad_accum_steps
            total_steps = updates_per_epoch * stage_num_epochs
            warmup_steps = int(total_steps * AUDIO_WARMUP_FRAC)
            scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

            logger.info(f"Batches/epoch: {batches_per_epoch}, Optimizer updates/epoch: {updates_per_epoch}")
            logger.info(f"Total optimizer steps: {total_steps}, Warmup: {warmup_steps}")
            log_gpu_memory(logger, "before_training")

            # Train stage
            checkpoint_dir = os.path.join(args.checkpoint_root, f"{args.encoder}_{stage}")
            best_val_loss = train_stage(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=args.device,
                logger=logger,
                patch_size=args.patch_size,
                num_epochs=stage_num_epochs,
                early_stop_patience=stage_patience,
                checkpoint_dir=checkpoint_dir,
                grad_clip_norm=AUDIO_GRAD_CLIP_NORM,
                grad_accum_steps=args.grad_accum_steps,
                dataset_weight_map=dataset_weight_map,
                label_smoothing=stage_label_smoothing,
                answer_tail_weight=stage_answer_tail_weight,
            )

            logger.info(f"Stage {stage} done - best val loss: {best_val_loss:.4f}")
            logger.info(f"  Checkpoint: {checkpoint_dir}")

            # Load best model for next stage
            if stage_idx < len(args.stages) - 1:
                best_ckpt = os.path.join(checkpoint_dir, "best_model.pt")
                logger.info(f"  Loading best checkpoint for next stage: {best_ckpt}")
                ckpt = torch.load(best_ckpt, map_location=args.device)
                if isinstance(ckpt, dict) and ("llm" in ckpt or "model_state" in ckpt):
                    model.load_from_file(best_ckpt)
                else:
                    model.load_state_dict(ckpt)
                logger.info("  ✓ Best model loaded for next stage")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Audio curriculum learning completed!")
        logger.info("=" * 60)
        logger.info(f"Checkpoints: {args.checkpoint_root}/")
        logger.info(f"Full log: {log_file}")

    except torch.cuda.OutOfMemoryError:
        logger.critical("FATAL: CUDA Out of Memory!")
        log_gpu_memory(logger, "FATAL_OOM")
        log_system_memory(logger, "FATAL_OOM")
        logger.critical("Try: --batch_size 1, or smaller --embed_dim, or smaller --llm_id")
        raise
    except Exception as e:
        logger.critical(f"FATAL ERROR: {e}")
        logger.critical(traceback.format_exc())
        log_gpu_memory(logger, "FATAL_ERROR")
        log_system_memory(logger, "FATAL_ERROR")
        raise


if __name__ == "__main__":
    main()
