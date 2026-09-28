# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# ---------------------------
# Audio Model Configuration
# ---------------------------
# This file contains configuration parameters optimized for audio language models.
# For time-series models, see model_config.py

# Audio-specific hyperparameters
AUDIO_SAMPLE_RATE = 16000  # Standard sample rate for speech (16kHz)
AUDIO_MAX_LENGTH = 80000   # 5 seconds at 16kHz (default for testing)
AUDIO_MAX_SECONDS = 5      # Maximum audio length in seconds
AUDIO_PATCH_SIZE = 640     # 40ms at 16kHz (640 samples) - fewer patches, less memory
AUDIO_MAX_PATCHES = 768    # Maximum number of patches (~30s at 640 patch size)

# Encoder configuration for audio (RawAudioTokenizer - light encoder for Flamingo)
AUDIO_EMBED_DIM = 256      # Embedding dimension (reduced for memory efficiency)
AUDIO_ENCODER_OUTPUT_DIM = AUDIO_EMBED_DIM
AUDIO_DROPOUT = 0.4        # Dropout probability (V5: increased from 0.3 to reduce overfitting on small data)

# Encoder selection and pretrained model defaults
# Choose: 'tokenizer', 'mel', 'wav2vec2', 'whisper', 'clap'
AUDIO_ENCODER_TYPE = "wav2vec2"
# HuggingFace model id used for pretrained encoders (wav2vec2 / whisper)
AUDIO_ENCODER_MODEL = "facebook/wav2vec2-base"
# Whether to freeze pretrained encoder weights (recommended to save memory)
AUDIO_FREEZE_ENCODER = True

# Training hyperparameters (adjusted for audio)
AUDIO_BATCH_SIZE = 2       # Smaller batch size due to longer sequences
AUDIO_NUM_EPOCHS = 50
AUDIO_EARLY_STOP_PAT = 3   # Reduced from 5 (models overfit after epoch 2-3)
AUDIO_LR_ENCODER = 5e-6    # Learning rate for encoder (V5: reduced from 1e-5 to slow convergence & reduce overfitting)
AUDIO_LR_PROJECTOR = 1.5e-5  # Learning rate for projector / perceiver / cross-attn (V5: halved)
AUDIO_LR_OTHER = 1.5e-5      # Learning rate for non-encoder trainable params (V5: halved from 3e-5)
AUDIO_WEIGHT_DECAY = 5e-2  # V5: increased from 1e-2 for stronger regularization on cross-attn
AUDIO_CROSS_ATTN_EVERY_N = 4  # Insert cross-attention every N LLM layers (was 1)
AUDIO_LORA_RANK = 16       # LoRA rank for cross-attention layers (0 = disable LoRA)
AUDIO_GRAD_CLIP_NORM = 1.0
AUDIO_WARMUP_FRAC = 0.03

# ---------------------------
# Stage 2: Chain-of-Thought (CoT) Fine-Tuning
# ---------------------------
# After Stage 1 alignment converges, freeze encoder/perceiver/projector
# and only train LoRA cross-attention adapters + gates with CoT data.
AUDIO_COT_LR = 2e-5           # Lower LR — alignment is stable, only adapt reasoning
AUDIO_COT_NUM_EPOCHS = 15     # CoT may need more epochs (longer targets)
AUDIO_COT_EARLY_STOP_PAT = 4  # Slightly more patience for CoT
AUDIO_COT_LABEL_SMOOTHING = 0.1  # Multiple valid rationales → soften targets
AUDIO_COT_DROPOUT = 0.3       # Strong dropout to prevent memorisation
AUDIO_COT_MAX_NEW_TOKENS = 512  # CoT generates longer outputs than QA

# Audio preprocessing
AUDIO_NORMALIZE = True     # Normalize audio to zero mean, unit variance
AUDIO_TO_MONO = True       # Convert stereo to mono

# Augmentation settings (for training)
AUDIO_AUGMENT = False      # Enable audio augmentation
AUDIO_TIME_STRETCH_RANGE = (0.9, 1.1)  # Time stretch factor range
AUDIO_NOISE_LEVEL = 0.005  # Gaussian noise standard deviation

# ---------------------------
# Original Time-Series Configuration
# ---------------------------
# Keep original parameters for backward compatibility

BATCH_SIZE = 4
PATCH_SIZE = 4
NUM_EPOCHS = 20
EARLY_STOP_PAT = 5
LR_ENCODER = 2e-4
LR_PROJECTOR = 1e-4
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0
WARMUP_FRAC = 0.03
MAX_SAMPLES = None
RESULTS_FILE = "test_predictions.jsonl"
EMBED_DIM = 128
ENCODER_OUTPUT_DIM = EMBED_DIM
TRANSFORMER_INPUT_DIM = EMBED_DIM
