# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torchaudio.functional as AF
from transformers import ClapModel, AutoFeatureExtractor

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class CLAPEncoder(TimeSeriesEncoderBase):
    """
    Audio encoder using LAION-AI's CLAP (Contrastive Language-Audio Pretraining).

    Uses the HTSAT (HTS-Audio-Transformer) backbone from CLAP, extracting
    intermediate temporal features BEFORE the contrastive projection head.
    This provides a sequence of audio tokens [B, N, 768] suitable for the
    Flamingo perceiver resampler and cross-attention mechanism.

    CLAP was pretrained on LAION-Audio-630K with audio-language contrastive
    supervision, so its features are inherently aligned with natural language.

    Architecture:
        Input [B, L] raw audio (16kHz)
            │
            ▼
        Resample to 48kHz (internal)
            │
            ▼
        Mel-Spectrogram (64 bins, internal)
            │
            ▼
        HTSAT (Swin-like Transformer)
            │
            ▼
        Output [B, N, hidden_size]  ← we use this (before pooling)
            │
            ▼  (skipped)
        Pool → Contrastive projection [B, 512]  ← NOT used

    Model options (HuggingFace):
        - laion/clap-htsat-unfused:  HTSAT, 768 dim, ~84M params (recommended)
        - laion/clap-htsat-fused:    HTSAT + feature fusion, 768 dim

    Args:
        output_dim: Output dimension (inherited, actual dim determined by model)
        dropout: Dropout probability
        model_name: HuggingFace model identifier for CLAP
        freeze_encoder: If True, freeze CLAP weights during training
        sample_rate: Input audio sample rate (resampled to 48kHz internally)
        max_length_s: Maximum audio length in seconds (default: 30)

    Reference:
        https://github.com/LAION-AI/CLAP
        Wu et al., "Large-scale Contrastive Language-Audio Pretraining with
        Feature Fusion and Keyword-to-Caption Augmentation", ICASSP 2023.
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        model_name: str = "laion/clap-htsat-unfused",
        freeze_encoder: bool = True,
        sample_rate: int = 16000,
        max_length_s: float = 30.0,
    ):
        super().__init__(output_dim, dropout)

        self.sample_rate = sample_rate
        self.freeze_encoder = freeze_encoder
        self.model_name = model_name
        self.max_length_s = max_length_s

        # CLAP expects 48kHz audio — will resample on-the-fly if needed
        self.target_sample_rate = 48000
        self._needs_resample = (sample_rate != self.target_sample_rate)
        if self._needs_resample:
            print(f"Will resample audio: {sample_rate}Hz → {self.target_sample_rate}Hz")

        # Load full CLAP model, extract audio encoder only
        print(f"Loading CLAP model: {model_name}")
        clap_model = ClapModel.from_pretrained(model_name)
        self.audio_model = clap_model.audio_model

        # Get hidden size from the audio config
        self.hidden_size = clap_model.config.audio_config.hidden_size  # 768

        # Free text model + projections (we only need the audio encoder)
        del clap_model

        # Feature extractor handles mel spectrogram conversion
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.audio_model.parameters():
                param.requires_grad = False

        # Optional dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for raw audio.

        Args:
            x: FloatTensor of shape [B, L], batch of raw audio waveforms
               where L is the number of audio samples at sample_rate (default 16kHz).
               Audio is resampled to 48kHz internally by the feature extractor.

        Returns:
            FloatTensor of shape [B, N, hidden_size], where N = H*W is the
            spatial grid flattened from HTSAT's [B, C, H, W] feature map,
            and hidden_size=768.
        """
        B, L = x.shape
        device = x.device

        # Resample 16kHz → 48kHz on CPU in float32 (CLAP was trained on 48kHz)
        # Use functional API (stateless) to avoid device/dtype issues with nn.Module buffers
        x_cpu = x.detach().cpu().float()
        if self._needs_resample:
            x_cpu = AF.resample(x_cpu, self.sample_rate, self.target_sample_rate)
        x_cpu = x_cpu.numpy()

        # Convert to list of arrays for the feature extractor
        audio_list = [x_cpu[i] for i in range(B)]

        # Process audio: computes mel spectrogram at 48kHz
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
            padding=True,
            max_length_s=self.max_length_s,
        )

        # Move to same device and dtype as model weights
        model_dtype = next(self.audio_model.parameters()).dtype
        input_features = inputs["input_features"].to(device=device, dtype=model_dtype)

        # Forward through CLAP audio encoder (HTSAT)
        outputs = self.audio_model(input_features)

        # HTSAT returns [B, C, H, W] (like a CNN feature map).
        # Reshape to [B, H*W, C] so downstream (projector / perceiver) gets [B, N, D].
        hidden_states = outputs.last_hidden_state          # [B, 768, H, W]
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.permute(0, 2, 3, 1)   # [B, H, W, C]
            B2, H, W, C = hidden_states.shape
            hidden_states = hidden_states.reshape(B2, H * W, C)  # [B, H*W, C]

        # Apply dropout
        hidden_states = self.dropout_layer(hidden_states)

        return hidden_states

    def get_output_dim(self) -> int:
        """Returns the actual output dimension of the encoder (768 for HTSAT)."""
        return self.hidden_size
