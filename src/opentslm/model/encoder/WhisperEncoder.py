# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from transformers import WhisperModel, WhisperFeatureExtractor

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class WhisperEncoder(TimeSeriesEncoderBase):
    """
    Audio encoder using OpenAI's pretrained Whisper model.

    This encoder uses only the encoder part of Whisper (not the decoder).
    Whisper processes audio via mel-spectrograms internally and uses a
    transformer encoder. Like Wav2Vec2, it's typically used frozen.

    Architecture:
        Input [B, L] raw audio (16kHz)
            │
            ▼
        Mel-Spectrogram (80 bins, internal)
            │
            ▼
        2 Conv layers (feature projection)
            │
            ▼
        Transformer Encoder (varies by model size)
            │
            ▼
        Output [B, T, hidden_size]

    Model sizes:
        - whisper-tiny:  4 layers,  384 dim,  ~39M params
        - whisper-base:  6 layers,  512 dim,  ~74M params
        - whisper-small: 12 layers, 768 dim,  ~244M params
        - whisper-medium: 24 layers, 1024 dim, ~769M params

    Args:
        output_dim: Output dimension (inherited, actual dim determined by model)
        dropout: Dropout probability
        model_name: HuggingFace model identifier for Whisper
        freeze_encoder: If True, freeze Whisper weights during training
        sample_rate: Expected audio sample rate (must be 16000 for Whisper)
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        model_name: str = "openai/whisper-tiny",
        freeze_encoder: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__(output_dim, dropout)

        if sample_rate != 16000:
            raise ValueError("Whisper requires 16kHz audio input")

        self.sample_rate = sample_rate
        self.freeze_encoder = freeze_encoder
        self.model_name = model_name

        # Load pretrained Whisper model (encoder only)
        self.whisper = WhisperModel.from_pretrained(model_name)
        
        # We only need the encoder, not the decoder
        self.encoder = self.whisper.encoder
        
        # Load feature extractor for mel-spectrogram conversion
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get the hidden size from the model config
        self.hidden_size = self.whisper.config.d_model

        # Optional dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], batch of raw audio waveforms
               where L is the number of audio samples at 16kHz

        Returns:
            FloatTensor of shape [B, T, hidden_size], where T is the number of
            time frames (1500 for Whisper's 30s window)
        """
        B, L = x.shape
        device = x.device

        # Convert to mel-spectrogram using Whisper's feature extractor
        # The feature extractor expects numpy arrays or lists
        audio_list = [x[i].cpu().numpy() for i in range(B)]
        
        # Extract mel features with padding to Whisper's expected length (30 seconds = 3000 mel frames)
        # Whisper expects fixed-length input of 30 seconds
        inputs = self.feature_extractor(
            audio_list,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding="max_length",      # Pad to max_length
            max_length=480000,         # 30 seconds at 16kHz
            truncation=True,           # Truncate if longer than 30s
        )
        
        # Move to same device as input
        input_features = inputs.input_features.to(device)

        # Pass through Whisper encoder
        encoder_outputs = self.encoder(input_features)
        
        # Get the last hidden state
        hidden_states = encoder_outputs.last_hidden_state

        # Apply dropout
        hidden_states = self.dropout_layer(hidden_states)

        return hidden_states

    def get_output_dim(self) -> int:
        """Returns the actual output dimension of the encoder."""
        return self.hidden_size
