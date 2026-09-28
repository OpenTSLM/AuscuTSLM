# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class Wav2Vec2Encoder(TimeSeriesEncoderBase):
    """
    Audio encoder using pretrained Wav2Vec2 model.

    This encoder processes raw audio waveforms using Meta's Wav2Vec2 model,
    which has been pretrained on large-scale speech data. The encoder can be
    used in frozen mode (for efficiency) or fine-tuned.

    Args:
        output_dim: Output dimension (inherited from base, not used directly)
        dropout: Dropout probability
        model_name: HuggingFace model identifier for Wav2Vec2
        freeze_encoder: If True, freeze Wav2Vec2 weights during training
        sample_rate: Expected audio sample rate (default: 16000 Hz)
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        model_name: str = "facebook/wav2vec2-base",
        freeze_encoder: bool = True,
        sample_rate: int = 16000,
    ):
        super().__init__(output_dim, dropout)

        self.sample_rate = sample_rate
        self.freeze_encoder = freeze_encoder

        # Load pretrained Wav2Vec2 model
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        # Get the hidden size from the model config
        self.hidden_size = self.wav2vec2.config.hidden_size

        # Optional dropout layer
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FloatTensor of shape [B, L], batch of raw audio waveforms
               where L is the number of audio samples (e.g., 16000 samples = 1 second at 16kHz)

        Returns:
            FloatTensor of shape [B, N, hidden_size], where N is the number of
            temporal features extracted by Wav2Vec2 (typically L/320 for base model)
        """

        # Wav2Vec2 expects input of shape [B, L]
        # It will internally process and downsample the audio
        outputs = self.wav2vec2(x)

        # Extract the last hidden state: [B, N, hidden_size]
        hidden_states = outputs.last_hidden_state

        # Apply dropout
        hidden_states = self.dropout_layer(hidden_states)

        return hidden_states

    def get_output_dim(self) -> int:
        """Returns the actual output dimension of the encoder."""
        return self.hidden_size
