# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class RawAudioTokenizer(TimeSeriesEncoderBase):
    """
    Light audio encoder for Flamingo architecture - NO transformer layers.
    
    This encoder matches the CNNTokenizer pattern used in OpenTSLMFlamingo.
    It only does patch embedding + positional encoding, relying on Flamingo's
    cross-attention (perceiver resampler) to process patch relationships.
    
    Architecture:
        Input [B, L] raw audio
            │
            ▼
        Conv1D patch embedding (kernel=patch_size, stride=patch_size)
            │
            ▼
        + Positional embeddings (learnable)
            │
            ▼
        LayerNorm + Dropout
            │
            ▼
        Output [B, N, embed_dim] where N = L // patch_size
    
    Why no transformer layers?
        - Flamingo has cross-attention layers that process patches contextually
        - The perceiver resampler in Flamingo does the heavy lifting
        - Adding transformer here would be redundant computation
    
    Args:
        output_dim: Output embedding dimension (default: 128)
        dropout: Dropout probability (default: 0.0)
        patch_size: Number of audio samples per patch (default: 640 = 40ms at 16kHz)
        max_patches: Maximum sequence length in patches (default: 768 = 30s audio)
                     Audio exceeding max_patches is truncated; shorter audio is fine.
    
    Notes:
        - Audio is automatically zero-padded to the nearest multiple of patch_size.
        - Audio exceeding max_patches is silently truncated (no error).
    
    Example:
        >>> encoder = RawAudioTokenizer(output_dim=128, patch_size=640)
        >>> audio = torch.randn(2, 80000)  # 2 audio clips, 5 seconds at 16kHz
        >>> output = encoder(audio)  # [2, 125, 128]
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        patch_size: int = 640,  # 40ms at 16kHz
        max_patches: int = 768,  # 30s at 16kHz: 480000/640 = 750 patches, +buffer
    ):
        super().__init__(output_dim, dropout)
        self.patch_size = patch_size
        self._output_dim = output_dim

        # Conv1d patch embedding: (B, 1, L) -> (B, output_dim, L/patch_size)
        # This is the same as ViT's patch embedding but for 1D audio
        self.patch_embed = nn.Conv1d(
            in_channels=1,
            out_channels=output_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )

        # Learnable positional embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_patches, output_dim)
        )

        # Input normalization and dropout
        self.input_norm = nn.LayerNorm(output_dim)
        self.input_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for raw audio.

        Args:
            x: FloatTensor of shape [B, L], batch of raw audio waveforms
               where L is the number of audio samples.
               Audio is automatically right-padded with zeros if not divisible
               by patch_size, and truncated if it exceeds max_patches.

        Returns:
            FloatTensor of shape [B, N, output_dim], where N = ceil(L / patch_size)
        """
        B, L = x.shape

        # Auto-pad audio length to the nearest multiple of patch_size
        remainder = L % self.patch_size
        if remainder != 0:
            pad_len = self.patch_size - remainder
            x = nn.functional.pad(x, (0, pad_len), value=0.0)

        # Truncate to max_patches if audio is too long
        max_samples = self.pos_embed.size(1) * self.patch_size
        if x.size(1) > max_samples:
            x = x[:, :max_samples]

        # Reshape to (B, 1, L) for Conv1d
        x = x.unsqueeze(1)

        # Conv patch embedding -> (B, output_dim, N)
        x = self.patch_embed(x)

        # Transpose to (B, N, output_dim)
        x = x.transpose(1, 2)

        # Add positional embeddings
        N = x.size(1)
        x = x + self.pos_embed[:, :N, :]

        # Normalize and apply dropout
        x = self.input_norm(x)
        x = self.input_dropout(x)

        return x

    def get_output_dim(self) -> int:
        """Returns the output dimension of the encoder."""
        return self._output_dim
    
    def get_num_patches(self, audio_length: int) -> int:
        """Calculate number of patches for a given audio length."""
        return audio_length // self.patch_size
