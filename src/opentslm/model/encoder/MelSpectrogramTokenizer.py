# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

import torch
import torch.nn as nn
import torchaudio.transforms as T

from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.model.encoder.TimeSeriesEncoderBase import TimeSeriesEncoderBase


class MelSpectrogramTokenizer(TimeSeriesEncoderBase):
    """
    Light mel-spectrogram encoder for Flamingo - NO transformer layers.
    
    This encoder converts raw audio to mel-spectrograms and processes them
    with a CNN, but WITHOUT transformer layers. Designed for Flamingo architecture
    where cross-attention provides the contextual processing.
    
    Architecture:
        Input [B, L] raw audio
            │
            ▼
        Mel-Spectrogram Transform (n_mels frequency bins)
            │ [B, n_mels, T]
            ▼
        2D CNN (3 conv blocks with pooling)
            │ [B, embed_dim, freq', T']
            ▼
        Global Average Pool (over frequency)
            │ [B, embed_dim, T']
            ▼
        Transpose + Positional Embeddings
            │ [B, T', embed_dim]
            ▼
        LayerNorm + Dropout
            │
            ▼
        Output [B, T', embed_dim]
    
    Why no transformer layers?
        - Flamingo has cross-attention layers that process patches contextually
        - The CNN already extracts local frequency patterns
        - Adding transformer here would be redundant computation
    
    Args:
        output_dim: Output embedding dimension (default: 256)
        dropout: Dropout probability (default: 0.0)
        sample_rate: Audio sample rate (default: 16000 Hz)
        n_fft: FFT window size (default: 400 = 25ms at 16kHz)
        hop_length: Hop length for STFT (default: 160 = 10ms at 16kHz)
        n_mels: Number of mel filterbanks (default: 64)
        max_time_frames: Maximum number of time frames (default: 1000)
    
    Example:
        >>> encoder = MelSpectrogramTokenizer(output_dim=256, n_mels=64)
        >>> audio = torch.randn(2, 80000)  # 2 audio clips, 5 seconds at 16kHz
        >>> output = encoder(audio)  # [2, T', 256] where T' depends on hop_length
    """

    def __init__(
        self,
        output_dim: int = ENCODER_OUTPUT_DIM,
        dropout: float = 0.0,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 64,
        max_time_frames: int = 1000,
    ):
        super().__init__(output_dim, dropout)

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self._output_dim = output_dim

        # 1) Mel-Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

        # Convert to log scale
        self.amplitude_to_db = T.AmplitudeToDB()

        # 2) 2D CNN for feature extraction from mel-spectrogram
        # Input: (B, 1, n_mels, time_frames)
        # Output: (B, output_dim, n_mels', time_frames')
        self.cnn = nn.Sequential(
            # Conv block 1: 1 -> 32 channels
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Downsample freq & time by 2

            # Conv block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # Downsample by 2

            # Conv block 3: 64 -> output_dim channels
            nn.Conv2d(64, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
        )

        # After 2 pooling layers in time: time_frames -> time_frames / 4
        self.time_downsample_factor = 4

        # 3) Positional embeddings for time dimension
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_time_frames, output_dim)
        )

        # 4) Layer norm and dropout (NO transformer!)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for mel-spectrogram tokenizer.

        Args:
            x: FloatTensor of shape [B, L], batch of raw audio waveforms
               where L is the number of audio samples

        Returns:
            FloatTensor of shape [B, T', output_dim], where T' is the number of time frames
            after CNN downsampling
        """
        B, L = x.shape

        # 1) Convert to mel-spectrogram
        # Output: (B, n_mels, time_frames)
        mel_spec = self.mel_transform(x)

        # Convert to log scale (dB)
        mel_spec = self.amplitude_to_db(mel_spec)

        # Normalize per sample
        mel_spec = (mel_spec - mel_spec.mean(dim=(1, 2), keepdim=True)) / (
            mel_spec.std(dim=(1, 2), keepdim=True) + 1e-6
        )

        # Add channel dimension: (B, 1, n_mels, time_frames)
        mel_spec = mel_spec.unsqueeze(1)

        # 2) Apply CNN
        # Output: (B, output_dim, freq_dim', time_frames')
        features = self.cnn(mel_spec)

        # 3) Global average pool over frequency dimension
        # (B, output_dim, freq_dim', time_frames') -> (B, output_dim, time_frames')
        features = features.mean(dim=2)

        # Transpose to (B, time_frames', output_dim)
        features = features.transpose(1, 2)

        # 4) Add positional embeddings (truncate if longer than max_time_frames)
        T_frames = features.size(1)
        if T_frames > self.pos_embed.size(1):
            features = features[:, :self.pos_embed.size(1), :]
            T_frames = self.pos_embed.size(1)
        features = features + self.pos_embed[:, :T_frames, :]

        # 5) Layer norm + dropout (NO transformer - Flamingo does the rest!)
        features = self.output_norm(features)
        features = self.output_dropout(features)

        return features

    def get_output_dim(self) -> int:
        """Returns the output dimension of the encoder."""
        return self._output_dim

    def get_num_frames(self, audio_length: int) -> int:
        """
        Calculate number of output frames for a given audio length.
        
        Args:
            audio_length: Number of audio samples
            
        Returns:
            Number of output time frames
        """
        # Mel-spectrogram frames
        mel_frames = (audio_length // self.hop_length) + 1
        # After CNN pooling
        return mel_frames // self.time_downsample_factor
