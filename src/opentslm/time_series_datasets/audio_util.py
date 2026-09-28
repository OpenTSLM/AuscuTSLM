# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Audio preprocessing utilities for audio language models.

This module provides functions for loading, preprocessing, and augmenting
audio data for use with audio language models.
"""

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import soundfile as sf
import warnings
from typing import Tuple, Optional
from opentslm.model_config import PATCH_SIZE


def load_audio(
    audio_path: str,
    target_sample_rate: int = 16000
) -> Tuple[torch.Tensor, int]:
    """
    Load audio file and resample to target sample rate.

    Uses soundfile as fallback when TorchCodec is not available.

    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate in Hz (default: 16000)

    Returns:
        Tuple of (waveform, sample_rate) where waveform is shape [channels, samples]
    """
    try:
        # Try using torchaudio first (requires TorchCodec in newer versions)
        waveform, sample_rate = torchaudio.load(audio_path)
    except (ImportError, ModuleNotFoundError):
        # Fallback to soundfile if TorchCodec is not available
        try:
            audio_data, sample_rate = sf.read(audio_path)
            # Convert to tensor and ensure correct shape [channels, samples]
            if audio_data.ndim == 1:
                waveform = torch.tensor(audio_data, dtype=torch.float32).unsqueeze(0)
            else:
                waveform = torch.tensor(audio_data, dtype=torch.float32).T
        except Exception as e:
            raise RuntimeError(
                f"Failed to load audio file {audio_path}. "
                f"TorchCodec is not available, and soundfile failed with: {e}"
            ) from e

    # Resample if necessary
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    return waveform, sample_rate


def preprocess_audio(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    target_sample_rate: int = 16000,
    normalize: bool = True,
    to_mono: bool = True,
    patch_size: int = PATCH_SIZE,
    max_length: Optional[int] = None,
) -> torch.Tensor:
    """
    Preprocess audio waveform for model input.

    Args:
        waveform: Audio tensor of shape [channels, samples] or [samples]
        sample_rate: Current sample rate
        target_sample_rate: Target sample rate
        normalize: Whether to normalize to zero mean and unit variance
        to_mono: Whether to convert to mono
        patch_size: Patch size for padding (length will be multiple of this)
        max_length: Maximum length in samples (will truncate if longer)

    Returns:
        Preprocessed audio tensor of shape [samples]
    """
    # Ensure tensor format
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # Handle shape
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # Add channel dimension

    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Convert to mono if needed
    if to_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Remove channel dimension
    waveform = waveform.squeeze(0)

    # Truncate if too long
    if max_length is not None and waveform.shape[0] > max_length:
        waveform = waveform[:max_length]

    # Normalize
    if normalize:
        mean = waveform.mean()
        std = waveform.std()
        if std > 1e-8:
            waveform = (waveform - mean) / std
        else:
            waveform = waveform - mean

    # Pad to multiple of patch_size
    current_length = waveform.shape[0]
    pad_length = (patch_size - current_length % patch_size) % patch_size
    if pad_length > 0:
        waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)

    return waveform


def extract_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int = 16000,
    n_fft: int = 400,
    hop_length: int = 160,
    n_mels: int = 80,
) -> torch.Tensor:
    """
    Extract mel-spectrogram features from audio waveform.

    Args:
        waveform: Audio tensor of shape [samples] or [channels, samples]
        sample_rate: Sample rate in Hz
        n_fft: FFT window size
        hop_length: Hop length for STFT
        n_mels: Number of mel filterbanks

    Returns:
        Mel-spectrogram of shape [n_mels, time_frames]
    """
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    mel_spec = mel_transform(waveform)

    # Convert to log scale
    mel_spec = torch.log(mel_spec + 1e-9)

    return mel_spec.squeeze(0)


def apply_time_stretch(
    waveform: torch.Tensor,
    rate: float = 1.0,
) -> torch.Tensor:
    """
    Apply time stretching augmentation.

    Args:
        waveform: Audio tensor of shape [samples]
        rate: Stretch rate (>1.0 speeds up, <1.0 slows down)

    Returns:
        Time-stretched waveform
    """
    if rate == 1.0:
        return waveform

    # Simple implementation using interpolation
    original_length = waveform.shape[0]
    new_length = int(original_length / rate)

    stretched = F.interpolate(
        waveform.unsqueeze(0).unsqueeze(0),
        size=new_length,
        mode='linear',
        align_corners=False
    )

    return stretched.squeeze(0).squeeze(0)


def apply_noise(
    waveform: torch.Tensor,
    noise_level: float = 0.005,
) -> torch.Tensor:
    """
    Add Gaussian noise to audio.

    Args:
        waveform: Audio tensor of shape [samples]
        noise_level: Standard deviation of noise

    Returns:
        Noisy waveform
    """
    noise = torch.randn_like(waveform) * noise_level
    return waveform + noise


def remove_silence(
    waveform: torch.Tensor,
    threshold: float = 0.01,
    min_silence_duration: int = 1600,  # 100ms at 16kHz
) -> torch.Tensor:
    """
    Remove silence from audio using simple energy-based VAD.

    Args:
        waveform: Audio tensor of shape [samples]
        threshold: Energy threshold for silence detection
        min_silence_duration: Minimum silence duration in samples

    Returns:
        Audio with silence removed
    """
    # Compute energy in sliding windows
    window_size = min_silence_duration
    energy = torch.zeros(len(waveform) - window_size + 1)

    for i in range(len(energy)):
        window = waveform[i:i + window_size]
        energy[i] = (window ** 2).mean()

    # Find non-silent regions
    non_silent = energy > threshold

    if not non_silent.any():
        return waveform  # Return original if all silent

    # Expand to original length
    non_silent_expanded = torch.zeros(len(waveform), dtype=torch.bool)
    for i in range(len(non_silent)):
        if non_silent[i]:
            non_silent_expanded[i:i + window_size] = True

    return waveform[non_silent_expanded]
