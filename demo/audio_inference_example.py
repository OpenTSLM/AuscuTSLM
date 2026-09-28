#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Simple example script to test audio language model inference.

This script demonstrates how to:
1. Load a pretrained audio language model
2. Process an audio file
3. Generate text descriptions/transcriptions

Usage:
    python demo/audio_inference_example.py --audio_path path/to/audio.wav --task captioning
"""

import argparse
import torch
from opentslm.model.encoder.Wav2Vec2Encoder import Wav2Vec2Encoder
from opentslm.model.projector.MLPProjector import MLPProjector
from opentslm.model.llm.OpenTSLMFlamingo import OpenTSLMFlamingo
from opentslm.time_series_datasets.audio_util import load_audio, preprocess_audio
from opentslm.prompt.audio_prompts import (
    get_audio_captioning_prompt,
    get_speech_recognition_prompt,
)
from opentslm.audio_model_config import (
    AUDIO_SAMPLE_RATE,
    AUDIO_MAX_LENGTH,
    AUDIO_PATCH_SIZE,
    AUDIO_ENCODER_MODEL,
)


def create_model(llm_id: str, checkpoint_path: str = None, device: str = "cuda"):
    """
    Create and optionally load a pretrained audio language model using Flamingo architecture.

    Args:
        llm_id: HuggingFace LLM model ID
        checkpoint_path: Path to model checkpoint (optional)
        device: Device to load model on

    Returns:
        Loaded model
    """
    print(f"Creating audio language model (Flamingo) with LLM: {llm_id}")

    # Create encoder
    encoder = Wav2Vec2Encoder(
        model_name=AUDIO_ENCODER_MODEL,
        freeze_encoder=True,
    )

    # Create Flamingo model (loads LLM backbone)
    model = OpenTSLMFlamingo(device=device, llm_id=llm_id, use_lora=True)

    # Create projector mapping encoder output -> Flamingo visual dim
    from opentslm.model_config import ENCODER_OUTPUT_DIM
    projector = MLPProjector(
        input_dim=encoder.get_output_dim(),
        output_dim=ENCODER_OUTPUT_DIM,
        device=device,
    )

    # Wrap encoder to apply projector so Flamingo receives expected vis_dim
    import torch.nn as nn

    class EncoderWithProjector(nn.Module):
        def __init__(self, encoder, projector):
            super().__init__()
            self.encoder = encoder
            self.projector = projector

        def forward(self, x):
            enc = self.encoder(x)
            b, n, d = enc.shape
            proj = self.projector(enc.view(b * n, d)).view(b, n, -1)
            return proj

    wrapped_encoder = EncoderWithProjector(encoder, projector).to(device)

    # Attach encoder/projector to the Flamingo model
    model.encoder = encoder
    model.projector = projector
    model.model.vision_encoder = wrapped_encoder

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"Loading checkpoint from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model = model.to(device)
    model.eval()

    return model


def process_audio(audio_path: str, task: str = "captioning"):
    """
    Process audio file and create input batch.

    Args:
        audio_path: Path to audio file
        task: Task type ("captioning" or "transcription")

    Returns:
        Batch dictionary ready for model input
    """
    print(f"Loading audio from {audio_path}")

    # Load and preprocess audio
    waveform, sr = load_audio(audio_path, target_sample_rate=AUDIO_SAMPLE_RATE)
    waveform = preprocess_audio(
        waveform,
        sample_rate=sr,
        target_sample_rate=AUDIO_SAMPLE_RATE,
        normalize=True,
        to_mono=True,
        max_length=AUDIO_MAX_LENGTH,
    )

    # Create prompt based on task
    if task == "captioning":
        pre_prompt = get_audio_captioning_prompt()
        post_prompt = "Description:"
    elif task == "transcription":
        pre_prompt = get_speech_recognition_prompt()
        post_prompt = "Transcription:"
    else:
        raise ValueError(f"Unknown task: {task}")

    # Create batch (single sample)
    batch = [{
        "pre_prompt": pre_prompt,
        "time_series": [waveform.tolist()],
        "time_series_text": ["Audio"],
        "post_prompt": post_prompt,
        "answer": "",  # Empty for inference
    }]

    return batch


def generate_text(model, batch, max_new_tokens: int = 100):
    """
    Generate text from audio using the model.

    Args:
        model: Audio language model
        batch: Input batch
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    print("Generating text...")

    with torch.no_grad():
        # Preprocess batch
        from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
        batch = extend_time_series_to_match_patch_size_and_aggregate(
            batch,
            patch_size=AUDIO_PATCH_SIZE,
            normalize=True
        )

        # Generate (this is a simplified version - actual generation depends on model implementation)
        # For a complete implementation, you'd need to use the model's generate method
        output = model.generate(batch, max_new_tokens=max_new_tokens)

    return output


def main():
    parser = argparse.ArgumentParser(description="Audio Language Model Inference (Flamingo)")
    parser.add_argument("--audio_path", type=str, required=True,
                       help="Path to audio file")
    parser.add_argument("--task", type=str, default="captioning",
                       choices=["captioning", "transcription"],
                       help="Task type")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace LLM model ID")
    parser.add_argument("--checkpoint", type=str, default=None,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--max_tokens", type=int, default=100,
                       help="Maximum tokens to generate")

    args = parser.parse_args()

    print("="*60)
    print("Audio Language Model Inference (OpenTSLMFlamingo)")
    print("="*60)
    print(f"Audio: {args.audio_path}")
    print(f"Task: {args.task}")
    print(f"Device: {args.device}")
    print()

    # Create model
    model = create_model(args.llm_id, args.checkpoint, args.device)

    # Process audio
    batch = process_audio(args.audio_path, args.task)

    # Generate text
    try:
        output = generate_text(model, batch, args.max_tokens)
        print("\n" + "="*60)
        print("Generated Output:")
        print("="*60)
        print(output)
    except Exception as e:
        print(f"\nError during generation: {e}")
        print("\nNote: This example requires a trained model checkpoint.")
        print("Train a model first using scripts/training/audio_curriculum_learning.py")

    print("\n" + "="*60)


if __name__ == "__main__":
    main()
