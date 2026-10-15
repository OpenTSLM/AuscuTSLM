#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test Audio Encoders Script

This script tests different audio encoders with Flamingo architecture:
1. RawAudioTokenizer - Light encoder for Flamingo (RECOMMENDED)
2. MelSpectrogramTokenizer - Uses mel-spectrogram features
3. Wav2Vec2Encoder - Uses pretrained Wav2Vec2 (baseline)
4. WhisperEncoder - Uses pretrained Whisper
5. CLAPEncoder - Uses pretrained CLAP (LAION)

Usage:
    # Quick test (encoder only, no LLM)
    python scripts/evaluation/test_audio_encoders.py --quick_test --encoder tokenizer
    python scripts/evaluation/test_audio_encoders.py --quick_test --encoder all

    # Full test with Flamingo LLM
    python scripts/evaluation/test_audio_encoders.py --encoder tokenizer
    python scripts/evaluation/test_audio_encoders.py --encoder all
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import argparse
import torch
import random
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from opentslm.model.encoder.RawAudioTokenizer import RawAudioTokenizer
from opentslm.model.encoder.MelSpectrogramTokenizer import MelSpectrogramTokenizer
from opentslm.model.encoder.Wav2Vec2Encoder import Wav2Vec2Encoder
from opentslm.model.encoder.WhisperEncoder import WhisperEncoder
from opentslm.model.encoder.CLAPEncoder import CLAPEncoder
from opentslm.model.projector.MLPProjector import MLPProjector
from opentslm.model.llm.AudioFlamingo import AudioFlamingo
from opentslm.model_config import ENCODER_OUTPUT_DIM
from opentslm.time_series_datasets.audio.CareSoundDataset import CareSoundDataset
from opentslm.time_series_datasets.util import extend_time_series_to_match_patch_size_and_aggregate
from opentslm.time_series_datasets.audio_util import load_audio, preprocess_audio
from opentslm.audio_model_config import (
    AUDIO_BATCH_SIZE,
    AUDIO_PATCH_SIZE,
    AUDIO_EMBED_DIM,
)

# Memory-saving defaults (override via command line)
MAX_AUDIO_SECONDS = 5  # Reduce from 30s to 5s
SAMPLE_RATE = 16000


def find_audio_files(audio_dir: str, num_files: int = 100) -> list:
    """
    Find audio files in directory and return random sample.
    
    Args:
        audio_dir: Directory to search for audio files
        num_files: Number of random files to return
        
    Returns:
        List of audio file paths
    """
    audio_files = []
    extensions = ['.wav', '.mp3', '.flac', '.ogg', '.WAV', '.MP3', '.FLAC', '.OGG']
    
    for ext in extensions:
        audio_files.extend(Path(audio_dir).rglob(f'*{ext}'))
    
    # Filter out hidden files
    audio_files = [f for f in audio_files if not f.name.startswith('.') and not f.name.startswith('._')]
    
    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {audio_dir}")
    
    # Random sample
    if len(audio_files) > num_files:
        audio_files = random.sample(audio_files, num_files)
    
    return audio_files


def test_encoder_only(
    encoder_type: str,
    audio_dir: str,
    device: str,
    num_files: int = 100,
    embed_dim: int = 256,
    num_layers: int = 2,
    patch_size: int = 640,
    max_audio_seconds: int = 5,
):
    """
    Test encoder on random audio files from a directory (no full model, no dataset loading).
    
    This is a quick functionality test to verify the encoder works.
    
    Args:
        encoder_type: Type of encoder ('raw', 'mel', 'wav2vec2')
        audio_dir: Directory containing audio files
        device: Device to run on
        num_files: Number of random files to test
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        patch_size: Patch size for audio
        max_audio_seconds: Maximum audio length in seconds
    """
    print("="*70)
    print(f"TESTING {encoder_type.upper()} ENCODER ON RANDOM AUDIO FILES")
    print("="*70)
    print()
    print(f"Audio directory: {audio_dir}")
    print(f"Number of files to test: {num_files}")
    print(f"Device: {device}")
    print(f"Embed dim: {embed_dim}")
    print(f"Num layers: {num_layers}")
    print(f"Patch size: {patch_size} ({patch_size/16:.0f}ms at 16kHz)")
    print(f"Max audio: {max_audio_seconds}s")
    print()
    
    # Find audio files
    print("Finding audio files...")
    audio_files = find_audio_files(audio_dir, num_files)
    print(f"Found {len(audio_files)} audio files to test")
    print()
    
    # Create encoder only (no full model - much faster and less memory)
    print(f"Creating {encoder_type.upper()} encoder...")
    
    if encoder_type == "tokenizer":
        encoder = RawAudioTokenizer(
            output_dim=embed_dim,
            patch_size=patch_size,
            dropout=0.0,
            max_patches=512,
        )
    elif encoder_type == "mel":
        encoder = MelSpectrogramTokenizer(
            output_dim=embed_dim,
            sample_rate=SAMPLE_RATE,
            n_mels=64,
            dropout=0.0,
            max_time_frames=1000,
        )
    elif encoder_type == "wav2vec2":
        encoder = Wav2Vec2Encoder(
            output_dim=embed_dim,
            model_name="facebook/wav2vec2-base",
            freeze_encoder=True,
        )
    elif encoder_type == "whisper":
        encoder = WhisperEncoder(
            output_dim=embed_dim,
            model_name="openai/whisper-tiny",  # Smallest model for testing
            freeze_encoder=True,
        )
    elif encoder_type == "clap":
        encoder = CLAPEncoder(
            output_dim=embed_dim,
            model_name="laion/clap-htsat-unfused",
            freeze_encoder=True,
        )
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    encoder = encoder.to(device)
    encoder.eval()
    # If encoder output dimension doesn't match desired embed_dim, add a projector
    enc_output_dim = None
    if hasattr(encoder, "get_output_dim"):
        try:
            enc_output_dim = encoder.get_output_dim()
        except Exception:
            enc_output_dim = None

    if enc_output_dim is None:
        enc_output_dim = getattr(encoder, "hidden_size", None)

    if enc_output_dim is not None and enc_output_dim != embed_dim:
        from opentslm.model.projector.MLPProjector import MLPProjector
        import torch.nn as nn

        projector = MLPProjector(enc_output_dim, embed_dim, device).to(device)
        # Wrap encoder + projector so forward returns the projected embeddings
        encoder = nn.Sequential(encoder, projector)

    # Count parameters (including any projector)
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder parameters: {total_params:,}")
    print()
    
    # Test on audio files
    print("Testing encoder on audio files...")
    print("-"*70)
    
    max_audio_length = max_audio_seconds * SAMPLE_RATE
    
    success = 0
    errors = 0
    
    with torch.no_grad():
        for i, audio_path in enumerate(tqdm(audio_files, desc="Processing", unit="files")):
            try:
                # Load and preprocess audio
                waveform, sr = load_audio(str(audio_path), SAMPLE_RATE)
                waveform = preprocess_audio(
                    waveform,
                    sample_rate=sr,
                    target_sample_rate=SAMPLE_RATE,
                    normalize=True,
                    to_mono=True,
                    max_length=max_audio_length,
                )
                
                # Pad to patch_size multiple
                audio_len = len(waveform)
                remainder = audio_len % patch_size
                if remainder != 0:
                    pad_len = patch_size - remainder
                    waveform = torch.nn.functional.pad(
                        torch.tensor(waveform), (0, pad_len), value=0.0
                    ).numpy()
                
                # Convert to tensor and add batch dimension
                audio_tensor = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Forward pass
                output = encoder(audio_tensor)
                
                # Verify output shape
                B, N, D = output.shape
                expected_patches = len(waveform) // patch_size
                
                if B != 1:
                    raise ValueError(f"Batch size mismatch: expected 1, got {B}")
                if D != embed_dim:
                    raise ValueError(f"Embed dim mismatch: expected {embed_dim}, got {D}")
                
                success += 1
                
                # Print details for first few files
                if i < 3:
                    print(f"\n  File {i+1}: {audio_path.name}")
                    print(f"    Input: {audio_tensor.shape} ({audio_len/SAMPLE_RATE:.2f}s)")
                    print(f"    Output: {output.shape} ({N} patches)")
                    print(f"    ✓ Success!")
                
            except Exception as e:
                errors += 1
                if errors <= 5:  # Only print first 5 errors
                    print(f"\n  ✗ Error with {audio_path.name}: {e}")
    
    # Summary
    print()
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    print()
    print(f"Total files tested: {len(audio_files)}")
    print(f"✓ Successful: {success} ({success/len(audio_files)*100:.1f}%)")
    print(f"✗ Errors: {errors} ({errors/len(audio_files)*100:.1f}%)")
    print()
    
    if success == len(audio_files):
        print("✓ All tests passed! Encoder is working correctly.")
    elif success > 0:
        print("⚠ Some tests passed. Check error messages above.")
    else:
        print("✗ All tests failed. Check encoder configuration.")
    
    print()
    return success, errors


def create_model_with_encoder(
    encoder_type: str, 
    llm_id: str, 
    device: str,
    embed_dim: int = 256,  # Reduced from 768
    num_layers: int = 2,   # Reduced from 4-6
    patch_size: int = 640, # Increased from 320 (40ms patches = fewer patches)
):
    """
    Create OpenTSLMFlamingo model with specified encoder.

    Args:
        encoder_type: Type of encoder ('raw', 'mel', 'wav2vec2')
        llm_id: HuggingFace LLM model ID
        device: Device to load model on

    Returns:
        Initialized model
    """
    print(f"\n{'='*70}")
    print(f"Creating model with {encoder_type.upper()} encoder")
    print(f"{'='*70}\n")

    # Print memory-saving settings
    print(f"  Embed dim: {embed_dim}")
    print(f"  Num layers: {num_layers}")
    print(f"  Patch size: {patch_size}")
    
    # Calculate expected patches for reference
    max_audio_samples = MAX_AUDIO_SECONDS * SAMPLE_RATE
    expected_patches = max_audio_samples // patch_size
    print(f"  Max audio: {MAX_AUDIO_SECONDS}s = {max_audio_samples} samples")
    print(f"  Expected patches: ~{expected_patches}")
    print()

    # Create encoder based on type
    if encoder_type == "tokenizer":
        print("Encoder: RawAudioTokenizer (light, NO transformer - BEST for Flamingo)")
        encoder = RawAudioTokenizer(
            output_dim=embed_dim,
            patch_size=patch_size,
            dropout=0.1,
            max_patches=512,
        )

    elif encoder_type == "mel":
        print("Encoder: MelSpectrogramTokenizer (light, CNN only - for Flamingo)")
        encoder = MelSpectrogramTokenizer(
            output_dim=embed_dim,
            sample_rate=16000,
            n_mels=64,
            dropout=0.1,
            max_time_frames=1000,
        )

    elif encoder_type == "wav2vec2":
        print("Encoder: Wav2Vec2Encoder (pretrained, frozen)")
        encoder = Wav2Vec2Encoder(
            output_dim=embed_dim,
            model_name="facebook/wav2vec2-base",
            freeze_encoder=True,
        )

    elif encoder_type == "whisper":
        print("Encoder: WhisperEncoder (pretrained, frozen)")
        encoder = WhisperEncoder(
            output_dim=embed_dim,
            model_name="openai/whisper-tiny",
            freeze_encoder=True,
        )

    elif encoder_type == "clap":
        print("Encoder: CLAPEncoder (pretrained, frozen)")
        encoder = CLAPEncoder(
            output_dim=embed_dim,
            model_name="laion/clap-htsat-unfused",
            freeze_encoder=True,
        )

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # Get encoder output dimension
    encoder_output_dim = encoder.get_output_dim()
    print(f"Encoder output dimension: {encoder_output_dim}")

    # Create projector
    print(f"Creating projector: {encoder_output_dim} -> LLM hidden dim")
    projector = MLPProjector(
        input_dim=encoder_output_dim,
        output_dim=ENCODER_OUTPUT_DIM,
        device=device,
    )

    # Create Flamingo model
    print(f"Creating AudioFlamingo with LLM: {llm_id}")
    model = AudioFlamingo(
        encoder=encoder,
        projector=projector,
        device=device,
        llm_id=llm_id,
        vis_dim=ENCODER_OUTPUT_DIM,
        freeze_encoder=True,
    )

    model = model.to(device)
    print(f"✓ Model created and moved to {device}")

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {total_params - trainable_params:,}")

    return model


def test_forward_pass(model, dataset, device, num_samples=3, patch_size=640):
    """
    Test forward pass with a few samples.

    Args:
        model: The model to test
        dataset: Dataset to load samples from
        device: Device to run on
        num_samples: Number of samples to test
        patch_size: Patch size for preprocessing
    """
    print(f"\n{'='*70}")
    print(f"Testing Forward Pass")
    print(f"{'='*70}\n")

    model.eval()

    # Create dataloader
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_samples:
                break

            print(f"Sample {i+1}/{num_samples}")
            print("-" * 70)

            # Preprocess batch
            batch = extend_time_series_to_match_patch_size_and_aggregate(
                batch,
                patch_size=patch_size,
                normalize=True
            )

            # Get audio info
            audio_length = len(batch[0]["time_series"])
            print(f"  Audio length: {audio_length} samples")
            print(f"  Question: {batch[0]['post_prompt'][:100]}...")

            try:
                # Forward pass
                loss = model(batch)

                print(f"  ✓ Forward pass successful!")
                print(f"  Loss: {loss.item():.4f}")

            except Exception as e:
                print(f"  ✗ Error during forward pass: {e}")
                import traceback
                traceback.print_exc()

            print()

    print("✓ Forward pass testing complete!")


def compare_encoders(
    audio_dir: str, 
    llm_id: str, 
    device: str,
    embed_dim: int = 256,
    num_layers: int = 2,
    patch_size: int = 640,
    max_audio_seconds: int = 5,
):
    """
    Compare all three encoders.

    Args:
        audio_dir: Directory containing audio files
        llm_id: LLM model ID
        device: Device to run on
        embed_dim: Embedding dimension
        num_layers: Number of transformer layers
        patch_size: Patch size for audio
        max_audio_seconds: Maximum audio length in seconds
    """
    print("\n" + "="*70)
    print("COMPARING AUDIO ENCODERS WITH FLAMINGO")
    print("="*70)

    # Load dataset once with memory-saving settings
    print("\nLoading dataset...")
    max_audio_length = max_audio_seconds * SAMPLE_RATE
    dataset = CareSoundDataset(
        split="train",
        EOS_TOKEN="</s>",
        audio_dir=audio_dir,
        max_audio_length=max_audio_length,
    )
    print(f"✓ Dataset loaded: {len(dataset)} samples")

    # Test each encoder
    encoders = ["tokenizer", "mel", "wav2vec2", "whisper", "clap"]

    for encoder_type in encoders:
        try:
            # Create model with memory-saving settings
            model = create_model_with_encoder(
                encoder_type, 
                llm_id, 
                device,
                embed_dim=embed_dim,
                num_layers=num_layers,
                patch_size=patch_size,
            )

            # Test forward pass
            test_forward_pass(model, dataset, device, num_samples=2, patch_size=patch_size)

            # Clean up
            del model
            torch.cuda.empty_cache() if device == "cuda" else None

        except Exception as e:
            print(f"\n✗ Error with {encoder_type} encoder: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "="*70 + "\n")

    print("✓ Encoder comparison complete!")


def main():
    parser = argparse.ArgumentParser(description="Test audio encoders with Flamingo")
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_organized",
                       help="Directory containing audio files")
    parser.add_argument("--encoder", type=str, default="tokenizer",
                       choices=["tokenizer", "mel", "wav2vec2", "whisper", "clap", "all"],
                       help="Encoder type to test (default: tokenizer - RECOMMENDED for Flamingo)")
    parser.add_argument("--llm_id", type=str, default="meta-llama/Llama-3.2-1B",
                       help="HuggingFace LLM model ID")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")
    parser.add_argument("--num_samples", type=int, default=2,
                       help="Number of samples to test per encoder")
    # Memory-saving options
    parser.add_argument("--embed_dim", type=int, default=256,
                       help="Embedding dimension (default: 256, reduce for less memory)")
    parser.add_argument("--num_layers", type=int, default=2,
                       help="Number of transformer layers (default: 2, reduce for less memory)")
    parser.add_argument("--patch_size", type=int, default=640,
                       help="Patch size (default: 640=40ms, increase for less memory)")
    parser.add_argument("--max_audio_seconds", type=int, default=5,
                       help="Maximum audio length in seconds (default: 5, reduce for less memory)")
    parser.add_argument("--cpu", action="store_true",
                       help="Force CPU mode (slower but uses system RAM)")
    # Quick test mode (encoder only, no full model)
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test: encoder only on random audio files (no LLM, much faster)")
    parser.add_argument("--num_files", type=int, default=100,
                       help="Number of random audio files to test in quick_test mode (default: 100)")

    args = parser.parse_args()
    
    # Override device if --cpu flag is set
    if args.cpu:
        args.device = "cpu"
    
    # Update global settings
    global MAX_AUDIO_SECONDS
    MAX_AUDIO_SECONDS = args.max_audio_seconds

    # Quick test mode - encoder only, no LLM
    if args.quick_test:
        print("="*70)
        print("QUICK TEST MODE - ENCODER ONLY (NO LLM)")
        print("="*70)
        print()
        print("This mode tests only the encoder on random audio files.")
        print("Much faster and uses much less memory than full model testing.")
        print()
        
        if args.encoder == "all":
            # Test all encoders
            encoders = ["tokenizer", "mel", "wav2vec2", "whisper", "clap"]
            for encoder_type in encoders:
                test_encoder_only(
                    encoder_type=encoder_type,
                    audio_dir=args.audio_dir,
                    device=args.device,
                    num_files=args.num_files,
                    embed_dim=args.embed_dim,
                    num_layers=args.num_layers,
                    patch_size=args.patch_size,
                    max_audio_seconds=args.max_audio_seconds,
                )
                print()
                # Clean up GPU memory between encoders
                if args.device == "cuda":
                    torch.cuda.empty_cache()
        else:
            test_encoder_only(
                encoder_type=args.encoder,
                audio_dir=args.audio_dir,
                device=args.device,
                num_files=args.num_files,
                embed_dim=args.embed_dim,
                num_layers=args.num_layers,
                patch_size=args.patch_size,
                max_audio_seconds=args.max_audio_seconds,
            )
        return

    # Full model testing (with LLM)
    print("="*70)
    print("AUDIO ENCODER TESTING WITH FLAMINGO")
    print("="*70)
    print(f"Audio directory: {args.audio_dir}")
    print(f"LLM: {args.llm_id}")
    print(f"Device: {args.device}")
    print()
    print("Memory-saving settings:")
    print(f"  Embed dim: {args.embed_dim}")
    print(f"  Num layers: {args.num_layers}")
    print(f"  Patch size: {args.patch_size} samples ({args.patch_size/16:.0f}ms at 16kHz)")
    print(f"  Max audio: {args.max_audio_seconds} seconds")
    print()
    
    # Calculate memory estimate
    max_patches = (args.max_audio_seconds * SAMPLE_RATE) // args.patch_size
    print(f"Estimated max patches per audio: {max_patches}")
    print()

    if args.encoder == "all":
        # Compare all encoders with memory-saving settings
        compare_encoders(
            args.audio_dir, 
            args.llm_id, 
            args.device,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            patch_size=args.patch_size,
            max_audio_seconds=args.max_audio_seconds,
        )
    else:
        # Test single encoder
        max_audio_length = args.max_audio_seconds * SAMPLE_RATE
        dataset = CareSoundDataset(
            split="train",
            EOS_TOKEN="</s>",
            audio_dir=args.audio_dir,
            max_audio_length=max_audio_length,  # Limit audio length
        )

        model = create_model_with_encoder(
            args.encoder, 
            args.llm_id, 
            args.device,
            embed_dim=args.embed_dim,
            num_layers=args.num_layers,
            patch_size=args.patch_size,
        )
        test_forward_pass(model, dataset, args.device, args.num_samples, patch_size=args.patch_size)


if __name__ == "__main__":
    main()