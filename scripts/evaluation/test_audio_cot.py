#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Quick test script for audio CoT generation.
Tests on 5 samples to verify setup before running full generation.

Usage:
    python scripts/evaluation/test_audio_cot.py --audio_dir data/caresound/audio
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Test imports
print("Testing imports...")
try:
    from openai import OpenAI
    print("✅ OpenAI library imported")
except ImportError:
    print("❌ OpenAI library not found. Install with: pip install openai")
    sys.exit(1)

try:
    import librosa
    print("✅ Librosa imported")
except ImportError:
    print("❌ Librosa not found. Install with: pip install librosa")
    sys.exit(1)

try:
    from datasets import load_dataset
    print("✅ Datasets library imported")
except ImportError:
    print("❌ Datasets library not found. Install with: pip install datasets")
    sys.exit(1)

# Test OpenAI API key
print("\nTesting OpenAI API key...")
try:
    client = OpenAI()
    # Try a minimal API call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say 'API works'"}],
        max_tokens=10
    )
    print(f"✅ OpenAI API key is valid")
    print(f"   Response: {response.choices[0].message.content}")
except Exception as e:
    error_str = str(e)
    # 429 = rate limited, which means the key IS valid, just throttled
    if "429" in error_str or "rate_limit" in error_str.lower():
        print("✅ OpenAI API key is valid (rate-limited right now, but that's OK)")
    else:
        print(f"❌ OpenAI API error: {e}")
        print("   Make sure OPENAI_API_KEY environment variable is set")
        sys.exit(1)

# Load a small sample from CaReSound
print("\nTesting CaReSound dataset loading...")
try:
    dataset = load_dataset("tsnngw/CaReSound")
    print(f"✅ CaReSound loaded: {len(dataset['train'])} train, {len(dataset['test'])} test")

    # Show sample
    sample = dataset['train'][0]
    print(f"\n📝 Sample data structure:")
    print(f"   Keys: {list(sample.keys())}")
    print(f"   Patient ID: {sample['patient_id']}")
    print(f"   Question: {sample['question'][:80]}...")
    print(f"   Answer: {sample['answer']}")
    print(f"   Dataset: {sample.get('dataset', 'N/A')}")
except Exception as e:
    print(f"❌ Error loading CaReSound: {e}")
    sys.exit(1)

# Test audio file access
print("\nTesting audio file access...")
import argparse
import glob as _glob
parser = argparse.ArgumentParser()
parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_merged")
args = parser.parse_args()

if not os.path.exists(args.audio_dir):
    print(f"❌ Audio directory not found: {args.audio_dir}")
    print("   Please specify correct path with --audio_dir")
    sys.exit(1)


def find_audio_files(patient_id, dataset_name, audio_dir):
    """Find audio file(s) for a patient, searching flat dir, subdirs, and suffixed names."""
    extensions = [".wav", ".flac", ".mp3", ".ogg"]
    found_files = []
    for ext in extensions:
        # Flat exact: {patient_id}.wav
        exact = os.path.join(audio_dir, f"{patient_id}{ext}")
        if os.path.exists(exact):
            found_files.append(exact)
        # Flat suffixed: {patient_id}_*.wav  (e.g. _p1, _AV, _MV)
        for p in _glob.glob(os.path.join(audio_dir, f"{patient_id}_*{ext}")):
            if p not in found_files:
                found_files.append(p)
        # Subdir exact: {dataset}/{patient_id}.wav
        sub_exact = os.path.join(audio_dir, dataset_name, f"{patient_id}{ext}")
        if os.path.exists(sub_exact) and sub_exact not in found_files:
            found_files.append(sub_exact)
        # Subdir suffixed: {dataset}/{patient_id}_*.wav
        for p in _glob.glob(os.path.join(audio_dir, dataset_name, f"{patient_id}_*{ext}")):
            if p not in found_files:
                found_files.append(p)
    return sorted(found_files)


# --- Scan ALL samples and report coverage ---
print("\n📊 Scanning audio coverage across dataset...")
total_train = len(dataset['train'])
total_test = len(dataset['test'])
missing_train = 0
missing_test = 0

for sample in dataset['train']:
    pid = str(sample['patient_id'])
    ds = sample.get('dataset', '')
    if not find_audio_files(pid, ds, args.audio_dir):
        missing_train += 1

for sample in dataset['test']:
    pid = str(sample['patient_id'])
    ds = sample.get('dataset', '')
    if not find_audio_files(pid, ds, args.audio_dir):
        missing_test += 1

valid_train = total_train - missing_train
valid_test = total_test - missing_test
skip_train_pct = 100.0 * missing_train / total_train if total_train else 0
skip_test_pct = 100.0 * missing_test / total_test if total_test else 0

print(f"   Train: {valid_train}/{total_train} valid samples "
      f"({missing_train} missing, {skip_train_pct:.2f}% skipped)")
print(f"   Test:  {valid_test}/{total_test} valid samples "
      f"({missing_test} missing, {skip_test_pct:.2f}% skipped)")

if valid_train == 0 and valid_test == 0:
    print(f"\n❌ No audio files found at all in: {args.audio_dir}")
    print("   Make sure audio files are downloaded and placed correctly")
    print("   Expected structure: audio_dir/{dataset}/{patient_id}[_suffix].wav")
    sys.exit(1)

if missing_train + missing_test > 0:
    print(f"   ⚠  {missing_train + missing_test} samples will be skipped during generation (audio not found)")
else:
    print("   ✅ All samples have matching audio files!")

# --- Find one valid sample to test loading ---
found = False
audio_path = None
for sample in dataset['train']:
    patient_id = str(sample['patient_id'])
    dataset_name = sample.get('dataset', '')
    files = find_audio_files(patient_id, dataset_name, args.audio_dir)
    if files:
        audio_path = files[0]
        found = True
        print(f"\n✅ Found audio file: {audio_path}")

        # Try to load it
        from opentslm.time_series_datasets.audio_util import load_audio, preprocess_audio
        waveform, sr = load_audio(audio_path, target_sample_rate=16000)
        waveform = preprocess_audio(waveform, sample_rate=sr, target_sample_rate=16000)
        print(f"   Duration: {len(waveform) / sr:.2f} seconds")
        print(f"   Sample rate: {sr} Hz")
        break

if not found:
    print(f"\n❌ Could not find any audio file for any sample")
    print(f"   Looked in: {args.audio_dir}")
    print(f"   Make sure audio files are downloaded and placed correctly")
    sys.exit(1)

# Test spectrogram generation
print("\nTesting spectrogram generation...")
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np

    # Generate mel-spectrogram (waveform is 1D after preprocess_audio)
    audio_np = waveform.numpy()
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np,
        sr=sr,
        n_mels=128,
        fmax=8000,
        hop_length=512
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Save test spectrogram
    fig, ax = plt.subplots(figsize=(12, 6))
    import librosa.display
    img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set_title('Test Mel-Spectrogram')

    test_spec_path = "test_spectrogram.png"
    plt.savefig(test_spec_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Spectrogram generated: {test_spec_path}")
    print(f"   Shape: {mel_spec.shape}")

    # Clean up
    os.remove(test_spec_path)

except Exception as e:
    print(f"❌ Error generating spectrogram: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS PASSED!")
print("="*60)
print("\nYou're ready to generate CoT rationales!")
print("\nTo run a small test (10 samples):")
print(f"  python scripts/data_prep/generate_audio_cot.py --audio_dir {args.audio_dir} --num_samples 10")
print("\nTo run full generation:")
print(f"  python scripts/data_prep/generate_audio_cot.py --audio_dir {args.audio_dir}")
print(f"\nEstimated costs (using GPT-4o-mini):")
print(f"  10 samples: ~$0.004 (less than 1 cent)")
print(f"  100 samples: ~$0.04 (4 cents)")
print(f"  1,000 samples: ~$0.40 (40 cents)")
print(f"  Full training set (26,100): ~$10")
print(f"  Full dataset (32,620): ~$13")
