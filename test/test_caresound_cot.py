#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Test script for CareSoundCoTDataset.

Tests loading and formatting of CoT samples.

Usage:
    python test/test_caresound_cot.py
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from opentslm.time_series_datasets.audio import CareSoundCoTDataset

def main():
    print("="*60)
    print("Testing CareSoundCoTDataset")
    print("="*60)

    try:
        # Load test split
        print("\nLoading test split...")
        dataset = CareSoundCoTDataset(
            split="test",
            EOS_TOKEN="<|endoftext|>",
            audio_dir="./data/caresound_audio/audio_merged",
            cot_data_dir="data/caresound_cot"
        )

        print(f"✅ Dataset loaded: {len(dataset)} samples")

        if len(dataset) == 0:
            print("❌ Dataset is empty!")
            return 1

        # Test first 3 samples
        print("\n" + "="*60)
        print("Testing first 3 samples:")
        print("="*60)

        for i in range(min(3, len(dataset))):
            sample = dataset[i]

            print(f"\n📝 Sample {i+1}:")
            print(f"   Patient ID: {sample['patient_id']}")
            print(f"   Dataset: {sample['dataset']}")
            print(f"   Question: {sample['question'][:60]}...")
            print(f"   Original Answer: {sample['original_answer']}")
            print(f"   CoT Rationale (first 100 chars): {sample['answer'][:100]}...")
            print(f"   Audio channels: {len(sample['time_series_text'])}")

            # Check audio loaded correctly
            if len(sample['time_series_text']) > 0:
                audio = sample['time_series_text'][0]
                if hasattr(audio, 'time_series'):
                    print(f"   Audio samples: {len(audio.time_series)}")
                else:
                    print("   ⚠️  Audio format unexpected")

        print("\n" + "="*60)
        print("✅ All tests passed!")
        print("="*60)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
