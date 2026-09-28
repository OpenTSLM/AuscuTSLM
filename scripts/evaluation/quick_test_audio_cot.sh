#!/bin/bash
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

# Quick test script for audio CoT generation
# Tests with 10 samples (~40 seconds, <1 cent)
#
# Usage:
#   bash scripts/evaluation/quick_test_audio_cot.sh sk-proj-your-key-here [audio_dir]
#
# Example:
#   bash scripts/evaluation/quick_test_audio_cot.sh sk-proj-abc123... ./data/caresound_audio/audio_merged

# Check if API key is provided
if [ -z "$1" ]; then
    echo "❌ Error: OpenAI API key required"
    echo ""
    echo "Usage:"
    echo "  bash scripts/evaluation/quick_test_audio_cot.sh YOUR_API_KEY [audio_dir]"
    echo ""
    echo "Example:"
    echo "  bash scripts/evaluation/quick_test_audio_cot.sh sk-proj-abc123... ./data/caresound_audio/audio_merged"
    echo ""
    exit 1
fi

API_KEY="$1"
AUDIO_DIR="${2:-./data/caresound_audio/audio_merged}"

echo "=========================================="
echo "Audio CoT Generation - Quick Test"
echo "=========================================="
echo ""
echo "API Key: ${API_KEY:0:10}...${API_KEY: -4}"
echo "Audio Directory: $AUDIO_DIR"
echo ""

# Step 1: Check balance BEFORE
echo "=========================================="
echo "Step 1: Checking OpenAI balance BEFORE..."
echo "=========================================="
echo ""
OPENAI_API_KEY="$API_KEY" python scripts/utils/check_openai_balance.py

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Balance check failed. Is your API key valid?"
    exit 1
fi

echo ""
read -p "Press Enter to continue with setup test or Ctrl+C to cancel..."

# Step 2: Test setup
echo ""
echo "=========================================="
echo "Step 2: Testing setup..."
echo "=========================================="
echo ""
OPENAI_API_KEY="$API_KEY" python scripts/evaluation/test_audio_cot.py --audio_dir "$AUDIO_DIR"

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Setup test failed. Please fix the issues above."
    exit 1
fi

# Step 3: Generate CoT
echo ""
echo "=========================================="
echo "Step 3: Generating CoT for 10 samples..."
echo "=========================================="
echo ""
echo "This will:"
echo "  - Generate spectrograms for 10 samples (~9 seconds)"
echo "  - Call GPT-4o-mini API (~30 seconds)"
echo "  - Save spectrograms for review"
echo "  - Cost: ~\$0.004 (<1 cent)"
echo ""
read -p "Press Enter to start generation or Ctrl+C to cancel..."

OPENAI_API_KEY="$API_KEY" python scripts/data_prep/generate_audio_cot.py \
    --audio_dir "$AUDIO_DIR" \
    --num_samples 10 \
    --save_spectrograms

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Generation failed. Check the error above."
    exit 1
fi

# Step 4: Check balance AFTER
echo ""
echo "=========================================="
echo "Step 4: Checking OpenAI balance AFTER..."
echo "=========================================="
echo ""
OPENAI_API_KEY="$API_KEY" python scripts/utils/check_openai_balance.py

# Step 5: Show results
echo ""
echo "=========================================="
echo "✅ Test Complete!"
echo "=========================================="
echo ""
echo "Review the results:"
echo "  1. CSV file: data/caresound_cot/caresound_train_cot.csv"
echo "  2. Spectrograms: data/caresound_cot/spectrograms/"
echo ""
echo "To view rationales:"
echo "  cat data/caresound_cot/caresound_train_cot.csv"
echo ""
echo "Or in Python:"
echo "  python -c \"import pandas as pd; df = pd.read_csv('data/caresound_cot/caresound_train_cot.csv'); print(df[['question', 'answer', 'rationale']].to_string())\""
echo ""
echo "If satisfied with quality, run full generation (~$10 for 26,100 samples):"
echo "  OPENAI_API_KEY=\"$API_KEY\" python scripts/data_prep/generate_audio_cot.py \\"
echo "      --audio_dir $AUDIO_DIR \\"
echo "      --checkpoint_every 100"
echo ""
