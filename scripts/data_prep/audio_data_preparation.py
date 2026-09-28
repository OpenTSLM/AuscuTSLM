#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Audio Data Preparation Script

This script handles all audio data preparation tasks:
1. Download audio from Polybox
2. Organize audio files by patient_id
3. Merge SPRSound segments
4. Verify alignment with HuggingFace dataset
5. Test audio loading

Usage:
    python scripts/data_prep/audio_data_preparation.py --polybox_url <url>  # Download & organize
    python scripts/data_prep/audio_data_preparation.py --merge_sprsound     # Merge SPRSound
    python scripts/data_prep/audio_data_preparation.py --verify             # Verify alignment
    python scripts/data_prep/audio_data_preparation.py --audio_dir <path>   # Test loading
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import argparse
import numpy as np
from opentslm.time_series_datasets.audio.CareSoundDataset import CareSoundDataset
import torch
import zipfile
import requests
import subprocess
from pathlib import Path
import shutil
from tqdm import tqdm
import re
from collections import defaultdict
import soundfile as sf
import json

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
from opentslm.time_series_datasets.audio.sprsound_merger import (
    find_sprsound_segments,
    load_and_merge_sprsound_patient,
    get_position_description,
    parse_sprsound_filename,
    group_segments_by_position,
)


def download_and_extract_polybox(polybox_url: str, extract_dir: str = "./data/caresound_audio"):
    """
    Download zip file from Polybox and extract it.
    Mimics the pattern from ecgqa_cot_loader.py

    Args:
        polybox_url: Polybox download URL
        extract_dir: Directory to extract files to

    Returns:
        Path to extracted directory
    """
    print("="*70)
    print("Downloading and Extracting Audio from Polybox")
    print("="*70)
    print()

    # Create extract directory
    os.makedirs(extract_dir, exist_ok=True)

    # Download zip file
    zip_path = os.path.join(extract_dir, "audio.zip")

    if not os.path.exists(zip_path):
        print(f"Downloading from Polybox...")
        print(f"URL: {polybox_url}")
        print()

        try:
            # Try wget first (more reliable for large files)
            print("Attempting download with wget...")
            subprocess.run([
                "wget", "-O", zip_path, polybox_url
            ], check=True)
            print("Download complete with wget")

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback to Python requests if wget is not available
            print("wget not available, using Python requests...")

            response = requests.get(polybox_url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(zip_path, 'wb') as f:
                if total_size == 0:
                    # No content-length header, download without progress
                    f.write(response.content)
                else:
                    # Download with progress bar
                    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

            print("Download complete")
    else:
        print(f"Zip file already exists: {zip_path}")

    # Extract zip file
    print(f"\nExtracting zip file...")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Get list of files to extract
        file_list = zip_ref.namelist()
        total_files = len(file_list)

        print(f"Extracting {total_files} files...")

        # Extract to a temporary directory first
        temp_extract_dir = os.path.join(extract_dir, "temp_extract")

        # Extract with progress bar
        for file_info in tqdm(file_list, desc="Extracting", unit="files"):
            zip_ref.extract(file_info, temp_extract_dir)

        # Find the actual data directory (it might be nested)
        extracted_items = os.listdir(temp_extract_dir)

        if len(extracted_items) == 1 and os.path.isdir(os.path.join(temp_extract_dir, extracted_items[0])):
            # If there's a single directory, move its contents up
            nested_dir = os.path.join(temp_extract_dir, extracted_items[0])
            final_dir = os.path.join(extract_dir, "audio_files")

            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)

            shutil.move(nested_dir, final_dir)
            shutil.rmtree(temp_extract_dir)

            print(f"Extraction complete")
            print(f"Files extracted to: {final_dir}")
            return final_dir
        else:
            # Multiple items or files at root level
            final_dir = os.path.join(extract_dir, "audio_files")

            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)

            shutil.move(temp_extract_dir, final_dir)

            print(f"Extraction complete")
            print(f"Files extracted to: {final_dir}")
            return final_dir


def extract_patient_id_from_filename(filename: str, strategy: str = "first_part") -> tuple:
    """
    Extract patient_id from filename.
    
    Args:
        filename: Original filename (without extension)
        strategy: Extraction strategy
            - "first_part": Use first part before underscore (keeps alphabet + numbers)
            - "numeric": Extract longest numeric sequence
    
    Returns:
        (patient_id, suffix) tuple
    """
    if strategy == "first_part":
        # Split by underscore, take first part (keeps alphabet + numbers)
        parts = filename.split("_")
        patient_id = parts[0] if parts else filename
        remaining = "_".join(parts[1:]) if len(parts) > 1 else ""
        return patient_id, remaining
    elif strategy == "numeric":
        # Extract longest numeric sequence
        numbers = re.findall(r'\d+', filename)
        if numbers:
            patient_id = max(numbers, key=len)
            idx = filename.find(patient_id)
            if idx != -1:
                after_id = filename[idx + len(patient_id):]
                suffix = after_id.strip("_")
            else:
                suffix = ""
            return patient_id, suffix
        else:
            # Fallback to first part
            parts = filename.split("_")
            return parts[0] if parts else filename, ""
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def organize_audio_files_by_patient_id(
    source_dir: str, 
    output_dir: str = "./data/caresound_organized",
    patient_id_extraction: str = "numeric",
    use_subdirectories: bool = True,
):
    """
    Organize audio files to match patient_id format required by CaReSound dataset.
    
    This function:
    1. Scans source_dir for audio files (in any folder structure)
    2. Extracts patient_id from filename (numeric extraction to match HuggingFace)
    3. Uses ONLY patient_id as filename (no suffixes, no dataset prefix in filename)
    4. Organizes in dataset subdirectories to avoid collisions
    
    Args:
        source_dir: Directory containing audio files (can be nested)
        output_dir: Output directory for organized files
        patient_id_extraction: How to extract patient_id ("numeric" recommended)
        use_subdirectories: If True, use {dataset}/{patient_id}.wav structure
    
    Returns:
        Path to organized directory
    """
    print("="*70)
    print("Organizing Audio Files by Patient ID")
    print("="*70)
    print()
    print(f"Patient ID extraction: {patient_id_extraction}")
    print(f"Use subdirectories: {use_subdirectories}")
    print("Note: Files named as {patient_id}.wav only (no suffixes, no dataset prefix)")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all audio files recursively (exclude hidden files)
    audio_files = []
    for ext in ['.wav', '.mp3', '.WAV', '.MP3']:
        all_files = Path(source_dir).rglob(f'*{ext}')
        # Filter out hidden files (starting with . or ._)
        for file_path in all_files:
            filename = file_path.name
            # Skip hidden files (starting with . or ._)
            if not filename.startswith('.') and not filename.startswith('._'):
                audio_files.append(file_path)
    
    print(f"Found {len(audio_files)} audio files in {source_dir}")
    print()
    
    if len(audio_files) == 0:
        print("⚠ No audio files found!")
        return output_dir
    
    # Simple dataset detection - just use folder name (user has unified them)
    def detect_dataset(file_path: Path) -> str:
        """Detect dataset name from file path - uses folder name directly."""
        parts = file_path.parts
        # Look for common dataset folder names
        dataset_names = ["ICBHI", "CirCor", "SPRSound", "ZCHSound", "KAUH"]
        for part in reversed(parts):
            if part in dataset_names:
                return part
        # If not found, use parent directory name
        if len(parts) > 1:
            return parts[-2]  # Parent directory
        return "Unknown"
    
    # Organize files
    copied = 0
    skipped = 0
    errors = 0
    collisions = 0
    stats = defaultdict(lambda: {"files": 0, "collisions": 0})
    
    # Track files per dataset to handle multiple files per patient_id
    # Key insight: If only ONE file per patient_id → use {patient_id}.wav
    # If MULTIPLE files per patient_id → use {patient_id}_{suffix}.wav to distinguish
    file_tracker = defaultdict(list)  # (dataset, patient_id) -> list of (source_path, suffix)
    
    for audio_file in tqdm(audio_files, desc="Organizing files", unit="files"):
        original_name = audio_file.name
        original_stem = audio_file.stem
        dataset = detect_dataset(audio_file)
        
        # Extract patient_id (first part) and suffix
        patient_id, remaining = extract_patient_id_from_filename(original_stem, patient_id_extraction)
        
        # Track this file
        key = (dataset, patient_id)
        file_tracker[key].append((str(audio_file), remaining))
        stats[dataset]["files"] += 1
    
    # Now organize files
    for (dataset, patient_id), file_list in file_tracker.items():
        if use_subdirectories:
            dataset_dir = os.path.join(output_dir, dataset)
            os.makedirs(dataset_dir, exist_ok=True)
        else:
            dataset_dir = output_dir
        
        if len(file_list) == 1:
            # Single file - keep original suffix (e.g. 102_1b1_Ar_sc_Meditron.wav)
            # so downstream scripts can match location codes from the filename
            file_path, suffix = file_list[0]
            if suffix:
                dest_filename = f"{patient_id}_{suffix}{Path(file_path).suffix}"
            else:
                dest_filename = f"{patient_id}{Path(file_path).suffix}"
            dest_path = os.path.join(dataset_dir, dest_filename)
            
            if not os.path.exists(dest_path):
                try:
                    shutil.copy2(file_path, dest_path)
                    copied += 1
                except Exception as e:
                    print(f"  ⚠ Error copying {Path(file_path).name}: {e}")
                    errors += 1
            else:
                skipped += 1
        else:
            # Multiple files for same patient_id - keep suffixes to distinguish
            collisions += len(file_list) - 1
            stats[dataset]["collisions"] += len(file_list) - 1
            #print(f"  ℹ Multiple files for {dataset}/{patient_id}: keeping suffixes to distinguish")
            
            for file_path, suffix in file_list:
                if suffix:
                    # Keep the suffix
                    dest_filename = f"{patient_id}_{suffix}{Path(file_path).suffix}"
                else:
                    # No suffix but multiple files - this shouldn't happen, but number them
                    idx = file_list.index((file_path, suffix)) + 1
                    dest_filename = f"{patient_id}_{idx}{Path(file_path).suffix}"
                
                dest_path = os.path.join(dataset_dir, dest_filename)
                
                if not os.path.exists(dest_path):
                    try:
                        shutil.copy2(file_path, dest_path)
                        copied += 1
                    except Exception as e:
                        print(f"  ⚠ Error copying {Path(file_path).name}: {e}")
                        errors += 1
                else:
                    skipped += 1
    
    print()
    print(f"✓ Organization complete!")
    print(f"  Copied: {copied} files")
    print(f"  Skipped (already exist): {skipped} files")
    print(f"  Errors: {errors} files")
    if collisions > 0:
        print(f"  Collisions (skipped): {collisions} files")
    print()
    print("Per-dataset statistics:")
    for dataset, data in sorted(stats.items()):
        if data["files"] > 0:
            print(f"  {dataset}: {data['files']} files", end="")
            if data["collisions"] > 0:
                print(f", {data['collisions']} collisions")
            else:
                print()
    print()
    print(f"Output directory: {output_dir}")
    if use_subdirectories:
        print(f"Structure: {output_dir}/{{dataset}}/{{patient_id}}.wav")
    else:
        print(f"Structure: {output_dir}/{{patient_id}}.wav")
        if collisions > 0:
            print("⚠ WARNING: Collisions detected! Consider using --use_subdirectories")
    print()
    
    return output_dir


def test_audio_loading(num_samples: int = 5, split: str = "train", audio_dir: str = None):
    """
    Test loading audio from CaReSound dataset.

    Args:
        num_samples: Number of samples to test
        split: Dataset split to test ('train' or 'test')
        audio_dir: Directory containing audio files (required!)
    """

    print("="*70)
    print("CaReSound Audio Loading Test")
    print("="*70)
    print()

    if audio_dir is None:
        print("ERROR: You must specify --audio_dir parameter!")
        print()
        print("The CaReSound HuggingFace dataset only contains metadata.")
        print("You need to download audio files separately and provide the path.")
        print()
        print("Usage:")
        print("  python scripts/data_prep/audio_data_preparation.py --audio_dir /path/to/audio/files")
        print()
        print("See CARESOUND_AUDIO_SETUP.md for detailed instructions.")
        return

    # Create dataset
    print(f"Loading {split} split...")
    print(f"Audio directory: {audio_dir}")
    try:
        dataset = CareSoundDataset(
            split=split,
            EOS_TOKEN="</s>",
            audio_dir=audio_dir,
            cache_dir="./data/caresound_cache"
        )
        print(f"✓ Dataset loaded successfully!")
        print(f"  Total samples: {len(dataset)}")
        print()
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        print("\nMake sure you have installed: pip install datasets")
        return

    # Test loading samples
    print(f"Testing {num_samples} samples...")
    print("-"*70)

    for i in range(min(num_samples, len(dataset))):
        print(f"\nSample {i+1}/{num_samples}")
        print("-"*70)

        try:
            # Get sample
            sample = dataset[i]

            # Print sample information
            print(f"Pre-prompt: {sample['pre_prompt'][:100]}...")
            print(f"Post-prompt: {sample['post_prompt'][:100]}...")
            print(f"Answer: {sample['answer'][:100]}...")
            print()

            # Check audio data
            if 'time_series' in sample and len(sample['time_series']) > 0:
                audio_data = sample['time_series'][0]

                # Convert to numpy for analysis
                if isinstance(audio_data, list):
                    audio_array = np.array(audio_data)
                elif isinstance(audio_data, torch.Tensor):
                    audio_array = audio_data.numpy()
                else:
                    audio_array = audio_data

                # Audio statistics
                print("Audio Information:")
                print(f"  ✓ Audio loaded successfully")
                print(f"  Length: {len(audio_array)} samples")
                print(f"  Duration: {len(audio_array) / 16000:.2f} seconds (at 16kHz)")
                print(f"  Min value: {audio_array.min():.4f}")
                print(f"  Max value: {audio_array.max():.4f}")
                print(f"  Mean: {audio_array.mean():.4f}")
                print(f"  Std: {audio_array.std():.4f}")

                # Check if audio is silent
                if np.abs(audio_array).max() < 1e-6:
                    print("  ⚠ WARNING: Audio appears to be silent!")
                else:
                    print("  ✓ Audio contains signal")

                # Check for NaN or Inf
                if np.isnan(audio_array).any():
                    print("  ✗ ERROR: Audio contains NaN values!")
                if np.isinf(audio_array).any():
                    print("  ✗ ERROR: Audio contains Inf values!")

            else:
                print("✗ No audio data found in sample!")

            print()

        except Exception as e:
            print(f"✗ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("="*70)
    print("Test Complete!")
    print("="*70)
    print()
    print("Next Steps:")
    print("1. If audio loaded successfully, you can proceed to training")
    print("2. The audio interface is in the dataset's __getitem__ method")
    print("3. Audio is returned as 'time_series' field in the sample dict")
    print("4. Format: sample['time_series'] = [audio_waveform_as_list]")
    print()
    print("To train with this dataset, use:")
    print("  python scripts/training/audio_curriculum_learning.py --encoder tokenizer --audio_dir ./data/caresound_audio/audio_merged")


def verify_audio_alignment(audio_dir: str, cache_dir: str = "./data/caresound_cache"):
    """
    Verify that organized audio files' patient_id (first part before _) 
    matches the HuggingFace CaReSound patient_id format.
    
    Args:
        audio_dir: Directory containing organized audio files
        cache_dir: Cache directory for HuggingFace dataset
    """
    print("="*70)
    print("Verifying Audio File Alignment with HuggingFace Patient IDs")
    print("="*70)
    print()
    
    # Load HuggingFace patient IDs
    try:
        from datasets import load_dataset
        print("Loading HuggingFace CaReSound dataset...")
        dataset = load_dataset("tsnngw/CaReSound", cache_dir=cache_dir)
        
        patient_ids_by_dataset = defaultdict(set)
        for split in ["train", "test"]:
            split_data = dataset[split]
            for sample in split_data:
                patient_id = sample["patient_id"]
                dataset_name = sample.get("dataset", "Unknown")
                patient_ids_by_dataset[dataset_name].add(patient_id)
        
        print(f"✓ Loaded patient IDs from HuggingFace")
        print()
        for dataset_name, patient_ids in patient_ids_by_dataset.items():
            print(f"  {dataset_name}: {len(patient_ids)} unique patient IDs")
        print()
    except Exception as e:
        print(f"✗ Error loading HuggingFace dataset: {e}")
        return
    
    # Scan organized audio files
    print(f"Scanning audio files in: {audio_dir}")
    print()
    
    audio_files = []
    for ext in ['.wav', '.mp3', '.WAV', '.MP3']:
        for file_path in Path(audio_dir).rglob(f'*{ext}'):
            filename = file_path.name
            # Skip hidden files
            if not filename.startswith('.') and not filename.startswith('._'):
                audio_files.append(file_path)
    
    print(f"Found {len(audio_files)} audio files")
    print()
    
    if len(audio_files) == 0:
        print("⚠ No audio files found!")
        return
    
    # Extract patient_id from each file and verify
    matches = 0
    mismatches = 0
    not_in_hf = 0
    stats = defaultdict(lambda: {"files": 0, "matches": 0, "mismatches": 0, "not_in_hf": 0})
    
    # Dataset detection from path
    dataset_mapping = {
        "ICBHI": "ICBHI", "icbhi": "ICBHI",
        "CirCor": "CirCor", "circor": "CirCor",
        "SPRSound": "SPRSound", "sprsound": "SPRSound",
        "ZCHSound": "ZCHSound", "zchsound": "ZCHSound",
        "KAUH": "KAUH", "kauh": "KAUH",
    }
    
    def detect_dataset(file_path: Path) -> str:
        """Detect dataset name from file path."""
        parts = file_path.parts
        for part in reversed(parts):
            if part in dataset_mapping:
                return dataset_mapping[part]
            for key, value in dataset_mapping.items():
                if part.lower() == key.lower():
                    return value
        return "Unknown"
    
    print("Verifying patient_id alignment...")
    print()
    
    mismatch_examples = []
    skipped_segments = 0
    
    for audio_file in tqdm(audio_files, desc="Verifying", unit="files"):
        filename = audio_file.name
        stem = audio_file.stem  # filename without extension
        
        # Skip SPRSound segment files (format: PatientID_Timestamp_Type_Position_ClipID)
        # These have 5 parts where part 2 is a float (timestamp)
        segment = parse_sprsound_filename(filename)
        if segment is not None:
            # This is an unmerged segment file - skip it
            skipped_segments += 1
            continue
        
        # Extract patient_id (first part before _)
        patient_id, suffix = extract_patient_id_from_filename(stem, strategy="first_part")
        
        # Detect dataset from path
        dataset = detect_dataset(audio_file)
        
        stats[dataset]["files"] += 1
        
        # Check if patient_id exists in HuggingFace for this dataset
        if dataset in patient_ids_by_dataset:
            if patient_id in patient_ids_by_dataset[dataset]:
                matches += 1
                stats[dataset]["matches"] += 1
            else:
                mismatches += 1
                stats[dataset]["mismatches"] += 1
                if len(mismatch_examples) < 10:
                    mismatch_examples.append((filename, patient_id, dataset))
        else:
            not_in_hf += 1
            stats[dataset]["not_in_hf"] += 1
    
    # Print results
    print()
    print("="*70)
    print("Verification Results")
    print("="*70)
    print()
    verified_count = len(audio_files) - skipped_segments
    print(f"Total audio files: {len(audio_files)}")
    if skipped_segments > 0:
        print(f"  Skipped SPRSound segments: {skipped_segments} (run --merge_sprsound first, then --delete_originals)")
        print(f"  Files verified: {verified_count}")
    if verified_count == 0:
        print()
        print("⚠ No files to verify! All files appear to be unmerged SPRSound segments.")
        print("  Run: python scripts/data_prep/audio_data_preparation.py --merge_sprsound --audio_dir <path>")
        print("  Then: python scripts/data_prep/audio_data_preparation.py --merge_sprsound --delete_originals --audio_dir <path>")
        return
    print(f"✓ Matches with HuggingFace: {matches} files ({matches/verified_count*100:.1f}%)")
    print(f"✗ Mismatches: {mismatches} files ({mismatches/verified_count*100:.1f}%)")
    if not_in_hf > 0:
        print(f"⚠ Unknown dataset: {not_in_hf} files ({not_in_hf/verified_count*100:.1f}%)")
    print()
    
    print("Per-dataset statistics:")
    for dataset in sorted(stats.keys()):
        data = stats[dataset]
        print(f"  {dataset}:")
        print(f"    Total files: {data['files']}")
        match_pct = (data['matches']/data['files']*100) if data['files'] > 0 else 0.0
        print(f"    Matches: {data['matches']} ({match_pct:.1f}%)")
        print(f"    Mismatches: {data['mismatches']}")
        if data['not_in_hf'] > 0:
            print(f"    Unknown dataset: {data['not_in_hf']}")
    print()
    
    if mismatch_examples:
        print("Example mismatches (first 10):")
        for filename, patient_id, dataset in mismatch_examples:
            print(f"  {filename} → patient_id='{patient_id}' (not found in HuggingFace {dataset})")
        print()
    
    # Recommendations
    if mismatches > 0:
        print("⚠ WARNING: Some patient_ids don't match HuggingFace format!")
        print()
        print("Possible reasons:")
        print("  1. Filename format doesn't match (e.g., contains extra prefixes)")
        print("  2. Patient_id extraction strategy needs adjustment")
        print("  3. Some files might not be in the HuggingFace dataset")
    else:
        print("✓ All patient_ids match HuggingFace format!")
    print()
    
    # Create manifest and handle mismatches
    manifest = {"matched": {}, "unmatched": []}
    unmatched_dir = os.path.join(audio_dir, "_unmatched")
    
    print("Creating audio manifest and handling mismatches...")
    moved_count = 0
    
    for audio_file in audio_files:
        filename = audio_file.name
        stem = audio_file.stem
        
        # Skip SPRSound segments
        segment = parse_sprsound_filename(filename)
        if segment is not None:
            continue
        
        patient_id, suffix = extract_patient_id_from_filename(stem, strategy="first_part")
        dataset = detect_dataset(audio_file)
        
        # Check if matched
        is_matched = False
        if dataset in patient_ids_by_dataset:
            if patient_id in patient_ids_by_dataset[dataset]:
                is_matched = True
        
        if is_matched:
            # Add to manifest
            if dataset not in manifest["matched"]:
                manifest["matched"][dataset] = {}
            if patient_id not in manifest["matched"][dataset]:
                manifest["matched"][dataset][patient_id] = []
            manifest["matched"][dataset][patient_id].append(str(audio_file))
        else:
            # Move to _unmatched folder
            manifest["unmatched"].append({
                "file": str(audio_file),
                "patient_id": patient_id,
                "dataset": dataset
            })
            
            # Create unmatched directory structure
            unmatched_dataset_dir = os.path.join(unmatched_dir, dataset)
            os.makedirs(unmatched_dataset_dir, exist_ok=True)
            
            # Move file
            dest_path = os.path.join(unmatched_dataset_dir, filename)
            try:
                shutil.move(str(audio_file), dest_path)
                moved_count += 1
            except Exception as e:
                print(f"  Warning: Could not move {filename}: {e}")
    
    # Save manifest
    manifest_path = os.path.join(audio_dir, "audio_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print()
    print("="*70)
    print("Manifest Created")
    print("="*70)
    print()
    print(f"✓ Manifest saved to: {manifest_path}")
    print(f"  Matched files: {sum(len(files) for d in manifest['matched'].values() for files in d.values())}")
    print(f"  Unmatched files moved: {moved_count}")
    if moved_count > 0:
        print(f"  Unmatched files location: {unmatched_dir}")
    print()
    print("For training, the CareSoundDataset will use matched files only.")
    print("Review unmatched files in _unmatched/ folder if needed.")
    print()


def merge_sprsound_segments(
    audio_dir: str,
    output_dir: str = None,
    sample_rate: int = 16000,
    crossfade_ms: float = 10.0,
    delete_originals: bool = False
):
    """
    Merge SPRSound segmented audio files and save to disk.
    
    SPRSound files are named: {PatientID}_{Timestamp}_{Type}_{Position}_{ClipID}.wav
    This function merges them into: {PatientID}_{Position}.wav (one file per position)
    
    Args:
        audio_dir: Directory containing SPRSound segments (or subdirectory SPRSound/)
        output_dir: Output directory for merged files (default: same as audio_dir)
        sample_rate: Target sample rate
        crossfade_ms: Crossfade duration in milliseconds
        delete_originals: If True, delete original segment files after merging
        
    Returns:
        Dictionary with merge statistics
    """
    print("="*70)
    print("Merging SPRSound Segments")
    print("="*70)
    print()
    
    # Create separate merged output directory
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_dir), "audio_merged")
    
    # Find SPRSound directory in source
    sprsound_dir = os.path.join(audio_dir, "SPRSound")
    if os.path.exists(sprsound_dir):
        search_dir = sprsound_dir
        out_subdir = os.path.join(output_dir, "SPRSound")
    else:
        search_dir = audio_dir
        out_subdir = output_dir
    
    # Clean up existing merged folder to avoid corrupted files
    if os.path.exists(out_subdir):
        print(f"Cleaning existing merged folder: {out_subdir}")
        shutil.rmtree(out_subdir)
    
    os.makedirs(out_subdir, exist_ok=True)
    
    print(f"Searching for segments in: {search_dir}")
    print(f"Output directory (merged files): {out_subdir}")
    print()
    
    # Find all segment files and group by patient_id
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
    all_segments = []
    non_segment_files = []
    
    for filename in os.listdir(search_dir):
        if filename.startswith('.') or filename.startswith('._'):
            continue
        if not any(filename.endswith(ext) for ext in audio_extensions):
            continue
        
        segment = parse_sprsound_filename(filename)
        if segment is not None:
            segment.filepath = os.path.join(search_dir, filename)
            all_segments.append(segment)
        else:
            non_segment_files.append(filename)
    
    # Show sample filenames for debugging
    all_audio_in_dir = [f for f in os.listdir(search_dir) 
                        if any(f.endswith(ext) for ext in audio_extensions)
                        and not f.startswith('.')]
    print(f"Total audio files in directory: {len(all_audio_in_dir)}")
    if all_audio_in_dir:
        print(f"Sample filenames (first 5):")
        for f in all_audio_in_dir[:5]:
            print(f"  - {f}")
    print()
    
    if len(all_segments) == 0:
        print("No SPRSound segment files found matching format:")
        print("  {PatientID}_{Timestamp}_{Type}_{Position}_{ClipID}.wav")
        print("  Example: 40069321_15.3_0_p4_984.wav")
        print()
        if non_segment_files:
            print(f"Found {len(non_segment_files)} files that don't match segment format:")
            for f in non_segment_files[:5]:
                print(f"  - {f}")
            print()
            print("These files might already be merged or have a different naming convention.")
        return {"patients": 0, "segments": 0, "merged_files": 0}
    
    # Group by patient_id
    patients = defaultdict(list)
    for segment in all_segments:
        patients[segment.patient_id].append(segment)
    
    print(f"Found {len(all_segments)} segments from {len(patients)} patients")
    print()
    
    # Process each patient
    merged_count = 0
    deleted_count = 0
    files_to_delete = []
    
    for patient_id in tqdm(sorted(patients.keys()), desc="Merging patients"):
        patient_segments = patients[patient_id]
        
        # Group by position
        position_groups = group_segments_by_position(patient_segments)
        
        for position, pos_segments in position_groups.items():
            # Merge segments for this position
            merged_results = load_and_merge_sprsound_patient(
                audio_dir=search_dir,
                patient_id=patient_id,
                target_sample_rate=sample_rate,
                crossfade_ms=crossfade_ms,
                insert_gaps=True,
                max_gap_seconds=2.0,
                merge_positions=False
            )
            
            if position in merged_results:
                merged_waveform, segment_info = merged_results[position]
                
                if len(merged_waveform) > 0:
                    # Save merged file
                    # Format: {patient_id}_{position}.wav (e.g., 40069321_p4.wav)
                    merged_filename = f"{patient_id}_{position}.wav"
                    merged_path = os.path.join(out_subdir, merged_filename)
                    
                    # Delete existing file if corrupted from previous attempt
                    if os.path.exists(merged_path):
                        os.remove(merged_path)
                    
                    try:
                        # Save using soundfile with explicit WAV format
                        sf.write(merged_path, merged_waveform, sample_rate, format='WAV', subtype='PCM_16')
                        merged_count += 1
                        
                        # Track original files for deletion
                        if delete_originals:
                            for seg in pos_segments:
                                files_to_delete.append(seg.filepath)
                    except Exception as e:
                        print(f"  Error saving {merged_filename}: {e}")
    
    # Delete original segment files if requested
    if delete_originals and files_to_delete:
        print()
        print(f"Deleting {len(files_to_delete)} original segment files...")
        for filepath in files_to_delete:
            try:
                os.remove(filepath)
                deleted_count += 1
            except Exception as e:
                print(f"  Warning: Could not delete {filepath}: {e}")
    
    # Also copy non-SPRSound datasets to the merged folder
    print()
    print("Copying other datasets to merged folder...")
    other_datasets_copied = 0
    
    # Look for other dataset folders (ICBHI, CirCor, ZCHSound, KAUH)
    other_datasets = ["ICBHI", "CirCor", "ZCHSound", "KAUH"]
    for dataset in other_datasets:
        src_dataset_dir = os.path.join(audio_dir, dataset)
        if os.path.exists(src_dataset_dir):
            dst_dataset_dir = os.path.join(output_dir, dataset)
            if not os.path.exists(dst_dataset_dir):
                shutil.copytree(src_dataset_dir, dst_dataset_dir)
                num_files = len([f for f in os.listdir(dst_dataset_dir) if not f.startswith('.')])
                print(f"  Copied {dataset}: {num_files} files")
                other_datasets_copied += num_files
            else:
                print(f"  {dataset} already exists in merged folder")
    
    print()
    print("="*70)
    print("Merge Complete!")
    print("="*70)
    print()
    print(f"  SPRSound patients processed: {len(patients)}")
    print(f"  SPRSound segments merged: {len(all_segments)} → {merged_count} files")
    if delete_originals:
        print(f"  Original segments deleted: {deleted_count}")
    if other_datasets_copied > 0:
        print(f"  Other datasets copied: {other_datasets_copied} files")
    
    return {
        "patients": len(patients),
        "segments": len(all_segments),
        "merged_files": merged_count,
        "deleted": deleted_count,
        "other_datasets_copied": other_datasets_copied
    }


def inspect_dataset_structure(split: str = "train"):
    """
    Inspect the raw dataset structure from HuggingFace.

    Args:
        split: Dataset split to inspect
    """
    print("="*70)
    print("Inspecting Raw Dataset Structure")
    print("="*70)
    print()

    try:
        from datasets import load_dataset

        print("Loading dataset from HuggingFace...")
        dataset = load_dataset("tsnngw/CaReSound", cache_dir="./data/caresound_cache")

        print(f"✓ Dataset loaded!")
        print()
        print("Available splits:", list(dataset.keys()))
        print()

        # Inspect train split
        train_data = dataset[split]
        print(f"{split.capitalize()} split:")
        print(f"  Number of samples: {len(train_data)}")
        print(f"  Features: {train_data.features}")
        print()

        # Show first sample
        print("First sample:")
        sample = train_data[0]
        for key, value in sample.items():
            if key == "audio":
                if isinstance(value, dict):
                    print(f"  {key}:")
                    print(f"    - sampling_rate: {value.get('sampling_rate', 'N/A')}")
                    print(f"    - array shape: {np.array(value.get('array', [])).shape}")
                else:
                    print(f"  {key}: {type(value)}")
            else:
                value_str = str(value)[:100]
                print(f"  {key}: {value_str}")
        print()

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description="Test CaReSound audio loading")
    parser.add_argument("--num_samples", type=int, default=5,
                       help="Number of samples to test")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"],
                       help="Dataset split to test")
    parser.add_argument("--audio_dir", type=str, default="./data/caresound_audio/audio_organized",
                       help="Directory containing audio files")
    parser.add_argument("--polybox_url", type=str, default=None,
                       help="Polybox URL to download audio zip file (only use first time)")
    parser.add_argument("--organize", type=str, default=None,
                       help="Organize audio files from source directory (extracts patient_id from filenames)")
    parser.add_argument("--patient_id_extraction", type=str, default="first_part",
                       choices=["first_part", "numeric", "all_before_last"],
                       help="How to extract patient_id from filename (default: first_part - uses first part before underscore)")
    parser.add_argument("--use_subdirectories", action="store_true", default=True,
                       help="Organize files in dataset subdirectories: {dataset}/{patient_id}.wav (default: True)")
    parser.add_argument("--flat_structure", action="store_true",
                       help="Use flat structure: {patient_id}.wav (may have collisions)")
    parser.add_argument("--inspect", action="store_true",
                       help="Inspect raw dataset structure")
    parser.add_argument("--verify", action="store_true",
                       help="Verify audio files align with HuggingFace patient_ids")
    parser.add_argument("--merge_sprsound", action="store_true",
                       help="Merge SPRSound segment files into single files per patient+position (deletes originals by default)")
    parser.add_argument("--keep_originals", action="store_true",
                       help="Keep original segment files after merging (use with --merge_sprsound)")

    args = parser.parse_args()

    # Handle Polybox download and organization
    if args.polybox_url:
        print("Polybox URL provided - downloading and organizing audio files...")
        print()

        # Step 1: Download and extract
        extracted_dir = download_and_extract_polybox(args.polybox_url)
        print()

        # Step 2: Organize files (patient_id only, in subdirectories)
        print("Organizing audio files...")
        print()
        organized_dir = organize_audio_files_by_patient_id(
            extracted_dir,
            output_dir=os.path.join(os.path.dirname(extracted_dir), "audio_organized"),
            patient_id_extraction=args.patient_id_extraction,
            use_subdirectories=not args.flat_structure
        )

        print("="*70)
        print("Download and organization complete!")
        print("="*70)
        print()
        return
    
    # Handle SPRSound merging
    if args.merge_sprsound:
        if not args.audio_dir:
            print("✗ Error: --audio_dir required for merging")
            print("   Use: python scripts/data_prep/audio_data_preparation.py --merge_sprsound --audio_dir <path>")
            return
        merge_sprsound_segments(
            audio_dir=args.audio_dir,
            output_dir=None,  # Will create audio_merged/ folder
            sample_rate=16000,
            crossfade_ms=10.0,
            delete_originals=not args.keep_originals  # Delete by default
        )
        return
    
    # Handle verification
    if args.verify:
        if not args.audio_dir:
            print("✗ Error: --audio_dir required for verification")
            print("   Use: python scripts/data_prep/audio_data_preparation.py --verify --audio_dir <path>")
            return
        verify_audio_alignment(args.audio_dir)
        return
    
    # Run test or inspect
    if args.inspect:
        inspect_dataset_structure(args.split)
    elif args.audio_dir:
        test_audio_loading(args.num_samples, args.split, args.audio_dir)
    else:
        print("No action specified. Use --inspect, --verify, --merge_sprsound, or provide --audio_dir for testing.")
        print()
        print("Examples:")
        print("  # Merge SPRSound segments (deletes originals by default):")
        print("  python scripts/data_prep/audio_data_preparation.py --merge_sprsound")
        print()
        print("  # Merge but keep original segment files:")
        print("  python scripts/data_prep/audio_data_preparation.py --merge_sprsound --keep_originals")
        print()
        print("  # Verify alignment:")
        print("  python scripts/data_prep/audio_data_preparation.py --verify --audio_dir ./data/caresound_audio/audio_merged")
        print()
        print("  # Test loading:")
        print("  python scripts/data_prep/audio_data_preparation.py --audio_dir ./data/caresound_audio/audio_merged --num_samples 5")
        print()
        print("  # Inspect dataset:")
        print("  python scripts/data_prep/audio_data_preparation.py --inspect")


if __name__ == "__main__":
    main()
