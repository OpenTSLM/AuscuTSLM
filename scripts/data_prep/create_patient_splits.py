# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Create patient-level train/val/test splits for the CaReSound dataset.

The HuggingFace CaReSound dataset only has train/test splits that are split
at the *question* level — the same patient can appear in both train and test.
This causes data leakage because the model sees a patient's audio during
training and is then tested on different questions about the same audio.

This script creates a proper **patient-disjoint** split:
  - All QA pairs for a given patient go to exactly ONE split
  - Stratified by source dataset (ICBHI, CirCor, SPRSound, ZCHSound, KAUH)
    so every source is represented in every split
  - Fixed random seed for reproducibility

Default ratios: 70% train / 15% val / 15% test (by patient count)

Output:
    data/caresound_patient_splits.json
    {
        "metadata": { ... },
        "splits": {
            "<patient_id>": "train" | "val" | "test",
            ...
        }
    }

Usage:
    python scripts/data_prep/create_patient_splits.py
    python scripts/data_prep/create_patient_splits.py --train_ratio 0.70 --val_ratio 0.15 --seed 42
"""

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime

from datasets import load_dataset


def create_patient_splits(
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
    cache_dir: str = "data/caresound_cache",
    output_path: str = "data/caresound_patient_splits.json",
):
    """
    Create patient-level splits stratified by source dataset.

    Args:
        train_ratio: Fraction of patients for training (per source dataset)
        val_ratio: Fraction of patients for validation
        seed: Random seed for reproducibility
        cache_dir: HuggingFace dataset cache directory
        output_path: Where to save the split mapping JSON

    Returns:
        Dictionary with split assignments and metadata
    """
    test_ratio = 1.0 - train_ratio - val_ratio
    assert test_ratio > 0, f"test_ratio = {test_ratio:.2f} must be > 0"

    print("=" * 60)
    print("CREATING PATIENT-LEVEL SPLITS FOR CaReSound")
    print("=" * 60)
    print(f"  Ratios: train={train_ratio:.0%}, val={val_ratio:.0%}, test={test_ratio:.0%}")
    print(f"  Seed: {seed}")
    print()

    # Load all data from HuggingFace (both existing splits)
    print("Loading CaReSound dataset from HuggingFace...")
    ds = load_dataset("tsnngw/CaReSound", cache_dir=cache_dir)

    # Pool all rows from both HF splits
    all_rows = []
    for split_name in ds:
        for i in range(len(ds[split_name])):
            all_rows.append(ds[split_name][i])
    print(f"  Total QA pairs: {len(all_rows)}")

    # Group patients by source dataset
    patients_by_source = defaultdict(set)
    qa_per_patient = defaultdict(int)
    for r in all_rows:
        patients_by_source[r["dataset"]].add(r["patient_id"])
        qa_per_patient[(r["dataset"], r["patient_id"])] += 1

    # Total unique patients
    all_patient_ids = set()
    for pids in patients_by_source.values():
        all_patient_ids |= pids
    print(f"  Total unique patients: {len(all_patient_ids)}")
    print()

    # Split patients within each source dataset
    rng = random.Random(seed)
    splits = {}  # patient_id -> "train" / "val" / "test"
    stats_per_source = {}

    for src in sorted(patients_by_source.keys()):
        pids = sorted(patients_by_source[src])  # sort for determinism
        rng.shuffle(pids)

        n = len(pids)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # Remaining go to test
        n_test = n - n_train - n_val

        # Ensure at least 1 patient in each split when source is large enough
        if n >= 3:
            n_train = max(n_train, 1)
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            # Rebalance if we over-allocated
            while n_train + n_val + n_test > n:
                n_train -= 1

        tr_pids = pids[:n_train]
        va_pids = pids[n_train : n_train + n_val]
        te_pids = pids[n_train + n_val :]

        for pid in tr_pids:
            splits[pid] = "train"
        for pid in va_pids:
            splits[pid] = "val"
        for pid in te_pids:
            splits[pid] = "test"

        # Count QA pairs per split
        tr_qs = sum(qa_per_patient[(src, p)] for p in tr_pids)
        va_qs = sum(qa_per_patient[(src, p)] for p in va_pids)
        te_qs = sum(qa_per_patient[(src, p)] for p in te_pids)

        stats_per_source[src] = {
            "patients": {"train": len(tr_pids), "val": len(va_pids), "test": len(te_pids)},
            "qa_pairs": {"train": tr_qs, "val": va_qs, "test": te_qs},
        }

        print(
            f"  {src:12s}: train={len(tr_pids):4d} pts ({tr_qs:5d} QAs), "
            f"val={len(va_pids):3d} pts ({va_qs:4d} QAs), "
            f"test={len(te_pids):3d} pts ({te_qs:4d} QAs)"
        )

    # Aggregate totals
    total_train_pts = sum(s["patients"]["train"] for s in stats_per_source.values())
    total_val_pts = sum(s["patients"]["val"] for s in stats_per_source.values())
    total_test_pts = sum(s["patients"]["test"] for s in stats_per_source.values())
    total_train_qs = sum(s["qa_pairs"]["train"] for s in stats_per_source.values())
    total_val_qs = sum(s["qa_pairs"]["val"] for s in stats_per_source.values())
    total_test_qs = sum(s["qa_pairs"]["test"] for s in stats_per_source.values())
    total_qs = total_train_qs + total_val_qs + total_test_qs

    print()
    print(f"  TOTAL patients: {total_train_pts} train + {total_val_pts} val + {total_test_pts} test = {total_train_pts + total_val_pts + total_test_pts}")
    print(f"  TOTAL QA pairs: {total_train_qs} train ({total_train_qs/total_qs*100:.1f}%) + "
          f"{total_val_qs} val ({total_val_qs/total_qs*100:.1f}%) + "
          f"{total_test_qs} test ({total_test_qs/total_qs*100:.1f}%) = {total_qs}")

    # Verify no overlap
    train_pids = {p for p, s in splits.items() if s == "train"}
    val_pids = {p for p, s in splits.items() if s == "val"}
    test_pids = {p for p, s in splits.items() if s == "test"}
    assert len(train_pids & val_pids) == 0, "Train/val patient overlap!"
    assert len(train_pids & test_pids) == 0, "Train/test patient overlap!"
    assert len(val_pids & test_pids) == 0, "Val/test patient overlap!"
    assert len(splits) == len(all_patient_ids), "Missing patients in split assignment!"
    print()
    print("  ✓ No patient overlap between any splits")

    # Build output
    output = {
        "metadata": {
            "description": "Patient-level train/val/test split for CaReSound dataset",
            "created": datetime.now().isoformat(),
            "seed": seed,
            "ratios": {"train": train_ratio, "val": val_ratio, "test": round(test_ratio, 2)},
            "total_patients": len(all_patient_ids),
            "total_qa_pairs": total_qs,
            "patients_per_split": {"train": total_train_pts, "val": total_val_pts, "test": total_test_pts},
            "qa_pairs_per_split": {"train": total_train_qs, "val": total_val_qs, "test": total_test_qs},
            "per_source": stats_per_source,
        },
        "splits": splits,
    }

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved to: {output_path}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Create patient-level train/val/test splits for CaReSound"
    )
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Fraction of patients for training (default: 0.70)")
    parser.add_argument("--val_ratio", type=float, default=0.15,
                        help="Fraction of patients for validation (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--cache_dir", type=str, default="data/caresound_cache",
                        help="HuggingFace dataset cache directory")
    parser.add_argument("--output", type=str, default="data/caresound_patient_splits.json",
                        help="Output path for split mapping JSON")
    args = parser.parse_args()

    create_patient_splits(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        cache_dir=args.cache_dir,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
