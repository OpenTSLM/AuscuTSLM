# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

#
# CirCorQAEvalDataset — clinician-grounded QA evaluation from CirCor DigiScope.
#
# Reads the JSONL output of scripts/data_prep/construct_circor_qa.py and provides the
# standard AudioQADataset interface for model evaluation.
#
# Every answer is derived from expert clinical annotations (cardiac physiologists
# and pediatric cardiologists), not from LLM-generated ground truth.
#

import json
import os
from typing import List, Literal, Optional, Tuple

import numpy as np

from opentslm.prompt.text_time_series_prompt import TextTimeSeriesPrompt
from opentslm.time_series_datasets.QADataset import QADataset
from opentslm.audio_model_config import AUDIO_SAMPLE_RATE as SAMPLE_RATE


class CirCorQAEvalDataset(QADataset):
    """
    Clinician-grounded QA evaluation set from the CirCor DigiScope dataset.

    Loads pre-generated QA pairs (from construct_circor_qa.py) where every
    answer is traceable to expert annotations. Provides the standard
    AudioQADataset interface for evaluation.
    """

    def __init__(
        self,
        split: Literal["train", "test", "validation"],
        EOS_TOKEN: str,
        qa_jsonl_path: str = "data/circor_qa/circor_qa.jsonl",
        audio_dir: str = "data/circor_qa/audio",
        tiers: Optional[List[str]] = None,
        max_audio_length: float = 5.0,
    ):
        self.qa_jsonl_path = qa_jsonl_path
        self.audio_dir = audio_dir
        self.tiers = tiers
        self.max_audio_length = max_audio_length
        self.max_audio_samples = int(max_audio_length * SAMPLE_RATE)

        super().__init__(split, EOS_TOKEN)

    def _load_splits(self) -> Tuple[list, list, list]:
        """Load QA pairs from JSONL and split by subject."""
        if not os.path.exists(self.qa_jsonl_path):
            raise FileNotFoundError(
                f"CirCor QA JSONL not found: {self.qa_jsonl_path}\n"
                "Run: python scripts/data_prep/construct_circor_qa.py --out_dir data/circor_qa"
            )

        # Load all QA pairs
        qa_pairs = []
        with open(self.qa_jsonl_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    qa_pairs.append(json.loads(line))

        # Filter by tiers if specified
        if self.tiers:
            qa_pairs = [qa for qa in qa_pairs if qa["question_tier"] in self.tiers]

        # Load audio for each QA pair
        samples = []
        for qa in qa_pairs:
            audio_path = os.path.join(self.audio_dir, qa["audio_file"])
            if not os.path.exists(audio_path):
                continue

            try:
                import soundfile as sf
                audio_array, sr = sf.read(audio_path)
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                audio_array = audio_array.astype(np.float32)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"Failed to load audio {audio_path}: {e}"
                )
                continue

            samples.append({
                "audio_array": audio_array,
                "question": qa["question"],
                "answer": qa["answer"],
                "question_id": qa["question_id"],
                "question_tier": qa["question_tier"],
                "subject_id": qa["subject_id"],
                "audio_file": qa["audio_file"],
                "auscultation_location": qa["auscultation_location"],
                "murmur_status": qa["murmur_status"],
                "outcome": qa["outcome"],
                "demographics": qa.get("demographics", {}),
                "source_fields": qa.get("source_fields", []),
                "source_values": qa.get("source_values", {}),
            })

        print(f"CirCor QA Eval: loaded {len(samples)} samples")

        # This is an evaluation-only dataset: all samples go to test split.
        # Train and validation are empty.
        return [], [], samples

    def _get_answer(self, row) -> str:
        return row["answer"]

    def _get_pre_prompt(self, row) -> str:
        return (
            "You are an expert clinician analyzing a heart sound recording. "
            "The recording is from the CirCor DigiScope phonocardiogram dataset "
            "and was obtained during a pediatric cardiac screening. "
            "Listen carefully to the audio signal and answer the clinical question.\n\n"
        )

    def _get_post_prompt(self, row) -> str:
        return f"\n\nQuestion: {row['question']}\n\nAnswer:"

    def _get_text_time_series_prompt_list(self, row) -> List[TextTimeSeriesPrompt]:
        audio_array = np.array(row["audio_array"], dtype=np.float32)

        # Normalize: zero mean, unit std
        mean = audio_array.mean()
        std = max(float(audio_array.std()), 1e-6)
        audio_array = (audio_array - mean) / std

        # Truncate to max length
        if len(audio_array) > self.max_audio_samples:
            audio_array = audio_array[: self.max_audio_samples]

        loc = row.get("auscultation_location", "unknown")
        dur = len(audio_array) / SAMPLE_RATE
        description_text = (
            f"The following is a phonocardiogram recording from the "
            f"{loc} auscultation position at {SAMPLE_RATE}Hz, duration {dur:.2f}s:"
        )
        return [TextTimeSeriesPrompt(description_text, audio_array)]

    def _format_sample(self, row):
        sample = super()._format_sample(row)
        sample["question_id"] = row.get("question_id", "")
        sample["question_tier"] = row.get("question_tier", "")
        sample["subject_id"] = row.get("subject_id", "")
        sample["murmur_status"] = row.get("murmur_status", "")
        sample["outcome"] = row.get("outcome", "")
        sample["label"] = row.get("murmur_status", "")
        sample["dataset_name"] = "CirCor_QA_Eval"
        sample["metadata"] = {
            "source_fields": row.get("source_fields", []),
            "source_values": row.get("source_values", {}),
            "demographics": row.get("demographics", {}),
        }
        return sample

    @staticmethod
    def get_labels() -> List[str]:
        return ["Present", "Absent"]
