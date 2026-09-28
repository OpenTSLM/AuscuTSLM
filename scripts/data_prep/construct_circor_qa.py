#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
Construct a clinician-grounded QA evaluation set from the CirCor DigiScope
Phonocardiogram Dataset (PhysioNet 2022).

Every answer is derived directly from expert annotations by cardiac
physiologists (murmur characterization) and pediatric cardiologists (outcome).

Produces:
  - circor_qa.jsonl   — QA pairs with provenance fields
  - metadata.json     — distribution statistics
  - audio/            — resampled 16kHz WAV files for selected subjects

Usage:
  python scripts/data_prep/construct_circor_qa.py --out_dir data/circor_qa

  # With existing local PhysioNet download:
  python scripts/data_prep/construct_circor_qa.py \\
      --out_dir data/circor_qa \\
      --physionet_dir /path/to/circor-heart-sound/1.0.3/training_data

  # Dry run (print sample QA pairs, don't write files):
  python scripts/data_prep/construct_circor_qa.py --dry_run --limit 5
"""

import argparse
import csv
import json
import logging
import os
import random
import re
import sys
from collections import Counter, defaultdict
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000

PHYSIONET_BASE = (
    "https://physionet.org/files/circor-heart-sound/1.0.3/"
)

LOCATION_EXPAND = {
    "AV": "aortic valve area",
    "PV": "pulmonic valve area",
    "TV": "tricuspid valve area",
    "MV": "mitral valve area (cardiac apex)",
    "Phc": "phono-cardiogram",
}

# Annotation fields that define murmur characterization
MURMUR_FIELDS = [
    "Murmur locations",
    "Most audible location",
    "Systolic murmur timing",
    "Systolic murmur shape",
    "Systolic murmur grading",
    "Systolic murmur pitch",
    "Systolic murmur quality",
    "Diastolic murmur timing",
    "Diastolic murmur shape",
    "Diastolic murmur grading",
    "Diastolic murmur pitch",
    "Diastolic murmur quality",
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def download_training_csv(cache_dir: str = "/tmp") -> str:
    """Download training_data.csv from PhysioNet if not already cached."""
    path = os.path.join(cache_dir, "circor_training_data.csv")
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        log.info(f"Using cached CSV: {path}")
        return path
    url = f"{PHYSIONET_BASE}.csv"
    log.info(f"Downloading {url} ...")
    import urllib.request

    urllib.request.urlretrieve(url, path)
    log.info(f"Saved to {path}")
    return path


def load_training_csv(csv_path: str) -> List[Dict[str, str]]:
    """Parse training_data.csv into list of dicts, one per subject."""
    subjects = []
    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Clean up field values
            cleaned = {}
            for k, v in row.items():
                k = k.strip().rstrip(":")
                v = v.strip() if v else ""
                if v.lower() == "nan" or v == "":
                    v = None
                cleaned[k] = v
            subjects.append(cleaned)
    log.info(f"Loaded {len(subjects)} subjects from CSV")
    return subjects


def _is_field_valid(value: Optional[str]) -> bool:
    """Check if an annotation field has a valid (non-nan, non-empty) value."""
    return value is not None and value.lower() not in ("nan", "")


def _expand_location(loc: str) -> str:
    """Expand location abbreviation to full clinical name."""
    return LOCATION_EXPAND.get(loc, loc)


def _expand_locations_list(locations_str: str) -> str:
    """Expand 'AV+MV+TV' to 'aortic valve area, mitral valve area, and tricuspid valve area'."""
    if not locations_str:
        return ""
    parts = [_expand_location(loc.strip()) for loc in locations_str.split("+")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


# ═══════════════════════════════════════════════════════════════════════════
# 2. SUBJECT SELECTION
# ═══════════════════════════════════════════════════════════════════════════


def select_subjects(
    subjects: List[Dict],
    n_present: int = 100,
    n_absent_normal: int = 80,
    n_absent_abnormal: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """
    Select evaluation subjects with stratification.

    - n_present murmur-present subjects (stratified by timing and outcome)
    - n_absent_normal murmur-absent, outcome-normal subjects
    - n_absent_abnormal murmur-absent, outcome-abnormal subjects
    - Excludes Murmur == Unknown
    """
    rng = random.Random(seed)

    present = [s for s in subjects if s.get("Murmur") == "Present"]
    absent_normal = [
        s
        for s in subjects
        if s.get("Murmur") == "Absent" and s.get("Outcome") == "Normal"
    ]
    absent_abnormal = [
        s
        for s in subjects
        if s.get("Murmur") == "Absent" and s.get("Outcome") == "Abnormal"
    ]

    # Stratified sampling for present group
    if len(present) <= n_present:
        selected_present = present
    else:
        # Stratify by systolic timing + outcome
        strata = defaultdict(list)
        for s in present:
            key = (
                s.get("Systolic murmur timing", "Unknown"),
                s.get("Outcome", "Unknown"),
            )
            strata[key].append(s)

        selected_present = []
        # Proportional allocation across strata
        for key, group in strata.items():
            n = max(1, round(len(group) / len(present) * n_present))
            rng.shuffle(group)
            selected_present.extend(group[:n])

        # Trim or pad to target
        if len(selected_present) > n_present:
            rng.shuffle(selected_present)
            selected_present = selected_present[:n_present]

    # Sample absent groups
    rng.shuffle(absent_normal)
    selected_absent_normal = absent_normal[: min(n_absent_normal, len(absent_normal))]

    rng.shuffle(absent_abnormal)
    selected_absent_abnormal = absent_abnormal[
        : min(n_absent_abnormal, len(absent_abnormal))
    ]

    selected = selected_present + selected_absent_normal + selected_absent_abnormal

    log.info(
        f"Selected {len(selected)} subjects: "
        f"{len(selected_present)} present, "
        f"{len(selected_absent_normal)} absent-normal, "
        f"{len(selected_absent_abnormal)} absent-abnormal"
    )
    return selected


# ═══════════════════════════════════════════════════════════════════════════
# 3. QUESTION TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════

QUESTION_TEMPLATES = {
    # ── Tier A: Detection (all subjects) ──────────────────────────────────
    "A1": {
        "tier": "A",
        "paraphrases": [
            "Is a cardiac murmur present in this heart sound recording?",
            "Can you detect any murmurs in the following auscultation audio?",
            "Listen to this heart sound. Are there any murmurs?",
            "Does this phonocardiogram contain evidence of a heart murmur?",
        ],
        "source_fields": ["Murmur"],
        "applicable": "all",
    },
    "A2": {
        "tier": "A",
        "paraphrases": [
            "Does this heart sound recording suggest a normal or abnormal cardiac condition?",
            "Based on the auscultation, is the cardiac condition normal or abnormal?",
            "What is the overall cardiac assessment from this heart sound?",
        ],
        "source_fields": ["Outcome"],
        "applicable": "all",
    },
    "A3": {
        "tier": "A",
        "paraphrases": [
            "Should this patient be referred for further cardiac evaluation based on this auscultation?",
            "Does this heart sound warrant further cardiac workup?",
            "Based on this recording, is a cardiology referral indicated?",
        ],
        "source_fields": ["Outcome"],
        "applicable": "all",
    },
    # ── Tier B: Localization (murmur present only) ────────────────────────
    "B1": {
        "tier": "B",
        "paraphrases": [
            "At which auscultation locations is the murmur heard?",
            "Where on the chest can the murmur be auscultated?",
            "Identify the auscultation sites where the murmur is detected.",
        ],
        "source_fields": ["Murmur locations"],
        "applicable": "present",
    },
    "B2": {
        "tier": "B",
        "paraphrases": [
            "Where is the murmur most audible?",
            "At which location is the murmur loudest?",
            "Identify the auscultation site with the most prominent murmur.",
        ],
        "source_fields": ["Most audible location"],
        "applicable": "present",
    },
    "B3": {
        "tier": "B",
        "paraphrases": [
            "Is this murmur systolic, diastolic, or both?",
            "During which phase of the cardiac cycle does the murmur occur?",
            "Classify the murmur as systolic, diastolic, or continuous.",
        ],
        "source_fields": ["Systolic murmur timing", "Diastolic murmur timing"],
        "applicable": "present",
    },
    # ── Tier C: Characterization (murmur present, field-dependent) ────────
    "C1": {
        "tier": "C",
        "paraphrases": [
            "What is the timing of the systolic murmur?",
            "Describe when during systole the murmur occurs.",
            "Characterize the systolic timing of this murmur.",
        ],
        "source_fields": ["Systolic murmur timing"],
        "applicable": "systolic",
    },
    "C2": {
        "tier": "C",
        "paraphrases": [
            "Describe the shape of the systolic murmur.",
            "What is the configuration or envelope of the systolic murmur?",
            "Characterize the intensity pattern of this systolic murmur.",
        ],
        "source_fields": ["Systolic murmur shape"],
        "applicable": "systolic",
    },
    "C3": {
        "tier": "C",
        "paraphrases": [
            "What is the pitch of the murmur?",
            "Is the murmur low-pitched, medium-pitched, or high-pitched?",
            "Describe the frequency characteristics of this murmur.",
        ],
        "source_fields": ["Systolic murmur pitch", "Diastolic murmur pitch"],
        "applicable": "has_murmur_pitch",
    },
    "C4": {
        "tier": "C",
        "paraphrases": [
            "What is the intensity grading of the murmur?",
            "Grade the intensity of this murmur on the Levine scale.",
            "How loud is this murmur?",
        ],
        "source_fields": ["Systolic murmur grading", "Diastolic murmur grading"],
        "applicable": "has_murmur_grading",
    },
    "C5": {
        "tier": "C",
        "paraphrases": [
            "Describe the quality of the murmur.",
            "What is the tonal quality or character of this murmur?",
            "Is the murmur blowing, harsh, or musical?",
        ],
        "source_fields": ["Systolic murmur quality", "Diastolic murmur quality"],
        "applicable": "has_murmur_quality",
    },
    "C6": {
        "tier": "C",
        "paraphrases": [
            "What is the timing of the diastolic murmur?",
            "Describe when during diastole the murmur occurs.",
            "Characterize the diastolic timing of this murmur.",
        ],
        "source_fields": ["Diastolic murmur timing"],
        "applicable": "diastolic",
    },
    # ── Tier D: Clinical reasoning (murmur present, composite) ────────────
    "D1": {
        "tier": "D",
        "paraphrases": [
            "Provide a complete description of the murmur heard in this recording.",
            "Give a detailed characterization of the murmur present in this auscultation.",
            "Describe all features of the murmur detected in this heart sound.",
        ],
        "source_fields": MURMUR_FIELDS,
        "applicable": "present",
    },
    "D2": {
        "tier": "D",
        "paraphrases": [
            "Based on the auscultation findings, what is your clinical assessment of this patient?",
            "Provide a clinical assessment combining the murmur characteristics and overall cardiac status.",
            "What is your overall clinical impression from this heart sound examination?",
        ],
        "source_fields": MURMUR_FIELDS + ["Outcome"],
        "applicable": "present",
    },
    "D3": {
        "tier": "D",
        "paraphrases": [
            "Considering the patient's age and the murmur characteristics, what clinical significance does this finding have?",
            "Given the patient demographics and auscultation findings, assess the clinical significance.",
            "Evaluate the clinical importance of these murmur findings in the context of the patient's age.",
        ],
        "source_fields": MURMUR_FIELDS + ["Outcome", "Age"],
        "applicable": "present",
    },
}


def _is_question_applicable(qid: str, template: dict, subject: Dict) -> bool:
    """Check if a question template is applicable to a given subject."""
    rule = template["applicable"]
    murmur = subject.get("Murmur", "")

    if rule == "all":
        return True
    if rule == "present":
        return murmur == "Present"
    if rule == "systolic":
        return murmur == "Present" and _is_field_valid(
            subject.get("Systolic murmur timing")
        )
    if rule == "diastolic":
        return murmur == "Present" and _is_field_valid(
            subject.get("Diastolic murmur timing")
        )
    if rule == "has_murmur_pitch":
        return murmur == "Present" and (
            _is_field_valid(subject.get("Systolic murmur pitch"))
            or _is_field_valid(subject.get("Diastolic murmur pitch"))
        )
    if rule == "has_murmur_grading":
        return murmur == "Present" and (
            _is_field_valid(subject.get("Systolic murmur grading"))
            or _is_field_valid(subject.get("Diastolic murmur grading"))
        )
    if rule == "has_murmur_quality":
        return murmur == "Present" and (
            _is_field_valid(subject.get("Systolic murmur quality"))
            or _is_field_valid(subject.get("Diastolic murmur quality"))
        )
    return False


# ═══════════════════════════════════════════════════════════════════════════
# 4. ANSWER CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════════════


def _timing_explanation(timing: str) -> str:
    """Map timing value to a short clinical explanation."""
    explanations = {
        "Early-systolic": "occurs during the early portion of the systolic phase",
        "Mid-systolic": "occurs during the middle portion of the systolic phase",
        "Late-systolic": "occurs during the late portion of the systolic phase",
        "Holosystolic": "persists throughout the entire systolic phase",
        "Early-diastolic": "occurs during the early portion of the diastolic phase",
        "Mid-diastolic": "occurs during the middle portion of the diastolic phase",
        "Holodiastolic": "persists throughout the entire diastolic phase",
    }
    return explanations.get(timing, f"occurs during {timing.lower()}")


def construct_answer(qid: str, subject: Dict) -> str:
    """Construct a natural-language answer from expert annotation fields."""
    murmur = subject.get("Murmur", "")
    outcome = subject.get("Outcome", "")

    # ── Tier A ────────────────────────────────────────────────────────────
    if qid == "A1":
        if murmur == "Present":
            locations = subject.get("Murmur locations", "")
            if _is_field_valid(locations):
                loc_text = _expand_locations_list(locations)
                return (
                    f"Yes, a cardiac murmur is present. "
                    f"The murmur was identified at the {loc_text} auscultation location(s)."
                )
            return "Yes, a cardiac murmur is present in this recording."
        return (
            "No, no cardiac murmur was detected in this recording. "
            "The heart sounds appear normal."
        )

    if qid == "A2":
        if outcome == "Abnormal":
            return "This heart sound recording suggests an abnormal cardiac condition."
        return "This heart sound recording suggests a normal cardiac condition."

    if qid == "A3":
        if outcome == "Abnormal":
            return (
                "Yes, this patient should be referred for further cardiac evaluation. "
                "The auscultation findings suggest an abnormal cardiac condition "
                "that warrants further workup."
            )
        return (
            "No, further cardiac evaluation is not urgently indicated. "
            "The auscultation findings are consistent with a normal cardiac condition."
        )

    # ── Tier B ────────────────────────────────────────────────────────────
    if qid == "B1":
        locations = subject.get("Murmur locations", "")
        loc_text = _expand_locations_list(locations)
        return f"The murmur is heard at the {loc_text} auscultation location(s)."

    if qid == "B2":
        loc = subject.get("Most audible location", "")
        return (
            f"The murmur is most audible at the "
            f"{_expand_location(loc)} position."
        )

    if qid == "B3":
        has_sys = _is_field_valid(subject.get("Systolic murmur timing"))
        has_dia = _is_field_valid(subject.get("Diastolic murmur timing"))
        if has_sys and has_dia:
            return (
                "The murmur has both systolic and diastolic components. "
                f"The systolic component has {subject['Systolic murmur timing'].lower()} timing, "
                f"and the diastolic component has {subject['Diastolic murmur timing'].lower()} timing."
            )
        if has_sys:
            return (
                f"This is a systolic murmur with {subject['Systolic murmur timing'].lower()} timing. "
                "No diastolic murmur was detected."
            )
        if has_dia:
            return (
                f"This is a diastolic murmur with {subject['Diastolic murmur timing'].lower()} timing. "
                "No systolic murmur was detected."
            )
        return "The cardiac cycle phase of the murmur could not be clearly determined."

    # ── Tier C ────────────────────────────────────────────────────────────
    if qid == "C1":
        timing = subject.get("Systolic murmur timing", "")
        return (
            f"The systolic murmur has {timing.lower()} timing. "
            f"This means the murmur {_timing_explanation(timing)} of the cardiac cycle."
        )

    if qid == "C2":
        shape = subject.get("Systolic murmur shape", "")
        shape_desc = {
            "Crescendo": "progressively increases in intensity",
            "Decrescendo": "progressively decreases in intensity",
            "Diamond": "increases then decreases in intensity (crescendo-decrescendo)",
            "Plateau": "maintains a constant intensity throughout",
        }
        desc = shape_desc.get(shape, f"has a {shape.lower()} pattern")
        return (
            f"The systolic murmur has a {shape.lower()} shape, meaning it {desc}."
        )

    if qid == "C3":
        sys_pitch = subject.get("Systolic murmur pitch")
        dia_pitch = subject.get("Diastolic murmur pitch")
        if _is_field_valid(sys_pitch) and _is_field_valid(dia_pitch):
            return (
                f"The systolic murmur is {sys_pitch.lower()}-pitched "
                f"and the diastolic murmur is {dia_pitch.lower()}-pitched."
            )
        if _is_field_valid(sys_pitch):
            return f"The murmur is {sys_pitch.lower()}-pitched."
        if _is_field_valid(dia_pitch):
            return f"The diastolic murmur is {dia_pitch.lower()}-pitched."
        return "The pitch of the murmur could not be clearly determined."

    if qid == "C4":
        sys_grade = subject.get("Systolic murmur grading")
        dia_grade = subject.get("Diastolic murmur grading")
        if _is_field_valid(sys_grade) and _is_field_valid(dia_grade):
            return (
                f"The systolic murmur is graded {sys_grade} on the Levine scale, "
                f"and the diastolic murmur is graded {dia_grade}."
            )
        if _is_field_valid(sys_grade):
            return f"The murmur is graded {sys_grade} on the Levine scale."
        if _is_field_valid(dia_grade):
            return f"The diastolic murmur is graded {dia_grade}."
        return "The intensity grading could not be clearly determined."

    if qid == "C5":
        sys_qual = subject.get("Systolic murmur quality")
        dia_qual = subject.get("Diastolic murmur quality")
        if _is_field_valid(sys_qual) and _is_field_valid(dia_qual):
            return (
                f"The systolic murmur has a {sys_qual.lower()} quality "
                f"and the diastolic murmur has a {dia_qual.lower()} quality."
            )
        if _is_field_valid(sys_qual):
            return f"The murmur has a {sys_qual.lower()} quality."
        if _is_field_valid(dia_qual):
            return f"The diastolic murmur has a {dia_qual.lower()} quality."
        return "The quality of the murmur could not be clearly determined."

    if qid == "C6":
        timing = subject.get("Diastolic murmur timing", "")
        return (
            f"The diastolic murmur has {timing.lower()} timing. "
            f"This means the murmur {_timing_explanation(timing)} of the cardiac cycle."
        )

    # ── Tier D ────────────────────────────────────────────────────────────
    if qid == "D1":
        return _compose_full_description(subject)

    if qid == "D2":
        desc = _compose_full_description(subject)
        if outcome == "Abnormal":
            desc += (
                " Given these findings, this patient should be referred for "
                "echocardiographic evaluation to rule out structural heart disease."
            )
        else:
            desc += (
                " Given these findings, the murmur is likely benign and does not "
                "require immediate further workup, though continued monitoring "
                "is advisable."
            )
        return desc

    if qid == "D3":
        age = subject.get("Age") or "unknown"
        desc = _compose_full_description(subject)
        if outcome == "Abnormal":
            desc += (
                f" In a patient of {age.lower()} age, these findings are clinically "
                "significant and warrant further investigation including echocardiography."
            )
        else:
            desc += (
                f" In a patient of {age.lower()} age, this murmur pattern is "
                "commonly encountered as an innocent or functional murmur and is "
                "not considered clinically significant."
            )
        return desc

    return ""


def _compose_full_description(subject: Dict) -> str:
    """Compose a complete murmur description from all available fields."""
    parts = []

    # Systolic component
    sys_timing = subject.get("Systolic murmur timing")
    sys_shape = subject.get("Systolic murmur shape")
    sys_grading = subject.get("Systolic murmur grading")
    sys_pitch = subject.get("Systolic murmur pitch")
    sys_quality = subject.get("Systolic murmur quality")

    if _is_field_valid(sys_timing):
        article = "An" if sys_timing[0].lower() in "aeiou" else "A"
        sys_desc = f"{article} {sys_timing.lower()}"
        if _is_field_valid(sys_shape):
            sys_desc += f", {sys_shape.lower()}-shaped"
        sys_desc += " systolic murmur is present"
        if _is_field_valid(sys_grading):
            sys_desc += f", graded {sys_grading} on the Levine scale"
        sys_desc += "."
        parts.append(sys_desc)

        qualifiers = []
        if _is_field_valid(sys_pitch):
            qualifiers.append(f"{sys_pitch.lower()}-pitched")
        if _is_field_valid(sys_quality):
            qualifiers.append(f"with a {sys_quality.lower()} quality")
        if qualifiers:
            parts.append(f"The murmur is {' '.join(qualifiers)}.")

    # Diastolic component
    dia_timing = subject.get("Diastolic murmur timing")
    dia_shape = subject.get("Diastolic murmur shape")
    dia_grading = subject.get("Diastolic murmur grading")
    dia_pitch = subject.get("Diastolic murmur pitch")
    dia_quality = subject.get("Diastolic murmur quality")

    if _is_field_valid(dia_timing):
        dia_article = "an" if dia_timing[0].lower() in "aeiou" else "a"
        dia_desc = f"Additionally, {dia_article} {dia_timing.lower()}"
        if _is_field_valid(dia_shape):
            dia_desc += f", {dia_shape.lower()}"
        dia_desc += " diastolic murmur is noted"
        if _is_field_valid(dia_grading):
            dia_desc += f", graded {dia_grading}"
        if _is_field_valid(dia_pitch):
            dia_desc += f", {dia_pitch.lower()}-pitched"
        if _is_field_valid(dia_quality):
            dia_desc += f", with a {dia_quality.lower()} quality"
        dia_desc += "."
        parts.append(dia_desc)
    elif _is_field_valid(sys_timing):
        parts.append("No diastolic murmur was detected.")

    # Location
    loc = subject.get("Most audible location")
    if _is_field_valid(loc):
        parts.append(
            f"The murmur is most audible at the {_expand_location(loc)} position."
        )

    # Outcome
    outcome = subject.get("Outcome", "")
    parts.append(f"The overall cardiac assessment is {outcome.lower()}.")

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
# 5. AUDIO HANDLING
# ═══════════════════════════════════════════════════════════════════════════


def get_audio_for_subject(
    subject: Dict,
    physionet_dir: Optional[str] = None,
    hf_dataset=None,
    hf_index: Optional[Dict[str, int]] = None,
) -> List[Dict[str, Any]]:
    """
    Get audio files for a subject.

    Returns list of dicts: [{"location": "TV", "audio_array": np.ndarray, "sr": 16000}, ...]

    Tries PhysioNet local files first, falls back to HuggingFace.
    """
    patient_id = subject["Patient ID"]
    recordings = []

    # Parse recording locations
    rec_locs_str = subject.get("Recording locations", "")
    if rec_locs_str:
        rec_locs = [loc.strip() for loc in rec_locs_str.split("+")]
    else:
        rec_locs = ["AV", "PV", "TV", "MV"]

    for loc in rec_locs:
        audio_array = None
        sr = SAMPLE_RATE

        # Try PhysioNet local directory first
        if physionet_dir:
            wav_path = os.path.join(physionet_dir, f"{patient_id}_{loc}.wav")
            if os.path.exists(wav_path):
                try:
                    import soundfile as sf

                    arr, file_sr = sf.read(wav_path)
                    if arr.ndim > 1:
                        arr = arr.mean(axis=1)
                    arr = arr.astype(np.float32)
                    if file_sr != SAMPLE_RATE:
                        import torchaudio
                        import torch

                        wf = torch.tensor(arr).unsqueeze(0)
                        wf = torchaudio.transforms.Resample(file_sr, SAMPLE_RATE)(wf)
                        arr = wf.squeeze(0).numpy()
                    audio_array = arr
                except Exception as e:
                    log.debug(f"Failed to load {wav_path}: {e}")

        # Fallback to HuggingFace
        if audio_array is None and hf_dataset is not None and hf_index is not None:
            key = f"{patient_id}_{loc}"
            idx = hf_index.get(key)
            if idx is not None:
                row = hf_dataset[idx]
                rec = row.get("recording", row.get("audio", {}))
                if isinstance(rec, dict):
                    import io
                    import soundfile as sf
                    # decode=False gives {"bytes": ..., "path": ...}
                    if rec.get("bytes"):
                        arr, file_sr = sf.read(io.BytesIO(rec["bytes"]))
                    elif rec.get("path") and os.path.exists(rec["path"]):
                        arr, file_sr = sf.read(rec["path"])
                    elif "array" in rec:
                        # fallback: already decoded (older datasets version)
                        arr = np.array(rec["array"], dtype=np.float32)
                        file_sr = rec.get("sampling_rate", SAMPLE_RATE)
                    else:
                        arr = None
                    if arr is not None:
                        if hasattr(arr, "ndim") and arr.ndim > 1:
                            arr = arr.mean(axis=1)
                        audio_array = arr.astype(np.float32)
                        if file_sr != SAMPLE_RATE:
                            import torchaudio
                            import torch
                            wf = torch.tensor(audio_array).unsqueeze(0)
                            wf = torchaudio.transforms.Resample(file_sr, SAMPLE_RATE)(wf)
                            audio_array = wf.squeeze(0).numpy()

        if audio_array is not None:
            recordings.append(
                {"location": loc, "audio_array": audio_array, "sr": SAMPLE_RATE}
            )

    return recordings


def _select_primary_audio(
    subject: Dict, recordings: List[Dict]
) -> Optional[Dict]:
    """Select the primary audio recording for a subject."""
    if not recordings:
        return None

    murmur = subject.get("Murmur", "")
    most_audible = subject.get("Most audible location")

    if murmur == "Present" and _is_field_valid(most_audible):
        # Use recording from most audible location
        for rec in recordings:
            if rec["location"] == most_audible:
                return rec

    # Fallback: use first available recording
    return recordings[0]


def save_audio(audio_array: np.ndarray, path: str, sr: int = SAMPLE_RATE):
    """Write audio to WAV file."""
    import soundfile as sf

    os.makedirs(os.path.dirname(path), exist_ok=True)
    sf.write(path, audio_array, sr, subtype="PCM_16")


# ═══════════════════════════════════════════════════════════════════════════
# 6. QA PAIR GENERATION
# ═══════════════════════════════════════════════════════════════════════════


def generate_qa_pairs(
    subject: Dict,
    audio_file: str,
    auscultation_location: str,
    seed: int = 42,
) -> List[Dict]:
    """Generate all applicable QA pairs for a subject."""
    rng = random.Random(seed + hash(subject.get("Patient ID", "")))
    qa_pairs = []

    for qid, template in QUESTION_TEMPLATES.items():
        if not _is_question_applicable(qid, template, subject):
            continue

        # Select a random paraphrase
        paraphrase_idx = rng.randint(0, len(template["paraphrases"]) - 1)
        question = template["paraphrases"][paraphrase_idx]
        variant = f"{qid}_v{paraphrase_idx}"

        # Construct answer
        answer = construct_answer(qid, subject)

        # Collect source values
        source_values = {}
        for field in template["source_fields"]:
            val = subject.get(field)
            if _is_field_valid(val):
                source_values[field] = val

        qa_pair = {
            "subject_id": subject["Patient ID"],
            "audio_file": audio_file,
            "auscultation_location": auscultation_location,
            "question_id": variant,
            "question_tier": template["tier"],
            "question": question,
            "answer": answer,
            "source_fields": template["source_fields"],
            "source_values": source_values,
            "murmur_status": subject.get("Murmur", ""),
            "outcome": subject.get("Outcome", ""),
            "demographics": {
                "age": subject.get("Age"),
                "sex": subject.get("Sex"),
            },
        }
        qa_pairs.append(qa_pair)

    return qa_pairs


# ═══════════════════════════════════════════════════════════════════════════
# 7. VALIDATION
# ═══════════════════════════════════════════════════════════════════════════


def validate_qa_pairs(qa_pairs: List[Dict]) -> List[str]:
    """Run automated validation checks. Returns list of issues."""
    issues = []
    for i, qa in enumerate(qa_pairs):
        # Check answer isn't empty
        if not qa["answer"].strip():
            issues.append(f"QA #{i} ({qa['question_id']}): empty answer")

        # Check answer doesn't contain 'nan'
        if "nan" in qa["answer"].lower().split():
            issues.append(
                f"QA #{i} ({qa['question_id']}): answer contains 'nan': "
                f"{qa['answer'][:100]}"
            )

        # Check audio file reference
        if not qa["audio_file"]:
            issues.append(f"QA #{i} ({qa['question_id']}): no audio file")

        # Check source_values isn't empty for non-detection questions
        if qa["question_tier"] in ("B", "C", "D") and not qa["source_values"]:
            issues.append(
                f"QA #{i} ({qa['question_id']}): "
                f"no source values for tier {qa['question_tier']}"
            )

    return issues


# ═══════════════════════════════════════════════════════════════════════════
# 8. OUTPUT
# ═══════════════════════════════════════════════════════════════════════════


def write_jsonl(qa_pairs: List[Dict], path: str):
    """Write QA pairs to JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for qa in qa_pairs:
            fh.write(json.dumps(qa, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(qa_pairs)} QA pairs to {path}")


def write_metadata(qa_pairs: List[Dict], path: str):
    """Write distribution statistics to JSON."""
    tier_counts = Counter(qa["question_tier"] for qa in qa_pairs)
    qid_counts = Counter(qa["question_id"].split("_v")[0] for qa in qa_pairs)
    murmur_counts = Counter(qa["murmur_status"] for qa in qa_pairs)
    outcome_counts = Counter(qa["outcome"] for qa in qa_pairs)
    subjects = set(qa["subject_id"] for qa in qa_pairs)
    age_counts = Counter(
        qa["demographics"].get("age") or "Unknown" for qa in qa_pairs
    )
    sex_counts = Counter(
        qa["demographics"].get("sex") or "Unknown" for qa in qa_pairs
    )

    metadata = {
        "total_qa_pairs": len(qa_pairs),
        "total_subjects": len(subjects),
        "tier_distribution": dict(sorted(tier_counts.items())),
        "question_distribution": dict(sorted(qid_counts.items())),
        "murmur_status_distribution": dict(murmur_counts),
        "outcome_distribution": dict(outcome_counts),
        "demographics": {
            "age_distribution": dict(age_counts),
            "sex_distribution": dict(sex_counts),
        },
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
    log.info(f"Wrote metadata to {path}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--out_dir",
        default="data/circor_qa",
        help="Output directory (default: data/circor_qa)",
    )
    parser.add_argument(
        "--physionet_dir",
        default=None,
        help="Path to local PhysioNet training_data/ directory (with WAV files). "
        "If not provided, audio is loaded from HuggingFace.",
    )
    parser.add_argument(
        "--csv_path",
        default=None,
        help="Path to training_data.csv. Downloaded from PhysioNet if not provided.",
    )
    parser.add_argument(
        "--n_present",
        type=int,
        default=100,
        help="Number of murmur-present subjects to select (default: 100)",
    )
    parser.add_argument(
        "--n_absent_normal",
        type=int,
        default=80,
        help="Number of murmur-absent, outcome-normal subjects (default: 80)",
    )
    parser.add_argument(
        "--n_absent_abnormal",
        type=int,
        default=20,
        help="Number of murmur-absent, outcome-abnormal subjects (default: 20)",
    )
    parser.add_argument(
        "--all_present",
        action="store_true",
        help="Use ALL murmur-present subjects instead of sampling",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print sample QA pairs without writing files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit subjects processed (for debugging)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation on existing output",
    )
    args = parser.parse_args()

    # ── Load CSV ──────────────────────────────────────────────────────────
    if args.csv_path:
        csv_path = args.csv_path
    else:
        csv_path = download_training_csv()

    subjects = load_training_csv(csv_path)

    # ── Validation-only mode ──────────────────────────────────────────────
    if args.validate:
        jsonl_path = os.path.join(args.out_dir, "circor_qa.jsonl")
        if not os.path.exists(jsonl_path):
            log.error(f"JSONL not found: {jsonl_path}")
            sys.exit(1)
        with open(jsonl_path, encoding="utf-8") as fh:
            qa_pairs = [json.loads(line) for line in fh]
        issues = validate_qa_pairs(qa_pairs)
        if issues:
            log.warning(f"Found {len(issues)} issues:")
            for issue in issues:
                log.warning(f"  {issue}")
        else:
            log.info("All QA pairs passed validation.")

        # Print sample QA pairs
        rng = random.Random(args.seed)
        sample = rng.sample(qa_pairs, min(10, len(qa_pairs)))
        print("\n=== Sample QA Pairs ===")
        for qa in sample:
            print(f"\n[{qa['question_id']}] Subject {qa['subject_id']} "
                  f"({qa['murmur_status']}, {qa['outcome']})")
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")
        return

    # ── Select subjects ───────────────────────────────────────────────────
    n_present = len([s for s in subjects if s.get("Murmur") == "Present"]) if args.all_present else args.n_present
    selected = select_subjects(
        subjects,
        n_present=n_present,
        n_absent_normal=args.n_absent_normal,
        n_absent_abnormal=args.n_absent_abnormal,
        seed=args.seed,
    )

    if args.limit:
        selected = selected[: args.limit]
        log.info(f"Limited to {len(selected)} subjects (--limit {args.limit})")

    # ── Load audio source ─────────────────────────────────────────────────
    hf_dataset = None
    hf_index = None
    if not args.physionet_dir:
        log.info("Loading audio from HuggingFace (no --physionet_dir provided)...")
        from datasets import load_dataset

        hf_dataset = load_dataset(
            "miguellmartins/circor-digiscope-physionet22",
            split="train",
            trust_remote_code=True,
        )
        # Disable HuggingFace audio auto-decoding (requires torchcodec which may
        # not be installed). We decode bytes manually with soundfile instead.
        from datasets import Audio as _HFAudio
        for _col, _feat in hf_dataset.features.items():
            if isinstance(_feat, _HFAudio):
                hf_dataset = hf_dataset.cast_column(_col, _HFAudio(decode=False))
                log.info(f"Disabled auto-decode for HF audio column '{_col}'")

        # Build filename -> index mapping
        hf_index = {}
        for i, row in enumerate(hf_dataset):
            fn = row.get("filename", "")
            basename = os.path.basename(fn)
            stem = os.path.splitext(basename)[0]
            hf_index[stem] = i
        log.info(f"HuggingFace index: {len(hf_index)} recordings")

    # ── Generate QA pairs ─────────────────────────────────────────────────
    all_qa_pairs = []
    audio_saved = 0
    subjects_skipped = 0

    for subject in selected:
        patient_id = subject["Patient ID"]

        # Get audio recordings
        recordings = get_audio_for_subject(
            subject,
            physionet_dir=args.physionet_dir,
            hf_dataset=hf_dataset,
            hf_index=hf_index,
        )

        # Select primary audio
        primary = _select_primary_audio(subject, recordings)
        if primary is None:
            log.warning(f"No audio found for subject {patient_id} — skipping")
            subjects_skipped += 1
            continue

        audio_filename = f"{patient_id}_{primary['location']}.wav"
        auscultation_location = primary["location"]

        # Save audio (unless dry run)
        if not args.dry_run:
            audio_path = os.path.join(args.out_dir, "audio", audio_filename)
            save_audio(primary["audio_array"], audio_path)
            audio_saved += 1

        # Generate QA pairs
        qa_pairs = generate_qa_pairs(
            subject, audio_filename, auscultation_location, seed=args.seed
        )
        all_qa_pairs.extend(qa_pairs)

    log.info(
        f"Generated {len(all_qa_pairs)} QA pairs from "
        f"{len(selected) - subjects_skipped} subjects "
        f"({subjects_skipped} skipped due to missing audio)"
    )

    # ── Validate ──────────────────────────────────────────────────────────
    issues = validate_qa_pairs(all_qa_pairs)
    if issues:
        log.warning(f"Validation found {len(issues)} issues:")
        for issue in issues[:20]:
            log.warning(f"  {issue}")
    else:
        log.info("All QA pairs passed validation.")

    # ── Dry run: print samples ────────────────────────────────────────────
    if args.dry_run:
        for tier in ["A", "B", "C", "D"]:
            tier_pairs = [qa for qa in all_qa_pairs if qa["question_tier"] == tier]
            print(f"\n{'='*60}")
            print(f"Tier {tier}: {len(tier_pairs)} QA pairs")
            print(f"{'='*60}")
            for qa in tier_pairs[:3]:
                print(f"\n  [{qa['question_id']}] Subject {qa['subject_id']} "
                      f"({qa['murmur_status']}, {qa['outcome']})")
                print(f"  Q: {qa['question']}")
                print(f"  A: {qa['answer']}")
                if qa["source_values"]:
                    print(f"  Sources: {qa['source_values']}")
        return

    # ── Write output ──────────────────────────────────────────────────────
    jsonl_path = os.path.join(args.out_dir, "circor_qa.jsonl")
    write_jsonl(all_qa_pairs, jsonl_path)

    metadata_path = os.path.join(args.out_dir, "metadata.json")
    write_metadata(all_qa_pairs, metadata_path)

    log.info(f"Audio files saved: {audio_saved}")
    log.info(f"Done. Output in {args.out_dir}/")


if __name__ == "__main__":
    main()
