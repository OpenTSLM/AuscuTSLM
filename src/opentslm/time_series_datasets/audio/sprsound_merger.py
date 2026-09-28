# SPDX-FileCopyrightText: 2025 Stanford University, ETH Zurich, and the project authors (see CONTRIBUTORS.md)
# SPDX-FileCopyrightText: 2025 This source file is part of the OpenTSLM open-source project.
#
# SPDX-License-Identifier: MIT

"""
SPRSound segment merger utility.

SPRSound audio files are named with the format:
    {PatientID}_{Timestamp}_{Type}_{Position}_{ClipID}.wav
    
Example: 40069321_15.3_0_p4_984.wav
    - PatientID: 40069321 (unique patient identifier)
    - Timestamp: 15.3 (start time in seconds within original recording)
    - Type: 0 (sound type, e.g., 0=Normal, 1=Adventitious)
    - Position: p4 (recording position on chest/back)
    - ClipID: 984 (unique segment identifier)

This module merges these segments back into coherent recordings:
1. Group by PatientID + Position (different positions = different lung lobes)
2. Sort by Timestamp (primary) and ClipID (secondary)
3. Handle overlaps with crossfade blending
4. Handle gaps with silence or smooth interpolation
"""

import os
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class SPRSoundSegment:
    """Represents a single SPRSound audio segment."""
    filepath: str
    patient_id: str
    timestamp: float
    sound_type: int
    position: str
    clip_id: int
    
    # Filled after loading audio
    waveform: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None
    duration: Optional[float] = None  # Duration in seconds
    
    @property
    def end_time(self) -> float:
        """Calculate end time based on timestamp and duration."""
        if self.duration is None:
            return self.timestamp
        return self.timestamp + self.duration


def parse_sprsound_filename(filename: str) -> Optional[SPRSoundSegment]:
    """
    Parse SPRSound filename into components.
    
    Format: {PatientID}_{Timestamp}_{Type}_{Position}_{ClipID}.wav
    Example: 40069321_15.3_0_p4_984.wav
    
    Args:
        filename: The filename (with or without path)
        
    Returns:
        SPRSoundSegment object or None if parsing fails
    """
    # Get just the filename without path and extension
    basename = os.path.basename(filename)
    name_without_ext = os.path.splitext(basename)[0]
    
    # Split by underscore
    parts = name_without_ext.split('_')
    
    if len(parts) < 5:
        # Not enough parts for SPRSound format
        return None
    
    try:
        # Parse components
        patient_id = parts[0]
        timestamp = float(parts[1])
        sound_type = int(parts[2])
        position = parts[3]
        clip_id = int(parts[4])
        
        return SPRSoundSegment(
            filepath=filename,
            patient_id=patient_id,
            timestamp=timestamp,
            sound_type=sound_type,
            position=position,
            clip_id=clip_id
        )
    except (ValueError, IndexError):
        return None


def is_sprsound_format(filename: str) -> bool:
    """Check if filename matches SPRSound naming format."""
    return parse_sprsound_filename(filename) is not None


def find_sprsound_segments(
    audio_dir: str, 
    patient_id: str,
    position: Optional[str] = None
) -> List[SPRSoundSegment]:
    """
    Find all SPRSound segments for a given patient (and optionally position).
    
    Args:
        audio_dir: Directory containing audio files
        patient_id: Patient ID to search for
        position: Optional specific position (e.g., 'p4'). If None, returns all positions.
        
    Returns:
        List of SPRSoundSegment objects
    """
    segments = []
    audio_extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.flac', '.FLAC']
    
    # Search in audio_dir and SPRSound subdirectory
    search_dirs = [audio_dir]
    sprsound_subdir = os.path.join(audio_dir, "SPRSound")
    if os.path.exists(sprsound_subdir):
        search_dirs.append(sprsound_subdir)
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
            
        for filename in os.listdir(search_dir):
            # Skip hidden files
            if filename.startswith('.') or filename.startswith('._'):
                continue
                
            # Check extension
            if not any(filename.endswith(ext) for ext in audio_extensions):
                continue
            
            # Try to parse as SPRSound format
            segment = parse_sprsound_filename(filename)
            if segment is None:
                continue
                
            # Check patient_id match
            if segment.patient_id != patient_id:
                continue
                
            # Check position filter
            if position is not None and segment.position != position:
                continue
            
            # Update filepath to full path
            segment.filepath = os.path.join(search_dir, filename)
            segments.append(segment)
    
    return segments


def group_segments_by_position(segments: List[SPRSoundSegment]) -> Dict[str, List[SPRSoundSegment]]:
    """
    Group segments by recording position.
    
    Different positions represent different lung lobes and should NOT be merged together.
    
    Args:
        segments: List of SPRSoundSegment objects
        
    Returns:
        Dictionary mapping position -> list of segments
    """
    groups = defaultdict(list)
    for segment in segments:
        groups[segment.position].append(segment)
    return dict(groups)


def sort_segments(segments: List[SPRSoundSegment]) -> List[SPRSoundSegment]:
    """
    Sort segments by timestamp (primary) and clip_id (secondary).
    
    Args:
        segments: List of SPRSoundSegment objects
        
    Returns:
        Sorted list of segments
    """
    return sorted(segments, key=lambda s: (s.timestamp, s.clip_id))


def crossfade(audio1: np.ndarray, audio2: np.ndarray, fade_samples: int) -> np.ndarray:
    """
    Apply crossfade between two audio arrays.
    
    Args:
        audio1: First audio array (end will be faded out)
        audio2: Second audio array (beginning will be faded in)
        fade_samples: Number of samples for crossfade
        
    Returns:
        Crossfaded audio array
    """
    if fade_samples <= 0:
        return np.concatenate([audio1, audio2])
    
    # Ensure fade_samples doesn't exceed audio lengths
    fade_samples = min(fade_samples, len(audio1), len(audio2))
    
    if fade_samples == 0:
        return np.concatenate([audio1, audio2])
    
    # Create fade curves (linear crossfade)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    fade_in = np.linspace(0.0, 1.0, fade_samples)
    
    # Get the overlap regions
    end_of_audio1 = audio1[-fade_samples:]
    start_of_audio2 = audio2[:fade_samples]
    
    # Blend the overlap region
    blended = end_of_audio1 * fade_out + start_of_audio2 * fade_in
    
    # Concatenate: [audio1 without overlap] + [blended] + [audio2 without overlap]
    result = np.concatenate([
        audio1[:-fade_samples],
        blended,
        audio2[fade_samples:]
    ])
    
    return result


def load_segment_audio(
    segment: SPRSoundSegment,
    target_sample_rate: int = 16000
) -> SPRSoundSegment:
    """
    Load audio for a segment and update its waveform and duration.
    
    Args:
        segment: SPRSoundSegment object
        target_sample_rate: Target sample rate for resampling
        
    Returns:
        Updated SPRSoundSegment with waveform loaded
    """
    from opentslm.time_series_datasets.audio_util import load_audio
    import torch
    
    try:
        waveform, sr = load_audio(segment.filepath, target_sample_rate)
        
        # Convert Tensor to numpy array
        if isinstance(waveform, torch.Tensor):
            # Handle shape: [channels, samples] -> [samples]
            if waveform.dim() == 2:
                waveform = waveform.mean(dim=0)  # Convert to mono
            waveform = waveform.numpy()
        
        segment.waveform = waveform
        segment.sample_rate = sr
        segment.duration = len(waveform) / sr
    except Exception as e:
        print(f"Error loading segment {segment.filepath}: {e}")
        segment.waveform = None
        segment.sample_rate = None
        segment.duration = None
    
    return segment


def merge_segments(
    segments: List[SPRSoundSegment],
    target_sample_rate: int = 16000,
    crossfade_ms: float = 10.0,
    insert_gaps: bool = True,
    max_gap_seconds: float = 2.0
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Merge multiple audio segments into a single continuous recording.
    
    Strategy:
    1. Sort segments by timestamp and clip_id
    2. Load each segment's audio
    3. For contiguous segments: use crossfade to avoid clicks
    4. For overlapping segments: crop overlap region, then crossfade
    5. For gaps: insert silence (optional) or just concatenate
    
    Args:
        segments: List of SPRSoundSegment objects (should be from same position)
        target_sample_rate: Target sample rate
        crossfade_ms: Crossfade duration in milliseconds for smooth transitions
        insert_gaps: If True, insert silence for time gaps; if False, concatenate directly
        max_gap_seconds: Maximum gap to fill with silence (larger gaps are capped)
        
    Returns:
        Tuple of (merged_waveform, segment_info_list)
        segment_info_list contains metadata about each segment's position in the merged audio
    """
    if len(segments) == 0:
        return np.array([]), []
    
    # Sort segments
    segments = sort_segments(segments)
    
    # Load audio for all segments
    for segment in segments:
        if segment.waveform is None:
            load_segment_audio(segment, target_sample_rate)
    
    # Filter out segments that failed to load
    segments = [s for s in segments if s.waveform is not None]
    
    if len(segments) == 0:
        return np.array([]), []
    
    if len(segments) == 1:
        return segments[0].waveform, [{
            'patient_id': segments[0].patient_id,
            'position': segments[0].position,
            'timestamp': segments[0].timestamp,
            'clip_id': segments[0].clip_id,
            'start_sample': 0,
            'end_sample': len(segments[0].waveform)
        }]
    
    # Calculate crossfade samples
    crossfade_samples = int(crossfade_ms * target_sample_rate / 1000)
    
    # Merge segments
    merged = segments[0].waveform.copy()
    segment_info = [{
        'patient_id': segments[0].patient_id,
        'position': segments[0].position,
        'timestamp': segments[0].timestamp,
        'clip_id': segments[0].clip_id,
        'start_sample': 0,
        'end_sample': len(merged)
    }]
    
    current_end_time = segments[0].end_time
    
    for i in range(1, len(segments)):
        prev_segment = segments[i - 1]
        curr_segment = segments[i]
        
        # Calculate time relationship
        gap_seconds = curr_segment.timestamp - current_end_time
        gap_samples = int(gap_seconds * target_sample_rate)
        
        if gap_seconds < -0.001:  # Overlap (negative gap, with small tolerance)
            # Overlapping segments
            overlap_samples = int(-gap_seconds * target_sample_rate)
            
            # Crop the overlap from the end of merged audio
            crop_amount = min(overlap_samples, len(merged) // 4)  # Don't crop more than 25%
            
            if crop_amount > 0:
                merged = merged[:-crop_amount]
            
            # Apply crossfade
            merged = crossfade(merged, curr_segment.waveform, crossfade_samples)
            
        elif gap_seconds <= 0.05:  # Nearly contiguous (< 50ms gap)
            # Use crossfade for smooth transition
            merged = crossfade(merged, curr_segment.waveform, crossfade_samples)
            
        else:  # Gap exists
            if insert_gaps:
                # Insert silence for the gap (capped at max_gap_seconds)
                actual_gap = min(gap_seconds, max_gap_seconds)
                silence_samples = int(actual_gap * target_sample_rate)
                silence = np.zeros(silence_samples, dtype=merged.dtype)
                
                # Concatenate with crossfade at boundaries
                merged = crossfade(merged, silence, crossfade_samples)
                merged = crossfade(merged, curr_segment.waveform, crossfade_samples)
            else:
                # Just crossfade directly
                merged = crossfade(merged, curr_segment.waveform, crossfade_samples)
        
        # Update segment info
        segment_info.append({
            'patient_id': curr_segment.patient_id,
            'position': curr_segment.position,
            'timestamp': curr_segment.timestamp,
            'clip_id': curr_segment.clip_id,
            'start_sample': len(merged) - len(curr_segment.waveform),
            'end_sample': len(merged)
        })
        
        # Update current end time
        current_end_time = max(current_end_time, curr_segment.end_time)
    
    return merged, segment_info


def load_and_merge_sprsound_patient(
    audio_dir: str,
    patient_id: str,
    target_sample_rate: int = 16000,
    crossfade_ms: float = 10.0,
    insert_gaps: bool = True,
    max_gap_seconds: float = 2.0,
    merge_positions: bool = False
) -> Dict[str, Tuple[np.ndarray, List[Dict]]]:
    """
    Load and merge all SPRSound segments for a patient.
    
    Args:
        audio_dir: Directory containing audio files
        patient_id: Patient ID to load
        target_sample_rate: Target sample rate
        crossfade_ms: Crossfade duration in milliseconds
        insert_gaps: If True, insert silence for time gaps
        max_gap_seconds: Maximum gap to fill with silence
        merge_positions: If True, merge all positions into one; if False, keep separate
        
    Returns:
        Dictionary mapping position -> (merged_waveform, segment_info_list)
        If merge_positions=True, returns {"all": (waveform, info)}
    """
    # Find all segments for this patient
    segments = find_sprsound_segments(audio_dir, patient_id)
    
    if len(segments) == 0:
        return {}
    
    # Group by position
    position_groups = group_segments_by_position(segments)
    
    if merge_positions:
        # Merge all positions together (not recommended for medical analysis)
        all_segments = []
        for pos_segments in position_groups.values():
            all_segments.extend(pos_segments)
        
        merged, info = merge_segments(
            all_segments,
            target_sample_rate=target_sample_rate,
            crossfade_ms=crossfade_ms,
            insert_gaps=insert_gaps,
            max_gap_seconds=max_gap_seconds
        )
        return {"all": (merged, info)}
    else:
        # Merge each position separately (recommended)
        results = {}
        for position, pos_segments in position_groups.items():
            merged, info = merge_segments(
                pos_segments,
                target_sample_rate=target_sample_rate,
                crossfade_ms=crossfade_ms,
                insert_gaps=insert_gaps,
                max_gap_seconds=max_gap_seconds
            )
            results[position] = (merged, info)
        
        return results


# Position name mapping for readable descriptions
# SPRSound recording locations:
#   p1 — Left Posterior
#   p2 — Left Lateral
#   p3 — Right Posterior
#   p4 — Right Lateral
POSITION_NAMES = {
    'p1': 'Left Posterior',
    'p2': 'Left Lateral',
    'p3': 'Right Posterior',
    'p4': 'Right Lateral',
}


def get_position_description(position: str) -> str:
    """Get human-readable description for a recording position."""
    return POSITION_NAMES.get(position, f"Position {position}")
