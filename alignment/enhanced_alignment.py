"""
Enhanced Alignment Module
Combines Whisper timestamps with diarization segments for word-level timing
"""

import logging
from typing import List, Dict, Optional, Any, Tuple
import numpy as np

logger = logging.getLogger(__name__)

def merge_word_timestamps_with_diarization(
    whisper_segments: List[Dict],
    diarization_segments: List[Dict]
) -> List[Dict]:
    """
    Merge Whisper word-level timestamps with speaker diarization segments.

    Args:
        whisper_segments: Whisper output with word timestamps
        diarization_segments: PyAnnote diarization segments

    Returns:
        Enhanced segments with speaker attribution and word-level timing
    """
    enhanced_segments = []

    for whisper_seg in whisper_segments:
        segment_start = whisper_seg.get("start", 0)
        segment_end = whisper_seg.get("end", 0)
        segment_text = whisper_seg.get("text", "")
        words = whisper_seg.get("words", [])

        # Find speaker for this segment
        speaker = _find_speaker_for_timestamp(segment_start, segment_end, diarization_segments)

        # Process words with speaker attribution
        enhanced_words = []
        for word in words:
            word_start = word.get("start", 0)
            word_end = word.get("end", 0)
            word_text = word.get("word", "")

            # Find speaker for this specific word (can differ from segment speaker)
            word_speaker = _find_speaker_for_timestamp(word_start, word_end, diarization_segments)

            enhanced_words.append({
                "word": word_text,
                "start": word_start,
                "end": word_end,
                "speaker": word_speaker,
                "confidence": word.get("probability", 0.0)
            })

        # Create enhanced segment
        enhanced_segment = {
            "speaker": speaker,
            "start": segment_start,
            "end": segment_end,
            "text": segment_text,
            "words": enhanced_words,
            "confidence": whisper_seg.get("avg_logprob", 0.0)
        }

        enhanced_segments.append(enhanced_segment)

    return enhanced_segments

def _find_speaker_for_timestamp(
    timestamp: float,
    segment_end: float,
    diarization_segments: List[Dict]
) -> str:
    """
    Find the most appropriate speaker for a given timestamp.

    Args:
        timestamp: Start time in seconds
        segment_end: End time in seconds
        diarization_segments: List of diarization segments

    Returns:
        Speaker identifier
    """
    if not diarization_segments:
        return "SPEAKER_00"

    # Find the segment that contains this timestamp
    for segment in diarization_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        if seg_start <= timestamp <= seg_end:
            return segment["speaker"]

    # If no exact match, find the closest segment
    min_distance = float('inf')
    closest_speaker = "SPEAKER_00"

    for segment in diarization_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]

        # Calculate distance to segment
        if timestamp < seg_start:
            distance = seg_start - timestamp
        elif timestamp > seg_end:
            distance = timestamp - seg_end
        else:
            distance = 0

        if distance < min_distance:
            min_distance = distance
            closest_speaker = segment["speaker"]

    return closest_speaker

def align_segments_with_translation(
    segments: List[Dict],
    translations: List[str]
) -> List[Dict]:
    """
    Align translated text with original segments.

    Args:
        segments: Original segments with timing
        translations: List of translated texts

    Returns:
        Enhanced segments with aligned translations
    """
    if len(segments) != len(translations):
        logger.warning(f"Segment count mismatch: {len(segments)} segments, {len(translations)} translations")
        # Truncate or pad to match
        translations = translations[:len(segments)] + [""] * max(0, len(segments) - len(translations))

    enhanced_segments = []
    for segment, translation in zip(segments, translations):
        enhanced_segment = segment.copy()
        enhanced_segment["translation"] = translation
        enhanced_segments.append(enhanced_segment)

    return enhanced_segments

def create_enhanced_transcript(
    whisper_result: Dict,
    diarization_segments: List[Dict],
    translations: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create an enhanced transcript with word-level timing and speaker attribution.

    Args:
        whisper_result: Whisper transcription result
        diarization_segments: PyAnnote diarization segments
        translations: Optional list of translated texts

    Returns:
        Enhanced transcript with complete information
    """
    # Extract segments from Whisper result
    whisper_segments = whisper_result.get("segments", [])

    if not whisper_segments:
        logger.warning("No segments found in Whisper result")
        return {"segments": [], "metadata": {}}

    # Merge word timestamps with diarization
    enhanced_segments = merge_word_timestamps_with_diarization(
        whisper_segments, diarization_segments
    )

    # Add translations if provided
    if translations:
        enhanced_segments = align_segments_with_translation(enhanced_segments, translations)

    # Calculate metadata
    total_duration = max([seg["end"] for seg in enhanced_segments]) if enhanced_segments else 0
    speakers = list(set([seg["speaker"] for seg in enhanced_segments]))
    word_count = sum([len(seg["words"]) for seg in enhanced_segments])

    metadata = {
        "total_duration": total_duration,
        "speaker_count": len(speakers),
        "speakers": speakers,
        "segment_count": len(enhanced_segments),
        "word_count": word_count,
        "language": whisper_result.get("language", "unknown"),
        "has_translation": translations is not None
    }

    return {
        "segments": enhanced_segments,
        "metadata": metadata
    }

def split_words_at_speaker_changes(
    segments: List[Dict]
) -> List[Dict]:
    """
    Split segments when speaker changes within a segment.

    Args:
        segments: Enhanced segments with word-level speaker attribution

    Returns:
        Segments split by speaker changes
    """
    split_segments = []

    for segment in segments:
        words = segment.get("words", [])
        if not words:
            split_segments.append(segment)
            continue

        # Group words by speaker
        current_speaker = words[0]["speaker"]
        current_words = [words[0]]
        current_start = words[0]["start"]

        for word in words[1:]:
            if word["speaker"] == current_speaker:
                current_words.append(word)
            else:
                # Create segment for current speaker
                if current_words:
                    split_segment = {
                        "speaker": current_speaker,
                        "start": current_start,
                        "end": current_words[-1]["end"],
                        "text": " ".join([w["word"] for w in current_words]),
                        "words": current_words.copy(),
                        "confidence": np.mean([w["confidence"] for w in current_words])
                    }
                    split_segments.append(split_segment)

                # Start new segment
                current_speaker = word["speaker"]
                current_words = [word]
                current_start = word["start"]

        # Add final segment for this original segment
        if current_words:
            split_segment = {
                "speaker": current_speaker,
                "start": current_start,
                "end": current_words[-1]["end"],
                "text": " ".join([w["word"] for w in current_words]),
                "words": current_words.copy(),
                "confidence": np.mean([w["confidence"] for w in current_words])
            }
            split_segments.append(split_segment)

    return split_segments

def smooth_transitions(
    segments: List[Dict],
    min_segment_duration: float = 0.5
) -> List[Dict]:
    """
    Smooth transitions between segments to avoid very short segments.

    Args:
        segments: Enhanced segments
        min_segment_duration: Minimum duration for segments

    Returns:
        Segments with smoothed transitions
    """
    if not segments:
        return segments

    smoothed_segments = []

    for i, segment in enumerate(segments):
        duration = segment["end"] - segment["start"]

        # If segment is too short, consider merging with previous or next
        if duration < min_segment_duration:
            if i > 0 and smoothed_segments:
                # Try to merge with previous segment if same speaker
                prev_segment = smoothed_segments[-1]
                if prev_segment["speaker"] == segment["speaker"]:
                    # Merge with previous
                    prev_segment["end"] = segment["end"]
                    prev_segment["text"] += " " + segment["text"]
                    prev_segment["words"].extend(segment["words"])
                    # Recalculate confidence
                    prev_segment["confidence"] = np.mean([
                        w["confidence"] for w in prev_segment["words"]
                    ])
                    continue

            # If we can't merge with previous, check if we can merge with next
            if i < len(segments) - 1:
                next_segment = segments[i + 1]
                if next_segment["speaker"] == segment["speaker"]:
                    # Mark for merging with next (will be handled in next iteration)
                    smoothed_segments.append(segment)
                    continue

            # If we can't merge, keep the segment as-is
            smoothed_segments.append(segment)
        else:
            smoothed_segments.append(segment)

    return smoothed_segments

def validate_alignment_result(
    enhanced_segments: List[Dict]
) -> Tuple[bool, Optional[str]]:
    """
    Validate enhanced alignment result for consistency.

    Args:
        enhanced_segments: Enhanced segments with word-level timing

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not enhanced_segments:
        return False, "No segments to validate"

    # Check for chronological order
    prev_end = -1
    for i, segment in enumerate(enhanced_segments):
        if segment["start"] < prev_end:
            return False, f"Segment {i} starts before previous segment ends"

        if segment["end"] < segment["start"]:
            return False, f"Segment {i} has end time before start time"

        # Validate word timestamps
        words = segment.get("words", [])
        prev_word_end = segment["start"]
        for j, word in enumerate(words):
            if word["start"] < prev_word_end:
                return False, f"Word {j} in segment {i} starts before previous word ends"

            if word["end"] < word["start"]:
                return False, f"Word {j} in segment {i} has end time before start time"

            prev_word_end = word["end"]

        prev_end = segment["end"]

    return True, None