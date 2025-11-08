"""
PyAnnote Audio Diarization Module
Speaker diarization using pyannote.audio pipeline
"""

import logging
import tempfile
import os
from typing import List, Dict, Optional, Tuple
import torch

logger = logging.getLogger(__name__)

# Global singleton for diarization pipeline
_diarization_pipeline = None

def get_diarization_pipeline():
    """
    Get diarization pipeline using singleton pattern.
    Loads pipeline on first call and returns cached instance on subsequent calls.

    Returns:
        PyAnnote diarization pipeline or None if loading fails
    """
    global _diarization_pipeline

    if _diarization_pipeline is None:
        try:
            logger.info("Loading PyAnnote diarization pipeline...")
            from pyannote.audio import Pipeline

            # Load the diarization pipeline
            # Note: This requires accepting the user agreement on HuggingFace
            _diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=False  # Set to True and provide HF_TOKEN if needed
            )

            # Move to GPU if available
            if torch.cuda.is_available():
                _diarization_pipeline = _diarization_pipeline.to(torch.device("cuda"))
                logger.info("Diarization pipeline moved to GPU")
            else:
                logger.info("Diarization pipeline using CPU")

            logger.info("PyAnnote diarization pipeline loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load PyAnnote diarization pipeline: {e}")
            logger.warning("Diarization functionality will be unavailable")
            _diarization_pipeline = None

    return _diarization_pipeline

def warm_up_diarization():
    """
    Warm up diarization pipeline by loading it into memory.
    Call this during application startup.
    """
    logger.info("Warming up diarization pipeline...")
    pipeline = get_diarization_pipeline()
    if pipeline is not None:
        logger.info("Diarization pipeline warmed up successfully")
    else:
        logger.error("Failed to warm up diarization pipeline")

def diarize_audio_file(audio_bytes: bytes, file_extension: str = "wav") -> List[Dict]:
    """
    Perform speaker diarization on audio file.

    Args:
        audio_bytes: Audio file content as bytes
        file_extension: Audio file extension (default: wav)

    Returns:
        List of diarization segments with speaker labels and timestamps
        Format: [{"speaker": "SPEAKER_00", "start": 0.5, "end": 3.2}, ...]

    Raises:
        RuntimeError: If diarization pipeline is not available
        ValueError: If audio data is invalid
    """
    if not audio_bytes:
        raise ValueError("Audio data cannot be empty")

    # Get diarization pipeline
    pipeline = get_diarization_pipeline()
    if pipeline is None:
        raise RuntimeError("Diarization pipeline is not available")

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        logger.info(f"Processing diarization for audio file: {len(audio_bytes)} bytes")

        # Load audio file for pyannote
        import torchaudio

        try:
            # Load audio waveform
            waveform, sample_rate = torchaudio.load(tmp_path)

            # Ensure mono audio
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Resample to 16kHz if needed (pyannote works best with 16kHz)
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                sample_rate = 16000

            # Create audio dictionary for pyannote
            audio = {"waveform": waveform, "sample_rate": sample_rate}

        except Exception as e:
            logger.error(f"Failed to load audio file: {e}")
            raise ValueError(f"Invalid audio file: {str(e)}")

        # Perform diarization
        try:
            diarization_result = pipeline(audio)

            # Convert results to standard format
            segments = []
            for turn, _, speaker in diarization_result.itertracks(yield_label=True):
                segment = {
                    "speaker": str(speaker),
                    "start": float(turn.start),
                    "end": float(turn.end),
                    "duration": float(turn.end - turn.start)
                }
                segments.append(segment)

            logger.info(f"Diarization completed: {len(segments)} segments found")

            # Sort segments by start time
            segments.sort(key=lambda x: x["start"])

            return segments

        except Exception as e:
            logger.error(f"Diarization processing failed: {e}")
            raise RuntimeError(f"Diarization failed: {str(e)}")

    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def diarize_audio_file_path(audio_path: str) -> List[Dict]:
    """
    Perform speaker diarization on audio file given file path.

    Args:
        audio_path: Path to audio file

    Returns:
        List of diarization segments with speaker labels and timestamps
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")

    # Read file content
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    # Extract file extension
    file_extension = os.path.splitext(audio_path)[1][1:]  # Remove the dot

    return diarize_audio_file(audio_bytes, file_extension)

def merge_overlapping_segments(segments: List[Dict], min_gap: float = 0.5) -> List[Dict]:
    """
    Merge segments from the same speaker that are close to each other.

    Args:
        segments: List of diarization segments
        min_gap: Minimum gap in seconds to consider merging

    Returns:
        List of merged segments
    """
    if not segments:
        return []

    # Sort by start time
    segments = sorted(segments, key=lambda x: x["start"])

    merged = []
    current = segments[0].copy()

    for next_segment in segments[1:]:
        # Check if segments are from same speaker and close enough
        if (next_segment["speaker"] == current["speaker"] and
            next_segment["start"] - current["end"] <= min_gap):

            # Merge segments
            current["end"] = next_segment["end"]
            current["duration"] = current["end"] - current["start"]
        else:
            # Add current segment and start new one
            merged.append(current)
            current = next_segment.copy()

    # Add last segment
    merged.append(current)

    return merged

def get_speaker_statistics(segments: List[Dict]) -> Dict:
    """
    Calculate speaking statistics for each speaker.

    Args:
        segments: List of diarization segments

    Returns:
        Dictionary with speaker statistics
    """
    if not segments:
        return {}

    stats = {}
    total_duration = 0

    for segment in segments:
        speaker = segment["speaker"]
        duration = segment["duration"]

        if speaker not in stats:
            stats[speaker] = {
                "total_duration": 0,
                "segment_count": 0,
                "avg_segment_duration": 0
            }

        stats[speaker]["total_duration"] += duration
        stats[speaker]["segment_count"] += 1
        total_duration += duration

    # Calculate averages and percentages
    for speaker in stats:
        speaker_stats = stats[speaker]
        speaker_stats["avg_segment_duration"] = (
            speaker_stats["total_duration"] / speaker_stats["segment_count"]
        )
        speaker_stats["percentage"] = (speaker_stats["total_duration"] / total_duration) * 100

    return stats

def validate_diarization_result(segments: List[Dict]) -> Tuple[bool, Optional[str]]:
    """
    Validate diarization result for common issues.

    Args:
        segments: List of diarization segments

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not segments:
        return False, "No diarization segments found"

    # Check for overlapping segments
    for i in range(len(segments) - 1):
        current = segments[i]
        next_seg = segments[i + 1]

        if current["end"] > next_seg["start"]:
            return False, f"Overlapping segments detected: {current} and {next_seg}"

    # Check for reasonable durations
    for segment in segments:
        if segment["duration"] < 0.1:  # Less than 100ms
            return False, f"Segment too short: {segment['duration']:.3f}s"

        if segment["duration"] > 60:  # More than 1 minute
            logger.warning(f"Very long segment detected: {segment['duration']:.3f}s")

    return True, None