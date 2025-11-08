"""
Full Pipeline Processing: Diarization -> ASR -> Translation -> Alignment
"""

import logging
import os
import tempfile
from typing import List, Dict, Optional, Any

# Import application modules
from diarization.pyannote_diarization import diarize_audio_file
from asr.whisper_asr import transcribe_audio_file
from pipeline.segment_utils import standardize_audio
from translation.nllb_translation import translate_text
from alignment.stable_ts import transcribe_with_timestamps

logger = logging.getLogger(__name__)

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    logger.warning("pydub not available - some audio processing features may be limited")

def crop_audio_segment(audio_bytes: bytes, start_sec: float, end_sec: float, file_extension: str = "wav") -> tuple[bytes, str]:
    """
    Crop audio segment from audio bytes.

    Args:
        audio_bytes: Audio data as bytes
        start_sec: Start time in seconds
        end_sec: End time in seconds
        file_extension: Audio file format

    Returns:
        Tuple of (cropped_audio_bytes, temp_file_path)
    """
    if not PYDUB_AVAILABLE:
        raise RuntimeError("pydub is required for audio cropping")

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_in:
        tmp_in.write(audio_bytes)
        in_path = tmp_in.name

    try:
        sound = AudioSegment.from_file(in_path)
        # Convert to milliseconds for pydub
        start_ms = int(start_sec * 1000)
        end_ms = int(end_sec * 1000)
        cropped = sound[start_ms:end_ms]

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_out:
            cropped.export(tmp_out.name, format=file_extension)
            with open(tmp_out.name, 'rb') as f:
                cropped_bytes = f.read()
            cropped_path = tmp_out.name

        logger.debug(f"Cropped audio segment: {start_sec}s - {end_sec}s ({len(cropped_bytes)} bytes)")
        return cropped_bytes, cropped_path

    except Exception as e:
        # Cleanup input file on error
        if os.path.exists(in_path):
            os.remove(in_path)
        raise RuntimeError(f"Failed to crop audio segment: {e}")
    finally:
        # Always cleanup input file
        if os.path.exists(in_path):
            os.remove(in_path)

def process_diarize_asr_translate(audio_path: str, source_lang: str = "hindi", target_lang: str = "english") -> List[Dict[str, Any]]:
    """
    Process audio through complete pipeline: Diarization -> ASR -> Translation.

    Args:
        audio_path: Path to audio file
        source_lang: Source language for ASR
        target_lang: Target language for translation

    Returns:
        List of processed segments with diarization, ASR, and translation results
    """
    logger.info(f"Starting full pipeline processing: {audio_path}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read audio file
    with open(audio_path, 'rb') as f:
        audio_bytes = f.read()

    file_extension = os.path.splitext(audio_path)[1][1:]  # Remove the dot
    return process_diarize_asr_translate_bytes(audio_bytes, file_extension, source_lang, target_lang)

def process_diarize_asr_translate_bytes(audio_bytes: bytes, file_extension: str = "mp3",
                                      src_lang: str = "hindi", tgt_lang: str = "english") -> List[Dict[str, Any]]:
    """
    Process audio bytes through complete pipeline: Diarization -> ASR -> Translation.

    Args:
        audio_bytes: Audio data as bytes
        file_extension: Audio file format
        src_lang: Source language for ASR
        tgt_lang: Target language for translation

    Returns:
        List of processed segments with diarization, ASR, and translation results
    """
    logger.info(f"Starting full pipeline processing: {len(audio_bytes)} bytes, {file_extension} format")

    try:
        # Step 1: Standardize audio to wav/mono/16k
        std_path = standardize_audio(audio_bytes, file_extension, "wav", 16000)
        with open(std_path, "rb") as af:
            std_audio_bytes = af.read()

        # Step 2: Diarization
        logger.info("Running speaker diarization...")
        segments = diarize_audio_file(std_audio_bytes, file_extension="wav")
        logger.info(f"Diarization completed: {len(segments)} segments found")

        if not segments:
            logger.warning("No diarization segments found - returning empty result")
            return []

        full_results = []

        # Step 3: For each segment, crop audio, run Whisper ASR, translate
        for i, seg in enumerate(segments):
            logger.debug(f"Processing segment {i+1}/{len(segments)}: {seg}")

            try:
                # Skip very short segments (less than 0.5 seconds)
                if seg["duration"] < 0.5:
                    logger.debug(f"Skipping short segment: {seg['duration']:.2f}s")
                    continue

                # Crop audio segment
                cropped_bytes, cropped_path = crop_audio_segment(
                    std_audio_bytes, seg["start"], seg["end"], "wav"
                )

                # Run ASR on cropped segment
                transcript = transcribe_audio_file(cropped_bytes, "wav")
                transcript = transcript.strip()

                # Clean up cropped audio file
                if os.path.exists(cropped_path):
                    os.remove(cropped_path)

                if not transcript:
                    logger.debug(f"Empty transcription for segment {i+1}")
                    continue

                # Translate transcript
                translation = translate_text(transcript, src_lang, tgt_lang)
                translation = translation.strip()

                # Add result
                result = {
                    "segment_id": i + 1,
                    "speaker": seg["speaker"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "duration": seg["duration"],
                    "transcript": transcript,
                    "translation": translation,
                    "confidence": None  # TODO: Extract from Whisper if available
                }
                full_results.append(result)

                logger.debug(f"Segment {i+1} processed: '{transcript}' -> '{translation}'")

            except Exception as e:
                logger.error(f"Failed to process segment {i+1}: {e}")
                # Continue with other segments
                continue

        # Clean up standardized audio file
        if os.path.exists(std_path):
            os.remove(std_path)

        logger.info(f"Pipeline completed: {len(full_results)} segments processed")
        return full_results

    except Exception as e:
        logger.error(f"Pipeline processing failed: {e}")
        # Cleanup any temporary files
        if 'std_path' in locals() and os.path.exists(std_path):
            os.remove(std_path)
        if 'cropped_path' in locals() and os.path.exists(cropped_path):
            os.remove(cropped_path)
        raise RuntimeError(f"Pipeline processing failed: {e}")

def process_with_timestamp_alignment(audio_path: str, source_lang: str = "hindi",
                                   target_lang: str = "english") -> Dict[str, Any]:
    """
    Process audio with enhanced timestamp alignment.

    Args:
        audio_path: Path to audio file
        source_lang: Source language
        target_lang: Target language

    Returns:
        Enhanced result with word-level timestamps
    """
    logger.info(f"Processing with timestamp alignment: {audio_path}")

    # Run standard pipeline first
    segments = process_diarize_asr_translate(audio_path, source_lang, target_lang)

    # TODO: Integrate stable-TS for word-level timestamps
    # This would require more complex integration with the alignment module

    return {
        "segments": segments,
        "metadata": {
            "total_duration": max([s["end"] for s in segments]) if segments else 0,
            "speaker_count": len(set([s["speaker"] for s in segments])),
            "word_count": sum([len(s["transcript"].split()) for s in segments])
        }
    }
