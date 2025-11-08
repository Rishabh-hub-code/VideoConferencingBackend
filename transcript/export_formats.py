"""
Transcript Export System
Exports transcripts in multiple formats: RTTM, JSON, SRT, VTT, Plain Text
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta
import os

logger = logging.getLogger(__name__)

def export_rttm(transcript_data: Dict[str, Any], output_path: str) -> bool:
    """
    Export transcript to RTTM (Rich Transcription Time Marked) format.
    Used for diarization evaluation.

    RTTM format:
    SPEAKER <filename> 1 <start> <duration> <speaker> <conf> <signal>

    Args:
        transcript_data: Enhanced transcript data with segments
        output_path: Output file path

    Returns:
        True if export successful, False otherwise
    """
    try:
        segments = transcript_data.get("segments", [])
        metadata = transcript_data.get("metadata", {})

        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                # Extract required fields
                speaker = segment.get("speaker", "UNKNOWN")
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                duration = end_time - start_time
                confidence = segment.get("confidence", 0.0)

                # Generate filename from metadata or use default
                filename = metadata.get("filename", "MEETING")

                # RTTM format: SPEAKER filename channel start duration speaker conf signal
                rttm_line = (
                    f"SPEAKER {filename} 1 {start_time:.3f} {duration:.3f} "
                    f"{speaker} {confidence:.3f} <NA>\n"
                )
                f.write(rttm_line)

        logger.info(f"RTTM export successful: {output_path} ({len(segments)} segments)")
        return True

    except Exception as e:
        logger.error(f"RTTM export failed: {e}")
        return False

def export_json(transcript_data: Dict[str, Any], output_path: str) -> bool:
    """
    Export transcript to JSON format with full metadata.

    Args:
        transcript_data: Enhanced transcript data
        output_path: Output file path

    Returns:
        True if export successful, False otherwise
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        logger.info(f"JSON export successful: {output_path}")
        return True

    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        return False

def export_srt(transcript_data: Dict[str, Any], output_path: str,
                include_translation: bool = False) -> bool:
    """
    Export transcript to SRT (SubRip) subtitle format.

    Args:
        transcript_data: Enhanced transcript data
        output_path: Output file path
        include_translation: Whether to include translations

    Returns:
        True if export successful, False otherwise
    """
    try:
        segments = transcript_data.get("segments", [])

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments, 1):
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "UNKNOWN")
                translation = segment.get("translation", "")

                # Format timestamps for SRT
                start_str = _seconds_to_srt_time(start_time)
                end_str = _seconds_to_srt_time(end_time)

                # Write subtitle block
                f.write(f"{i}\n")
                f.write(f"{start_str} --> {end_str}\n")

                # Add speaker label
                subtitle_text = f"[{speaker}] {text}"

                # Add translation if available and requested
                if include_translation and translation:
                    subtitle_text += f"\n{translation}"

                f.write(f"{subtitle_text}\n\n")

        logger.info(f"SRT export successful: {output_path} ({len(segments)} segments)")
        return True

    except Exception as e:
        logger.error(f"SRT export failed: {e}")
        return False

def export_vtt(transcript_data: Dict[str, Any], output_path: str,
               include_translation: bool = False) -> bool:
    """
    Export transcript to WebVTT (Web Video Text Tracks) format.

    Args:
        transcript_data: Enhanced transcript data
        output_path: Output file path
        include_translation: Whether to include translations

    Returns:
        True if export successful, False otherwise
    """
    try:
        segments = transcript_data.get("segments", [])

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write VTT header
            f.write("WEBVTT\n\n")

            for segment in segments:
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "UNKNOWN")
                translation = segment.get("translation", "")

                # Format timestamps for VTT
                start_str = _seconds_to_vtt_time(start_time)
                end_str = _seconds_to_vtt_time(end_time)

                # Write cue
                f.write(f"{start_str} --> {end_str}\n")

                # Add speaker label with VTT styling
                subtitle_text = f"<v {speaker}>{text}</v>"

                # Add translation if available and requested
                if include_translation and translation:
                    subtitle_text += f"\n{translation}"

                f.write(f"{subtitle_text}\n\n")

        logger.info(f"VTT export successful: {output_path} ({len(segments)} segments)")
        return True

    except Exception as e:
        logger.error(f"VTT export failed: {e}")
        return False

def export_text(transcript_data: Dict[str, Any], output_path: str,
               include_timestamps: bool = True, include_speakers: bool = True) -> bool:
    """
    Export transcript to plain text format.

    Args:
        transcript_data: Enhanced transcript data
        output_path: Output file path
        include_timestamps: Whether to include timestamps
        include_speakers: Whether to include speaker labels

    Returns:
        True if export successful, False otherwise
    """
    try:
        segments = transcript_data.get("segments", [])
        metadata = transcript_data.get("metadata", {})

        with open(output_path, 'w', encoding='utf-8') as f:
            # Write header information
            f.write("TRANSCRIPT\n")
            f.write("=" * 50 + "\n\n")

            if metadata:
                f.write(f"Total Duration: {metadata.get('total_duration', 0):.2f} seconds\n")
                f.write(f"Number of Speakers: {metadata.get('speaker_count', 0)}\n")
                f.write(f"Language: {metadata.get('language', 'Unknown')}\n")
                f.write(f"Segments: {metadata.get('segment_count', 0)}\n")
                f.write(f"Words: {metadata.get('word_count', 0)}\n\n")
                f.write("=" * 50 + "\n\n")

            # Write segments
            for i, segment in enumerate(segments, 1):
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                text = segment.get("text", "").strip()
                speaker = segment.get("speaker", "UNKNOWN")
                translation = segment.get("translation", "")

                # Write segment header
                line_parts = []
                if include_timestamps:
                    start_str = _seconds_to_readable_time(start_time)
                    end_str = _seconds_to_readable_time(end_time)
                    line_parts.append(f"[{start_str} - {end_str}]")

                if include_speakers:
                    line_parts.append(f"{speaker}:")

                header = " ".join(line_parts)

                # Write segment content
                if header:
                    f.write(f"{header}\n")
                f.write(f"{text}\n")

                # Add translation if available
                if translation:
                    f.write(f"Translation: {translation}\n")

                f.write("\n")

        logger.info(f"Text export successful: {output_path} ({len(segments)} segments)")
        return True

    except Exception as e:
        logger.error(f"Text export failed: {e}")
        return False

def export_csv(transcript_data: Dict[str, Any], output_path: str) -> bool:
    """
    Export transcript to CSV format for analysis.

    Args:
        transcript_data: Enhanced transcript data
        output_path: Output file path

    Returns:
        True if export successful, False otherwise
    """
    try:
        import csv

        segments = transcript_data.get("segments", [])

        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'Segment_ID', 'Speaker', 'Start_Time', 'End_Time', 'Duration',
                'Text', 'Translation', 'Confidence', 'Word_Count'
            ])

            # Write segments
            for i, segment in enumerate(segments, 1):
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                duration = end_time - start_time
                text = segment.get("text", "").strip()
                translation = segment.get("translation", "")
                confidence = segment.get("confidence", 0.0)
                word_count = len(text.split())

                writer.writerow([
                    i,
                    segment.get("speaker", "UNKNOWN"),
                    f"{start_time:.3f}",
                    f"{end_time:.3f}",
                    f"{duration:.3f}",
                    text,
                    translation,
                    f"{confidence:.3f}",
                    word_count
                ])

        logger.info(f"CSV export successful: {output_path} ({len(segments)} segments)")
        return True

    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return False

def export_all_formats(transcript_data: Dict[str, Any], output_dir: str,
                      base_filename: str = "transcript") -> Dict[str, bool]:
    """
    Export transcript to all supported formats.

    Args:
        transcript_data: Enhanced transcript data
        output_dir: Output directory path
        base_filename: Base filename without extension

    Returns:
        Dictionary mapping format names to export success status
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    # Export to each format
    formats = [
        ("json", export_json),
        ("rttm", export_rttm),
        ("srt", export_srt),
        ("vtt", export_vtt),
        ("txt", export_text),
        ("csv", export_csv)
    ]

    for format_name, export_func in formats:
        output_path = os.path.join(output_dir, f"{base_filename}.{format_name}")
        try:
            success = export_func(transcript_data, output_path)
            results[format_name] = success
        except Exception as e:
            logger.error(f"Failed to export {format_name}: {e}")
            results[format_name] = False

    return results

def _seconds_to_srt_time(seconds: float) -> str:
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{milliseconds:03d}"

def _seconds_to_vtt_time(seconds: float) -> str:
    """Convert seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = int((seconds - int(seconds)) * 1000)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{milliseconds:03d}"

def _seconds_to_readable_time(seconds: float) -> str:
    """Convert seconds to human-readable time format."""
    td = timedelta(seconds=seconds)
    hours, remainder = divmod(td.total_seconds(), 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
    elif minutes > 0:
        return f"{int(minutes)}m {int(seconds)}s"
    else:
        return f"{int(seconds)}s"

def validate_export_data(transcript_data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
    """
    Validate transcript data before export.

    Args:
        transcript_data: Transcript data to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(transcript_data, dict):
        return False, "Transcript data must be a dictionary"

    if "segments" not in transcript_data:
        return False, "Transcript data must contain 'segments' key"

    segments = transcript_data["segments"]
    if not isinstance(segments, list):
        return False, "Segments must be a list"

    if not segments:
        return False, "No segments to export"

    # Validate each segment
    for i, segment in enumerate(segments):
        if not isinstance(segment, dict):
            return False, f"Segment {i} must be a dictionary"

        required_fields = ["speaker", "start", "end", "text"]
        for field in required_fields:
            if field not in segment:
                return False, f"Segment {i} missing required field: {field}"

        # Validate timestamps
        if segment["start"] < 0:
            return False, f"Segment {i} has invalid start time"

        if segment["end"] < segment["start"]:
            return False, f"Segment {i} has end time before start time"

    return True, None