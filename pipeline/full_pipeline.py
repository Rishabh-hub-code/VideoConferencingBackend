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

def crop_audio_segment(audio_bytes, start_sec, end_sec, file_extension="wav"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_in:
        tmp_in.write(audio_bytes)
        in_path = tmp_in.name

    sound = AudioSegment.from_file(in_path)
    cropped = sound[start_sec * 1000 : end_sec * 1000]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_out:
        cropped.export(tmp_out.name, format=file_extension)
        cropped_bytes = tmp_out.read()
        cropped_path = tmp_out.name
    os.remove(in_path)
    return cropped_bytes, cropped_path

def process_diarize_asr_translate(audio_bytes, file_extension="mp3", src_lang="hindi", tgt_lang="english"):
    # Step 1: Standardize audio to wav/mono/16k
    std_path = standardize_audio(audio_bytes, file_extension, "wav", 16000)
    with open(std_path, "rb") as af:
        std_audio_bytes = af.read()

    # Step 2: Diarization
    segments = diarize_audio_file(std_audio_bytes, file_extension="wav")
    full_results = []

    # Step 3: For each segment, crop audio, run Whisper ASR, translate
    for seg in segments:
        cropped_bytes, cropped_path = crop_audio_segment(std_audio_bytes, seg["start"], seg["end"], "wav")
        transcript = transcribe_audio_file(cropped_bytes, "wav")
        translation = translate_text(transcript, src_lang, tgt_lang)
        full_results.append({
            "speaker": seg["speaker"],
            "start": seg["start"],
            "end": seg["end"],
            "transcript": transcript,
            "translation": translation
        })
        os.remove(cropped_path)

    os.remove(std_path)
    return full_results
