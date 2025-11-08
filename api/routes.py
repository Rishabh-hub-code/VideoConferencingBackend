"""
API Routes for WebRTC Video Conferencing Backend
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from typing import Optional, Dict, Any
import logging
import tempfile
import os

# Import application modules
from asr.whisper_asr import transcribe_audio_file
from translation.nllb_translation import translate_text
from pipeline.full_pipeline import process_diarize_asr_translate

logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

@router.post("/asr")
async def transcribe_audio(
    audio_file: UploadFile = File(...),
    file_extension: Optional[str] = Form("wav")
):
    """
    Transcribe audio file using Whisper ASR.

    Args:
        audio_file: Audio file to transcribe
        file_extension: Audio file extension (default: wav)

    Returns:
        JSON response with transcribed text
    """
    try:
        logger.info(f"Processing ASR request for file: {audio_file.filename}")

        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )

        # Read file content
        audio_bytes = await audio_file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty."
            )

        # Transcribe audio
        transcribed_text = transcribe_audio_file(audio_bytes, file_extension)

        logger.info(f"ASR transcription completed: {len(transcribed_text)} characters")

        return {
            "success": True,
            "data": {
                "text": transcribed_text,
                "filename": audio_file.filename,
                "file_size": len(audio_bytes)
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ASR processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process audio file for transcription."
        )

@router.post("/translate")
async def translate_text_endpoint(
    text: str = Form(...),
    source_language: str = Form("hindi"),
    target_language: str = Form("english")
):
    """
    Translate text using NLLB model.

    Args:
        text: Text to translate
        source_language: Source language code
        target_language: Target language code

    Returns:
        JSON response with translated text
    """
    try:
        logger.info(f"Processing translation request: {source_language} -> {target_language}")

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text to translate cannot be empty."
            )

        # Translate text
        translated_text = translate_text(text, source_language, target_language)

        logger.info(f"Translation completed: {len(translated_text)} characters")

        return {
            "success": True,
            "data": {
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": target_language
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Translation processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to translate text."
        )

@router.post("/pipeline/full")
async def full_pipeline_endpoint(
    audio_file: UploadFile = File(...),
    source_language: str = Form("hindi"),
    target_language: str = Form("english"),
    file_extension: Optional[str] = Form("wav")
):
    """
    Process audio through complete pipeline: ASR -> Diarization -> Translation -> Alignment.

    Args:
        audio_file: Audio file to process
        source_language: Source language for ASR
        target_language: Target language for translation
        file_extension: Audio file extension

    Returns:
        JSON response with complete processed results
    """
    try:
        logger.info(f"Processing full pipeline for file: {audio_file.filename}")

        # Validate file type
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Please upload an audio file."
            )

        # Read file content
        audio_bytes = await audio_file.read()

        if len(audio_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Audio file is empty."
            )

        # Save to temporary file for pipeline processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_extension}') as tmp:
            tmp.write(audio_bytes)
            tmp_filepath = tmp.name

        try:
            # Process through full pipeline
            result = process_diarize_asr_translate(
                audio_path=tmp_filepath,
                source_lang=source_language,
                target_lang=target_language
            )

            logger.info(f"Full pipeline processing completed for {audio_file.filename}")

            return {
                "success": True,
                "data": {
                    "filename": audio_file.filename,
                    "file_size": len(audio_bytes),
                    "pipeline_result": result
                }
            }

        finally:
            # Cleanup temporary file
            if os.path.exists(tmp_filepath):
                os.remove(tmp_filepath)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full pipeline processing error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to process audio through full pipeline."
        )

@router.get("/pipeline/status")
async def pipeline_status():
    """
    Get the status of all pipeline components.

    Returns:
        JSON response with component status information
    """
    try:
        # Check if models are loaded by importing them
        from asr.whisper_asr import model as whisper_model
        from translation.nllb_translation import get_nllb_model

        # Check Whisper model
        whisper_loaded = whisper_model is not None

        # Check NLLB model (this will trigger lazy loading if not already loaded)
        try:
            nllb_model, nllb_tokenizer = get_nllb_model()
            nllb_loaded = nllb_model is not None and nllb_tokenizer is not None
        except Exception:
            nllb_loaded = False

        # Check diarization module
        try:
            from diarization.pyannote_diarization import diarize_audio_file
            diarization_available = True
        except ImportError:
            diarization_available = False

        return {
            "success": True,
            "data": {
                "components": {
                    "whisper_asr": {
                        "status": "loaded" if whisper_loaded else "not_loaded",
                        "available": whisper_loaded
                    },
                    "nllb_translation": {
                        "status": "loaded" if nllb_loaded else "not_loaded",
                        "available": nllb_loaded
                    },
                    "diarization": {
                        "status": "available" if diarization_available else "not_available",
                        "available": diarization_available
                    },
                    "stable_alignment": {
                        "status": "available",
                        "available": True  # Always available as it uses Whisper
                    }
                },
                "overall_status": "ready" if all([whisper_loaded, nllb_loaded]) else "loading"
            }
        }

    except Exception as e:
        logger.error(f"Pipeline status check error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to check pipeline status."
        )