"""
NLLB Translation Module with Singleton Model Loading
Optimized for performance by loading models only once
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Global singleton variables for model and tokenizer
_model = None
_tokenizer = None

# Full Indian language code map
INDIAN_LANG_CODES = {
    "hindi": "hin_Deva",
    "marathi": "mar_Deva",
    "gujarati": "guj_Gujr",
    "bengali": "ben_Beng",
    "punjabi": "pan_Guru",
    "tamil": "tam_Taml",
    "telugu": "tel_Telu",
    "malayalam": "mal_Mlym",
    "kannada": "kan_Knda",
    "oriya": "ory_Orya",
    "assamese": "asm_Beng",
    "urdu": "urd_Arab",
    "sanskrit": "san_Deva",
    "english": "eng_Latn"
}

def get_nllb_model() -> Tuple[Optional[object], Optional[object]]:
    """
    Get NLLB model and tokenizer using singleton pattern.
    Loads models on first call and returns cached instance on subsequent calls.

    Returns:
        Tuple of (model, tokenizer) or (None, None) if loading fails
    """
    global _model, _tokenizer

    if _model is None or _tokenizer is None:
        try:
            logger.info("Loading NLLB model and tokenizer (first time)...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

            _tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
            _model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")

            logger.info("NLLB model and tokenizer loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load NLLB model: {e}")
            _model = None
            _tokenizer = None

    return _model, _tokenizer

def warm_up_models():
    """
    Warm up NLLB models by loading them into memory.
    Call this during application startup.
    """
    logger.info("Warming up NLLB translation models...")
    model, tokenizer = get_nllb_model()
    if model is not None and tokenizer is not None:
        logger.info("NLLB models warmed up successfully")
    else:
        logger.error("Failed to warm up NLLB models")

def translate_text(text: str, src_lang: str = "hindi", tgt_lang: str = "english") -> str:
    """
    Translate text using NLLB-200 model.

    Args:
        text: Text to translate
        src_lang: Source language name (e.g., "hindi", "english")
        tgt_lang: Target language name (e.g., "english", "hindi")

    Returns:
        Translated text string

    Raises:
        ValueError: If text is empty or languages are not supported
        RuntimeError: If model loading fails
    """
    if not text or not text.strip():
        raise ValueError("Text to translate cannot be empty")

    # Get model and tokenizer (singleton pattern)
    model, tokenizer = get_nllb_model()

    if model is None or tokenizer is None:
        raise RuntimeError("NLLB model is not available. Check model loading.")

    try:
        # Map language names to NLLB language codes
        src_code = INDIAN_LANG_CODES.get(src_lang.lower(), "hin_Deva")
        tgt_code = INDIAN_LANG_CODES.get(tgt_lang.lower(), "eng_Latn")

        # Set source language for tokenizer
        tokenizer.src_lang = src_code

        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt")

        # Map target language code to BOS token ID
        def get_bos_token_id(tokenizer, lang_code):
            tokens = tokenizer.additional_special_tokens
            ids = tokenizer.additional_special_tokens_ids
            if lang_code not in tokens:
                lang_code = "eng_Latn"  # Fallback to English
            idx = tokens.index(lang_code)
            return ids[idx]

        bos_token_id = get_bos_token_id(tokenizer, tgt_code)

        # Generate translation
        output_tokens = model.generate(
            **inputs,
            forced_bos_token_id=bos_token_id,
            max_length=200  # Reasonable limit for most translations
        )

        # Decode and return translation
        translated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

        logger.debug(f"Translation: {src_lang}({src_code}) -> {tgt_lang}({tgt_code})")
        logger.debug(f"Original: {text[:100]}...")
        logger.debug(f"Translated: {translated_text[:100]}...")

        return translated_text

    except Exception as e:
        logger.error(f"Translation error: {e}")
        raise RuntimeError(f"Translation failed: {str(e)}")

def get_supported_languages() -> dict:
    """
    Get list of supported languages with their codes.

    Returns:
        Dictionary mapping language names to NLLB language codes
    """
    return INDIAN_LANG_CODES.copy()

def is_language_supported(language: str) -> bool:
    """
    Check if a language is supported for translation.

    Args:
        language: Language name to check

    Returns:
        True if language is supported, False otherwise
    """
    return language.lower() in INDIAN_LANG_CODES
