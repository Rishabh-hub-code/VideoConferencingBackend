"""
Real-time Audio Processing Pipeline
Handles streaming audio chunks through ASR -> Diarization -> Translation -> Alignment
"""

import asyncio
import logging
import io
import wave
import struct
from typing import Dict, List, Optional, Callable, Any
from collections import deque
import threading
import time

logger = logging.getLogger(__name__)

class RealtimeAudioProcessor:
    """
    Real-time audio processor for streaming ASR and translation.
    Uses sliding window approach with configurable buffer sizes.
    """

    def __init__(self,
                 sample_rate: int = 16000,
                 channels: int = 1,
                 sample_width: int = 2,
                 window_duration: float = 3.0,
                 overlap_duration: float = 0.5,
                 on_caption_callback: Optional[Callable] = None):
        """
        Initialize the real-time audio processor.

        Args:
            sample_rate: Audio sample rate (Hz)
            channels: Number of audio channels
            sample_width: Sample width in bytes
            window_duration: Duration of processing window in seconds
            overlap_duration: Overlap between windows in seconds
            on_caption_callback: Callback function for caption results
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.window_duration = window_duration
        self.overlap_duration = overlap_duration

        # Calculate buffer sizes
        self.window_size = int(sample_rate * window_duration)
        self.overlap_size = int(sample_rate * overlap_duration)
        self.chunk_size = 4096  # Default chunk size for audio processing

        # Audio buffer
        self.audio_buffer = deque(maxlen=self.window_size + self.overlap_size)

        # Processing state
        self.is_processing = False
        self.processing_lock = threading.Lock()

        # Callback for caption results
        self.on_caption_callback = on_caption_callback

        # Statistics
        self.chunks_processed = 0
        self.last_processing_time = 0
        self.total_processing_time = 0

        logger.info(f"RealtimeAudioProcessor initialized: {sample_rate}Hz, {window_duration}s windows")

    def add_audio_chunk(self, audio_data: bytes):
        """
        Add audio chunk to the processing buffer.

        Args:
            audio_data: Raw audio bytes
        """
        # Convert bytes to samples
        samples = self._bytes_to_samples(audio_data)

        # Add samples to buffer
        for sample in samples:
            self.audio_buffer.append(sample)

        self.chunks_processed += 1

        # Check if we should trigger processing
        if len(self.audio_buffer) >= self.window_size:
            if not self.is_processing:
                # Start processing in background thread
                threading.Thread(
                    target=self._process_window,
                    daemon=True
                ).start()

    def _bytes_to_samples(self, audio_data: bytes) -> List[int]:
        """
        Convert audio bytes to integer samples.

        Args:
            audio_data: Raw audio bytes

        Returns:
            List of integer samples
        """
        if self.sample_width == 2:
            # 16-bit PCM
            samples = list(struct.unpack(f'<{len(audio_data)//2}h', audio_data))
        elif self.sample_width == 1:
            # 8-bit PCM
            samples = list(struct.unpack(f'{len(audio_data)}B', audio_data))
            # Convert 0-255 to -128 to 127
            samples = [s - 128 for s in samples]
        else:
            raise ValueError(f"Unsupported sample width: {self.sample_width}")

        return samples

    def _samples_to_bytes(self, samples: List[int]) -> bytes:
        """
        Convert integer samples to audio bytes.

        Args:
            samples: List of integer samples

        Returns:
            Raw audio bytes
        """
        if self.sample_width == 2:
            # 16-bit PCM
            return struct.pack(f'<{len(samples)}h', *samples)
        elif self.sample_width == 1:
            # 8-bit PCM
            # Convert -128 to 127 to 0-255
            samples = [s + 128 for s in samples]
            return struct.pack(f'{len(samples)}B', *samples)
        else:
            raise ValueError(f"Unsupported sample width: {self.sample_width}")

    def _samples_to_wav_bytes(self, samples: List[int]) -> bytes:
        """
        Convert samples to WAV format bytes.

        Args:
            samples: List of integer samples

        Returns:
            WAV format bytes
        """
        # Create in-memory WAV file
        wav_buffer = io.BytesIO()

        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(self.channels)
            wav_file.setsampwidth(self.sample_width)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(self._samples_to_bytes(samples))

        wav_buffer.seek(0)
        return wav_buffer.read()

    def _process_window(self):
        """
        Process current audio window through ASR pipeline.
        Runs in background thread.
        """
        if not self.processing_lock.acquire(blocking=False):
            return  # Already processing

        try:
            self.is_processing = True
            start_time = time.time()

            # Get current window
            if len(self.audio_buffer) < self.window_size:
                return

            # Extract window samples
            window_samples = list(self.audio_buffer)[:self.window_size]

            # Convert to WAV bytes for processing
            wav_bytes = self._samples_to_wav_bytes(window_samples)

            # Process through ASR pipeline
            try:
                # Import here to avoid circular imports
                from asr.whisper_asr import transcribe_audio_file
                from translation.nllb_translation import translate_text

                # Run ASR
                transcription = transcribe_audio_file(wav_bytes, "wav")
                transcription = transcription.strip()

                if transcription:
                    logger.debug(f"ASR result: {transcription}")

                    # TODO: Add diarization for multi-speaker scenarios
                    # For now, assume single speaker
                    speaker_id = "SPEAKER_00"

                    # TODO: Make language configurable
                    source_lang = "english"  # Default to English for now
                    target_lang = "hindi"

                    # Translate if needed
                    try:
                        translation = translate_text(transcription, source_lang, target_lang)
                        translation = translation.strip()
                    except Exception as e:
                        logger.warning(f"Translation failed: {e}")
                        translation = transcription  # Fallback to original

                    # Create caption result
                    caption_result = {
                        "speaker": speaker_id,
                        "text": transcription,
                        "translation": translation,
                        "timestamp": time.time(),
                        "confidence": None,  # TODO: Extract from Whisper
                        "processing_time": time.time() - start_time
                    }

                    # Call callback if provided
                    if self.on_caption_callback:
                        try:
                            self.on_caption_callback(caption_result)
                        except Exception as e:
                            logger.error(f"Caption callback error: {e}")

                    # Update statistics
                    self.total_processing_time += time.time() - start_time
                    self.last_processing_time = time.time()

                    logger.info(f"Processed {len(window_samples)} samples in {time.time() - start_time:.3f}s")

            except Exception as e:
                logger.error(f"ASR processing error: {e}")

        except Exception as e:
            logger.error(f"Window processing error: {e}")
        finally:
            self.is_processing = False
            self.processing_lock.release()

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get processing statistics.

        Returns:
            Dictionary with processing statistics
        """
        avg_processing_time = (
            self.total_processing_time / max(1, self.chunks_processed)
        )

        return {
            "chunks_processed": self.chunks_processed,
            "buffer_size": len(self.audio_buffer),
            "window_size": self.window_size,
            "is_processing": self.is_processing,
            "last_processing_time": self.last_processing_time,
            "average_processing_time": avg_processing_time,
            "total_processing_time": self.total_processing_time
        }

    def reset(self):
        """Reset the processor state."""
        self.audio_buffer.clear()
        self.chunks_processed = 0
        self.total_processing_time = 0
        self.last_processing_time = 0
        logger.info("RealtimeAudioProcessor reset")

class RoomAudioProcessor:
    """
    Manages multiple audio processors for different rooms.
    Handles room-based audio processing and caption broadcasting.
    """

    def __init__(self):
        self.processors: Dict[str, RealtimeAudioProcessor] = {}
        self.room_settings: Dict[str, Dict] = {}
        self.connection_manager = None

    def set_connection_manager(self, connection_manager):
        """Set the connection manager for broadcasting captions."""
        self.connection_manager = connection_manager

    def create_processor(self, room_id: str, **kwargs) -> RealtimeAudioProcessor:
        """
        Create a new audio processor for a room.

        Args:
            room_id: Room identifier
            **kwargs: Additional processor configuration

        Returns:
            RealtimeAudioProcessor instance
        """
        if room_id in self.processors:
            logger.warning(f"Processor already exists for room {room_id}")
            return self.processors[room_id]

        # Create caption callback for this room
        def room_caption_callback(caption_result):
            self._handle_caption_result(room_id, caption_result)

        # Create processor with room-specific callback
        processor = RealtimeAudioProcessor(
            on_caption_callback=room_caption_callback,
            **kwargs
        )

        self.processors[room_id] = processor
        self.room_settings[room_id] = kwargs

        logger.info(f"Created audio processor for room {room_id}")
        return processor

    def get_processor(self, room_id: str) -> Optional[RealtimeAudioProcessor]:
        """Get existing processor for a room."""
        return self.processors.get(room_id)

    def remove_processor(self, room_id: str):
        """Remove processor for a room."""
        if room_id in self.processors:
            processor = self.processors[room_id]
            processor.reset()
            del self.processors[room_id]
            del self.room_settings[room_id]
            logger.info(f"Removed audio processor for room {room_id}")

    def process_audio_chunk(self, room_id: str, audio_data: bytes):
        """
        Process audio chunk for a specific room.

        Args:
            room_id: Room identifier
            audio_data: Audio chunk bytes
        """
        processor = self.get_processor(room_id)
        if processor:
            processor.add_audio_chunk(audio_data)
        else:
            logger.warning(f"No processor found for room {room_id}")

    def _handle_caption_result(self, room_id: str, caption_result: Dict):
        """
        Handle caption result from a room processor.

        Args:
            room_id: Room identifier
            caption_result: Caption processing result
        """
        if self.connection_manager:
            # Broadcast caption to all clients in the room
            asyncio.create_task(
                self.connection_manager.broadcast_captions(room_id, caption_result)
            )

    def get_room_statistics(self, room_id: str) -> Optional[Dict]:
        """Get processing statistics for a room."""
        processor = self.get_processor(room_id)
        if processor:
            return processor.get_statistics()
        return None

    def get_all_statistics(self) -> Dict[str, Dict]:
        """Get statistics for all active rooms."""
        return {
            room_id: processor.get_statistics()
            for room_id, processor in self.processors.items()
        }

# Global room processor instance
room_processor = RoomAudioProcessor()