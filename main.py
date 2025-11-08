"""
FastAPI Main Application Entry Point
WebRTC Video Conferencing with Real-time ASR and Translation
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.websockets import WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
import logging
import os
from typing import Dict, List
import json
import asyncio
from collections import defaultdict

# Import application modules
from api.routes import router as api_router
from api.websockets import connection_manager
from asr.whisper_asr import model as whisper_model
from translation.nllb_translation import warm_up_models
from utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - handles startup and shutdown."""
    logger.info("Starting WebRTC Video Conferencing Backend...")

    # Warm up ML models at startup
    try:
        logger.info("Loading ML models...")
        warm_up_models()
        logger.info("NLLB translation model loaded successfully")

        # Whisper model is already loaded at import time
        logger.info("Whisper ASR model loaded successfully")

        logger.info("All models loaded successfully - server ready")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    yield

    logger.info("Shutting down server...")

# Create FastAPI application
app = FastAPI(
    title="WebRTC Video Conferencing API",
    description="Real-time video conferencing with ASR, translation, and speaker diarization",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions with consistent error format."""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code
    }

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return {
        "success": False,
        "error": "Internal server error",
        "status_code": 500
    }

@app.get("/")
async def root():
    """Root endpoint - API information."""
    return {
        "message": "WebRTC Video Conferencing API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "api_docs": "/docs",
            "asr": "/api/asr",
            "translation": "/api/translate",
            "pipeline": "/api/pipeline/full",
            "signaling": "/ws/signaling/{room_id}",
            "captions": "/ws/captions/{room_id}"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "models_loaded": {
            "whisper": whisper_model is not None,
        }
    }

# WebSocket endpoints for real-time communication
@app.websocket("/ws/signaling/{room_id}")
async def websocket_signaling(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for WebRTC signaling in a specific room."""
    # Extract peer_id from query parameters
    peer_id = websocket.query_params.get("peer_id")
    if not peer_id:
        await websocket.close(code=4000, reason="peer_id is required")
        return

    try:
        await connection_manager.connect_signaling(websocket, room_id, peer_id)

        # Handle incoming messages
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)

                logger.debug(f"Signaling message from {peer_id} in room {room_id}: {message.get('type')}")

                # Relay message to appropriate target
                if message.get("to"):
                    # Send to specific peer
                    target_peer = message["to"]
                    await connection_manager.broadcast_signaling(room_id, {
                        **message,
                        "from": peer_id
                    })
                else:
                    # Broadcast to all peers in room (except sender)
                    await connection_manager.broadcast_signaling(room_id, {
                        **message,
                        "from": peer_id
                    }, exclude_peer=peer_id)

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling signaling message: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket signaling error: {e}")
    finally:
        await connection_manager.disconnect_signaling(websocket, room_id, peer_id)

@app.websocket("/ws/captions/{room_id}")
async def websocket_captions(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for real-time caption streaming in a specific room."""
    try:
        await connection_manager.connect_captions(websocket, room_id)

        # Handle incoming messages (if needed for future features)
        while True:
            try:
                # Keep connection alive with ping/pong
                data = await websocket.receive_text()

                # For now, we don't expect incoming messages on caption WebSocket
                # But we handle them gracefully if they arrive
                if data == "ping":
                    await websocket.send_text("pong")

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error handling caption connection: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket caption error: {e}")
    finally:
        await connection_manager.disconnect_captions(websocket, room_id)

@app.websocket("/ws/audio/{room_id}")
async def websocket_audio_stream(websocket: WebSocket, room_id: str):
    """WebSocket endpoint for real-time audio streaming for ASR processing."""
    try:
        await websocket.accept()
        logger.info(f"Audio streaming client connected to room {room_id}")

        # Buffer for audio chunks
        audio_buffer = bytearray()
        chunk_count = 0

        while True:
            try:
                # Receive audio chunk
                data = await websocket.receive_bytes()
                audio_buffer.extend(data)
                chunk_count += 1

                logger.debug(f"Received audio chunk {chunk_count} in room {room_id}: {len(data)} bytes")

                # Process audio when buffer reaches certain size (e.g., 3 seconds of audio)
                if len(audio_buffer) >= 16000 * 2 * 3:  # 16kHz * 2 bytes * 3 seconds
                    try:
                        # TODO: Process audio buffer through ASR pipeline
                        # This is where real-time processing would happen
                        logger.info(f"Processing audio buffer: {len(audio_buffer)} bytes")

                        # For now, just send a placeholder response
                        # In production, this would trigger the ASR -> translation pipeline
                        await connection_manager.broadcast_captions(room_id, {
                            "speaker": "SPEAKER_00",
                            "text": f"Audio chunk {chunk_count} processed",
                            "translation": f"Audio chunk {chunk_count} processed",
                            "confidence": 0.95
                        })

                        # Clear buffer after processing
                        audio_buffer.clear()

                    except Exception as e:
                        logger.error(f"Error processing audio buffer: {e}")
                        audio_buffer.clear()

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Error receiving audio chunk: {e}")
                break

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket audio streaming error: {e}")
    finally:
        logger.info(f"Audio streaming client disconnected from room {room_id}")

# Room management endpoints
@app.get("/api/rooms/{room_id}/status")
async def get_room_status(room_id: str):
    """Get status information for a specific room."""
    status = connection_manager.get_room_status(room_id)
    return {
        "success": True,
        "data": status
    }

if __name__ == "__main__":
    import uvicorn

    # Read configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("DEBUG", "false").lower() == "true"

    logger.info(f"Starting server on {host}:{port}")

    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        log_level="info"
    )