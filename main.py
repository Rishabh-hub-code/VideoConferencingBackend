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
from api.websockets import ConnectionManager
from asr.whisper_asr import model as whisper_model
from translation.nllb_translation import warm_up_models
from utils.logger import setup_logging

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Global state for WebSocket connections
connection_manager = ConnectionManager()

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

# WebSocket endpoints will be added in api/websockets.py module
# This main file focuses on application setup and configuration

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