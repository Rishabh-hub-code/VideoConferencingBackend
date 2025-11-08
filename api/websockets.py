"""
WebSocket endpoints for WebRTC signaling and real-time caption streaming
"""

from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List, Set
import json
import logging
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages WebSocket connections for real-time communication."""

    def __init__(self):
        # Room-based connection management
        self.signaling_connections: Dict[str, List[WebSocket]] = defaultdict(list)
        self.caption_connections: Dict[str, List[WebSocket]] = defaultdict(list)

        # Peer tracking for signaling
        self.room_peers: Dict[str, Set[str]] = defaultdict(set)

        # Active caption data per room
        self.room_captions: Dict[str, List[Dict]] = defaultdict(list)

    async def connect_signaling(self, websocket: WebSocket, room_id: str, peer_id: str):
        """Connect a peer to signaling WebSocket for a specific room."""
        await websocket.accept()

        # Store connection
        self.signaling_connections[room_id].append({
            'websocket': websocket,
            'peer_id': peer_id
        })

        # Add peer to room
        self.room_peers[room_id].add(peer_id)

        logger.info(f"Peer {peer_id} connected to signaling room {room_id}")

        # Notify other peers in the room
        await self.broadcast_signaling(room_id, {
            "type": "peer-joined",
            "peer_id": peer_id,
            "room_peers": list(self.room_peers[room_id])
        }, exclude_peer=peer_id)

        # Send current room state to new peer
        await websocket.send_text(json.dumps({
            "type": "room-state",
            "peer_id": peer_id,
            "room_peers": list(self.room_peers[room_id]),
            "peer_count": len(self.room_peers[room_id])
        }))

    async def disconnect_signaling(self, websocket: WebSocket, room_id: str, peer_id: str):
        """Disconnect a peer from signaling WebSocket."""
        # Remove connection
        self.signaling_connections[room_id] = [
            conn for conn in self.signaling_connections[room_id]
            if conn['websocket'] != websocket
        ]

        # Remove peer from room
        self.room_peers[room_id].discard(peer_id)

        logger.info(f"Peer {peer_id} disconnected from signaling room {room_id}")

        # Notify other peers
        await self.broadcast_signaling(room_id, {
            "type": "peer-left",
            "peer_id": peer_id,
            "room_peers": list(self.room_peers[room_id])
        })

    async def connect_captions(self, websocket: WebSocket, room_id: str):
        """Connect a client to caption WebSocket for a specific room."""
        await websocket.accept()
        self.caption_connections[room_id].append(websocket)

        logger.info(f"Client connected to captions room {room_id}")

        # Send recent caption history (last 10 captions)
        recent_captions = self.room_captions[room_id][-10:]
        if recent_captions:
            await websocket.send_text(json.dumps({
                "type": "caption-history",
                "captions": recent_captions
            }))

    async def disconnect_captions(self, websocket: WebSocket, room_id: str):
        """Disconnect a client from caption WebSocket."""
        if websocket in self.caption_connections[room_id]:
            self.caption_connections[room_id].remove(websocket)
            logger.info(f"Client disconnected from captions room {room_id}")

    async def broadcast_signaling(self, room_id: str, message: Dict, exclude_peer: str = None):
        """Broadcast signaling message to all peers in a room."""
        if room_id not in self.signaling_connections:
            return

        message_str = json.dumps(message)
        disconnected_peers = []

        for connection in self.signaling_connections[room_id]:
            peer_id = connection['peer_id']
            websocket = connection['websocket']

            # Skip excluded peer
            if exclude_peer and peer_id == exclude_peer:
                continue

            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send message to peer {peer_id}: {e}")
                disconnected_peers.append(connection)

        # Clean up disconnected peers
        for connection in disconnected_peers:
            self.signaling_connections[room_id].remove(connection)
            self.room_peers[room_id].discard(connection['peer_id'])

    async def broadcast_captions(self, room_id: str, caption_data: Dict):
        """Broadcast caption message to all clients in a room."""
        if room_id not in self.caption_connections:
            return

        # Store caption in room history
        caption_message = {
            "type": "caption",
            "timestamp": asyncio.get_event_loop().time(),
            **caption_data
        }
        self.room_captions[room_id].append(caption_message)

        # Keep only last 100 captions in memory
        if len(self.room_captions[room_id]) > 100:
            self.room_captions[room_id] = self.room_captions[room_id][-100:]

        # Broadcast to all connected caption clients
        message_str = json.dumps(caption_message)
        disconnected_clients = []

        for websocket in self.caption_connections[room_id]:
            try:
                await websocket.send_text(message_str)
            except Exception as e:
                logger.warning(f"Failed to send caption to client: {e}")
                disconnected_clients.append(websocket)

        # Clean up disconnected clients
        for websocket in disconnected_clients:
            self.caption_connections[room_id].remove(websocket)

    def get_room_status(self, room_id: str) -> Dict:
        """Get status information for a room."""
        return {
            "room_id": room_id,
            "peer_count": len(self.room_peers[room_id]),
            "peers": list(self.room_peers[room_id]),
            "caption_clients": len(self.caption_connections[room_id]),
            "caption_history_count": len(self.room_captions[room_id])
        }

# Global connection manager instance
connection_manager = ConnectionManager()