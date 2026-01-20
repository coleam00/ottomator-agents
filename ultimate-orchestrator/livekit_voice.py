"""
LIVEKIT VOICE INTEGRATION
Real-time voice communication with AI agents using LiveKit WebRTC
"""

import os
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
import json

try:
    from livekit import api, rtc
    from livekit.agents import JobContext, WorkerOptions, cli, llm
    from livekit.agents.voice_assistant import VoiceAssistant
    from livekit.plugins import deepgram, openai, silero
except ImportError:
    print("⚠️  LiveKit not installed. Run: pip install livekit livekit-agents livekit-plugins-deepgram livekit-plugins-openai livekit-plugins-silero")

from dotenv import load_dotenv
load_dotenv()

@dataclass
class LiveKitConfig:
    """LiveKit configuration"""
    url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    api_key: str = os.getenv("LIVEKIT_API_KEY", "")
    api_secret: str = os.getenv("LIVEKIT_API_SECRET", "")
    room_name: str = "ultimate-orchestrator"

    # Voice settings
    stt_provider: str = "deepgram"  # Speech-to-text
    tts_provider: str = "openai"     # Text-to-speech
    vad_provider: str = "silero"     # Voice activity detection

    # AI settings
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096


class LiveKitVoiceAgent:
    """LiveKit-powered voice agent for real-time conversations"""

    def __init__(self, config: LiveKitConfig = None):
        self.config = config or LiveKitConfig()
        self.room: Optional[rtc.Room] = None
        self.assistant: Optional[VoiceAssistant] = None

    async def create_room(self, room_name: str = None) -> dict:
        """Create a LiveKit room and return connection token"""
        room_name = room_name or self.config.room_name

        # Create room service
        room_service = api.RoomService(
            self.config.url,
            self.config.api_key,
            self.config.api_secret
        )

        # Create or get room
        try:
            room = await room_service.create_room(
                api.CreateRoomRequest(name=room_name)
            )
        except Exception:
            # Room may already exist
            rooms = await room_service.list_rooms(api.ListRoomsRequest())
            room = next((r for r in rooms.rooms if r.name == room_name), None)

            if not room:
                raise Exception(f"Failed to create room: {room_name}")

        # Generate access token for participant
        token = api.AccessToken(
            self.config.api_key,
            self.config.api_secret
        )
        token.with_identity("voice-user")
        token.with_name("Voice User")
        token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True
            )
        )

        jwt_token = token.to_jwt()

        return {
            "room_name": room_name,
            "token": jwt_token,
            "url": self.config.url,
            "sid": room.sid
        }

    async def entrypoint(self, ctx: JobContext):
        """Voice assistant entry point (for agent worker)"""

        # Setup STT (Speech-to-Text)
        if self.config.stt_provider == "deepgram":
            stt = deepgram.STT(
                api_key=os.getenv("DEEPGRAM_API_KEY"),
                model="nova-2-general"
            )
        else:
            stt = openai.STT()

        # Setup TTS (Text-to-Speech)
        if self.config.tts_provider == "openai":
            tts = openai.TTS(
                voice="alloy"  # Options: alloy, echo, fable, onyx, nova, shimmer
            )
        else:
            tts = openai.TTS()

        # Setup VAD (Voice Activity Detection)
        vad = silero.VAD.load()

        # Setup LLM for conversation
        initial_ctx = llm.ChatContext().append(
            role="system",
            text="""You are an advanced AI assistant with access to 70+ specialized agents.

You can help with:
- Code generation and review
- Content creation and research
- Data analysis and visualization
- API integration and automation
- Database design and optimization
- Security audits and performance optimization

When a user asks for something, you can delegate to specialized agents or handle it directly.

Keep responses concise and natural for voice conversation. Ask clarifying questions when needed."""
        )

        # Create voice assistant
        assistant = VoiceAssistant(
            vad=vad,
            stt=stt,
            llm=openai.LLM(
                model=os.getenv("MODEL_CHOICE", "gpt-4o"),
                temperature=self.config.llm_temperature,
                max_tokens=self.config.llm_max_tokens
            ),
            tts=tts,
            chat_ctx=initial_ctx
        )

        # Start the assistant
        assistant.start(ctx.room)

        # Greet the user
        await assistant.say("Hello! I'm your Ultimate AI Orchestrator. How can I help you today?", allow_interruptions=True)

    async def start_worker(self):
        """Start LiveKit worker to handle voice sessions"""
        print("🎙️  Starting LiveKit Voice Worker...")
        print(f"   URL: {self.config.url}")
        print(f"   Room: {self.config.room_name}")

        # Run worker
        cli.run_app(
            WorkerOptions(
                entrypoint_fnc=self.entrypoint,
                api_key=self.config.api_key,
                api_secret=self.config.api_secret,
                ws_url=self.config.url
            )
        )


class LiveKitIntegration:
    """Integration layer for LiveKit with FastAPI"""

    def __init__(self):
        self.config = LiveKitConfig()
        self.agent = LiveKitVoiceAgent(self.config)
        self.active_rooms = {}

    async def create_voice_session(self, user_id: str = None) -> dict:
        """Create a new voice session for a user"""
        room_name = f"voice-{user_id or 'default'}"

        try:
            session = await self.agent.create_room(room_name)
            self.active_rooms[room_name] = session

            return {
                "success": True,
                "session": session,
                "message": "Voice session created"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    async def end_voice_session(self, room_name: str) -> dict:
        """End a voice session"""
        if room_name in self.active_rooms:
            del self.active_rooms[room_name]
            return {"success": True, "message": "Session ended"}

        return {"success": False, "error": "Session not found"}

    def get_active_sessions(self) -> list:
        """Get all active voice sessions"""
        return list(self.active_rooms.values())


# WebSocket handler for real-time voice data
class VoiceWebSocket:
    """WebSocket handler for browser-based voice communication"""

    def __init__(self):
        self.connections = {}

    async def handle_connection(self, websocket, user_id: str):
        """Handle WebSocket connection for voice streaming"""
        self.connections[user_id] = websocket

        try:
            while True:
                # Receive audio data from browser
                data = await websocket.receive_text()
                message = json.loads(data)

                if message["type"] == "audio":
                    # Process audio through LiveKit
                    audio_data = message["data"]
                    response = await self.process_voice(audio_data, user_id)

                    # Send response back
                    await websocket.send_json({
                        "type": "response",
                        "data": response
                    })

                elif message["type"] == "ping":
                    await websocket.send_json({"type": "pong"})

        except Exception as e:
            print(f"WebSocket error: {e}")

        finally:
            if user_id in self.connections:
                del self.connections[user_id]

    async def process_voice(self, audio_data: str, user_id: str) -> dict:
        """Process voice input and return AI response"""
        # This would integrate with the meta orchestrator
        # For now, return a simple response
        return {
            "text": "Voice processing integrated with 70+ AI agents",
            "audio": None  # TTS audio would go here
        }


# FastAPI routes integration
def setup_livekit_routes(app):
    """Setup LiveKit routes in FastAPI app"""

    livekit = LiveKitIntegration()
    voice_ws = VoiceWebSocket()

    @app.post("/voice/create-session")
    async def create_voice_session(user_id: str = None):
        """Create a new LiveKit voice session"""
        return await livekit.create_voice_session(user_id)

    @app.post("/voice/end-session")
    async def end_voice_session(room_name: str):
        """End a voice session"""
        return await livekit.end_voice_session(room_name)

    @app.get("/voice/active-sessions")
    async def get_active_sessions():
        """Get all active voice sessions"""
        return {"sessions": livekit.get_active_sessions()}

    @app.websocket("/ws/voice/{user_id}")
    async def voice_websocket(websocket, user_id: str):
        """WebSocket endpoint for voice streaming"""
        await websocket.accept()
        await voice_ws.handle_connection(websocket, user_id)


# Standalone server for testing
async def main():
    """Run standalone LiveKit voice agent"""
    print("=" * 80)
    print("  🎙️  LIVEKIT VOICE AGENT - ULTIMATE ORCHESTRATOR")
    print("=" * 80)

    config = LiveKitConfig()
    agent = LiveKitVoiceAgent(config)

    # Create a room for testing
    session = await agent.create_room()

    print("\n✅ Voice session created!")
    print(f"\n🔗 Connection Details:")
    print(f"   Room: {session['room_name']}")
    print(f"   URL: {session['url']}")
    print(f"   Token: {session['token'][:50]}...")
    print(f"\n💡 Use these credentials in the frontend to connect")
    print(f"\n🚀 Starting voice worker...")

    # Start worker
    await agent.start_worker()


if __name__ == "__main__":
    asyncio.run(main())
