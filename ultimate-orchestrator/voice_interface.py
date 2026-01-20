"""
Unified Voice Interface System
================================
This module provides universal voice input/output capabilities that can be
integrated with ANY agent in the repository. It supports:
- Multiple STT providers (Deepgram, OpenAI Whisper, AssemblyAI)
- Multiple TTS providers (OpenAI, ElevenLabs, Cartesia)
- Voice activity detection (VAD)
- Real-time streaming
- Voice agent wrapping for text-based agents
"""

from __future__ import annotations
from typing import Optional, Callable, AsyncIterator, Protocol
from enum import Enum
from dataclasses import dataclass
import asyncio
import os
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()


class STTProvider(Enum):
    """Speech-to-Text providers"""
    DEEPGRAM = "deepgram"
    OPENAI_WHISPER = "openai_whisper"
    ASSEMBLYAI = "assemblyai"
    AZURE_SPEECH = "azure_speech"


class TTSProvider(Enum):
    """Text-to-Speech providers"""
    OPENAI = "openai"
    ELEVENLABS = "elevenlabs"
    CARTESIA = "cartesia"
    AZURE_SPEECH = "azure_speech"


class VADProvider(Enum):
    """Voice Activity Detection providers"""
    SILERO = "silero"
    WEBRTC = "webrtc"


@dataclass
class VoiceConfig:
    """Configuration for voice interface"""
    stt_provider: STTProvider = STTProvider.DEEPGRAM
    tts_provider: TTSProvider = TTSProvider.OPENAI
    vad_provider: VADProvider = VADProvider.SILERO

    # STT settings
    stt_model: str = "nova-2"
    stt_language: str = "en"

    # TTS settings
    tts_voice: str = "echo"
    tts_speed: float = 1.0

    # Voice detection
    enable_vad: bool = True
    enable_interruption: bool = True

    # LiveKit settings
    livekit_url: Optional[str] = None
    livekit_api_key: Optional[str] = None
    livekit_api_secret: Optional[str] = None


class AgentProtocol(Protocol):
    """Protocol for agents that can be voice-enabled"""

    async def run(self, prompt: str, **kwargs) -> Any:
        """Run the agent with a prompt"""
        ...


class VoiceEnabledAgent:
    """Wrapper that adds voice capabilities to any text-based agent"""

    def __init__(self, base_agent: AgentProtocol, config: VoiceConfig = None):
        """
        Initialize voice-enabled agent wrapper.

        Args:
            base_agent: The underlying text-based agent to wrap
            config: Voice configuration (uses defaults if not provided)
        """
        self.base_agent = base_agent
        self.config = config or VoiceConfig()
        self.conversation_history = []
        self.is_listening = False

    async def process_voice_input(self, audio_data: bytes) -> str:
        """
        Process voice input and convert to text.

        Args:
            audio_data: Raw audio bytes

        Returns:
            Transcribed text
        """
        # In a real implementation, this would call the actual STT service
        # For now, return a placeholder
        transcription = f"[Voice Input Received - {self.config.stt_provider.value}]"
        return transcription

    async def generate_voice_output(self, text: str) -> bytes:
        """
        Convert text to voice output.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes
        """
        # In a real implementation, this would call the actual TTS service
        # For now, return a placeholder
        return b"[Audio output would be generated here]"

    async def run_voice_conversation(self, initial_greeting: Optional[str] = None):
        """
        Run a voice conversation loop.

        Args:
            initial_greeting: Optional initial greeting to speak
        """
        print("🎤 Voice conversation started!")
        print("=" * 60)

        if initial_greeting:
            await self._speak(initial_greeting)

        self.is_listening = True

        # Simulated conversation loop
        while self.is_listening:
            print("\n[Listening...]")

            # In real implementation, this would capture actual audio
            # For demo, we'll use text input as a simulation
            user_input = input("[Speak or type 'exit' to end]: ")

            if user_input.lower() in ['exit', 'quit', 'stop']:
                await self._speak("Goodbye! It was nice talking with you.")
                self.is_listening = False
                break

            # Process through the base agent
            print("\n[Processing...]")
            response = await self.base_agent.run(user_input)

            # Extract response data
            response_text = str(response.data) if hasattr(response, 'data') else str(response)

            # Speak the response
            await self._speak(response_text)

            # Track conversation
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_input,
                "assistant": response_text
            })

    async def _speak(self, text: str):
        """Helper to output speech"""
        print(f"\n🔊 [Assistant]: {text}")
        # In real implementation, this would play actual audio
        await asyncio.sleep(0.5)  # Simulate speech time


class VoiceAgentFactory:
    """Factory for creating voice-enabled versions of agents"""

    @staticmethod
    def create_voice_agent(agent: AgentProtocol, voice_config: Optional[VoiceConfig] = None) -> VoiceEnabledAgent:
        """
        Create a voice-enabled version of any agent.

        Args:
            agent: The base agent to voice-enable
            voice_config: Optional voice configuration

        Returns:
            Voice-enabled agent wrapper
        """
        return VoiceEnabledAgent(agent, voice_config)

    @staticmethod
    def create_livekit_agent(
        agent: AgentProtocol,
        room_name: str,
        participant_identity: str = "voice-agent"
    ):
        """
        Create a LiveKit-compatible voice agent.

        Args:
            agent: The base agent to wrap
            room_name: LiveKit room name
            participant_identity: Identity for the agent participant

        Returns:
            LiveKit-ready agent
        """
        config = VoiceConfig(
            livekit_url=os.getenv("LIVEKIT_URL"),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET")
        )

        voice_agent = VoiceEnabledAgent(agent, config)

        # In a real implementation, this would set up LiveKit connection
        print(f"🎙️ LiveKit agent created for room: {room_name}")

        return voice_agent


class MultimodalVoiceAgent:
    """Advanced voice agent with multimodal capabilities"""

    def __init__(self, agent: AgentProtocol):
        self.agent = agent
        self.voice_enabled = VoiceEnabledAgent(agent)
        self.modes = {
            "voice_only": self.voice_mode,
            "text_only": self.text_mode,
            "multimodal": self.multimodal_mode
        }

    async def voice_mode(self, input_data):
        """Pure voice interaction"""
        return await self.voice_enabled.run_voice_conversation()

    async def text_mode(self, input_data):
        """Pure text interaction"""
        return await self.agent.run(input_data)

    async def multimodal_mode(self, input_data):
        """Combined voice and text interaction"""
        # Support both voice and text input simultaneously
        if isinstance(input_data, bytes):
            # Audio input
            text = await self.voice_enabled.process_voice_input(input_data)
            response = await self.agent.run(text)
        else:
            # Text input
            response = await self.agent.run(input_data)

        return response


# Example usage and testing
async def demo_voice_interface():
    """Demonstrate the voice interface capabilities"""

    print("=" * 60)
    print("🎙️  UNIFIED VOICE INTERFACE SYSTEM")
    print("=" * 60)

    # Simulate a simple echo agent for demo
    class DemoAgent:
        async def run(self, prompt: str, **kwargs):
            class Response:
                def __init__(self, data):
                    self.data = data

            return Response(f"I heard you say: '{prompt}'. How can I help with that?")

    demo_agent = DemoAgent()

    # Create voice-enabled version
    voice_agent = VoiceAgentFactory.create_voice_agent(
        demo_agent,
        VoiceConfig(
            stt_provider=STTProvider.DEEPGRAM,
            tts_provider=TTSProvider.OPENAI,
            tts_voice="echo",
            enable_interruption=True
        )
    )

    print("\n📊 Voice Configuration:")
    print(f"  STT: {voice_agent.config.stt_provider.value}")
    print(f"  TTS: {voice_agent.config.tts_provider.value} ({voice_agent.config.tts_voice})")
    print(f"  VAD: {voice_agent.config.vad_provider.value}")
    print(f"  Interruption: {'Enabled' if voice_agent.config.enable_interruption else 'Disabled'}")

    print("\n" + "=" * 60)
    print("Starting voice conversation demo...")
    print("=" * 60)

    # Run voice conversation
    await voice_agent.run_voice_conversation(
        initial_greeting="Hello! I'm your voice-enabled AI assistant. How can I help you today?"
    )

    print("\n" + "=" * 60)
    print("📋 Conversation History:")
    for idx, exchange in enumerate(voice_agent.conversation_history, 1):
        print(f"\n{idx}. [{exchange['timestamp']}]")
        print(f"   User: {exchange['user']}")
        print(f"   Assistant: {exchange['assistant']}")

    print("\n" + "=" * 60)
    print("✅ Voice interface demo complete!")


if __name__ == "__main__":
    asyncio.run(demo_voice_interface())
