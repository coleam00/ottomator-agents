"""
Unified Configuration System
=============================
Centralized configuration management for all 70+ agents in the repository.
Supports environment variables, config files, and dynamic configuration.
"""

from __future__ import annotations
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field, asdict
from enum import Enum
import os
import json
from pathlib import Path

from dotenv import load_dotenv


class ModelProvider(Enum):
    """LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    GOOGLE = "google"


class DatabaseProvider(Enum):
    """Database providers"""
    SUPABASE = "supabase"
    POSTGRES = "postgres"
    MONGODB = "mongodb"
    PINECONE = "pinecone"
    CHROMADB = "chromadb"


@dataclass
class LLMConfig:
    """LLM configuration"""
    provider: ModelProvider = ModelProvider.OPENAI
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 60


@dataclass
class DatabaseConfig:
    """Database configuration"""
    provider: DatabaseProvider = DatabaseProvider.SUPABASE
    url: str = ""
    api_key: str = ""
    connection_string: str = ""


@dataclass
class VoiceConfig:
    """Voice configuration"""
    stt_provider: str = "deepgram"
    stt_api_key: str = ""
    tts_provider: str = "openai"
    tts_api_key: str = ""
    livekit_url: str = ""
    livekit_api_key: str = ""
    livekit_api_secret: str = ""


@dataclass
class MCPConfig:
    """Model Context Protocol configuration"""
    enabled: bool = False
    servers: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    enabled: bool = True
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"


@dataclass
class AgentConfig:
    """Agent-specific configuration"""
    name: str = ""
    enabled: bool = True
    custom_settings: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UnifiedConfig:
    """Unified configuration for all agents"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    voice: VoiceConfig = field(default_factory=VoiceConfig)
    mcp: MCPConfig = field(default_factory=MCPConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    agents: Dict[str, AgentConfig] = field(default_factory=dict)
    environment: str = "development"


class ConfigManager:
    """Centralized configuration management"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Optional path to config file
        """
        self.config_path = config_path or os.getenv("CONFIG_PATH", ".env")
        self.config = UnifiedConfig()
        self._load_configuration()

    def _load_configuration(self):
        """Load configuration from environment and files"""

        # Load environment variables
        load_dotenv(self.config_path)

        # LLM Configuration
        self.config.llm = LLMConfig(
            provider=ModelProvider(os.getenv("LLM_PROVIDER", "openai")),
            model=os.getenv("MODEL_CHOICE", "gpt-4o"),
            api_key=os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", "")),
            base_url=os.getenv("BASE_URL", "https://api.openai.com/v1"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "4096")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60"))
        )

        # Database Configuration
        self.config.database = DatabaseConfig(
            provider=DatabaseProvider(os.getenv("DB_PROVIDER", "supabase")),
            url=os.getenv("SUPABASE_URL", os.getenv("DATABASE_URL", "")),
            api_key=os.getenv("SUPABASE_KEY", ""),
            connection_string=os.getenv("DATABASE_CONNECTION_STRING", "")
        )

        # Voice Configuration
        self.config.voice = VoiceConfig(
            stt_provider=os.getenv("STT_PROVIDER", "deepgram"),
            stt_api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            tts_provider=os.getenv("TTS_PROVIDER", "openai"),
            tts_api_key=os.getenv("OPENAI_API_KEY", ""),
            livekit_url=os.getenv("LIVEKIT_URL", ""),
            livekit_api_key=os.getenv("LIVEKIT_API_KEY", ""),
            livekit_api_secret=os.getenv("LIVEKIT_API_SECRET", "")
        )

        # MCP Configuration
        self.config.mcp = MCPConfig(
            enabled=os.getenv("MCP_ENABLED", "false").lower() == "true",
            servers=self._parse_mcp_servers()
        )

        # Monitoring Configuration
        self.config.monitoring = MonitoringConfig(
            enabled=os.getenv("MONITORING_ENABLED", "true").lower() == "true",
            langfuse_enabled=os.getenv("LANGFUSE_ENABLED", "false").lower() == "true",
            langfuse_public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            langfuse_secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            langfuse_host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
        )

        # Environment
        self.config.environment = os.getenv("ENVIRONMENT", "development")

        # Load JSON config if exists
        self._load_json_config()

    def _parse_mcp_servers(self) -> List[Dict[str, str]]:
        """Parse MCP servers from environment"""
        servers = []

        # Check for Airtable MCP
        if os.getenv("AIRTABLE_API_KEY"):
            servers.append({
                "name": "airtable",
                "api_key": os.getenv("AIRTABLE_API_KEY")
            })

        # Check for GitHub MCP
        if os.getenv("GITHUB_TOKEN"):
            servers.append({
                "name": "github",
                "token": os.getenv("GITHUB_TOKEN")
            })

        # Check for Slack MCP
        if os.getenv("SLACK_BOT_TOKEN"):
            servers.append({
                "name": "slack",
                "token": os.getenv("SLACK_BOT_TOKEN"),
                "team_id": os.getenv("SLACK_TEAM_ID", "")
            })

        # Check for Brave Search MCP
        if os.getenv("BRAVE_API_KEY"):
            servers.append({
                "name": "brave_search",
                "api_key": os.getenv("BRAVE_API_KEY")
            })

        # Check for Firecrawl MCP
        if os.getenv("FIRECRAWL_API_KEY"):
            servers.append({
                "name": "firecrawl",
                "api_key": os.getenv("FIRECRAWL_API_KEY")
            })

        return servers

    def _load_json_config(self):
        """Load configuration from JSON file if exists"""
        json_config_path = Path("config.json")

        if json_config_path.exists():
            try:
                with open(json_config_path, 'r') as f:
                    json_config = json.load(f)

                # Merge JSON config with environment config
                # JSON config takes precedence
                if "agents" in json_config:
                    for agent_name, agent_config in json_config["agents"].items():
                        self.config.agents[agent_name] = AgentConfig(**agent_config)

            except Exception as e:
                print(f"Warning: Failed to load JSON config: {e}")

    def get_llm_config(self) -> LLMConfig:
        """Get LLM configuration"""
        return self.config.llm

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration"""
        return self.config.database

    def get_voice_config(self) -> VoiceConfig:
        """Get voice configuration"""
        return self.config.voice

    def get_mcp_config(self) -> MCPConfig:
        """Get MCP configuration"""
        return self.config.mcp

    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration"""
        return self.config.monitoring

    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent"""
        return self.config.agents.get(agent_name)

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.config.environment == "production"

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.config.environment == "development"

    def save_config(self, filepath: str = "config.json"):
        """Save current configuration to JSON file"""
        config_dict = asdict(self.config)

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        print(f"✅ Configuration saved to {filepath}")

    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return warnings/errors"""
        issues = {
            "errors": [],
            "warnings": []
        }

        # Check LLM configuration
        if not self.config.llm.api_key:
            issues["errors"].append("LLM API key is not set")

        if self.config.llm.temperature < 0 or self.config.llm.temperature > 2:
            issues["warnings"].append(f"LLM temperature {self.config.llm.temperature} is outside recommended range (0-2)")

        # Check voice configuration if needed
        if self.config.voice.livekit_url and not self.config.voice.livekit_api_key:
            issues["errors"].append("LiveKit URL is set but API key is missing")

        # Check MCP configuration
        if self.config.mcp.enabled and not self.config.mcp.servers:
            issues["warnings"].append("MCP is enabled but no servers are configured")

        return issues

    def get_config_summary(self) -> str:
        """Get a human-readable configuration summary"""
        lines = [
            "=" * 60,
            "UNIFIED CONFIGURATION SUMMARY",
            "=" * 60,
            "",
            f"Environment: {self.config.environment}",
            "",
            "LLM Configuration:",
            f"  Provider: {self.config.llm.provider.value}",
            f"  Model: {self.config.llm.model}",
            f"  Temperature: {self.config.llm.temperature}",
            f"  Max Tokens: {self.config.llm.max_tokens}",
            f"  API Key: {'✓ Set' if self.config.llm.api_key else '✗ Missing'}",
            "",
            "Database Configuration:",
            f"  Provider: {self.config.database.provider.value}",
            f"  URL: {'✓ Set' if self.config.database.url else '✗ Missing'}",
            "",
            "Voice Configuration:",
            f"  STT: {self.config.voice.stt_provider}",
            f"  TTS: {self.config.voice.tts_provider}",
            f"  LiveKit: {'✓ Configured' if self.config.voice.livekit_url else '✗ Not configured'}",
            "",
            "MCP Configuration:",
            f"  Enabled: {self.config.mcp.enabled}",
            f"  Servers: {len(self.config.mcp.servers)}",
        ]

        if self.config.mcp.servers:
            lines.append("  Available MCP servers:")
            for server in self.config.mcp.servers:
                lines.append(f"    - {server['name']}")

        lines.extend([
            "",
            "Monitoring:",
            f"  Enabled: {self.config.monitoring.enabled}",
            f"  Langfuse: {self.config.monitoring.langfuse_enabled}",
            "",
            f"Configured Agents: {len(self.config.agents)}",
        ])

        # Add validation results
        validation = self.validate_config()
        if validation["errors"]:
            lines.append("")
            lines.append("⚠️  ERRORS:")
            for error in validation["errors"]:
                lines.append(f"  • {error}")

        if validation["warnings"]:
            lines.append("")
            lines.append("⚠️  WARNINGS:")
            for warning in validation["warnings"]:
                lines.append(f"  • {warning}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigManager()

    return _config_manager


# Example usage
if __name__ == "__main__":
    config = get_config()
    print(config.get_config_summary())

    # Save configuration to file
    # config.save_config()
