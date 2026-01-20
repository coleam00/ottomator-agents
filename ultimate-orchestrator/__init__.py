"""
Ultimate Orchestrator System
=============================
Supreme AI agent coordination platform for the ottomator-agents ecosystem.

This package provides:
- Meta-orchestrator for coordinating 70+ agents
- Universal voice interface for any agent
- 12 new ultra-powerful specialized agents
- Unified configuration management
- Real-time monitoring and analytics
"""

__version__ = "1.0.0"
__author__ = "oTTomator"

from .meta_orchestrator import (
    MetaOrchestrator,
    AgentRegistry,
    AgentMetadata,
    AgentCategory
)

from .voice_interface import (
    VoiceEnabledAgent,
    VoiceAgentFactory,
    VoiceConfig,
    STTProvider,
    TTSProvider,
    VADProvider,
    MultimodalVoiceAgent
)

from .new_ultra_agents import (
    UltraAgentRegistry,
    CodeReviewAgent,
    DeploymentOrchestratorAgent,
    DatabaseArchitectAgent,
    APIArchitectAgent,
    TestingStrategistAgent,
    ArchitectureAdvisorAgent,
    PerformanceOptimizerAgent,
    SecurityAuditorAgent,
    DocumentationGeneratorAgent,
    CostOptimizerAgent,
    AccessibilityExpertAgent,
    AIMLArchitectAgent
)

from .unified_config import (
    ConfigManager,
    UnifiedConfig,
    LLMConfig,
    DatabaseConfig,
    VoiceConfig as ConfigVoiceConfig,
    MCPConfig,
    MonitoringConfig,
    get_config
)

from .monitoring_dashboard import (
    MonitoringDashboard,
    Metric,
    MetricType,
    AgentStats,
    get_dashboard
)

__all__ = [
    # Meta Orchestrator
    "MetaOrchestrator",
    "AgentRegistry",
    "AgentMetadata",
    "AgentCategory",

    # Voice Interface
    "VoiceEnabledAgent",
    "VoiceAgentFactory",
    "VoiceConfig",
    "STTProvider",
    "TTSProvider",
    "VADProvider",
    "MultimodalVoiceAgent",

    # Ultra Agents
    "UltraAgentRegistry",
    "CodeReviewAgent",
    "DeploymentOrchestratorAgent",
    "DatabaseArchitectAgent",
    "APIArchitectAgent",
    "TestingStrategistAgent",
    "ArchitectureAdvisorAgent",
    "PerformanceOptimizerAgent",
    "SecurityAuditorAgent",
    "DocumentationGeneratorAgent",
    "CostOptimizerAgent",
    "AccessibilityExpertAgent",
    "AIMLArchitectAgent",

    # Configuration
    "ConfigManager",
    "UnifiedConfig",
    "LLMConfig",
    "DatabaseConfig",
    "ConfigVoiceConfig",
    "MCPConfig",
    "MonitoringConfig",
    "get_config",

    # Monitoring
    "MonitoringDashboard",
    "Metric",
    "MetricType",
    "AgentStats",
    "get_dashboard",
]
