"""
Real-Time Monitoring and Analytics Dashboard
============================================
Comprehensive monitoring system for tracking agent performance, usage,
and system health across all 70+ agents.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import json
from collections import defaultdict, deque


class MetricType(Enum):
    """Types of metrics to track"""
    AGENT_EXECUTION = "agent_execution"
    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    ERROR = "error"
    SUCCESS = "success"
    COST = "cost"


@dataclass
class Metric:
    """Individual metric data point"""
    timestamp: datetime
    agent_name: str
    metric_type: MetricType
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentStats:
    """Statistics for a single agent"""
    agent_name: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_latency_ms: float = 0
    total_tokens: int = 0
    total_cost: float = 0
    avg_latency_ms: float = 0
    success_rate: float = 0
    last_execution: Optional[datetime] = None
    errors: List[str] = field(default_factory=list)


class MonitoringDashboard:
    """Real-time monitoring and analytics dashboard"""

    def __init__(self, retention_hours: int = 24):
        """
        Initialize monitoring dashboard.

        Args:
            retention_hours: How many hours of metrics to retain
        """
        self.retention_hours = retention_hours
        self.metrics: deque = deque(maxlen=10000)  # Keep last 10k metrics
        self.agent_stats: Dict[str, AgentStats] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.start_time = datetime.now()

    def record_metric(self, metric: Metric):
        """Record a new metric"""
        self.metrics.append(metric)

        # Update agent stats
        if metric.agent_name not in self.agent_stats:
            self.agent_stats[metric.agent_name] = AgentStats(agent_name=metric.agent_name)

        stats = self.agent_stats[metric.agent_name]

        if metric.metric_type == MetricType.AGENT_EXECUTION:
            stats.total_executions += 1
            stats.last_execution = metric.timestamp

        elif metric.metric_type == MetricType.SUCCESS:
            stats.successful_executions += 1

        elif metric.metric_type == MetricType.ERROR:
            stats.failed_executions += 1
            if "error_message" in metric.metadata:
                stats.errors.append(metric.metadata["error_message"])

        elif metric.metric_type == MetricType.LATENCY:
            stats.total_latency_ms += metric.value

        elif metric.metric_type == MetricType.TOKEN_USAGE:
            stats.total_tokens += int(metric.value)

        elif metric.metric_type == MetricType.COST:
            stats.total_cost += metric.value

        # Update derived stats
        if stats.total_executions > 0:
            stats.avg_latency_ms = stats.total_latency_ms / stats.total_executions
            stats.success_rate = (stats.successful_executions / stats.total_executions) * 100

        # Check for alerts
        self._check_alerts(metric, stats)

    def _check_alerts(self, metric: Metric, stats: AgentStats):
        """Check if any alert conditions are met"""

        # Alert if success rate drops below 80%
        if stats.total_executions >= 10 and stats.success_rate < 80:
            self.alerts.append({
                "timestamp": datetime.now(),
                "severity": "warning",
                "agent": metric.agent_name,
                "message": f"Success rate dropped to {stats.success_rate:.1f}%"
            })

        # Alert if latency is very high
        if metric.metric_type == MetricType.LATENCY and metric.value > 10000:  # 10 seconds
            self.alerts.append({
                "timestamp": datetime.now(),
                "severity": "warning",
                "agent": metric.agent_name,
                "message": f"High latency detected: {metric.value:.0f}ms"
            })

        # Alert on errors
        if metric.metric_type == MetricType.ERROR:
            self.alerts.append({
                "timestamp": datetime.now(),
                "severity": "error",
                "agent": metric.agent_name,
                "message": metric.metadata.get("error_message", "Unknown error")
            })

    def get_system_summary(self) -> Dict[str, Any]:
        """Get overall system summary"""
        total_executions = sum(s.total_executions for s in self.agent_stats.values())
        total_successes = sum(s.successful_executions for s in self.agent_stats.values())
        total_failures = sum(s.failed_executions for s in self.agent_stats.values())
        total_tokens = sum(s.total_tokens for s in self.agent_stats.values())
        total_cost = sum(s.total_cost for s in self.agent_stats.values())

        avg_success_rate = (total_successes / total_executions * 100) if total_executions > 0 else 0

        uptime = datetime.now() - self.start_time

        return {
            "uptime_seconds": uptime.total_seconds(),
            "total_agents": len(self.agent_stats),
            "active_agents": sum(1 for s in self.agent_stats.values() if s.total_executions > 0),
            "total_executions": total_executions,
            "total_successes": total_successes,
            "total_failures": total_failures,
            "success_rate": avg_success_rate,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "active_alerts": len([a for a in self.alerts if (datetime.now() - a["timestamp"]).seconds < 300])
        }

    def get_agent_summary(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific agent"""
        stats = self.agent_stats.get(agent_name)

        if not stats:
            return None

        return {
            "agent_name": stats.agent_name,
            "total_executions": stats.total_executions,
            "successful_executions": stats.successful_executions,
            "failed_executions": stats.failed_executions,
            "success_rate": stats.success_rate,
            "avg_latency_ms": stats.avg_latency_ms,
            "total_tokens": stats.total_tokens,
            "total_cost": stats.total_cost,
            "last_execution": stats.last_execution.isoformat() if stats.last_execution else None,
            "recent_errors": stats.errors[-5:] if stats.errors else []
        }

    def get_top_agents(self, metric: str = "executions", limit: int = 10) -> List[Dict[str, Any]]:
        """Get top agents by a specific metric"""
        agents_list = list(self.agent_stats.values())

        if metric == "executions":
            agents_list.sort(key=lambda x: x.total_executions, reverse=True)
        elif metric == "latency":
            agents_list.sort(key=lambda x: x.avg_latency_ms, reverse=True)
        elif metric == "tokens":
            agents_list.sort(key=lambda x: x.total_tokens, reverse=True)
        elif metric == "cost":
            agents_list.sort(key=lambda x: x.total_cost, reverse=True)
        elif metric == "errors":
            agents_list.sort(key=lambda x: x.failed_executions, reverse=True)

        return [
            {
                "rank": idx + 1,
                "agent_name": agent.agent_name,
                "executions": agent.total_executions,
                "success_rate": agent.success_rate,
                "avg_latency_ms": agent.avg_latency_ms,
                "total_tokens": agent.total_tokens,
                "total_cost": agent.total_cost
            }
            for idx, agent in enumerate(agents_list[:limit])
        ]

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return sorted(
            self.alerts,
            key=lambda x: x["timestamp"],
            reverse=True
        )[:limit]

    def get_metrics_by_timerange(
        self,
        start_time: datetime,
        end_time: datetime,
        metric_type: Optional[MetricType] = None
    ) -> List[Metric]:
        """Get metrics within a time range"""
        filtered_metrics = [
            m for m in self.metrics
            if start_time <= m.timestamp <= end_time
        ]

        if metric_type:
            filtered_metrics = [m for m in filtered_metrics if m.metric_type == metric_type]

        return filtered_metrics

    def export_metrics(self, filepath: str):
        """Export all metrics to JSON file"""
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "system_summary": self.get_system_summary(),
            "agent_stats": {
                name: self.get_agent_summary(name)
                for name in self.agent_stats.keys()
            },
            "recent_alerts": self.get_recent_alerts(50),
            "metrics_count": len(self.metrics)
        }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"✅ Metrics exported to {filepath}")

    def render_dashboard(self) -> str:
        """Render a text-based dashboard"""
        lines = [
            "=" * 80,
            "🎯 AGENT MONITORING DASHBOARD",
            "=" * 80,
            ""
        ]

        # System summary
        summary = self.get_system_summary()
        uptime = timedelta(seconds=summary["uptime_seconds"])

        lines.extend([
            "SYSTEM OVERVIEW",
            "-" * 80,
            f"Uptime: {uptime}",
            f"Total Agents: {summary['total_agents']} | Active: {summary['active_agents']}",
            f"Total Executions: {summary['total_executions']:,}",
            f"Success Rate: {summary['success_rate']:.1f}% ({summary['total_successes']:,} successes, {summary['total_failures']:,} failures)",
            f"Total Tokens: {summary['total_tokens']:,}",
            f"Total Cost: ${summary['total_cost']:.4f}",
            f"Active Alerts: {summary['active_alerts']}",
            ""
        ])

        # Top agents by executions
        lines.extend([
            "TOP 5 MOST USED AGENTS",
            "-" * 80
        ])

        top_agents = self.get_top_agents("executions", 5)
        for agent in top_agents:
            lines.append(
                f"{agent['rank']}. {agent['agent_name']}: "
                f"{agent['executions']:,} executions | "
                f"{agent['success_rate']:.1f}% success | "
                f"{agent['avg_latency_ms']:.0f}ms avg latency"
            )

        lines.append("")

        # Recent alerts
        recent_alerts = self.get_recent_alerts(5)
        if recent_alerts:
            lines.extend([
                "RECENT ALERTS",
                "-" * 80
            ])

            for alert in recent_alerts:
                severity_icon = "⚠️" if alert["severity"] == "warning" else "❌"
                lines.append(
                    f"{severity_icon} [{alert['timestamp'].strftime('%H:%M:%S')}] "
                    f"{alert['agent']}: {alert['message']}"
                )

            lines.append("")

        # Performance breakdown
        lines.extend([
            "PERFORMANCE BREAKDOWN",
            "-" * 80,
        ])

        top_latency = self.get_top_agents("latency", 3)
        if top_latency:
            lines.append("Slowest Agents:")
            for agent in top_latency:
                lines.append(f"  • {agent['agent_name']}: {agent['avg_latency_ms']:.0f}ms")

        lines.append("")

        top_cost = self.get_top_agents("cost", 3)
        if top_cost:
            lines.append("Most Expensive Agents:")
            for agent in top_cost:
                lines.append(f"  • {agent['agent_name']}: ${agent['total_cost']:.4f}")

        lines.append("")
        lines.append("=" * 80)

        return "\n".join(lines)


# Global dashboard instance
_dashboard: Optional[MonitoringDashboard] = None


def get_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance"""
    global _dashboard

    if _dashboard is None:
        _dashboard = MonitoringDashboard()

    return _dashboard


# Example usage
async def demo_monitoring():
    """Demonstrate monitoring dashboard"""

    dashboard = get_dashboard()

    print("🎯 MONITORING DASHBOARD DEMO")
    print("=" * 80)
    print("\nSimulating agent executions...\n")

    # Simulate some metrics
    agents = ["livekit-voice", "code-reviewer", "rag-agent", "api-architect", "security-auditor"]

    for i in range(50):
        agent = agents[i % len(agents)]

        # Execution start
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=agent,
            metric_type=MetricType.AGENT_EXECUTION,
            value=1
        ))

        # Latency
        import random
        latency = random.uniform(100, 3000)
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=agent,
            metric_type=MetricType.LATENCY,
            value=latency
        ))

        # Success or error
        if random.random() > 0.1:  # 90% success rate
            dashboard.record_metric(Metric(
                timestamp=datetime.now(),
                agent_name=agent,
                metric_type=MetricType.SUCCESS,
                value=1
            ))
        else:
            dashboard.record_metric(Metric(
                timestamp=datetime.now(),
                agent_name=agent,
                metric_type=MetricType.ERROR,
                value=1,
                metadata={"error_message": "Simulated error"}
            ))

        # Token usage
        tokens = random.randint(100, 5000)
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=agent,
            metric_type=MetricType.TOKEN_USAGE,
            value=tokens
        ))

        # Cost (rough estimate: $0.01 per 1000 tokens)
        cost = tokens * 0.00001
        dashboard.record_metric(Metric(
            timestamp=datetime.now(),
            agent_name=agent,
            metric_type=MetricType.COST,
            value=cost
        ))

        await asyncio.sleep(0.01)

    # Display dashboard
    print(dashboard.render_dashboard())

    # Export metrics
    dashboard.export_metrics("monitoring_export.json")


if __name__ == "__main__":
    asyncio.run(demo_monitoring())
