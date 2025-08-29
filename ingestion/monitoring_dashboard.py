"""
Real-time Monitoring Dashboard for Document Ingestion
Provides progress tracking, performance metrics, and system health monitoring.
"""

import asyncio
import logging
import time
import psutil
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetric:
    """Single performance metric measurement."""
    timestamp: float
    value: float
    label: str
    
    def age_seconds(self) -> float:
        """Get age of metric in seconds."""
        return time.time() - self.timestamp


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_sent_mb: float
    network_recv_mb: float
    active_connections: int
    timestamp: float = field(default_factory=time.time)


@dataclass 
class IngestionMetrics:
    """Ingestion-specific metrics."""
    documents_total: int = 0
    documents_completed: int = 0
    documents_failed: int = 0
    documents_in_progress: int = 0
    chunks_total: int = 0
    chunks_processed: int = 0
    episodes_created: int = 0
    entities_extracted: int = 0
    avg_doc_processing_time: float = 0
    avg_chunk_processing_time: float = 0
    success_rate: float = 0
    estimated_time_remaining: float = 0
    current_document: Optional[str] = None
    current_phase: str = "idle"
    errors_last_hour: int = 0
    warnings_last_hour: int = 0


class MonitoringDashboard:
    """Real-time monitoring dashboard for ingestion pipeline."""
    
    def __init__(
        self,
        update_interval: float = 1.0,
        metrics_window_size: int = 300,  # 5 minutes of metrics
        enable_system_monitoring: bool = True,
        enable_console_output: bool = True
    ):
        """
        Initialize monitoring dashboard.
        
        Args:
            update_interval: Update interval in seconds
            metrics_window_size: Number of metric samples to keep
            enable_system_monitoring: Monitor system resources
            enable_console_output: Print to console
        """
        self.update_interval = update_interval
        self.metrics_window_size = metrics_window_size
        self.enable_system_monitoring = enable_system_monitoring
        self.enable_console_output = enable_console_output
        
        # Metrics storage
        self.performance_metrics: Dict[str, deque] = {
            "doc_processing_time": deque(maxlen=metrics_window_size),
            "chunk_processing_time": deque(maxlen=metrics_window_size),
            "neo4j_response_time": deque(maxlen=metrics_window_size),
            "embedding_generation_time": deque(maxlen=metrics_window_size),
            "throughput_docs_per_min": deque(maxlen=metrics_window_size),
            "throughput_chunks_per_min": deque(maxlen=metrics_window_size)
        }
        
        self.system_metrics_history = deque(maxlen=metrics_window_size)
        self.ingestion_metrics = IngestionMetrics()
        
        # Performance tracking
        self.document_start_times: Dict[str, float] = {}
        self.chunk_processing_times: List[float] = []
        self.last_throughput_check = time.time()
        self.docs_completed_at_last_check = 0
        self.chunks_processed_at_last_check = 0
        
        # Alert thresholds
        self.alert_thresholds = {
            "cpu_percent": 80,
            "memory_percent": 85,
            "doc_processing_time": 300,  # 5 minutes
            "chunk_processing_time": 30,  # 30 seconds
            "error_rate": 0.1,  # 10%
            "neo4j_timeout_rate": 0.05  # 5%
        }
        
        # Active alerts
        self.active_alerts: List[Dict[str, Any]] = []
        
        # Monitoring task
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start monitoring dashboard."""
        if self._running:
            return
        
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Monitoring dashboard started")
    
    async def stop(self):
        """Stop monitoring dashboard."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring dashboard stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect metrics
                if self.enable_system_monitoring:
                    system_metrics = await self._collect_system_metrics()
                    self.system_metrics_history.append(system_metrics)
                
                # Calculate throughput
                self._calculate_throughput()
                
                # Check for alerts
                self._check_alerts()
                
                # Update display
                if self.enable_console_output:
                    self._display_dashboard()
                
                await asyncio.sleep(self.update_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_mb = memory.used / (1024 * 1024)
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0
            
            # Network I/O
            net_io = psutil.net_io_counters()
            net_sent_mb = net_io.bytes_sent / (1024 * 1024) if net_io else 0
            net_recv_mb = net_io.bytes_recv / (1024 * 1024) if net_io else 0
            
            # Active connections
            connections = len(psutil.net_connections())
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb,
                disk_io_read_mb=disk_read_mb,
                disk_io_write_mb=disk_write_mb,
                network_sent_mb=net_sent_mb,
                network_recv_mb=net_recv_mb,
                active_connections=connections
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_throughput(self):
        """Calculate document and chunk throughput."""
        current_time = time.time()
        time_elapsed = current_time - self.last_throughput_check
        
        if time_elapsed >= 60:  # Calculate every minute
            # Document throughput
            docs_processed = self.ingestion_metrics.documents_completed - self.docs_completed_at_last_check
            doc_throughput = (docs_processed / time_elapsed) * 60
            
            self.performance_metrics["throughput_docs_per_min"].append(
                PerformanceMetric(current_time, doc_throughput, "docs/min")
            )
            
            # Chunk throughput
            chunks_processed = self.ingestion_metrics.chunks_processed - self.chunks_processed_at_last_check
            chunk_throughput = (chunks_processed / time_elapsed) * 60
            
            self.performance_metrics["throughput_chunks_per_min"].append(
                PerformanceMetric(current_time, chunk_throughput, "chunks/min")
            )
            
            # Update checkpoints
            self.last_throughput_check = current_time
            self.docs_completed_at_last_check = self.ingestion_metrics.documents_completed
            self.chunks_processed_at_last_check = self.ingestion_metrics.chunks_processed
    
    def _check_alerts(self):
        """Check for alert conditions."""
        alerts = []
        current_time = time.time()
        
        # System resource alerts
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            
            if latest_system.cpu_percent > self.alert_thresholds["cpu_percent"]:
                alerts.append({
                    "type": "high_cpu",
                    "message": f"High CPU usage: {latest_system.cpu_percent:.1f}%",
                    "severity": "warning",
                    "timestamp": current_time
                })
            
            if latest_system.memory_percent > self.alert_thresholds["memory_percent"]:
                alerts.append({
                    "type": "high_memory",
                    "message": f"High memory usage: {latest_system.memory_percent:.1f}%",
                    "severity": "warning",
                    "timestamp": current_time
                })
        
        # Performance alerts
        if self.ingestion_metrics.avg_doc_processing_time > self.alert_thresholds["doc_processing_time"]:
            alerts.append({
                "type": "slow_processing",
                "message": f"Slow document processing: {self.ingestion_metrics.avg_doc_processing_time:.1f}s avg",
                "severity": "warning",
                "timestamp": current_time
            })
        
        # Error rate alert
        if self.ingestion_metrics.documents_total > 0:
            error_rate = self.ingestion_metrics.documents_failed / self.ingestion_metrics.documents_total
            if error_rate > self.alert_thresholds["error_rate"]:
                alerts.append({
                    "type": "high_error_rate",
                    "message": f"High error rate: {error_rate*100:.1f}%",
                    "severity": "critical",
                    "timestamp": current_time
                })
        
        # Update active alerts
        self.active_alerts = alerts
    
    def _display_dashboard(self):
        """Display dashboard in console."""
        # Clear screen (works on Unix-like systems)
        print("\033[2J\033[H")
        
        # Header
        print("=" * 80)
        print("📊 INGESTION MONITORING DASHBOARD")
        print("=" * 80)
        
        # Progress overview
        progress = (
            self.ingestion_metrics.documents_completed / self.ingestion_metrics.documents_total * 100
            if self.ingestion_metrics.documents_total > 0 else 0
        )
        
        print(f"\n📄 DOCUMENT PROGRESS")
        print(f"  Total: {self.ingestion_metrics.documents_total}")
        print(f"  Completed: {self.ingestion_metrics.documents_completed} ({progress:.1f}%)")
        print(f"  Failed: {self.ingestion_metrics.documents_failed}")
        print(f"  In Progress: {self.ingestion_metrics.documents_in_progress}")
        
        if self.ingestion_metrics.current_document:
            print(f"  Current: {self.ingestion_metrics.current_document}")
            print(f"  Phase: {self.ingestion_metrics.current_phase}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE")
        print(f"  Chunks Processed: {self.ingestion_metrics.chunks_processed}/{self.ingestion_metrics.chunks_total}")
        print(f"  Episodes Created: {self.ingestion_metrics.episodes_created}")
        print(f"  Entities Extracted: {self.ingestion_metrics.entities_extracted}")
        print(f"  Success Rate: {self.ingestion_metrics.success_rate:.1f}%")
        
        if self.ingestion_metrics.avg_doc_processing_time > 0:
            print(f"  Avg Doc Time: {self.ingestion_metrics.avg_doc_processing_time:.1f}s")
        if self.ingestion_metrics.avg_chunk_processing_time > 0:
            print(f"  Avg Chunk Time: {self.ingestion_metrics.avg_chunk_processing_time:.2f}s")
        
        # Throughput
        if self.performance_metrics["throughput_docs_per_min"]:
            latest_doc_throughput = self.performance_metrics["throughput_docs_per_min"][-1].value
            print(f"  Doc Throughput: {latest_doc_throughput:.1f} docs/min")
        
        if self.performance_metrics["throughput_chunks_per_min"]:
            latest_chunk_throughput = self.performance_metrics["throughput_chunks_per_min"][-1].value
            print(f"  Chunk Throughput: {latest_chunk_throughput:.1f} chunks/min")
        
        # System resources
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            print(f"\n💻 SYSTEM RESOURCES")
            print(f"  CPU: {latest_system.cpu_percent:.1f}%")
            print(f"  Memory: {latest_system.memory_percent:.1f}% ({latest_system.memory_mb:.0f} MB)")
            print(f"  Active Connections: {latest_system.active_connections}")
        
        # Estimated time remaining
        if self.ingestion_metrics.estimated_time_remaining > 0:
            eta_minutes = self.ingestion_metrics.estimated_time_remaining / 60
            print(f"\n⏱️  ESTIMATED TIME REMAINING: {eta_minutes:.1f} minutes")
        
        # Active alerts
        if self.active_alerts:
            print(f"\n⚠️  ACTIVE ALERTS")
            for alert in self.active_alerts[:5]:  # Show max 5 alerts
                severity_icon = "🔴" if alert["severity"] == "critical" else "🟡"
                print(f"  {severity_icon} {alert['message']}")
        
        # Footer
        print("\n" + "=" * 80)
        print(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def record_document_start(self, document_id: str, title: str):
        """Record document processing start."""
        self.document_start_times[document_id] = time.time()
        self.ingestion_metrics.documents_in_progress += 1
        self.ingestion_metrics.current_document = title
        self.ingestion_metrics.current_phase = "processing"
    
    def record_document_complete(
        self,
        document_id: str,
        chunks_created: int,
        episodes_created: int,
        entities_extracted: int,
        success: bool = True
    ):
        """Record document processing completion."""
        if document_id in self.document_start_times:
            processing_time = time.time() - self.document_start_times[document_id]
            
            self.performance_metrics["doc_processing_time"].append(
                PerformanceMetric(time.time(), processing_time, "seconds")
            )
            
            # Update average
            recent_times = [m.value for m in self.performance_metrics["doc_processing_time"]]
            if recent_times:
                self.ingestion_metrics.avg_doc_processing_time = statistics.mean(recent_times)
            
            del self.document_start_times[document_id]
        
        self.ingestion_metrics.documents_in_progress -= 1
        
        if success:
            self.ingestion_metrics.documents_completed += 1
        else:
            self.ingestion_metrics.documents_failed += 1
        
        self.ingestion_metrics.chunks_processed += chunks_created
        self.ingestion_metrics.episodes_created += episodes_created
        self.ingestion_metrics.entities_extracted += entities_extracted
        
        # Update success rate
        total_processed = self.ingestion_metrics.documents_completed + self.ingestion_metrics.documents_failed
        if total_processed > 0:
            self.ingestion_metrics.success_rate = (
                self.ingestion_metrics.documents_completed / total_processed * 100
            )
        
        # Estimate time remaining
        if self.ingestion_metrics.documents_completed > 0:
            remaining = self.ingestion_metrics.documents_total - total_processed
            avg_time = self.ingestion_metrics.avg_doc_processing_time
            self.ingestion_metrics.estimated_time_remaining = remaining * avg_time
        
        self.ingestion_metrics.current_document = None
        self.ingestion_metrics.current_phase = "idle"
    
    def record_chunk_processing_time(self, processing_time: float):
        """Record chunk processing time."""
        self.performance_metrics["chunk_processing_time"].append(
            PerformanceMetric(time.time(), processing_time, "seconds")
        )
        
        # Update average
        recent_times = [m.value for m in self.performance_metrics["chunk_processing_time"]]
        if recent_times:
            self.ingestion_metrics.avg_chunk_processing_time = statistics.mean(recent_times)
    
    def record_neo4j_response_time(self, response_time: float):
        """Record Neo4j response time."""
        self.performance_metrics["neo4j_response_time"].append(
            PerformanceMetric(time.time(), response_time, "seconds")
        )
    
    def record_embedding_generation_time(self, generation_time: float):
        """Record embedding generation time."""
        self.performance_metrics["embedding_generation_time"].append(
            PerformanceMetric(time.time(), generation_time, "seconds")
        )
    
    def update_totals(self, total_documents: int, total_chunks: int):
        """Update total counts."""
        self.ingestion_metrics.documents_total = total_documents
        self.ingestion_metrics.chunks_total = total_chunks
    
    def update_phase(self, phase: str):
        """Update current processing phase."""
        self.ingestion_metrics.current_phase = phase
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        summary = {
            "ingestion": asdict(self.ingestion_metrics),
            "performance": {},
            "system": {},
            "alerts": self.active_alerts
        }
        
        # Add performance metrics averages
        for metric_name, metric_deque in self.performance_metrics.items():
            if metric_deque:
                values = [m.value for m in metric_deque]
                summary["performance"][metric_name] = {
                    "avg": statistics.mean(values),
                    "min": min(values),
                    "max": max(values),
                    "latest": metric_deque[-1].value
                }
        
        # Add system metrics
        if self.system_metrics_history:
            cpu_values = [m.cpu_percent for m in self.system_metrics_history]
            memory_values = [m.memory_percent for m in self.system_metrics_history]
            
            summary["system"] = {
                "cpu": {
                    "avg": statistics.mean(cpu_values),
                    "max": max(cpu_values),
                    "latest": self.system_metrics_history[-1].cpu_percent
                },
                "memory": {
                    "avg": statistics.mean(memory_values),
                    "max": max(memory_values),
                    "latest": self.system_metrics_history[-1].memory_percent
                }
            }
        
        return summary
    
    def export_metrics(self, filepath: str):
        """Export metrics to JSON file."""
        try:
            metrics = self.get_metrics_summary()
            with open(filepath, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)
            logger.info(f"Metrics exported to {filepath}")
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")