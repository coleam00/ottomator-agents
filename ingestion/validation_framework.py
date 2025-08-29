"""
Validation Framework for Pre-flight Checks and Data Integrity
Ensures system readiness and data quality before and during ingestion.
"""

import os
import asyncio
import logging
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import asyncpg
from neo4j import AsyncGraphDatabase
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class ValidationStatus(Enum):
    """Validation check status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    status: ValidationStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    @property
    def is_critical(self) -> bool:
        """Check if this is a critical failure."""
        return self.status == ValidationStatus.FAILED and self.details.get("critical", True)


@dataclass
class ValidationReport:
    """Complete validation report."""
    session_id: str
    started_at: str
    completed_at: Optional[str] = None
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    warning_checks: int = 0
    skipped_checks: int = 0
    results: List[ValidationResult] = field(default_factory=list)
    can_proceed: bool = True
    recommendations: List[str] = field(default_factory=list)
    
    def add_result(self, result: ValidationResult):
        """Add a validation result to the report."""
        self.results.append(result)
        self.total_checks += 1
        
        if result.status == ValidationStatus.PASSED:
            self.passed_checks += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed_checks += 1
            if result.is_critical:
                self.can_proceed = False
        elif result.status == ValidationStatus.WARNING:
            self.warning_checks += 1
        elif result.status == ValidationStatus.SKIPPED:
            self.skipped_checks += 1
    
    def finalize(self):
        """Finalize the report."""
        self.completed_at = datetime.now(timezone.utc).isoformat()
        
        # Generate recommendations
        if self.failed_checks > 0:
            self.recommendations.append(f"Fix {self.failed_checks} failed validation checks before proceeding")
        if self.warning_checks > 0:
            self.recommendations.append(f"Review {self.warning_checks} warnings for potential issues")
        if not self.can_proceed:
            self.recommendations.append("Critical failures detected - ingestion cannot proceed")


class ValidationFramework:
    """Comprehensive validation framework for ingestion pipeline."""
    
    def __init__(
        self,
        enable_pre_flight: bool = True,
        enable_data_validation: bool = True,
        enable_system_checks: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize validation framework.
        
        Args:
            enable_pre_flight: Run pre-flight checks
            enable_data_validation: Validate data integrity
            enable_system_checks: Check system resources
            strict_mode: Treat warnings as failures
        """
        self.enable_pre_flight = enable_pre_flight
        self.enable_data_validation = enable_data_validation
        self.enable_system_checks = enable_system_checks
        self.strict_mode = strict_mode
        
        # Connection details from environment
        self.db_provider = os.getenv("DB_PROVIDER", "postgres")
        self.database_url = os.getenv("DATABASE_URL")
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        self.neo4j_uri = os.getenv("NEO4J_URI")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD")
        
        # Validation thresholds
        self.thresholds = {
            "min_disk_space_gb": 1.0,
            "min_memory_gb": 2.0,
            "max_file_size_mb": 100.0,
            "max_chunk_size": 5000,
            "min_chunk_size": 100,
            "max_documents": 1000,
            "connection_timeout": 10.0
        }
    
    async def run_pre_flight_checks(
        self,
        documents_folder: str,
        session_id: str
    ) -> ValidationReport:
        """
        Run comprehensive pre-flight checks.
        
        Args:
            documents_folder: Folder containing documents
            session_id: Session identifier
            
        Returns:
            Validation report
        """
        report = ValidationReport(
            session_id=session_id,
            started_at=datetime.now(timezone.utc).isoformat()
        )
        
        logger.info(f"Starting pre-flight checks for session {session_id}")
        
        # 1. Environment validation
        if self.enable_pre_flight:
            await self._validate_environment(report)
        
        # 2. Database connectivity
        if self.enable_pre_flight:
            await self._validate_database_connections(report)
        
        # 3. Documents folder validation
        if self.enable_data_validation:
            await self._validate_documents_folder(documents_folder, report)
        
        # 4. System resource checks
        if self.enable_system_checks:
            await self._validate_system_resources(report)
        
        # 5. Configuration validation
        if self.enable_pre_flight:
            await self._validate_configuration(report)
        
        # Finalize report
        report.finalize()
        
        logger.info(
            f"Pre-flight checks complete: "
            f"{report.passed_checks} passed, {report.failed_checks} failed, "
            f"{report.warning_checks} warnings. Can proceed: {report.can_proceed}"
        )
        
        return report
    
    async def _validate_environment(self, report: ValidationReport):
        """Validate environment variables."""
        logger.debug("Validating environment variables...")
        
        required_vars = {
            "postgres": ["DATABASE_URL"],
            "supabase": ["SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY"]
        }
        
        # Check database provider
        if self.db_provider not in ["postgres", "supabase"]:
            report.add_result(ValidationResult(
                check_name="db_provider",
                status=ValidationStatus.FAILED,
                message=f"Invalid DB_PROVIDER: {self.db_provider}",
                details={"critical": True}
            ))
            return
        
        # Check required variables for provider
        missing_vars = []
        for var in required_vars.get(self.db_provider, []):
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            report.add_result(ValidationResult(
                check_name="environment_variables",
                status=ValidationStatus.FAILED,
                message=f"Missing required environment variables: {', '.join(missing_vars)}",
                details={"missing": missing_vars, "critical": True}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="environment_variables",
                status=ValidationStatus.PASSED,
                message="All required environment variables present"
            ))
        
        # Check Neo4j configuration
        if not self.neo4j_uri or not self.neo4j_password:
            report.add_result(ValidationResult(
                check_name="neo4j_config",
                status=ValidationStatus.WARNING,
                message="Neo4j credentials not configured - graph building will be skipped",
                details={"critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="neo4j_config",
                status=ValidationStatus.PASSED,
                message="Neo4j configuration present"
            ))
        
        # Check embedding configuration
        embedding_provider = os.getenv("EMBEDDING_PROVIDER")
        if not embedding_provider:
            report.add_result(ValidationResult(
                check_name="embedding_config",
                status=ValidationStatus.FAILED,
                message="EMBEDDING_PROVIDER not configured",
                details={"critical": True}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="embedding_config",
                status=ValidationStatus.PASSED,
                message=f"Embedding provider configured: {embedding_provider}"
            ))
    
    async def _validate_database_connections(self, report: ValidationReport):
        """Validate database connections."""
        logger.debug("Validating database connections...")
        
        # Test PostgreSQL/Supabase connection
        if self.db_provider == "postgres" and self.database_url:
            try:
                conn = await asyncio.wait_for(
                    asyncpg.connect(self.database_url),
                    timeout=self.thresholds["connection_timeout"]
                )
                
                # Test query
                result = await conn.fetchval("SELECT 1")
                await conn.close()
                
                report.add_result(ValidationResult(
                    check_name="postgres_connection",
                    status=ValidationStatus.PASSED,
                    message="PostgreSQL connection successful"
                ))
                
                # Check required tables
                conn = await asyncpg.connect(self.database_url)
                tables = await conn.fetch(
                    "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
                )
                table_names = {t['tablename'] for t in tables}
                required_tables = {'documents', 'chunks', 'sessions', 'messages'}
                missing_tables = required_tables - table_names
                
                if missing_tables:
                    report.add_result(ValidationResult(
                        check_name="database_schema",
                        status=ValidationStatus.FAILED,
                        message=f"Missing required tables: {', '.join(missing_tables)}",
                        details={"missing_tables": list(missing_tables), "critical": True}
                    ))
                else:
                    report.add_result(ValidationResult(
                        check_name="database_schema",
                        status=ValidationStatus.PASSED,
                        message="All required tables present"
                    ))
                
                await conn.close()
                
            except asyncio.TimeoutError:
                report.add_result(ValidationResult(
                    check_name="postgres_connection",
                    status=ValidationStatus.FAILED,
                    message="PostgreSQL connection timeout",
                    details={"critical": True}
                ))
            except Exception as e:
                report.add_result(ValidationResult(
                    check_name="postgres_connection",
                    status=ValidationStatus.FAILED,
                    message=f"PostgreSQL connection failed: {str(e)}",
                    details={"error": str(e), "critical": True}
                ))
        
        elif self.db_provider == "supabase":
            # For Supabase, we'll check via API
            try:
                from supabase import create_client
                client = create_client(self.supabase_url, self.supabase_key)
                
                # Test query
                response = client.table("documents").select("id").limit(1).execute()
                
                report.add_result(ValidationResult(
                    check_name="supabase_connection",
                    status=ValidationStatus.PASSED,
                    message="Supabase connection successful"
                ))
                
            except Exception as e:
                report.add_result(ValidationResult(
                    check_name="supabase_connection",
                    status=ValidationStatus.FAILED,
                    message=f"Supabase connection failed: {str(e)}",
                    details={"error": str(e), "critical": True}
                ))
        
        # Test Neo4j connection
        if self.neo4j_uri and self.neo4j_password:
            try:
                driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password)
                )
                
                async with driver.session() as session:
                    result = await asyncio.wait_for(
                        session.run("RETURN 1 as test"),
                        timeout=self.thresholds["connection_timeout"]
                    )
                    await result.single()
                
                await driver.close()
                
                report.add_result(ValidationResult(
                    check_name="neo4j_connection",
                    status=ValidationStatus.PASSED,
                    message="Neo4j connection successful"
                ))
                
            except asyncio.TimeoutError:
                report.add_result(ValidationResult(
                    check_name="neo4j_connection",
                    status=ValidationStatus.WARNING,
                    message="Neo4j connection timeout - graph building may be slow",
                    details={"critical": False}
                ))
            except Exception as e:
                report.add_result(ValidationResult(
                    check_name="neo4j_connection",
                    status=ValidationStatus.WARNING,
                    message=f"Neo4j connection failed: {str(e)} - graph building will be skipped",
                    details={"error": str(e), "critical": False}
                ))
    
    async def _validate_documents_folder(self, documents_folder: str, report: ValidationReport):
        """Validate documents folder and files."""
        logger.debug(f"Validating documents folder: {documents_folder}")
        
        # Check folder exists
        if not os.path.exists(documents_folder):
            report.add_result(ValidationResult(
                check_name="documents_folder",
                status=ValidationStatus.FAILED,
                message=f"Documents folder not found: {documents_folder}",
                details={"critical": True}
            ))
            return
        
        # Find documents
        doc_patterns = ["*.md", "*.markdown", "*.txt"]
        documents = []
        for pattern in doc_patterns:
            documents.extend(Path(documents_folder).rglob(pattern))
        
        if not documents:
            report.add_result(ValidationResult(
                check_name="documents_found",
                status=ValidationStatus.FAILED,
                message=f"No documents found in {documents_folder}",
                details={"critical": True}
            ))
            return
        
        # Check document count
        if len(documents) > self.thresholds["max_documents"]:
            report.add_result(ValidationResult(
                check_name="document_count",
                status=ValidationStatus.WARNING,
                message=f"Large number of documents ({len(documents)}) may take significant time",
                details={"count": len(documents), "critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="document_count",
                status=ValidationStatus.PASSED,
                message=f"Found {len(documents)} documents to process"
            ))
        
        # Check file sizes
        large_files = []
        total_size_mb = 0
        max_size_mb = self.thresholds["max_file_size_mb"]
        
        for doc_path in documents:
            size_mb = doc_path.stat().st_size / (1024 * 1024)
            total_size_mb += size_mb
            
            if size_mb > max_size_mb:
                large_files.append({
                    "file": str(doc_path),
                    "size_mb": round(size_mb, 2)
                })
        
        if large_files:
            report.add_result(ValidationResult(
                check_name="file_sizes",
                status=ValidationStatus.WARNING,
                message=f"{len(large_files)} files exceed {max_size_mb}MB",
                details={"large_files": large_files[:5], "critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="file_sizes",
                status=ValidationStatus.PASSED,
                message=f"All files within size limits (total: {total_size_mb:.1f}MB)"
            ))
        
        # Check file readability
        unreadable_files = []
        for doc_path in documents[:10]:  # Check first 10 files
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    f.read(100)  # Try reading first 100 chars
            except Exception as e:
                unreadable_files.append({
                    "file": str(doc_path),
                    "error": str(e)
                })
        
        if unreadable_files:
            report.add_result(ValidationResult(
                check_name="file_readability",
                status=ValidationStatus.WARNING,
                message=f"{len(unreadable_files)} files may have encoding issues",
                details={"unreadable_files": unreadable_files, "critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="file_readability",
                status=ValidationStatus.PASSED,
                message="Document files are readable"
            ))
    
    async def _validate_system_resources(self, report: ValidationReport):
        """Validate system resources."""
        logger.debug("Validating system resources...")
        
        try:
            import psutil
            
            # Check disk space
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < self.thresholds["min_disk_space_gb"]:
                report.add_result(ValidationResult(
                    check_name="disk_space",
                    status=ValidationStatus.WARNING,
                    message=f"Low disk space: {free_gb:.1f}GB free",
                    details={"free_gb": free_gb, "critical": False}
                ))
            else:
                report.add_result(ValidationResult(
                    check_name="disk_space",
                    status=ValidationStatus.PASSED,
                    message=f"Sufficient disk space: {free_gb:.1f}GB free"
                ))
            
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < self.thresholds["min_memory_gb"]:
                report.add_result(ValidationResult(
                    check_name="memory",
                    status=ValidationStatus.WARNING,
                    message=f"Low memory: {available_gb:.1f}GB available",
                    details={"available_gb": available_gb, "critical": False}
                ))
            else:
                report.add_result(ValidationResult(
                    check_name="memory",
                    status=ValidationStatus.PASSED,
                    message=f"Sufficient memory: {available_gb:.1f}GB available"
                ))
            
            # Check CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > 80:
                report.add_result(ValidationResult(
                    check_name="cpu_usage",
                    status=ValidationStatus.WARNING,
                    message=f"High CPU usage: {cpu_percent}%",
                    details={"cpu_percent": cpu_percent, "critical": False}
                ))
            else:
                report.add_result(ValidationResult(
                    check_name="cpu_usage",
                    status=ValidationStatus.PASSED,
                    message=f"CPU usage normal: {cpu_percent}%"
                ))
                
        except ImportError:
            report.add_result(ValidationResult(
                check_name="system_resources",
                status=ValidationStatus.SKIPPED,
                message="psutil not installed - skipping system resource checks",
                details={"critical": False}
            ))
    
    async def _validate_configuration(self, report: ValidationReport):
        """Validate ingestion configuration."""
        logger.debug("Validating configuration...")
        
        # Check chunk size settings
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "200"))
        
        if chunk_size < self.thresholds["min_chunk_size"]:
            report.add_result(ValidationResult(
                check_name="chunk_size",
                status=ValidationStatus.WARNING,
                message=f"Small chunk size ({chunk_size}) may create too many chunks",
                details={"chunk_size": chunk_size, "critical": False}
            ))
        elif chunk_size > self.thresholds["max_chunk_size"]:
            report.add_result(ValidationResult(
                check_name="chunk_size",
                status=ValidationStatus.WARNING,
                message=f"Large chunk size ({chunk_size}) may exceed model context",
                details={"chunk_size": chunk_size, "critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="chunk_size",
                status=ValidationStatus.PASSED,
                message=f"Chunk size configured: {chunk_size} (overlap: {chunk_overlap})"
            ))
        
        # Check vector dimensions
        vector_dim = int(os.getenv("VECTOR_DIMENSION", "768"))
        if vector_dim not in [768, 1536, 3072]:
            report.add_result(ValidationResult(
                check_name="vector_dimension",
                status=ValidationStatus.WARNING,
                message=f"Unusual vector dimension: {vector_dim}",
                details={"vector_dim": vector_dim, "critical": False}
            ))
        else:
            report.add_result(ValidationResult(
                check_name="vector_dimension",
                status=ValidationStatus.PASSED,
                message=f"Vector dimension configured: {vector_dim}"
            ))
    
    async def validate_chunk_data(
        self,
        chunk_content: str,
        chunk_index: int,
        document_id: str
    ) -> ValidationResult:
        """
        Validate individual chunk data.
        
        Args:
            chunk_content: Chunk content
            chunk_index: Chunk index
            document_id: Document ID
            
        Returns:
            Validation result
        """
        # Check chunk size
        if len(chunk_content) < self.thresholds["min_chunk_size"]:
            return ValidationResult(
                check_name="chunk_validation",
                status=ValidationStatus.WARNING,
                message=f"Chunk {chunk_index} is very small ({len(chunk_content)} chars)",
                details={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "size": len(chunk_content),
                    "critical": False
                }
            )
        
        if len(chunk_content) > self.thresholds["max_chunk_size"]:
            return ValidationResult(
                check_name="chunk_validation",
                status=ValidationStatus.WARNING,
                message=f"Chunk {chunk_index} exceeds max size ({len(chunk_content)} chars)",
                details={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "size": len(chunk_content),
                    "critical": False
                }
            )
        
        # Check for empty or whitespace-only content
        if not chunk_content.strip():
            return ValidationResult(
                check_name="chunk_validation",
                status=ValidationStatus.FAILED,
                message=f"Chunk {chunk_index} is empty",
                details={
                    "document_id": document_id,
                    "chunk_index": chunk_index,
                    "critical": True
                }
            )
        
        return ValidationResult(
            check_name="chunk_validation",
            status=ValidationStatus.PASSED,
            message=f"Chunk {chunk_index} validated",
            details={
                "document_id": document_id,
                "chunk_index": chunk_index,
                "size": len(chunk_content)
            }
        )
    
    async def validate_embedding(
        self,
        embedding: List[float],
        expected_dim: int,
        chunk_id: str
    ) -> ValidationResult:
        """
        Validate embedding vector.
        
        Args:
            embedding: Embedding vector
            expected_dim: Expected dimension
            chunk_id: Chunk identifier
            
        Returns:
            Validation result
        """
        if not embedding:
            return ValidationResult(
                check_name="embedding_validation",
                status=ValidationStatus.FAILED,
                message=f"No embedding generated for chunk {chunk_id}",
                details={"chunk_id": chunk_id, "critical": True}
            )
        
        if len(embedding) != expected_dim:
            return ValidationResult(
                check_name="embedding_validation",
                status=ValidationStatus.FAILED,
                message=f"Embedding dimension mismatch: got {len(embedding)}, expected {expected_dim}",
                details={
                    "chunk_id": chunk_id,
                    "actual_dim": len(embedding),
                    "expected_dim": expected_dim,
                    "critical": True
                }
            )
        
        # Check for all zeros (failed embedding)
        if all(v == 0 for v in embedding):
            return ValidationResult(
                check_name="embedding_validation",
                status=ValidationStatus.FAILED,
                message=f"Embedding is all zeros for chunk {chunk_id}",
                details={"chunk_id": chunk_id, "critical": True}
            )
        
        return ValidationResult(
            check_name="embedding_validation",
            status=ValidationStatus.PASSED,
            message=f"Embedding validated for chunk {chunk_id}",
            details={"chunk_id": chunk_id, "dimension": len(embedding)}
        )
    
    def print_report(self, report: ValidationReport):
        """Print validation report to console."""
        print("\n" + "="*60)
        print("VALIDATION REPORT")
        print("="*60)
        print(f"Session: {report.session_id}")
        print(f"Started: {report.started_at}")
        print(f"Completed: {report.completed_at}")
        print()
        print(f"Total Checks: {report.total_checks}")
        print(f"  ✅ Passed: {report.passed_checks}")
        print(f"  ❌ Failed: {report.failed_checks}")
        print(f"  ⚠️  Warnings: {report.warning_checks}")
        print(f"  ⏭️  Skipped: {report.skipped_checks}")
        print()
        print(f"Can Proceed: {'YES' if report.can_proceed else 'NO'}")
        print()
        
        if report.failed_checks > 0:
            print("Failed Checks:")
            for result in report.results:
                if result.status == ValidationStatus.FAILED:
                    print(f"  ❌ {result.check_name}: {result.message}")
            print()
        
        if report.warning_checks > 0:
            print("Warnings:")
            for result in report.results:
                if result.status == ValidationStatus.WARNING:
                    print(f"  ⚠️  {result.check_name}: {result.message}")
            print()
        
        if report.recommendations:
            print("Recommendations:")
            for rec in report.recommendations:
                print(f"  • {rec}")
        
        print("="*60)