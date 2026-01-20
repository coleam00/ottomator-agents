"""
Database Configuration and Management
======================================
Unified interface for all database providers: Supabase, MySQL, Qdrant (Quadrant)
"""

import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from dotenv import load_dotenv

load_dotenv()


# ========== SUPABASE CONNECTION ==========

class SupabaseClient:
    """Supabase database client"""

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")
        self.service_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY are required")

        # Import supabase client
        try:
            from supabase import create_client, Client
            self.client: Client = create_client(self.url, self.key)
        except ImportError:
            print("Warning: supabase-py not installed. Run: pip install supabase")
            self.client = None

    async def insert(self, table: str, data: Dict) -> Dict:
        """Insert data into table"""
        if not self.client:
            return {"error": "Supabase client not initialized"}

        try:
            result = self.client.table(table).insert(data).execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"error": str(e)}

    async def query(self, table: str, filters: Dict = None) -> List[Dict]:
        """Query data from table"""
        if not self.client:
            return []

        try:
            query = self.client.table(table).select("*")

            if filters:
                for key, value in filters.items():
                    query = query.eq(key, value)

            result = query.execute()
            return result.data
        except Exception as e:
            print(f"Query error: {e}")
            return []

    async def update(self, table: str, data: Dict, filters: Dict) -> Dict:
        """Update data in table"""
        if not self.client:
            return {"error": "Supabase client not initialized"}

        try:
            query = self.client.table(table).update(data)

            for key, value in filters.items():
                query = query.eq(key, value)

            result = query.execute()
            return {"success": True, "data": result.data}
        except Exception as e:
            return {"error": str(e)}

    async def delete(self, table: str, filters: Dict) -> Dict:
        """Delete data from table"""
        if not self.client:
            return {"error": "Supabase client not initialized"}

        try:
            query = self.client.table(table).delete()

            for key, value in filters.items():
                query = query.eq(key, value)

            result = query.execute()
            return {"success": True}
        except Exception as e:
            return {"error": str(e)}


# ========== MYSQL CONNECTION ==========

class MySQLClient:
    """MySQL database client"""

    def __init__(self):
        self.host = os.getenv("MYSQL_HOST")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DATABASE")

        if not all([self.host, self.user, self.password, self.database]):
            raise ValueError("MySQL credentials are incomplete")

        self.connection = None
        self.connect()

    def connect(self):
        """Establish MySQL connection"""
        try:
            import mysql.connector
            self.connection = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
            print(f"✓ Connected to MySQL: {self.database}")
        except ImportError:
            print("Warning: mysql-connector-python not installed. Run: pip install mysql-connector-python")
        except Exception as e:
            print(f"MySQL connection error: {e}")

    async def execute(self, query: str, params: tuple = None) -> Dict:
        """Execute SQL query"""
        if not self.connection:
            return {"error": "Not connected to MySQL"}

        try:
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())

            if query.strip().upper().startswith("SELECT"):
                result = cursor.fetchall()
                cursor.close()
                return {"success": True, "data": result}
            else:
                self.connection.commit()
                cursor.close()
                return {"success": True, "affected_rows": cursor.rowcount}
        except Exception as e:
            return {"error": str(e)}

    async def insert(self, table: str, data: Dict) -> Dict:
        """Insert data into table"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(["%s"] * len(data))
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
        return await self.execute(query, tuple(data.values()))

    async def query(self, table: str, filters: Dict = None) -> List[Dict]:
        """Query data from table"""
        query = f"SELECT * FROM {table}"

        if filters:
            where_clause = " AND ".join([f"{k} = %s" for k in filters.keys()])
            query += f" WHERE {where_clause}"
            result = await self.execute(query, tuple(filters.values()))
        else:
            result = await self.execute(query)

        return result.get("data", [])

    def close(self):
        """Close connection"""
        if self.connection:
            self.connection.close()


# ========== QDRANT (QUADRANT) VECTOR DATABASE ==========

class QdrantClient:
    """Qdrant vector database client"""

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        if not self.url:
            raise ValueError("QDRANT_URL is required")

        try:
            from qdrant_client import QdrantClient as QC
            from qdrant_client.models import Distance, VectorParams

            if self.api_key:
                self.client = QC(url=self.url, api_key=self.api_key)
            else:
                self.client = QC(url=self.url)

            self.Distance = Distance
            self.VectorParams = VectorParams
            print(f"✓ Connected to Qdrant: {self.url}")
        except ImportError:
            print("Warning: qdrant-client not installed. Run: pip install qdrant-client")
            self.client = None

    async def create_collection(
        self,
        collection_name: str,
        vector_size: int = 1536,
        distance: str = "Cosine"
    ) -> Dict:
        """Create a vector collection"""
        if not self.client:
            return {"error": "Qdrant client not initialized"}

        try:
            distance_metric = getattr(self.Distance, distance.upper())
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.VectorParams(
                    size=vector_size,
                    distance=distance_metric
                )
            )
            return {"success": True, "collection": collection_name}
        except Exception as e:
            return {"error": str(e)}

    async def insert_vectors(
        self,
        collection_name: str,
        vectors: List[List[float]],
        payloads: List[Dict],
        ids: Optional[List[str]] = None
    ) -> Dict:
        """Insert vectors into collection"""
        if not self.client:
            return {"error": "Qdrant client not initialized"}

        try:
            from qdrant_client.models import PointStruct

            if not ids:
                ids = [str(i) for i in range(len(vectors))]

            points = [
                PointStruct(id=id_, vector=vector, payload=payload)
                for id_, vector, payload in zip(ids, vectors, payloads)
            ]

            self.client.upsert(
                collection_name=collection_name,
                points=points
            )

            return {"success": True, "inserted": len(points)}
        except Exception as e:
            return {"error": str(e)}

    async def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 5
    ) -> List[Dict]:
        """Search for similar vectors"""
        if not self.client:
            return []

        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit
            )

            return [
                {
                    "id": hit.id,
                    "score": hit.score,
                    "payload": hit.payload
                }
                for hit in results
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return []

    async def list_collections(self) -> List[str]:
        """List all collections"""
        if not self.client:
            return []

        try:
            collections = self.client.get_collections()
            return [col.name for col in collections.collections]
        except Exception as e:
            print(f"Error listing collections: {e}")
            return []


# ========== UNIFIED DATABASE MANAGER ==========

class DatabaseManager:
    """Unified manager for all databases"""

    def __init__(self):
        self.supabase = None
        self.mysql = None
        self.qdrant = None

        self._initialize_databases()

    def _initialize_databases(self):
        """Initialize all configured databases"""

        # Supabase
        if os.getenv("SUPABASE_URL"):
            try:
                self.supabase = SupabaseClient()
                print("✓ Supabase initialized")
            except Exception as e:
                print(f"Supabase initialization failed: {e}")

        # MySQL
        if os.getenv("MYSQL_HOST"):
            try:
                self.mysql = MySQLClient()
                print("✓ MySQL initialized")
            except Exception as e:
                print(f"MySQL initialization failed: {e}")

        # Qdrant
        if os.getenv("QDRANT_URL"):
            try:
                self.qdrant = QdrantClient()
                print("✓ Qdrant initialized")
            except Exception as e:
                print(f"Qdrant initialization failed: {e}")

    def get_supabase(self) -> Optional[SupabaseClient]:
        """Get Supabase client"""
        return self.supabase

    def get_mysql(self) -> Optional[MySQLClient]:
        """Get MySQL client"""
        return self.mysql

    def get_qdrant(self) -> Optional[QdrantClient]:
        """Get Qdrant client"""
        return self.qdrant

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all databases"""
        status = {}

        if self.supabase:
            try:
                await self.supabase.query("users", filters={"id": "health-check"})
                status["supabase"] = "healthy"
            except:
                status["supabase"] = "unhealthy"

        if self.mysql:
            try:
                result = await self.mysql.execute("SELECT 1")
                status["mysql"] = "healthy" if result.get("success") else "unhealthy"
            except:
                status["mysql"] = "unhealthy"

        if self.qdrant:
            try:
                await self.qdrant.list_collections()
                status["qdrant"] = "healthy"
            except:
                status["qdrant"] = "unhealthy"

        return status


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global _db_manager

    if _db_manager is None:
        _db_manager = DatabaseManager()

    return _db_manager


# Example usage and testing
async def demo():
    """Demonstrate database connections"""

    print("=" * 80)
    print("🗄️  DATABASE CONFIGURATION DEMO")
    print("=" * 80)
    print()

    db = get_database_manager()

    # Health check
    print("Running health check...")
    health = await db.health_check()

    for db_name, status in health.items():
        icon = "✅" if status == "healthy" else "❌"
        print(f"{icon} {db_name.upper()}: {status}")

    print()
    print("=" * 80)
    print()

    # Test Supabase
    if db.supabase:
        print("Testing Supabase...")
        result = await db.supabase.query("users")
        print(f"  Query result: {len(result)} records")

    # Test MySQL
    if db.mysql:
        print("Testing MySQL...")
        result = await db.mysql.execute("SHOW TABLES")
        print(f"  Tables: {result}")

    # Test Qdrant
    if db.qdrant:
        print("Testing Qdrant...")
        collections = await db.qdrant.list_collections()
        print(f"  Collections: {collections}")

    print()
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(demo())
