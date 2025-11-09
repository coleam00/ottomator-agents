#!/usr/bin/env python3
"""
Archon API Client - Standardized interface for Archon REST API
Provides helper functions for all MCP tool equivalents
"""
import requests
import json
from typing import Dict, List, Optional, Any


class ArchonClient:
    """Client for interacting with Archon API Server"""
    
    def __init__(self, base_url: str = "http://localhost:8181"):
        """Initialize Archon client
        
        Args:
            base_url: Base URL for Archon API Server (default: http://localhost:8181)
        """
        self.base_url = base_url
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """Make standardized request with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            response = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timed out"}
        except requests.exceptions.HTTPError as e:
            return {"success": False, "error": f"HTTP {e.response.status_code}: {e.response.text}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # ===== Knowledge Management =====
    
    def search_knowledge(
        self,
        query: str,
        top_k: int = 10,
        use_reranking: bool = True,
        search_strategy: str = "hybrid",
        filters: Optional[Dict] = None
    ) -> Dict:
        """Search knowledge base using RAG
        
        Args:
            query: Search query text
            top_k: Number of results to return (default: 10)
            use_reranking: Apply cross-encoder reranking (default: True)
            search_strategy: "hybrid", "semantic", or "keyword" (default: "hybrid")
            filters: Optional filters (source_type, tags, etc.)
        """
        return self._request("POST", "/api/knowledge-items/search", json={
            "query": query,
            "top_k": top_k,
            "use_reranking": use_reranking,
            "search_strategy": search_strategy,
            "filters": filters or {}
        })
    
    def list_knowledge_items(
        self,
        limit: int = 50,
        offset: int = 0,
        source_type: Optional[str] = None
    ) -> Dict:
        """List all knowledge base items

        Args:
            limit: Maximum results (default: 50)
            offset: Pagination offset (default: 0)
            source_type: Filter by source type
        """
        params = {"limit": limit, "offset": offset}
        if source_type:
            params["source_type"] = source_type
        return self._request("GET", "/api/knowledge-items", params=params)

    def get_rag_sources(self) -> Dict:
        """Get list of all available RAG sources

        Returns:
            Dictionary containing list of sources with metadata
        """
        return self._request("GET", "/api/rag/sources")

    def get_database_metrics(self) -> Dict:
        """Get database metrics and statistics

        Returns:
            Dictionary containing total documents, chunks, code examples, etc.
        """
        return self._request("GET", "/api/database/metrics")
    
    def crawl_website(
        self,
        url: str,
        crawl_depth: int = 3,
        follow_links: bool = True,
        sitemap_url: Optional[str] = None
    ) -> Dict:
        """Crawl website and add to knowledge base
        
        Args:
            url: Starting URL to crawl
            crawl_depth: Depth of recursive crawling (default: 3)
            follow_links: Follow links in pages (default: True)
            sitemap_url: Optional direct sitemap URL
        """
        return self._request("POST", "/api/knowledge-items/crawl", json={
            "url": url,
            "crawl_depth": crawl_depth,
            "follow_links": follow_links,
            "sitemap_url": sitemap_url
        })
    
    def upload_document(self, file_path: str, metadata: Optional[Dict] = None) -> Dict:
        """Upload document to knowledge base
        
        Args:
            file_path: Path to file to upload
            metadata: Optional metadata (source_type, tags)
        """
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': json.dumps(metadata)} if metadata else {}
            # Don't use self.headers for multipart
            response = requests.post(
                f"{self.base_url}/api/knowledge-items/upload",
                files=files,
                data=data,
                timeout=60
            )
            response.raise_for_status()
            return response.json()
    
    # ===== Project Management =====
    
    def list_projects(self) -> Dict:
        """Get all projects"""
        return self._request("GET", "/api/projects")
    
    def get_project(self, project_id: str) -> Dict:
        """Get project details
        
        Args:
            project_id: UUID of the project
        """
        return self._request("GET", f"/api/projects/{project_id}")
    
    def create_project(self, name: str, description: str = "") -> Dict:
        """Create new project
        
        Args:
            name: Project name
            description: Project description
        """
        return self._request("POST", "/api/projects", json={
            "name": name,
            "description": description
        })
    
    # ===== Task Management =====
    
    def create_task(
        self,
        project_id: str,
        title: str,
        description: str = "",
        status: str = "todo"
    ) -> Dict:
        """Create task in project
        
        Args:
            project_id: UUID of the project
            title: Task title
            description: Task description
            status: Task status (todo, in_progress, done)
        """
        return self._request("POST", "/api/tasks", json={
            "project_id": project_id,
            "title": title,
            "description": description,
            "status": status
        })
    
    def update_task(self, task_id: str, updates: Dict) -> Dict:
        """Update task
        
        Args:
            task_id: UUID of the task
            updates: Dictionary of fields to update
        """
        return self._request("PUT", f"/api/tasks/{task_id}", json=updates)
    
    def list_tasks(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50
    ) -> Dict:
        """List tasks with filters
        
        Args:
            project_id: Filter by project UUID
            status: Filter by status
            limit: Maximum results
        """
        params = {"limit": limit}
        if project_id:
            params["project_id"] = project_id
        if status:
            params["status"] = status
        return self._request("GET", "/api/tasks", params=params)
    
    def get_task(self, task_id: str) -> Dict:
        """Get task details
        
        Args:
            task_id: UUID of the task
        """
        return self._request("GET", f"/api/tasks/{task_id}")
    
    # ===== Document Management =====
    
    def list_documents(self, project_id: Optional[str] = None) -> Dict:
        """Get project documents
        
        Args:
            project_id: Optional filter by project UUID
        """
        params = {"project_id": project_id} if project_id else {}
        return self._request("GET", "/api/documents", params=params)
    
    def create_document(
        self,
        title: str,
        content: str,
        project_id: Optional[str] = None
    ) -> Dict:
        """Create new document
        
        Args:
            title: Document title
            content: Document content
            project_id: Optional project UUID
        """
        return self._request("POST", "/api/documents", json={
            "title": title,
            "content": content,
            "project_id": project_id
        })
    
    def update_document(self, document_id: str, updates: Dict) -> Dict:
        """Update document with version tracking
        
        Args:
            document_id: UUID of the document
            updates: Dictionary of fields to update
        """
        return self._request("PUT", f"/api/documents/{document_id}", json=updates)


# Convenience function for quick usage
def search(query: str, top_k: int = 10) -> Dict:
    """Quick search helper"""
    client = ArchonClient()
    return client.search_knowledge(query, top_k=top_k)


if __name__ == "__main__":
    # Example usage
    client = ArchonClient()
    
    # Search knowledge base
    results = client.search_knowledge("authentication implementation", top_k=5)
    print(json.dumps(results, indent=2))
