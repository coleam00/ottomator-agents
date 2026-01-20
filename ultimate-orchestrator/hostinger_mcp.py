"""
Hostinger MCP Server Integration
==================================
Model Context Protocol server for Hostinger Business Plan integration.
Provides tools for managing hosting, domains, databases, and deployments.
"""

from typing import Any, Dict, List, Optional
import os
import httpx
from datetime import datetime
from pydantic_ai import Agent
from dotenv import load_dotenv

load_dotenv()


class HostingerMCPServer:
    """
    Hostinger MCP Server for managing hosting infrastructure.

    Features:
    - VPS/Cloud server management
    - MySQL database operations
    - Domain management
    - File operations via FTP/SFTP
    - Email account management
    - SSL certificate management
    """

    def __init__(self):
        self.api_key = os.getenv("HOSTINGER_API_KEY")
        self.api_url = os.getenv("HOSTINGER_API_URL", "https://api.hostinger.com/v1")
        self.server_ip = os.getenv("HOSTINGER_SERVER_IP")
        self.ssh_user = os.getenv("HOSTINGER_SSH_USER", "root")

        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to Hostinger API"""

        if not self.api_key:
            return {
                "error": "Missing HOSTINGER_API_KEY",
                "message": "Please set HOSTINGER_API_KEY in your .env file"
            }

        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{self.api_url}/{endpoint}"

            try:
                if method.upper() == "GET":
                    response = await client.get(url, headers=self.base_headers)
                elif method.upper() == "POST":
                    response = await client.post(url, headers=self.base_headers, json=data)
                elif method.upper() == "PUT":
                    response = await client.put(url, headers=self.base_headers, json=data)
                elif method.upper() == "DELETE":
                    response = await client.delete(url, headers=self.base_headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                return {
                    "error": f"HTTP {e.response.status_code}",
                    "message": str(e),
                    "details": e.response.text
                }
            except Exception as e:
                return {
                    "error": "Request failed",
                    "message": str(e)
                }

    # ========== Server Management ==========

    async def list_servers(self) -> Dict[str, Any]:
        """List all VPS/Cloud servers"""
        return await self._make_request("GET", "servers")

    async def get_server_info(self, server_id: str = None) -> Dict[str, Any]:
        """Get server information"""
        if not server_id:
            server_id = os.getenv("HOSTINGER_SERVER_ID", "default")
        return await self._make_request("GET", f"servers/{server_id}")

    async def restart_server(self, server_id: str = None) -> Dict[str, Any]:
        """Restart server"""
        if not server_id:
            server_id = os.getenv("HOSTINGER_SERVER_ID", "default")
        return await self._make_request("POST", f"servers/{server_id}/restart")

    # ========== MySQL Database Management ==========

    async def list_databases(self) -> Dict[str, Any]:
        """List all MySQL databases"""
        return await self._make_request("GET", "databases")

    async def create_database(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create MySQL database"""
        data = {"name": name, "description": description}
        return await self._make_request("POST", "databases", data)

    async def get_database_info(self, database_id: str) -> Dict[str, Any]:
        """Get database connection info"""
        return await self._make_request("GET", f"databases/{database_id}")

    async def create_database_backup(self, database_id: str) -> Dict[str, Any]:
        """Create database backup"""
        return await self._make_request("POST", f"databases/{database_id}/backups")

    # ========== Domain Management ==========

    async def list_domains(self) -> Dict[str, Any]:
        """List all domains"""
        return await self._make_request("GET", "domains")

    async def get_domain_dns(self, domain: str) -> Dict[str, Any]:
        """Get DNS records"""
        return await self._make_request("GET", f"domains/{domain}/dns")

    async def install_ssl(self, domain: str) -> Dict[str, Any]:
        """Install SSL certificate"""
        data = {"domain": domain, "type": "lets_encrypt"}
        return await self._make_request("POST", "ssl", data)


# Create Hostinger-enabled Agent
class HostingerAgent:
    """Agent with Hostinger management capabilities"""

    def __init__(self):
        self.hostinger = HostingerMCPServer()
        self.agent = Agent(
            model=os.getenv("MODEL_CHOICE", "gpt-4o"),
            system_prompt="""You are a Hostinger infrastructure expert.

            You can manage:
            - VPS/Cloud servers
            - MySQL databases
            - Domains and DNS
            - SSL certificates

            Provide clear, actionable responses.
            """
        )
        self._register_tools()

    def _register_tools(self):
        """Register Hostinger tools"""

        @self.agent.tool_plain
        async def list_servers() -> Dict[str, Any]:
            """List all servers"""
            return await self.hostinger.list_servers()

        @self.agent.tool_plain
        async def get_server_info(server_id: str = None) -> Dict[str, Any]:
            """Get server information"""
            return await self.hostinger.get_server_info(server_id)

        @self.agent.tool_plain
        async def list_databases() -> Dict[str, Any]:
            """List MySQL databases"""
            return await self.hostinger.list_databases()

        @self.agent.tool_plain
        async def create_database(name: str) -> Dict[str, Any]:
            """Create MySQL database"""
            return await self.hostinger.create_database(name)

        @self.agent.tool_plain
        async def list_domains() -> Dict[str, Any]:
            """List domains"""
            return await self.hostinger.list_domains()

        @self.agent.tool_plain
        async def install_ssl(domain: str) -> Dict[str, Any]:
            """Install SSL certificate"""
            return await self.hostinger.install_ssl(domain)

    async def run(self, query: str) -> str:
        """Run agent with query"""
        result = await self.agent.run(query)
        return result.data


if __name__ == "__main__":
    import asyncio

    async def demo():
        print("🌐 Hostinger MCP Agent Demo")
        print("=" * 60)

        try:
            agent = HostingerAgent()
            result = await agent.run("List all my databases")
            print(f"Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            print("\nMake sure to set HOSTINGER_API_KEY in CREDENTIALS.txt")

    asyncio.run(demo())
