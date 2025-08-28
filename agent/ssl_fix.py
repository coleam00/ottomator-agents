"""
SSL connection fix for Supabase Cloudflare issues.
This module provides robust HTTP client configuration for dealing with SSL handshake failures.
"""

import httpx
import ssl
import certifi
from typing import Optional

def create_robust_http_client(timeout: int = 60) -> httpx.Client:
    """
    Create an HTTP client with robust SSL configuration.
    
    This handles:
    - SSL handshake failures (Error 525)
    - Connection errors (Error 520)
    - Proper certificate verification
    - Connection pooling and retry logic
    """
    # Create SSL context with proper configuration
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Configure limits for connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0  # Close idle connections after 30 seconds
    )
    
    # Create transport with proper SSL configuration
    transport = httpx.HTTPTransport(
        verify=ssl_context,
        retries=3,  # Retry failed requests
        limits=limits
    )
    
    # Create client with robust configuration
    client = httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(
            connect=10.0,  # Connection timeout
            read=timeout,   # Read timeout
            write=10.0,     # Write timeout
            pool=5.0        # Pool timeout
        ),
        headers={
            "User-Agent": "MaryPause-AI/1.0",
            "Accept": "application/json"
        },
        follow_redirects=True
    )
    
    return client


def create_async_robust_http_client(timeout: int = 60) -> httpx.AsyncClient:
    """
    Create an async HTTP client with robust SSL configuration.
    """
    # Create SSL context with proper configuration
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    
    # Configure limits for connection pooling
    limits = httpx.Limits(
        max_keepalive_connections=5,
        max_connections=10,
        keepalive_expiry=30.0
    )
    
    # Create async transport
    transport = httpx.AsyncHTTPTransport(
        verify=ssl_context,
        retries=3,
        limits=limits
    )
    
    # Create async client
    client = httpx.AsyncClient(
        transport=transport,
        timeout=httpx.Timeout(
            connect=10.0,
            read=timeout,
            write=10.0,
            pool=5.0
        ),
        headers={
            "User-Agent": "MaryPause-AI/1.0",
            "Accept": "application/json"
        },
        follow_redirects=True
    )
    
    return client