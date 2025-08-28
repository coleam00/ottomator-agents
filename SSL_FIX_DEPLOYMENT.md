# SSL Handshake Fix for Supabase Connection Issues

## Problem
The API server is experiencing SSL handshake failures (Cloudflare errors 525 and 520) when connecting to Supabase, causing the agent to fail when responding to chat requests.

## Root Causes
1. **SSL/TLS Configuration Mismatch**: Cloudflare (protecting Supabase) is unable to establish SSL connection
2. **Connection Pooling Issues**: HTTP connections may be getting stale or timing out
3. **Worker Cycling**: Render free tier restarts workers every ~15 minutes (normal behavior)

## Solutions Implemented

### 1. Added Retry Logic with Tenacity
- Added `tenacity` library to requirements.txt
- Implemented retry decorators on critical Supabase operations:
  - `create_session()`: 3 retries with exponential backoff
  - `get_session()`: 3 retries with exponential backoff  
  - `test_connection()`: 3 retries with exponential backoff
- Retry logic specifically handles SSL/connection errors

### 2. Enhanced Error Handling
- Better logging for connection failures
- Graceful fallbacks when SSL issues occur
- Proper exception handling to prevent cascading failures

### 3. Created SSL Fix Module
- `agent/ssl_fix.py`: Robust HTTP client configuration
- Proper SSL context with certificate verification
- Connection pooling optimization
- Timeout configurations for reliability

## Deployment Steps

### 1. Push Changes to Repository
```bash
git add .
git commit -m "fix: implement SSL retry logic for Supabase connection issues

- Add tenacity library for retry logic with exponential backoff
- Implement retry decorators on critical Supabase operations
- Create ssl_fix module with robust HTTP client configuration
- Handle Cloudflare SSL handshake failures (Error 525/520)
- Improve connection pooling and timeout settings

This fixes intermittent connection failures between Render and Supabase
caused by SSL/TLS configuration mismatches through Cloudflare proxy."

git push origin fix/api-error-handling
```

### 2. Verify Environment Variables on Render

Ensure these are set correctly:
```
SUPABASE_URL=https://bpopugzfbokjzgawshov.supabase.co
SUPABASE_SERVICE_ROLE_KEY=<your-key>
DB_PROVIDER=supabase
LOG_LEVEL=INFO
```

### 3. Consider Upgrading Render Plan

The free tier has limitations that contribute to these issues:
- Automatic worker cycling every 15 minutes
- Limited resources (0.25 CPU, 512 MB RAM)
- No guaranteed uptime

Consider upgrading to at least the Starter plan for:
- More stable workers
- Better resource allocation
- Reduced connection issues

## Alternative Solutions (If Issues Persist)

### 1. Direct PostgreSQL Connection
Instead of going through Supabase API (which uses Cloudflare), connect directly to the PostgreSQL database:

```env
DB_PROVIDER=postgres
DATABASE_URL=postgresql://postgres:[password]@db.bpopugzfbokjzgawshov.supabase.co:5432/postgres?sslmode=require
```

### 2. Use Supabase Connection Pooler
Supabase provides a connection pooler that might handle SSL better:
```
# Use port 6543 instead of 5432 for pooled connections
DATABASE_URL=postgresql://postgres:[password]@db.bpopugzfbokjzgawshov.supabase.co:6543/postgres?sslmode=require&pgbouncer=true
```

### 3. Implement Circuit Breaker Pattern
If retries don't work, implement a circuit breaker to temporarily bypass Supabase and use cached data or alternative storage.

## Monitoring

After deployment, monitor these metrics:
1. SSL handshake errors in logs (should decrease)
2. Successful retry attempts (should see retry logs)
3. API response times (should remain stable)
4. Worker restart frequency (normal on free tier)

## Testing

Test the deployment with:
```bash
# Health check
curl https://marypause-ai.onrender.com/health

# Test chat endpoint
curl -X POST https://marypause-ai.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, test message"}'
```

## Files Modified
- `/agent/supabase_db_utils.py`: Added retry logic and improved error handling
- `/agent/ssl_fix.py`: New module for robust SSL configuration
- `/requirements.txt`: Added tenacity==9.0.0
- `/SSL_FIX_DEPLOYMENT.md`: This documentation

## Status
Ready for deployment. The retry logic should handle transient SSL issues, making the application more resilient to connection problems.