# Neo4j User Integration Documentation

## Overview

This document describes the implementation of user isolation in Neo4j/Graphiti for episodic conversation memories. The system ensures complete privacy between users while maintaining a shared knowledge base accessible to all.

## Problem Statement

Previously, all episodic conversation nodes in Neo4j were mixed together without user isolation, causing:
- Privacy concerns with user conversations being accessible to others
- Data contamination between different users' sessions
- Incorrect context retrieval in multi-user environments

## Solution Architecture

### Key Components

1. **User Isolation via `group_id`**
   - Graphiti's `group_id` field is used for user-specific namespacing
   - Each user's Supabase UUID serves as their Neo4j `group_id`
   - Ensures complete isolation of conversation episodes per user

2. **Shared Knowledge Base**
   - Documents ingested from medical sources remain without `user_id`
   - Accessible to all users for knowledge queries
   - Only conversation episodes are user-specific

3. **Automatic User Registration**
   - Users are automatically registered in Neo4j when created in Supabase
   - Idempotent registration ensures no duplicates
   - Retry logic handles temporary failures

## Implementation Details

### 1. Backend Changes

#### Graph Utils (`agent/graph_utils.py`)

- **Added `user_id` parameter to `add_episode()`**: Episodes are now created with user-specific `group_id`
- **Fixed `add_fact_triples()`**: Now uses `user_id` for `group_id` instead of `episode_id`
- **New `register_user()` method**: Registers users in Neo4j knowledge graph
- **New `ensure_user_exists()` method**: Idempotent user registration
- **Updated search methods**: Accept optional `user_id` for filtering results

```python
# Example usage
await graph_client.add_episode(
    episode_id="conversation_123",
    content="User conversation content",
    source="chat_session",
    user_id="user_uuid_here"  # This ensures isolation
)
```

#### Episodic Memory Service (`agent/episodic_memory.py`)

- Extracts `user_id` from metadata
- Passes `user_id` to all graph operations
- Filters search results by user when searching conversation history
- Keeps knowledge base searches unfiltered

#### API Layer (`agent/api.py`)

- New endpoint: `POST /api/users/register-neo4j`
- Accepts `user_id` in request body
- Calls `graph_client.ensure_user_exists()` for idempotent registration
- Returns success/error status

### 2. Database Integration

#### Migration (`sql/migrations/003_neo4j_user_integration.sql`)

Creates the following database objects:

- **`neo4j_users` table**: Tracks registration status for each user
- **`register_user_in_neo4j()` trigger**: Fires on new user creation
- **`update_neo4j_registration_status()` function**: Updates registration status
- **`get_pending_neo4j_registrations()` function**: Returns users pending registration
- **RLS policies**: Ensures users can only see their own registration status

### 3. Supabase Edge Function

#### `supabase/functions/register-neo4j-user/index.ts`

Handles the HTTP call to the backend API:

- Receives user_id from database trigger
- Calls backend `/api/users/register-neo4j` endpoint
- Implements retry logic with exponential backoff
- Updates registration status in database
- Supports batch processing of pending registrations

## Deployment Instructions

### 1. Deploy Database Migration

```sql
-- Run in Supabase SQL Editor
-- Copy contents of sql/migrations/003_neo4j_user_integration.sql
```

### 2. Deploy Edge Function

```bash
# Install Supabase CLI if not already installed
npm install -g supabase

# Link to your project
supabase link --project-ref your-project-ref

# Deploy the Edge Function
supabase functions deploy register-neo4j-user

# Set environment variables
supabase secrets set BACKEND_API_URL=https://your-backend-url.com
```

### 3. Configure Backend

Ensure your backend API is deployed with the new changes:
- Updated graph_utils.py
- Updated episodic_memory.py
- Updated api.py with registration endpoint

### 4. Test the Integration

```bash
# Test user registration endpoint directly
curl -X POST https://your-backend/api/users/register-neo4j \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test-user-uuid"}'

# Check registration status in Supabase
SELECT * FROM neo4j_users WHERE user_id = 'test-user-uuid';
```

## Data Flow

1. **User Creation**
   ```
   User signs up → Supabase auth.users → Trigger fires → neo4j_users record created
   ```

2. **Neo4j Registration**
   ```
   Edge Function called → Backend API → Neo4j registration → Status updated
   ```

3. **Conversation Episode Creation**
   ```
   User chats → API extracts user_id → Episode created with user_id as group_id
   ```

4. **Search Isolation**
   ```
   User searches → user_id passed to search → Results filtered by group_id
   ```

## Monitoring and Troubleshooting

### Check Registration Status

```sql
-- View all pending registrations
SELECT * FROM neo4j_users WHERE registration_status IN ('pending', 'retry');

-- Check specific user
SELECT * FROM neo4j_users WHERE user_id = 'user-uuid';

-- View failed registrations
SELECT * FROM neo4j_users WHERE registration_status = 'failed';
```

### Manual Registration Retry

```sql
-- Reset a user for retry
SELECT update_neo4j_registration_status(
  'user-uuid'::UUID,
  'retry',
  NULL
);
```

### Edge Function Logs

```bash
# View Edge Function logs
supabase functions logs register-neo4j-user
```

## Rollback Procedure

If needed, the migration can be rolled back:

```sql
BEGIN;
DROP TRIGGER IF EXISTS trigger_register_neo4j_user ON auth.users;
DROP TRIGGER IF EXISTS update_neo4j_users_updated_at ON neo4j_users;
DROP FUNCTION IF EXISTS register_user_in_neo4j() CASCADE;
DROP FUNCTION IF EXISTS update_neo4j_registration_status(UUID, TEXT, TEXT) CASCADE;
DROP FUNCTION IF EXISTS get_pending_neo4j_registrations() CASCADE;
DROP FUNCTION IF EXISTS process_neo4j_registration(UUID) CASCADE;
DROP FUNCTION IF EXISTS update_neo4j_users_updated_at() CASCADE;
DROP TABLE IF EXISTS neo4j_users CASCADE;
COMMIT;
```

## Security Considerations

1. **User Isolation**: Each user's conversations are completely isolated via `group_id`
2. **RLS Policies**: Users can only view their own registration status
3. **Service Role**: Only service role can modify registration records
4. **Idempotent Operations**: Safe to retry without creating duplicates

## Performance Considerations

1. **Indexed Queries**: All lookups use indexed fields
2. **Batch Processing**: Pending registrations processed in batches
3. **Exponential Backoff**: Prevents overwhelming the system with retries
4. **Namespace Filtering**: Queries are faster due to group_id filtering

## Future Enhancements

1. **Webhook Integration**: Direct Neo4j registration via webhooks
2. **User Deletion**: Cascade delete user data from Neo4j
3. **Analytics**: Track registration metrics and success rates
4. **Bulk Migration**: Tool for migrating existing conversations to user namespaces

## Support

For issues or questions:
1. Check Edge Function logs for registration errors
2. Verify backend API is accessible from Edge Function
3. Ensure Neo4j is running and accessible
4. Check database for registration status and errors