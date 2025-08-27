# Client Integration Guide - Supabase User ID

## Overview
For proper user isolation in Neo4j, the client application MUST pass the user's Supabase UUID with every API request.

## Required: Passing User ID to the Agent

### 1. Getting the User's Supabase UUID

In your client application (React, Next.js, etc.), get the user's UUID from Supabase auth:

```typescript
import { createClient } from '@supabase/supabase-js'

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
)

// Get the current user
const { data: { user } } = await supabase.auth.getUser()

// The user's UUID is in user.id
const userUUID = user?.id  // This is what you need to pass
```

### 2. Sending the UUID with Chat Requests

When calling the agent API, include the user's UUID in the request:

```typescript
// Example API call from your frontend
const response = await fetch('https://your-api.com/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${session?.access_token}`
  },
  body: JSON.stringify({
    message: userMessage,
    session_id: sessionId,
    user_id: user.id,  // ← CRITICAL: Pass the Supabase UUID here
    metadata: {
      // any additional metadata
    }
  })
})
```

### 3. For Streaming Requests

```typescript
const response = await fetch('https://your-api.com/chat/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'Authorization': `Bearer ${session?.access_token}`
  },
  body: JSON.stringify({
    message: userMessage,
    session_id: sessionId,
    user_id: user.id,  // ← CRITICAL: Pass the Supabase UUID here
  })
})
```

## How It Works

### Data Flow
```
Frontend App
    ↓
Gets user.id from Supabase Auth
    ↓
Sends user_id with every request
    ↓
Backend API receives user_id
    ↓
Agent uses user_id as group_id in Neo4j
    ↓
User's conversations are isolated
```

### What Happens in the Backend

1. **Episodic Memory Creation**:
   - The `user_id` is used as `group_id` when storing conversations
   - Each user's episodes are partitioned in the graph

2. **Searching Past Conversations**:
   - When the agent searches episodic memory, it filters by the user's `group_id`
   - Only that user's conversations are returned

3. **Shared Knowledge Access**:
   - Medical knowledge base uses `group_id="0"`
   - All users can access shared medical information

## Example: Next.js Integration

```typescript
// app/api/chat/route.ts
import { createRouteHandlerClient } from '@supabase/auth-helpers-nextjs'
import { cookies } from 'next/headers'

export async function POST(request: Request) {
  const supabase = createRouteHandlerClient({ cookies })
  
  // Get the current user
  const { data: { user } } = await supabase.auth.getUser()
  
  if (!user) {
    return new Response('Unauthorized', { status: 401 })
  }
  
  const body = await request.json()
  
  // Forward to your agent API with user UUID
  const agentResponse = await fetch(process.env.AGENT_API_URL + '/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ...body,
      user_id: user.id,  // Always include the Supabase UUID
    })
  })
  
  return agentResponse
}
```

## Example: React Hook

```typescript
// hooks/useAgentChat.ts
import { useUser } from '@supabase/auth-helpers-react'

export function useAgentChat() {
  const user = useUser()
  
  const sendMessage = async (message: string, sessionId: string) => {
    if (!user) {
      throw new Error('User not authenticated')
    }
    
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        message,
        session_id: sessionId,
        user_id: user.id,  // Always pass user UUID
      })
    })
    
    return response
  }
  
  return { sendMessage }
}
```

## Testing User Isolation

To verify user isolation is working:

1. **Log in as User A**
   - Have a conversation about specific symptoms
   - Note the user UUID

2. **Log in as User B** 
   - Try searching for User A's symptoms
   - Should not find User A's conversations

3. **Check Shared Knowledge**
   - Both users should find general medical information
   - But not each other's personal conversations

## Security Considerations

### ⚠️ IMPORTANT
- **Never** use email or username as the user_id
- **Always** use the Supabase UUID (user.id)
- **Validate** the user_id on the backend if possible
- **Don't** allow clients to specify arbitrary user_ids

### Backend Validation (Optional but Recommended)

```python
# In your API endpoint
from fastapi import Header, HTTPException
import jwt

async def validate_user_id(
    authorization: str = Header(...),
    user_id: str
) -> bool:
    """Validate that the user_id matches the JWT token"""
    try:
        # Decode the Supabase JWT
        token = authorization.replace("Bearer ", "")
        decoded = jwt.decode(token, options={"verify_signature": False})
        
        # Check if user_id matches the sub claim
        if decoded.get("sub") != user_id:
            raise HTTPException(status_code=403, detail="User ID mismatch")
        
        return True
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")
```

## Troubleshooting

### Problem: User can't see their past conversations
**Solution**: Ensure you're passing the exact Supabase UUID, not a different identifier

### Problem: Users see each other's data
**Solution**: Verify the user_id is being passed correctly and matches the Supabase UUID

### Problem: No episodic memories are created
**Solution**: Check that user_id is included in the metadata when creating episodes

## API Reference

### Required Fields for Chat Endpoint

```json
POST /chat
{
  "message": "string (required)",
  "session_id": "string (optional)",
  "user_id": "string (REQUIRED - Supabase UUID)",
  "metadata": {}
}
```

### Response Will Include
- The assistant's response
- Tools used (including episodic memory searches)
- All searches will be automatically filtered by user's group_id

## Summary

✅ **DO**: Always pass `user.id` from Supabase auth  
✅ **DO**: Include it in every API request  
✅ **DO**: Validate on the backend if possible  

❌ **DON'T**: Use email or username as user_id  
❌ **DON'T**: Allow arbitrary user_id values  
❌ **DON'T**: Forget to pass user_id  

The user isolation is automatic once you pass the correct Supabase UUID!