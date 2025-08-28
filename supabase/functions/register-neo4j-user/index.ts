// Supabase Edge Function: register-neo4j-user
// Handles automatic Neo4j user registration when users are created in Supabase

import { serve } from 'https://deno.land/std@0.168.0/http/server.ts'
import { createClient } from 'https://esm.sh/@supabase/supabase-js@2'

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Headers': 'authorization, x-client-info, apikey, content-type',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
}

interface RegistrationRequest {
  user_id: string
  retry_count?: number
}

interface RegistrationResponse {
  status: 'success' | 'error' | 'retry'
  user_id: string
  message: string
  should_retry?: boolean
}

// Environment variables
const SUPABASE_URL = Deno.env.get('SUPABASE_URL')!
const SUPABASE_SERVICE_ROLE_KEY = Deno.env.get('SUPABASE_SERVICE_ROLE_KEY')!
const BACKEND_API_URL = Deno.env.get('BACKEND_API_URL') || 'http://localhost:8058'
const MAX_RETRIES = 5
const RETRY_DELAY_BASE = 5000 // 5 seconds base delay

// Initialize Supabase client
const supabase = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

/**
 * Safely normalize unknown errors to string
 */
function normalizeError(error: unknown): string {
  if (error instanceof Error) {
    return error.message
  }
  if (typeof error === 'string') {
    return error
  }
  if (error && typeof error === 'object' && 'message' in error) {
    return String(error.message)
  }
  return 'An unknown error occurred'
}

/**
 * Register a user in Neo4j/Graphiti knowledge graph
 */
async function registerUserInNeo4j(userId: string): Promise<RegistrationResponse> {
  try {
    // Call backend API to register user
    const response = await fetch(`${BACKEND_API_URL}/api/users/register-neo4j`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
      body: JSON.stringify({ user_id: userId }),
    })

    if (response.ok) {
      const data = await response.json()
      
      // Update registration status in database
      const { error: updateError } = await supabase.rpc('update_neo4j_registration_status', {
        p_user_id: userId,
        p_status: 'registered',
        p_error: null,
      })
      
      if (updateError) {
        console.error('Failed to update registration status:', updateError)
        // Continue anyway since the main operation succeeded
      }

      return {
        status: 'success',
        user_id: userId,
        message: 'User successfully registered in Neo4j',
      }
    } else if (response.status >= 500) {
      // Server error - should retry
      throw new Error(`Server error: ${response.status}`)
    } else {
      // Client error - should not retry
      const errorText = await response.text()
      
      const { error: updateError } = await supabase.rpc('update_neo4j_registration_status', {
        p_user_id: userId,
        p_status: 'failed',
        p_error: `HTTP ${response.status}: ${errorText}`,
      })
      
      if (updateError) {
        console.error('Failed to update registration status:', updateError)
      }

      return {
        status: 'error',
        user_id: userId,
        message: `Registration failed: ${errorText}`,
        should_retry: false,
      }
    }
  } catch (error) {
    console.error('Error registering user in Neo4j:', error)
    
    const errorMessage = normalizeError(error)
    
    // Update status to retry
    const { error: updateError } = await supabase.rpc('update_neo4j_registration_status', {
      p_user_id: userId,
      p_status: 'retry',
      p_error: errorMessage,
    })
    
    if (updateError) {
      console.error('Failed to update registration status:', updateError)
    }

    return {
      status: 'retry',
      user_id: userId,
      message: `Registration error: ${errorMessage}`,
      should_retry: true,
    }
  }
}

/**
 * Process pending registrations with retry logic
 */
async function processPendingRegistrations() {
  try {
    // Get pending registrations from database
    const { data: pendingUsers, error } = await supabase.rpc('get_pending_neo4j_registrations')
    
    if (error) {
      console.error('Error fetching pending registrations:', error)
      return
    }

    if (!pendingUsers || pendingUsers.length === 0) {
      return
    }

    // Process each pending user
    for (const user of pendingUsers) {
      const result = await registerUserInNeo4j(user.user_id)
      
      if (result.should_retry && user.attempt_count < MAX_RETRIES) {
        // Schedule retry with exponential backoff
        const delay = RETRY_DELAY_BASE * Math.pow(2, user.attempt_count)
        console.log(`Scheduling retry for user ${user.user_id} in ${delay}ms`)
        
        // Note: In production, you'd use a proper job queue
        // For now, we'll just mark it for retry
        const { error: updateError } = await supabase.rpc('update_neo4j_registration_status', {
          p_user_id: user.user_id,
          p_status: 'retry',
          p_error: result.message,
        })
        
        if (updateError) {
          console.error('Failed to update retry status:', updateError)
        }
      }
    }
  } catch (error) {
    console.error('Error processing pending registrations:', error)
  }
}

serve(async (req) => {
  // Handle CORS preflight
  if (req.method === 'OPTIONS') {
    return new Response('ok', { headers: corsHeaders })
  }

  try {
    // Try to parse JSON body
    let body: any
    try {
      body = await req.json()
    } catch (jsonError) {
      console.error('Invalid JSON in request body:', jsonError)
      return new Response(
        JSON.stringify({ 
          error: 'Invalid JSON',
          message: 'Request body must be valid JSON' 
        }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 400 
        }
      )
    }

    const { user_id, trigger_type } = body

    if (trigger_type === 'batch_process') {
      // Process all pending registrations
      await processPendingRegistrations()
      
      return new Response(
        JSON.stringify({ 
          status: 'success', 
          message: 'Batch processing initiated' 
        }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 200 
        }
      )
    }

    // Validate input
    if (!user_id) {
      return new Response(
        JSON.stringify({ error: 'user_id is required' }),
        { 
          headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          status: 400 
        }
      )
    }

    // Register single user
    const result = await registerUserInNeo4j(user_id)
    
    // Determine appropriate status code based on result
    let statusCode: number
    if (result.status === 'success') {
      statusCode = 200
    } else if (result.status === 'error' && !result.should_retry) {
      statusCode = 400 // Client error, won't retry
    } else if (result.status === 'retry' && result.should_retry) {
      statusCode = 503 // Service unavailable, will retry
    } else {
      statusCode = 500 // Other server error
    }
    
    return new Response(
      JSON.stringify(result),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: statusCode
      }
    )
  } catch (error) {
    console.error('Edge function error:', error)
    
    const errorMessage = normalizeError(error)
    
    return new Response(
      JSON.stringify({ 
        error: 'Internal server error',
        message: errorMessage 
      }),
      { 
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: 500
      }
    )
  }
})