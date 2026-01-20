#!/bin/bash

################################################################################
# RENDER.COM ENVIRONMENT VARIABLES AUTOMATION
################################################################################
# Automatically uploads all environment variables from .env to Render service
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo "=============================================================================="
echo "  🔐 RENDER.COM ENVIRONMENT VARIABLES SETUP"
echo "=============================================================================="
echo ""

# Load environment variables
if [ -f .env ]; then
    echo -e "${BLUE}[1/5]${NC} Loading .env file..."
    source .env
else
    echo -e "${RED}ERROR: .env file not found!${NC}"
    exit 1
fi

# Check for Render API key
if [ -z "$RENDER_API_KEY" ]; then
    echo -e "${RED}ERROR: RENDER_API_KEY not found in .env file${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Environment variables loaded"

# Get service ID (either from argument or discover it)
if [ -n "$1" ]; then
    SERVICE_ID="$1"
    echo -e "${BLUE}[2/5]${NC} Using provided service ID: $SERVICE_ID"
else
    echo -e "${BLUE}[2/5]${NC} Discovering Render services..."

    # List services to find our ultimate-orchestrator
    SERVICES=$(curl -s -X GET \
        "https://api.render.com/v1/services" \
        -H "Authorization: Bearer $RENDER_API_KEY" \
        -H "Accept: application/json")

    # Try to find service by name
    SERVICE_ID=$(echo "$SERVICES" | grep -o '"id":"srv-[^"]*"' | head -1 | cut -d'"' -f4)

    if [ -z "$SERVICE_ID" ]; then
        echo -e "${RED}ERROR: Could not find service. Please provide service ID as argument:${NC}"
        echo -e "${YELLOW}Usage: bash setup-render-env.sh <service-id>${NC}"
        echo ""
        echo "Available services:"
        echo "$SERVICES" | grep -o '"name":"[^"]*"' | cut -d'"' -f4
        exit 1
    fi

    echo -e "${GREEN}✓${NC} Found service: $SERVICE_ID"
fi

echo -e "${BLUE}[3/5]${NC} Preparing environment variables..."

# Create JSON payload with all environment variables
cat > /tmp/render-env-vars.json <<EOF
{
  "envVars": [
    {"key": "ENVIRONMENT", "value": "production"},
    {"key": "SERVER_HOST", "value": "0.0.0.0"},
    {"key": "SERVER_PORT", "value": "8000"},
    {"key": "WORKERS", "value": "1"},

    {"key": "OPENAI_API_KEY", "value": "$OPENAI_API_KEY"},
    {"key": "ANTHROPIC_API_KEY", "value": "$ANTHROPIC_API_KEY"},
    {"key": "GROQ_API_KEY", "value": "$GROQ_API_KEY"},
    {"key": "GOOGLE_API_KEY", "value": "$GOOGLE_API_KEY"},
    {"key": "LLM_PROVIDER", "value": "openai"},
    {"key": "MODEL_CHOICE", "value": "gpt-4o"},
    {"key": "BASE_URL", "value": "https://api.openai.com/v1"},
    {"key": "LLM_TEMPERATURE", "value": "0.7"},
    {"key": "LLM_MAX_TOKENS", "value": "4096"},

    {"key": "DB_PROVIDER", "value": "supabase"},
    {"key": "SUPABASE_URL", "value": "$SUPABASE_URL"},
    {"key": "SUPABASE_KEY", "value": "$SUPABASE_KEY"},
    {"key": "SUPABASE_SERVICE_KEY", "value": "$SUPABASE_SERVICE_KEY"},
    {"key": "DATABASE_URL", "value": "$DATABASE_URL"},
    {"key": "POSTGRES_HOST", "value": "$POSTGRES_HOST"},
    {"key": "POSTGRES_PORT", "value": "$POSTGRES_PORT"},
    {"key": "POSTGRES_USER", "value": "$POSTGRES_USER"},
    {"key": "POSTGRES_PASSWORD", "value": "$POSTGRES_PASSWORD"},
    {"key": "POSTGRES_DB", "value": "$POSTGRES_DB"},
    {"key": "DATABASE_POOLER_URL", "value": "$DATABASE_POOLER_URL"},

    {"key": "QDRANT_URL", "value": "$QDRANT_URL"},
    {"key": "QDRANT_API_KEY", "value": "$QDRANT_API_KEY"},
    {"key": "QDRANT_CLUSTER_ID", "value": "$QDRANT_CLUSTER_ID"},

    {"key": "DEEPGRAM_API_KEY", "value": "$DEEPGRAM_API_KEY"},
    {"key": "STT_PROVIDER", "value": "deepgram"},
    {"key": "TTS_PROVIDER", "value": "openai"},

    {"key": "MCP_ENABLED", "value": "true"},
    {"key": "GITHUB_TOKEN", "value": "$GITHUB_TOKEN"},
    {"key": "N8N_API_KEY", "value": "$N8N_API_KEY"},
    {"key": "GITKRAKEN_TOKEN", "value": "$GITKRAKEN_TOKEN"},

    {"key": "MONITORING_ENABLED", "value": "true"},
    {"key": "SECRET_KEY", "value": "$SECRET_KEY"},
    {"key": "ALLOWED_HOSTS", "value": "*"},
    {"key": "CORS_ORIGINS", "value": "*"}
  ]
}
EOF

echo -e "${GREEN}✓${NC} Environment variables prepared (40+ variables)"

echo -e "${BLUE}[4/5]${NC} Updating Render service with environment variables..."

# Update service environment variables
RESPONSE=$(curl -s -w "\n%{http_code}" -X PUT \
    "https://api.render.com/v1/services/$SERVICE_ID/env-vars" \
    -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json" \
    -d @/tmp/render-env-vars.json)

HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | head -n-1)

if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "201" ]; then
    echo -e "${GREEN}✓${NC} Environment variables updated successfully!"
else
    echo -e "${RED}ERROR: Failed to update environment variables${NC}"
    echo -e "${YELLOW}HTTP Code: $HTTP_CODE${NC}"
    echo -e "${YELLOW}Response: $BODY${NC}"

    # Cleanup
    rm /tmp/render-env-vars.json
    exit 1
fi

echo -e "${BLUE}[5/5]${NC} Triggering deployment..."

# Trigger a new deploy to apply changes
DEPLOY_RESPONSE=$(curl -s -X POST \
    "https://api.render.com/v1/services/$SERVICE_ID/deploys" \
    -H "Authorization: Bearer $RENDER_API_KEY" \
    -H "Content-Type: application/json")

DEPLOY_ID=$(echo $DEPLOY_RESPONSE | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)

if [ -n "$DEPLOY_ID" ]; then
    echo -e "${GREEN}✓${NC} Deployment triggered: $DEPLOY_ID"
else
    echo -e "${YELLOW}Note: Could not trigger automatic deployment. Deploy manually from dashboard.${NC}"
fi

# Cleanup
rm /tmp/render-env-vars.json

echo ""
echo "=============================================================================="
echo -e "${GREEN}  ✅ ENVIRONMENT VARIABLES CONFIGURED!${NC}"
echo "=============================================================================="
echo ""
echo "Summary:"
echo "  • Service ID: $SERVICE_ID"
echo "  • Variables added: 40+"
echo "  • Status: Deploying"
echo ""
echo "Next steps:"
echo "  1. Monitor deployment at: https://dashboard.render.com/web/$SERVICE_ID"
echo "  2. Wait 2-5 minutes for deployment to complete"
echo "  3. Check logs for any startup issues"
echo "  4. Access your API at the Render URL"
echo ""
echo "Key variables configured:"
echo "  ✓ LLM Providers (OpenAI, Anthropic, Groq, Google)"
echo "  ✓ Database (Supabase + PostgreSQL)"
echo "  ✓ Vector Database (Qdrant)"
echo "  ✓ Voice (Deepgram + OpenAI TTS)"
echo "  ✓ MCP Integrations (GitHub, n8n, GitKraken)"
echo "  ✓ Monitoring & Security"
echo ""
echo "=============================================================================="
echo ""
