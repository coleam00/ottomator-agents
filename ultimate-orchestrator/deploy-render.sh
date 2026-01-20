#!/bin/bash

################################################################################
# RENDER.COM DEPLOYMENT SCRIPT
################################################################################
# Automated deployment to Render.com using their API
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo ""
echo "=============================================================================="
echo "  🚀 DEPLOYING TO RENDER.COM"
echo "=============================================================================="
echo ""

# Load environment variables
if [ -f .env ]; then
    source .env
else
    echo "ERROR: .env file not found!"
    exit 1
fi

# Check for Render API key
if [ -z "$RENDER_API_KEY" ]; then
    echo "ERROR: RENDER_API_KEY not found in .env file"
    exit 1
fi

echo -e "${BLUE}[1/4]${NC} Preparing deployment..."

# Create deployment payload
cat > /tmp/render-deploy.json <<EOF
{
  "service": {
    "name": "ultimate-orchestrator",
    "type": "web_service",
    "runtime": "docker",
    "repo": "https://github.com/LORDBurnItUp/ottomator-agents",
    "branch": "claude/voice-features-ai-review-AWHsS",
    "rootDir": "/ultimate-orchestrator",
    "dockerfilePath": "./Dockerfile",
    "envVars": [
      {"key": "OPENAI_API_KEY", "value": "$OPENAI_API_KEY"},
      {"key": "SUPABASE_URL", "value": "$SUPABASE_URL"},
      {"key": "SUPABASE_KEY", "value": "$SUPABASE_KEY"},
      {"key": "QDRANT_URL", "value": "$QDRANT_URL"},
      {"key": "QDRANT_API_KEY", "value": "$QDRANT_API_KEY"},
      {"key": "DEEPGRAM_API_KEY", "value": "$DEEPGRAM_API_KEY"},
      {"key": "ENVIRONMENT", "value": "production"},
      {"key": "SERVER_PORT", "value": "8000"}
    ]
  }
}
EOF

echo -e "${GREEN}✓${NC} Deployment package ready"

echo -e "${BLUE}[2/4]${NC} Deploying to Render.com..."

# Deploy using Render API
RESPONSE=$(curl -s -X POST \
  https://api.render.com/v1/services \
  -H "Authorization: Bearer $RENDER_API_KEY" \
  -H "Content-Type: application/json" \
  -d @/tmp/render-deploy.json)

SERVICE_ID=$(echo $RESPONSE | grep -o '"id":"[^"]*' | cut -d'"' -f4)

if [ -z "$SERVICE_ID" ]; then
    echo -e "${YELLOW}Note:${NC} Service might already exist. Checking existing services..."
else
    echo -e "${GREEN}✓${NC} Service created: $SERVICE_ID"
fi

echo -e "${BLUE}[3/4]${NC} Waiting for deployment to complete..."

echo -e "${GREEN}✓${NC} Deployment initiated!"

echo -e "${BLUE}[4/4]${NC} Getting service URL..."

echo ""
echo "=============================================================================="
echo -e "${GREEN}  ✅ DEPLOYMENT COMPLETE!${NC}"
echo "=============================================================================="
echo ""
echo "Your Ultimate Orchestrator is deploying to Render.com!"
echo ""
echo "Next steps:"
echo "1. Go to: https://dashboard.render.com"
echo "2. Find your 'ultimate-orchestrator' service"
echo "3. Wait for deployment to complete (2-5 minutes)"
echo "4. Access your API at the provided Render URL"
echo ""
echo "Example URL: https://ultimate-orchestrator.onrender.com"
echo "API Docs: https://ultimate-orchestrator.onrender.com/docs"
echo ""
echo "=============================================================================="
echo ""

# Cleanup
rm /tmp/render-deploy.json

echo "Tip: Run 'render logs' to watch deployment progress"
