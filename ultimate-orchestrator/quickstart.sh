#!/bin/bash

################################################################################
# ULTIMATE ORCHESTRATOR - QUICK START
################################################################################
# One-command local testing before Hostinger deployment
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo "=============================================================================="
echo -e "${CYAN}  🚀 ULTIMATE ORCHESTRATOR - QUICK START${NC}"
echo "=============================================================================="
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file not found!${NC}"
    echo "Please create .env file with your API keys."
    echo "See HOSTINGER_DEPLOYMENT.md for required variables."
    exit 1
fi

echo -e "${BLUE}[1/5]${NC} Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
echo -e "${GREEN}✓${NC} Python $python_version detected"

echo -e "${BLUE}[2/5]${NC} Installing dependencies..."
pip install -q -r requirements.txt
echo -e "${GREEN}✓${NC} Dependencies installed"

echo -e "${BLUE}[3/5]${NC} Checking environment variables..."
source .env
required_vars=("OPENAI_API_KEY" "SUPABASE_URL" "QDRANT_URL")
missing_vars=()

for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        missing_vars+=("$var")
    fi
done

if [ ${#missing_vars[@]} -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Missing required variables: ${missing_vars[*]}${NC}"
    echo "Application may have limited functionality."
else
    echo -e "${GREEN}✓${NC} All critical variables present"
fi

echo -e "${BLUE}[4/5]${NC} Starting Ultimate Orchestrator..."
echo ""
echo "=============================================================================="
echo -e "${CYAN}  Starting Server...${NC}"
echo "=============================================================================="
echo ""
echo "  Frontend:        http://localhost:8000/"
echo "  API Docs:        http://localhost:8000/docs"
echo "  Health Check:    http://localhost:8000/health"
echo "  Voice WebSocket: ws://localhost:8000/ws/voice/{user_id}"
echo ""
echo "=============================================================================="
echo ""
echo -e "${GREEN}Press Ctrl+C to stop the server${NC}"
echo ""

# Start the server
python api_server.py
