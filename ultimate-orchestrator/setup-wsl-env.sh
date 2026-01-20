#!/bin/bash

################################################################################
# ULTIMATE ORCHESTRATOR - ENVIRONMENT SETUP FOR WSL/KALI
################################################################################
# Quick environment configuration for fresh Windows 11 + WSL setup
################################################################################

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

clear

echo ""
echo "=============================================================================="
echo -e "${CYAN}  🚀 ULTIMATE ORCHESTRATOR - WSL ENVIRONMENT SETUP${NC}"
echo "=============================================================================="
echo ""
echo -e "${YELLOW}This script will help you configure your environment variables.${NC}"
echo ""

# Navigate to the right directory
cd /home/user/ottomator-agents/ultimate-orchestrator

echo -e "${BLUE}[1/3]${NC} Checking current directory..."
echo "   📁 Current path: $(pwd)"
echo ""

# Check if .env already exists
if [ -f .env ]; then
    echo -e "${YELLOW}⚠️  .env file already exists!${NC}"
    echo ""
    read -p "Do you want to overwrite it? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${GREEN}✓${NC} Keeping existing .env file"
        echo ""
        echo "Your environment is already configured!"
        echo "Run: ${CYAN}bash quickstart.sh${NC} to start the server"
        exit 0
    fi
fi

echo -e "${BLUE}[2/3]${NC} Creating .env file..."

# Create basic .env with placeholders
cat > .env << 'EOF'
################################################################################
# ULTIMATE ORCHESTRATOR - ENVIRONMENT VARIABLES
################################################################################

# ===================
# LLM PROVIDERS
# ===================
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GROQ_API_KEY=your_groq_key_here
GOOGLE_API_KEY=your_google_key_here

LLM_PROVIDER=openai
MODEL_CHOICE=gpt-4o
BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# ===================
# DATABASES
# ===================
DB_PROVIDER=supabase

# Supabase
SUPABASE_URL=your_supabase_url_here
SUPABASE_KEY=your_supabase_anon_key_here
SUPABASE_SERVICE_KEY=your_supabase_service_key_here

# PostgreSQL (from Supabase)
DATABASE_URL=your_database_url_here
POSTGRES_HOST=your_postgres_host_here
POSTGRES_PORT=5432
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_postgres_password_here
POSTGRES_DB=postgres
DATABASE_POOLER_URL=your_pooler_url_here

# Qdrant Vector Database
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_CLUSTER_ID=your_cluster_id_here

# ===================
# VOICE
# ===================
DEEPGRAM_API_KEY=your_deepgram_key_here
STT_PROVIDER=deepgram
TTS_PROVIDER=openai

# LiveKit (Optional)
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# ===================
# MCP INTEGRATIONS
# ===================
MCP_ENABLED=true
GITHUB_TOKEN=your_github_token_here
N8N_API_KEY=your_n8n_key_here
GITKRAKEN_TOKEN=your_gitkraken_token_here

# ===================
# HOSTINGER
# ===================
HOSTINGER_SSH_HOST=46.202.197.97
HOSTINGER_SSH_PORT=65002
HOSTINGER_SSH_USER=u472811699
HOSTINGER_SSH_PASSWORD=your_hostinger_password_here
HOSTINGER_DOMAIN=your_domain_here

# ===================
# SERVER CONFIG
# ===================
ENVIRONMENT=production
SERVER_HOST=0.0.0.0
PORT=8000
SERVER_PORT=8000
WORKERS=1

# ===================
# SECURITY
# ===================
SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || echo "change-this-secret-key-$(date +%s)")
ALLOWED_HOSTS=*
CORS_ORIGINS=*

# ===================
# MONITORING
# ===================
MONITORING_ENABLED=true
EOF

echo -e "${GREEN}✓${NC} Created .env file with placeholders"
echo ""

echo -e "${BLUE}[3/3]${NC} Next steps:"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}OPTION 1: Edit the .env file with your credentials${NC}"
echo ""
echo "  Open the file:"
echo -e "  ${YELLOW}nano .env${NC}"
echo ""
echo "  Replace these placeholders with your actual keys:"
echo "    • your_openai_key_here"
echo "    • your_supabase_url_here"
echo "    • your_qdrant_url_here"
echo "    • your_deepgram_key_here"
echo "    • etc..."
echo ""
echo "  Save: Ctrl+O, Enter, Ctrl+X"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}OPTION 2: Use VS Code (easier on Windows)${NC}"
echo ""
echo "  From Windows, navigate to WSL:"
echo -e "  ${YELLOW}\\\\wsl\$\\Kali-linux\\home\\user\\ottomator-agents\\ultimate-orchestrator\\.env${NC}"
echo ""
echo "  Or use VS Code:"
echo -e "  ${YELLOW}code .env${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${CYAN}AFTER EDITING:${NC}"
echo ""
echo "  Test locally:"
echo -e "  ${YELLOW}bash quickstart.sh${NC}"
echo ""
echo "  Deploy to Hostinger:"
echo -e "  ${YELLOW}python3 deploy_hostinger.py${NC}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}✓ Environment setup complete!${NC}"
echo ""
