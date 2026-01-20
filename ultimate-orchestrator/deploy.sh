#!/bin/bash

################################################################################
# QUICK CLOUD DEPLOYMENT SCRIPT
################################################################################
# This script deploys Ultimate Orchestrator to your Hostinger VPS
# Run this locally, it will automatically deploy to your cloud server
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo ""
echo "=============================================================================="
echo "  🚀 ULTIMATE ORCHESTRATOR - CLOUD DEPLOYMENT"
echo "=============================================================================="
echo ""

# Check if credentials are set
if [[ ! -f "CREDENTIALS.txt" ]]; then
    echo -e "${RED}ERROR:${NC} CREDENTIALS.txt not found!"
    echo "Please create and fill out CREDENTIALS.txt first"
    exit 1
fi

# Read server details
echo -e "${BLUE}[1/8]${NC} Reading server configuration..."

# Prompt for server details
read -p "Enter your Hostinger VPS IP address: " SERVER_IP
read -p "Enter SSH username (default: root): " SSH_USER
SSH_USER=${SSH_USER:-root}
read -p "Enter SSH port (default: 22): " SSH_PORT
SSH_PORT=${SSH_PORT:-22}

echo -e "${GREEN}✓${NC} Server: $SSH_USER@$SERVER_IP:$SSH_PORT"

# Test SSH connection
echo -e "${BLUE}[2/8]${NC} Testing SSH connection..."

if ssh -p $SSH_PORT -o ConnectTimeout=5 $SSH_USER@$SERVER_IP "echo 'SSH connection successful'" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} SSH connection successful"
else
    echo -e "${RED}ERROR:${NC} Cannot connect to server"
    echo "Please ensure:"
    echo "  1. Your VPS is running"
    echo "  2. SSH is enabled"
    echo "  3. Your SSH key is added or you have password access"
    exit 1
fi

# Create remote directory
echo -e "${BLUE}[3/8]${NC} Creating remote directory..."

ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "mkdir -p /opt/ultimate-orchestrator"

echo -e "${GREEN}✓${NC} Remote directory created"

# Copy files to server
echo -e "${BLUE}[4/8]${NC} Copying files to server..."

# Create temporary archive
tar -czf /tmp/orchestrator-deploy.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='venv' \
    --exclude='.git' \
    --exclude='node_modules' \
    *

# Upload archive
scp -P $SSH_PORT /tmp/orchestrator-deploy.tar.gz $SSH_USER@$SERVER_IP:/opt/ultimate-orchestrator/

# Extract on server
ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "cd /opt/ultimate-orchestrator && tar -xzf orchestrator-deploy.tar.gz && rm orchestrator-deploy.tar.gz"

# Clean up local archive
rm /tmp/orchestrator-deploy.tar.gz

echo -e "${GREEN}✓${NC} Files copied successfully"

# Upload credentials
echo -e "${BLUE}[5/8]${NC} Uploading credentials..."

scp -P $SSH_PORT CREDENTIALS.txt $SSH_USER@$SERVER_IP:/opt/ultimate-orchestrator/.env

echo -e "${GREEN}✓${NC} Credentials uploaded"

# Make install script executable
ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "chmod +x /opt/ultimate-orchestrator/install.sh"

# Run installation
echo -e "${BLUE}[6/8]${NC} Running automated installation..."
echo ""
echo "This will:"
echo "  - Install system dependencies"
echo "  - Install Docker & Docker Compose"
echo "  - Set up Python environment"
echo "  - Configure Nginx"
echo "  - Set up SSL (if domain provided)"
echo "  - Start the service"
echo ""
read -p "Continue with installation? (y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Deployment cancelled"
    exit 1
fi

# Run installation on server
ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "cd /opt/ultimate-orchestrator && bash install.sh"

echo -e "${GREEN}✓${NC} Installation complete"

# Start services
echo -e "${BLUE}[7/8]${NC} Starting services..."

ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose up -d"

echo -e "${GREEN}✓${NC} Services started"

# Get service status
echo -e "${BLUE}[8/8]${NC} Checking service status..."

ssh -p $SSH_PORT $SSH_USER@$SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose ps"

echo ""
echo "=============================================================================="
echo -e "${GREEN}  ✅ DEPLOYMENT COMPLETE!${NC}"
echo "=============================================================================="
echo ""
echo "Your Ultimate Orchestrator is now running at:"
echo ""
echo "  HTTP:  http://$SERVER_IP:8000"
echo "  HTTPS: https://$SERVER_IP (if SSL configured)"
echo ""
echo "API Documentation:"
echo "  http://$SERVER_IP:8000/docs"
echo ""
echo "Useful Commands:"
echo ""
echo "  # Check logs"
echo "  ssh -p $SSH_PORT $SSH_USER@$SERVER_IP 'cd /opt/ultimate-orchestrator && docker-compose logs -f'"
echo ""
echo "  # Restart services"
echo "  ssh -p $SSH_PORT $SSH_USER@$SERVER_IP 'cd /opt/ultimate-orchestrator && docker-compose restart'"
echo ""
echo "  # Stop services"
echo "  ssh -p $SSH_PORT $SSH_USER@$SERVER_IP 'cd /opt/ultimate-orchestrator && docker-compose down'"
echo ""
echo "  # Update deployment"
echo "  bash deploy.sh"
echo ""
echo "=============================================================================="
echo ""

# Ask if user wants to open browser
read -p "Open API documentation in browser? (y/n): " -n 1 -r
echo

if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v xdg-open &> /dev/null; then
        xdg-open "http://$SERVER_IP:8000/docs"
    elif command -v open &> /dev/null; then
        open "http://$SERVER_IP:8000/docs"
    else
        echo "Please open http://$SERVER_IP:8000/docs in your browser"
    fi
fi

echo ""
echo "🎉 Enjoy your Ultimate Orchestrator!"
echo ""
