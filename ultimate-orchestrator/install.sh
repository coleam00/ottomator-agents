#!/bin/bash

################################################################################
# Ultimate Orchestrator - Automated Installation Script
################################################################################
# This script automates the complete installation and setup of the
# Ultimate Orchestrator system on any Linux server (local or cloud)
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo ""
    echo "=============================================================================="
    echo "       🚀 ULTIMATE ORCHESTRATOR - AUTOMATED INSTALLATION 🚀"
    echo "=============================================================================="
    echo "  Cloud-Ready AI Agent Orchestration Platform"
    echo "  Version: 1.0.0"
    echo "=============================================================================="
    echo ""
}

# Check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root. This is not recommended for production."
        read -p "Continue anyway? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Detect OS
detect_os() {
    log_info "Detecting operating system..."

    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        OS=$ID
        VERSION=$VERSION_ID
        log_success "Detected: $PRETTY_NAME"
    else
        log_error "Cannot detect OS. /etc/os-release not found."
        exit 1
    fi
}

# Install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."

    case $OS in
        ubuntu|debian)
            sudo apt-get update
            sudo apt-get install -y \
                python3 \
                python3-pip \
                python3-venv \
                git \
                curl \
                wget \
                build-essential \
                libssl-dev \
                libffi-dev \
                python3-dev \
                nodejs \
                npm \
                nginx \
                supervisor \
                redis-server \
                certbot \
                python3-certbot-nginx
            ;;
        centos|rhel|fedora)
            sudo yum install -y \
                python3 \
                python3-pip \
                git \
                curl \
                wget \
                gcc \
                openssl-devel \
                bffi-devel \
                python3-devel \
                nodejs \
                npm \
                nginx \
                supervisor \
                redis \
                certbot \
                python3-certbot-nginx
            ;;
        *)
            log_error "Unsupported OS: $OS"
            log_info "Please install dependencies manually:"
            log_info "  - Python 3.10+"
            log_info "  - pip"
            log_info "  - Node.js 18+"
            log_info "  - npm"
            log_info "  - nginx"
            log_info "  - supervisor"
            log_info "  - redis"
            exit 1
            ;;
    esac

    log_success "System dependencies installed"
}

# Install Docker
install_docker() {
    log_info "Installing Docker..."

    if command -v docker &> /dev/null; then
        log_success "Docker already installed: $(docker --version)"
        return
    fi

    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh

    log_success "Docker installed"
    log_warning "You may need to log out and back in for Docker permissions to take effect"
}

# Install Docker Compose
install_docker_compose() {
    log_info "Installing Docker Compose..."

    if command -v docker-compose &> /dev/null; then
        log_success "Docker Compose already installed: $(docker-compose --version)"
        return
    fi

    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose

    log_success "Docker Compose installed"
}

# Create project directory
create_project_directory() {
    log_info "Setting up project directory..."

    PROJECT_DIR="${1:-/opt/ultimate-orchestrator}"

    if [[ -d "$PROJECT_DIR" ]]; then
        log_warning "Directory $PROJECT_DIR already exists"
        read -p "Remove and recreate? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo rm -rf "$PROJECT_DIR"
        else
            log_info "Using existing directory"
        fi
    fi

    sudo mkdir -p "$PROJECT_DIR"
    sudo chown -R $USER:$USER "$PROJECT_DIR"

    log_success "Project directory created: $PROJECT_DIR"
    echo "$PROJECT_DIR"
}

# Create Python virtual environment
create_virtualenv() {
    log_info "Creating Python virtual environment..."

    cd "$PROJECT_DIR"
    python3 -m venv venv
    source venv/bin/activate

    pip install --upgrade pip setuptools wheel

    log_success "Virtual environment created"
}

# Install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."

    cd "$PROJECT_DIR"
    source venv/bin/activate

    # Copy requirements.txt to project directory
    if [[ -f "requirements.txt" ]]; then
        pip install -r requirements.txt
        log_success "Python dependencies installed"
    else
        log_warning "requirements.txt not found. Skipping Python dependencies."
    fi
}

# Setup environment configuration
setup_environment() {
    log_info "Setting up environment configuration..."

    cd "$PROJECT_DIR"

    # Interactive configuration
    echo ""
    echo "=============================================================================="
    echo "  ENVIRONMENT CONFIGURATION"
    echo "=============================================================================="
    echo ""

    # Create .env file
    cat > .env << 'EOF'
# Ultimate Orchestrator Environment Configuration
# Generated by automated installer

# ===================
# LLM Configuration
# ===================
LLM_PROVIDER=openai
MODEL_CHOICE=gpt-4o
OPENAI_API_KEY=
BASE_URL=https://api.openai.com/v1
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=4096

# ===================
# Database Configuration
# ===================
DB_PROVIDER=supabase

# Supabase
SUPABASE_URL=
SUPABASE_KEY=

# MySQL
MYSQL_HOST=
MYSQL_PORT=3306
MYSQL_USER=
MYSQL_PASSWORD=
MYSQL_DATABASE=

# ===================
# Voice Configuration
# ===================
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=

TTS_PROVIDER=openai

# LiveKit
LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=

# ===================
# MCP Servers
# ===================
MCP_ENABLED=true

# Hostinger MCP
HOSTINGER_API_KEY=
HOSTINGER_API_URL=https://api.hostinger.com/v1

# Other MCP Servers
GITHUB_TOKEN=
SLACK_BOT_TOKEN=
BRAVE_API_KEY=

# ===================
# Monitoring
# ===================
MONITORING_ENABLED=true
LANGFUSE_ENABLED=false

# ===================
# Server Configuration
# ===================
ENVIRONMENT=production
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
WORKERS=4

# ===================
# Security
# ===================
SECRET_KEY=$(openssl rand -hex 32)
ALLOWED_HOSTS=*
CORS_ORIGINS=*
EOF

    log_success ".env file created"
    log_info "Please edit .env file with your actual API keys and configuration"
}

# Interactive configuration wizard
run_configuration_wizard() {
    log_info "Starting interactive configuration wizard..."

    echo ""
    echo "=============================================================================="
    echo "  CONFIGURATION WIZARD"
    echo "=============================================================================="
    echo ""

    # OpenAI API Key
    read -p "Enter your OpenAI API Key (or press Enter to skip): " OPENAI_KEY
    if [[ -n "$OPENAI_KEY" ]]; then
        sed -i "s/OPENAI_API_KEY=.*/OPENAI_API_KEY=$OPENAI_KEY/" .env
    fi

    # Supabase
    echo ""
    log_info "Supabase Configuration"
    read -p "Enter your Supabase URL (or press Enter to skip): " SUPABASE_URL_INPUT
    if [[ -n "$SUPABASE_URL_INPUT" ]]; then
        sed -i "s|SUPABASE_URL=.*|SUPABASE_URL=$SUPABASE_URL_INPUT|" .env
    fi

    read -p "Enter your Supabase API Key (or press Enter to skip): " SUPABASE_KEY_INPUT
    if [[ -n "$SUPABASE_KEY_INPUT" ]]; then
        sed -i "s/SUPABASE_KEY=.*/SUPABASE_KEY=$SUPABASE_KEY_INPUT/" .env
    fi

    # MySQL
    echo ""
    log_info "MySQL Configuration"
    read -p "Configure MySQL? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -p "MySQL Host: " MYSQL_HOST_INPUT
        read -p "MySQL User: " MYSQL_USER_INPUT
        read -s -p "MySQL Password: " MYSQL_PASSWORD_INPUT
        echo
        read -p "MySQL Database: " MYSQL_DATABASE_INPUT

        sed -i "s/MYSQL_HOST=.*/MYSQL_HOST=$MYSQL_HOST_INPUT/" .env
        sed -i "s/MYSQL_USER=.*/MYSQL_USER=$MYSQL_USER_INPUT/" .env
        sed -i "s/MYSQL_PASSWORD=.*/MYSQL_PASSWORD=$MYSQL_PASSWORD_INPUT/" .env
        sed -i "s/MYSQL_DATABASE=.*/MYSQL_DATABASE=$MYSQL_DATABASE_INPUT/" .env
    fi

    # Hostinger
    echo ""
    log_info "Hostinger Configuration"
    read -p "Enter your Hostinger API Key (or press Enter to skip): " HOSTINGER_KEY
    if [[ -n "$HOSTINGER_KEY" ]]; then
        sed -i "s/HOSTINGER_API_KEY=.*/HOSTINGER_API_KEY=$HOSTINGER_KEY/" .env
    fi

    # Deepgram
    echo ""
    log_info "Voice Configuration (Optional)"
    read -p "Enter your Deepgram API Key for voice (or press Enter to skip): " DEEPGRAM_KEY
    if [[ -n "$DEEPGRAM_KEY" ]]; then
        sed -i "s/DEEPGRAM_API_KEY=.*/DEEPGRAM_API_KEY=$DEEPGRAM_KEY/" .env
    fi

    log_success "Configuration wizard completed"
}

# Create systemd service
create_systemd_service() {
    log_info "Creating systemd service..."

    cat | sudo tee /etc/systemd/system/ultimate-orchestrator.service > /dev/null << EOF
[Unit]
Description=Ultimate Orchestrator AI Agent System
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
Environment="PATH=$PROJECT_DIR/venv/bin"
ExecStart=$PROJECT_DIR/venv/bin/python meta_orchestrator.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl daemon-reload
    sudo systemctl enable ultimate-orchestrator.service

    log_success "Systemd service created"
}

# Setup Nginx reverse proxy
setup_nginx() {
    log_info "Setting up Nginx reverse proxy..."

    read -p "Enter your domain name (or press Enter to use IP address): " DOMAIN

    if [[ -z "$DOMAIN" ]]; then
        DOMAIN=$(curl -s ifconfig.me)
        log_info "Using IP address: $DOMAIN"
    fi

    cat | sudo tee /etc/nginx/sites-available/ultimate-orchestrator > /dev/null << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /ws {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "Upgrade";
        proxy_set_header Host \$host;
    }
}
EOF

    sudo ln -sf /etc/nginx/sites-available/ultimate-orchestrator /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx

    log_success "Nginx configured"

    # Ask about SSL
    echo ""
    read -p "Would you like to set up SSL with Let's Encrypt? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] && [[ -n "$DOMAIN" ]] && [[ "$DOMAIN" != *.*.*.* ]]; then
        read -p "Enter email for SSL certificate: " EMAIL
        sudo certbot --nginx -d "$DOMAIN" --non-interactive --agree-tos -m "$EMAIL"
        log_success "SSL certificate installed"
    fi
}

# Setup firewall
setup_firewall() {
    log_info "Setting up firewall..."

    if command -v ufw &> /dev/null; then
        sudo ufw allow 22/tcp   # SSH
        sudo ufw allow 80/tcp   # HTTP
        sudo ufw allow 443/tcp  # HTTPS
        sudo ufw allow 8000/tcp # Application

        # Enable UFW with confirmation
        echo "y" | sudo ufw enable

        log_success "Firewall configured"
    else
        log_warning "UFW not found. Please configure firewall manually."
    fi
}

# Create startup script
create_startup_script() {
    log_info "Creating startup script..."

    cat > "$PROJECT_DIR/start.sh" << 'EOF'
#!/bin/bash

# Ultimate Orchestrator Startup Script

cd "$(dirname "$0")"
source venv/bin/activate

echo "Starting Ultimate Orchestrator..."

# Start the meta orchestrator
python meta_orchestrator.py
EOF

    chmod +x "$PROJECT_DIR/start.sh"

    log_success "Startup script created"
}

# Print completion message
print_completion() {
    echo ""
    echo "=============================================================================="
    echo "  ✅ INSTALLATION COMPLETE!"
    echo "=============================================================================="
    echo ""
    log_success "Ultimate Orchestrator has been installed successfully!"
    echo ""
    echo "Installation Directory: $PROJECT_DIR"
    echo ""
    echo "📋 NEXT STEPS:"
    echo ""
    echo "1. Edit configuration:"
    echo "   nano $PROJECT_DIR/.env"
    echo ""
    echo "2. Start the service:"
    echo "   sudo systemctl start ultimate-orchestrator"
    echo ""
    echo "3. Check status:"
    echo "   sudo systemctl status ultimate-orchestrator"
    echo ""
    echo "4. View logs:"
    echo "   sudo journalctl -u ultimate-orchestrator -f"
    echo ""
    echo "5. Access your instance:"
    if [[ -n "$DOMAIN" ]]; then
        echo "   https://$DOMAIN"
    else
        echo "   http://$(curl -s ifconfig.me):8000"
    fi
    echo ""
    echo "📚 DOCUMENTATION:"
    echo "   - README: $PROJECT_DIR/README.md"
    echo "   - Features: $PROJECT_DIR/FEATURES.md"
    echo "   - Deployment: $PROJECT_DIR/DEPLOYMENT.md"
    echo ""
    echo "🎯 QUICK START:"
    echo "   cd $PROJECT_DIR"
    echo "   source venv/bin/activate"
    echo "   python meta_orchestrator.py"
    echo ""
    echo "=============================================================================="
    echo ""
}

# Main installation flow
main() {
    print_banner
    check_root
    detect_os

    echo ""
    read -p "Install system dependencies? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_dependencies
    fi

    echo ""
    read -p "Install Docker? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_docker
        install_docker_compose
    fi

    echo ""
    read -p "Enter installation directory (default: /opt/ultimate-orchestrator): " INSTALL_DIR
    PROJECT_DIR=$(create_project_directory "${INSTALL_DIR:-/opt/ultimate-orchestrator}")

    # Copy current files to project directory
    if [[ -d "$(dirname "$0")" ]]; then
        log_info "Copying project files..."
        cp -r "$(dirname "$0")"/* "$PROJECT_DIR/"
        log_success "Project files copied"
    fi

    create_virtualenv
    install_python_dependencies
    setup_environment

    echo ""
    read -p "Run configuration wizard? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_configuration_wizard
    fi

    create_startup_script

    echo ""
    read -p "Create systemd service? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        create_systemd_service
    fi

    echo ""
    read -p "Setup Nginx reverse proxy? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_nginx
    fi

    echo ""
    read -p "Configure firewall? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        setup_firewall
    fi

    print_completion
}

# Run main installation
main "$@"
