# 🚀 Ultimate Orchestrator - Cloud Deployment Guide

## Quick Start (5 Minutes!)

### Step 1: Fill in Your Credentials

1. Open `CREDENTIALS.txt`
2. Replace ALL `PASTE_YOUR_...` placeholders with your actual API keys
3. Save the file

**Required Credentials:**
- ✅ OpenAI API Key (REQUIRED)
- ✅ Hostinger VPS IP and SSH credentials
- ✅ At least ONE database: Supabase OR MySQL

**Optional but Recommended:**
- Deepgram API Key (for voice)
- Qdrant URL and API Key (for vector search)
- Domain name (for SSL)

### Step 2: Deploy to Cloud

```bash
# Make deployment script executable
chmod +x deploy.sh

# Run deployment
bash deploy.sh
```

The script will ask you for:
- Your Hostinger VPS IP address
- SSH username (usually `root`)
- SSH port (usually `22`)

Then it will automatically:
- ✅ Copy all files to your server
- ✅ Install all dependencies
- ✅ Configure databases
- ✅ Set up Nginx reverse proxy
- ✅ Install SSL certificate (if domain provided)
- ✅ Start all services

### Step 3: Access Your System

Once deployed, access at:
- **API**: `http://YOUR_SERVER_IP:8000`
- **Documentation**: `http://YOUR_SERVER_IP:8000/docs`
- **Monitoring**: `http://YOUR_SERVER_IP:8000/api/v1/monitoring/dashboard`

---

## Manual Installation (Alternative)

If you prefer manual control:

### On Your Local Machine

1. **Prepare credentials:**
```bash
# Edit CREDENTIALS.txt with your API keys
nano CREDENTIALS.txt
```

2. **Copy files to server:**
```bash
scp -r ultimate-orchestrator root@YOUR_SERVER_IP:/opt/
```

### On Your Hostinger VPS

1. **SSH into server:**
```bash
ssh root@YOUR_SERVER_IP
```

2. **Navigate to directory:**
```bash
cd /opt/ultimate-orchestrator
```

3. **Copy credentials:**
```bash
# Copy from local CREDENTIALS.txt
nano .env
# Paste all content
# Save: Ctrl+O, Enter, Ctrl+X
```

4. **Run installation:**
```bash
chmod +x install.sh
bash install.sh
```

5. **Start services:**
```bash
docker-compose up -d
```

---

## Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down
```

### Individual Docker Container

```bash
# Build image
docker build -t ultimate-orchestrator .

# Run container
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name orchestrator \
  ultimate-orchestrator
```

---

## Configuration

### Environment Variables

All configuration is in `.env` file. Key settings:

```bash
# LLM Provider
OPENAI_API_KEY=your-key-here
MODEL_CHOICE=gpt-4o

# Databases
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

MYSQL_HOST=your-mysql-host
MYSQL_USER=your-username
MYSQL_PASSWORD=your-password

# Qdrant Vector DB
QDRANT_URL=https://your-cluster.qdrant.io
QDRANT_API_KEY=your-api-key

# Voice (Optional)
DEEPGRAM_API_KEY=your-key
LIVEKIT_URL=wss://your-app.livekit.cloud

# Hostinger
HOSTINGER_API_KEY=your-key
HOSTINGER_SERVER_IP=your-vps-ip
```

### Database Setup

#### Supabase

1. Create project at [supabase.com](https://supabase.com)
2. Copy URL and anon key to `.env`
3. Create tables (auto-created on first run)

#### MySQL

1. Create database in Hostinger control panel
2. Note: host, username, password, database name
3. Add to `.env`

#### Qdrant (Vector Database)

1. Sign up at [qdrant.io](https://qdrant.io)
2. Create cluster
3. Copy URL and API key to `.env`

---

## Service Management

### SystemD (Production)

```bash
# Start service
sudo systemctl start ultimate-orchestrator

# Stop service
sudo systemctl stop ultimate-orchestrator

# Restart service
sudo systemctl restart ultimate-orchestrator

# Check status
sudo systemctl status ultimate-orchestrator

# View logs
sudo journalctl -u ultimate-orchestrator -f

# Enable auto-start on boot
sudo systemctl enable ultimate-orchestrator
```

### Docker Compose

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# Logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f orchestrator
docker-compose logs -f redis
docker-compose logs -f postgres
```

---

## Monitoring & Maintenance

### Health Check

```bash
curl http://YOUR_SERVER_IP:8000/health
```

### View Metrics

```bash
curl http://YOUR_SERVER_IP:8000/api/v1/monitoring/summary
```

### Database Health

```bash
# Run health check script
python database_config.py
```

### Backup

```bash
# Backup databases
docker exec orchestrator-postgres pg_dump -U orchestrator orchestrator > backup.sql

# Backup .env
cp .env .env.backup.$(date +%Y%m%d)

# Backup logs
tar -czf logs-backup-$(date +%Y%m%d).tar.gz logs/
```

---

## SSL/HTTPS Setup

### Automatic (Let's Encrypt)

During installation, the script will ask:
```
Would you like to set up SSL with Let's Encrypt? (y/n)
```

Answer `y` and provide:
- Your domain name
- Email address

SSL will be automatically configured!

### Manual

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo certbot renew --dry-run
```

---

## Scaling

### Vertical Scaling (More Resources)

Upgrade your Hostinger VPS plan to larger instance.

### Horizontal Scaling (Multiple Instances)

1. **Load Balancer Setup:**
```nginx
upstream orchestrator {
    server 10.0.0.1:8000;
    server 10.0.0.2:8000;
    server 10.0.0.3:8000;
}

server {
    location / {
        proxy_pass http://orchestrator;
    }
}
```

2. **Shared Redis:**
```bash
# Use external Redis for all instances
REDIS_HOST=your-redis-server.com
```

3. **Shared Database:**
```bash
# All instances use same database
SUPABASE_URL=https://shared-db.supabase.co
```

---

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs orchestrator

# Check if port is in use
sudo netstat -tulpn | grep 8000

# Restart all services
docker-compose restart
```

### Database Connection Errors

```bash
# Test database connectivity
python database_config.py

# Check credentials in .env
cat .env | grep -E "SUPABASE|MYSQL|QDRANT"
```

### API Errors

```bash
# Check API logs
docker-compose logs -f orchestrator

# Test API endpoint
curl http://localhost:8000/health

# Restart API
docker-compose restart orchestrator
```

### Out of Memory

```bash
# Check memory usage
free -h

# Check Docker memory
docker stats

# Reduce workers in .env
WORKERS=2
```

---

## Security Best Practices

1. **Change default passwords:**
```bash
# Update admin password
ADMIN_PASSWORD=your-strong-password

# Update Postgres password
POSTGRES_PASSWORD=your-strong-password
```

2. **Firewall configuration:**
```bash
# Allow only necessary ports
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
```

3. **Regular updates:**
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Update Docker images
docker-compose pull
docker-compose up -d
```

4. **API Key rotation:**
- Rotate API keys monthly
- Use separate keys for dev/prod
- Store keys in environment variables only

---

## Performance Tuning

### Nginx

```nginx
# In nginx.conf
worker_processes auto;
worker_connections 4096;

# Enable gzip
gzip on;
gzip_types text/plain application/json;

# Enable caching
proxy_cache_path /var/cache/nginx levels=1:2 keys_zone=api_cache:10m;
```

### Redis

```bash
# In docker-compose.yml
redis:
  command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
```

### Python Workers

```bash
# In .env
WORKERS=4  # Set to number of CPU cores
```

---

## Updating

### Update Code

```bash
# On local machine
bash deploy.sh

# Or manually on server
cd /opt/ultimate-orchestrator
git pull origin main
docker-compose down
docker-compose up -d --build
```

### Update Dependencies

```bash
cd /opt/ultimate-orchestrator
source venv/bin/activate
pip install -r requirements.txt --upgrade
```

---

## Support

- **Documentation**: See README.md and FEATURES.md
- **API Docs**: http://YOUR_SERVER_IP:8000/docs
- **Logs**: `docker-compose logs -f`
- **Health Check**: http://YOUR_SERVER_IP:8000/health

---

## Quick Reference

| Task | Command |
|------|---------|
| Deploy | `bash deploy.sh` |
| Start | `docker-compose up -d` |
| Stop | `docker-compose down` |
| Logs | `docker-compose logs -f` |
| Restart | `docker-compose restart` |
| Status | `docker-compose ps` |
| Health | `curl http://localhost:8000/health` |
| Backup | `bash backup.sh` |

---

**🎉 You're all set! Your Ultimate Orchestrator is now running in the cloud!**
