# 🚀 ULTIMATE ORCHESTRATOR - QUICKSTART GUIDE

## Get Your System Live in 5 Minutes!

---

## 📋 What You Need

Before starting, gather these credentials:

### Required:
- ✅ **OpenAI API Key** (get at platform.openai.com)
- ✅ **Hostinger VPS** (IP address, SSH username, password)
- ✅ **ONE Database**: Supabase OR MySQL

### Optional (but recommended):
- 🎙️ **Deepgram API Key** (for voice features)
- 📊 **Qdrant API Key** (for vector search)
- 🌐 **Domain Name** (for SSL/HTTPS)

---

## 🎯 Step 1: Fill in Your Credentials (2 minutes)

1. **Open `CREDENTIALS.txt`** in this folder

2. **Find each section** and paste your credentials:

```bash
# Example - Replace with YOUR actual values:
OPENAI_API_KEY=sk-proj-ABC123XYZ...     # ← Paste your key here
SUPABASE_URL=https://abc123.supabase.co # ← Paste your URL here
SUPABASE_KEY=eyJhbGc...                  # ← Paste your key here
MYSQL_HOST=mysql.hostinger.com           # ← Your MySQL host
MYSQL_USER=u123456_dbuser                # ← Your MySQL username
MYSQL_PASSWORD=your_password             # ← Your MySQL password
MYSQL_DATABASE=u123456_dbname            # ← Your database name
HOSTINGER_API_KEY=your-hostinger-key     # ← From Hostinger dashboard
```

3. **Save the file** (Ctrl+S or Cmd+S)

### Where to Get Each Credential:

| Service | Where to Get It |
|---------|----------------|
| **OpenAI** | https://platform.openai.com/api-keys |
| **Supabase** | https://supabase.com → Your Project → Settings → API |
| **MySQL** | Hostinger Control Panel → Databases → Details |
| **Hostinger API** | Hostinger → Advanced → API |
| **Deepgram** | https://deepgram.com → API Keys |
| **Qdrant** | https://qdrant.io → Clusters → Your Cluster |

---

## 🚀 Step 2: Deploy to Cloud (3 minutes)

### From Your Local Machine:

```bash
# Navigate to the ultimate-orchestrator folder
cd ultimate-orchestrator

# Make deployment script executable (if not already)
chmod +x deploy.sh

# Run the deployment
bash deploy.sh
```

### The script will ask you:

```
Enter your Hostinger VPS IP address: 123.456.789.10
Enter SSH username (default: root): root
Enter SSH port (default: 22): 22
Continue with installation? (y/n): y
```

### What Happens Automatically:

The script will:
1. ✅ Test SSH connection
2. ✅ Copy all files to your server
3. ✅ Upload credentials
4. ✅ Install system dependencies (Python, Docker, Nginx)
5. ✅ Set up Python environment
6. ✅ Install all Python packages
7. ✅ Configure Nginx reverse proxy
8. ✅ Set up SSL (if domain provided)
9. ✅ Start all services with Docker Compose
10. ✅ Verify deployment

**Total time:** ~3-5 minutes (depending on server speed)

---

## ✅ Step 3: Verify It's Working

Once deployment completes, you'll see:

```
============================================================================
  ✅ DEPLOYMENT COMPLETE!
============================================================================

Your Ultimate Orchestrator is now running at:
  HTTP:  http://123.456.789.10:8000
  HTTPS: https://yourdomain.com (if SSL configured)

API Documentation:
  http://123.456.789.10:8000/docs
```

### Test Your Deployment:

1. **Open your browser** to:
   ```
   http://YOUR_SERVER_IP:8000/docs
   ```

2. **You should see** the FastAPI interactive documentation

3. **Try the health check**:
   ```bash
   curl http://YOUR_SERVER_IP:8000/health
   ```

   Should return:
   ```json
   {
     "status": "healthy",
     "version": "1.0.0",
     "services": {
       "orchestrator": "running",
       "monitoring": "running"
     }
   }
   ```

4. **Test agent orchestration**:
   - Go to `http://YOUR_SERVER_IP:8000/docs`
   - Find `POST /api/v1/orchestrate`
   - Click "Try it out"
   - Enter task: `"List all available AI agents"`
   - Click "Execute"

---

## 🎉 You're Live!

Your Ultimate Orchestrator is now running with:

✅ **70+ AI Agents** - Ready to orchestrate
✅ **Voice Capabilities** - If you added Deepgram key
✅ **Real-time Monitoring** - Check `/api/v1/monitoring/summary`
✅ **Production Security** - SSL, firewall, rate limiting
✅ **Auto-restart** - Services recover automatically
✅ **Multi-database** - Supabase + MySQL + Qdrant

---

## 🔧 Common Commands

### View Logs
```bash
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose logs -f"
```

### Restart Services
```bash
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose restart"
```

### Check Status
```bash
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose ps"
```

### Stop Services
```bash
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose down"
```

### Start Services
```bash
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose up -d"
```

---

## 🎯 Next Steps

### 1. Test Agent Orchestration

Visit `http://YOUR_SERVER_IP:8000/docs` and try:

**Discover Agents:**
```json
POST /api/v1/discover
{
  "task_description": "I need to review code for security issues"
}
```

**Orchestrate Task:**
```json
POST /api/v1/orchestrate
{
  "task": "Review this Python code for security vulnerabilities",
  "enable_voice": false
}
```

### 2. Enable Voice Features

If you added Deepgram key:
```json
POST /api/v1/orchestrate
{
  "task": "Hello, I want to talk to you",
  "enable_voice": true
}
```

### 3. Monitor Performance

Check real-time metrics:
```bash
curl http://YOUR_SERVER_IP:8000/api/v1/monitoring/summary
```

View dashboard:
```bash
curl http://YOUR_SERVER_IP:8000/api/v1/monitoring/dashboard
```

### 4. Set Up Domain (Optional)

If you have a domain name:

1. **Point DNS** to your server IP:
   - Add A record: `@` → `YOUR_SERVER_IP`
   - Add A record: `www` → `YOUR_SERVER_IP`

2. **Get SSL certificate** (already set up during deployment):
   ```bash
   ssh root@YOUR_SERVER_IP "certbot --nginx -d yourdomain.com"
   ```

3. **Access via HTTPS**:
   ```
   https://yourdomain.com
   ```

---

## 🆘 Troubleshooting

### Issue: Can't connect to server
```bash
# Check if server is running
ping YOUR_SERVER_IP

# Check if SSH port is open
telnet YOUR_SERVER_IP 22

# Try connecting with password
ssh -o PubkeyAuthentication=no root@YOUR_SERVER_IP
```

### Issue: Services not starting
```bash
# SSH into server
ssh root@YOUR_SERVER_IP

# Check Docker containers
cd /opt/ultimate-orchestrator
docker-compose ps

# View logs
docker-compose logs orchestrator

# Restart services
docker-compose restart
```

### Issue: Database connection failed
```bash
# Test database connections
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && python database_config.py"

# Check credentials in .env
ssh root@YOUR_SERVER_IP "cat /opt/ultimate-orchestrator/.env | grep -E 'SUPABASE|MYSQL'"
```

### Issue: API returns errors
```bash
# Check API logs
ssh root@YOUR_SERVER_IP "docker logs orchestrator-orchestrator-1"

# Check if port 8000 is accessible
curl http://YOUR_SERVER_IP:8000/health

# Restart API service
ssh root@YOUR_SERVER_IP "cd /opt/ultimate-orchestrator && docker-compose restart orchestrator"
```

---

## 📚 More Help

- **Full Documentation**: See `DEPLOYMENT.md`
- **Features List**: See `FEATURES.md`
- **API Reference**: http://YOUR_SERVER_IP:8000/docs
- **Hostinger Support**: Check Hostinger knowledge base
- **Database Guides**:
  - Supabase: https://supabase.com/docs
  - MySQL: https://dev.mysql.com/doc/
  - Qdrant: https://qdrant.tech/documentation/

---

## 🎓 Example Usage

### Example 1: Code Review
```bash
curl -X POST "http://YOUR_SERVER_IP:8000/api/v1/orchestrate" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Review this code for security issues: def login(user, pwd): exec(f\"SELECT * FROM users WHERE name={user}\")"
  }'
```

### Example 2: Discover RAG Agents
```bash
curl -X POST "http://YOUR_SERVER_IP:8000/api/v1/discover" \
  -H "Content-Type: application/json" \
  -d '{
    "task_description": "I need RAG with PDF processing",
    "required_capabilities": ["rag", "pdf_processing"]
  }'
```

### Example 3: List All Agents
```bash
curl http://YOUR_SERVER_IP:8000/api/v1/agents
```

### Example 4: Monitoring
```bash
# System summary
curl http://YOUR_SERVER_IP:8000/api/v1/monitoring/summary

# Top agents
curl "http://YOUR_SERVER_IP:8000/api/v1/monitoring/top?metric=executions&limit=10"

# Recent alerts
curl http://YOUR_SERVER_IP:8000/api/v1/monitoring/alerts
```

---

## 🎉 Congratulations!

You now have a **production-ready AI agent orchestration system** running in the cloud!

**What you can do:**
- 🎯 Orchestrate 70+ AI agents
- 🎙️ Voice-enable any agent
- 📊 Monitor everything in real-time
- 🗄️ Use multiple databases
- 🔒 Production-grade security
- ☁️ Scale as needed

**Start building amazing things!** 🚀

---

*Need help? Check the full documentation in DEPLOYMENT.md or README.md*
