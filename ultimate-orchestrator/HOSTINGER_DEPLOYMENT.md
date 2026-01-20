# 🚀 HOSTINGER DEPLOYMENT GUIDE
## Ultimate Orchestrator with LiveKit Voice & 4D Frontend

---

## 🎯 What You Get

After deployment, you'll have:

✅ **70+ AI Agents** - Accessible via voice and API
✅ **LiveKit Voice Interface** - Real-time voice conversations with AI
✅ **4D Futuristic Frontend** - Stunning animated UI with particle effects
✅ **REST API** - Full programmatic access to all agents
✅ **WebSocket Support** - Real-time voice streaming
✅ **Multi-Database** - Supabase + MySQL + Qdrant integration
✅ **Auto-Scaling** - Handles multiple concurrent users

---

## 📋 Prerequisites

### 1. Hostinger Business Plan Requirements

- ✅ SSH access enabled
- ✅ Python 3.9+ installed
- ✅ At least 2GB RAM
- ✅ 10GB+ disk space

### 2. Required API Keys (Already in .env)

All credentials are configured in your `.env` file:

```bash
# LLM Providers
OPENAI_API_KEY=✅
ANTHROPIC_API_KEY=✅
GROQ_API_KEY=✅
GOOGLE_API_KEY=✅

# Databases
SUPABASE_URL=✅
SUPABASE_KEY=✅
QDRANT_URL=✅
QDRANT_API_KEY=✅

# Voice
DEEPGRAM_API_KEY=✅

# MCP Integrations
GITHUB_TOKEN=✅
N8N_API_KEY=✅
```

### 3. LiveKit Server (Optional)

For production voice:
- **Option A**: Self-hosted LiveKit (included in deployment)
- **Option B**: LiveKit Cloud (get free tier at https://livekit.io)

---

## 🚀 ONE-COMMAND DEPLOYMENT

### Step 1: Install Dependencies

```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
pip install -r requirements.txt
```

### Step 2: Deploy to Hostinger

```bash
python deploy_hostinger.py
```

That's it! The script will:

1. ✅ Connect to Hostinger via SSH
2. ✅ Setup Python virtual environment
3. ✅ Upload all application files
4. ✅ Install dependencies
5. ✅ Configure Nginx reverse proxy
6. ✅ Setup process manager
7. ✅ Upload futuristic frontend
8. ✅ Start the application
9. ✅ Verify deployment

---

## 🎬 What Happens During Deployment

### Automated Steps

```
[1/10] Connecting to Hostinger via SSH...
   ✓ Connected to 46.202.197.97:65002

[2/10] Setting up remote Python environment...
   ✓ Python environment configured

[3/10] Uploading application files...
   ✓ Uploaded 12 files

[4/10] Installing Python dependencies...
   ✓ Dependencies installed

[5/10] Configuring process manager...
   ✓ Process manager configured

[6/10] Configuring Nginx reverse proxy...
   ✓ Nginx proxy configured

[7/10] Uploading futuristic frontend...
   ✓ Frontend uploaded

[8/10] Starting application...
   ✓ Application started (PID: 12345)

[9/10] Verifying deployment...
   ✓ Process is running
   ✓ Application started successfully

[10/10] Deployment complete!
```

---

## 🌐 Access Your Deployment

After successful deployment:

### Frontend Interface
```
http://your-domain.com/
```

**Features:**
- 4D animated particle background
- Real-time voice visualizer
- Voice button with "Activate Voice"
- Live agent grid showing 70+ agents
- Real-time stats dashboard

### API Endpoints

```bash
# Health check
curl http://your-domain.com/health

# API documentation
http://your-domain.com/docs

# List all agents
curl http://your-domain.com/api/v1/agents

# Create voice session
curl -X POST http://your-domain.com/api/v1/voice/create-session

# Orchestrate a task
curl -X POST http://your-domain.com/api/v1/orchestrate \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a Python function to calculate fibonacci"}'
```

### WebSocket Endpoints

```javascript
// Voice WebSocket
ws://your-domain.com/ws/voice/{user_id}

// Monitoring WebSocket
ws://your-domain.com/ws/monitoring
```

---

## 🎙️ Using Voice Features

### Browser-Based Voice

1. Open the frontend: `http://your-domain.com/`
2. Click the **"ACTIVATE VOICE"** button
3. Allow microphone access
4. Start speaking!

The AI will:
- Listen to your voice
- Process with Deepgram STT
- Route to appropriate agent
- Respond with OpenAI TTS

### LiveKit Voice (Production)

For better quality and scalability:

1. Sign up at https://livekit.io (free tier available)
2. Get your LiveKit credentials
3. Add to `.env`:

```bash
LIVEKIT_URL=wss://your-project.livekit.cloud
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_secret
```

4. Redeploy with updated credentials

---

## 🛠️ Management Commands

### View Logs

```bash
ssh -p 65002 u472811699@46.202.197.97 \
  'tail -f /home/u472811699/ultimate-orchestrator/logs/app.log'
```

### Check Status

```bash
ssh -p 65002 u472811699@46.202.197.97 \
  'pgrep -f api_server.py'
```

### Restart Application

```bash
ssh -p 65002 u472811699@46.202.197.97 \
  'pkill -f api_server.py && cd /home/u472811699/ultimate-orchestrator && nohup venv/bin/python api_server.py > logs/app.log 2>&1 &'
```

### Update Application

```bash
# Run deployment script again
python deploy_hostinger.py
```

---

## 🎨 Frontend Features

### 4D Visual Effects

- **Particle System**: 200+ 3D particles with orbital motion
- **Gradient Orbs**: Animated background gradients
- **Voice Visualizer**: Real-time audio waveform
- **Glassmorphism**: Translucent card designs
- **Pulse Rings**: Animated rings on voice activation

### Interaction

- **Space Bar**: Toggle voice mode
- **Escape**: Deactivate voice
- **Agent Search**: Filter 70+ agents
- **Voice Button**: Click or use keyboard

### Responsive Design

- Desktop: Full experience
- Tablet: Optimized layout
- Mobile: Touch-friendly controls

---

## 🔧 Troubleshooting

### Issue: Can't Connect to Hostinger

**Solution:**
```bash
# Test SSH connection
ssh -p 65002 u472811699@46.202.197.97

# Check if password is correct in .env
cat .env | grep HOSTINGER_SSH_PASSWORD
```

### Issue: Application Won't Start

**Solution:**
```bash
# Check logs
ssh -p 65002 u472811699@46.202.197.97 \
  'cat /home/u472811699/ultimate-orchestrator/logs/app.log'

# Check if port is in use
ssh -p 65002 u472811699@46.202.197.97 \
  'netstat -tuln | grep 8000'
```

### Issue: Voice Not Working

**Solutions:**

1. **Check Microphone Permissions**: Browser must allow mic access
2. **Check Deepgram Key**: Verify in `.env`
3. **Check Browser Console**: Look for WebSocket errors
4. **Try Different Browser**: Chrome/Edge recommended

### Issue: Frontend Not Loading

**Solutions:**

1. **Check Nginx**: Make sure proxy is configured
2. **Check Static Files**: Verify frontend files uploaded
3. **Clear Browser Cache**: Hard refresh (Ctrl+Shift+R)

---

## 📊 Performance Tuning

### For High Traffic

Edit `.env`:

```bash
# Increase workers (if your plan supports it)
WORKERS=4

# Enable caching
REDIS_ENABLED=true
REDIS_URL=redis://localhost:6379
```

### For Low Resources

```bash
# Reduce workers
WORKERS=1

# Disable monitoring
MONITORING_ENABLED=false
```

---

## 🔐 Security Recommendations

### 1. Environment Variables

Never commit `.env` to git:

```bash
# .env is already in .gitignore
git status # should not show .env
```

### 2. HTTPS Setup

For production, enable SSL:

```bash
# Install certbot on Hostinger
sudo certbot --nginx -d your-domain.com
```

### 3. API Rate Limiting

Add to `api_server.py`:

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/api/v1/orchestrate")
@limiter.limit("10/minute")
async def orchestrate_task(request: AgentRequest):
    ...
```

---

## 🚀 Advanced Configuration

### Custom Domain Setup

1. Point your domain to Hostinger IP: `46.202.197.97`
2. Update Nginx config with your domain
3. Enable SSL with Let's Encrypt

### Database Optimization

```bash
# Enable connection pooling
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10

# Enable Qdrant caching
QDRANT_CACHE_ENABLED=true
```

### Voice Quality Enhancement

```bash
# Use premium Deepgram model
STT_MODEL=nova-2-phonecall

# Use premium OpenAI voice
TTS_VOICE=nova  # or shimmer, alloy, etc.
```

---

## 📈 Monitoring & Analytics

### Built-in Dashboard

Access at: `http://your-domain.com/api/v1/monitoring/dashboard`

Shows:
- Total requests processed
- Average response time
- Agent usage statistics
- Error rates
- System health

### Real-time Monitoring

WebSocket connection:

```javascript
const ws = new WebSocket('ws://your-domain.com/ws/monitoring');
ws.onmessage = (event) => {
  const stats = JSON.parse(event.data);
  console.log('System stats:', stats);
};
```

---

## 🎯 Next Steps

After deployment:

1. ✅ Test the frontend interface
2. ✅ Try voice mode
3. ✅ Test API endpoints
4. ✅ Check monitoring dashboard
5. ✅ Setup custom domain (optional)
6. ✅ Enable SSL (recommended)
7. ✅ Configure backups

---

## 💡 Tips & Best Practices

### Cost Optimization

- Start with 1 worker, scale up as needed
- Use Deepgram's free tier (200 hours/month)
- Cache frequently used responses
- Monitor API usage to avoid overage

### Performance

- Enable CDN for static assets
- Use connection pooling for databases
- Implement response caching
- Compress API responses

### Reliability

- Setup health check monitoring
- Configure auto-restart on crash
- Enable error logging
- Backup database regularly

---

## 📞 Support & Resources

### Documentation

- API Docs: `http://your-domain.com/docs`
- This Guide: `HOSTINGER_DEPLOYMENT.md`
- Main README: `README.md`

### Logs Location

```bash
/home/u472811699/ultimate-orchestrator/logs/
  ├── app.log        # Application logs
  ├── access.log     # HTTP access logs
  └── error.log      # Error logs
```

### Quick Reference

| Component | Location |
|-----------|----------|
| Application | `/home/u472811699/ultimate-orchestrator` |
| Logs | `/home/u472811699/ultimate-orchestrator/logs` |
| Frontend | `/home/u472811699/public_html` |
| Nginx Config | `/home/u472811699/.nginx/ultimate-orchestrator.conf` |
| Virtual Env | `/home/u472811699/ultimate-orchestrator/venv` |

---

## 🎉 You're Ready!

Your Ultimate Orchestrator is now deployed with:

- ⚡ 70+ AI agents ready to assist
- 🎙️ Real-time voice capabilities
- 🎨 Stunning 4D futuristic interface
- 🔌 Full REST API access
- 📊 Built-in monitoring
- 🚀 Production-ready infrastructure

**Start using it now at: `http://your-domain.com/`**

---

*Need help? Check the logs or restart the application using the management commands above.*
