# 🚀 WSL QUICK START GUIDE
## For Fresh Windows 11 + Kali Linux WSL

---

## 📍 WHERE AM I?

**Your project location:**
```
\\wsl$\Kali-linux\home\user\ottomator-agents\ultimate-orchestrator
```

**In WSL terminal:**
```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
```

**Current user:** `root`
**System:** Kali Linux (WSL on Windows 11)

---

## 🗺️ DIRECTORY STRUCTURE

```
/home/user/ottomator-agents/
├── ultimate-orchestrator/       ← YOU ARE HERE!
│   ├── frontend/                ← 4D Futuristic UI
│   │   ├── index.html
│   │   ├── style.css
│   │   ├── app.js
│   │   └── particles.js
│   ├── api_server.py            ← Main FastAPI server
│   ├── livekit_voice.py         ← Voice integration
│   ├── meta_orchestrator.py     ← 70+ agents coordinator
│   ├── deploy_hostinger.py      ← Deploy to Hostinger
│   ├── quickstart.sh            ← Test locally
│   ├── .env                     ← YOUR CREDENTIALS HERE
│   ├── requirements.txt         ← Python dependencies
│   └── ...
└── [70+ other AI agent projects]
```

---

## ✏️ EDIT YOUR .ENV FILE

### **Option 1: From Windows (Easiest)**

1. Open File Explorer
2. In address bar, paste:
   ```
   \\wsl$\Kali-linux\home\user\ottomator-agents\ultimate-orchestrator
   ```
3. Right-click `.env` → Open with Notepad or VS Code
4. Replace `your_*_key_here` with your actual API keys
5. Save (Ctrl+S)

### **Option 2: From WSL Terminal**

```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
nano .env
```

Replace these placeholders:
- `your_openai_key_here` → Your actual OpenAI API key
- `your_supabase_url_here` → Your Supabase project URL
- `your_qdrant_url_here` → Your Qdrant cluster URL
- `your_deepgram_key_here` → Your Deepgram API key
- etc...

**Save:** `Ctrl+O`, `Enter`, `Ctrl+X`

### **Option 3: VS Code (Best)**

```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
code .
```

Then edit `.env` in VS Code.

---

## 🔑 REQUIRED API KEYS

You need at minimum:

| Service | Required? | Get it from |
|---------|-----------|-------------|
| **OpenAI** | ✅ YES | https://platform.openai.com/api-keys |
| **Supabase** | ✅ YES | https://supabase.com/dashboard |
| **Qdrant** | ✅ YES | https://qdrant.tech/cloud/ |
| **Deepgram** | ✅ YES | https://console.deepgram.com/ |
| Anthropic | Optional | https://console.anthropic.com/ |
| Groq | Optional | https://console.groq.com/ |
| Google | Optional | https://makersuite.google.com/ |

---

## 🚀 START THE APPLICATION

### **Test Locally First:**

```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
bash quickstart.sh
```

This will start the server at: **http://localhost:8000/**

Open in Windows browser:
- Frontend: http://localhost:8000/
- API Docs: http://localhost:8000/docs

**Stop server:** Press `Ctrl+C`

### **Deploy to Hostinger:**

After testing locally works:

```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
python3 deploy_hostinger.py
```

This will automatically:
1. Connect to your Hostinger server via SSH
2. Upload all files
3. Install dependencies
4. Start the application
5. Give you the live URL!

---

## 🎯 COMMON WSL COMMANDS

```bash
# Navigate to project
cd /home/user/ottomator-agents/ultimate-orchestrator

# List files
ls -lah

# Edit .env
nano .env

# Check Python
python3 --version

# Install dependencies
pip3 install -r requirements.txt

# Start server
bash quickstart.sh

# Deploy to Hostinger
python3 deploy_hostinger.py

# View logs (if deployed)
tail -f logs/app.log
```

---

## 🪟 ACCESS FROM WINDOWS

### **Open WSL files in Windows Explorer:**

```
\\wsl$\Kali-linux\home\user\ottomator-agents\ultimate-orchestrator
```

### **Open VS Code in WSL:**

From WSL terminal:
```bash
code .
```

### **Access localhost from Windows:**

When running `bash quickstart.sh`, open browser:
```
http://localhost:8000/
```

WSL automatically forwards ports to Windows!

---

## 🛠️ TROUBLESHOOTING

### **Can't find the directory?**

From WSL:
```bash
cd /home/user/ottomator-agents/ultimate-orchestrator
pwd  # Should show: /home/user/ottomator-agents/ultimate-orchestrator
```

From Windows Explorer:
```
\\wsl$\Kali-linux\home\user\ottomator-agents\ultimate-orchestrator
```

### **Permission denied?**

```bash
sudo su  # Switch to root if needed
cd /home/user/ottomator-agents/ultimate-orchestrator
```

### **Dependencies not installing?**

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt --no-cache-dir
```

### **Port already in use?**

```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or change port in .env
nano .env
# Change: PORT=8001
```

---

## 📝 QUICK CHEAT SHEET

| Task | Command |
|------|---------|
| Go to project | `cd /home/user/ottomator-agents/ultimate-orchestrator` |
| Edit credentials | `nano .env` |
| Test locally | `bash quickstart.sh` |
| Deploy | `python3 deploy_hostinger.py` |
| View files | `ls -lah` |
| Check status | `ps aux \| grep api_server` |
| Open in Windows | `\\wsl$\Kali-linux\home\user\ottomator-agents` |

---

## 🎨 WHAT YOU GET

After running `bash quickstart.sh` or deploying:

### **Frontend (http://localhost:8000/)**
- 🌌 3D animated particle background
- 🎙️ Voice button to talk to AI
- 🤖 Grid of 70+ AI agents
- 📊 Live stats dashboard
- ⚡ Real-time visualizations

### **API (http://localhost:8000/docs)**
- 📚 Interactive API documentation
- 🔌 All endpoints documented
- 🧪 Test APIs in browser

### **Voice Interface**
- Click "ACTIVATE VOICE" button
- Speak naturally
- AI responds with voice!

---

## 🎯 NEXT STEPS

1. **Edit .env** with your API keys
2. **Test locally:** `bash quickstart.sh`
3. **Open browser:** http://localhost:8000/
4. **Try voice mode:** Click the big voice button!
5. **Deploy live:** `python3 deploy_hostinger.py`

---

## 🆘 NEED HELP?

**Check documentation:**
- Main README: `cat README.md`
- Hostinger Guide: `cat HOSTINGER_DEPLOYMENT.md`
- Feature List: `cat FEATURES.md`

**Run diagnostics:**
```bash
python3 --version  # Should be 3.11+
pip3 list | grep pydantic-ai  # Should show installed
cat .env | grep OPENAI  # Check if keys are set
```

---

**You're all set! Start with `bash quickstart.sh` to test locally! 🚀**
