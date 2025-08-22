# Quick Setup Guide: Using Supabase with Medical RAG Agent

This is a quick setup guide to get your Medical RAG agent working with Supabase instead of direct PostgreSQL connections.

## ✅ What You Have

Your Supabase project is already set up:
- ✅ Database URL: `https://bpopugzfbokjzgawshov.supabase.co`
- ✅ Database schema applied (documents, chunks, sessions, messages tables)
- ✅ Vector embeddings configured (3072 dimensions for Gemini)
- ✅ Custom functions: `match_chunks`, `hybrid_search`, `get_document_chunks`
- ✅ Supabase API keys available

## 🚀 Quick Setup (5 steps)

### 1. Install Dependencies
```bash
pip install supabase==2.10.0
# or update all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
Update your `.env` file:
```bash
# Set Supabase as the database provider
DB_PROVIDER=supabase

# Add your Supabase credentials
SUPABASE_URL=https://bpopugzfbokjzgawshov.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# Keep your other settings
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=google
# ... etc
```

### 3. Get Your API Keys
1. Go to [Supabase Dashboard](https://supabase.com/dashboard)
2. Navigate to your project
3. Go to **Settings** → **API**
4. Copy:
   - **URL** (already known: `https://bpopugzfbokjzgawshov.supabase.co`)
   - **anon public** key → `SUPABASE_ANON_KEY`
   - **service_role** key → `SUPABASE_SERVICE_ROLE_KEY`

### 4. Test Configuration
```bash
# Test your Supabase setup
python test_supabase_connection.py
```

Expected output:
```
🎉 All tests passed! Supabase is properly configured.
```

### 5. Run the System
```bash
# Start the API server
python -m agent.api

# In another terminal, test the health endpoint
curl http://localhost:8058/health
```

You should see:
```json
{
  "status": "healthy",
  "database": true,
  "provider": "supabase",
  "stats": {...}
}
```

## 🧪 Test Your Setup

### Health Check
```bash
curl http://localhost:8058/health
```

### Provider Information
```bash
curl http://localhost:8058/provider/info
curl http://localhost:8058/provider/validate
```

### Database Stats
```bash
curl http://localhost:8058/database/stats
```

### CLI Interface
```bash
python cli.py
```

## 📊 What Changed

The system now automatically uses Supabase when `DB_PROVIDER=supabase`:

| Operation | Before (asyncpg) | After (Supabase) |
|-----------|------------------|------------------|
| Connection | Direct PostgreSQL pool | Supabase HTTP API |
| Vector Search | Direct SQL function | RPC call to `match_chunks` |
| Session Management | SQL INSERT/SELECT | REST API calls |
| Authentication | Database password | API keys |

## 🔧 Key Benefits

- ✅ **No database password needed** - only API keys
- ✅ **Automatic scaling** - Supabase handles connection management
- ✅ **Built-in monitoring** - Dashboard with metrics
- ✅ **Same API** - Your application code doesn't change
- ✅ **Easy switching** - Change `DB_PROVIDER` to switch back

## 🚨 Troubleshooting

### Common Issues

**API Key Error**
```
Error: Invalid API key
```
→ Check `SUPABASE_SERVICE_ROLE_KEY` is correctly set

**Vector Dimension Error**
```
Error: vector has wrong dimensions
```
→ Your schema is already configured for 3072 dimensions (Gemini)

**Function Not Found**
```
Error: function "match_chunks" does not exist
```
→ Functions are already created in your Supabase project

**Connection Timeout**
```
Error: Request timeout
```
→ Check internet connection and Supabase project status

### Debug Mode
```bash
export LOG_LEVEL=DEBUG
python -m agent.api
```

## 📈 Next Steps

1. **✅ Complete Setup**: Follow the 5 steps above
2. **📄 Ingest Documents**: Run `python -m ingestion.ingest`
3. **🤖 Test Agent**: Use CLI or API to ask questions
4. **📊 Monitor Usage**: Check Supabase dashboard
5. **🔄 Scale**: Supabase automatically handles scaling

## 🆘 Need Help?

- **Configuration issues**: Run `python test_supabase_connection.py`
- **API problems**: Check `curl http://localhost:8058/provider/validate`
- **Detailed guide**: See `SUPABASE_MIGRATION_GUIDE.md`
- **Original setup**: Set `DB_PROVIDER=postgres` to use direct connection

## 🎯 Summary

Your Medical RAG agent can now run with **just Supabase API keys** instead of database passwords. The system automatically detects your configuration and uses the appropriate database provider.

**Ready to test? Run:**
```bash
python test_supabase_connection.py
```