# CI/CD Setup for MaryPause AI

## Current Setup ✅

Your Render deployment is **already configured** with automatic CI/CD:

### Render Auto-Deploy Configuration
- **Status**: ENABLED
- **Trigger**: On successful checks (`checksPass`)
- **Branch**: `main`
- **Repository**: https://github.com/marypause/marypause_ai
- **Service URL**: https://marypause-ai.onrender.com

### How It Works

1. **Push to Main Branch** → Render automatically detects the new commit
2. **Build Process** → Runs `pip install -r requirements.txt`
3. **Deploy** → Starts service with `gunicorn --config gunicorn.conf.py agent.api:app`
4. **Health Check** → Verifies deployment is successful

## Deployment Workflow

### Automatic Deployments (Current Setup)
Every push to the `main` branch triggers an automatic deployment:

```bash
# Local development
git add .
git commit -m "feat: your changes"
git push origin main
# → Render automatically deploys
```

### Manual Deployments
If needed, you can trigger manual deployments from:
- Render Dashboard: https://dashboard.render.com/web/srv-d2m587bipnbc738tbb8g
- Click "Manual Deploy" → Select commit → Deploy

## GitHub Actions Integration (Optional Enhancement)

To add pre-deployment checks, create `.github/workflows/ci.yml`:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
        LLM_PROVIDER: openai
        LLM_API_KEY: ${{ secrets.LLM_API_KEY }}
      run: |
        pytest tests/ --cov=agent --cov=ingestion
    
    - name: Lint with ruff
      run: |
        pip install ruff
        ruff check .
    
    - name: Type check with mypy
      run: |
        pip install mypy
        mypy agent/ --ignore-missing-imports

  notify-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    
    steps:
    - name: Deployment Notification
      run: |
        echo "✅ Tests passed! Render will auto-deploy commit ${{ github.sha }}"
```

## Environment Variables Management

### Required Variables in Render
Ensure these are set in Render Dashboard → Environment:

```bash
# Database
DATABASE_URL=your_postgres_url
# OR for Supabase:
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your_key

# Neo4j
NEO4J_URI=bolt://your-neo4j-host:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4o-mini

# Embeddings
EMBEDDING_PROVIDER=openai
EMBEDDING_API_KEY=sk-...
EMBEDDING_MODEL=text-embedding-3-small

# Application
APP_ENV=production
LOG_LEVEL=INFO
```

## Deployment Best Practices

### 1. Use Feature Branches
```bash
# Create feature branch
git checkout -b feature/new-feature

# Make changes and test locally
python -m agent.api  # Test locally

# Push to GitHub
git push origin feature/new-feature

# Create PR to main branch
# After review and merge → Auto-deploy
```

### 2. Pre-commit Hooks (Recommended)
Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff
        args: [--fix]
  
  - repo: https://github.com/psf/black
    rev: 23.0.0
    hooks:
      - id: black
        language_version: python3.11
```

Install:
```bash
pip install pre-commit
pre-commit install
```

### 3. Rollback Strategy
If a deployment fails:

1. **Quick Rollback via Render Dashboard**:
   - Go to Deploys tab
   - Find last successful deployment
   - Click "Rollback to this deploy"

2. **Git Revert**:
   ```bash
   git revert HEAD
   git push origin main
   # Triggers new deployment with reverted code
   ```

## Monitoring Deployments

### Check Deployment Status
```bash
# View live logs
curl https://marypause-ai.onrender.com/health

# Check specific deployment in Render Dashboard
# https://dashboard.render.com/web/srv-d2m587bipnbc738tbb8g/deploys
```

### Deployment Notifications
Render can send notifications for:
- Successful deployments
- Failed deployments
- Service downtime

Configure in: Dashboard → Settings → Notifications

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check `requirements.txt` for conflicts
   - Verify Python version matches (3.11)
   - Check build logs in Render Dashboard

2. **Runtime Failures**
   - Verify all environment variables are set
   - Check application logs: Dashboard → Logs
   - Test health endpoint: `curl https://marypause-ai.onrender.com/health`

3. **Auto-deploy Not Triggering**
   - Verify branch name is `main`
   - Check GitHub webhook in repo settings
   - Ensure auto-deploy is enabled in Render

### Debug Commands
```bash
# Check if service is running
curl https://marypause-ai.onrender.com/health

# View recent commits
git log --oneline -10

# Check which commit is deployed
# View in Render Dashboard → Deploys
```

## Performance Optimization

### Build Cache
Render caches Python packages between builds. To clear cache:
1. Dashboard → Settings → Clear build cache
2. Trigger new deployment

### Deployment Speed
- Keep `requirements.txt` minimal
- Use specific versions to avoid resolution delays
- Consider using `pip-compile` for locked dependencies

## Security Considerations

1. **Never commit secrets** - Use environment variables
2. **Use GitHub Secrets** for API keys in Actions
3. **Enable branch protection** for main branch
4. **Review PRs** before merging to main
5. **Monitor dependencies** for vulnerabilities

## Summary

Your CI/CD pipeline is **ready to use**:

✅ Auto-deploy enabled on Render
✅ Pushes to `main` trigger deployments
✅ Service monitored with health checks
✅ Rollback available if needed

To deploy: Simply push to the `main` branch of https://github.com/marypause/marypause_ai