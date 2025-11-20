# 🚀 GitHub Pages Deployment Guide

Complete guide for deploying the Live Agent Studio static website to GitHub Pages with continuous deployment via GitHub Actions.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [GitHub Pages Configuration](#github-pages-configuration)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Custom Domain Setup](#custom-domain-setup-optional)
- [Deployment Workflow](#deployment-workflow)
- [Monitoring & Maintenance](#monitoring--maintenance)
- [Troubleshooting](#troubleshooting)

## ✅ Prerequisites

Before deploying, ensure you have:

- [ ] GitHub account with repository access
- [ ] Repository forked or cloned: `miiwins/ottomator-agents`
- [ ] Git installed locally
- [ ] Basic knowledge of Git commands
- [ ] (Optional) Custom domain for professional URL

## 🎯 Initial Setup

### Step 1: Verify Repository Structure

Ensure your repository has the correct structure:

```
ottomator-agents/
├── www/                          # Website files
│   ├── index.html
│   ├── css/styles.css
│   ├── js/main.js
│   ├── .nojekyll                # Important for GitHub Pages
│   ├── CNAME                    # For custom domain
│   └── README.md
├── .github/
│   └── workflows/
│       └── deploy-pages.yml     # CI/CD workflow
└── DEPLOYMENT_GUIDE.md          # This file
```

### Step 2: Local Testing

Test the website locally before deploying:

```bash
# Navigate to the www directory
cd ottomator-agents/www

# Start a local server (choose one)

# Option 1: Python 3
python -m http.server 8000

# Option 2: Python 2
python -m SimpleHTTPServer 8000

# Option 3: Node.js
npx http-server -p 8000

# Option 4: PHP
php -S localhost:8000
```

Open your browser to `http://localhost:8000` and verify everything works.

## ⚙️ GitHub Pages Configuration

### Step 1: Enable GitHub Pages

1. **Navigate to Repository Settings**
   ```
   GitHub Repository → Settings → Pages (left sidebar)
   ```

2. **Configure Source**
   - **Source**: Select "GitHub Actions"
   - **Branch**: Leave as default (managed by Actions)
   - Click **Save**

   ![GitHub Pages Settings](https://docs.github.com/assets/images/help/pages/select-github-actions-source.png)

### Step 2: Set Repository Permissions

1. **Navigate to Actions Settings**
   ```
   GitHub Repository → Settings → Actions → General
   ```

2. **Workflow Permissions**
   - Select: ✅ "Read and write permissions"
   - Check: ✅ "Allow GitHub Actions to create and approve pull requests"
   - Click **Save**

   This allows the workflow to deploy to GitHub Pages.

### Step 3: Verify GitHub Pages is Active

After configuration, GitHub will show:

```
✅ Your site is published at https://miiwins.github.io/ottomator-agents/
```

## 🔄 GitHub Actions CI/CD

The repository includes an automated CI/CD pipeline that:

1. ✅ **Validates** HTML, CSS, and JavaScript
2. 🔨 **Builds** the static site
3. 🚀 **Deploys** to GitHub Pages
4. 📊 **Tests** performance with Lighthouse

### Workflow Triggers

The deployment workflow triggers on:

- **Push to main branch** (automatic deployment)
- **Pull requests to main** (validation only, no deployment)
- **Manual trigger** (via GitHub Actions UI)

### Workflow File

Location: `.github/workflows/deploy-pages.yml`

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
    paths: ['www/**']
  workflow_dispatch:

jobs:
  validate:
    # Validates HTML/CSS/JS
  build:
    # Prepares files for deployment
  deploy:
    # Deploys to GitHub Pages
  lighthouse:
    # Performance testing
```

### Manual Deployment

To manually trigger deployment:

1. Go to **Actions** tab in GitHub
2. Select **Deploy to GitHub Pages** workflow
3. Click **Run workflow**
4. Select branch: `main`
5. Click **Run workflow**

## 🌐 Custom Domain Setup (Optional)

### Step 1: Prepare Domain

1. **Edit CNAME file** in `www/CNAME`:
   ```
   example.com
   ```
   Or for subdomain:
   ```
   www.example.com
   ```

2. **Commit and push**:
   ```bash
   git add www/CNAME
   git commit -m "Add custom domain"
   git push origin main
   ```

### Step 2: Configure DNS

At your domain registrar (e.g., Namecheap, GoDaddy, Cloudflare):

**For Apex Domain (example.com):**

```
Type: A
Name: @
Value: 185.199.108.153
```

```
Type: A
Name: @
Value: 185.199.109.153
```

```
Type: A
Name: @
Value: 185.199.110.153
```

```
Type: A
Name: @
Value: 185.199.111.153
```

**For Subdomain (www.example.com):**

```
Type: CNAME
Name: www
Value: miiwins.github.io
```

### Step 3: Enable HTTPS

1. Go to **Settings** → **Pages**
2. Check: ✅ **Enforce HTTPS**
3. Wait for SSL certificate to provision (can take 24 hours)

### Step 4: Verify Domain

In GitHub Pages settings:

1. Enter your custom domain in the "Custom domain" field
2. Click **Save**
3. Wait for DNS check to complete (green checkmark)

## 📝 Deployment Workflow

### Standard Deployment Process

```bash
# 1. Make changes to website files
cd www
# Edit index.html, css/styles.css, js/main.js, etc.

# 2. Test locally
python -m http.server 8000
# Visit http://localhost:8000

# 3. Stage changes
git add www/

# 4. Commit with descriptive message
git commit -m "Update agent showcase section"

# 5. Push to main branch (triggers automatic deployment)
git push origin main

# 6. Monitor deployment
# Go to GitHub → Actions tab
# Watch the workflow progress

# 7. Verify deployment
# Visit https://miiwins.github.io/ottomator-agents/
```

### Deployment Timeline

```
Push to main
    ↓
GitHub Actions triggered (instant)
    ↓
Validation job (~1 min)
    ↓
Build job (~30 sec)
    ↓
Deploy job (~30 sec)
    ↓
Site live (~5 min total)
    ↓
Lighthouse testing (~2 min)
```

### Rollback Process

If a deployment breaks the site:

```bash
# Option 1: Revert the last commit
git revert HEAD
git push origin main

# Option 2: Reset to previous commit
git log  # Find the good commit SHA
git reset --hard <commit-sha>
git push origin main --force

# Option 3: Redeploy a specific commit
git checkout <commit-sha> -- www/
git commit -m "Rollback to working version"
git push origin main
```

## 📊 Monitoring & Maintenance

### Monitor Deployment Status

**GitHub Actions Tab:**
- View all workflow runs
- Check for failures
- Download artifacts
- View logs

**GitHub Pages Settings:**
- See current deployment status
- View deployment history
- Check custom domain status

### Performance Monitoring

The workflow includes Lighthouse CI for automated performance testing:

- **Performance Score**: Target > 90
- **Accessibility Score**: Target > 90
- **Best Practices Score**: Target > 90
- **SEO Score**: Target > 90

View Lighthouse reports:
1. Go to **Actions** tab
2. Click on latest workflow run
3. Scroll to **Lighthouse CI** job
4. Click on report link

### Update Schedule

Recommended maintenance schedule:

- **Weekly**: Check for broken links
- **Monthly**: Review performance metrics
- **Quarterly**: Update dependencies and content
- **Annually**: Review and refresh design

### Content Updates

To add new agents or update content:

1. **Edit `www/index.html`**
   - Add agent cards in appropriate sections
   - Update statistics in hero section
   - Add new categories if needed

2. **Update styles** (if needed)
   - Edit `www/css/styles.css`
   - Maintain design consistency

3. **Test changes locally**
   ```bash
   python -m http.server 8000
   ```

4. **Deploy**
   ```bash
   git add www/
   git commit -m "Add new agent: [name]"
   git push origin main
   ```

## 🔧 Troubleshooting

### Issue: Site Not Deploying

**Symptoms**: Pushed to main but site not updating

**Solutions**:

1. **Check GitHub Actions**
   ```
   Actions tab → Latest workflow → Check for errors
   ```

2. **Verify GitHub Pages is enabled**
   ```
   Settings → Pages → Source should be "GitHub Actions"
   ```

3. **Check workflow file syntax**
   ```bash
   # Validate YAML syntax
   yamllint .github/workflows/deploy-pages.yml
   ```

4. **Verify permissions**
   ```
   Settings → Actions → General → Workflow permissions
   Should be: "Read and write permissions"
   ```

### Issue: 404 Error on GitHub Pages

**Symptoms**: Site shows 404 Not Found

**Solutions**:

1. **Verify .nojekyll file exists**
   ```bash
   ls www/.nojekyll  # Should exist
   ```

2. **Check file paths**
   - Ensure all paths are relative
   - GitHub Pages is case-sensitive
   - Example: `css/styles.css` NOT `CSS/Styles.css`

3. **Verify index.html exists in www/**
   ```bash
   ls www/index.html  # Should exist
   ```

### Issue: CSS/JS Not Loading

**Symptoms**: Site loads but no styling

**Solutions**:

1. **Check browser console**
   - Press F12
   - Look for 404 errors
   - Verify file paths

2. **Verify file paths in HTML**
   ```html
   <!-- Correct -->
   <link rel="stylesheet" href="css/styles.css">
   <script src="js/main.js"></script>

   <!-- Incorrect -->
   <link rel="stylesheet" href="/css/styles.css">
   <script src="/js/main.js"></script>
   ```

3. **Clear browser cache**
   - Hard refresh: Ctrl+Shift+R (Windows/Linux)
   - Hard refresh: Cmd+Shift+R (Mac)

### Issue: Custom Domain Not Working

**Symptoms**: Custom domain shows error

**Solutions**:

1. **Verify DNS propagation**
   ```bash
   # Check DNS
   dig example.com
   nslookup example.com
   ```
   DNS can take 24-48 hours to propagate.

2. **Verify CNAME file**
   ```bash
   cat www/CNAME  # Should contain only your domain
   ```

3. **Check GitHub Pages settings**
   ```
   Settings → Pages → Custom domain
   Should show green checkmark
   ```

4. **Disable and re-enable custom domain**
   - Remove domain in settings
   - Wait 5 minutes
   - Re-add domain

### Issue: Workflow Failing

**Symptoms**: GitHub Actions shows red X

**Solutions**:

1. **Check workflow logs**
   ```
   Actions tab → Failed workflow → View logs
   ```

2. **Common failures**:

   **Validation errors:**
   ```
   Solution: Fix HTML/CSS/JS syntax errors
   ```

   **Permission errors:**
   ```
   Solution: Settings → Actions → Enable write permissions
   ```

   **Artifact upload errors:**
   ```
   Solution: Check www/ directory exists and has content
   ```

3. **Test workflow locally** (optional)
   ```bash
   # Install act (GitHub Actions local runner)
   brew install act  # Mac
   # or
   curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash

   # Run workflow locally
   act -j build
   ```

### Issue: Slow Load Times

**Symptoms**: Site takes too long to load

**Solutions**:

1. **Optimize images**
   - Compress images
   - Use appropriate formats (WebP, AVIF)
   - Implement lazy loading

2. **Minify assets**
   ```bash
   # Install minification tools
   npm install -g html-minifier clean-css-cli terser

   # Minify HTML
   html-minifier --collapse-whitespace --remove-comments www/index.html -o www/index.min.html

   # Minify CSS
   cleancss -o www/css/styles.min.css www/css/styles.css

   # Minify JS
   terser www/js/main.js -o www/js/main.min.js
   ```

3. **Use CDN** (optional)
   - Implement Cloudflare
   - Enable caching
   - Use compression

### Issue: Mobile Display Issues

**Symptoms**: Site broken on mobile devices

**Solutions**:

1. **Test responsiveness**
   - Chrome DevTools → Toggle device toolbar
   - Test multiple viewport sizes

2. **Check viewport meta tag**
   ```html
   <meta name="viewport" content="width=device-width, initial-scale=1.0">
   ```

3. **Verify CSS media queries**
   ```css
   @media (max-width: 768px) {
       /* Mobile styles */
   }
   ```

## 📞 Getting Help

If you encounter issues not covered here:

1. **GitHub Issues**
   ```
   https://github.com/miiwins/ottomator-agents/issues
   ```

2. **GitHub Discussions**
   ```
   https://github.com/miiwins/ottomator-agents/discussions
   ```

3. **GitHub Pages Documentation**
   ```
   https://docs.github.com/en/pages
   ```

4. **GitHub Actions Documentation**
   ```
   https://docs.github.com/en/actions
   ```

## 🎉 Success Checklist

After deployment, verify:

- [ ] Site accessible at GitHub Pages URL
- [ ] All pages load correctly
- [ ] Images and assets load
- [ ] Navigation works
- [ ] Links are functional
- [ ] Mobile responsive
- [ ] Custom domain working (if configured)
- [ ] HTTPS enabled
- [ ] GitHub Actions passing
- [ ] Lighthouse score > 90

## 📚 Additional Resources

- [GitHub Pages Documentation](https://docs.github.com/en/pages)
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Custom Domain Configuration](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site)
- [Lighthouse Documentation](https://developers.google.com/web/tools/lighthouse)
- [Web Performance Best Practices](https://web.dev/fast/)

---

**Happy Deploying! 🚀**

For questions or support, please open an issue in the repository.
