# Live Agent Studio - Static Website

A beautiful, modern static website showcasing 71+ AI agent implementations from the ottomator-agents repository. Built with vanilla HTML, CSS, and JavaScript, following the design aesthetic of writer.com/agents.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black)

## 🌟 Features

- **Modern Design**: Inspired by writer.com/agents with a clean, professional aesthetic
- **Responsive Layout**: Fully responsive design that works on all devices
- **71+ Agent Showcase**: Comprehensive listing of all AI agents organized by category
- **Fast & Lightweight**: Pure vanilla JavaScript, no frameworks, lightning-fast load times
- **SEO Optimized**: Semantic HTML and meta tags for better search engine visibility
- **Accessible**: WCAG compliant with proper ARIA labels and keyboard navigation
- **Smooth Animations**: Intersection Observer API for performant scroll animations
- **GitHub Pages Ready**: Optimized for deployment on GitHub Pages

## 📁 Structure

```
www/
├── index.html              # Main homepage
├── css/
│   └── styles.css          # All styles (CSS variables, components, responsive)
├── js/
│   └── main.js             # JavaScript functionality
├── assets/
│   └── images/             # Images and icons
├── .nojekyll               # GitHub Pages configuration
├── CNAME                   # Custom domain configuration (optional)
└── README.md               # This file
```

## 🎨 Design System

### Color Palette

```css
--primary-purple: #5551ff
--primary-dark: #2e2ae8
--primary-light: #7875ff
--accent-green: #83e6ad
--accent-pink: #ffebfc
```

### Typography

- **Primary Font**: Poppins (300, 400, 500, 600, 700)
- **Display Font**: Playfair Display (700, 900)
- **Base Size**: 16px with responsive scaling

### Components

- Hero Section with gradient background
- Learning Path cards with difficulty levels
- Agent category sections with filterable grids
- Technology showcase with glassmorphism effects
- Use case cards with hover animations

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone https://github.com/miiwins/ottomator-agents.git
   cd ottomator-agents/www
   ```

2. **Serve locally**

   **Option A: Using Python**
   ```bash
   python -m http.server 8000
   ```

   **Option B: Using Node.js**
   ```bash
   npx http-server -p 8000
   ```

   **Option C: Using PHP**
   ```bash
   php -S localhost:8000
   ```

3. **Open in browser**
   ```
   http://localhost:8000
   ```

### Production Deployment

The site is automatically deployed to GitHub Pages via GitHub Actions when changes are pushed to the `main` branch.

**GitHub Pages URL**: `https://miiwins.github.io/ottomator-agents/`

## 📦 Deployment Guide

### Setting Up GitHub Pages

1. **Navigate to Repository Settings**
   - Go to your GitHub repository
   - Click on **Settings** > **Pages**

2. **Configure Source**
   - **Source**: GitHub Actions
   - The deployment workflow will handle the rest

3. **Custom Domain (Optional)**
   - Edit `CNAME` file with your domain
   - Configure DNS settings at your domain registrar:
     ```
     Type: CNAME
     Name: www (or @)
     Value: miiwins.github.io
     ```

### GitHub Actions CI/CD

The repository includes a comprehensive CI/CD pipeline (`.github/workflows/deploy-pages.yml`):

**Pipeline Stages:**

1. **Validate** - HTML/CSS/JS linting
2. **Build** - Prepare static assets
3. **Deploy** - Deploy to GitHub Pages
4. **Lighthouse** - Performance testing

**Triggering Deployment:**

- **Automatic**: Push to `main` branch
- **Manual**: Via GitHub Actions UI (workflow_dispatch)

```bash
# Make changes
git add www/
git commit -m "Update website content"
git push origin main

# GitHub Actions will automatically deploy
```

### Environment Variables

No environment variables are required for the static site. All configuration is handled through:

- HTML meta tags
- CSS variables in `styles.css`
- JavaScript constants in `main.js`

## 🛠️ Customization

### Updating Content

**Add a New Agent:**

1. Open `index.html`
2. Find the appropriate category section
3. Add a new card following this template:

```html
<div class="agent-card">
    <div class="card-icon">🤖</div>
    <h4>Your Agent Name</h4>
    <p>Agent description goes here.</p>
    <div class="card-tags">
        <span class="tag">Tech 1</span>
        <span class="tag">Tech 2</span>
    </div>
    <a href="https://github.com/miiwins/ottomator-agents/tree/main/your-agent"
       target="_blank" class="card-link">View →</a>
</div>
```

### Changing Colors

Edit CSS variables in `www/css/styles.css`:

```css
:root {
    --primary-purple: #5551ff;  /* Change to your color */
    --primary-dark: #2e2ae8;    /* Change to your color */
    /* ... more variables ... */
}
```

### Adding Pages

1. Create new HTML file in `www/` directory
2. Link from navigation in `index.html`
3. Follow the same structure and styling

## 🧪 Testing

### Manual Testing

```bash
# Start local server
python -m http.server 8000

# Test in multiple browsers
- Chrome/Edge (Chromium)
- Firefox
- Safari

# Test responsive design
- Mobile (375px, 414px)
- Tablet (768px, 1024px)
- Desktop (1280px, 1920px)
```

### Automated Testing

The GitHub Actions workflow includes:

- **HTML Validation**: Using htmlhint
- **CSS Validation**: Using stylelint
- **JS Validation**: Using eslint
- **Performance**: Using Lighthouse CI

### Accessibility Testing

```bash
# Install pa11y
npm install -g pa11y

# Run accessibility tests
pa11y http://localhost:8000
```

## 📊 Performance

### Optimization Checklist

- ✅ Minified CSS and JS (can be added to build step)
- ✅ Lazy loading images with Intersection Observer
- ✅ Async font loading
- ✅ Optimized animations with CSS transforms
- ✅ No render-blocking resources
- ✅ Mobile-first responsive design

### Performance Metrics (Target)

- **First Contentful Paint**: < 1.5s
- **Time to Interactive**: < 3.0s
- **Lighthouse Score**: > 90
- **Bundle Size**: < 100KB (uncompressed)

## 🔧 Troubleshooting

### Site Not Deploying

1. Check GitHub Actions logs in the **Actions** tab
2. Verify GitHub Pages is enabled in Settings > Pages
3. Ensure source is set to "GitHub Actions"
4. Check branch protection rules

### Styles Not Loading

1. Verify file paths are correct (case-sensitive on Linux)
2. Check browser console for 404 errors
3. Clear browser cache
4. Check `.nojekyll` file exists

### Custom Domain Issues

1. Verify CNAME file contains only the domain
2. Check DNS propagation (can take 24-48 hours)
3. Enable "Enforce HTTPS" in GitHub Pages settings

## 📝 Browser Support

- Chrome/Edge (Chromium) - Latest 2 versions
- Firefox - Latest 2 versions
- Safari - Latest 2 versions
- Mobile Safari - iOS 12+
- Chrome Mobile - Latest

## 🤝 Contributing

Contributions are welcome! To update the website:

1. Fork the repository
2. Create a feature branch
3. Make your changes in the `www/` directory
4. Test locally
5. Submit a pull request

## 📄 License

This project is open source under the MIT License.

## 🔗 Links

- **Live Site**: https://miiwins.github.io/ottomator-agents/
- **GitHub Repository**: https://github.com/miiwins/ottomator-agents
- **Issues**: https://github.com/miiwins/ottomator-agents/issues

## 📧 Support

For questions or issues:

- Open a GitHub Issue
- Check the main repository README
- Review the GitHub Discussions

---

**Built with ❤️ for the AI Agent community**
