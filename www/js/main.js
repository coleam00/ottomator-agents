// ===========================
// Mobile Navigation
// ===========================
const mobileMenuToggle = document.querySelector('.mobile-menu-toggle');
const navLinks = document.querySelector('.nav-links');

if (mobileMenuToggle) {
    mobileMenuToggle.addEventListener('click', () => {
        navLinks.classList.toggle('active');
        mobileMenuToggle.classList.toggle('active');
    });

    // Close menu when clicking on a link
    const navItems = document.querySelectorAll('.nav-links a');
    navItems.forEach(item => {
        item.addEventListener('click', () => {
            navLinks.classList.remove('active');
            mobileMenuToggle.classList.remove('active');
        });
    });
}

// ===========================
// Smooth Scrolling
// ===========================
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        const href = this.getAttribute('href');
        if (href !== '#' && href.length > 1) {
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const offsetTop = target.offsetTop - 80;
                window.scrollTo({
                    top: offsetTop,
                    behavior: 'smooth'
                });
            }
        }
    });
});

// ===========================
// Scroll Animations
// ===========================
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in-up');
            observer.unobserve(entry.target);
        }
    });
}, observerOptions);

// Observe all cards
const cards = document.querySelectorAll('.learning-card, .agent-card, .use-case-card, .tech-category');
cards.forEach(card => {
    observer.observe(card);
});

// ===========================
// Navbar Scroll Effect
// ===========================
let lastScroll = 0;
const navbar = document.querySelector('.navbar');

window.addEventListener('scroll', () => {
    const currentScroll = window.pageYOffset;

    if (currentScroll <= 0) {
        navbar.classList.remove('scroll-up');
        return;
    }

    if (currentScroll > lastScroll && !navbar.classList.contains('scroll-down')) {
        // Scrolling down
        navbar.classList.remove('scroll-up');
        navbar.classList.add('scroll-down');
    } else if (currentScroll < lastScroll && navbar.classList.contains('scroll-down')) {
        // Scrolling up
        navbar.classList.remove('scroll-down');
        navbar.classList.add('scroll-up');
    }

    lastScroll = currentScroll;
});

// ===========================
// Dynamic Stats Counter
// ===========================
function animateCounter(element, target, duration = 2000) {
    let start = 0;
    const increment = target / (duration / 16);
    const timer = setInterval(() => {
        start += increment;
        if (start >= target) {
            element.textContent = target + (element.dataset.suffix || '');
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(start) + (element.dataset.suffix || '');
        }
    }, 16);
}

// Animate stats when they come into view
const statsObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const statNumbers = entry.target.querySelectorAll('.stat-number');
            statNumbers.forEach(stat => {
                const target = stat.textContent.replace(/\D/g, '');
                if (target) {
                    stat.dataset.suffix = stat.textContent.replace(/\d/g, '');
                    animateCounter(stat, parseInt(target));
                }
            });
            statsObserver.unobserve(entry.target);
        }
    });
}, { threshold: 0.5 });

const heroStats = document.querySelector('.hero-stats');
if (heroStats) {
    statsObserver.observe(heroStats);
}

// ===========================
// Filter Functionality (Future Enhancement)
// ===========================
// This can be enhanced to add filtering by technology, difficulty, etc.
const filterButtons = document.querySelectorAll('[data-filter]');
if (filterButtons.length > 0) {
    filterButtons.forEach(button => {
        button.addEventListener('click', () => {
            const filter = button.dataset.filter;
            const cards = document.querySelectorAll('[data-category]');

            cards.forEach(card => {
                if (filter === 'all' || card.dataset.category === filter) {
                    card.style.display = 'block';
                    card.classList.add('fade-in-up');
                } else {
                    card.style.display = 'none';
                }
            });

            // Update active button
            filterButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');
        });
    });
}

// ===========================
// Search Functionality (Future Enhancement)
// ===========================
const searchInput = document.querySelector('[data-search]');
if (searchInput) {
    searchInput.addEventListener('input', (e) => {
        const searchTerm = e.target.value.toLowerCase();
        const cards = document.querySelectorAll('.agent-card');

        cards.forEach(card => {
            const title = card.querySelector('h4').textContent.toLowerCase();
            const description = card.querySelector('p').textContent.toLowerCase();
            const tags = Array.from(card.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());

            const matches = title.includes(searchTerm) ||
                          description.includes(searchTerm) ||
                          tags.some(tag => tag.includes(searchTerm));

            card.style.display = matches ? 'block' : 'none';
        });
    });
}

// ===========================
// Copy Code Button (Future Enhancement)
// ===========================
const copyButtons = document.querySelectorAll('[data-copy]');
copyButtons.forEach(button => {
    button.addEventListener('click', async () => {
        const code = button.dataset.copy;
        try {
            await navigator.clipboard.writeText(code);
            button.textContent = 'Copied!';
            setTimeout(() => {
                button.textContent = 'Copy';
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    });
});

// ===========================
// Load More Functionality (Future Enhancement)
// ===========================
const loadMoreButtons = document.querySelectorAll('[data-load-more]');
loadMoreButtons.forEach(button => {
    button.addEventListener('click', () => {
        const target = button.dataset.loadMore;
        const hiddenItems = document.querySelectorAll(`[data-group="${target}"][data-hidden="true"]`);

        hiddenItems.forEach((item, index) => {
            if (index < 6) { // Load 6 more items
                item.dataset.hidden = 'false';
                item.style.display = 'block';
                item.classList.add('fade-in-up');
            }
        });

        // Hide button if no more items
        const remainingItems = document.querySelectorAll(`[data-group="${target}"][data-hidden="true"]`);
        if (remainingItems.length === 0) {
            button.style.display = 'none';
        }
    });
});

// ===========================
// External Links (Open in new tab)
// ===========================
document.querySelectorAll('a[href^="http"]').forEach(link => {
    if (!link.hasAttribute('target')) {
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    }
});

// ===========================
// Console Message
// ===========================
console.log('%c🤖 Live Agent Studio', 'font-size: 20px; font-weight: bold; color: #5551ff;');
console.log('%cExplore 71+ AI Agent implementations', 'font-size: 14px; color: #6c757d;');
console.log('%cGitHub: https://github.com/miiwins/ottomator-agents', 'font-size: 12px; color: #5551ff;');

// ===========================
// Performance Monitoring
// ===========================
if (window.performance) {
    window.addEventListener('load', () => {
        setTimeout(() => {
            const perfData = window.performance.timing;
            const pageLoadTime = perfData.loadEventEnd - perfData.navigationStart;
            console.log(`Page loaded in ${pageLoadTime}ms`);
        }, 0);
    });
}
