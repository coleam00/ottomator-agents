/**
 * ADVANCED 3D PARTICLE SYSTEM
 * Creates stunning visual effects with WebGL-powered particles
 */

class Particle3D {
    constructor(x, y, z) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.vx = (Math.random() - 0.5) * 2;
        this.vy = (Math.random() - 0.5) * 2;
        this.vz = (Math.random() - 0.5) * 2;
        this.size = Math.random() * 3 + 1;
        this.color = this.randomColor();
        this.life = 1.0;
        this.decay = Math.random() * 0.01 + 0.005;
    }

    randomColor() {
        const colors = [
            'rgba(0, 240, 255, 0.8)',
            'rgba(255, 0, 255, 0.8)',
            'rgba(0, 255, 136, 0.8)',
            'rgba(255, 255, 255, 0.5)'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    update(centerX, centerY, centerZ) {
        // Apply forces towards center (orbital motion)
        const dx = centerX - this.x;
        const dy = centerY - this.y;
        const dz = centerZ - this.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

        if (distance > 1) {
            this.vx += (dx / distance) * 0.01;
            this.vy += (dy / distance) * 0.01;
            this.vz += (dz / distance) * 0.01;
        }

        // Apply velocity with damping
        this.vx *= 0.99;
        this.vy *= 0.99;
        this.vz *= 0.99;

        this.x += this.vx;
        this.y += this.vy;
        this.z += this.vz;

        // Fade out
        this.life -= this.decay;

        return this.life > 0;
    }

    project(width, height, fov) {
        // 3D to 2D projection
        const scale = fov / (fov + this.z);
        const x2d = this.x * scale + width / 2;
        const y2d = this.y * scale + height / 2;
        const size2d = this.size * scale;

        return { x: x2d, y: y2d, size: size2d, brightness: this.life };
    }
}

class ParticleSystem3D {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.maxParticles = 200;
        this.fov = 500;
        this.centerX = 0;
        this.centerY = 0;
        this.centerZ = 0;
        this.rotation = 0;

        this.init();
        this.animate();
    }

    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());

        // Create initial particles
        for (let i = 0; i < this.maxParticles; i++) {
            this.createParticle();
        }
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticle() {
        const angle = Math.random() * Math.PI * 2;
        const radius = Math.random() * 300 + 100;
        const height = (Math.random() - 0.5) * 400;

        const x = Math.cos(angle) * radius;
        const y = height;
        const z = Math.sin(angle) * radius;

        this.particles.push(new Particle3D(x, y, z));
    }

    update() {
        // Rotate center point
        this.rotation += 0.002;

        // Update particles
        this.particles = this.particles.filter(particle =>
            particle.update(this.centerX, this.centerY, this.centerZ)
        );

        // Maintain particle count
        while (this.particles.length < this.maxParticles) {
            this.createParticle();
        }
    }

    draw() {
        this.ctx.fillStyle = 'rgba(10, 10, 20, 0.1)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        // Sort particles by z-index (depth)
        this.particles.sort((a, b) => b.z - a.z);

        // Draw particles
        this.particles.forEach(particle => {
            const projected = particle.project(this.canvas.width, this.canvas.height, this.fov);

            if (projected.x > 0 && projected.x < this.canvas.width &&
                projected.y > 0 && projected.y < this.canvas.height) {

                this.ctx.beginPath();
                this.ctx.arc(projected.x, projected.y, projected.size, 0, Math.PI * 2);

                const alpha = projected.brightness;
                this.ctx.fillStyle = particle.color.replace('0.8', alpha);
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = particle.color;
                this.ctx.fill();
            }
        });

        // Draw connections between nearby particles
        this.drawConnections();
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const p1 = this.particles[i];
                const p2 = this.particles[j];

                const dx = p1.x - p2.x;
                const dy = p1.y - p2.y;
                const dz = p1.z - p2.z;
                const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

                if (distance < 150) {
                    const proj1 = p1.project(this.canvas.width, this.canvas.height, this.fov);
                    const proj2 = p2.project(this.canvas.width, this.canvas.height, this.fov);

                    this.ctx.beginPath();
                    this.ctx.moveTo(proj1.x, proj1.y);
                    this.ctx.lineTo(proj2.x, proj2.y);

                    const alpha = (1 - distance / 150) * 0.3;
                    this.ctx.strokeStyle = `rgba(0, 240, 255, ${alpha})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.stroke();
                }
            }
        }
    }

    animate() {
        this.update();
        this.draw();
        requestAnimationFrame(() => this.animate());
    }
}

// Auto-initialize if particles canvas exists
document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('particles');
    if (canvas) {
        new ParticleSystem3D(canvas);
        console.log('✨ 3D Particle system initialized');
    }
});
