/**
 * ULTIMATE ORCHESTRATOR - FRONTEND APPLICATION
 * 4D Futuristic Voice-Enabled AI Interface
 */

// Configuration
const CONFIG = {
    apiUrl: window.location.origin + '/api',
    wsUrl: (window.location.protocol === 'https:' ? 'wss://' : 'ws://') + window.location.host + '/ws',
    liveKitUrl: 'ws://localhost:7880' // Will be replaced with actual LiveKit server
};

// State
const state = {
    voiceActive: false,
    liveKitRoom: null,
    audioTrack: null,
    conversation: [],
    agents: [],
    stats: {
        requests: 0,
        activeAgents: 70,
        avgResponse: 0.8,
        uptime: 99.9
    }
};

// DOM Elements
let elements = {};

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', init);

async function init() {
    console.log('🚀 Initializing Ultimate Orchestrator...');

    // Cache DOM elements
    cacheElements();

    // Setup event listeners
    setupEventListeners();

    // Initialize visualizations
    initVoiceVisualizer();
    initParticles();

    // Load agents
    await loadAgents();

    // Hide loading overlay
    setTimeout(() => {
        document.getElementById('loadingOverlay').classList.add('hidden');
        showToast('System initialized successfully', 'success');
    }, 2000);
}

function cacheElements() {
    elements = {
        voiceButton: document.getElementById('voiceButton'),
        voiceStatus: document.getElementById('voiceStatus'),
        voiceCanvas: document.getElementById('voiceCanvas'),
        messages: document.getElementById('messages'),
        clearTranscript: document.getElementById('clearTranscript'),
        agentGrid: document.getElementById('agentGrid'),
        agentSearch: document.getElementById('agentSearch'),
        requestCount: document.getElementById('requestCount'),
        toastContainer: document.getElementById('toastContainer')
    };
}

function setupEventListeners() {
    // Voice button
    elements.voiceButton.addEventListener('click', toggleVoice);

    // Clear transcript
    elements.clearTranscript.addEventListener('click', clearTranscript);

    // Agent search
    elements.agentSearch.addEventListener('input', filterAgents);

    // Keyboard shortcuts
    document.addEventListener('keydown', handleKeyboard);
}

// ===== VOICE INTERFACE =====

async function toggleVoice() {
    if (!state.voiceActive) {
        await startVoice();
    } else {
        await stopVoice();
    }
}

async function startVoice() {
    console.log('🎙️ Starting voice interface...');

    try {
        // Request microphone permission
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

        // Connect to LiveKit
        await connectLiveKit();

        // Update UI
        state.voiceActive = true;
        elements.voiceButton.classList.add('active');
        elements.voiceButton.querySelector('.voice-text').textContent = 'LISTENING...';
        updateVoiceStatus('Listening', 'online');

        // Add system message
        addMessage('system', '🎙️ Voice mode activated. Start speaking!');

        showToast('Voice mode activated', 'success');

        // Start audio analysis
        startAudioAnalysis(stream);

    } catch (error) {
        console.error('Voice activation failed:', error);
        showToast('Microphone access denied', 'error');
    }
}

async function stopVoice() {
    console.log('🛑 Stopping voice interface...');

    // Disconnect LiveKit
    if (state.liveKitRoom) {
        state.liveKitRoom.disconnect();
        state.liveKitRoom = null;
    }

    // Update UI
    state.voiceActive = false;
    elements.voiceButton.classList.remove('active');
    elements.voiceButton.querySelector('.voice-text').textContent = 'ACTIVATE VOICE';
    updateVoiceStatus('Ready', 'idle');

    showToast('Voice mode deactivated', 'success');
}

async function connectLiveKit() {
    try {
        // Request session from backend
        const response = await fetch(`${CONFIG.apiUrl}/voice/create-session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        const session = await response.json();

        if (!session.success) {
            throw new Error('Failed to create voice session');
        }

        // Connect to LiveKit room
        const room = new LivekitClient.Room();

        room.on(LivekitClient.RoomEvent.TrackSubscribed, handleTrackSubscribed);
        room.on(LivekitClient.RoomEvent.TrackUnsubscribed, handleTrackUnsubscribed);
        room.on(LivekitClient.RoomEvent.Disconnected, handleDisconnected);

        await room.connect(session.session.url, session.session.token);

        state.liveKitRoom = room;

        console.log('✅ Connected to LiveKit room:', session.session.room_name);

    } catch (error) {
        console.error('LiveKit connection failed:', error);

        // Fallback to WebSocket
        connectWebSocket();
    }
}

function connectWebSocket() {
    console.log('📡 Connecting to WebSocket...');

    const ws = new WebSocket(`${CONFIG.wsUrl}/voice/user-${Date.now()}`);

    ws.onopen = () => {
        console.log('✅ WebSocket connected');
        state.ws = ws;
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === 'response') {
            addMessage('ai', data.data.text);

            // Play TTS audio if available
            if (data.data.audio) {
                playAudio(data.data.audio);
            }
        }
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        showToast('Connection error', 'error');
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        state.ws = null;
    };
}

function handleTrackSubscribed(track, publication, participant) {
    console.log('Track subscribed:', track.kind);

    if (track.kind === 'audio') {
        const audioElement = track.attach();
        document.body.appendChild(audioElement);
    }
}

function handleTrackUnsubscribed(track) {
    track.detach().forEach(element => element.remove());
}

function handleDisconnected() {
    console.log('Disconnected from LiveKit');
    stopVoice();
}

// ===== AUDIO ANALYSIS =====

function startAudioAnalysis(stream) {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = audioContext.createAnalyser();
    const microphone = audioContext.createMediaStreamSource(stream);

    analyser.fftSize = 256;
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    microphone.connect(analyser);

    // Visualize audio
    function visualize() {
        if (!state.voiceActive) return;

        analyser.getByteFrequencyData(dataArray);

        // Update wave bars
        const waveBars = document.querySelectorAll('.wave-bar');
        for (let i = 0; i < waveBars.length; i++) {
            const index = Math.floor(i * bufferLength / waveBars.length);
            const value = dataArray[index];
            const height = (value / 255) * 60;
            waveBars[i].style.height = `${Math.max(20, height)}px`;
        }

        // Update canvas visualization
        updateVoiceCanvas(dataArray);

        requestAnimationFrame(visualize);
    }

    visualize();
}

function updateVoiceCanvas(dataArray) {
    const canvas = elements.voiceCanvas;
    const ctx = canvas.getContext('2d');

    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;

    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) * 0.6;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw circular visualizer
    ctx.beginPath();
    for (let i = 0; i < dataArray.length; i++) {
        const angle = (i / dataArray.length) * Math.PI * 2;
        const value = dataArray[i] / 255;
        const r = radius + (value * 50);

        const x = centerX + Math.cos(angle) * r;
        const y = centerY + Math.sin(angle) * r;

        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
    }

    ctx.closePath();
    ctx.strokeStyle = 'rgba(0, 240, 255, 0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Add glow effect
    ctx.shadowBlur = 20;
    ctx.shadowColor = 'rgba(0, 240, 255, 0.8)';
    ctx.stroke();
}

// ===== MESSAGE HANDLING =====

function addMessage(type, text) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;

    const icon = type === 'user' ? '👤' : type === 'ai' ? '🤖' : 'ℹ️';

    messageDiv.innerHTML = `
        <span class="message-icon">${icon}</span>
        <span>${text}</span>
    `;

    elements.messages.appendChild(messageDiv);
    elements.messages.scrollTop = elements.messages.scrollHeight;

    // Add to state
    state.conversation.push({ type, text, timestamp: Date.now() });
}

function clearTranscript() {
    elements.messages.innerHTML = `
        <div class="message system">
            <span class="message-icon">🤖</span>
            <span>Say "Hello" to activate voice mode and start talking to 70+ AI agents</span>
        </div>
    `;
    state.conversation = [];
}

function updateVoiceStatus(text, status) {
    elements.voiceStatus.innerHTML = `
        <span class="status-dot ${status === 'online' ? 'status-online' : ''}"></span>
        <span>${text}</span>
    `;
}

// ===== AGENT MANAGEMENT =====

async function loadAgents() {
    console.log('📦 Loading agents...');

    try {
        const response = await fetch(`${CONFIG.apiUrl}/agents/list`);
        const data = await response.json();

        state.agents = data.agents || [];
        renderAgents(state.agents);

        console.log(`✅ Loaded ${state.agents.length} agents`);

    } catch (error) {
        console.error('Failed to load agents:', error);

        // Load sample agents
        state.agents = getSampleAgents();
        renderAgents(state.agents);
    }
}

function renderAgents(agents) {
    elements.agentGrid.innerHTML = agents.map(agent => `
        <div class="agent-item" onclick="selectAgent('${agent.name}')">
            <div class="agent-icon">${agent.icon || '🤖'}</div>
            <div class="agent-name">${agent.name}</div>
            <div class="agent-category">${agent.category}</div>
        </div>
    `).join('');
}

function filterAgents() {
    const query = elements.agentSearch.value.toLowerCase();
    const filtered = state.agents.filter(agent =>
        agent.name.toLowerCase().includes(query) ||
        agent.category.toLowerCase().includes(query)
    );
    renderAgents(filtered);
}

function selectAgent(agentName) {
    console.log('Selected agent:', agentName);
    showToast(`Agent selected: ${agentName}`, 'success');
    addMessage('system', `🎯 Switched to ${agentName} agent`);
}

function getSampleAgents() {
    return [
        { name: 'Code Generator', category: 'CODE_GENERATION', icon: '💻' },
        { name: 'Content Writer', category: 'CONTENT_CREATION', icon: '✍️' },
        { name: 'Research Assistant', category: 'RESEARCH', icon: '🔬' },
        { name: 'Data Analyst', category: 'DATA_ANALYSIS', icon: '📊' },
        { name: 'API Integrator', category: 'INTEGRATION', icon: '🔌' },
        { name: 'Security Auditor', category: 'SPECIALIZED', icon: '🔒' },
        { name: 'Performance Optimizer', category: 'SPECIALIZED', icon: '⚡' },
        { name: 'Database Architect', category: 'CODE_GENERATION', icon: '🗄️' },
        { name: 'Voice Agent', category: 'VOICE', icon: '🎙️' },
        { name: 'RAG Assistant', category: 'RAG', icon: '📚' },
        { name: 'Deployment Manager', category: 'AUTOMATION', icon: '🚀' },
        { name: 'Testing Strategist', category: 'CODE_GENERATION', icon: '🧪' }
    ];
}

// ===== PARTICLE SYSTEM =====

function initParticles() {
    const canvas = document.getElementById('particles');
    const ctx = canvas.getContext('2d');

    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const particles = [];
    const particleCount = 100;

    class Particle {
        constructor() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.vx = (Math.random() - 0.5) * 0.5;
            this.vy = (Math.random() - 0.5) * 0.5;
            this.radius = Math.random() * 2;
        }

        update() {
            this.x += this.vx;
            this.y += this.vy;

            if (this.x < 0 || this.x > canvas.width) this.vx *= -1;
            if (this.y < 0 || this.y > canvas.height) this.vy *= -1;
        }

        draw() {
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
            ctx.fillStyle = 'rgba(0, 240, 255, 0.5)';
            ctx.fill();
        }
    }

    // Create particles
    for (let i = 0; i < particleCount; i++) {
        particles.push(new Particle());
    }

    // Animation loop
    function animate() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Update and draw particles
        particles.forEach(particle => {
            particle.update();
            particle.draw();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < 100) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 240, 255, ${1 - distance / 100})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        requestAnimationFrame(animate);
    }

    animate();

    // Resize handler
    window.addEventListener('resize', () => {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    });
}

function initVoiceVisualizer() {
    // Initial static visualization
    const canvas = elements.voiceCanvas;
    const ctx = canvas.getContext('2d');

    function drawStatic() {
        canvas.width = canvas.offsetWidth;
        canvas.height = canvas.offsetHeight;

        const centerX = canvas.width / 2;
        const centerY = canvas.height / 2;
        const radius = Math.min(centerX, centerY) * 0.6;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw static circle
        ctx.beginPath();
        ctx.arc(centerX, centerY, radius, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)';
        ctx.lineWidth = 2;
        ctx.stroke();
    }

    drawStatic();
    window.addEventListener('resize', drawStatic);
}

// ===== UTILITY FUNCTIONS =====

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;

    elements.toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function handleKeyboard(e) {
    // Space bar to toggle voice
    if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        toggleVoice();
    }

    // Escape to stop voice
    if (e.code === 'Escape' && state.voiceActive) {
        stopVoice();
    }
}

// ===== QUICK ACTIONS =====

function openDocs() {
    window.open(`${CONFIG.apiUrl}/docs`, '_blank');
}

function openAPI() {
    window.open(`${CONFIG.apiUrl}/docs`, '_blank');
}

function openMonitoring() {
    showToast('Monitoring dashboard coming soon', 'success');
}

function openSettings() {
    showToast('Settings panel coming soon', 'success');
}

// ===== EXPORT FOR GLOBAL ACCESS =====

window.app = {
    toggleVoice,
    addMessage,
    selectAgent,
    showToast,
    state
};

console.log('✅ Ultimate Orchestrator initialized');
