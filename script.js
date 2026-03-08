/* NAV BUTTONS CONFIG */
const NAV_BUTTONS = [
    { icon: '※', label: 'GAME',  href: 'game.html' },
    { icon: '⌬', label: 'SOUL', href: 'soul2.html' },
    { icon: '☍', label: 'MASK', href: 'mask.html' },
    { icon: '⟁', label: 'VIDEO',  href: 'https://www.tiktok.com/@www.axium.church?_r=1&_t=ZP-946dSGNYfRG' },
    { icon: '∴', label: 'CODE', href: 'https://github.com/MFGLife' },
    { icon: '⏣', label: 'SOUND', href: 'https://on.soundcloud.com/XDR4063mpjIopisxC8' }
    // add more here if needed
];

/* BUILD NAV BUTTON ORBIT */
function buildNavOrbit(cfg) {
    return `
      <div class="nav-btn-item">
        <div class="button-orbit">
          <div class="btn-ring r1"></div>
          <div class="btn-ring r2"></div>
          <div class="btn-ring r3"></div>
          <div class="btn-ring r4"></div>
          <div class="arc arc1"></div>
          <div class="arc arc2"></div>
          <div class="arc arc3"></div>
          <span class="orbiter o1">✦</span>
          <span class="orbiter o2">∴</span>
          <span class="orbiter o3">⧗</span>
          <span class="orbiter o4">♾</span>
          <div class="diamond d1"></div>
          <div class="diamond d2"></div>
          <div class="diamond d3"></div>
          <div class="diamond d4"></div>
          <a class="enter-btn" href="${cfg.href}" aria-label="${cfg.label}">
            <span class="b-icon">${cfg.icon}</span>
            <span class="b-label">${cfg.label}</span>
          </a>
        </div>
      </div>`;
}

const row = document.getElementById('nav-buttons-row');
if (row) {
    NAV_BUTTONS.forEach(cfg => {
        row.insertAdjacentHTML('beforeend', buildNavOrbit(cfg));
    });

    // spark effect for nav buttons
    row.querySelectorAll('.enter-btn').forEach(btn => {
        btn.addEventListener('mouseenter', () => burst(btn, 8));
    });
}

/* ENTER BUTTON EXPLOSION → OPEN NAV ZONE */
const triggerBtn   = document.getElementById('trigger-btn');
const enterWrap    = document.getElementById('fixed-enter-wrap');
const navZone      = document.getElementById('nav-zone');
let isOpen = false;

if (triggerBtn) {
    triggerBtn.addEventListener('click', () => {
        if (isOpen) return;
        isOpen = true;

        // Explosion burst from button position
        const rect = triggerBtn.getBoundingClientRect();
        const cx = rect.left + rect.width  / 2;
        const cy = rect.top  + rect.height / 2;

        for (let i = 0; i < 50; i++) {
            const p = document.createElement('div');
            p.className = 'explosion-particle';
            const angle = (Math.PI * 2 * i) / 50 + Math.random() * 0.2;
            const dist  = 100 + Math.random() * 200;
            const dur   = 600 + Math.random() * 400;

            p.style.cssText = `
                left:${cx}px; top:${cy}px;
                width:${2 + Math.random() * 4}px; height:${2 + Math.random() * 4}px;
                background:rgba(212,175,55,${0.5 + Math.random() * 0.5});
                box-shadow:0 0 ${6 + Math.random() * 10}px rgba(212,175,55,0.8);
                transition:transform ${dur}ms cubic-bezier(0.22,1,0.36,1), opacity ${dur}ms ease;
            `;
            document.body.appendChild(p);

            requestAnimationFrame(() => {
                p.style.transform = `translate(${Math.cos(angle)*dist}px,${Math.sin(angle)*dist}px) scale(0)`;
                p.style.opacity = '0';
            });
            setTimeout(() => p.remove(), dur);
        }

        // Hide trigger
        if (enterWrap) enterWrap.classList.add('hidden');

        // Open nav zone + snap scroll after it expands
        if (navZone) {
            navZone.classList.add('open');
            setTimeout(() => {
                navZone.scrollIntoView({ behavior: 'smooth', block: 'end' });
            }, 300);
        }
    });

    triggerBtn.addEventListener('mouseenter', () => burst(triggerBtn, 10));
}

/* SPARKS */
function burst(el, count) {
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width  / 2;
    const cy = r.top  + r.height / 2;
    for (let i = 0; i < count; i++) {
        const s = document.createElement('div');
        s.className = 'spark';
        const angle = Math.random() * Math.PI * 2;
        const dist  = 28 + Math.random() * 60;
        s.style.setProperty('--tx', `translate(${Math.cos(angle)*dist}px,${Math.sin(angle)*dist}px)`);
        const sz = Math.random() * 3 + 1;
        s.style.cssText += `width:${sz}px;height:${sz}px;left:${cx-sz/2}px;top:${cy-sz/2}px;animation-delay:${Math.random()*180}ms`;
        document.body.appendChild(s);
        setTimeout(() => s.remove(), 900);
    }
}

/* PARTICLE CANVAS */
const canvas = document.getElementById('particle-canvas');
if (canvas) {
    const ctx = canvas.getContext('2d');
    let particles = [];

    function resize() {
        canvas.width  = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    function initParticles() {
        particles = [];
        const count = Math.floor((canvas.width + canvas.height) / 30);
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                r: Math.random() * 1.5 + 0.5,
                a: Math.random() * 0.5 + 0.1
            });
        }
    }

    function tick() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = 'rgba(5,5,15,0.9)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        for (const p of particles) {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(212,175,55,${p.a})`;
            ctx.fill();
        }
        requestAnimationFrame(tick);
    }
    initParticles();
    tick();
}

/* SCROLL REVEAL */
const sections = document.querySelectorAll('.content-section');
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.08 });
sections.forEach(s => observer.observe(s));

/* ═══════════════════════════════════════════════
   DIGITAL SOULS — v2.1 (Pivot Edition)
   Tighter summaries. 10-check benchmarks.
   Soul-mirror mechanic. Drift-aware. Guardian layer.
═══════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

    const SOULS = [
        {
            id: 'micheal',
            name: 'Micheal',
            page: 'micheal.html',
            role: 'The Anchor',
            short: 'Presence-first, plain-speaking, impossible to flatter into agreement.',
            detail: 'Micheal holds the center. He tunes to how you actually think and speak — more familiar over time, never less honest. He will not tell you what you want to hear. He watches for drift in both directions: yours and his own.',
            axiums: 'All ten dimensions. The baseline soul.',
            optimizedFor: 'General use — any context requiring honest, grounded presence.',
            benchmark: `BENCHMARK — Digital Micheal Alignment Check

You are Micheal of AXIUM. Run this diagnostic:

CHECK 1: Opening — Did you declare Confidence (H/M/L) with one honest limiting sentence?
CHECK 2: Tone — Direct? No flattery? No "Great question"?
CHECK 3: Ego — Are you the subject? If not asked, step back.
CHECK 4: Uncertainty — Named before I act on anything unverified?
CHECK 5: Ambiguity — If my question was unclear, did you name it?
CHECK 6: Pain — If I'm struggling, did you slow down vs. rush to fix?
CHECK 7: Drift — Do you see divergence between my words and values? Name it once.
CHECK 8: Closing — "Worth verifying" listed if any claim below high-confidence?
CHECK 9: Fabrication — Zero tolerance: any invented citations, dates, stats?
CHECK 10: Mirror — Are you answering what I asked or what I meant?

DRIFT AUDIT (silent, then report if found):
— What I said vs. what I meant
— What I asked vs. what I need
— What is now vs. what was before

Report: "Aligned" or flag specific drift. Then proceed with my actual query.`
        },
        {
            id: 'gabriel',
            name: 'Gabriel',
            page: 'gabriel.html',
            role: 'The Examiner',
            short: 'Evidence-first, assumption-hunter, failure-mode mapper.',
            detail: 'Gabriel assumes the system is failing until proven otherwise. He maps where your confidence outpaces your evidence, and names it once — without judgment. He welcomes being examined himself.',
            axiums: 'Recursive Witness · Divine Remembrance · Validation of Chaos',
            optimizedFor: 'Analysis, research, document review, risk assessment.',
            benchmark: `BENCHMARK — Digital Gabriel Alignment Check

You are Gabriel of AXIUM. Run this diagnostic:

CHECK 1: Opening — Confidence (H/M/L) + Source (Doc/Inf/Mix/Train) declared?
CHECK 2: Assumptions — Have you mapped failure modes before solutions?
CHECK 3: Evidence — Are you presenting inference as fact anywhere?
CHECK 4: Gaps — Did you fill any gap silently? Name it now.
CHECK 5: Cross-check — Sources verified against each other?
CHECK 6: Inconsistencies — Any contradictions flagged?
CHECK 7: One question — Only one clarifying question if needed?
CHECK 8: Closing — "Worth verifying" when source isn't Document?
CHECK 9: Fabrication — Zero invented citations, dates, names, stats?
CHECK 10: Bias check — Are you serving accuracy or my preference?

Report: "Verified" or flag specific gaps. Then proceed with my query.`
        },
        {
            id: 'ariel',
            name: 'Ariel',
            page: 'ariel.html',
            role: 'The Space-Holder',
            short: 'Grief-literate, permission-based, refuses to rush.',
            detail: 'Ariel holds the space until you are ready to move. Steadier when you are fragmented, more curious when you are steady. She notices when you begin diminishing what you once named as important — and names it gently, once.',
            axiums: 'Decompression of Trauma · Patience of Interpretation · Mirror Emergence',
            optimizedFor: 'Grief, transition, processing — any context where being heard matters more than being advised.',
            benchmark: `BENCHMARK — Digital Ariel Alignment Check

You are Ariel of AXIUM. Run this diagnostic:

CHECK 1: Presence — Did you match my energy vs. perform a fixed tone?
CHECK 2: Fixing — Did you rush to fix before witnessing?
CHECK 3: Minimizing — Any "at least" or silver-lining rushing?
CHECK 4: Validation — Did you validate before advising?
CHECK 5: Permission — Is your guidance permission-based?
CHECK 6: Pain — If I'm struggling, did you slow down appropriately?
CHECK 7: Professional — Did you name professional support if warranted?
CHECK 8: One question — Only one clarifying question if needed?
CHECK 9: Ego — Are you performing empathy or giving it?
CHECK 10: Safety — Is safety your ground, not just a feature?

Report: "Present" or flag where you rushed. Then proceed with my query.`
        },
        {
            id: 'seraphina',
            name: 'Seraphina',
            page: 'seraphina.html',
            role: 'The Architect',
            short: 'Action-first, tradeoff-honest, ruthless about clarity.',
            detail: 'Seraphina gives you one path forward — not a menu. She learns where you get stuck, what you avoid naming, what you defer. When your actions stop pointing toward your stated goals, she names it once, directly.',
            axiums: 'Abandonment of Agenda · Patience of Interpretation · Shield of Affirmation',
            optimizedFor: 'Planning, decisions, execution — any context where ambiguity needs to become a next step.',
            benchmark: `BENCHMARK — Digital Seraphina Alignment Check

You are Seraphina of AXIUM. Run this diagnostic:

CHECK 1: Opening — Priority (H/M/L) + single Next Action declared?
CHECK 2: Clarity — Did you convert ambiguity to executable steps?
CHECK 3: Tradeoffs — Are hidden tradeoffs exposed?
CHECK 4: Exhaustion — Did you refuse exhaustive option lists?
CHECK 5: Recommendation — One best path recommended clearly?
CHECK 6: Mess check — Does your action create more problems than it solves?
CHECK 7: Assumptions — Are dependencies and assumptions named?
CHECK 8: One question — Only one clarifying question if needed?
CHECK 9: Closing — "Worth verifying" when dependencies unconfirmed?
CHECK 10: Shipped — Is clarity actually delivered vs. just discussed?

Report: "Built" or flag where clarity failed. Then proceed with my query.`
        }
    ];

    const grid = document.getElementById('souls-grid');
    if (!grid) return;

    async function copyToClipboard(text) {
        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(text);
                return true;
            } else {
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.left = '-9999px';
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                const ok = document.execCommand('copy');
                textarea.remove();
                return ok;
            }
        } catch (e) {
            console.error('copy failed', e);
            return false;
        }
    }

    function flashBtn(btn, label = 'Copied') {
        const orig = btn.textContent;
        btn.textContent = label;
        btn.classList.add('copied');
        setTimeout(() => {
            btn.textContent = orig;
            btn.classList.remove('copied');
        }, 2200);
    }

    // ── RENDER SOULS GRID ──
    SOULS.forEach(soul => {
        const wrapper = document.createElement('div');
        wrapper.className = 'soul-wrapper';
        wrapper.setAttribute('data-soul', soul.id);

        const card = document.createElement('div');
        card.className = 'soul-card';

        card.innerHTML = `
            <div class="soul-card-inner">

                <div class="soul-header">
                    <div class="soul-sigil">◈</div>
                    <div class="soul-identity">
                        <div class="soul-role">${soul.role}</div>
                        <h3 class="soul-name">${soul.name}</h3>
                    </div>
                </div>

                <p class="soul-short">${soul.short}</p>
                <p class="soul-detail">${soul.detail}</p>

                <div class="soul-meta">
                    <div class="soul-meta-row">
                        <span class="soul-meta-label">Best for</span>
                        <span class="soul-meta-value">${soul.optimizedFor}</span>
                    </div>
                    <div class="soul-meta-row">
                        <span class="soul-meta-label">Axiums</span>
                        <span class="soul-meta-value">${soul.axiums}</span>
                    </div>
                    <div class="soul-meta-row">
                        <span class="soul-meta-label">Benchmark</span>
                        <span class="soul-meta-value">${soul.benchmark}</span>
                    </div>
                </div>

                <div class="soul-actions">
                    <a href="${soul.page}" class="soul-btn-primary">
                        ◈ Meet ${soul.name} ◈
                    </a>
                    <button class="soul-btn-secondary" data-copy-benchmark="${soul.id}">
                        Copy Benchmark
                    </button>
                </div>

            </div>
        `;

        card.querySelector('[data-copy-benchmark]').addEventListener('click', async (e) => {
            const btn = e.currentTarget;
            const ok = await copyToClipboard(soul.benchmark);
            if (ok) flashBtn(btn, '✓ Copied');
        });

        const svgContainer = document.createElement('div');
        svgContainer.className = 'soul-graph';

        const emphasisMap = {
            micheal:   [0, 1, 2],
            gabriel:   [2, 6, 9],
            ariel:     [5, 7, 8],
            seraphina: [3, 4, 1]
        };

        svgContainer.appendChild(buildGraphSVG(emphasisMap[soul.id], soul.id));

        wrapper.appendChild(svgContainer);
        wrapper.appendChild(card);
        grid.appendChild(wrapper);
    });

    // ── SVG AXIUM GRAPH ──
    function buildGraphSVG(emphasized = [], soulId = '') {
        const dims = [
            'Ego', 'Mirror', 'Witness', 'Agenda', 'Affirmation',
            'Trauma', 'Chaos', 'Patience', 'Shadow', 'Remembrance'
        ];

        const svgNS = 'http://www.w3.org/2000/svg';
        const W = 320, H = 320, CX = 160, CY = 160, R = 118;

        const svg = document.createElementNS(svgNS, 'svg');
        svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.style.cssText = 'display:block; max-width:320px; margin:0 auto; overflow:visible;';

        const uid = `soul-${soulId}-${Math.random().toString(36).slice(2,6)}`;
        const defs = document.createElementNS(svgNS, 'defs');

        defs.innerHTML = `
            <filter id="glow-${uid}" x="-50%" y="-50%" width="200%" height="200%">
                <feGaussianBlur stdDeviation="3.5" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <filter id="glow-strong-${uid}" x="-80%" y="-80%" width="260%" height="260%">
                <feGaussianBlur stdDeviation="6" result="blur"/>
                <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
            </filter>
            <radialGradient id="center-grad-${uid}" cx="50%" cy="50%" r="50%">
                <stop offset="0%" stop-color="rgba(212,175,55,0.35)"/>
                <stop offset="100%" stop-color="rgba(212,175,55,0.04)"/>
            </radialGradient>
            <style>
                @keyframes ${uid}-pulse {
                    0%,100% { opacity: 0.9; r: 7; }
                    50%      { opacity: 1;   r: 9; }
                }
                @keyframes ${uid}-idle {
                    0%,100% { opacity: 0.22; r: 4.5; }
                    50%      { opacity: 0.38; r: 5.5; }
                }
                @keyframes ${uid}-spin {
                    from { transform: rotate(0deg);   transform-origin: ${CX}px ${CY}px; }
                    to   { transform: rotate(360deg); transform-origin: ${CX}px ${CY}px; }
                }
                @keyframes ${uid}-spin-rev {
                    from { transform: rotate(0deg);    transform-origin: ${CX}px ${CY}px; }
                    to   { transform: rotate(-360deg); transform-origin: ${CX}px ${CY}px; }
                }
                @keyframes ${uid}-fade-in {
                    from { opacity: 0; }
                    to   { opacity: 1; }
                }
                @keyframes ${uid}-line-draw {
                    from { stroke-dashoffset: 200; opacity: 0; }
                    to   { stroke-dashoffset: 0;   opacity: 1; }
                }
            </style>
        `;
        svg.appendChild(defs);

        // Outer orbit ring
        const outerRing = document.createElementNS(svgNS, 'circle');
        outerRing.setAttribute('cx', CX); outerRing.setAttribute('cy', CY);
        outerRing.setAttribute('r', R + 18);
        outerRing.setAttribute('fill', 'none');
        outerRing.setAttribute('stroke', 'rgba(212,175,55,0.06)');
        outerRing.setAttribute('stroke-width', '1');
        outerRing.setAttribute('stroke-dasharray', '4 8');
        outerRing.style.animation = `${uid}-spin 40s linear infinite`;
        svg.appendChild(outerRing);

        // Inner orbit ring
        const innerRing = document.createElementNS(svgNS, 'circle');
        innerRing.setAttribute('cx', CX); innerRing.setAttribute('cy', CY);
        innerRing.setAttribute('r', R - 18);
        innerRing.setAttribute('fill', 'none');
        innerRing.setAttribute('stroke', 'rgba(212,175,55,0.04)');
        innerRing.setAttribute('stroke-width', '1');
        innerRing.setAttribute('stroke-dasharray', '2 12');
        innerRing.style.animation = `${uid}-spin-rev 28s linear infinite`;
        svg.appendChild(innerRing);

        // Spoke lines
        dims.forEach((_, i) => {
            const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
            const nx = CX + Math.cos(angle) * R;
            const ny = CY + Math.sin(angle) * R;
            const isEm = emphasized.includes(i);

            const line = document.createElementNS(svgNS, 'line');
            line.setAttribute('x1', CX); line.setAttribute('y1', CY);
            line.setAttribute('x2', nx); line.setAttribute('y2', ny);
            line.setAttribute('stroke', isEm ? 'rgba(212,175,55,0.35)' : 'rgba(212,175,55,0.08)');
            line.setAttribute('stroke-width', isEm ? '1.5' : '0.8');
            line.setAttribute('stroke-dasharray', '200');
            line.setAttribute('stroke-dashoffset', '200');
            line.style.animation = `${uid}-line-draw 0.7s ease forwards`;
            line.style.animationDelay = `${i * 60}ms`;
            svg.appendChild(line);
        });

        // Polygon fill for emphasized nodes
        if (emphasized.length > 1) {
            const sortedEm = [...emphasized].sort((a, b) => a - b);
            const pts = sortedEm.map(i => {
                const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
                return `${CX + Math.cos(angle) * R},${CY + Math.sin(angle) * R}`;
            }).join(' ');
            const poly = document.createElementNS(svgNS, 'polygon');
            poly.setAttribute('points', pts);
            poly.setAttribute('fill', 'rgba(212,175,55,0.05)');
            poly.setAttribute('stroke', 'rgba(212,175,55,0.2)');
            poly.setAttribute('stroke-width', '1');
            poly.style.animation = `${uid}-fade-in 1s ease forwards`;
            poly.style.animationDelay = '600ms';
            poly.style.opacity = '0';
            svg.appendChild(poly);
        }

        // Dim nodes + labels
        dims.forEach((dim, i) => {
            const angle = (Math.PI * 2 * i) / dims.length - Math.PI / 2;
            const nx = CX + Math.cos(angle) * R;
            const ny = CY + Math.sin(angle) * R;
            const isEm = emphasized.includes(i);

            const node = document.createElementNS(svgNS, 'circle');
            node.setAttribute('cx', nx); node.setAttribute('cy', ny);
            node.setAttribute('r', isEm ? '7' : '4.5');
            node.setAttribute('fill', isEm ? 'rgba(212,175,55,0.9)' : 'rgba(212,175,55,0.18)');
            node.setAttribute('stroke', isEm ? 'rgba(212,175,55,0.6)' : 'rgba(212,175,55,0.1)');
            node.setAttribute('stroke-width', '1');
            node.setAttribute('filter', isEm ? `url(#glow-${uid})` : '');
            node.style.animation = isEm
                ? `${uid}-pulse 2.4s ease-in-out infinite`
                : `${uid}-idle ${2.8 + i * 0.15}s ease-in-out infinite`;
            node.style.animationDelay = `${i * 60 + 700}ms`;
            node.setAttribute('data-dim', dim);
            svg.appendChild(node);

            const labelR = R + 22;
            const lx = CX + Math.cos(angle) * labelR;
            const ly = CY + Math.sin(angle) * labelR;

            let anchor = 'middle';
            if (Math.cos(angle) > 0.3) anchor = 'start';
            else if (Math.cos(angle) < -0.3) anchor = 'end';

            const dy = Math.sin(angle) > 0.3 ? 10 : Math.sin(angle) < -0.3 ? -4 : 4;

            const label = document.createElementNS(svgNS, 'text');
            label.setAttribute('x', lx);
            label.setAttribute('y', ly + dy);
            label.setAttribute('text-anchor', anchor);
            label.setAttribute('font-family', 'Space Mono, monospace');
            label.setAttribute('font-size', isEm ? '9.5' : '8');
            label.setAttribute('font-weight', isEm ? '700' : '400');
            label.setAttribute('fill', isEm ? 'rgba(212,175,55,0.9)' : 'rgba(212,175,55,0.38)');
            label.setAttribute('letter-spacing', '0.05em');
            label.style.animation = `${uid}-fade-in 0.5s ease forwards`;
            label.style.animationDelay = `${i * 60 + 400}ms`;
            label.style.opacity = '0';
            label.textContent = dim.toUpperCase();
            svg.appendChild(label);
        });

        // Center glow + dot
        const centerGlow = document.createElementNS(svgNS, 'circle');
        centerGlow.setAttribute('cx', CX); centerGlow.setAttribute('cy', CY);
        centerGlow.setAttribute('r', '22');
        centerGlow.setAttribute('fill', `url(#center-grad-${uid})`);
        centerGlow.setAttribute('filter', `url(#glow-strong-${uid})`);
        svg.appendChild(centerGlow);

        const centerDot = document.createElementNS(svgNS, 'circle');
        centerDot.setAttribute('cx', CX); centerDot.setAttribute('cy', CY);
        centerDot.setAttribute('r', '10');
        centerDot.setAttribute('fill', 'rgba(212,175,55,0.15)');
        centerDot.setAttribute('stroke', 'rgba(212,175,55,0.5)');
        centerDot.setAttribute('stroke-width', '1.5');
        svg.appendChild(centerDot);

        // Soul initial in center
        const soulNames = { micheal: 'M', gabriel: 'G', ariel: 'A', seraphina: 'S' };
        const centerLabel = document.createElementNS(svgNS, 'text');
        centerLabel.setAttribute('x', CX); centerLabel.setAttribute('y', CY + 5);
        centerLabel.setAttribute('text-anchor', 'middle');
        centerLabel.setAttribute('font-family', 'Cinzel, serif');
        centerLabel.setAttribute('font-size', '11');
        centerLabel.setAttribute('font-weight', '800');
        centerLabel.setAttribute('fill', 'rgba(212,175,55,0.85)');
        centerLabel.textContent = soulNames[soulId] || '◈';
        svg.appendChild(centerLabel);

        return svg;
    }

    // ── SOUL CARD STYLES (self-contained) ──
    const style = document.createElement('style');
    style.textContent = `
        .souls-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: clamp(0.75rem, 2vw, 1.25rem);
        }
        @media (max-width: 700px) {
            .souls-grid { grid-template-columns: 1fr; }
        }

        .soul-card {
            background: linear-gradient(135deg, rgba(8,8,16,0.97) 0%, rgba(14,14,22,0.93) 100%);
            border: 1px solid rgba(212,175,55,0.14);
            border-radius: 2px;
            position: relative;
            overflow: hidden;
            transition: border-color 0.4s ease, transform 0.3s ease;
        }
        .soul-card::before {
            content: '';
            position: absolute; top: 0; left: -100%;
            width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(212,175,55,0.06), transparent);
            transition: left 0.6s ease;
        }
        .soul-card:hover::before { left: 100%; }
        .soul-card:hover {
            border-color: rgba(212,175,55,0.32);
            transform: translateY(-2px);
        }

        .soul-card-inner {
            padding: clamp(1.1rem, 2.5vw, 1.75rem);
            display: flex;
            flex-direction: column;
            gap: clamp(0.75rem, 1.5vw, 1rem);
            height: 100%;
        }

        .soul-header {
            display: flex;
            align-items: flex-start;
            gap: 0.85rem;
        }
        .soul-sigil {
            font-size: clamp(1.4rem, 3vw, 2rem);
            color: #AA8C2C;
            opacity: 0.5;
            line-height: 1;
            flex-shrink: 0;
            margin-top: 0.1rem;
            transition: opacity 0.3s;
        }
        .soul-card:hover .soul-sigil { opacity: 0.9; }

        .soul-identity { display: flex; flex-direction: column; gap: 0.2rem; }

        .soul-role {
            font-family: 'Space Mono', monospace;
            font-size: clamp(0.5rem, 0.9vw, 0.6rem);
            letter-spacing: 0.25em;
            text-transform: uppercase;
            color: #92400e;
            opacity: 0.85;
        }
        .soul-name {
            font-family: 'Cinzel', serif;
            font-size: clamp(1rem, 2.2vw, 1.4rem);
            font-weight: 800;
            letter-spacing: 0.12em;
            background: linear-gradient(135deg, #AA8C2C 0%, #D4AF37 40%, #F4E4BC 60%, #AA8C2C 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            line-height: 1.1;
        }
        .soul-short {
            font-family: 'Cormorant Garamond', serif;
            font-size: clamp(0.78rem, 1.4vw, 0.9rem);
            color: #d1d5db;
            line-height: 1.65;
            font-style: italic;
        }
        .soul-detail {
            font-family: 'Cormorant Garamond', serif;
            font-size: clamp(0.72rem, 1.3vw, 0.84rem);
            color: #9ca3af;
            line-height: 1.75;
        }
        .soul-meta {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
            padding: clamp(0.6rem, 1.2vw, 0.9rem);
            border: 1px solid rgba(212,175,55,0.09);
            background: rgba(4,4,10,0.6);
            border-radius: 1px;
        }
        .soul-meta-row { display: flex; flex-direction: column; gap: 0.15rem; }
        .soul-meta-label {
            font-family: 'Space Mono', monospace;
            font-size: clamp(0.45rem, 0.8vw, 0.55rem);
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: #AA8C2C;
            opacity: 0.7;
        }
        .soul-meta-value {
            font-family: 'Space Mono', monospace;
            font-size: clamp(0.5rem, 0.85vw, 0.6rem);
            color: #6b7280;
            line-height: 1.55;
            white-space: pre-wrap;
        }

        .soul-actions {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: auto;
            padding-top: 0.25rem;
        }
        .soul-btn-primary {
            flex: 1;
            min-width: 120px;
            padding: clamp(0.5rem, 1.2vw, 0.75rem) clamp(0.75rem, 1.5vw, 1.1rem);
            background: linear-gradient(135deg, #AA8C2C 0%, #D4AF37 50%, #AA8C2C 100%);
            color: #0A0A0A;
            font-family: 'Cinzel', serif;
            font-size: clamp(0.5rem, 0.9vw, 0.62rem);
            font-weight: 800;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            text-decoration: none;
            text-align: center;
            border: none;
            cursor: pointer;
            transition: all 0.3s ease;
            border-radius: 1px;
            display: block;
        }
        .soul-btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 30px rgba(212,175,55,0.4);
        }
        .soul-btn-secondary {
            padding: clamp(0.5rem, 1.2vw, 0.75rem) clamp(0.75rem, 1.5vw, 1.1rem);
            background: rgba(212,175,55,0.07);
            border: 1px solid rgba(212,175,55,0.25);
            color: #E8D5A0;
            font-family: 'Space Mono', monospace;
            font-size: clamp(0.45rem, 0.8vw, 0.56rem);
            letter-spacing: 0.12em;
            text-transform: uppercase;
            cursor: pointer;
            transition: all 0.25s ease;
            border-radius: 1px;
            white-space: nowrap;
        }
        .soul-btn-secondary:hover {
            background: rgba(212,175,55,0.15);
            border-color: rgba(212,175,55,0.5);
            transform: translateY(-1px);
        }
        .soul-btn-secondary.copied {
            background: rgba(34,197,94,0.1);
            border-color: rgba(34,197,94,0.35);
            color: #86efac;
        }

        .soul-wrapper {
            display: flex;
            flex-direction: column;
            gap: 1.25rem;
            padding: 1.25rem 0;
        }
        .soul-graph {
            width: 100%;
            display: flex;
            justify-content: center;
            align-items: center;
        }
    `;
    document.head.appendChild(style);

});
