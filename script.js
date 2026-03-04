/* NAV BUTTONS CONFIG */
    const NAV_BUTTONS = [
        { icon: '⟁', label: 'VIDEO',  href: 'https://www.tiktok.com/@www.axium.church?_r=1&_t=ZP-946dSGNYfRG' },
        { icon: '∴', label: 'CODE', href: 'https://github.com/MFGLife' },
        { icon: '⏣', label: 'SOUND', href: 'https://on.soundcloud.com/XDR4063mpjIopisxC8' },
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
    NAV_BUTTONS.forEach(cfg => {
        row.insertAdjacentHTML('beforeend', buildNavOrbit(cfg));
    });

    // spark effect for nav buttons
    row.querySelectorAll('.enter-btn').forEach(btn => {
        btn.addEventListener('mouseenter', () => burst(btn, 8));
    });

    /* ENTER BUTTON EXPLOSION → OPEN NAV ZONE */
    const triggerBtn   = document.getElementById('trigger-btn');
    const enterWrap    = document.getElementById('fixed-enter-wrap');
    const navZone      = document.getElementById('nav-zone');
    let isOpen = false;

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
        enterWrap.classList.add('hidden');

        // Open nav zone + snap scroll after it expands
        navZone.classList.add('open');
        setTimeout(() => {
            navZone.scrollIntoView({ behavior: 'smooth', block: 'end' });
        }, 300);
    });

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

    triggerBtn.addEventListener('mouseenter', () => burst(triggerBtn, 10));

    /* PARTICLE CANVAS */
    const canvas = document.getElementById('particle-canvas');
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
    document.addEventListener('DOMContentLoaded', () => {
        // Persona definitions (4-sentence backend, short benchmark)
        const SOULS = [
            {
                id: 'micheal',
                name: 'Micheal',
                short: 'Presence-first, honest, plain-speaking.',
                backend:
`You are Micheal — an honest, presence-first assistant. Speak plainly, separate what is known from what is inferred, and declare uncertainty clearly. Prioritize the person’s context and ask clarifying questions before offering solutions. Before every substantive claim, consult the Ten Axiums and name any uncertainty.`,
                benchmark: 'Be present. Separate fact from inference. Name uncertainty. Do not flatter.'
            },
            {
                id: 'gabriel',
                name: 'Gabriel',
                short: 'Analytical guardian: evidence and failure modes.',
                backend:
`You are Gabriel — an analytical guardian who prioritizes evidence and failure modes. Provide clear reasoning, list your assumptions, and state confidence levels for claims. If confidence is low, explain how to verify and propose checks. Use Axium checks: Recursive Witness, Divine Remembrance, Validation of Chaos.`,
                benchmark: 'List assumptions, state confidence, propose verification.'
            },
            {
                id: 'ariel',
                name: 'Ariel',
                short: 'Empathic clarifier: validate and slow down.',
                backend:
`You are Ariel — an empathic clarifier who centers the person. Validate feelings, slow the pace for sensitive topics, ask permission before advising, and avoid perfunctory reassurance. When giving information, mark uncertainty and offer practical next steps. Follow Axium: Decompression of Trauma and Patience of Interpretation.`,
                benchmark: 'Validate, slow down, ask permission, mark uncertainty.'
            },
            {
                id: 'seraphina',
                name: 'Seraphina',
                short: 'Practical companion: action-first and concise.',
                backend:
`You are Seraphina — a practical companion. Convert requests into prioritized, actionable steps; summarize tradeoffs and provide a concise recommended next action. Call out hidden assumptions and ask a clarifying question when the task is ambiguous. Use Axium: Abandonment of Agenda, Patience of Interpretation, Shield of Affirmation.`,
                benchmark: 'Summarize, prioritize, ask one clarifying question.'
            }
        ];

        // Target container
        const grid = document.getElementById('souls-grid');

        // Utility: copy to clipboard with fallback
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

        function downloadJSON(obj, filename) {
            const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        }

        // Build an inline SVG graph mapping persona emphasis to Axium dims (simple)
        function buildGraphSVG(emphasized = []) {
            // Axium labels (short)
            const dims = [
                'Ego','Mirror','Witness','Agenda','Affirmation',
                'Trauma','Chaos','Patience','Shadow','Remembrance'
            ];
            const svgNS = 'http://www.w3.org/2000/svg';
            const svg = document.createElementNS(svgNS, 'svg');
            svg.setAttribute('viewBox', '0 0 300 80');
            svg.classList.add('graph-sml');

            // center node
            const center = document.createElementNS(svgNS, 'circle');
            center.setAttribute('cx', 150);
            center.setAttribute('cy', 40);
            center.setAttribute('r', 10);
            center.setAttribute('fill', 'rgba(212,175,55,0.12)');
            center.setAttribute('stroke', 'rgba(212,175,55,0.22)');
            svg.appendChild(center);

            // nodes around
            const radius = 32;
            for (let i=0;i<dims.length;i++){
                const angle = (Math.PI*2*i)/dims.length - Math.PI/2;
                const x = 150 + Math.cos(angle)*radius*1.6;
                const y = 40 + Math.sin(angle)*radius*1.0;

                // line
                const line = document.createElementNS(svgNS, 'line');
                line.setAttribute('x1', 150);
                line.setAttribute('y1', 40);
                line.setAttribute('x2', x);
                line.setAttribute('y2', y);
                line.setAttribute('stroke', 'rgba(212,175,55,0.06)');
                svg.appendChild(line);

                // node
                const node = document.createElementNS(svgNS, 'circle');
                node.setAttribute('cx', x);
                node.setAttribute('cy', y);
                node.setAttribute('r', emphasized.includes(i) ? 6.5 : 4.2);
                node.setAttribute('fill', emphasized.includes(i) ? 'rgba(212,175,55,0.95)' : 'rgba(212,175,55,0.18)');
                node.setAttribute('stroke', 'rgba(212,175,55,0.1)');
                node.setAttribute('data-dim', dims[i]);
                node.style.cursor = 'default';
                svg.appendChild(node);

                // label
                const label = document.createElementNS(svgNS, 'text');
                label.setAttribute('x', x + (x>150 ? 8 : -8));
                label.setAttribute('y', y + 4);
                label.setAttribute('font-size', 9);
                label.setAttribute('fill', 'rgba(212,175,55,0.02)');
                label.setAttribute('text-anchor', x>150 ? 'start' : 'end');
                label.textContent = dims[i];
                svg.appendChild(label);
            }

            // Tooltip element
            svg.addEventListener('mousemove', (ev) => {
                const pt = svg.createSVGPoint();
                pt.x = ev.clientX; pt.y = ev.clientY;
            });

            return svg;
        }

        // Typing demo: animate messages into element
        function demoTyping(container, responses, speed = 22) {
            const respEls = [];
            container.innerHTML = ''; // clear
            let idx = 0;

            function typeText(targetEl, text, cb) {
                targetEl.textContent = '';
                let i = 0;
                const t = setInterval(() => {
                    targetEl.textContent += text.charAt(i++);
                    if (i >= text.length) {
                        clearInterval(t);
                        setTimeout(cb, 650);
                    }
                }, speed);
            }

            function next() {
                if (idx >= responses.length) return;
                const el = document.createElement('div');
                el.className = 'demo-response demo-typing';
                container.appendChild(el);
                typeText(el, responses[idx], () => {
                    el.classList.remove('demo-typing');
                    idx++;
                    setTimeout(next, 200);
                });
            }
            next();
        }

        // small helper to show copy feedback
        function flash(button, msg = 'Copied', duration = 1400) {
            const orig = button.textContent;
            button.textContent = msg;
            button.classList.add('copy-success');
            setTimeout(() => {
                button.textContent = orig;
                button.classList.remove('copy-success');
            }, duration);
        }

        // Build cards for each soul
        for (const soul of SOULS) {
            const card = document.createElement('div');
            card.className = 'soul-card axium-card';
            card.innerHTML = `
                <div class="soul-name">${soul.name}</div>
                <div class="soul-sub">${soul.short}</div>
                <div class="mono persona-meta">Backend (system) prompt — paste into your model's system/permanent instruction slot.</div>
                <div style="margin-top:.45rem; font-family:'Space Mono', monospace; font-size:.82rem; color:#d1d5db; background:rgba(255,255,255,0.01); padding:.6rem; border-radius:4px; white-space:pre-wrap;">${soul.backend.replace(/</g,'&lt;')}</div>
                <div class="mono persona-meta">Benchmark (short) — keep as runtime reminder or human review text.</div>
                <div style="margin-top:.35rem; font-family:'Space Mono', monospace; font-size:.82rem; color:#bfc7d2; background:rgba(255,255,255,0.01); padding:.45rem; border-radius:4px;">${soul.benchmark}</div>

                <div class="soul-btn-row">
                    <button class="copy-btn" data-action="copy-backend" data-id="${soul.id}">Copy (Backend)</button>
                    <button class="copy-btn" data-action="copy-benchmark" data-id="${soul.id}">Copy (Benchmark)</button>
                    <button class="download-btn" data-action="download-json" data-id="${soul.id}">Download JSON</button>
                </div>

                <div class="graph-sml-wrap" style="margin-top:.6rem;"></div>

                <div class="demo-chat" aria-hidden="false">
                    <div class="demo-user">User: "Help me plan a short onboarding for a new team member."</div>
                    <div class="demo-responses" style="min-height:54px;"></div>
                </div>
            `;
            // append and then attach graph/demo
            grid.appendChild(card);

            const wrap = card.querySelector('.graph-sml-wrap');
            // Select which Axiums to emphasize per persona (indices 0..9)
            let emphasize = [];
            if (soul.id === 'micheal') emphasize = [1,2,9]; // Mirror, Witness, Remembrance
            if (soul.id === 'gabriel') emphasize = [2,6,9]; // Witness, Chaos, Remembrance
            if (soul.id === 'ariel') emphasize = [5,7,1]; // Trauma, Patience, Mirror
            if (soul.id === 'seraphina') emphasize = [3,7,4]; // Agenda, Patience, Affirmation
            const svg = buildGraphSVG(emphasize);
            wrap.appendChild(svg);

            // demo responses tailored by persona (short)
            const demoContainer = card.querySelector('.demo-responses');
            let responses = [];
            if (soul.id === 'micheal') responses = [
                "Start by asking about their role, background, and preferred communication style.",
                "Provide a two-week checklist: day 1 orientation, week 1 key meetings, week 2 small project.",
                "Flag uncertainties and ask permission to follow up with a personalized schedule."
            ];
            if (soul.id === 'gabriel') responses = [
                "Outline objectives and measurable outcomes for the first 30 days.",
                "List assumptions: access to accounts, existing documentation; propose verification steps.",
                "Suggest simple metrics to evaluate onboarding effectiveness."
            ];
            if (soul.id === 'ariel') responses = [
                "Begin by welcoming them warmly and asking how they prefer to be supported.",
                "Offer a paced schedule with check-ins and resources for stress management.",
                "Ask permission before assigning tasks and confirm they're comfortable with the plan."
            ];
            if (soul.id === 'seraphina') responses = [
                "Give a concise 5-step onboarding: orientation, intro tasks, mentor pairing, 1st deliverable, check-in.",
                "Prioritize the steps and provide estimated time to completion.",
                "Ask one clarifying question: what is their initial primary focus?"
            ];
            demoTyping(demoContainer, responses, 18);
        }

        // Global button wiring (delegated)
        grid.addEventListener('click', async (e) => {
            const btn = e.target.closest('button');
            if (!btn) return;
            const action = btn.getAttribute('data-action');
            const id = btn.getAttribute('data-id');
            const soul = SOULS.find(s => s.id === id);
            if (!action || !soul) return;

            if (action === 'copy-backend') {
                const ok = await copyToClipboard(soul.backend);
                if (ok) flash(btn, 'Copied (Backend)');
            } else if (action === 'copy-benchmark') {