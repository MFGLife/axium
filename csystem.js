/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — CARD SYSTEM v1.0
 * Full card rendering engine: portraits, particles, hand picker,
 * detail modal (flip), field zones. Drop-in replacement for all
 * card rendering in game.html.
 *
 * REQUIRES: cards.js loaded first (PLAYER_CARDS, EGO_CARDS,
 *   ID_CARDS, ATTN_STATES, SYNERGIES, getAttnState, shuffle)
 *
 * EXPORTS (window globals):
 *   CardRenderer      — builds a living card DOM element
 *   HandPickerModal   — full-screen card fan picker
 *   CardDetailModal   — flip-to-back detail view
 *   FieldZone         — renders mini cards in field areas
 *   CardParticles     — standalone particle emitter per card
 * ═══════════════════════════════════════════════════════════════
 */

;(function(global) {
'use strict';

// ─────────────────────────────────────────────────────────────
// CONSTANTS & HELPERS
// ─────────────────────────────────────────────────────────────
const RAF  = requestAnimationFrame.bind(window);
const CRAF = cancelAnimationFrame.bind(window);

function hexA(v) {
  return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
}
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t)  { return a + (b - a) * t; }
function rand(a, b)     { return a + Math.random() * (b - a); }
function cap(s)         { return s ? s[0].toUpperCase() + s.slice(1) : ''; }

function cardLayerLabel(card) {
  const layer = card.layer || '';
  if (layer === 'Superego') {
    const num = card.number !== undefined ? toRoman(card.number) : '';
    return num ? `Superego · ${num}` : 'Superego';
  }
  if (layer === 'Ego') return `Ego · ${cap(card.suit || '')}`;
  if (layer === 'ID')  return `ID · ${cap(card.suit || '')}`;
  return layer || 'Card';
}

function toRoman(n) {
  const map = [[21,'XXI'],[20,'XX'],[19,'XIX'],[18,'XVIII'],[17,'XVII'],[16,'XVI'],[15,'XV'],
               [14,'XIV'],[13,'XIII'],[12,'XII'],[11,'XI'],[10,'X'],[9,'IX'],[8,'VIII'],
               [7,'VII'],[6,'VI'],[5,'V'],[4,'IV'],[3,'III'],[2,'II'],[1,'I'],[0,'0']];
  return (map.find(([v]) => v === n) || [n, String(n)])[1];
}

// Layer colour palette
function layerColor(card) {
  if (card.layer === 'Superego') return '#D4AF37';
  if (card.layer === 'Ego')      return '#7EB8E8';
  if (card.layer === 'ID')       return '#86EFAC';
  return '#ffffff';
}

// Parse hex → [r,g,b]
function hexRGB(hex) {
  const h = hex.replace('#','');
  if (h.length === 3) {
    return [parseInt(h[0]+h[0],16), parseInt(h[1]+h[1],16), parseInt(h[2]+h[2],16)];
  }
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}

// ─────────────────────────────────────────────────────────────
// CONSTELLATION ANIMATOR
// Shared animation logic for all canvas sizes.
// ─────────────────────────────────────────────────────────────
class ConstellationAnim {
  constructor(canvas, card, opts = {}) {
    this.canvas  = canvas;
    this.card    = card;
    this.opts    = Object.assign({ speed: 0.018, nodeSize: 1.8, edgeAlpha: 0.38, glowMult: 3 }, opts);
    this.rafId   = null;
    this.mapped  = [];
    this.edges   = [];
    this.phases  = [];
    this._build();
  }

  _build() {
    const { card } = this;
    const pts = card.pts;
    if (!pts || !pts.length) return;
    const W = this.canvas.offsetWidth  || this.canvas.width  || 120;
    const H = this.canvas.offsetHeight || this.canvas.height || 80;
    this.canvas.width  = W;
    this.canvas.height = H;
    let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
    pts.forEach(([x,y])=>{ mnX=Math.min(mnX,x);mnY=Math.min(mnY,y);mxX=Math.max(mxX,x);mxY=Math.max(mxY,y); });
    const pad = 8;
    const sc  = Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
    const oX  = W/2-(mnX+mxX)/2*sc;
    const oY  = H/2-(mnY+mxY)/2*sc;
    this.mapped = pts.map(([x,y]) => ({ x: x*sc+oX, y: y*sc+oY }));
    // Build nearest-2 edges
    const edgeSet = new Set();
    this.mapped.forEach((p, i) => {
      this.mapped
        .map((q, j) => ({ j, d: Math.hypot(q.x-p.x, q.y-p.y) }))
        .filter(v => v.j !== i)
        .sort((a, b) => a.d - b.d)
        .slice(0, 2)
        .forEach(({ j }) => {
          const k = Math.min(i,j) + '-' + Math.max(i,j);
          if (!edgeSet.has(k)) { edgeSet.add(k); this.edges.push([i,j]); }
        });
    });
    this.phases = this.mapped.map(() => Math.random() * Math.PI * 2);
    this.t = 0;
  }

  start() {
    if (this.rafId) CRAF(this.rafId);
    const frame = () => {
      if (!this.canvas.isConnected) { this.rafId = null; return; }
      this._draw();
      this.rafId = RAF(frame);
    };
    this.rafId = RAF(frame);
    return this;
  }

  stop() {
    if (this.rafId) { CRAF(this.rafId); this.rafId = null; }
  }

  // ── Charge API ───────────────────────────────────────────
  // progress 0→1: nodes light up sequentially.
  // At 1.0 all nodes blaze, then burstDecay kicks in.
  // Call setCharge(0) to reset after a trigger fires.
  setCharge(progress) {
    this.charge      = clamp(progress, 0, 1);
    this.burstDecay  = 0; // cleared; only set on trigger
  }

  triggerBurst() {
    this.charge     = 0;
    this.burstDecay = 1.0;
  }

  _draw() {
    const { canvas, card, opts, mapped, edges, phases } = this;
    if (!mapped.length) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width, H = canvas.height;
    this.t += opts.speed;
    const t = this.t;
    const [r,g,b] = hexRGB(card.color || '#ffffff');

    // Charge state (0 if not in battle)
    const charge     = this.charge     || 0;
    const burst      = this.burstDecay || 0;
    const nodeCount  = mapped.length;

    // Decay burst each frame
    if (this.burstDecay > 0) {
      this.burstDecay = Math.max(0, this.burstDecay - 0.055);
    }

    ctx.clearRect(0, 0, W, H);

    // Background radial glow — intensifies with charge
    const bgAlpha = 0.05 + charge * 0.12 + burst * 0.18;
    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W,H) * 0.55);
    bg.addColorStop(0, `rgba(${r},${g},${b},${bgAlpha.toFixed(3)})`);
    bg.addColorStop(1, 'rgba(2,2,10,0)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // How many nodes are "lit" based on charge progress
    // e.g. charge=0.5 with 8 nodes → 4 nodes lit
    const litCount = charge >= 1.0 ? nodeCount : Math.floor(charge * nodeCount);

    // Edges — brighten between two lit nodes
    edges.forEach(([a, eb]) => {
      const aLit  = a  < litCount;
      const bLit  = eb < litCount;
      const edgeLit = aLit && bLit;
      const baseAlpha = edgeLit
        ? (opts.edgeAlpha * 2.5 + burst * 0.5)
        : (opts.edgeAlpha + 0.12 * Math.sin(t * 1.4 + (a + eb) * 0.6));
      const na = mapped[a], nb = mapped[eb];
      ctx.beginPath();
      ctx.strokeStyle = `rgba(${r},${g},${b},${Math.min(baseAlpha, 1).toFixed(3)})`;
      ctx.lineWidth   = edgeLit ? 1.1 : 0.7;
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
      ctx.stroke();
    });

    // Nodes
    mapped.forEach((p, i) => {
      const isLit   = i < litCount;
      const isFront = i === litCount; // the "charging" leading node
      const tw      = 0.55 + 0.45 * Math.sin(t * 0.95 + phases[i]);

      // Burst at trigger: all nodes flash white
      const burstBoost = burst * (0.6 + 0.4 * Math.sin(i * 0.8));

      // Node size: lit nodes bigger, leading node pulses
      let sz = opts.nodeSize + tw * 0.7;
      if (isLit)   sz *= (1.0 + charge * 0.5 + burstBoost * 1.2);
      if (isFront) sz *= (1.0 + 0.35 * Math.sin(t * 4)); // rapid pulse on leading node

      // Glow halo — skip for dim unlits to save draw calls
      if (isLit || isFront || burst > 0.05) {
        const glowR  = sz * (isLit ? opts.glowMult * 1.1 : opts.glowMult);
        const glowA  = isLit
          ? (tw * 0.28 + charge * 0.15 + burstBoost * 0.4)
          : (tw * 0.12);
        ctx.beginPath();
        ctx.arc(p.x, p.y, glowR, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${r},${g},${b},${Math.min(glowA, 0.9).toFixed(3)})`;
        ctx.fill();
      }

      // Core node
      const coreAlpha = isLit
        ? Math.min(0.95 + burstBoost * 0.5, 1)
        : (0.55 + tw * 0.2);
      ctx.beginPath();
      ctx.arc(p.x, p.y, sz, 0, Math.PI * 2);
      ctx.fillStyle = isLit
        ? `rgba(${r},${g},${b},${coreAlpha.toFixed(3)})`
        : `rgba(${r},${g},${b},${(0.45 + tw * 0.2).toFixed(3)})`;
      ctx.fill();

      // White centre — lit nodes get full-white, unlits get dim
      const whiteAlpha = isLit
        ? Math.min(0.85 + burstBoost, 1)
        : (0.45 + tw * 0.2);
      ctx.beginPath();
      ctx.arc(p.x, p.y, sz * 0.38, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${whiteAlpha.toFixed(3)})`;
      ctx.fill();
    });
  }

  resize() {
    this._build();
  }
}

// ─────────────────────────────────────────────────────────────
// CARD PARTICLE SYSTEM
// Each card gets its own WebGL-free particle layer on a canvas.
// ─────────────────────────────────────────────────────────────
class CardParticles {
  constructor(canvas, card) {
    this.canvas = canvas;
    this.card   = card;
    this.rafId  = null;
    this.pool   = [];
    this.t      = 0;
    this._resize();
  }

  _resize() {
    this.W = this.canvas.offsetWidth  || this.canvas.width  || 100;
    this.H = this.canvas.offsetHeight || this.canvas.height || 140;
    this.canvas.width  = this.W;
    this.canvas.height = this.H;
    this._initPool();
  }

  _initPool() {
    const { W, H, card } = this;
    const [r,g,b] = hexRGB(card.color || '#ffffff');
    const count = 28;
    this.pool = Array.from({ length: count }, (_, i) => ({
      x:     rand(0, W),
      y:     rand(0, H),
      vx:    rand(-0.18, 0.18),
      vy:    rand(-0.55, -0.08),   // mostly drift upward
      life:  rand(0, 1),
      maxLife: rand(0.6, 1.4),
      sz:    rand(0.6, 2.2),
      // Each particle gets slight color variation
      r: clamp(r + rand(-20, 20), 0, 255),
      g: clamp(g + rand(-20, 20), 0, 255),
      b: clamp(b + rand(-20, 20), 0, 255),
      twinkle: rand(0, Math.PI * 2),
      type: i % 5 === 0 ? 'spark' : i % 7 === 0 ? 'rune' : 'dust',
    }));
  }

  start() {
    if (this.rafId) CRAF(this.rafId);
    const frame = () => {
      if (!this.canvas.isConnected) { this.rafId = null; return; }
      this._draw();
      this.rafId = RAF(frame);
    };
    this.rafId = RAF(frame);
    return this;
  }

  stop() {
    if (this.rafId) { CRAF(this.rafId); this.rafId = null; }
  }

  _draw() {
    const { canvas, pool, card } = this;
    const W = this.W, H = this.H;
    const ctx = canvas.getContext('2d');
    this.t += 0.016;

    ctx.clearRect(0, 0, W, H);

    pool.forEach(p => {
      // Advance
      p.life += 0.005;
      if (p.life > p.maxLife) {
        // Respawn from bottom edge with slight spread
        p.x     = rand(W * 0.1, W * 0.9);
        p.y     = H + 2;
        p.life  = 0;
        p.maxLife = rand(0.5, 1.3);
        p.vx    = rand(-0.22, 0.22);
        p.vy    = rand(-0.6, -0.1);
      }
      p.x += p.vx + 0.04 * Math.sin(this.t * 0.7 + p.twinkle);
      p.y += p.vy;

      const prog   = p.life / p.maxLife;
      // Fade in/out
      const alpha  = prog < 0.2
        ? prog / 0.2
        : prog > 0.75
          ? 1 - (prog - 0.75) / 0.25
          : 1;
      const tw     = 0.5 + 0.5 * Math.sin(this.t * 2.2 + p.twinkle);

      if (p.type === 'spark') {
        // Elongated spark line
        const len = p.sz * (2 + tw * 3);
        const grad = ctx.createLinearGradient(p.x, p.y, p.x + p.vx * len * 8, p.y + p.vy * len * 8);
        grad.addColorStop(0, `rgba(${p.r},${p.g},${p.b},${(alpha * 0.9).toFixed(2)})`);
        grad.addColorStop(1, `rgba(${p.r},${p.g},${p.b},0)`);
        ctx.beginPath();
        ctx.strokeStyle = grad;
        ctx.lineWidth   = p.sz * 0.5;
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + p.vx * len * 8, p.y + p.vy * len * 8);
        ctx.stroke();
        // Core bright dot
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.sz * 0.6, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${(alpha * tw * 0.85).toFixed(2)})`;
        ctx.fill();

      } else if (p.type === 'rune') {
        // Small diamond / cross rune
        const sz2 = p.sz * (1.2 + tw * 0.8);
        ctx.save();
        ctx.translate(p.x, p.y);
        ctx.rotate(this.t * 0.4 + p.twinkle);
        ctx.beginPath();
        ctx.moveTo(0, -sz2 * 2);
        ctx.lineTo(sz2, 0);
        ctx.lineTo(0, sz2 * 2);
        ctx.lineTo(-sz2, 0);
        ctx.closePath();
        ctx.strokeStyle = `rgba(${p.r},${p.g},${p.b},${(alpha * 0.55).toFixed(2)})`;
        ctx.lineWidth   = 0.4;
        ctx.stroke();
        ctx.restore();

      } else {
        // Standard dust mote
        const sz2 = p.sz * (0.7 + tw * 0.4);
        const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, sz2 * 3);
        grd.addColorStop(0, `rgba(${p.r},${p.g},${p.b},${(alpha * 0.55).toFixed(2)})`);
        grd.addColorStop(1, `rgba(${p.r},${p.g},${p.b},0)`);
        ctx.beginPath(); ctx.arc(p.x, p.y, sz2 * 3, 0, Math.PI * 2);
        ctx.fillStyle = grd; ctx.fill();

        ctx.beginPath(); ctx.arc(p.x, p.y, sz2, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(255,255,255,${(alpha * tw * 0.6).toFixed(2)})`; ctx.fill();
      }
    });
  }
}

// ─────────────────────────────────────────────────────────────
// CSS INJECTION  (styles live here, not in an external file)
// ─────────────────────────────────────────────────────────────
(function injectStyles() {
  if (document.getElementById('axium-card-system-styles')) return;
  const style = document.createElement('style');
  style.id = 'axium-card-system-styles';
  style.textContent = `
/* ══════════════════════════════════════════
   AXIUM CARD SYSTEM — Base Card
══════════════════════════════════════════ */
.axc-card {
  position: relative;
  width: var(--card-w, 110px);
  aspect-ratio: 63/88;
  border-radius: 10px;
  background: linear-gradient(168deg, rgba(14,11,30,.98) 0%, rgba(6,4,16,.99) 60%, rgba(10,8,22,.98) 100%);
  border: 1px solid rgba(255,255,255,.1);
  overflow: hidden;
  cursor: pointer;
  flex-shrink: 0;
  transform-style: preserve-3d;
  transition: transform .28s cubic-bezier(.22,1,.36,1),
              box-shadow .28s cubic-bezier(.22,1,.36,1),
              border-color .28s;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
}

/* Size variants */
.axc-card.axc-sm  { --card-w: 72px;  }
.axc-card.axc-md  { --card-w: 110px; }
.axc-card.axc-lg  { --card-w: 160px; }
.axc-card.axc-xl  { --card-w: 220px; }

/* Hover lift */
.axc-card:hover   { transform: translateY(-6px) scale(1.03); }
.axc-card.axc-sm:hover { transform: translateY(-3px) scale(1.04); }

/* Staged (selected) ring */
.axc-card.axc-staged {
  border-color: #D4AF37 !important;
  box-shadow: 0 0 0 1px #D4AF3755, 0 0 28px rgba(212,175,55,.45) !important;
}

/* Enemy card tint */
.axc-card.axc-enemy { border-color: rgba(224,85,85,.35); }

/* Exhausted / used */
.axc-card.axc-exhausted { opacity: .28; filter: grayscale(.85); pointer-events: none; }

/* ── Inner layers ───────────────────────── */
.axc-border-glow {
  position: absolute; inset: 0; border-radius: 10px; pointer-events: none; z-index: 1;
  background: radial-gradient(ellipse at 30% 15%, var(--card-color-a, transparent) 0%, transparent 60%);
  opacity: .18; transition: opacity .35s;
}
.axc-card:hover .axc-border-glow { opacity: .3; }

/* Particle canvas — full-card coverage */
.axc-particle-cvs {
  position: absolute; inset: 0; width: 100%; height: 100%;
  border-radius: 10px; pointer-events: none; z-index: 2;
}

/* ── Header strip — name only, no layer label ───────────────── */
.axc-header {
  position: relative; z-index: 5;
  padding: 5px 7px 4px;
  flex-shrink: 0;
}
.axc-layer-lbl {
  display: none;
}
.axc-card-name {
  font-family: 'Cinzel', serif;
  font-size: clamp(8px, 1.8vw, 11px);
  font-weight: 600; letter-spacing: .04em; line-height: 1.15;
  display: block;
}
/* Ghost numeral behind name */
.axc-numeral-ghost {
  position: absolute; top: 2px; right: 5px;
  font-family: 'Cinzel Decorative', serif;
  font-size: 20px; font-weight: 900;
  letter-spacing: .04em; line-height: 1;
  color: rgba(255,255,255,.04);
  pointer-events: none; user-select: none; z-index: 0;
}

/* ── Constellation canvas ───────────────── */
.axc-constellation-wrap {
  position: relative; z-index: 5;
  flex: 1; min-height: 0; overflow: hidden;
}
.axc-const-cvs {
  width: 100%; height: 100%; display: block;
}

/* ── Footer strip — hidden, expand modal shows details ───────── */
.axc-footer {
  display: none;
}
.axc-type-pip {
  font-family: 'Space Mono', monospace;
  font-size: 5px; letter-spacing: .06em; text-transform: uppercase;
  padding: 1px 4px; border-radius: 2px; border: 1px solid;
}
.axc-type-pip.compression   { color: rgba(220,100,60,.85); border-color: rgba(220,100,60,.3); }
.axc-type-pip.decompression { color: rgba(126,184,232,.85); border-color: rgba(126,184,232,.3); }
.axc-type-pip.both          { color: rgba(212,175,55,.85); border-color: rgba(212,175,55,.3); }
.axc-axium-score {
  font-family: 'JetBrains Mono', monospace;
  font-size: 9px; font-weight: 600; color: rgba(255,255,255,.38);
}
/* Layer colour dot */
.axc-layer-dot {
  width: 5px; height: 5px; border-radius: 50%;
  box-shadow: 0 0 5px currentColor;
}

/* Staged check badge */
.axc-staged-badge {
  position: absolute; top: 5px; right: 5px; z-index: 10;
  width: 16px; height: 16px; border-radius: 50%;
  background: rgba(212,175,55,.18); border: 1.5px solid #D4AF37;
  display: flex; align-items: center; justify-content: center;
  opacity: 0; transition: opacity .2s; pointer-events: none;
}
.axc-card.axc-staged .axc-staged-badge { opacity: 1; }
.axc-staged-badge svg { width: 8px; height: 8px; color: #D4AF37; }

/* ══════════════════════════════════════════
   HAND PICKER MODAL
══════════════════════════════════════════ */
#axc-hand-picker {
  position: absolute; inset: 0; z-index: 60;
  background: rgba(2, 2, 10, .96);
  display: none; flex-direction: column;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}
#axc-hand-picker.axc-open { display: flex; }

.axc-picker-hdr {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 16px; border-bottom: 1px solid rgba(255,255,255,.06);
  flex-shrink: 0;
}
.axc-picker-title {
  font-family: 'Cinzel', serif; font-size: 17px;
  letter-spacing: .08em; color: #D4AF37;
}
.axc-picker-count {
  font-family: 'JetBrains Mono', monospace;
  font-size: 12px; color: rgba(255,255,255,.3);
}
.axc-picker-close {
  font-family: 'Space Mono', monospace; font-size: 9px;
  letter-spacing: .12em; text-transform: uppercase;
  color: rgba(255,255,255,.3); background: none;
  border: 1px solid rgba(255,255,255,.1); padding: 6px 10px;
  cursor: pointer; border-radius: 2px; transition: all .2s;
}
.axc-picker-close:hover { color: rgba(255,255,255,.6); border-color: rgba(255,255,255,.25); }

/* Filter tabs */
.axc-picker-tabs {
  display: flex; gap: 6px; padding: 8px 16px;
  border-bottom: 1px solid rgba(255,255,255,.04);
  flex-shrink: 0; overflow-x: auto; scrollbar-width: none;
}
.axc-picker-tabs::-webkit-scrollbar { display: none; }
.axc-ptab {
  font-family: 'Space Mono', monospace; font-size: 8px;
  letter-spacing: .14em; text-transform: uppercase;
  padding: 4px 10px; border-radius: 20px; border: 1px solid rgba(255,255,255,.1);
  color: rgba(255,255,255,.28); background: none; cursor: pointer;
  transition: all .2s; white-space: nowrap; flex-shrink: 0;
}
.axc-ptab.active {
  color: #0a0a0a; background: #D4AF37; border-color: #D4AF37;
}
.axc-ptab[data-layer="Ego"].active    { background: #7EB8E8; border-color: #7EB8E8; }
.axc-ptab[data-layer="ID"].active     { background: #86EFAC; border-color: #86EFAC; color: #0a0a0a; }

/* Card scroll area */
.axc-picker-scroll {
  flex: 1; overflow-y: auto; padding: 14px 12px;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 10px; align-content: start;
}
.axc-picker-scroll::-webkit-scrollbar { width: 3px; }
.axc-picker-scroll::-webkit-scrollbar-thumb { background: rgba(212,175,55,.2); border-radius: 2px; }

/* ── Card Tooltip (mechanic on hover) ── */
.axc-card-tooltip {
  position: absolute; inset: 0; z-index: 8;
  background: linear-gradient(180deg, transparent 30%, rgba(2,2,10,.96) 70%);
  border-radius: 10px;
  display: flex; flex-direction: column; justify-content: flex-end;
  padding: 6px; opacity: 0;
  transition: opacity .25s cubic-bezier(.22,1,.36,1);
  pointer-events: none;
}
.axc-card:hover .axc-card-tooltip { opacity: 1; }
.axc-tooltip-mech {
  font-family: 'Cormorant Garamond', serif; font-style: italic;
  font-size: 9px; line-height: 1.5; color: rgba(255,255,255,.65);
  text-align: center;
}
.axc-tooltip-kw {
  font-family: 'Space Mono', monospace; font-size: 7px;
  letter-spacing: .1em; text-transform: uppercase;
  color: rgba(255,255,255,.3); text-align: center; margin-top: 2px;
}

/* Picker footer */
.axc-picker-footer {
  padding: 10px 14px 14px; border-top: 1px solid rgba(255,255,255,.05);
  display: flex; align-items: center; gap: 10px; flex-shrink: 0;
}
.axc-picker-resolve {
  flex: 1; font-family: 'Cinzel', serif; font-size: 13px;
  letter-spacing: .16em; text-transform: uppercase;
  color: #0a0a0a; background: linear-gradient(135deg,#AA8C2C,#D4AF37,#AA8C2C);
  border: none; padding: 12px; cursor: pointer; border-radius: 2px;
  opacity: 0; pointer-events: none;
  transition: opacity .25s, box-shadow .25s;
}
.axc-picker-resolve.ready { opacity: 1; pointer-events: auto; }
.axc-picker-resolve:hover.ready { box-shadow: 0 0 22px rgba(212,175,55,.5); }
.axc-picker-msg {
  font-family: 'Cormorant Garamond', serif; font-style: italic;
  font-size: 12px; color: rgba(255,255,255,.28);
}

/* Polarity toggle pill — shown on staged cards inside the picker */
.axc-pol-pill {
  position: absolute; bottom: 6px; left: 50%; transform: translateX(-50%);
  font-family: 'Space Mono', monospace; font-size: 7px; letter-spacing: .08em;
  text-transform: uppercase; padding: 3px 9px; border-radius: 10px; border: 1px solid;
  cursor: pointer; z-index: 20; white-space: nowrap; user-select: none;
  transition: all .18s;
}
.axc-pol-pill.normal   { color:rgba(212,175,55,.9); border-color:rgba(212,175,55,.45); background:rgba(212,175,55,.12); }
.axc-pol-pill.reversed { color:rgba(224,85,85,.9);  border-color:rgba(224,85,85,.5);  background:rgba(224,85,85,.13); }

/* Synergy indicators */
.axc-syn-ready {
  font-family: 'Space Mono', monospace; font-size: 7.5px;
  letter-spacing: .1em; text-transform: uppercase;
  color: #D4AF37; background: rgba(212,175,55,.08);
  border: 1px solid rgba(212,175,55,.25);
  padding: 2px 7px; border-radius: 20px;
  white-space: nowrap;
}

/* ══════════════════════════════════════════
   CARD DETAIL MODAL (full-screen flip)
══════════════════════════════════════════ */
#axc-detail-modal {
  position: fixed; inset: 0; z-index: 300;
  background: rgba(2,2,10,.0);
  display: flex; align-items: center; justify-content: center;
  pointer-events: none;
  transition: background .35s;
}
#axc-detail-modal.axc-open {
  background: rgba(2,2,10,.88);
  pointer-events: auto;
}

.axc-detail-scene {
  perspective: 900px;
  width: min(320px, 88vw);
}

.axc-detail-flipper {
  width: 100%; position: relative;
  transform-style: preserve-3d;
  transition: transform .65s cubic-bezier(.22,1,.36,1);
}
.axc-detail-flipper.axc-flipped { transform: rotateY(180deg); }

.axc-detail-face {
  width: 100%; border-radius: 16px; backface-visibility: hidden;
  -webkit-backface-visibility: hidden;
  overflow: hidden;
}
.axc-detail-front {
  /* Full-size card face — we borrow the card element */
  transform: rotateY(0deg);
}
.axc-detail-back {
  position: absolute; inset: 0;
  transform: rotateY(180deg);
  background: linear-gradient(168deg, rgba(14,11,30,.99) 0%, rgba(6,4,16,1) 100%);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 16px;
  display: flex; flex-direction: column;
  overflow: hidden;
}

/* Back face contents */
.axcd-back-header {
  padding: 16px 18px 12px; border-bottom: 1px solid rgba(255,255,255,.06);
  flex-shrink: 0;
}
.axcd-back-layer {
  font-family: 'Space Mono', monospace; font-size: 8px;
  letter-spacing: .24em; text-transform: uppercase;
  opacity: .38; margin-bottom: 4px;
}
.axcd-back-name {
  font-family: 'Cinzel Decorative', serif; font-size: 22px;
  font-weight: 700; letter-spacing: .06em; line-height: 1.1;
}
.axcd-back-keywords {
  font-family: 'Cormorant Garamond', serif; font-style: italic;
  font-size: 13px; color: rgba(255,255,255,.32); margin-top: 4px;
}

/* Large constellation canvas on back */
.axcd-const-wrap {
  height: 130px; flex-shrink: 0; position: relative; overflow: hidden;
  background: rgba(0,0,0,.25);
}
.axcd-const-cvs { width: 100%; height: 100%; display: block; }

/* Mechanic body */
.axcd-body {
  flex: 1; padding: 12px 18px;
  overflow-y: auto; display: flex; flex-direction: column; gap: 10px;
}
.axcd-body::-webkit-scrollbar { width: 2px; }
.axcd-body::-webkit-scrollbar-thumb { background: rgba(212,175,55,.2); }

.axcd-section-lbl {
  font-family: 'Space Mono', monospace; font-size: 7.5px;
  letter-spacing: .2em; text-transform: uppercase;
  color: rgba(255,255,255,.18); margin-bottom: 3px;
}
.axcd-effect-text {
  font-family: 'Cormorant Garamond', serif; font-style: italic;
  font-size: 13px; line-height: 1.8; color: rgba(255,255,255,.62);
}
.axcd-stats-row {
  display: grid; grid-template-columns: 1fr 1fr;
  gap: 6px;
}
.axcd-stat {
  padding: 6px 9px; background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.05); border-radius: 4px;
}
.axcd-stat-val {
  font-family: 'JetBrains Mono', monospace; font-size: 14px;
  font-weight: 600;
}
.axcd-stat-lbl {
  font-family: 'Space Mono', monospace; font-size: 7px;
  letter-spacing: .14em; text-transform: uppercase;
  color: rgba(255,255,255,.2); margin-top: 2px;
}
.axcd-synergy-row {
  display: flex; flex-wrap: wrap; gap: 4px;
}
.axcd-syn-pill {
  font-family: 'Space Mono', monospace; font-size: 7px;
  letter-spacing: .1em; text-transform: uppercase;
  padding: 2px 7px; border-radius: 20px; border: 1px solid;
}
.axcd-syn-pill.ready { color: #D4AF37; border-color: rgba(212,175,55,.45); background: rgba(212,175,55,.07); }
.axcd-syn-pill.potential { color: rgba(255,255,255,.32); border-color: rgba(255,255,255,.1); }

/* Footer actions */
.axcd-back-footer {
  padding: 10px 16px 14px; border-top: 1px solid rgba(255,255,255,.05);
  display: flex; gap: 8px; flex-shrink: 0;
}
.axcd-btn {
  font-family: 'Cinzel', serif; letter-spacing: .16em; text-transform: uppercase;
  padding: 10px 16px; cursor: pointer; border-radius: 2px; font-size: 11px;
  transition: all .22s;
}
.axcd-btn.stage {
  flex: 1;
  background: linear-gradient(135deg,#AA8C2C,#D4AF37,#AA8C2C);
  color: #0a0a0a; border: none;
}
.axcd-btn.stage:hover { box-shadow: 0 0 20px rgba(212,175,55,.5); }
.axcd-btn.stage.staged { background: rgba(212,175,55,.1); color: #D4AF37; border: 1px solid rgba(212,175,55,.35); }
.axcd-btn.close-back {
  background: rgba(255,255,255,.04); color: rgba(255,255,255,.35);
  border: 1px solid rgba(255,255,255,.09);
}
.axcd-btn.close-back:hover { color: rgba(255,255,255,.6); }

/* Modal backdrop click target */
.axc-detail-backdrop {
  position: fixed; inset: 0; z-index: -1;
}

/* ══════════════════════════════════════════
   FIELD ZONE
══════════════════════════════════════════ */
.axc-field-zone {
  display: flex; flex-wrap: wrap; gap: 5px;
  align-items: flex-end; padding: 4px 0; min-height: 40px;
}
.axc-field-empty {
  font-family: 'Space Mono', monospace; font-size: 8px;
  letter-spacing: .12em; text-transform: uppercase;
  color: rgba(255,255,255,.07); align-self: center; padding: 4px 0;
}
.axc-field-empty.enemy { color: rgba(224,85,85,.12); }

/* Remove badge on field mini cards */
.axc-remove-badge {
  position: absolute; top: -5px; right: -5px; z-index: 20;
  width: 16px; height: 16px; border-radius: 50%;
  background: rgba(224,85,85,.85); border: 1px solid #e05555;
  display: flex; align-items: center; justify-content: center;
  cursor: pointer; transition: transform .15s, background .15s;
  font-size: 9px; color: white; font-weight: bold; line-height: 1;
}
.axc-remove-badge:hover { background: #e05555; transform: scale(1.2); }

/* ══════════════════════════════════════════
   ENTRANCE ANIMATIONS
══════════════════════════════════════════ */
@keyframes axcCardIn {
  from { opacity: 0; transform: translateY(12px) scale(.9); }
  to   { opacity: 1; transform: translateY(0) scale(1); }
}
.axc-card { animation: axcCardIn .32s cubic-bezier(.22,1,.36,1) both; }

@keyframes axcPickerIn {
  from { opacity: 0; transform: translateX(0) scale(.97); }
  to   { opacity: 1; transform: scale(1); }
}
#axc-hand-picker.axc-open { animation: axcPickerIn .3s cubic-bezier(.22,1,.36,1) both; }

@keyframes axcModalIn {
  from { opacity: 0; transform: scale(.88) rotateY(-8deg); }
  to   { opacity: 1; transform: scale(1) rotateY(0deg); }
}
#axc-detail-modal.axc-open .axc-detail-scene {
  animation: axcModalIn .45s cubic-bezier(.22,1,.36,1) both;
}
  `;
  document.head.appendChild(style);
})();

// ─────────────────────────────────────────────────────────────
// CARD RENDERER
// Builds a full living card DOM element.
// opts: { size:'sm'|'md'|'lg'|'xl', showTooltip, enemy, staged, onLongPress }
// ─────────────────────────────────────────────────────────────
class CardRenderer {
  constructor(card, opts = {}) {
    this.card    = card;
    this.opts    = Object.assign({
      size:        'md',
      showTooltip: true,
      enemy:       false,
      staged:      false,
      animDelay:   0,
    }, opts);
    this._anims  = [];
    this.el      = this._build();
  }

  _build() {
    const { card, opts } = this;
    const [r,g,b] = hexRGB(card.color || '#ffffff');
    const lc      = layerColor(card);
    const cardType = card.type || 'compression';

    const el = document.createElement('div');
    el.className = `axc-card axc-${opts.size}` +
      (opts.enemy  ? ' axc-enemy'    : '') +
      (opts.staged ? ' axc-staged'   : '');
    el.dataset.cardId = card.id;
    el.style.animationDelay = opts.animDelay + 'ms';
    el.style.setProperty('--card-color-a', `rgba(${r},${g},${b},.6)`);
    el.style.borderColor  = `rgba(${r},${g},${b},.25)`;
    el.style.boxShadow    = `0 0 18px rgba(${r},${g},${b},.1), inset 0 0 30px rgba(${r},${g},${b},.03)`;

    // Hover intensify shadow
    el.addEventListener('mouseenter', () => {
      el.style.boxShadow = `0 8px 32px rgba(0,0,0,.7), 0 0 28px rgba(${r},${g},${b},.35), inset 0 0 30px rgba(${r},${g},${b},.06)`;
      el.style.borderColor = `rgba(${r},${g},${b},.55)`;
    });
    el.addEventListener('mouseleave', () => {
      if (!opts.staged) {
        el.style.boxShadow = `0 0 18px rgba(${r},${g},${b},.1), inset 0 0 30px rgba(${r},${g},${b},.03)`;
        el.style.borderColor = `rgba(${r},${g},${b},.25)`;
      }
    });

    // Border glow layer
    const borderGlow = document.createElement('div');
    borderGlow.className = 'axc-border-glow';
    el.appendChild(borderGlow);

    // Particle canvas — only for picker/shop cards (md/lg/xl), not field cards (sm)
    // Field cards use the constellation charge animation instead
    let ptclCvs = null;
    if (opts.size !== 'sm') {
      ptclCvs = document.createElement('canvas');
      ptclCvs.className = 'axc-particle-cvs';
      el.appendChild(ptclCvs);
    }

    // Header
    const header = document.createElement('div');
    header.className = 'axc-header';
    header.innerHTML = `
      <span class="axc-layer-lbl" style="color:${lc}">${cardLayerLabel(card)}</span>
      <span class="axc-card-name" style="color:${card.color}">${card.name}</span>
    `;
    // Ghost numeral for Superego
    if (card.layer === 'Superego' && card.number !== undefined) {
      const ghost = document.createElement('span');
      ghost.className = 'axc-numeral-ghost';
      ghost.textContent = toRoman(card.number);
      header.appendChild(ghost);
    }
    el.appendChild(header);

    // Constellation canvas wrapper
    const constWrap = document.createElement('div');
    constWrap.className = 'axc-constellation-wrap';
    const constCvs = document.createElement('canvas');
    constCvs.className = 'axc-const-cvs';
    constWrap.appendChild(constCvs);
    el.appendChild(constWrap);

    // Tooltip overlay
    if (opts.showTooltip) {
      const mechText = this._mechText();
      const tooltip  = document.createElement('div');
      tooltip.className = 'axc-card-tooltip';
      tooltip.innerHTML = `
        <div class="axc-tooltip-mech">${mechText}</div>
        <div class="axc-tooltip-kw">${card.keywords || ''}</div>
      `;
      el.appendChild(tooltip);
    }

    // Footer
    const footer = document.createElement('div');
    footer.className = 'axc-footer';
    const layerDot = document.createElement('div');
    layerDot.className = 'axc-layer-dot';
    layerDot.style.background = lc;
    layerDot.style.color = lc;
    footer.innerHTML = `
      <span class="axc-type-pip ${cardType}">${cardType.slice(0,4)}</span>
      <span class="axc-axium-score" style="color:rgba(${r},${g},${b},.55)">⬡${card.axiumScore ?? '?'}</span>
    `;
    footer.prepend(layerDot);
    el.appendChild(footer);

    // Staged check badge
    const badge = document.createElement('div');
    badge.className = 'axc-staged-badge';
    badge.innerHTML = `<svg viewBox="0 0 12 12" fill="none"><polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
    el.appendChild(badge);

    // Start animations after DOM insertion
    requestAnimationFrame(() => {
      if (!constCvs.offsetWidth) return;
      const constAnim = new ConstellationAnim(constCvs, card, {
        speed:     0.016,
        nodeSize:  opts.size === 'sm' ? 1.1 : opts.size === 'lg' ? 2.4 : 1.8,
        edgeAlpha: opts.size === 'sm' ? 0.22 : 0.32,
        glowMult:  opts.size === 'sm' ? 1.2 : 3.5,
      });
      constAnim.start();
      this._anims.push(constAnim);
      this._constAnim = constAnim; // direct ref for setCharge

      // Particles only on non-field cards
      if (ptclCvs) {
        const ptclAnim = new CardParticles(ptclCvs, card);
        ptclAnim.start();
        this._anims.push(ptclAnim);
      }
    });

    return el;
  }

  _mechText() {
    const { card } = this;
    if (card.layer === 'Superego') {
      return `Shield +${card.shieldVal || 0}` +
        (card.capacityVal ? ` · Cap ${card.capacityVal > 0 ? '+' : ''}${card.capacityVal}` : '');
    }
    if (card.layer === 'Ego') {
      return [
        card.chunkFlat  ? `+${card.chunkFlat} flat`    : '',
        card.chunkPct   ? `×${card.chunkPct}`           : '',
        card.studyMult  ? `Study ×${card.studyMult}`    : '',
      ].filter(Boolean).join(' · ');
    }
    if (card.layer === 'ID') {
      return `Recharge +${card.rechargeVal || 0}/stack`;
    }
    return card.keywords || '';
  }

  setStaged(bool) {
    this.opts.staged = bool;
    this.el.classList.toggle('axc-staged', bool);
    const [r,g,b] = hexRGB(this.card.color || '#ffffff');
    if (bool) {
      this.el.style.borderColor = '#D4AF37';
      this.el.style.boxShadow   = '0 0 0 1px rgba(212,175,55,.45), 0 0 28px rgba(212,175,55,.4)';
    } else {
      this.el.style.borderColor = `rgba(${r},${g},${b},.25)`;
      this.el.style.boxShadow   = `0 0 18px rgba(${r},${g},${b},.1)`;
    }
  }

  // ── Charge / burst API (called from battle playback) ────────
  // progress 0→1: lights constellation nodes sequentially.
  // At 1.0 all nodes are fully lit — signals card is about to fire.
  setCharge(progress) {
    if (this._constAnim) this._constAnim.setCharge(progress);
  }

  // Called when the card fires — flashes all nodes white then resets.
  triggerBurst() {
    if (this._constAnim) this._constAnim.triggerBurst();
  }

  destroy() {
    this._anims.forEach(a => a.stop());
    this._anims = [];
    if (this.el.parentElement) this.el.parentElement.removeChild(this.el);
  }
}

// ─────────────────────────────────────────────────────────────
// HAND PICKER MODAL
// Full-screen overlay with card grid, filters, and staging.
// Usage: new HandPickerModal(parentEl, hand, playerPlayed, opts)
// opts: { maxStaged, onStagedChange, onResolve, onClose }
// ─────────────────────────────────────────────────────────────
class HandPickerModal {
  constructor(parentEl, hand, playerPlayed, opts = {}) {
    this.parent       = parentEl;
    this.hand         = hand;
    this.playerPlayed = playerPlayed;   // shared array reference
    this.opts         = Object.assign({
      maxStaged:        10,
      onStagedChange:   null,
      onResolve:        null,
      onClose:          null,
      getSynergies:     null,
    }, opts);
    this._renderers   = new Map();  // cardId → CardRenderer
    this._activeFilter = 'All';
    this.el           = null;
    this._build();
  }

  _build() {
    // Remove any existing
    const old = this.parent.querySelector('#axc-hand-picker');
    if (old) old.remove();

    const el = document.createElement('div');
    el.id = 'axc-hand-picker';
    el.innerHTML = `
      <div class="axc-picker-hdr">
        <span class="axc-picker-title">Your Hand</span>
        <span class="axc-picker-count" id="axcp-count">0/10</span>
        <button class="axc-picker-close" id="axcp-close">Done ×</button>
      </div>
      <div class="axc-picker-tabs" id="axcp-tabs">
        <button class="axc-ptab active" data-layer="All">All</button>
        <button class="axc-ptab" data-layer="Superego">Superego</button>
        <button class="axc-ptab" data-layer="Ego">Ego</button>
        <button class="axc-ptab" data-layer="ID">ID</button>
      </div>
      <div class="axc-picker-scroll" id="axcp-grid"></div>
      <div class="axc-picker-footer">
        <span class="axc-picker-msg" id="axcp-msg">Select cards to stage</span>
        <button class="axc-picker-resolve" id="axcp-resolve">Resolve ✦</button>
      </div>
    `;
    this.parent.appendChild(el);
    this.el = el;

    // Close btn
    el.querySelector('#axcp-close').addEventListener('click', () => this.close());

    // Tabs
    el.querySelectorAll('.axc-ptab').forEach(tab => {
      tab.addEventListener('click', () => {
        el.querySelectorAll('.axc-ptab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        this._activeFilter = tab.dataset.layer;
        this._renderGrid();
      });
    });

    // Resolve btn
    el.querySelector('#axcp-resolve').addEventListener('click', () => {
      if (this.playerPlayed.length > 0) {
        this.close();
        this.opts.onResolve?.();
      }
    });

    this._renderGrid();
    this._updateCount();
  }

  _renderGrid() {
    const grid = this.el.querySelector('#axcp-grid');
    grid.innerHTML = '';
    this._renderers.forEach((r, id) => {
      if (!this.hand.find(c => c.id === id)) r.destroy();
    });
    this._renderers.clear();

    const filtered = this._activeFilter === 'All'
      ? this.hand
      : this.hand.filter(c => c.layer === this._activeFilter);

    filtered.forEach((card, i) => {
      // playerPlayed holds {card, reversed} entries
      const entry    = this.playerPlayed.find(e => (e.card || e).id === card.id);
      const isStaged = !!entry;
      const isRev    = entry ? entry.reversed : false;

      const renderer = new CardRenderer(card, {
        size:        'md',
        showTooltip: true,
        staged:      isStaged,
        animDelay:   i * 35,
      });
      renderer.el.style.cursor = 'pointer';

      // Apply reversed tint if staged-reversed
      if (isStaged && isRev) this._applyReversedStyle(renderer.el, card);

      // Tap = toggle staged (but NOT if tapping the polarity pill)
      renderer.el.addEventListener('click', e => {
        if (e.target.closest('.axc-pol-pill')) return;
        this._toggleCard(card, renderer);
      });
      renderer.el.addEventListener('contextmenu', e => {
        e.preventDefault();
        this._openDetail(card);
      });

      let pressTimer;
      renderer.el.addEventListener('pointerdown', e => {
        if (e.target.closest('.axc-pol-pill')) return;
        pressTimer = setTimeout(() => this._openDetail(card), 600);
      });
      renderer.el.addEventListener('pointerup',   () => clearTimeout(pressTimer));
      renderer.el.addEventListener('pointermove', () => clearTimeout(pressTimer));

      // Polarity pill — only visible when staged
      if (isStaged) {
        this._addPolarityPill(renderer.el, card, entry);
      }

      grid.appendChild(renderer.el);
      this._renderers.set(card.id, renderer);
    });

    this._updateCount();
  }

  // Adds the ↑ NRM / ↓ REV pill onto a card element
  _addPolarityPill(cardEl, card, entry) {
    const pill = document.createElement('div');
    pill.className = 'axc-pol-pill ' + (entry.reversed ? 'reversed' : 'normal');
    pill.textContent = entry.reversed ? '↓ REV' : '↑ NRM';
    pill.title = entry.reversed ? 'Reversed — tap to set Normal' : 'Normal — tap to set Reversed';
    pill.addEventListener('click', e => {
      e.stopPropagation();
      entry.reversed = !entry.reversed;
      // Update pill appearance
      pill.className  = 'axc-pol-pill ' + (entry.reversed ? 'reversed' : 'normal');
      pill.textContent = entry.reversed ? '↓ REV' : '↑ NRM';
      pill.title = entry.reversed ? 'Reversed — tap to set Normal' : 'Normal — tap to set Reversed';
      // Update card border/shadow tint
      if (entry.reversed) {
        this._applyReversedStyle(cardEl, card);
      } else {
        cardEl.style.borderColor = '#D4AF37';
        cardEl.style.boxShadow   = '0 0 0 1px rgba(212,175,55,.45), 0 0 28px rgba(212,175,55,.4)';
      }
      this.opts.onStagedChange?.([...this.playerPlayed]);
    });
    cardEl.appendChild(pill);
  }

  // Red tint for reversed-staged cards
  _applyReversedStyle(cardEl, card) {
    cardEl.style.borderColor = 'rgba(224,85,85,.7)';
    cardEl.style.boxShadow   = '0 0 0 1px rgba(224,85,85,.4), 0 0 22px rgba(224,85,85,.35)';
  }

  _toggleCard(card, renderer) {
    const idx = this.playerPlayed.findIndex(e => (e.card || e).id === card.id);
    if (idx >= 0) {
      // Unstage
      this.playerPlayed.splice(idx, 1);
      renderer.setStaged(false);
      // Reset any reversed styling
      const [r,g,b] = hexRGB(card.color || '#ffffff');
      renderer.el.style.borderColor = `rgba(${r},${g},${b},.25)`;
      renderer.el.style.boxShadow   = `0 0 18px rgba(${r},${g},${b},.1)`;
      // Remove polarity pill
      renderer.el.querySelector('.axc-pol-pill')?.remove();
    } else {
      if (this.playerPlayed.length >= this.opts.maxStaged) {
        renderer.el.style.animation = 'none';
        renderer.el.style.transform = 'translateX(-4px)';
        setTimeout(() => { renderer.el.style.transform = ''; renderer.el.style.animation = ''; }, 120);
        return;
      }
      // Stage as {card, reversed:false}
      const entry = { card, reversed: false };
      this.playerPlayed.push(entry);
      renderer.setStaged(true);
      this._addPolarityPill(renderer.el, card, entry);
    }
    this._updateCount();
    this.opts.onStagedChange?.([...this.playerPlayed]);
  }

  _updateCount() {
    const n   = this.playerPlayed.length;
    const max = this.opts.maxStaged;
    const countEl   = this.el.querySelector('#axcp-count');
    const msgEl     = this.el.querySelector('#axcp-msg');
    const resolveEl = this.el.querySelector('#axcp-resolve');
    if (countEl) countEl.textContent = `${n}/${max}`;
    if (msgEl)   msgEl.textContent   = n === 0 ? 'Select cards to stage'
                                     : n >= max  ? 'Hand full — ready to Resolve!'
                                     : `${n} staged · tap to toggle`;
    if (resolveEl) resolveEl.classList.toggle('ready', n > 0);

    // Synergy indicators — extract card ids from {card,reversed} entries
    if (this.opts.getSynergies && n > 0) {
      const ids  = this.playerPlayed.map(e => (e.card || e).id);
      const syns = this.opts.getSynergies(ids);
      if (syns.length > 0 && msgEl) {
        const names = syns.slice(0, 2).map(s => s.name).join(' · ');
        msgEl.innerHTML = `<span class="axc-syn-ready">✦ ${names}</span>`;
      }
    }
  }

  _openDetail(card) {
    if (!window.AxiumCardDetail) return;
    // AxiumCardDetail.open expects plain card objects in the played array
    const playedCards = this.playerPlayed.map(e => e.card || e);
    window.AxiumCardDetail.open(card, playedCards, {
      onStageChange: () => {
        this._renderers.forEach((r, id) => {
          const entry = this.playerPlayed.find(e => (e.card || e).id === id);
          r.setStaged(!!entry);
        });
        this._updateCount();
        this.opts.onStagedChange?.([...this.playerPlayed]);
      },
    });
  }

  open() {
    this._renderGrid();
    this.el.classList.add('axc-open');
  }

  close() {
    this.el.classList.remove('axc-open');
    this.opts.onClose?.();
  }

  destroy() {
    this._renderers.forEach(r => r.destroy());
    this._renderers.clear();
    this.el?.remove();
  }
}

// ─────────────────────────────────────────────────────────────
// CARD DETAIL MODAL  (singleton)
// ─────────────────────────────────────────────────────────────
const CardDetailModal = (() => {
  let modalEl    = null;
  let constAnim  = null;
  let ptclAnim   = null;
  let currentCard = null;
  let currentPlayed = null;
  let currentOpts = {};

  function _inject() {
    if (document.getElementById('axc-detail-modal')) return;
    const el = document.createElement('div');
    el.id = 'axc-detail-modal';
    el.innerHTML = `
      <div class="axc-detail-backdrop" id="axcd-backdrop"></div>
      <div class="axc-detail-scene">
        <div class="axc-detail-flipper" id="axcd-flipper">
          <div class="axc-detail-face axc-detail-front" id="axcd-front">
            <!-- Large card portrait rendered here -->
          </div>
          <div class="axc-detail-face axc-detail-back">
            <div class="axcd-back-header">
              <div class="axcd-back-layer"  id="axcd-blayer"></div>
              <div class="axcd-back-name"   id="axcd-bname"></div>
              <div class="axcd-back-keywords" id="axcd-bkw"></div>
            </div>
            <div class="axcd-const-wrap">
              <canvas class="axcd-const-cvs" id="axcd-bcvs"></canvas>
            </div>
            <div class="axcd-body">
              <div>
                <div class="axcd-section-lbl">Effect</div>
                <div class="axcd-effect-text" id="axcd-beffect"></div>
              </div>
              <div class="axcd-stats-row" id="axcd-bstats"></div>
              <div id="axcd-bsyn-wrap">
                <div class="axcd-section-lbl">Synergies</div>
                <div class="axcd-synergy-row" id="axcd-bsyn"></div>
              </div>
            </div>
            <div class="axcd-back-footer">
              <button class="axcd-btn stage" id="axcd-stage-btn">Stage Card</button>
              <button class="axcd-btn close-back" id="axcd-flip-back">←</button>
            </div>
          </div>
        </div>
      </div>
    `;
    document.body.appendChild(el);
    modalEl = el;

    el.querySelector('#axcd-backdrop').addEventListener('click', close);
    el.querySelector('#axcd-flip-back').addEventListener('click', () => {
      el.querySelector('#axcd-flipper').classList.remove('axc-flipped');
    });
    el.querySelector('#axcd-stage-btn').addEventListener('click', _toggleStage);
  }

  function _toggleStage() {
    if (!currentCard || !currentPlayed) return;
    const idx = currentPlayed.findIndex(c => c.id === currentCard.id);
    if (idx >= 0) {
      currentPlayed.splice(idx, 1);
    } else {
      if (currentPlayed.length < 10) currentPlayed.push(currentCard);
    }
    _updateStageBtn();
    currentOpts.onStageChange?.();
  }

  function _updateStageBtn() {
    const btn = modalEl?.querySelector('#axcd-stage-btn');
    if (!btn || !currentCard || !currentPlayed) return;
    const staged = !!currentPlayed.find(c => c.id === currentCard.id);
    btn.textContent = staged ? 'Remove from Stage' : 'Stage Card';
    btn.classList.toggle('staged', staged);
  }

  function open(card, playerPlayed, opts = {}) {
    _inject();
    currentCard   = card;
    currentPlayed = playerPlayed;
    currentOpts   = opts;
    const [r,g,b] = hexRGB(card.color || '#ffffff');
    const lc      = layerColor(card);

    // Front — large card portrait
    const front = modalEl.querySelector('#axcd-front');
    front.innerHTML = '';
    const frontRenderer = new CardRenderer(card, { size: 'xl', showTooltip: false });
    // Make the portrait card fill the scene width
    frontRenderer.el.style.setProperty('--card-w', '100%');
    frontRenderer.el.style.cursor = 'pointer';
    frontRenderer.el.addEventListener('click', () => {
      modalEl.querySelector('#axcd-flipper').classList.add('axc-flipped');
      _populateBack(card);
    });
    front.appendChild(frontRenderer.el);

    // Populate back face (without flipping yet)
    _populateBack(card);
    _updateStageBtn();

    // Ensure front face visible
    modalEl.querySelector('#axcd-flipper').classList.remove('axc-flipped');
    modalEl.classList.add('axc-open');
  }

  function _populateBack(card) {
    const lc = layerColor(card);
    const [r,g,b] = hexRGB(card.color || '#ffffff');

    modalEl.querySelector('#axcd-blayer').textContent  = cardLayerLabel(card);
    modalEl.querySelector('#axcd-blayer').style.color  = lc;
    modalEl.querySelector('#axcd-bname').textContent   = card.name;
    modalEl.querySelector('#axcd-bname').style.color   = card.color;
    modalEl.querySelector('#axcd-bkw').textContent     = card.keywords || '';

    // Effect text
    let fx = '';
    if (card.layer === 'Superego') {
      fx = (card.shieldDesc || '') + (card.capacityDesc ? '\n\n' + card.capacityDesc : '');
    } else if (card.layer === 'Ego') {
      fx = (card.chunkDesc || '') + (card.divideDesc ? '\n\nReversed: ' + card.divideDesc : '');
    } else if (card.layer === 'ID') {
      fx = (card.rechargeDesc || '') + (card.drainDesc ? '\n\nReversed: ' + card.drainDesc : '');
    } else {
      fx = card.effectDesc || card.keywords || '';
    }
    modalEl.querySelector('#axcd-beffect').textContent = fx;

    // Stats
    const statsEl = modalEl.querySelector('#axcd-bstats');
    statsEl.innerHTML = '';
    const stats = [
      { label: 'Type',   val: card.type || '—', color: null },
      { label: 'Axium',  val: `${card.axiumScore ?? '?'} / 10`, color: `rgba(${r},${g},${b},.8)` },
    ];
    if (card.layer === 'Superego') {
      stats.push({ label: 'Shield',   val: `+${card.shieldVal || 0}`,   color: '#D4AF37' });
      stats.push({ label: 'Capacity', val: `${card.capacityVal >= 0 ? '+' : ''}${card.capacityVal || 0}`, color: '#D4AF37' });
    }
    if (card.layer === 'ID') {
      stats.push({ label: 'Recharge', val: `+${card.rechargeVal || 0}/stack`, color: '#86EFAC' });
      if (card.drainVal) stats.push({ label: 'Drain', val: `-${card.drainVal}/stack`, color: '#e05555' });
    }
    if (card.layer === 'Ego') {
      if (card.chunkFlat) stats.push({ label: 'Flat Bonus', val: `+${card.chunkFlat}`, color: '#7EB8E8' });
      if (card.chunkPct)  stats.push({ label: 'Multiplier', val: `×${card.chunkPct}`, color: '#7EB8E8' });
    }
    stats.forEach(s => {
      const d = document.createElement('div'); d.className = 'axcd-stat';
      d.innerHTML = `<div class="axcd-stat-val" style="color:${s.color||'rgba(255,255,255,.65)'}">${s.val}</div><div class="axcd-stat-lbl">${s.label}</div>`;
      statsEl.appendChild(d);
    });

    // Synergies
    const synEl = modalEl.querySelector('#axcd-bsyn');
    synEl.innerHTML = '';
    if (typeof SYNERGIES !== 'undefined') {
      const relevantSyns = SYNERGIES.filter(s => s.cards.includes(card.id));
      const played = currentPlayed || [];
      const playedIds = played.map(c => c.id).concat(card.id);
      relevantSyns.forEach(syn => {
        const isReady = syn.cards.every(id => playedIds.includes(id));
        const pill = document.createElement('span');
        pill.className = 'axcd-syn-pill ' + (isReady ? 'ready' : 'potential');
        pill.textContent = isReady ? `✦ ${syn.name}` : syn.name;
        if (isReady) pill.style.borderColor = syn.visual || 'rgba(212,175,55,.45)';
        synEl.appendChild(pill);
      });
    }
    const synWrap = modalEl.querySelector('#axcd-bsyn-wrap');
    if (synWrap) synWrap.style.display = synEl.children.length ? 'block' : 'none';

    // Back constellation
    const cvs = modalEl.querySelector('#axcd-bcvs');
    if (constAnim) constAnim.stop();
    setTimeout(() => {
      if (!cvs.offsetWidth) return;
      constAnim = new ConstellationAnim(cvs, card, {
        speed: 0.015, nodeSize: 2.6, edgeAlpha: 0.4, glowMult: 4,
      });
      constAnim.start();
    }, 50);
  }

  function close() {
    if (constAnim) { constAnim.stop(); constAnim = null; }
    modalEl?.classList.remove('axc-open');
    currentCard = currentPlayed = null;
  }

  return { open, close };
})();

// ─────────────────────────────────────────────────────────────
// FIELD ZONE
// Renders actual small cards in player/enemy field areas.
// Usage: new FieldZone(containerEl, isEnemy)
// ─────────────────────────────────────────────────────────────
class FieldZone {
  constructor(containerEl, isEnemy = false) {
    this.container = containerEl;
    this.isEnemy   = isEnemy;
    this._renderers = new Map();  // cardId → renderer
  }

  render(cards, onRemove = null) {
    // Destroy cards that are no longer in the list
    const newIds = new Set(cards.map(c => c.id));
    this._renderers.forEach((renderer, id) => {
      if (!newIds.has(id)) { renderer.destroy(); this._renderers.delete(id); }
    });

    this.container.innerHTML = '';

    if (!cards.length) {
      const empty = document.createElement('div');
      empty.className = 'axc-field-empty' + (this.isEnemy ? ' enemy' : '');
      empty.textContent = this.isEnemy ? '— enemy awaiting —' : '— no cards staged —';
      this.container.appendChild(empty);
      return;
    }

    cards.forEach((card, i) => {
      let renderer = this._renderers.get(card.id);
      if (!renderer) {
        renderer = new CardRenderer(card, {
          size:        'sm',
          showTooltip: !this.isEnemy,
          enemy:       this.isEnemy,
          animDelay:   i * 40,
        });
        this._renderers.set(card.id, renderer);
      }

      // Wrap in position:relative for remove badge
      const wrap = document.createElement('div');
      wrap.style.cssText = 'position:relative;display:inline-flex;flex-shrink:0;';
      wrap.appendChild(renderer.el);

      // Remove badge for player cards
      if (!this.isEnemy && onRemove) {
        const badge = document.createElement('div');
        badge.className = 'axc-remove-badge';
        badge.textContent = '×';
        badge.addEventListener('click', e => { e.stopPropagation(); onRemove(i, card); });
        wrap.appendChild(badge);
      }

      this.container.appendChild(wrap);
    });
  }

  destroy() {
    this._renderers.forEach(r => r.destroy());
    this._renderers.clear();
    this.container.innerHTML = '';
  }
}

// ─────────────────────────────────────────────────────────────
// EXPORTS
// ─────────────────────────────────────────────────────────────
global.CardRenderer    = CardRenderer;
global.HandPickerModal = HandPickerModal;
global.AxiumCardDetail = CardDetailModal;
global.FieldZone       = FieldZone;
global.CardParticles   = CardParticles;
global.ConstellationAnim = ConstellationAnim;

})(window);
