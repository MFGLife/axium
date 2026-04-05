/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — Card Expand Patch  (inject after csystem.js)
 * Adds a ⊞ expand button to every .axc-card element and
 * replaces the detail modal with a rich full-screen overlay.
 * ═══════════════════════════════════════════════════════════════
 */
;(function(global) {
'use strict';

// ─────────────────────────────────────────────────────────────
// INJECT STYLES
// ─────────────────────────────────────────────────────────────
(function() {
  if (document.getElementById('axium-expand-styles')) return;
  const s = document.createElement('style');
  s.id = 'axium-expand-styles';
  s.textContent = `

/* ── Expand button on every card ──────────────────────── */
.axc-expand-btn {
  position: absolute;
  top: 4px; right: 4px;
  z-index: 30;
  width: 22px; height: 22px;
  border-radius: 4px;
  background: rgba(0,0,0,.55);
  border: 1px solid rgba(255,255,255,.12);
  display: flex; align-items: center; justify-content: center;
  cursor: pointer;
  /* Always visible — opacity controlled by touch vs pointer media */
  opacity: 0;
  transition: opacity .2s, background .2s, border-color .2s, transform .15s;
  font-size: 9px;
  color: rgba(255,255,255,.55);
  pointer-events: auto;
  line-height: 1;
  font-family: 'Space Mono', monospace;
  letter-spacing: 0;
  /* Prevent the button tap from selecting text on the card */
  -webkit-user-select: none;
  user-select: none;
  -webkit-tap-highlight-color: transparent;
  touch-action: manipulation;
}

/* Pointer devices: reveal on card hover */
@media (hover: hover) and (pointer: fine) {
  .axc-card:hover .axc-expand-btn,
  .axc-card:focus-within .axc-expand-btn {
    opacity: 1;
  }
  .axc-expand-btn:hover {
    background: rgba(212,175,55,.22);
    border-color: rgba(212,175,55,.55);
    color: #D4AF37;
    transform: scale(1.12);
  }
}

/* Touch devices: always visible, bigger tap target */
@media (hover: none), (pointer: coarse) {
  .axc-expand-btn {
    opacity: 1;
    width: 28px; height: 28px;
    font-size: 11px;
    background: rgba(0,0,0,.65);
    border-color: rgba(255,255,255,.18);
  }
  .axc-expand-btn:active {
    background: rgba(212,175,55,.25);
    border-color: rgba(212,175,55,.6);
    color: #D4AF37;
    transform: scale(0.93);
  }
}

/* Suppress text selection on the whole card on touch */
.axc-card {
  -webkit-user-select: none;
  user-select: none;
  -webkit-touch-callout: none;
}

/* ── Full-screen expand overlay ───────────────────────── */
#axc-expand-modal {
  position: fixed;
  inset: 0;
  z-index: 400;
  background: rgba(2,2,10,.0);
  display: flex;
  align-items: center;
  justify-content: center;
  pointer-events: none;
  transition: background .38s cubic-bezier(.22,1,.36,1);
  padding: clamp(12px,3vw,28px);
  box-sizing: border-box;
}
#axc-expand-modal.open {
  background: rgba(2,2,10,.93);
  pointer-events: auto;
}
#axc-expand-modal .axc-exp-backdrop {
  position: absolute;
  inset: 0;
  cursor: pointer;
}

.axc-exp-panel {
  position: relative;
  z-index: 1;
  width: 100%;
  max-width: 520px;
  max-height: 92vh;
  border-radius: 16px;
  background: linear-gradient(168deg, rgba(16,12,34,.99) 0%, rgba(6,4,18,.99) 55%, rgba(12,9,26,.99) 100%);
  border: 1px solid rgba(255,255,255,.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  opacity: 0;
  transform: scale(.88) translateY(24px);
  transition: opacity .42s cubic-bezier(.22,1,.36,1), transform .42s cubic-bezier(.22,1,.36,1);
  box-shadow: 0 32px 80px rgba(0,0,0,.8), 0 0 60px rgba(0,0,0,.5);
}
#axc-expand-modal.open .axc-exp-panel {
  opacity: 1;
  transform: scale(1) translateY(0);
}

/* Color accent line at top */
.axc-exp-accent {
  height: 2px;
  width: 100%;
  flex-shrink: 0;
}

/* Header */
.axc-exp-header {
  display: flex;
  align-items: flex-start;
  gap: 14px;
  padding: 18px 20px 14px;
  border-bottom: 1px solid rgba(255,255,255,.06);
  flex-shrink: 0;
  position: relative;
}
.axc-exp-constellation {
  width: 88px;
  height: 88px;
  flex-shrink: 0;
  border-radius: 10px;
  background: rgba(0,0,0,.35);
  border: 1px solid rgba(255,255,255,.07);
  overflow: hidden;
}
.axc-exp-constellation canvas {
  width: 100%;
  height: 100%;
  display: block;
}
.axc-exp-title-group {
  flex: 1;
  min-width: 0;
  padding-top: 2px;
}
.axc-exp-layer-lbl {
  font-family: 'Space Mono', monospace;
  font-size: 7.5px;
  letter-spacing: .22em;
  text-transform: uppercase;
  opacity: .4;
  margin-bottom: 5px;
  display: block;
}
.axc-exp-name {
  font-family: 'Cinzel Decorative', serif;
  font-size: clamp(16px, 4vw, 24px);
  font-weight: 700;
  letter-spacing: .04em;
  line-height: 1.1;
  display: block;
  margin-bottom: 5px;
}
.axc-exp-keywords {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic;
  font-size: 13px;
  color: rgba(255,255,255,.35);
  line-height: 1.5;
  display: block;
  margin-bottom: 8px;
}
.axc-exp-badges {
  display: flex;
  gap: 6px;
  flex-wrap: wrap;
  align-items: center;
}
.axc-exp-type-pip {
  font-family: 'Space Mono', monospace;
  font-size: 6px; letter-spacing: .08em; text-transform: uppercase;
  padding: 2px 6px; border-radius: 3px; border: 1px solid;
}
.axc-exp-type-pip.compression   { color: rgba(220,100,60,.9); border-color: rgba(220,100,60,.4); }
.axc-exp-type-pip.decompression { color: rgba(126,184,232,.9); border-color: rgba(126,184,232,.4); }
.axc-exp-type-pip.both          { color: rgba(212,175,55,.9);  border-color: rgba(212,175,55,.4); }
.axc-exp-axium-badge {
  font-family: 'JetBrains Mono', monospace;
  font-size: 10px; font-weight: 600;
  padding: 2px 7px; border-radius: 3px;
  background: rgba(255,255,255,.04);
  border: 1px solid rgba(255,255,255,.08);
  color: rgba(255,255,255,.45);
}
.axc-exp-tier-badge {
  font-family: 'Space Mono', monospace;
  font-size: 6px; letter-spacing: .12em; text-transform: uppercase;
  padding: 2px 6px; border-radius: 3px;
  border: 1px solid rgba(255,255,255,.08);
  color: rgba(255,255,255,.22);
}
.axc-exp-close {
  position: absolute;
  top: 14px; right: 16px;
  width: 28px; height: 28px;
  border-radius: 50%;
  background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.1);
  color: rgba(255,255,255,.4);
  font-size: 14px; line-height: 1;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all .2s;
  font-family: 'Space Mono', monospace;
}
.axc-exp-close:hover {
  background: rgba(224,85,85,.15);
  border-color: rgba(224,85,85,.4);
  color: #e05555;
}

/* Body scroll area */
.axc-exp-body {
  flex: 1;
  overflow-y: auto;
  padding: 16px 20px 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  scrollbar-width: thin;
  scrollbar-color: rgba(212,175,55,.2) transparent;
}
.axc-exp-body::-webkit-scrollbar { width: 3px; }
.axc-exp-body::-webkit-scrollbar-thumb { background: rgba(212,175,55,.2); border-radius: 2px; }

/* Section label */
.axc-exp-section-lbl {
  font-family: 'Space Mono', monospace;
  font-size: 7px; letter-spacing: .24em; text-transform: uppercase;
  color: rgba(255,255,255,.18);
  margin-bottom: 6px;
  display: flex; align-items: center; gap: 8px;
}
.axc-exp-section-lbl::after {
  content: '';
  flex: 1;
  height: 1px;
  background: rgba(255,255,255,.05);
}

/* Stats grid */
.axc-exp-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
  gap: 7px;
}
.axc-exp-stat {
  padding: 8px 10px;
  border-radius: 6px;
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.06);
}
.axc-exp-stat-val {
  font-family: 'JetBrains Mono', monospace;
  font-size: 15px; font-weight: 600;
  line-height: 1.1;
  margin-bottom: 3px;
}
.axc-exp-stat-lbl {
  font-family: 'Space Mono', monospace;
  font-size: 7px; letter-spacing: .14em; text-transform: uppercase;
  color: rgba(255,255,255,.22);
}

/* Effect block */
.axc-exp-effect-block {
  padding: 12px 14px;
  border-radius: 8px;
  background: rgba(255,255,255,.025);
  border: 1px solid rgba(255,255,255,.06);
  border-left: 2px solid;
}
.axc-exp-effect-title {
  font-family: 'Space Mono', monospace;
  font-size: 7px; letter-spacing: .18em; text-transform: uppercase;
  opacity: .45; margin-bottom: 6px; display: block;
}
.axc-exp-effect-text {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic;
  font-size: clamp(12px, 2.5vw, 14px);
  line-height: 1.85;
  color: rgba(255,255,255,.68);
  white-space: pre-line;
}
.axc-exp-effect-block.reversed-block {
  background: rgba(224,85,85,.04);
  border-left-color: rgba(224,85,85,.5) !important;
}
.axc-exp-effect-block.reversed-block .axc-exp-effect-text {
  color: rgba(224,120,120,.7);
}

/* Synergies */
.axc-exp-synergy {
  padding: 10px 12px;
  border-radius: 7px;
  background: rgba(255,255,255,.025);
  border: 1px solid rgba(255,255,255,.07);
  cursor: default;
  transition: border-color .2s;
}
.axc-exp-synergy.syn-active {
  background: rgba(212,175,55,.05);
  border-color: rgba(212,175,55,.3);
}
.axc-exp-syn-header {
  display: flex;
  align-items: center;
  gap: 7px;
  margin-bottom: 4px;
}
.axc-exp-syn-name {
  font-family: 'Cinzel', serif;
  font-size: 11px; font-weight: 600; letter-spacing: .06em;
}
.axc-exp-syn-cards {
  font-family: 'Space Mono', monospace;
  font-size: 7px; letter-spacing: .1em; text-transform: uppercase;
  color: rgba(255,255,255,.25);
  margin-bottom: 3px;
}
.axc-exp-syn-desc {
  font-family: 'Cormorant Garamond', serif;
  font-style: italic;
  font-size: 12px; line-height: 1.7;
  color: rgba(255,255,255,.42);
}
.axc-exp-syn-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.axc-exp-syn-badge {
  font-family: 'Space Mono', monospace;
  font-size: 6px; letter-spacing: .1em; text-transform: uppercase;
  padding: 1px 6px; border-radius: 10px;
  border: 1px solid rgba(212,175,55,.4);
  color: rgba(212,175,55,.8);
  background: rgba(212,175,55,.07);
  margin-left: auto; flex-shrink: 0;
}

/* Rare badge */
.axc-exp-rare-badge {
  font-family: 'Cinzel', serif;
  font-size: 7px; letter-spacing: .18em; text-transform: uppercase;
  padding: 2px 8px; border-radius: 10px;
  background: linear-gradient(135deg, rgba(212,175,55,.15), rgba(255,220,80,.08));
  border: 1px solid rgba(212,175,55,.4);
  color: #D4AF37;
}

/* Layer mechanic description block */
.axc-exp-mechanic-overview {
  padding: 10px 13px;
  border-radius: 7px;
  background: rgba(255,255,255,.02);
  border: 1px solid rgba(255,255,255,.05);
}
.axc-exp-mechanic-overview p {
  font-family: 'Cormorant Garamond', serif;
  font-size: 13px; line-height: 1.8;
  color: rgba(255,255,255,.35);
  margin: 0;
}

/* Chapter & availability */
.axc-exp-meta-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.axc-exp-meta-chip {
  font-family: 'Space Mono', monospace;
  font-size: 7px; letter-spacing: .14em; text-transform: uppercase;
  padding: 3px 9px; border-radius: 3px;
  background: rgba(255,255,255,.03);
  border: 1px solid rgba(255,255,255,.07);
  color: rgba(255,255,255,.28);
}

`;
  document.head.appendChild(s);
})();

// ─────────────────────────────────────────────────────────────
// MODAL ELEMENT
// ─────────────────────────────────────────────────────────────
let _modal = null;
let _constAnim = null;

function _getModal() {
  if (_modal) return _modal;
  const el = document.createElement('div');
  el.id = 'axc-expand-modal';
  el.innerHTML = `
    <div class="axc-exp-backdrop"></div>
    <div class="axc-exp-panel">
      <div class="axc-exp-accent" id="axce-accent"></div>
      <div class="axc-exp-header">
        <div class="axc-exp-constellation" id="axce-constellation">
          <canvas id="axce-cvs" width="88" height="88"></canvas>
        </div>
        <div class="axc-exp-title-group">
          <span class="axc-exp-layer-lbl" id="axce-layer"></span>
          <span class="axc-exp-name"      id="axce-name"></span>
          <span class="axc-exp-keywords"  id="axce-keywords"></span>
          <div class="axc-exp-badges"     id="axce-badges"></div>
        </div>
        <button class="axc-exp-close" id="axce-close">×</button>
      </div>
      <div class="axc-exp-body" id="axce-body"></div>
    </div>
  `;
  document.body.appendChild(el);
  el.querySelector('.axc-exp-backdrop').addEventListener('click', closeExpandModal);
  el.querySelector('#axce-close').addEventListener('click', closeExpandModal);
  // ESC to close
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeExpandModal(); });
  _modal = el;
  return el;
}

function closeExpandModal() {
  const el = _modal; if (!el) return;
  el.classList.remove('open');
  if (_constAnim) { _constAnim.stop(); _constAnim = null; }
}

// ─────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────
function _hexRGB(hex) {
  const h = hex.replace('#','');
  if (h.length === 3) return [parseInt(h[0]+h[0],16),parseInt(h[1]+h[1],16),parseInt(h[2]+h[2],16)];
  return [parseInt(h.slice(0,2),16),parseInt(h.slice(2,4),16),parseInt(h.slice(4,6),16)];
}

function _layerLabel(card) {
  const romanMap = {0:'0',1:'I',2:'II',3:'III',4:'IV',5:'V',6:'VI',7:'VII',8:'VIII',9:'IX',
    10:'X',11:'XI',12:'XII',13:'XIII',14:'XIV',15:'XV',16:'XVI',17:'XVII',18:'XVIII',19:'XIX',20:'XX',21:'XXI'};
  if (card.layer === 'Superego') {
    const n = card.number !== undefined ? romanMap[card.number] || String(card.number) : '';
    return n ? `Superego · ${n}` : 'Superego';
  }
  if (card.layer === 'Ego') return `Ego · ${card.suit ? card.suit[0].toUpperCase()+card.suit.slice(1) : ''}`;
  if (card.layer === 'ID')  return `ID · ${card.suit ? card.suit[0].toUpperCase()+card.suit.slice(1) : ''}`;
  return card.layer || 'Card';
}

function _layerColor(card) {
  if (card.layer === 'Superego') return '#D4AF37';
  if (card.layer === 'Ego')      return '#7EB8E8';
  if (card.layer === 'ID')       return '#86EFAC';
  return '#ffffff';
}

function _mechOverview(card) {
  if (card.layer === 'Superego') return 'Superego cards are your Moral Center. Upright: SHIELD — a flat attention boost. While in deck: CAPACITY — alters your max attention pool.';
  if (card.layer === 'Ego')      return 'Ego cards are the Rational Mediator. Upright: CHUNK — multiplies your attention gains. Reversed: DIVIDE — applies a divisor, representing misinformation or conspiracy.';
  if (card.layer === 'ID')       return 'ID cards are the Unconscious. Upright: RECHARGE — stackable passive attention gain. Reversed (trauma-played): DRAIN — stackable passive attention loss.';
  return '';
}

// ─────────────────────────────────────────────────────────────
// OPEN EXPAND MODAL
// ─────────────────────────────────────────────────────────────
function openExpandModal(card) {
  const el     = _getModal();
  const [r,g,b]= _hexRGB(card.color || '#ffffff');
  const lc     = _layerColor(card);

  // Accent line
  const accent = el.querySelector('#axce-accent');
  accent.style.background = `linear-gradient(90deg, ${card.color}, rgba(${r},${g},${b},.15), transparent)`;

  // Header
  const layerEl = el.querySelector('#axce-layer');
  layerEl.textContent = _layerLabel(card);
  layerEl.style.color = lc;

  const nameEl = el.querySelector('#axce-name');
  nameEl.textContent = card.name;
  nameEl.style.color  = card.color;

  el.querySelector('#axce-keywords').textContent = card.keywords || '';

  // Badges
  const badgesEl = el.querySelector('#axce-badges');
  const type = card.type || 'compression';
  badgesEl.innerHTML = `
    <span class="axc-exp-type-pip ${type}">${type}</span>
    <span class="axc-exp-axium-badge" style="color:rgba(${r},${g},${b},.7)">⬡ ${card.axiumScore ?? '?'} / 10</span>
    ${card.tier ? `<span class="axc-exp-tier-badge">Tier ${card.tier}</span>` : ''}
    ${(card.axiumScore === 10) ? `<span class="axc-exp-rare-badge">✦ Perfect</span>` : ''}
  `;

  // Constellation canvas
  const cvs = el.querySelector('#axce-cvs');
  if (_constAnim) { _constAnim.stop(); _constAnim = null; }
  cvs.width = 88; cvs.height = 88;
  if (card.pts && typeof ConstellationAnim !== 'undefined') {
    requestAnimationFrame(() => {
      _constAnim = new ConstellationAnim(cvs, card, { speed:0.018, nodeSize:2.2, edgeAlpha:0.38, glowMult:3.5 });
      _constAnim.start();
    });
  }

  // Body
  const body = el.querySelector('#axce-body');
  body.innerHTML = '';

  // ── Stats ──
  const statItems = _buildStats(card, r, g, b, lc);
  if (statItems.length) {
    body.appendChild(_section('Core Stats'));
    const statsGrid = document.createElement('div');
    statsGrid.className = 'axc-exp-stats';
    statItems.forEach(s => {
      const d = document.createElement('div');
      d.className = 'axc-exp-stat';
      d.innerHTML = `<div class="axc-exp-stat-val" style="color:${s.color||'rgba(255,255,255,.65)'}">${s.val}</div><div class="axc-exp-stat-lbl">${s.lbl}</div>`;
      statsGrid.appendChild(d);
    });
    body.appendChild(statsGrid);
  }

  // ── Layer mechanic overview ──
  const mechOv = _mechOverview(card);
  if (mechOv) {
    const ov = document.createElement('div');
    ov.className = 'axc-exp-mechanic-overview';
    ov.innerHTML = `<p>${mechOv}</p>`;
    body.appendChild(ov);
  }

  // ── Upright effect ──
  const uprightText = _uprightText(card);
  if (uprightText) {
    body.appendChild(_section('Upright Effect'));
    const block = document.createElement('div');
    block.className = 'axc-exp-effect-block';
    block.style.borderLeftColor = `rgba(${r},${g},${b},.7)`;
    block.innerHTML = `<span class="axc-exp-effect-title">${_uprightLabel(card)}</span><div class="axc-exp-effect-text">${uprightText}</div>`;
    body.appendChild(block);
  }

  // ── Reversed effect ──
  const revText = _reversedText(card);
  if (revText) {
    body.appendChild(_section('Reversed'));
    const block = document.createElement('div');
    block.className = 'axc-exp-effect-block reversed-block';
    block.innerHTML = `<span class="axc-exp-effect-title" style="color:rgba(224,85,85,.6)">↓ Reversed Mechanic</span><div class="axc-exp-effect-text">${revText}</div>`;
    body.appendChild(block);
  }

  // ── Synergies ──
  const allSyns = (typeof SYNERGIES !== 'undefined') ? SYNERGIES.filter(s => s.cards.includes(card.id)) : [];
  if (allSyns.length) {
    body.appendChild(_section(`Synergies (${allSyns.length})`));
    allSyns.forEach(syn => {
      const synEl = document.createElement('div');
      synEl.className = 'axc-exp-synergy' + (syn.rare ? ' syn-active' : '');
      const dotCol = syn.visual || '#D4AF37';
      const partnerNames = syn.cards.filter(id => id !== card.id).map(id => {
        const c = (typeof getCard === 'function') ? getCard(id) : null;
        return c ? c.name : id.replace(/_/g,' ');
      });
      synEl.innerHTML = `
        <div class="axc-exp-syn-header">
          <div class="axc-exp-syn-dot" style="background:${dotCol};box-shadow:0 0 6px ${dotCol}"></div>
          <span class="axc-exp-syn-name" style="color:${dotCol}">${syn.name}</span>
          ${syn.rare ? '<span class="axc-exp-rare-badge">✦ Rare</span>' : ''}
          <span class="axc-exp-syn-badge">Synergy</span>
        </div>
        <div class="axc-exp-syn-cards">Requires: ${syn.cards.map(id => {
          const c = (typeof getCard === 'function') ? getCard(id) : null;
          return c ? c.name : id.replace(/_/g,' ');
        }).join(' · ')}</div>
        <div class="axc-exp-syn-desc">${syn.desc}</div>
      `;
      body.appendChild(synEl);
    });
  }

  // ── Availability ──
  const metaChips = _buildMeta(card);
  if (metaChips.length) {
    body.appendChild(_section('Availability'));
    const metaRow = document.createElement('div');
    metaRow.className = 'axc-exp-meta-row';
    metaChips.forEach(txt => {
      const chip = document.createElement('span');
      chip.className = 'axc-exp-meta-chip';
      chip.textContent = txt;
      metaRow.appendChild(chip);
    });
    body.appendChild(metaRow);
  }

  // Open
  el.classList.add('open');
  body.scrollTop = 0;
}

// ─────────────────────────────────────────────────────────────
// CONTENT HELPERS
// ─────────────────────────────────────────────────────────────
function _section(label) {
  const d = document.createElement('div');
  d.className = 'axc-exp-section-lbl';
  d.textContent = label;
  return d;
}

function _buildStats(card, r, g, b, lc) {
  const stats = [];
  const col = card.color || '#fff';

  if (card.layer === 'Superego') {
    if (card.shieldVal)    stats.push({ lbl:'Shield Boost',  val:`+${card.shieldVal}`,  color:'#D4AF37' });
    if (card.capacityVal != null) stats.push({ lbl:'Capacity Mod', val:`${card.capacityVal>=0?'+':''}${card.capacityVal}`, color: card.capacityVal >= 0 ? '#86EFAC' : '#e05555' });
    if (card.rechargeVal)  stats.push({ lbl:'Passive Hold',  val:`+${card.rechargeVal}`, color:'rgba(255,255,255,.55)' });
  }
  if (card.layer === 'Ego') {
    if (card.chunkFlat)    stats.push({ lbl:'Flat Bonus',   val:`+${card.chunkFlat}`,  color:'#7EB8E8' });
    if (card.chunkPct)     stats.push({ lbl:'Multiplier',   val:`×${card.chunkPct}`,   color:'#7EB8E8' });
    if (card.studyMult)    stats.push({ lbl:'Study Amp',    val:`×${card.studyMult}`,  color:'#D4AF37' });
  }
  if (card.layer === 'ID') {
    if (card.rechargeVal)  stats.push({ lbl:'Recharge/Stack', val:`+${card.rechargeVal}`, color:'#86EFAC' });
    if (card.drainVal)     stats.push({ lbl:'Drain/Stack',    val:`−${card.drainVal}`,    color:'#e05555' });
    if (card.traumaHealing) stats.push({ lbl:'Trauma Heal',   val:`+${card.traumaHealing}`, color:'rgba(126,184,232,.8)' });
  }
  if (card.axiumScore != null) stats.push({ lbl:'Axium Score', val:`${card.axiumScore} / 10`, color:`rgba(${r},${g},${b},.75)` });
  return stats;
}

function _uprightLabel(card) {
  if (card.layer === 'Superego') return '◈ Shield / Capacity';
  if (card.layer === 'Ego')      return '◇ Chunk Multiplier';
  if (card.layer === 'ID')       return '○ Recharge';
  return '↑ Upright';
}

function _uprightText(card) {
  if (card.layer === 'Superego') {
    const parts = [];
    if (card.shieldDesc)   parts.push(card.shieldDesc);
    if (card.capacityDesc) parts.push(card.capacityDesc);
    if (card.rechargeDesc) parts.push(card.rechargeDesc);
    return parts.join('\n\n');
  }
  if (card.layer === 'Ego')  return card.chunkDesc   || '';
  if (card.layer === 'ID')   return card.rechargeDesc || '';
  return card.effectDesc || '';
}

function _reversedText(card) {
  if (card.layer === 'Superego') return card.reversedDesc || '';
  if (card.layer === 'Ego')      return card.divideDesc   || '';
  if (card.layer === 'ID')       return card.drainDesc    || '';
  return '';
}

function _buildMeta(card) {
  const chips = [];
  if (card.chapter)     chips.push(`Chapter ${card.chapter}`);
  if (card.tier)        chips.push(`Tier ${card.tier}`);
  if (card.shopChapter) chips.push(`Shop: Chapter ${card.shopChapter}`);
  if (card.suit)        chips.push(`Suit: ${card.suit[0].toUpperCase()+card.suit.slice(1)}`);
  if (card.layer)       chips.push(`Layer: ${card.layer}`);
  return chips;
}

// ─────────────────────────────────────────────────────────────
// ADD EXPAND BUTTON TO EXISTING + FUTURE CARDS
// Uses MutationObserver to catch dynamically-created cards
// ─────────────────────────────────────────────────────────────
function _addExpandBtn(cardEl) {
  if (cardEl.querySelector('.axc-expand-btn')) return; // already done
  const cardId = cardEl.dataset.cardId || cardEl.dataset.id;
  if (!cardId) return;

  const btn = document.createElement('button');
  btn.className = 'axc-expand-btn';
  btn.title = 'Expand card details';
  btn.innerHTML = '⊞';
  btn.setAttribute('aria-label', 'Expand card details');
  // Prevent the button's own touch from propagating to card drag/select
  btn.setAttribute('type', 'button');

  // Use pointerdown + pointerup tracking so a short tap opens the modal
  // but a scroll-drag on the card does NOT trigger it.
  let _pdX = 0, _pdY = 0;
  btn.addEventListener('pointerdown', e => {
    _pdX = e.clientX; _pdY = e.clientY;
    e.stopPropagation();
  }, { passive: true });

  btn.addEventListener('pointerup', e => {
    e.stopPropagation();
    const dx = Math.abs(e.clientX - _pdX);
    const dy = Math.abs(e.clientY - _pdY);
    if (dx < 8 && dy < 8) {
      // It's a tap, not a drag
      e.preventDefault();
      const card = (typeof getCard === 'function') ? getCard(cardId) : null;
      if (card) openExpandModal(card);
    }
  });

  // Also handle plain click for mouse users (already covered by pointerup,
  // but this catches keyboard activation via Enter/Space)
  btn.addEventListener('click', e => {
    e.stopPropagation();
    e.preventDefault();
  });

  cardEl.appendChild(btn);
}

function _scanAndAddBtns() {
  document.querySelectorAll('.axc-card').forEach(_addExpandBtn);
}

// Scan on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', _scanAndAddBtns);
} else {
  _scanAndAddBtns();
}

// Watch for new cards
if (typeof MutationObserver !== 'undefined') {
  const obs = new MutationObserver(mutations => {
    mutations.forEach(m => {
      m.addedNodes.forEach(node => {
        if (!(node instanceof Element)) return;
        if (node.classList.contains('axc-card')) {
          _addExpandBtn(node);
        } else {
          node.querySelectorAll?.('.axc-card').forEach(_addExpandBtn);
        }
      });
    });
  });
  obs.observe(document.body, { childList: true, subtree: true });
}

// ─────────────────────────────────────────────────────────────
// EXPOSE GLOBALLY
// ─────────────────────────────────────────────────────────────
global.openExpandModal  = openExpandModal;
global.closeExpandModal = closeExpandModal;

// Also hook into AxiumCardDetail.open so long-press/right-click
// on existing cards still routes through the richer panel
if (global.AxiumCardDetail) {
  const _origOpen = global.AxiumCardDetail.open.bind(global.AxiumCardDetail);
  global.AxiumCardDetail.open = function(card, played, opts) {
    openExpandModal(card);
  };
}

})(window);
