/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — UI.js  v2.0  (Overlay Navigation Architecture)
 * Depends on: cards.js (ATTN_STATES, getAttnState, getSynergies)
 * Must load BEFORE shop.js and battle.js
 *
 * KEY CHANGE v2.0:
 *   goTo() no longer toggles .active on .screen elements.
 *   Instead it toggles .axm-active on .axm-overlay elements.
 *   #screen-battle is permanently visible as the base layer.
 *   All other screens are position:fixed overlays.
 * ═══════════════════════════════════════════════════════════════
 */

'use strict';

// ─────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────
const MAX_STAGED = 10;
const MIN_ATTN   = 5;

// ─────────────────────────────────────────────────────────
// DOM HELPERS
// ─────────────────────────────────────────────────────────
function gel(id)        { return document.getElementById(id); }
function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t)  { return a + (b - a) * t; }
function hexA(val)       { return Math.max(0, Math.min(255, Math.round(val))).toString(16).padStart(2, '0'); }
function cap(s)          { return s ? s[0].toUpperCase() + s.slice(1) : ''; }
function numberToRoman(n) {
  const map = [[21,'XXI'],[20,'XX'],[19,'XIX'],[18,'XVIII'],[17,'XVII'],[16,'XVI'],[15,'XV'],
               [14,'XIV'],[13,'XIII'],[12,'XII'],[11,'XI'],[10,'X'],[9,'IX'],[8,'VIII'],
               [7,'VII'],[6,'VI'],[5,'V'],[4,'IV'],[3,'III'],[2,'II'],[1,'I'],[0,'0']];
  return (map.find(([v]) => v === n) || [n, String(n)])[1];
}
function cardLabel(card) {
  const layer = card.layer || '';
  if (layer === 'Superego') {
    const num = card.number !== undefined ? numberToRoman(card.number) : '';
    return num ? `Superego · ${num}` : 'Superego';
  }
  if (layer === 'Ego') return `Ego · ${cap(card.suit || '')}`;
  if (layer === 'ID')  return `ID · ${cap(card.suit || '')}`;
  return layer || 'Card';
}

// ─────────────────────────────────────────────────────────
// OVERLAY NAVIGATION  (replaces old screen .active switching)
//
// Overlay IDs that goTo() can manage:
//   screen-intro, screen-seed-load, screen-shop,
//   screen-deck-review, screen-outcome
//   (screen-battle is always visible — never in this list)
//   (screen-hand-picker is managed separately by picker functions)
// ─────────────────────────────────────────────────────────
const _MANAGED_OVERLAYS = [
  'screen-intro',
  'screen-seed-load',
  'screen-shop',
  'screen-deck-review',
  'screen-outcome',
];

function _overlayOpen(id) {
  const el = gel(id);
  if (!el) return;
  el.classList.add('axm-active');
}

function _overlayClose(id) {
  const el = gel(id);
  if (!el) return;
  el.classList.remove('axm-active');
}

function _closeAllManaged() {
  _MANAGED_OVERLAYS.forEach(_overlayClose);
}

/**
 * goTo(screenId, afterFade?)
 * Drop-in replacement for the old goTo().
 * - 'screen-battle' → closes all overlays (battle shell is always visible)
 * - anything else   → closes all overlays, opens the target
 * afterFade callback fires after the transition delay.
 */
function goTo(screenId, afterFade) {
  _closeAllManaged();

  if (screenId !== 'screen-battle') {
    // Small rAF delay so the close transition starts before open
    requestAnimationFrame(() => {
      _overlayOpen(screenId);
    });
  }

  // Match old goTo() timing (450ms fade) so callers' afterFade works correctly
  setTimeout(() => afterFade?.(), 60);
}

// ─────────────────────────────────────────────────────────
// STARFIELD BACKGROUND
// ─────────────────────────────────────────────────────────
(function initStars() {
  const cv = gel('stars'); if (!cv) return;
  const ctx = cv.getContext('2d');
  let W, H, stars = [];
  function resize() {
    W = cv.width  = window.innerWidth;
    H = cv.height = window.innerHeight;
    stars = Array.from({ length: 140 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      r: .3 + Math.random() * 1.1,
      a: .07 + Math.random() * .36,
      sp: .22 + Math.random() * .55,
      ph: Math.random() * Math.PI * 2,
    }));
  }
  function draw(t) {
    requestAnimationFrame(draw);
    ctx.fillStyle = '#02020a'; ctx.fillRect(0, 0, W, H);
    const g = ctx.createRadialGradient(W * .4, H * .35, 0, W * .4, H * .35, W * .5);
    g.addColorStop(0, 'rgba(40,20,70,.13)'); g.addColorStop(1, 'rgba(2,2,10,0)');
    ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
    stars.forEach(s => {
      const tw = .4 + .6 * Math.abs(Math.sin(t * .0008 * s.sp + s.ph));
      ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${s.a * tw})`; ctx.fill();
    });
  }
  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(draw);
})();

// ─────────────────────────────────────────────────────────
// TOAST
// ─────────────────────────────────────────────────────────
let _toastTimer;
function toast(heading, body) {
  gel('toast-h').textContent = heading;
  gel('toast-b').textContent = body;
  const t = gel('toast');
  t.classList.add('show');
  clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

// ─────────────────────────────────────────────────────────
// BURST PARTICLES
// ─────────────────────────────────────────────────────────
function burst(x, y, color, n) {
  for (let i = 0; i < n; i++) {
    const p = document.createElement('div');
    p.className = 'ptcl';
    const ang  = (Math.PI * 2 * i) / n + Math.random() * .5;
    const dist = 20 + Math.random() * 65;
    const dur  = 380 + Math.random() * 260;
    p.style.cssText = `left:${x}px;top:${y}px;width:${1.4 + Math.random() * 2.8}px;height:${1.4 + Math.random() * 2.8}px;background:${color};box-shadow:0 0 5px ${color};transition:transform ${dur}ms cubic-bezier(.22,1,.36,1),opacity ${dur}ms ease;position:fixed;border-radius:50%;pointer-events:none;z-index:900;`;
    document.body.appendChild(p);
    requestAnimationFrame(() => {
      p.style.transform = `translate(${Math.cos(ang) * dist}px,${Math.sin(ang) * dist}px) scale(0)`;
      p.style.opacity = '0';
    });
    setTimeout(() => p.remove(), dur);
  }
}

// ─────────────────────────────────────────────────────────
// SYNERGY FLASH
// ─────────────────────────────────────────────────────────
function flashSynergy(syn) {
  const flash = gel('synergy-flash'), msg = gel('synergy-msg');
  if (!flash || !msg) return;
  flash.style.background = `radial-gradient(ellipse at center,${syn.visual || '#D4AF37'}22 0%,transparent 70%)`;
  gel('syn-name').textContent = syn.name;
  gel('syn-name').style.color = syn.visual || '#D4AF37';
  gel('syn-desc').textContent = syn.desc || '';
  flash.classList.add('show'); msg.classList.add('show');
  setTimeout(() => { flash.classList.remove('show'); msg.classList.remove('show'); }, 2100);
}

// ─────────────────────────────────────────────────────────
// ATTENTION BARS
// ─────────────────────────────────────────────────────────
function buildPips(id) {
  const c = gel(id); if (!c) return; c.innerHTML = '';
  ATTN_STATES.forEach(s => {
    const p = document.createElement('div');
    p.className = 'b-bar-pip';
    p.style.left = (s.pos * 100) + '%';
    p.style.setProperty('--pip-col', s.col);
    c.appendChild(p);
  });
}

function updatePips(id, pct) {
  document.querySelectorAll(`#${id} .b-bar-pip`).forEach((p, i) => {
    p.classList.toggle('lit', ATTN_STATES[i] && ATTN_STATES[i].pos <= pct / 100);
  });
}

function setBar(which, pct, state) {
  const fill = gel(which + '-fill');      if (fill) fill.style.width    = pct + '%';
  const cur  = gel(which + '-cursor');   if (cur)  cur.style.left      = pct + '%';
  const lbl  = gel(which + '-state-lbl'); if (lbl) { lbl.textContent = state.label; lbl.style.color = state.col; }
}

function animateBarTo(which, newAttn, maxAttn) {
  const pct = clamp(newAttn / maxAttn * 100, 0, 100);
  setBar(which, pct, getAttnState(pct));
  updatePips(which + '-pips', pct);
}

// ─────────────────────────────────────────────────────────
// AXIUM METER
// ─────────────────────────────────────────────────────────
function updateAxiumMeter(playerPlayed) {
  const n   = playerPlayed.length;
  const avg = n ? playerPlayed.reduce((s, e) => s + ((e.card || e).axiumScore || 0), 0) / n : 0;
  const lit = Math.round(avg);
  for (let i = 0; i < 10; i++) {
    const p = gel(`axp-${i}`);
    if (p) p.classList.toggle('lit', i < lit);
  }
}

// ─────────────────────────────────────────────────────────
// BATTLE FIELD RENDER
// ─────────────────────────────────────────────────────────
function renderBattleField(B, onUnstage) {
  const pf = gel('field-player');
  if (pf) {
    pf.innerHTML = '';
    if (!B.playerPlayed.length) {
      pf.innerHTML = '<div class="field-empty">— no cards staged —</div>';
    } else {
      B.playerPlayed.forEach((entry, i) => {
        const { card, reversed } = entry;
        const chip = document.createElement('div');
        chip.className = 'f-chip' + (reversed ? ' reversed-chip' : '');
        chip.style.borderColor = card.color + (reversed ? '66' : '55');
        chip.style.color = card.color;
        const icon = card.layer === 'Superego' ? '◈' : card.layer === 'Ego' ? '◇' : '○';
        chip.innerHTML = `<span>${icon}</span><span>${card.name}</span>`;
        const polBtn = document.createElement('span');
        polBtn.className = 'f-chip-polarity ' + (reversed ? 'reversed' : 'normal');
        polBtn.textContent = reversed ? '↓' : '↑';
        polBtn.title = reversed ? 'Reversed' : 'Normal';
        polBtn.onclick = e => {
          e.stopPropagation();
          entry.reversed = !entry.reversed;
          renderBattleField(B, onUnstage);
          updateBattlePhaseUI(B);
        };
        chip.appendChild(polBtn);
        const rm = document.createElement('span');
        rm.className = 'f-chip-remove'; rm.textContent = '×';
        rm.onclick = e => { e.stopPropagation(); onUnstage(i); };
        chip.appendChild(rm);
        pf.appendChild(chip);
      });
    }
  }

  const ef = gel('field-enemy');
  if (ef) {
    ef.innerHTML = '';
    if (!B.enemyHand.length) {
      ef.innerHTML = '<div class="field-empty field-enemy-empty">— enemy awaiting —</div>';
    } else {
      B.enemyHand.forEach(({ card, drain }) => {
        const chip = document.createElement('div');
        chip.className = 'f-chip enemy';
        const icon = card.layer === 'Superego' ? '◈' : card.layer === 'Ego' ? '◇' : '○';
        chip.innerHTML = `<span>${icon}</span><span>${card.name}</span><span style="font-family:'JetBrains Mono',monospace;font-size:7px;opacity:.55;margin-left:4px;">−${drain || ''}</span>`;
        ef.appendChild(chip);
      });
    }
  }
  updateBattlePhaseUI(B);
}

// ─────────────────────────────────────────────────────────
// BATTLE PHASE UI
// ─────────────────────────────────────────────────────────
function updateBattlePhaseUI(B) {
  const n  = B.playerPlayed.length;
  const rb = gel('resolve-btn');
  if (rb) rb.classList.toggle('show', n > 0 && B.phase === 'build');
  const pb = gel('pass-btn');
  if (pb) pb.disabled = B.phase !== 'build';
  const pm = gel('b-phase-msg');
  if (pm) pm.textContent = n === 0
    ? 'Open your hand to choose cards'
    : n >= MAX_STAGED ? 'Hand full — Resolve!'
    : `${n} card${n > 1 ? 's' : ''} staged · ↑/↓ sets polarity`;
  updateAxiumMeter(B.playerPlayed);
}

// ─────────────────────────────────────────────────────────
// HAND PICKER  —  now uses #screen-hand-picker overlay
// ─────────────────────────────────────────────────────────
const _miniAnims = new Map();
let _pickerFilter = 'All';

function openHandPicker(B) {
  _pickerFilter = 'All';
  // Reset tab state
  document.querySelectorAll('.picker-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.layer === 'All')
  );
  _renderPickerGrid(B);
  _overlayOpen('screen-hand-picker');
}

function closeHandPicker() {
  _overlayClose('screen-hand-picker');
}

function _renderPickerGrid(B) {
  const list = gel('picker-list');
  if (!list) return;
  list.innerHTML = '';

  const hand = B.playerHand || [];
  const filtered = _pickerFilter === 'All'
    ? hand
    : hand.filter(c => c.layer === _pickerFilter);

  filtered.forEach(card => {
    const existing   = B.playerPlayed.find(c => (c.card || c).id === card.id);
    const isStaged   = !!existing;
    const isReversed = existing ? existing.reversed : false;
    const layerCol   = card.layer === 'Superego' ? '#D4AF37'
                     : card.layer === 'Ego'      ? '#7EB8E8' : '#86EFAC';
    const normalMech = card.layer === 'Superego'
      ? `Shield +${card.shieldVal || 0}`
      : card.layer === 'Ego'
        ? [(card.chunkFlat ? `+${card.chunkFlat} flat` : ''), (card.chunkPct ? `×${card.chunkPct}` : '')].filter(Boolean).join(' ')
        : `Recharge +${card.rechargeVal || 0}/stack`;
    const revMech = `Reversed: −${Math.round(Math.abs(card.reversedShift || card.shieldVal * 0.4 || 8))}`;

    // Use csystem CardRenderer if available, otherwise build a simple row
    if (typeof CardRenderer !== 'undefined') {
      const renderer = new CardRenderer(card, {
        size: 'md', showTooltip: true, staged: isStaged,
      });
      renderer.el.style.cursor = 'pointer';
      renderer.el.style.position = 'relative';

      // Staged highlight
      if (isStaged) {
        if (isReversed) {
          renderer.el.style.borderColor = 'rgba(224,85,85,.7)';
          renderer.el.style.boxShadow   = '0 0 0 1px rgba(224,85,85,.4),0 0 18px rgba(224,85,85,.3)';
        } else {
          renderer.el.style.borderColor = '#D4AF37';
          renderer.el.style.boxShadow   = '0 0 0 1px rgba(212,175,55,.45),0 0 18px rgba(212,175,55,.3)';
        }
        // Polarity pill
        _attachPickerPill(renderer.el, card, existing, B);
      }

      // Tap to toggle stage
      let _pd = { x: 0, y: 0 };
      renderer.el.addEventListener('pointerdown', e => { _pd = { x: e.clientX, y: e.clientY }; }, { passive: true });
      renderer.el.addEventListener('pointerup', e => {
        if (e.target.closest('.picker-pol-pill')) return;
        if (Math.abs(e.clientX - _pd.x) < 10 && Math.abs(e.clientY - _pd.y) < 10) {
          _pickerToggle(card, B);
        }
      });
      list.appendChild(renderer.el);

    } else {
      // Fallback: simple list row (original ui.js style)
      const row = document.createElement('div');
      row.className = 'picker-row' + (isStaged ? (isReversed ? ' staged-reversed' : ' staged') : '');
      row.dataset.id = card.id;
      row.innerHTML = `
        <div class="picker-check ${isStaged ? (isReversed ? 'on-rev' : 'on') : ''}" id="chk-${card.id}">
          <svg viewBox="0 0 12 12" fill="none"><polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
        </div>
        <canvas class="picker-mini-cvs" width="40" height="40"></canvas>
        <div class="picker-info">
          <div class="picker-name" style="color:${card.color}">${card.name}</div>
          <div class="picker-layer-lbl" style="color:${layerCol}">${cardLabel(card)}</div>
          <div class="picker-mechanic">${isReversed ? revMech : normalMech}</div>
        </div>
        <button class="picker-polarity-btn ${isStaged ? (isReversed ? 'reversed' : 'normal') : 'normal'}">
          ${isReversed ? '↓ REV' : '↑ NRM'}
        </button>
        <div class="picker-axium-lbl" style="color:${card.color}">⬡${card.axiumScore || '?'}</div>
      `;
      row.addEventListener('click', e => {
        if (e.target.closest('.picker-polarity-btn')) return;
        _pickerToggle(card, B);
      });
      const polBtn = row.querySelector('.picker-polarity-btn');
      if (polBtn) polBtn.addEventListener('click', e => {
        e.stopPropagation();
        const entry = B.playerPlayed.find(c => (c.card || c).id === card.id);
        if (!entry) return;
        entry.reversed = !entry.reversed;
        _renderPickerGrid(B);
      });
      setTimeout(() => {
        const cvs = row.querySelector('.picker-mini-cvs');
        if (cvs) animatePickerMini(cvs, card);
      }, 30);
      list.appendChild(row);
    }
  });

  _updatePickerFooter(B);
}

function _attachPickerPill(cardEl, card, entry, B) {
  const old = cardEl.querySelector('.picker-pol-pill');
  if (old) old.remove();
  const pill = document.createElement('div');
  pill.className = 'picker-pol-pill';
  const isRev = entry.reversed;
  pill.style.cssText = `
    position:absolute;bottom:6px;left:50%;transform:translateX(-50%);
    font-family:'Space Mono',monospace;font-size:7px;letter-spacing:.08em;
    text-transform:uppercase;padding:3px 9px;border-radius:10px;border:1px solid;
    cursor:pointer;z-index:20;white-space:nowrap;user-select:none;
    touch-action:manipulation;-webkit-tap-highlight-color:transparent;
    ${isRev
      ? 'color:rgba(224,85,85,.9);border-color:rgba(224,85,85,.5);background:rgba(224,85,85,.13);'
      : 'color:rgba(212,175,55,.9);border-color:rgba(212,175,55,.45);background:rgba(212,175,55,.12);'}
  `;
  pill.textContent = isRev ? '↓ REV' : '↑ NRM';
  pill.addEventListener('click', e => {
    e.stopPropagation();
    entry.reversed = !entry.reversed;
    _renderPickerGrid(B);
    // Also update field chips if visible
    if (typeof renderBattleField === 'function') renderBattleField(B, i => {
      B.playerPlayed.splice(i, 1);
      _renderPickerGrid(B);
      updateBattlePhaseUI(B);
    });
    updateBattlePhaseUI(B);
  });
  cardEl.appendChild(pill);
}

function _pickerToggle(card, B) {
  const idx = B.playerPlayed.findIndex(e => (e.card || e).id === card.id);
  if (idx >= 0) {
    B.playerPlayed.splice(idx, 1);
  } else {
    if (B.playerPlayed.length >= MAX_STAGED) { toast('Limit', 'Max 10 cards'); return; }
    B.playerPlayed.push({ card, reversed: false });
  }
  _renderPickerGrid(B);
  updateBattlePhaseUI(B);
}

function _updatePickerFooter(B) {
  const n = (B?.playerPlayed || []).length;
  const countEl  = gel('picker-count-lbl');
  const statusEl = gel('picker-status-msg');
  const resolveEl= gel('picker-confirm');
  if (countEl)   countEl.textContent   = `${n}/10`;
  if (statusEl)  statusEl.textContent  = n === 0 ? 'Select cards to stage'
                                       : n >= 10  ? 'Hand full — ready to Resolve!'
                                       : `${n} staged · tap card to toggle`;
  if (resolveEl) resolveEl.classList.toggle('ready', n > 0);
}

// ─────────────────────────────────────────────────────────
// CARD CANVAS ANIMATIONS
// ─────────────────────────────────────────────────────────
function animateCardCanvas(canvas, card) {
  if (!canvas) return;
  const W = canvas.offsetWidth || 120, H = canvas.offsetHeight || 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts || !pts.length) return;
  const { mapped, edges } = _buildConstellation(pts, W, H, 4);
  const phases = mapped.map(() => Math.random() * Math.PI * 2);
  let t = 0; const key = 'card-' + card.id + '-' + Math.random();
  if (_miniAnims.has(key)) cancelAnimationFrame(_miniAnims.get(key));
  function frame() {
    if (!canvas.isConnected) { _miniAnims.delete(key); return; }
    t += .018; ctx.clearRect(0, 0, W, H);
    edges.forEach(([a, b]) => {
      const p = .15 + .08 * Math.sin(t + (a + b) * .5);
      ctx.beginPath(); ctx.strokeStyle = card.color + hexA(p * 220); ctx.lineWidth = .7;
      ctx.moveTo(mapped[a].x, mapped[a].y); ctx.lineTo(mapped[b].x, mapped[b].y); ctx.stroke();
    });
    mapped.forEach((p, i) => {
      const tw = .55 + .45 * Math.sin(t * .95 + phases[i]); const r = 1.1 + tw * .6;
      ctx.beginPath(); ctx.arc(p.x, p.y, r * 2.5, 0, Math.PI * 2); ctx.fillStyle = card.color + hexA(tw * 25); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fillStyle = card.color + 'bb'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, .7, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fill();
    });
    _miniAnims.set(key, requestAnimationFrame(frame));
  }
  frame();
}

function animatePickerMini(canvas, card) {
  if (!canvas) return;
  const W = 40, H = 40; canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts || !pts.length) return;
  const { mapped, edges } = _buildConstellation(pts, W, H, 3);
  const phases = mapped.map(() => Math.random() * Math.PI * 2);
  let t = 0; const key = 'mini-' + card.id + '-' + Math.random();
  if (_miniAnims.has(key)) cancelAnimationFrame(_miniAnims.get(key));
  function frame() {
    if (!canvas.isConnected) { _miniAnims.delete(key); return; }
    t += .02; ctx.clearRect(0, 0, W, H);
    edges.forEach(([a, b]) => {
      ctx.beginPath(); ctx.strokeStyle = card.color + hexA(.2 * 220); ctx.lineWidth = .7;
      ctx.moveTo(mapped[a].x, mapped[a].y); ctx.lineTo(mapped[b].x, mapped[b].y); ctx.stroke();
    });
    mapped.forEach((p, i) => {
      const tw = .5 + .5 * Math.sin(t + phases[i]); const r = .9 + tw * .5;
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.fillStyle = card.color + 'cc'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, .45, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,255,255,.85)'; ctx.fill();
    });
    _miniAnims.set(key, requestAnimationFrame(frame));
  }
  frame();
}

function _buildConstellation(pts, W, H, pad) {
  let mnX = 1e9, mnY = 1e9, mxX = -1e9, mxY = -1e9;
  pts.forEach(([x, y]) => { mnX = Math.min(mnX, x); mnY = Math.min(mnY, y); mxX = Math.max(mxX, x); mxY = Math.max(mxY, y); });
  const sc = Math.min((W - pad * 2) / (mxX - mnX || 1), (H - pad * 2) / (mxY - mnY || 1));
  const oX = W / 2 - (mnX + mxX) / 2 * sc;
  const oY = H / 2 - (mnY + mxY) / 2 * sc;
  const mapped = pts.map(([x, y]) => ({ x: x * sc + oX, y: y * sc + oY }));
  const edgeSet = new Set(), edges = [];
  mapped.forEach((p, i) => {
    mapped.map((q, j) => ({ j, d: Math.hypot(q.x - p.x, q.y - p.y) }))
      .filter(v => v.j !== i).sort((a, b) => a.d - b.d).slice(0, 2)
      .forEach(({ j }) => {
        const k = Math.min(i, j) + '-' + Math.max(i, j);
        if (!edgeSet.has(k)) { edgeSet.add(k); edges.push([i, j]); }
      });
  });
  return { mapped, edges };
}

// ─────────────────────────────────────────────────────────
// PURGE SAVE DATA
// ─────────────────────────────────────────────────────────
function purgeSaveData() {
  if (confirm('This will permanently dissolve your current constellation and progress. Proceed?')) {
    localStorage.removeItem('axium_save');
    localStorage.removeItem('axium_history');
    toast('System Purged', 'Returning to the void.');
  }
}
