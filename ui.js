/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — UI.js  (Shared UI Utilities)
 * Depends on: cards.js (ATTN_STATES, getAttnState, getSynergies)
 * Must load BEFORE shop.js and battle.js
 * ═══════════════════════════════════════════════════════════════
 */

'use strict';

// ─────────────────────────────────────────────────────────
// CONSTANTS — set by chapter config in game.html
// ─────────────────────────────────────────────────────────
const MAX_STAGED = 10;
const MIN_ATTN   = 5;

// ─────────────────────────────────────────────────────────
// DOM HELPERS
// ─────────────────────────────────────────────────────────
function gel(id)          { return document.getElementById(id); }
function clamp(v, a, b)   { return Math.max(a, Math.min(b, v)); }
function lerp(a, b, t)    { return a + (b - a) * t; }
function hexA(val)         { return Math.max(0, Math.min(255, Math.round(val))).toString(16).padStart(2, '0'); }
function cap(s)            { return s ? s[0].toUpperCase() + s.slice(1) : ''; }
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
// SCREEN NAVIGATION
// ─────────────────────────────────────────────────────────
function goTo(screenId, afterFade) {
  const overlay = gel('fade-overlay');
  overlay.classList.add('dark');
  setTimeout(() => {
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    gel(screenId)?.classList.add('active');
    overlay.classList.remove('dark');
    afterFade?.();
  }, 450);
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
  const fill = gel(which + '-fill');   if (fill)   fill.style.width  = pct + '%';
  const cur  = gel(which + '-cursor'); if (cur)    cur.style.left   = pct + '%';
  const lbl  = gel(which + '-state-lbl'); if (lbl) { lbl.textContent = state.label; lbl.style.color = state.col; }
}

/** Animate bar to new attention value */
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
// Expects B.playerPlayed as [{card, reversed}] and B.enemyHand as [{card, drain}]
// ─────────────────────────────────────────────────────────
function renderBattleField(B, onUnstage) {
  // Player field
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
        // Polarity toggle
        const polBtn = document.createElement('span');
        polBtn.className = 'f-chip-polarity ' + (reversed ? 'reversed' : 'normal');
        polBtn.textContent = reversed ? '↓' : '↑';
        polBtn.title = reversed ? 'Reversed (drain)' : 'Normal (shield/gain)';
        polBtn.onclick = e => {
          e.stopPropagation();
          entry.reversed = !entry.reversed;
          renderBattleField(B, onUnstage);
          updateBattlePhaseUI(B);
        };
        chip.appendChild(polBtn);
        // Remove
        const rm = document.createElement('span');
        rm.className = 'f-chip-remove'; rm.textContent = '×';
        rm.onclick = e => { e.stopPropagation(); onUnstage(i); };
        chip.appendChild(rm);
        pf.appendChild(chip);
      });
    }
  }

  // Enemy field
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
        chip.innerHTML = `<span>${icon}</span><span>${card.name}</span><span style="font-family:'JetBrains Mono',monospace;font-size:7px;opacity:.55;margin-left:4px;">−${drain}</span>`;
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
    : `${n} card${n > 1 ? 's' : ''} staged · ↑/↓ sets polarity · Resolve or add more`;
  updateAxiumMeter(B.playerPlayed);
}

// ─────────────────────────────────────────────────────────
// HAND PICKER  (with polarity toggle)
// ─────────────────────────────────────────────────────────
const _miniAnims = new Map();

function openHandPicker(B) {
  const list = gel('picker-list');
  list.innerHTML = '';

  B.playerHand.forEach(card => {
    const existing  = B.playerPlayed.find(c => c.card.id === card.id);
    const isStaged  = !!existing;
    const isReversed = existing ? existing.reversed : false;
    const layerCol  = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
    const normalMech = card.layer === 'Superego'
      ? `Shield +${card.shieldVal || 0}`
      : card.layer === 'Ego'
        ? [(card.chunkFlat ? `+${card.chunkFlat} flat` : ''), (card.chunkPct ? `×${card.chunkPct}` : '')].filter(Boolean).join(' ')
        : `Recharge +${card.rechargeVal || 0}/stack`;
    const revMech = `Reversed: ${card.reversedDesc || `−${Math.round(Math.abs(card.reversedShift || card.shieldVal * 0.4 || 8))} drain`}`;

    const row = document.createElement('div');
    row.className = 'picker-row' + (isStaged ? (isReversed ? ' staged-reversed' : ' staged') : '');
    row.dataset.id = card.id;
    const checkClass = isStaged ? (isReversed ? 'on-rev' : 'on') : '';

    row.innerHTML = `
      <div class="picker-check ${checkClass}" id="chk-${card.id}">
        <svg viewBox="0 0 12 12" fill="none"><polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </div>
      <canvas class="picker-mini-cvs" width="40" height="40"></canvas>
      <div class="picker-info">
        <div class="picker-name" style="color:${card.color}">${card.name}</div>
        <div class="picker-layer-lbl" style="color:${layerCol}">${cardLabel(card)}</div>
        <div class="picker-mechanic" id="mech-${card.id}">${isReversed ? revMech : normalMech}</div>
        ${isReversed ? `<div class="picker-reversed-hint">${revMech}</div>` : ''}
      </div>
      <button class="picker-polarity-btn ${isStaged ? (isReversed ? 'reversed' : 'normal') : 'normal'}" id="pol-${card.id}" title="Toggle polarity">
        ${isReversed ? '↓ REV' : '↑ NRM'}
      </button>
      <div class="picker-axium-lbl" style="color:${card.color}">⬡${card.axiumScore || '?'}</div>
    `;

    // Click row = toggle staged
    row.addEventListener('click', e => {
      if (e.target.closest('.picker-polarity-btn')) return;
      _toggleStageCard(card, row, B);
    });

    // Polarity toggle
    row.querySelector(`#pol-${card.id}`)?.addEventListener('click', e => {
      e.stopPropagation();
      _togglePolarity(card.id, B, row);
    });

    list.appendChild(row);

    setTimeout(() => {
      const cvs = row.querySelector('.picker-mini-cvs');
      if (cvs) animatePickerMini(cvs, card);
    }, 30);
  });

  gel('picker-count-lbl').textContent = `${B.playerPlayed.length}/10`;
  gel('hand-picker').classList.add('show');
}

function closeHandPicker() {
  gel('hand-picker').classList.remove('show');
}

function _togglePolarity(cardId, B, rowEl) {
  const entry = B.playerPlayed.find(c => c.card.id === cardId);
  if (!entry) return;
  entry.reversed = !entry.reversed;
  const isRev = entry.reversed;
  const card  = entry.card;

  rowEl.classList.toggle('staged-reversed', isRev);
  rowEl.classList.toggle('staged', !isRev);

  const chk = gel('chk-' + cardId);
  if (chk) { chk.classList.remove('on', 'on-rev'); chk.classList.add(isRev ? 'on-rev' : 'on'); }

  const polBtn = gel('pol-' + cardId);
  if (polBtn) { polBtn.className = 'picker-polarity-btn ' + (isRev ? 'reversed' : 'normal'); polBtn.textContent = isRev ? '↓ REV' : '↑ NRM'; }

  const normalMech = card.layer === 'Superego'
    ? `Shield +${card.shieldVal || 0}`
    : card.layer === 'Ego'
      ? [(card.chunkFlat ? `+${card.chunkFlat} flat` : ''), (card.chunkPct ? `×${card.chunkPct}` : '')].filter(Boolean).join(' ')
      : `Recharge +${card.rechargeVal || 0}/stack`;
  const revMech = `Reversed: ${card.reversedDesc || `−${Math.round(Math.abs(card.reversedShift || card.shieldVal * 0.4 || 8))} drain`}`;
  const mechEl = gel('mech-' + cardId);
  if (mechEl) mechEl.textContent = isRev ? revMech : normalMech;
}

function _toggleStageCard(card, rowEl, B) {
  const already = B.playerPlayed.findIndex(c => c.card.id === card.id);
  if (already >= 0) {
    B.playerPlayed.splice(already, 1);
    rowEl.classList.remove('staged', 'staged-reversed');
    const chk = rowEl.querySelector('.picker-check');
    if (chk) chk.classList.remove('on', 'on-rev');
    const pol = rowEl.querySelector('.picker-polarity-btn');
    if (pol) { pol.className = 'picker-polarity-btn normal'; pol.textContent = '↑ NRM'; }
  } else {
    if (B.playerPlayed.length >= MAX_STAGED) { toast('Limit', 'Max 10 cards'); return; }
    B.playerPlayed.push({ card, reversed: false });
    rowEl.classList.add('staged');
    const chk = rowEl.querySelector('.picker-check');
    if (chk) chk.classList.add('on');
  }
  gel('picker-count-lbl').textContent = `${B.playerPlayed.length}/10`;
  updateBattlePhaseUI(B);
  renderBattleField(B, i => {
    B.playerPlayed.splice(i, 1);
    const r = gel('hand-picker')?.querySelector(`[data-id="${B.playerPlayed[i]?.card?.id}"]`);
    if (r) { r.classList.remove('staged', 'staged-reversed'); }
    gel('picker-count-lbl').textContent = `${B.playerPlayed.length}/10`;
    renderBattleField(B, arguments.callee);
    updateBattlePhaseUI(B);
  });
}

// ─────────────────────────────────────────────────────────
// CARD CANVAS ANIMATIONS
// ─────────────────────────────────────────────────────────

/** Full card canvas animation (used in shop) */
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

/** Tiny picker mini canvas */
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

/** Shared constellation builder */
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
// PURGE SAVE DATA (shared utility)
// ─────────────────────────────────────────────────────────
function purgeSaveData() {
  if (confirm('This will permanently dissolve your current constellation and progress. Proceed?')) {
    localStorage.removeItem('axium_save');
    localStorage.removeItem('axium_history');
    toast('System Purged', 'Returning to the void.');
  }
}
