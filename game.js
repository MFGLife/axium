// ═══════════════════════════════════════════════════════════════
// AXIUM — GAME ENGINE v6.0
// Constellation vs Constellation · cards.js required
//
// LAYOUT:
//   TOP    → Enemy attention bar + enemy cards (reversed)
//   CENTER → VS line
//   BOTTOM → Player cards + player attention bar
//
// BATTLE:
//   Cards activate one by one. Particle beam fires to the bar
//   it affects. Bar animates live. Math floats up on bar.
//   No trauma deck — enemy plays reversed player cards.
// ═══════════════════════════════════════════════════════════════

const CHAPTER_CONFIG = window.CHAPTER_CONFIG || {
  id:          1,
  label:       '01 · Binding of Ego',
  title:       'Binding of Ego',
  axium:       '"The absence of love binds ego, allowing the awareness of God."',
  narrative:   'Build your constellation. Stage cards, then Resolve — your hand against theirs.',
  shopChapter: 1,
  playerStart: 55,
  enemyStart:  80,
  baseMaxAttn: 100,
  handSize:    6,
  winAttn:     90,
  loseAttn:    10,
  introLabel:  'The Cathedral · Chapter I',
  introBtnText:'Enter the Fight',
  winTitle:    'Attention Held',
  winDesc:     'Your constellation held. The chapter is complete.',
  winBtn:      'Visit the Shop',
  loseTitle:   'Attention Lost',
  loseDesc:    'The constellation overwhelmed your attention. Try again.',
  loseBtn:     'Try Again',
  // ms per card activation during battle
  cardDelay:   1100,
};

// ── CONSTANTS ────────────────────────────────────────────────
const STARTING_HAND   = CHAPTER_CONFIG.handSize || 6;
const MAX_STAGED      = 10;
const MIN_ATTN        = 5;
const BASE_MAX_ATTN   = CHAPTER_CONFIG.baseMaxAttn || 100;
const CURRENT_CHAPTER = CHAPTER_CONFIG.id || 1;

// Enemy deck: 6 reversed player cards chosen at chapter start
const ENEMY_CARD_COUNT = 4;

// ── GAME STATE ───────────────────────────────────────────────
const S = {
  playerAttn:   CHAPTER_CONFIG.playerStart,
  enemyAttn:    CHAPTER_CONFIG.enemyStart,
  maxAttn:      BASE_MAX_ATTN,
  turn:         1,
  phase:        'build',
  playerDeck:   [],
  playerHand:   [],
  playerPlayed: [],
  enemyHand:    [],   // reversed cards enemy will play
  modalCard:    null,
  won:  false,
  lost: false,
};

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
function startChapter() {
  el('intro').style.display = 'none';
  S.playerDeck  = shuffle([...PLAYER_CARDS]);
  S.playerAttn  = CHAPTER_CONFIG.playerStart;
  S.enemyAttn   = CHAPTER_CONFIG.enemyStart;
  S.turn        = 1;
  S.phase       = 'build';
  S.won = false; S.lost = false;
  S.playerPlayed = [];
  recalcMaxAttn();
  buildEnemyHand();
  initBars();
  dealHand();
  renderField();
  updatePhaseUI();
  log(`── Chapter ${CURRENT_CHAPTER} · ${CHAPTER_CONFIG.title} ──`, 'sys');
  log('Stage your cards, then Resolve.', 'sys');
}

function recalcMaxAttn() {
  const cap = calculateCapacity(S.playerDeck);
  S.maxAttn = clamp(BASE_MAX_ATTN + cap, 20, 300);
}

// Enemy plays reversed versions of randomly chosen player cards
function buildEnemyHand() {
  const pool = shuffle([...PLAYER_CARDS]).slice(0, ENEMY_CARD_COUNT);
  S.enemyHand = pool.map(card => ({
    ...card,
    reversed: true,
    // Reversed effect: drain player by reversedShift (positive = drain)
    drainVal: Math.abs(card.reversedShift || Math.round((card.shieldVal || 10) * 0.4)),
    displayName: card.name + ' ✦ reversed',
  }));
}

// ═══════════════════════════════════════════════════════════════
// ATTENTION BARS  (horizontal, top = enemy, bottom = player)
// ═══════════════════════════════════════════════════════════════
function initBars() {
  buildPips('player-pips');
  buildPips('enemy-pips');
  updateBars();
}

function buildPips(id) {
  const c = el(id); if (!c) return; c.innerHTML = '';
  ATTN_STATES.forEach(s => {
    const p = document.createElement('div');
    p.className = 'bar-pip';
    p.style.left = (s.pos * 100) + '%';
    p.style.setProperty('--pip-col', s.col);
    c.appendChild(p);
  });
}

function updateBars() {
  _updateBar('player', S.playerAttn, S.maxAttn, 'player-fill', 'player-cursor', 'player-state-lbl', 'player-pips');
  _updateBar('enemy',  S.enemyAttn,  S.maxAttn, 'enemy-fill',  'enemy-cursor',  'enemy-state-lbl',  'enemy-pips');
}

function _updateBar(who, val, max, fillId, curId, lblId, pipsId) {
  const pct = clamp(val / max * 100, 0, 100);
  const fill = el(fillId); if (fill) fill.style.width = pct + '%';
  const cur  = el(curId);  if (cur)  cur.style.left  = pct + '%';
  const attnFor100 = clamp(val / max * 100, 0, 100);
  const state = getAttnState(attnFor100);
  const lbl = el(lblId);
  if (lbl) { lbl.textContent = state.label; lbl.style.color = state.col; }
  updatePips(pipsId, pct);
}

function updatePips(id, pct) {
  document.querySelectorAll(`#${id} .bar-pip`).forEach((p, i) => {
    p.classList.toggle('lit', ATTN_STATES[i] && ATTN_STATES[i].pos <= pct / 100);
  });
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

// Animate a bar smoothly to targetVal, fire onDone when complete
function animateBarTo(who, startVal, targetVal, duration, onDone) {
  const start = performance.now();
  const fillId  = who + '-fill';
  const curId   = who + '-cursor';
  const lblId   = who + '-state-lbl';
  const pipsId  = who + '-pips';
  const maxVal  = S.maxAttn;

  function tick(now) {
    const p   = Math.min(1, (now - start) / duration);
    const cur = startVal + (targetVal - startVal) * easeOut(p);
    const val = clamp(cur, 0, maxVal);
    if (who === 'player') S.playerAttn = val;
    else                  S.enemyAttn  = val;
    _updateBar(who, val, maxVal, fillId, curId, lblId, pipsId);
    if (p < 1) requestAnimationFrame(tick);
    else if (onDone) onDone();
  }
  requestAnimationFrame(tick);
}

// ═══════════════════════════════════════════════════════════════
// FLOATING MATH TEXT on bar
// ═══════════════════════════════════════════════════════════════
function floatMath(barId, text, color) {
  const barEl = el(barId); if (!barEl) return;
  const rect  = barEl.getBoundingClientRect();
  const div   = document.createElement('div');
  div.style.cssText = `
    position:fixed;
    left:${rect.left + rect.width * 0.5}px;
    top:${rect.top - 6}px;
    transform:translateX(-50%);
    font-family:'JetBrains Mono',monospace;
    font-size:clamp(11px,2.2vw,15px);
    font-weight:700;
    color:${color};
    text-shadow:0 0 12px ${color};
    pointer-events:none;
    z-index:9999;
    white-space:nowrap;
    opacity:1;
    transition:transform 0.85s cubic-bezier(.22,1,.36,1), opacity 0.85s ease;
  `;
  div.textContent = text;
  document.body.appendChild(div);
  requestAnimationFrame(() => {
    div.style.transform = 'translateX(-50%) translateY(-38px)';
    div.style.opacity   = '0';
  });
  setTimeout(() => div.remove(), 950);
}

// ═══════════════════════════════════════════════════════════════
// PARTICLE BEAM  (card → bar)
// Mobile-safe: lightweight canvas, 20 particles max
// ═══════════════════════════════════════════════════════════════
function fireBeam(fromEl, toBarId, color, onDone) {
  const toEl  = el(toBarId); if (!fromEl || !toEl) { onDone?.(); return; }
  const fRect = fromEl.getBoundingClientRect();
  const tRect = toEl.getBoundingClientRect();

  const sx = fRect.left + fRect.width  / 2;
  const sy = fRect.top  + fRect.height / 2;
  const ex = tRect.left + tRect.width  / 2;
  const ey = tRect.top  + tRect.height / 2;

  const cvs = document.createElement('canvas');
  cvs.style.cssText = 'position:fixed;inset:0;width:100%;height:100%;pointer-events:none;z-index:800;';
  cvs.width  = window.innerWidth;
  cvs.height = window.innerHeight;
  document.body.appendChild(cvs);
  const ctx = cvs.getContext('2d');

  const [r, g, b] = hexRGB(color || '#D4AF37');
  const DURATION  = 420; // ms
  const start     = performance.now();
  let   rafId;

  function frame(now) {
    const prog = Math.min(1, (now - start) / DURATION);
    ctx.clearRect(0, 0, cvs.width, cvs.height);

    // Travelling orb along the line
    const px = sx + (ex - sx) * easeOut(prog);
    const py = sy + (ey - sy) * easeOut(prog);

    // Trail
    const steps = 12;
    for (let i = 0; i < steps; i++) {
      const tp   = Math.max(0, prog - i * 0.018);
      const tx   = sx + (ex - sx) * easeOut(tp);
      const ty   = sy + (ey - sy) * easeOut(tp);
      const alpha = (1 - i / steps) * (1 - prog * 0.5) * 0.65;
      const radius = (3.5 - i * 0.22) * (1 - prog * 0.3);
      if (radius < 0.1) continue;
      ctx.beginPath();
      ctx.arc(tx, ty, radius, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
      ctx.fill();
    }

    // Leading orb
    const glowSize = 8 + 4 * Math.sin(prog * Math.PI);
    const grd = ctx.createRadialGradient(px, py, 0, px, py, glowSize * 2);
    grd.addColorStop(0, `rgba(255,255,255,0.9)`);
    grd.addColorStop(0.3, `rgba(${r},${g},${b},0.8)`);
    grd.addColorStop(1,   `rgba(${r},${g},${b},0)`);
    ctx.beginPath();
    ctx.arc(px, py, glowSize * 2, 0, Math.PI * 2);
    ctx.fillStyle = grd;
    ctx.fill();

    // Inner bright core
    ctx.beginPath();
    ctx.arc(px, py, 3.5, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(255,255,255,0.95)`;
    ctx.fill();

    if (prog < 1) {
      rafId = requestAnimationFrame(frame);
    } else {
      // Impact burst on bar
      _impactBurst(ctx, ex, ey, r, g, b);
      setTimeout(() => { cvs.remove(); onDone?.(); }, 180);
    }
  }

  rafId = requestAnimationFrame(frame);
}

function _impactBurst(ctx, x, y, r, g, b) {
  const N = 14;
  const start = performance.now();
  function burst(now) {
    const p = Math.min(1, (now - start) / 160);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    for (let i = 0; i < N; i++) {
      const angle = (i / N) * Math.PI * 2;
      const dist  = p * (18 + i * 2.5);
      const bx    = x + Math.cos(angle) * dist;
      const by    = y + Math.sin(angle) * dist;
      const alpha = (1 - p) * 0.7;
      ctx.beginPath();
      ctx.arc(bx, by, 2.5 * (1 - p * 0.6), 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha.toFixed(3)})`;
      ctx.fill();
    }
    if (p < 1) requestAnimationFrame(burst);
  }
  requestAnimationFrame(burst);
}

// ═══════════════════════════════════════════════════════════════
// CARD GLOW PULSE (during activation)
// ═══════════════════════════════════════════════════════════════
function pulseCard(cardEl, color) {
  if (!cardEl) return;
  const prev = cardEl.style.boxShadow;
  const prevBorder = cardEl.style.borderColor;
  cardEl.style.transition = 'box-shadow 0.18s, border-color 0.18s, transform 0.18s';
  cardEl.style.boxShadow  = `0 0 32px ${color}, 0 0 8px ${color} inset`;
  cardEl.style.borderColor = color;
  cardEl.style.transform  = 'scale(1.08)';
  setTimeout(() => {
    cardEl.style.boxShadow   = prev;
    cardEl.style.borderColor = prevBorder;
    cardEl.style.transform   = '';
  }, 600);
}

// ═══════════════════════════════════════════════════════════════
// STATUS + PHASE UI
// ═══════════════════════════════════════════════════════════════
function updateStatusStrip() {
  const strip = el('status-strip'); if (!strip) return; strip.innerHTML = '';
  if (S.playerPlayed.length) {
    const d = document.createElement('div');
    d.className = 'status-pill shield';
    d.textContent = `${S.playerPlayed.length} staged`;
    strip.appendChild(d);
  }
}

function updatePhaseUI() {
  const n  = S.playerPlayed.length;
  const rb = el('resolve-btn');
  if (rb) rb.classList.toggle('show', n > 0 && S.phase === 'build');
  const pb = el('pass-btn');
  if (pb) pb.disabled = S.phase !== 'build';
  const pm = el('phase-msg');
  if (pm) pm.textContent = n === 0
    ? 'Open your hand to choose cards'
    : n >= MAX_STAGED
      ? 'Hand full — Resolve!'
      : `${n} card${n > 1 ? 's' : ''} staged · Resolve or add more`;
  const tn = el('turn-n');
  if (tn) tn.textContent = S.turn;
  updateAxiumMeter();
  updateStatusStrip();
}

function updateAxiumMeter() {
  const n   = S.playerPlayed.length;
  const avg = n ? S.playerPlayed.reduce((s, c) => s + (c.axiumScore || 0), 0) / n : 0;
  const lit = Math.round(avg);
  for (let i = 0; i < 10; i++) {
    const p = el(`axp-${i}`); if (p) p.classList.toggle('lit', i < lit);
  }
}

// ═══════════════════════════════════════════════════════════════
// CARD RENDERING
// ═══════════════════════════════════════════════════════════════
const cardAnimations = new Map();

function hexRGB(hex) {
  const h = (hex || '#ffffff').replace('#', '');
  if (h.length === 3) return [parseInt(h[0]+h[0],16), parseInt(h[1]+h[1],16), parseInt(h[2]+h[2],16)];
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}

function hexA(v) {
  return Math.max(0, Math.min(255, Math.round(v))).toString(16).padStart(2, '0');
}

function cardLabel(card) {
  if (card.traumaRole) return card.traumaRole;
  const layer = card.layer || '';
  if (layer === 'Superego') { const num = card.number !== undefined ? numberToRoman(card.number) : ''; return num ? `Superego · ${num}` : 'Superego'; }
  if (layer === 'Ego')  return `Ego · ${cap(card.suit || '')}`;
  if (layer === 'ID')   return `ID · ${cap(card.suit || '')}`;
  return layer || 'Card';
}
function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : ''; }
function numberToRoman(n) {
  const map = [[21,'XXI'],[20,'XX'],[19,'XIX'],[18,'XVIII'],[17,'XVII'],[16,'XVI'],[15,'XV'],[14,'XIV'],[13,'XIII'],[12,'XII'],[11,'XI'],[10,'X'],[9,'IX'],[8,'VIII'],[7,'VII'],[6,'VI'],[5,'V'],[4,'IV'],[3,'III'],[2,'II'],[1,'I'],[0,'0']];
  return (map.find(([v]) => v === n) || [n, String(n)])[1];
}

function makeCard(card, classes = []) {
  const isEnemy   = classes.includes('enemy-placed') || card.reversed;
  const nameCol   = isEnemy ? 'rgba(220,120,120,.95)' : '#fff';
  const layerCol  = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
  const div       = document.createElement('div');
  div.className   = 'card ' + classes.join(' ');
  div.dataset.id  = card.id;

  const glow = document.createElement('div');
  glow.className = 'card-glow';
  glow.style.background = `radial-gradient(ellipse at 35% 25%, ${card.color}44, transparent 65%)`;
  div.appendChild(glow);

  const top = document.createElement('div');
  top.className = 'card-top';
  top.innerHTML = `<div class="card-axium-lbl" style="color:${layerCol}">${isEnemy ? '✦ reversed' : cardLabel(card)}</div><div class="card-name" style="color:${nameCol}">${card.name}</div>`;
  div.appendChild(top);

  const cvs = document.createElement('canvas');
  cvs.className = 'card-cvs';
  div.appendChild(cvs);

  const cardType = card.type || 'compression';
  const axium    = card.axiumScore !== undefined ? card.axiumScore : '?';
  const bot      = document.createElement('div');
  bot.className  = 'card-bot';
  bot.innerHTML  = `<span class="card-type-pip ${cardType}">${cardType.slice(0,4)}</span><span class="card-intensity">⬡${axium}</span>`;
  div.appendChild(bot);

  if (card.reversed) {
    div.style.opacity      = '0.9';
    div.style.borderColor  = 'rgba(224,85,85,.45)';
    div.style.boxShadow    = '0 0 14px rgba(224,85,85,.2)';
  }

  setTimeout(() => animateCardCanvas(cvs, card, isEnemy), 20);
  return div;
}

function animateCardCanvas(canvas, card, isEnemy = false) {
  if (!canvas || !canvas.parentElement) return;
  const W = canvas.offsetWidth || 70, H = canvas.offsetHeight || 50;
  if (W < 4 || H < 4) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts || !pts.length) return;

  let mnX=1e9, mnY=1e9, mxX=-1e9, mxY=-1e9;
  pts.forEach(([x,y]) => { if(x<mnX)mnX=x; if(y<mnY)mnY=y; if(x>mxX)mxX=x; if(y>mxY)mxY=y; });
  const pad = 4, sc = Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX = W/2-(mnX+mxX)/2*sc, oY = H/2-(mnY+mxY)/2*sc;
  const mapped = pts.map(([x,y]) => ({ x: x*sc+oX, y: y*sc+oY }));

  const edgeSet = new Set(), edges = [];
  mapped.forEach((p,i) => {
    mapped.map((q,j) => ({ j, d: Math.hypot(q.x-p.x,q.y-p.y) }))
      .filter(v => v.j !== i).sort((a,b) => a.d-b.d).slice(0,2)
      .forEach(({ j }) => {
        const k = Math.min(i,j)+'-'+Math.max(i,j);
        if (!edgeSet.has(k)) { edgeSet.add(k); edges.push([i,j]); }
      });
  });

  const phases = mapped.map(() => Math.random() * Math.PI * 2);
  const baseColor = isEnemy ? '#e05555' : card.color;
  let t = 0, rafId;

  function frame() {
    if (!canvas.isConnected) { cancelAnimationFrame(rafId); cardAnimations.delete(canvas); return; }
    t += 0.018;
    ctx.clearRect(0,0,W,H);
    edges.forEach(([a,b]) => {
      const pulse = 0.15 + 0.08*Math.sin(t+(a+b)*0.5);
      ctx.beginPath();
      ctx.strokeStyle = baseColor + hexA(pulse * 255);
      ctx.lineWidth = 0.6;
      ctx.moveTo(mapped[a].x, mapped[a].y);
      ctx.lineTo(mapped[b].x, mapped[b].y);
      ctx.stroke();
    });
    mapped.forEach((p,i) => {
      const tw = 0.55 + 0.45*Math.sin(t*0.95+phases[i]);
      const r  = 1.2 + tw*0.5;
      ctx.beginPath(); ctx.arc(p.x,p.y,r*2.5,0,Math.PI*2);
      ctx.fillStyle = baseColor + hexA(tw*28); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fillStyle = baseColor + 'bb'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,0.7,0,Math.PI*2);
      ctx.fillStyle = 'rgba(255,255,255,.9)'; ctx.fill();
    });
    rafId = requestAnimationFrame(frame);
    cardAnimations.set(canvas, rafId);
  }
  if (cardAnimations.has(canvas)) cancelAnimationFrame(cardAnimations.get(canvas));
  frame();
}

// ═══════════════════════════════════════════════════════════════
// HAND PICKER MODAL
// ═══════════════════════════════════════════════════════════════
function openHandPicker() {
  const modal = el('hand-picker'), list = el('hand-picker-list');
  list.innerHTML = '';
  S.playerHand.forEach(card => {
    const isStaged = !!S.playerPlayed.find(c => c.id === card.id);
    const row = document.createElement('div');
    row.className = 'picker-row' + (isStaged ? ' picker-staged' : '');
    row.dataset.id = card.id;
    const layerCol = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
    const mechanic = card.layer === 'Superego'
      ? `Shield +${card.shieldVal||0}` + (card.capacityVal ? ` · Cap ${card.capacityVal>0?'+':''}${card.capacityVal}` : '')
      : card.layer === 'Ego'
        ? [(card.chunkFlat?`+${card.chunkFlat} flat`:''),(card.chunkPct?`×${card.chunkPct}`:''),(card.studyMult?`Study ×${card.studyMult}`:'')].filter(Boolean).join(' ')
        : `Recharge +${card.rechargeVal||0}/stack`;

    row.innerHTML = `
      <div class="picker-check${isStaged?' checked':''}" id="chk-${card.id}">
        <svg viewBox="0 0 12 12" fill="none"><polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </div>
      <canvas class="picker-mini-cvs" data-card="${card.id}" width="44" height="44"></canvas>
      <div class="picker-info">
        <div class="picker-name" style="color:${card.color}">${card.name}</div>
        <div class="picker-layer" style="color:${layerCol}">${cardLabel(card)}</div>
        <div class="picker-mechanic">${mechanic}</div>
        <div class="picker-keywords">${card.keywords||''}</div>
      </div>
      <div class="picker-axium" style="color:${card.color}">⬡${card.axiumScore||'?'}</div>
    `;
    row.addEventListener('click', () => toggleStageCard(card, row));
    list.appendChild(row);
    setTimeout(() => { const cvs = row.querySelector('.picker-mini-cvs'); if (cvs) animatePickerCanvas(cvs, card); }, 40);
  });
  el('picker-count').textContent = `${S.playerPlayed.length}/${MAX_STAGED}`;
  modal.classList.add('show');
}

function animatePickerCanvas(canvas, card) {
  if (!canvas) return;
  const W = 44, H = 44; canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts || !pts.length) return;
  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=3,sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc,oY=H/2-(mnY+mxY)/2*sc;
  const mapped=pts.map(([x,y])=>({x:x*sc+oX,y:y*sc+oY}));
  const edgeSet=new Set(),edges=[];
  mapped.forEach((p,i)=>{mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)})).filter(v=>v.j!==i).sort((a,b)=>a.d-b.d).slice(0,2).forEach(({j})=>{const k=Math.min(i,j)+'-'+Math.max(i,j);if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);}});});
  const phases=mapped.map(()=>Math.random()*Math.PI*2);
  let t=0,rafId; const key='picker-'+card.id;
  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(rafId);cardAnimations.delete(key);return;}
    t+=0.02; ctx.clearRect(0,0,W,H);
    edges.forEach(([a,b])=>{const p=0.18+0.08*Math.sin(t+(a+b)*0.5);ctx.beginPath();ctx.strokeStyle=card.color+hexA(p*180);ctx.lineWidth=0.7;ctx.moveTo(mapped[a].x,mapped[a].y);ctx.lineTo(mapped[b].x,mapped[b].y);ctx.stroke();});
    mapped.forEach((p,i)=>{const tw=0.5+0.5*Math.sin(t+phases[i]);const r=1+tw*0.6;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=card.color+'cc';ctx.fill();ctx.beginPath();ctx.arc(p.x,p.y,0.5,0,Math.PI*2);ctx.fillStyle='rgba(255,255,255,.85)';ctx.fill();});
    rafId=requestAnimationFrame(frame); cardAnimations.set(key,rafId);
  }
  if(cardAnimations.has(key)) cancelAnimationFrame(cardAnimations.get(key));
  frame();
}

function toggleStageCard(card, rowEl) {
  const already = S.playerPlayed.findIndex(c => c.id === card.id);
  const attnFor100 = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const state = getAttnState(attnFor100);
  const limit = state.debuff?.maxActiveCards || MAX_STAGED;
  if (already >= 0) {
    S.playerPlayed.splice(already, 1);
    rowEl.classList.remove('picker-staged');
    rowEl.querySelector('.picker-check')?.classList.remove('checked');
  } else {
    if (S.playerPlayed.length >= limit) { toast('Limit Reached', `Max ${limit} cards in this state`); return; }
    if (S.playerPlayed.length >= MAX_STAGED) { toast('Hand Full', 'Maximum 10 cards per battle'); return; }
    S.playerPlayed.push(card);
    rowEl.classList.add('picker-staged');
    rowEl.querySelector('.picker-check')?.classList.add('checked');
  }
  el('picker-count').textContent = `${S.playerPlayed.length}/${MAX_STAGED}`;
  updatePhaseUI(); renderField();
}

function closeHandPicker() { el('hand-picker').classList.remove('show'); }

// ═══════════════════════════════════════════════════════════════
// FIELD RENDER  — chip rows (player bottom, enemy top)
// ═══════════════════════════════════════════════════════════════
function renderField() {
  // Player chips (bottom field)
  const pField = el('field-player'); if (!pField) return; pField.innerHTML = '';
  if (S.playerPlayed.length === 0) {
    pField.innerHTML = '<div class="field-empty">— no cards staged —</div>';
  } else {
    S.playerPlayed.forEach((card, i) => {
      const chip = document.createElement('div');
      chip.className = 'field-chip';
      chip.style.borderColor = card.color + '55';
      chip.style.color = card.color;
      const icon = card.layer === 'Superego' ? '◈' : card.layer === 'Ego' ? '◇' : '○';
      chip.innerHTML = `<span class="chip-icon">${icon}</span><span class="chip-name">${card.name}</span><span class="chip-remove" title="Remove">×</span>`;
      chip.querySelector('.chip-remove').addEventListener('click', e => { e.stopPropagation(); unstageCard(i); });
      pField.appendChild(chip);
    });
  }

  // Enemy chips (top field)
  const eField = el('field-enemy'); if (!eField) return; eField.innerHTML = '';
  if (!S.enemyHand || S.enemyHand.length === 0) {
    eField.innerHTML = '<div class="field-empty enemy-empty">— enemy awaiting —</div>';
  } else {
    S.enemyHand.forEach(card => {
      const chip = document.createElement('div');
      chip.className = 'field-chip enemy-chip';
      chip.style.borderColor = 'rgba(224,85,85,.4)';
      chip.style.color = 'rgba(224,85,85,.8)';
      chip.innerHTML = `<span class="chip-icon">✦</span><span class="chip-name">${card.name}</span>`;
      eField.appendChild(chip);
    });
  }
  updatePhaseUI();
}

function unstageCard(idx) {
  const card = S.playerPlayed[idx]; if (!card) return;
  S.playerPlayed.splice(idx, 1);
  const row = el('hand-picker')?.querySelector(`[data-id="${card.id}"]`);
  if (row) { row.classList.remove('picker-staged'); row.querySelector('.picker-check')?.classList.remove('checked'); }
  el('picker-count').textContent = `${S.playerPlayed.length}/${MAX_STAGED}`;
  log(`${card.name} removed`, 'sys');
  renderField();
}

// ═══════════════════════════════════════════════════════════════
// CARD DETAIL MODAL
// ═══════════════════════════════════════════════════════════════
function openModal(card) {
  S.modalCard = card;
  el('modal-axium').textContent = cardLabel(card); el('modal-axium').style.color = card.color;
  el('modal-name').textContent  = card.name; el('modal-name').style.color = card.color;
  el('modal-liturgy').textContent = card.keywords || '';
  let effectText = '';
  if (card.layer === 'Superego') effectText = (card.shieldDesc||'') + (card.capacityDesc ? '\n\n'+card.capacityDesc : '');
  else if (card.layer === 'Ego') effectText = (card.chunkDesc||'') + (card.divideDesc ? '\n\nReversed: '+card.divideDesc : '');
  else if (card.layer === 'ID')  effectText = (card.rechargeDesc||card.drainDesc||'');
  else effectText = card.effectDesc || '';
  el('modal-effect').textContent = effectText;
  el('modal-type').textContent    = card.type || '—';
  el('modal-int').textContent     = card.axiumScore !== undefined ? card.axiumScore+' / 10' : '—';
  let shiftEst = card.layer === 'Superego' ? card.shieldVal||0 : card.layer === 'ID' ? card.rechargeVal||0 : card.chunkFlat||0;
  el('modal-shift').textContent = (shiftEst>=0?'+':'')+shiftEst;
  el('modal-shift').style.color = shiftEst >= 0 ? '#86EFAC' : '#e05555';
  el('modal-exhaust').textContent = card.tier ? `Tier ${card.tier}` : '—';
  const myIds = S.playerPlayed.map(c => c.id).concat(card.id);
  const active = SYNERGIES.filter(s => s.cards.includes(card.id) && s.cards.every(id => myIds.includes(id)));
  const potent = SYNERGIES.filter(s => s.cards.includes(card.id) && !active.includes(s));
  let synTxt = '';
  if (active.length) synTxt += '✦ '+active.map(s=>s.name).join(' · ')+' (ready!) ';
  if (potent.length) synTxt += (active.length?'  ':'')+'Needs: '+potent.map(s=>s.cards.filter(id=>id!==card.id).join('+')).join(' / ');
  el('modal-synergies').textContent = synTxt || (card.synergies ? card.synergies.join(' · ') : '');
  el('card-modal').classList.add('show');
}

function closeModal(e) {
  if (e && e.target !== el('card-modal') && !String(e.target.id).includes('modal-close')) return;
  el('card-modal').classList.remove('show'); S.modalCard = null;
}

// ═══════════════════════════════════════════════════════════════
// RESOLVE ENTRY POINT
// ═══════════════════════════════════════════════════════════════
function resolveRound() {
  if (S.phase !== 'build') return;
  if (!S.playerPlayed.length) { toast('No Cards', 'Stage at least one card'); return; }
  S.phase = 'battling';
  closeHandPicker();
  el('resolve-btn').classList.remove('show');
  el('pass-btn').disabled = true;
  setTimeout(() => startBattleSequence(), 300);
}

// ═══════════════════════════════════════════════════════════════
// BATTLE SEQUENCE v6.0
// ─────────────────────────────────────────────────────────────
// Layout:
//   TOP:    enemy attention bar + enemy field row (reversed cards)
//   CENTER: VS divider
//   BOTTOM: player field row + player attention bar
//
// Sequence:
//   1. Flash overlay briefly to signal start
//   2. Player cards activate one by one:
//      - card pulses gold
//      - particle beam fires to player bar (gain) or enemy bar (damage)
//      - bar animates, math floats
//   3. Enemy cards activate one by one:
//      - card pulses red
//      - beam fires to player bar (drain)
//      - bar animates, math floats
//   4. Synergies fire
//   5. Resolve
// ═══════════════════════════════════════════════════════════════
function startBattleSequence() {
  const DELAY = CHAPTER_CONFIG.cardDelay || 1100;

  // Build the battle overlay
  const overlay = document.createElement('div');
  overlay.id = 'battle-overlay';
  overlay.style.cssText = `
    position:fixed;inset:0;z-index:400;
    background:rgba(3,3,14,0.96);
    display:flex;flex-direction:column;
    opacity:0;transition:opacity .4s;
    overflow:hidden;
  `;

  const attnFor100P = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const attnFor100E = clamp(S.enemyAttn  / S.maxAttn * 100, 0, 100);
  const psP = getAttnState(attnFor100P);
  const psE = getAttnState(attnFor100E);
  const pPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const ePct = clamp(S.enemyAttn  / S.maxAttn * 100, 0, 100);

  overlay.innerHTML = `
    <!-- ── ENEMY BAR (top) ── -->
    <div id="ovr-enemy-bar-wrap" style="
      flex-shrink:0;padding:14px 18px 10px;
      background:linear-gradient(180deg,rgba(3,3,14,.98) 0%,rgba(3,3,14,.85) 100%);
      border-bottom:1px solid rgba(224,85,85,.12);
    ">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
        <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:rgba(224,85,85,.5);">Enemy Attention</span>
        <div style="display:flex;align-items:baseline;gap:8px;">
          <span id="ovr-enemy-val" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(224,85,85,.6);">${Math.round(S.enemyAttn)} / ${S.maxAttn}</span>
          <span id="ovr-enemy-state" style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:16px;color:${psE.col};">${psE.label}</span>
        </div>
      </div>
      <div id="ovr-enemy-track" style="height:7px;background:rgba(255,255,255,.06);border-radius:4px;position:relative;border:1px solid rgba(255,255,255,.06);">
        <div id="ovr-enemy-fill" style="height:100%;border-radius:4px;width:${ePct}%;background:linear-gradient(90deg,rgba(20,20,20,.5) 0%,rgba(107,33,168,.7) 20%,rgba(220,38,38,.85) 55%,rgba(249,115,22,.9) 80%,rgba(255,200,50,1) 100%);transition:width .55s cubic-bezier(.22,1,.36,1);box-shadow:0 0 10px rgba(224,85,85,.3);"></div>
        <div id="ovr-enemy-cursor" style="position:absolute;top:-5px;left:${ePct}%;width:14px;height:14px;border-radius:50%;background:white;transform:translateX(-50%);transition:left .55s cubic-bezier(.22,1,.36,1);border:2px solid #e05555;box-shadow:0 0 10px #e05555;"></div>
      </div>
    </div>

    <!-- ── ENEMY FIELD ── -->
    <div id="ovr-enemy-field" style="
      flex-shrink:0;display:flex;align-items:flex-end;justify-content:center;
      gap:8px;padding:12px 10px 8px;flex-wrap:wrap;min-height:90px;
    "></div>

    <!-- ── VS DIVIDER ── -->
    <div style="
      display:flex;align-items:center;gap:10px;
      padding:0 18px;flex-shrink:0;
    ">
      <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent);"></div>
      <span id="ovr-phase-lbl" style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.24em;text-transform:uppercase;color:rgba(255,255,255,.18);flex-shrink:0;">vs</span>
      <div style="flex:1;height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,.07),transparent);"></div>
    </div>

    <!-- ── PLAYER FIELD ── -->
    <div id="ovr-player-field" style="
      flex-shrink:0;display:flex;align-items:flex-start;justify-content:center;
      gap:8px;padding:8px 10px 12px;flex-wrap:wrap;min-height:90px;
    "></div>

    <!-- ── PLAYER BAR (bottom) ── -->
    <div id="ovr-player-bar-wrap" style="
      flex-shrink:0;padding:10px 18px 16px;
      background:linear-gradient(0deg,rgba(3,3,14,.98) 0%,rgba(3,3,14,.85) 100%);
      border-top:1px solid rgba(212,175,55,.12);
    ">
      <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:5px;">
        <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:rgba(212,175,55,.5);">Your Attention</span>
        <div style="display:flex;align-items:baseline;gap:8px;">
          <span id="ovr-player-val" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(212,175,55,.65);">${Math.round(S.playerAttn)} / ${S.maxAttn}</span>
          <span id="ovr-player-state" style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:16px;color:${psP.col};">${psP.label}</span>
        </div>
      </div>
      <div id="ovr-player-track" style="height:7px;background:rgba(255,255,255,.06);border-radius:4px;position:relative;border:1px solid rgba(255,255,255,.06);">
        <div id="ovr-player-fill" style="height:100%;border-radius:4px;width:${pPct}%;background:linear-gradient(90deg,rgba(107,33,168,.7) 0%,rgba(29,78,216,.7) 18%,rgba(55,65,81,.7) 35%,rgba(126,184,232,.8) 52%,rgba(212,175,55,.9) 70%,rgba(134,239,172,.9) 85%,rgba(255,255,255,1) 100%);transition:width .55s cubic-bezier(.22,1,.36,1);box-shadow:0 0 10px rgba(212,175,55,.3);"></div>
        <div id="ovr-player-cursor" style="position:absolute;top:-5px;left:${pPct}%;width:14px;height:14px;border-radius:50%;background:white;transform:translateX(-50%);transition:left .55s cubic-bezier(.22,1,.36,1);border:2px solid #D4AF37;box-shadow:0 0 10px #D4AF37;"></div>
      </div>
    </div>

    <!-- ── BATTLE LOG ── -->
    <div style="position:absolute;bottom:100px;left:10px;width:min(200px,38vw);z-index:10;pointer-events:none;">
      <div id="ovr-log" style="display:flex;flex-direction:column;gap:3px;"></div>
    </div>
  `;

  document.body.appendChild(overlay);
  requestAnimationFrame(() => { overlay.style.opacity = '1'; });

  // ── Populate fields with cards ──
  const pFieldEl = overlay.querySelector('#ovr-player-field');
  const eFieldEl = overlay.querySelector('#ovr-enemy-field');

  // Render enemy cards reversed, facing down toward player
  const enemyCardEls = S.enemyHand.map(card => {
    const cardEl = makeCard(card, ['field-card', 'enemy-placed']);
    cardEl.style.cssText += ';width:62px;opacity:0.75;transition:opacity .3s,box-shadow .3s,transform .3s;cursor:default;';
    eFieldEl.appendChild(cardEl);
    return { el: cardEl, card };
  });

  // Render player cards upright
  const playerCardEls = S.playerPlayed.map(card => {
    const cardEl = makeCard(card, ['field-card', 'player-placed']);
    cardEl.style.cssText += ';width:62px;transition:opacity .3s,box-shadow .3s,transform .3s;cursor:default;';
    pFieldEl.appendChild(cardEl);
    return { el: cardEl, card };
  });

  function ovrAddLog(msg, type) {
    const logEl = overlay.querySelector('#ovr-log'); if (!logEl) return;
    const line  = document.createElement('div');
    line.style.cssText = `font-family:'Space Mono',monospace;font-size:8px;letter-spacing:.05em;line-height:1.5;color:${
      type==='p'?'rgba(212,175,55,.8)':type==='drain'?'rgba(224,85,85,.75)':type==='syn'?'rgba(212,175,55,.9)':'rgba(255,255,255,.3)'};animation:logSlide .2s ease;`;
    line.textContent = msg;
    logEl.appendChild(line);
    while (logEl.children.length > 12) logEl.removeChild(logEl.firstChild);
  }

  // Live bar updaters inside overlay
  function updateOvrBars() {
    const pPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
    const ePct = clamp(S.enemyAttn  / S.maxAttn * 100, 0, 100);
    const psP  = getAttnState(pPct);
    const psE  = getAttnState(ePct);

    const pFill = overlay.querySelector('#ovr-player-fill');
    const pCur  = overlay.querySelector('#ovr-player-cursor');
    const pVal  = overlay.querySelector('#ovr-player-val');
    const pSt   = overlay.querySelector('#ovr-player-state');
    if (pFill)  pFill.style.width  = pPct + '%';
    if (pCur)   pCur.style.left    = pPct + '%';
    if (pVal)   pVal.textContent   = `${Math.round(S.playerAttn)} / ${S.maxAttn}`;
    if (pSt)  { pSt.textContent    = psP.label; pSt.style.color = psP.col; }

    const eFill = overlay.querySelector('#ovr-enemy-fill');
    const eCur  = overlay.querySelector('#ovr-enemy-cursor');
    const eVal  = overlay.querySelector('#ovr-enemy-val');
    const eSt   = overlay.querySelector('#ovr-enemy-state');
    if (eFill)  eFill.style.width  = ePct + '%';
    if (eCur)   eCur.style.left    = ePct + '%';
    if (eVal)   eVal.textContent   = `${Math.round(S.enemyAttn)} / ${S.maxAttn}`;
    if (eSt)  { eSt.textContent    = psE.label; eSt.style.color = psE.col; }
  }

  // Float math relative to overlay bar elements
  function floatOnBar(barWrapId, text, color) {
    const barWrap = overlay.querySelector(`#${barWrapId}`); if (!barWrap) return;
    const rect    = barWrap.getBoundingClientRect();
    const div     = document.createElement('div');
    div.style.cssText = `
      position:fixed;
      left:${rect.left + rect.width * 0.5}px;
      top:${rect.top + rect.height * 0.5}px;
      transform:translateX(-50%);
      font-family:'JetBrains Mono',monospace;
      font-size:clamp(12px,2.4vw,16px);font-weight:700;
      color:${color};text-shadow:0 0 14px ${color};
      pointer-events:none;z-index:9999;white-space:nowrap;
      opacity:1;transition:transform .9s cubic-bezier(.22,1,.36,1),opacity .9s ease;
    `;
    div.textContent = text;
    document.body.appendChild(div);
    requestAnimationFrame(() => {
      div.style.transform = 'translateX(-50%) translateY(-44px)';
      div.style.opacity   = '0';
    });
    setTimeout(() => div.remove(), 1000);
  }

  const phaseLbl = overlay.querySelector('#ovr-phase-lbl');
  function setPhase(txt) { if (phaseLbl) phaseLbl.textContent = txt; }

  // ── CALCULATE CARD EFFECTS ──────────────────────────────
  function getPlayerCardEffect(card) {
    // Upright player card: gain to player
    let gain = 0;
    if (card.layer === 'Superego') gain = card.shieldVal || 0;
    if (card.layer === 'ID')       gain = (card.rechargeVal || 0) * 3; // 3 stacks worth
    if (card.layer === 'Ego')      gain = card.chunkFlat || 0;
    // Also deal some attention damage to enemy from strong cards
    const enemyHit = card.layer === 'Superego'
      ? Math.round((card.shieldVal || 0) * 0.25)
      : card.layer === 'ID'
        ? (card.rechargeVal || 0)
        : 0;
    return { playerGain: gain, enemyHit };
  }

  function getEnemyCardEffect(card) {
    // Reversed card: drains player
    return { playerDrain: card.drainVal || 8 };
  }

  // ── SEQUENCE RUNNER ─────────────────────────────────────
  let seqIndex = 0;
  // interleave: player[0], enemy[0], player[1], enemy[1], ...
  const allCards = [];
  const maxLen = Math.max(playerCardEls.length, enemyCardEls.length);
  for (let i = 0; i < maxLen; i++) {
    if (i < playerCardEls.length) allCards.push({ type: 'player', ...playerCardEls[i] });
    if (i < enemyCardEls.length)  allCards.push({ type: 'enemy',  ...enemyCardEls[i] });
  }

  function activateNext() {
    if (seqIndex >= allCards.length) {
      // All cards done — check synergies then finalise
      setTimeout(() => runSynergies(() => endBattle(overlay)), 600);
      return;
    }

    const item = allCards[seqIndex++];
    const isPlayer = item.type === 'player';
    const cardEl   = item.el;
    const card     = item.card;

    // Pulse the card
    const pulseColor = isPlayer ? '#D4AF37' : '#e05555';
    pulseCard(cardEl, pulseColor);

    // Determine what bar to hit and what values to show
    let beamTargetId, barWrapId, mathText, mathColor, deltaApply;

    if (isPlayer) {
      const fx = getPlayerCardEffect(card);
      // Primary: beam → player bar (gain)
      beamTargetId = 'ovr-player-track';
      barWrapId    = 'ovr-player-bar-wrap';
      const gainLabel = card.layer === 'Superego' ? `+${fx.playerGain} Shield` : card.layer === 'ID' ? `+${fx.playerGain} Recharge` : `+${fx.playerGain} Chunk`;
      mathText  = gainLabel;
      mathColor = '#D4AF37';
      deltaApply = () => {
        const prev = S.playerAttn;
        S.playerAttn = clamp(S.playerAttn + fx.playerGain, MIN_ATTN, S.maxAttn);
        updateOvrBars();
        ovrAddLog(`${card.name}: ${gainLabel}`, 'p');
        // Secondary beam to enemy if we hit them
        if (fx.enemyHit > 0) {
          setTimeout(() => {
            pulseCard(cardEl, '#86EFAC');
            fireBeam(cardEl, 'ovr-enemy-track', '#86EFAC', () => {
              S.enemyAttn = clamp(S.enemyAttn - fx.enemyHit, MIN_ATTN, S.maxAttn);
              updateOvrBars();
              floatOnBar('ovr-enemy-bar-wrap', `-${fx.enemyHit}`, '#86EFAC');
              ovrAddLog(`${card.name}: enemy −${fx.enemyHit}`, 'p');
            });
          }, 250);
        }
      };
      setPhase(gainLabel);
    } else {
      // Enemy reversed card drains player
      const fx = getEnemyCardEffect(card);
      beamTargetId = 'ovr-player-track';
      barWrapId    = 'ovr-player-bar-wrap';
      mathText     = `-${fx.playerDrain} Drain`;
      mathColor    = '#e05555';
      deltaApply   = () => {
        S.playerAttn = clamp(S.playerAttn - fx.playerDrain, MIN_ATTN, S.maxAttn);
        updateOvrBars();
        floatOnBar('ovr-player-bar-wrap', `-${fx.playerDrain}`, '#e05555');
        ovrAddLog(`${card.name} reversed: −${fx.playerDrain} drain`, 'drain');
      };
      setPhase(`${card.name} reversed — Drain ${fx.playerDrain}`);
    }

    // Fire beam from card to bar
    const barTrackEl = overlay.querySelector(`#${beamTargetId}`);
    fireBeam(cardEl, beamTargetId, isPlayer ? (card.color || '#D4AF37') : '#e05555', () => {
      deltaApply();
      if (isPlayer) floatOnBar(barWrapId, mathText, mathColor);
    });

    setTimeout(activateNext, DELAY);
  }

  // Kick off sequence after short intro pause
  setPhase('Constellations align...');
  setTimeout(activateNext, 700);
}

// ═══════════════════════════════════════════════════════════════
// SYNERGY RESOLUTION
// ═══════════════════════════════════════════════════════════════
function runSynergies(onDone) {
  const playerIds  = S.playerPlayed.map(c => c.id);
  const activeSyns = getSynergies(playerIds);
  if (!activeSyns.length) { onDone(); return; }

  let i = 0;
  function nextSyn() {
    if (i >= activeSyns.length) { onDone(); return; }
    const syn = activeSyns[i++];
    flashSynergy(syn);
    // Synergy gives a bonus attention bump
    const bonus = 8 + (syn.rare ? 12 : 0);
    S.playerAttn = clamp(S.playerAttn + bonus, MIN_ATTN, S.maxAttn);
    log(`✦ Synergy: ${syn.name} +${bonus}`, 'synerg');
    setTimeout(nextSyn, 2200);
  }
  nextSyn();
}

// ═══════════════════════════════════════════════════════════════
// END BATTLE — tear down overlay, finalise
// ═══════════════════════════════════════════════════════════════
function endBattle(overlay) {
  overlay.style.opacity = '0';
  overlay.style.transition = 'opacity .5s';
  setTimeout(() => {
    overlay.remove();
    updateBars();
    finaliseBattle();
  }, 520);
}

// ═══════════════════════════════════════════════════════════════
// FINALISE
// ═══════════════════════════════════════════════════════════════
function finaliseBattle() {
  updateBars();
  const pop   = el('resolve-pop');
  const state = getAttnState(clamp(S.playerAttn / S.maxAttn * 100, 0, 100));
  if (pop) { pop.textContent = state.label; pop.style.color = state.col; pop.classList.add('show'); }
  burst(window.innerWidth / 2, window.innerHeight * 0.5, state.col, 22);
  setTimeout(() => { if (pop) pop.classList.remove('show'); checkWinLose(); }, 1100);
}

// ═══════════════════════════════════════════════════════════════
// WIN / LOSE / NEXT TURN
// ═══════════════════════════════════════════════════════════════
function checkWinLose() {
  if (S.playerAttn <= CHAPTER_CONFIG.loseAttn) { triggerLose(); return; }
  if (S.playerAttn >= CHAPTER_CONFIG.winAttn)  { triggerWin();  return; }
  if (S.playerPlayed.length >= 10) {
    const avg = S.playerPlayed.reduce((s, c) => s + (c.axiumScore || 0), 0) / S.playerPlayed.length;
    if (avg >= 10) { triggerWin(true); return; }
  }
  setTimeout(() => nextTurn(), 600);
}

function triggerWin(perfect = false) {
  S.won = true; S.phase = 'done';
  el('out-title').textContent = perfect ? 'The Axium' : CHAPTER_CONFIG.winTitle;
  el('out-title').style.color = '#D4AF37';
  el('out-title').style.textShadow = '0 0 50px rgba(212,175,55,.4)';
  el('out-desc').textContent  = perfect ? 'Perfect constellation. The final boss awakens.' : CHAPTER_CONFIG.winDesc;
  el('out-btn').textContent   = CHAPTER_CONFIG.winBtn;
  el('out-btn').style.cssText = 'background:linear-gradient(135deg,#AA8C2C,#D4AF37,#AA8C2C);color:#0a0a0a;border:none;padding:11px 34px;cursor:pointer;border-radius:2px;';
  setTimeout(() => el('outcome').classList.add('show'), 600);
  burst(window.innerWidth/2, window.innerHeight/2, '#D4AF37', 44);
  setTimeout(() => burst(window.innerWidth/2, window.innerHeight/2, '#86EFAC', 28), 500);
}

function triggerLose() {
  S.lost = true; S.phase = 'done';
  el('out-title').textContent = CHAPTER_CONFIG.loseTitle;
  el('out-title').style.color = '#e05555';
  el('out-title').style.textShadow = '0 0 50px rgba(224,85,85,.4)';
  el('out-desc').textContent  = CHAPTER_CONFIG.loseDesc;
  el('out-btn').textContent   = CHAPTER_CONFIG.loseBtn;
  el('out-btn').style.cssText = 'background:rgba(224,85,85,.08);color:#e05555;border:1px solid rgba(224,85,85,.3);padding:11px 34px;cursor:pointer;border-radius:2px;';
  setTimeout(() => el('outcome').classList.add('show'), 400);
  burst(window.innerWidth/2, window.innerHeight/2, '#DC2626', 28);
}

function handleOutcome() { el('outcome').classList.remove('show'); if (S.won) openShop(); else resetChapter(); }

function nextTurn() {
  S.turn++; S.phase = 'build'; S.playerPlayed = [];
  buildEnemyHand();  // new reversed hand each turn
  const tn = el('turn-n'); if (tn) tn.textContent = S.turn;
  const pb = el('pass-btn'); if (pb) pb.disabled = false;
  recalcMaxAttn(); dealHand(); renderField();
  log(`── Turn ${S.turn} ──`, 'sys');
}

function passAndEndTurn() {
  if (S.phase !== 'build') return;
  S.playerAttn = clamp(S.playerAttn - 6, MIN_ATTN, S.maxAttn);
  updateBars(); log('Passed — attention −6', 'sys');
  S.playerPlayed = [];
  S.phase = 'battling';
  const pb = el('pass-btn'); if (pb) pb.disabled = true;
  const rb = el('resolve-btn'); if (rb) rb.classList.remove('show');
  setTimeout(() => checkWinLose(), 900);
}

function resetChapter() {
  S.playerDeck  = shuffle([...PLAYER_CARDS]);
  S.playerAttn  = CHAPTER_CONFIG.playerStart;
  S.enemyAttn   = CHAPTER_CONFIG.enemyStart;
  S.turn = 1; S.phase = 'build';
  S.playerHand = []; S.playerPlayed = [];
  S.won = false; S.lost = false;
  buildEnemyHand();
  recalcMaxAttn(); updateBars();
  const tn = el('turn-n'); if (tn) tn.textContent = '1';
  const pb = el('pass-btn'); if (pb) pb.disabled = false;
  const rb = el('resolve-btn'); if (rb) rb.classList.remove('show');
  const lg = el('log'); if (lg) lg.innerHTML = '';
  renderField(); dealHand(); log('── Chapter reset ──', 'sys');
}

// ═══════════════════════════════════════════════════════════════
// SHOP
// ═══════════════════════════════════════════════════════════════
function openShop() {
  const npc    = getShopNPC(CHAPTER_CONFIG.shopChapter || CURRENT_CHAPTER);
  const offers = getShopOffers(CHAPTER_CONFIG.shopChapter || CURRENT_CHAPTER, 3).filter(Boolean);
  el('sh-npc-name').textContent  = npc.name;
  el('sh-npc-role').textContent  = npc.role;
  el('sh-speech').textContent    = npc.speeches[Math.floor(Math.random() * npc.speeches.length)];
  const cardsEl = el('sh-cards'); cardsEl.innerHTML = '';
  offers.forEach(card => {
    if (!card) return;
    const wrap = document.createElement('div'); wrap.className = 'sh-card-wrap';
    wrap.appendChild(makeCard(card, []));
    const desc = card.chunkDesc||card.shieldDesc||card.rechargeDesc||card.keywords||'';
    const lbl  = document.createElement('div'); lbl.className = 'sh-upgrade-lbl';
    lbl.textContent = desc.length > 52 ? desc.slice(0,52)+'…' : desc;
    wrap.appendChild(lbl);
    wrap.addEventListener('click', () => {
      if (!S.playerDeck.find(c => c.id === card.id)) { S.playerDeck.push(card); toast('Added', card.name); }
      el('shop').classList.remove('show'); resetChapter();
    });
    cardsEl.appendChild(wrap);
  });
  el('shop').classList.add('show');
}

function skipShop() { el('shop').classList.remove('show'); resetChapter(); }

// ═══════════════════════════════════════════════════════════════
// SYNERGY FLASH
// ═══════════════════════════════════════════════════════════════
function flashSynergy(syn) {
  const flash = el('synergy-flash'), msg = el('synergy-msg'); if (!flash || !msg) return;
  flash.style.background = `radial-gradient(ellipse at center,${syn.visual||'#D4AF37'}22 0%,transparent 70%)`;
  el('synergy-msg-name').textContent = syn.name;
  el('synergy-msg-desc').textContent = syn.desc || '';
  el('synergy-msg-name').style.color = syn.visual || '#D4AF37';
  flash.classList.add('show'); msg.classList.add('show');
  setTimeout(() => { flash.classList.remove('show'); msg.classList.remove('show'); }, 2100);
}

// ═══════════════════════════════════════════════════════════════
// DEAL HAND
// ═══════════════════════════════════════════════════════════════
function dealHand() {
  const attnFor100 = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const state      = getAttnState(attnFor100);
  const drawMod    = state.debuff?.drawMod || 0;
  const size       = clamp(STARTING_HAND + drawMod, 1, STARTING_HAND);
  const available  = S.playerDeck.filter(c => !S.playerPlayed.find(p => p.id === c.id));
  S.playerHand     = shuffle(available).slice(0, size);
  S.playerPlayed.forEach(c => { if (!S.playerHand.find(h => h.id === c.id)) S.playerHand.unshift(c); });
}

// ═══════════════════════════════════════════════════════════════
// UTILS
// ═══════════════════════════════════════════════════════════════
function el(id)            { return document.getElementById(id); }
function clamp(v, mn, mx)  { return Math.max(mn, Math.min(mx, v)); }

function log(msg, type = 'sys') {
  const logEl = el('log'); if (!logEl) return;
  const line  = document.createElement('div');
  line.className = `log-line ${type}`; line.textContent = msg;
  logEl.appendChild(line);
  while (logEl.children.length > 16) logEl.removeChild(logEl.firstChild);
}

let toastTimer;
function toast(h, b) {
  el('toast-h').textContent = h; el('toast-b').textContent = b;
  const t = el('toast'); t.classList.add('show'); clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

function burst(x, y, color, n) {
  for (let i = 0; i < n; i++) {
    const p    = document.createElement('div'); p.className = 'ptcl';
    const ang  = (Math.PI * 2 * i) / n + Math.random() * .5;
    const dist = 20 + Math.random() * 65;
    const dur  = 380 + Math.random() * 260;
    p.style.cssText = `left:${x}px;top:${y}px;width:${1.4+Math.random()*2.8}px;height:${1.4+Math.random()*2.8}px;background:${color};box-shadow:0 0 5px ${color};transition:transform ${dur}ms cubic-bezier(.22,1,.36,1),opacity ${dur}ms ease;position:fixed;border-radius:50%;pointer-events:none;z-index:900;`;
    document.body.appendChild(p);
    requestAnimationFrame(() => {
      p.style.transform = `translate(${Math.cos(ang)*dist}px,${Math.sin(ang)*dist}px) scale(0)`;
      p.style.opacity   = '0';
    });
    setTimeout(() => p.remove(), dur);
  }
}

// ── Stars ─────────────────────────────────────────────────────
(function () {
  const cv = el('stars'); if (!cv) return;
  const ctx = cv.getContext('2d');
  let W, H, stars = [];
  function resize() {
    W = cv.width = window.innerWidth; H = cv.height = window.innerHeight;
    stars = Array.from({ length: 140 }, () => ({
      x: Math.random()*W, y: Math.random()*H,
      r: .3+Math.random()*1.1, a: .07+Math.random()*.36,
      sp: .22+Math.random()*.55, ph: Math.random()*Math.PI*2
    }));
  }
  function draw(t) {
    requestAnimationFrame(draw);
    ctx.fillStyle = '#03030e'; ctx.fillRect(0,0,W,H);
    const g = ctx.createRadialGradient(W*.4,H*.35,0,W*.4,H*.35,W*.5);
    g.addColorStop(0,'rgba(40,20,70,.13)'); g.addColorStop(1,'rgba(3,3,14,0)');
    ctx.fillStyle = g; ctx.fillRect(0,0,W,H);
    stars.forEach(s => {
      const tw = .4+.6*Math.abs(Math.sin(t*.0008*s.sp+s.ph));
      ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle = `rgba(255,255,255,${s.a*tw})`; ctx.fill();
    });
  }
  resize(); window.addEventListener('resize', resize); requestAnimationFrame(draw);
})();

// ── Inject logSlide keyframe ──────────────────────────────────
(function () {
  if (!document.getElementById('game-keyframes')) {
    const s = document.createElement('style'); s.id = 'game-keyframes';
    s.textContent = '@keyframes logSlide{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}';
    document.head.appendChild(s);
  }
})();
