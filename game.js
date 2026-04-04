// ═══════════════════════════════════════════════════════════════
// AXIUM — GAME ENGINE v6.1
// Both sides draw from the same card pool (PLAYER_CARDS).
// CHAPTER_CONFIG.enemyDeck = array of card IDs the enemy uses.
// CHAPTER_CONFIG.enemyStart / enemyMax = enemy attention values.
// ═══════════════════════════════════════════════════════════════

const CHAPTER_CONFIG = window.CHAPTER_CONFIG || {
  id:           1,
  label:        '01 · Binding of Ego',
  title:        'Binding of Ego',
  axium:        '"The absence of love binds ego, allowing the awareness of God."',
  narrative:    "Build a constellation of cards. Stage them, then Resolve.",
  shopChapter:  1,
  playerStart:  100,
  baseMaxAttn:  100,
  handSize:     6,
  winAttn:      90,
  loseAttn:     10,
  enemyStart:   80,
  enemyMax:     100,
  enemyDeck:    ['fool','magician','high_priestess','empress','emperor','lovers'],
  enemyHandSize: 3,
  winTitle:     'Attention Held',
  winDesc:      'Your constellation held. The chapter is complete.',
  winBtn:       'Visit the Shop',
  loseTitle:    'Attention Lost',
  loseDesc:     'Your attention collapsed. Try again.',
  loseBtn:      'Try Again',
  battleTiming: { charge: 2000, nodeCycle: 500, done: 1800 },
};

const STARTING_HAND   = CHAPTER_CONFIG.handSize    || 6;
const ENEMY_HAND_SIZE = CHAPTER_CONFIG.enemyHandSize || 3;
const MAX_STAGED      = 10;
const MIN_ATTN        = 5;
const BASE_MAX_ATTN   = CHAPTER_CONFIG.baseMaxAttn  || 100;
const CURRENT_CHAPTER = CHAPTER_CONFIG.id           || 1;

// ── GAME STATE ───────────────────────────────────────────────
const S = {
  playerAttn:   CHAPTER_CONFIG.playerStart,
  maxAttn:      BASE_MAX_ATTN,
  enemyAttn:    CHAPTER_CONFIG.enemyStart || 80,
  enemyMaxAttn: CHAPTER_CONFIG.enemyMax   || 100,
  turn:         1,
  phase:        'build',
  playerDeck:   [],
  playerHand:   [],
  playerPlayed: [],
  enemyDeck:    [],
  enemyHand:    [],
  enemyPlayed:  [],
  modalCard:    null,
  won:          false,
  lost:         false,
};

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
function startChapter() {
  const introEl = el('intro');
  if (introEl) introEl.style.display = 'none';

  const gameEl = el('game');
  if (gameEl) gameEl.style.display = 'grid';

  const config = window.CHAPTER_CONFIG || CHAPTER_CONFIG;

  S.playerDeck   = shuffle([...PLAYER_CARDS]);
  S.playerAttn   = config.playerStart;

  const eDeckIds = config.enemyDeck || [];
  S.enemyDeck    = eDeckIds.map(id => PLAYER_CARDS.find(c => c.id === id)).filter(Boolean);

  S.enemyAttn    = config.enemyStart || 80;
  S.enemyMaxAttn = config.enemyMax   || 100;

  S.turn         = 1;
  S.phase        = 'build';
  S.playerPlayed = [];
  S.enemyPlayed  = [];
  S.playerHand   = [];

  recalcMaxAttn();
  initBars();
  dealHand();
  renderField();
  updatePhaseUI();

  log(`── Chapter Started: ${config.title} ──`, 'sys');
}

function recalcMaxAttn() {
  const cap = calculateCapacity(S.playerDeck);
  S.maxAttn = clamp(BASE_MAX_ATTN + cap, 20, 300);
}

// ═══════════════════════════════════════════════════════════════
// BARS
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
  const ppPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const pFill = el('player-fill'); if (pFill) pFill.style.width = ppPct + '%';
  const pCur  = el('player-cursor'); if (pCur) pCur.style.left = ppPct + '%';
  const ps    = getAttnState(ppPct);
  const pLbl  = el('player-state-lbl');
  if (pLbl) { pLbl.textContent = ps.label; pLbl.style.color = ps.col; }
  updatePips('player-pips', ppPct);

  const epPct = clamp(S.enemyAttn / S.enemyMaxAttn * 100, 0, 100);
  const eFill = el('enemy-fill'); if (eFill) eFill.style.width = epPct + '%';
  const eCur  = el('enemy-cursor'); if (eCur) eCur.style.left = epPct + '%';
  const es    = getAttnState(epPct);
  const eLbl  = el('enemy-state-lbl');
  if (eLbl) { eLbl.textContent = es.label; eLbl.style.color = es.col; }
  updatePips('enemy-pips', epPct);
}

function updatePips(id, pct) {
  document.querySelectorAll(`#${id} .bar-pip`).forEach((p, i) => {
    p.classList.toggle('lit', ATTN_STATES[i] && ATTN_STATES[i].pos <= pct / 100);
  });
}

function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

function shiftPlayer(delta) {
  S.playerAttn = clamp(S.playerAttn + delta, MIN_ATTN, S.maxAttn);
  updateBars();
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
    : n >= MAX_STAGED ? 'Hand full — Resolve!'
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
// SAFE HEX ALPHA
// ═══════════════════════════════════════════════════════════════
function hexA(val) {
  const v = Math.max(0, Math.min(255, Math.round(val)));
  return v.toString(16).padStart(2, '0');
}

// ═══════════════════════════════════════════════════════════════
// CARD RENDERING
// ═══════════════════════════════════════════════════════════════
const cardAnimations = new Map();

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
function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : ''; }
function numberToRoman(n) {
  const map = [[21,'XXI'],[20,'XX'],[19,'XIX'],[18,'XVIII'],[17,'XVII'],[16,'XVI'],[15,'XV'],
               [14,'XIV'],[13,'XIII'],[12,'XII'],[11,'XI'],[10,'X'],[9,'IX'],[8,'VIII'],
               [7,'VII'],[6,'VI'],[5,'V'],[4,'IV'],[3,'III'],[2,'II'],[1,'I'],[0,'0']];
  return (map.find(([v]) => v === n) || [n, String(n)])[1];
}

function makeCard(card, classes = []) {
  const isEnemy  = classes.includes('enemy-placed');
  const nameCol  = isEnemy ? 'rgba(220,120,120,.95)' : '#fff';
  const layerCol = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
  const el2 = document.createElement('div');
  el2.className = 'card ' + classes.join(' ');
  el2.dataset.id = card.id;
  const glow = document.createElement('div'); glow.className = 'card-glow';
  glow.style.background = `radial-gradient(ellipse at 35% 25%, ${card.color}44, transparent 65%)`;
  el2.appendChild(glow);
  const top = document.createElement('div'); top.className = 'card-top';
  top.innerHTML = `<div class="card-axium-lbl" style="color:${layerCol}">${cardLabel(card)}</div><div class="card-name" style="color:${nameCol}">${card.name}</div>`;
  el2.appendChild(top);
  const cvs = document.createElement('canvas'); cvs.className = 'card-cvs'; el2.appendChild(cvs);
  const cardType = card.type || 'compression';
  const axium    = card.axiumScore !== undefined ? card.axiumScore : '?';
  const bot = document.createElement('div'); bot.className = 'card-bot';
  bot.innerHTML = `<span class="card-type-pip ${cardType}">${cardType.slice(0, 4)}</span><span class="card-intensity">⬡${axium}</span>`;
  el2.appendChild(bot);
  setTimeout(() => animateCardCanvas(cvs, card), 20);
  return el2;
}

function animateCardCanvas(canvas, card) {
  if (!canvas || !canvas.parentElement) return;
  const W = canvas.offsetWidth || 70, H = canvas.offsetHeight || 50;
  if (W < 4 || H < 4) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts || !pts.length) return;
  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=4, sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc, oY=H/2-(mnY+mxY)/2*sc;
  const mapped=pts.map(([x,y])=>({x:x*sc+oX,y:y*sc+oY}));
  const edgeSet=new Set(), edges=[];
  mapped.forEach((p,i)=>{
    mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)}))
      .filter(v=>v.j!==i).sort((a,b)=>a.d-b.d).slice(0,2)
      .forEach(({j})=>{ const k=Math.min(i,j)+'-'+Math.max(i,j); if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);} });
  });
  const phases=mapped.map((_,i)=>i*0.8+Math.random()*Math.PI*2);
  let t=0, rafId;
  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(rafId);cardAnimations.delete(canvas);return;}
    t+=0.018; ctx.clearRect(0,0,W,H);
    edges.forEach(([a,b])=>{
      const pulse=0.15+0.08*Math.sin(t+(a+b)*0.5);
      ctx.beginPath(); ctx.strokeStyle=card.color+hexA(pulse*255); ctx.lineWidth=0.6;
      ctx.moveTo(mapped[a].x,mapped[a].y); ctx.lineTo(mapped[b].x,mapped[b].y); ctx.stroke();
    });
    mapped.forEach((p,i)=>{
      const tw=0.55+0.45*Math.sin(t*0.95+phases[i]); const r=1.2+tw*0.5;
      ctx.beginPath(); ctx.arc(p.x,p.y,r*2.5,0,Math.PI*2); ctx.fillStyle=card.color+hexA(tw*28); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=card.color+'bb'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,0.7,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.9)'; ctx.fill();
    });
    rafId=requestAnimationFrame(frame); cardAnimations.set(canvas,rafId);
  }
  if(cardAnimations.has(canvas)) cancelAnimationFrame(cardAnimations.get(canvas));
  frame();
}

function animateModalCanvas(card) {
  const canvas = el('modal-canvas'); if (!canvas) return;
  const W = canvas.offsetWidth || 280, H = 140;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts; if (!pts) return;
  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=20, sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc, oY=H/2-(mnY+mxY)/2*sc;
  const mapped=pts.map(([x,y])=>({x:x*sc+oX,y:y*sc+oY}));
  const edgeSet=new Set(), edges=[];
  mapped.forEach((p,i)=>{
    mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)}))
      .filter(v=>v.j!==i).sort((a,b)=>a.d-b.d).slice(0,2)
      .forEach(({j})=>{ const k=Math.min(i,j)+'-'+Math.max(i,j); if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);} });
  });
  const phases=mapped.map((_,i)=>i*0.8+Math.random()*Math.PI*2);
  let t2=0, raf2; const key='modal-'+card.id;
  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(raf2);cardAnimations.delete(key);return;}
    t2+=0.018; ctx.clearRect(0,0,W,H);
    const bg=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.55);
    bg.addColorStop(0,card.color+'10'); bg.addColorStop(1,'rgba(3,3,14,0)');
    ctx.fillStyle=bg; ctx.fillRect(0,0,W,H);
    edges.forEach(([a,b])=>{
      const pulse=0.28+0.1*Math.sin(t2+(a+b)*0.5);
      ctx.beginPath(); ctx.strokeStyle=card.color+hexA(pulse*255); ctx.lineWidth=1.1;
      ctx.moveTo(mapped[a].x,mapped[a].y); ctx.lineTo(mapped[b].x,mapped[b].y); ctx.stroke();
    });
    mapped.forEach((p,i)=>{
      const tw=0.6+0.4*Math.sin(t2*0.95+phases[i]); const r=2.2+tw*1.2;
      ctx.beginPath(); ctx.arc(p.x,p.y,r*3,0,Math.PI*2); ctx.fillStyle=card.color+hexA(tw*22); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=card.color+'cc'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,1,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.95)'; ctx.fill();
    });
    raf2=requestAnimationFrame(frame); cardAnimations.set(key,raf2);
  }
  if(cardAnimations.has(key)) cancelAnimationFrame(cardAnimations.get(key));
  frame();
}

// ═══════════════════════════════════════════════════════════════
// HAND PICKER
// ═══════════════════════════════════════════════════════════════
function openHandPicker() {
  const modal = el('hand-picker');
  const list  = el('hand-picker-list');
  list.innerHTML = '';

  S.playerHand.forEach(card => {
    const isStaged = !!S.playerPlayed.find(c => c.id === card.id);
    const row = document.createElement('div');
    row.className  = 'picker-row' + (isStaged ? ' picker-staged' : '');
    row.dataset.id = card.id;
    const layerCol = card.layer==='Superego' ? '#D4AF37' : card.layer==='Ego' ? '#7EB8E8' : '#86EFAC';
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
    setTimeout(() => {
      const cvs = row.querySelector('.picker-mini-cvs');
      if (cvs) animatePickerCanvas(cvs, card);
    }, 40);
  });

  el('picker-count').textContent = `${S.playerPlayed.length}/${MAX_STAGED}`;
  modal.classList.add('show');
}

function animatePickerCanvas(canvas, card) {
  if (!canvas) return;
  const W=44, H=44; canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext('2d');
  const pts=card.pts; if(!pts||!pts.length) return;
  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=3, sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc, oY=H/2-(mnY+mxY)/2*sc;
  const mapped=pts.map(([x,y])=>({x:x*sc+oX,y:y*sc+oY}));
  const edgeSet=new Set(), edges=[];
  mapped.forEach((p,i)=>{
    mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)}))
      .filter(v=>v.j!==i).sort((a,b)=>a.d-b.d).slice(0,2)
      .forEach(({j})=>{ const k=Math.min(i,j)+'-'+Math.max(i,j); if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);} });
  });
  const phases=mapped.map(()=>Math.random()*Math.PI*2);
  let t=0, rafId; const key='picker-'+card.id;
  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(rafId);cardAnimations.delete(key);return;}
    t+=0.02; ctx.clearRect(0,0,W,H);
    edges.forEach(([a,b])=>{
      const p=0.18+0.08*Math.sin(t+(a+b)*0.5);
      ctx.beginPath(); ctx.strokeStyle=card.color+hexA(p*180); ctx.lineWidth=0.7;
      ctx.moveTo(mapped[a].x,mapped[a].y); ctx.lineTo(mapped[b].x,mapped[b].y); ctx.stroke();
    });
    mapped.forEach((p,i)=>{
      const tw=0.5+0.5*Math.sin(t+phases[i]); const r=1+tw*0.6;
      ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=card.color+'cc'; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x,p.y,0.5,0,Math.PI*2); ctx.fillStyle='rgba(255,255,255,.85)'; ctx.fill();
    });
    rafId=requestAnimationFrame(frame); cardAnimations.set(key,rafId);
  }
  if(cardAnimations.has(key)) cancelAnimationFrame(cardAnimations.get(key));
  frame();
}

function toggleStageCard(card, rowEl) {
  const already = S.playerPlayed.findIndex(c => c.id === card.id);
  const attnPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const state   = getAttnState(attnPct);
  const limit   = state.debuff?.maxActiveCards || MAX_STAGED;

  if (already >= 0) {
    S.playerPlayed.splice(already, 1);
    rowEl.classList.remove('picker-staged');
    rowEl.querySelector('.picker-check')?.classList.remove('checked');
  } else {
    if (S.playerPlayed.length >= limit)      { toast('Limit Reached', `Max ${limit} cards in this state`); return; }
    if (S.playerPlayed.length >= MAX_STAGED) { toast('Hand Full', 'Maximum 10 cards per battle');          return; }
    S.playerPlayed.push(card);
    rowEl.classList.add('picker-staged');
    rowEl.querySelector('.picker-check')?.classList.add('checked');
  }
  el('picker-count').textContent = `${S.playerPlayed.length}/${MAX_STAGED}`;
  updatePhaseUI();
  renderField();
}

function closeHandPicker() { el('hand-picker').classList.remove('show'); }

// ═══════════════════════════════════════════════════════════════
// FIELD RENDER
// ═══════════════════════════════════════════════════════════════
function renderField() {
  const pField = el('field-player'); if (!pField) return;
  pField.innerHTML = '';
  if (!S.playerPlayed.length) {
    pField.innerHTML = '<div class="field-empty">— no cards staged —</div>';
  } else {
    S.playerPlayed.forEach((card, i) => {
      const chip = document.createElement('div'); chip.className = 'field-chip';
      chip.style.borderColor = card.color + '55'; chip.style.color = card.color;
      const icon = card.layer==='Superego' ? '◈' : card.layer==='Ego' ? '◇' : '○';
      chip.innerHTML = `<span class="chip-icon">${icon}</span><span class="chip-name">${card.name}</span><span class="chip-remove" title="Remove">×</span>`;
      chip.querySelector('.chip-remove').addEventListener('click', e => { e.stopPropagation(); unstageCard(i); });
      pField.appendChild(chip);
    });
  }

  const eField = el('field-enemy');
  if (eField) {
    eField.innerHTML = '';
    if (!S.enemyHand.length) {
      eField.innerHTML = '<div class="field-empty enemy-empty">— enemy awaiting —</div>';
    } else {
      S.enemyHand.forEach(card => {
        const chip = document.createElement('div'); chip.className = 'field-chip enemy-chip';
        chip.style.borderColor = card.color + '55'; chip.style.color = card.color;
        const icon = card.layer==='Superego' ? '◈' : card.layer==='Ego' ? '◇' : '○';
        chip.innerHTML = `<span class="chip-icon">${icon}</span><span class="chip-name">${card.name}</span>`;
        eField.appendChild(chip);
      });
    }
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
  el('modal-name').textContent  = card.name;       el('modal-name').style.color  = card.color;
  el('modal-liturgy').textContent = card.keywords || '';
  let effectText = '';
  if (card.layer==='Superego') effectText = (card.shieldDesc||'') + (card.capacityDesc ? '\n\n'+card.capacityDesc : '');
  else if (card.layer==='Ego') effectText = (card.chunkDesc||'') + (card.divideDesc ? '\n\nReversed: '+card.divideDesc : '');
  else if (card.layer==='ID')  effectText = card.rechargeDesc || card.drainDesc || '';
  else effectText = card.effectDesc || '';
  el('modal-effect').textContent = effectText;
  el('modal-type').textContent   = card.type || '—';
  el('modal-int').textContent    = card.axiumScore !== undefined ? card.axiumScore+' / 10' : '—';
  let shiftEst = 0;
  if (card.layer==='Superego') shiftEst = card.shieldVal   || 0;
  if (card.layer==='ID')       shiftEst = card.rechargeVal || 0;
  if (card.layer==='Ego')      shiftEst = card.chunkFlat   || 0;
  el('modal-shift').textContent  = (shiftEst >= 0 ? '+' : '') + shiftEst;
  el('modal-shift').style.color  = shiftEst >= 0 ? '#86EFAC' : '#e05555';
  el('modal-exhaust').textContent = card.tier ? `Tier ${card.tier}` : '—';
  const myIds  = S.playerPlayed.map(c => c.id).concat(card.id);
  const active = SYNERGIES.filter(s => s.cards.includes(card.id) && s.cards.every(id => myIds.includes(id)));
  const potent = SYNERGIES.filter(s => s.cards.includes(card.id) && !active.includes(s));
  let synTxt = '';
  if (active.length) synTxt += '✦ ' + active.map(s => s.name).join(' · ') + ' (ready!) ';
  if (potent.length) synTxt += (active.length ? '  ' : '') + 'Needs: ' + potent.map(s => s.cards.filter(id => id !== card.id).join('+')).join(' / ');
  el('modal-synergies').textContent = synTxt || (card.synergies ? card.synergies.join(' · ') : '');
  el('card-modal').classList.add('show');
  setTimeout(() => animateModalCanvas(card), 50);
}

function closeModal(e) {
  if (e && e.target !== el('card-modal') && !String(e.target.id).includes('modal-close')) return;
  el('card-modal').classList.remove('show'); S.modalCard = null;
}

// ═══════════════════════════════════════════════════════════════
// DEAL HANDS
// ═══════════════════════════════════════════════════════════════
function dealHand() {
  const attnPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const state   = getAttnState(attnPct);
  const drawMod = state.debuff?.drawMod || 0;
  const size    = clamp(STARTING_HAND + drawMod, 1, STARTING_HAND);

  const available = S.playerDeck.filter(c => !S.playerPlayed.find(p => p.id === c.id));
  S.playerHand = shuffle(available).slice(0, size);
  S.playerPlayed.forEach(c => { if (!S.playerHand.find(h => h.id === c.id)) S.playerHand.unshift(c); });

  S.enemyHand = shuffle([...S.enemyDeck]).slice(0, CHAPTER_CONFIG.enemyHandSize || 3);

  renderField();
}

// ═══════════════════════════════════════════════════════════════
// RESOLVE
// ═══════════════════════════════════════════════════════════════
function resolveRound() {
  if (S.phase !== 'build') return;
  if (!S.playerPlayed.length) { toast('No Cards', 'Stage at least one card'); return; }
  S.enemyPlayed = [...S.enemyHand];
  S.phase = 'battling';
  closeHandPicker();
  el('resolve-btn').classList.remove('show');
  el('pass-btn').disabled = true;
  setTimeout(() => startBattleSequence(), 700);
}

// ═══════════════════════════════════════════════════════════════
// BATTLE SEQUENCE
// ═══════════════════════════════════════════════════════════════
function startBattleSequence() {
  const BT = CHAPTER_CONFIG.battleTiming;

  const ppPct   = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const psState = getAttnState(ppPct);
  const epPct   = clamp(S.enemyAttn / S.enemyMaxAttn * 100, 0, 100);
  const esState = getAttnState(epPct);

  const overlay = document.createElement('div');
  overlay.id = 'battle-overlay';
  overlay.style.cssText = 'position:fixed;inset:0;z-index:500;background:#03030e;opacity:0;transition:opacity .5s;pointer-events:all;display:flex;flex-direction:column;';

  overlay.innerHTML = `
    <canvas id="battle-cvs" style="position:absolute;inset:0;width:100%;height:100%;display:block;z-index:1;"></canvas>
    <div id="ovr-bars" style="position:relative;z-index:10;flex-shrink:0;padding:14px 18px 10px;background:linear-gradient(180deg,rgba(3,3,14,.95) 0%,rgba(3,3,14,.85) 70%,rgba(3,3,14,0) 100%);border-bottom:1px solid rgba(255,255,255,.06);pointer-events:none;">
      <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:rgba(212,175,55,.55);">Your Attention</span>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <span id="ovr-player-val" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(212,175,55,.65);">${Math.round(S.playerAttn)} / ${S.maxAttn}</span>
            <span id="ovr-player-state" style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:17px;color:${psState.col};">${psState.label}</span>
          </div>
        </div>
        <div style="height:6px;background:rgba(255,255,255,.06);border-radius:3px;position:relative;border:1px solid rgba(255,255,255,.08);">
          <div id="ovr-player-fill" style="height:100%;border-radius:3px;width:${ppPct}%;background:linear-gradient(90deg,rgba(107,33,168,.7) 0%,rgba(29,78,216,.7) 18%,rgba(55,65,81,.7) 35%,rgba(126,184,232,.8) 52%,rgba(212,175,55,.9) 70%,rgba(134,239,172,.9) 85%,rgba(255,255,255,1) 100%);transition:width 0.3s;"></div>
          <div id="ovr-player-cursor" style="position:absolute;top:-5px;left:${ppPct}%;width:14px;height:14px;border-radius:50%;background:white;transform:translateX(-50%);transition:left 0.3s;border:2px solid #D4AF37;box-shadow:0 0 10px #D4AF37;"></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:baseline;">
          <span style="font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.18em;text-transform:uppercase;color:rgba(224,85,85,.55);">Enemy Attention</span>
          <div style="display:flex;align-items:baseline;gap:8px;">
            <span id="ovr-enemy-val" style="font-family:'JetBrains Mono',monospace;font-size:11px;color:rgba(224,85,85,.65);">${Math.round(S.enemyAttn)} / ${S.enemyMaxAttn}</span>
            <span id="ovr-enemy-state" style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:17px;color:${esState.col};">${esState.label}</span>
          </div>
        </div>
        <div style="height:6px;background:rgba(255,255,255,.06);border-radius:3px;position:relative;border:1px solid rgba(255,255,255,.08);">
          <div id="ovr-enemy-fill" style="height:100%;border-radius:3px;width:${epPct}%;background:linear-gradient(90deg,rgba(107,33,168,.7) 0%,rgba(220,38,38,.85) 55%,rgba(249,115,22,.9) 80%,rgba(255,200,50,1) 100%);transition:width 0.3s;"></div>
          <div id="ovr-enemy-cursor" style="position:absolute;top:-5px;left:${epPct}%;width:14px;height:14px;border-radius:50%;background:white;transform:translateX(-50%);transition:left 0.3s;border:2px solid #e05555;box-shadow:0 0 10px #e05555;"></div>
        </div>
      </div>
    </div>
    <div id="constellation-area" style="flex:1;position:relative;z-index:5;min-height:0;">
      <div id="battle-phase-lbl" style="position:absolute;left:50%;top:8%;transform:translateX(-50%);font-family:'Cormorant Garamond',serif;font-style:italic;font-size:clamp(16px,4.5vw,24px);letter-spacing:.12em;color:#D4AF37;text-align:center;text-shadow:0 0 40px rgba(212,175,55,.6);pointer-events:none;transition:color .5s;z-index:11;white-space:nowrap;">Constellation Awakening...</div>
    </div>
    <div style="position:relative;z-index:10;display:flex;padding:14px;gap:10px;flex-shrink:0;background:linear-gradient(0deg,rgba(3,3,14,.95) 0%,rgba(3,3,14,0) 100%);pointer-events:none;">
      <div id="battle-log" style="width:min(240px,45vw);display:flex;flex-direction:column;gap:4px;"></div>
    </div>
  `;

  document.body.appendChild(overlay);

  requestAnimationFrame(() => {
    overlay.style.opacity = '1';
    const canvas = document.getElementById('battle-cvs');
    const rect   = overlay.getBoundingClientRect();
    canvas.width  = rect.width;
    canvas.height = rect.height;
    runBattleAnimation(canvas, BT);
  });
}

// ═══════════════════════════════════════════════════════════════
// BATTLE ANIMATION — clean, single frame function
// ═══════════════════════════════════════════════════════════════
function runBattleAnimation(canvas, BT) {
  const ctx          = canvas.getContext('2d');
  const W            = canvas.width;
  const H            = canvas.height;
  const barsPanel    = document.getElementById('ovr-bars');
  const barsHeight   = barsPanel ? barsPanel.offsetHeight : 140;
  const bottomPanel  = document.querySelector('#battle-overlay > div:last-child');
  const bottomHeight = bottomPanel ? bottomPanel.offsetHeight : 60;
  const phaseLbl     = document.getElementById('battle-phase-lbl');
  const battleLogEl  = document.getElementById('battle-log');

  const visibleTop    = barsHeight + 40;
  const visibleBottom = H - bottomHeight - 30;
  const midY          = visibleTop + (visibleBottom - visibleTop) * 0.5;

  // ── Node pulse timings ──────────────────────────────────────
  const NODE_CYCLE   = 600;  // ms for one node to pulse through
  const NODE_STAGGER = 80;   // ms between successive node starts

  // ── Layout helpers ──────────────────────────────────────────
  function layoutCards(played, yPos) {
    const count   = played.length;
    const spacing = Math.min(W * 0.8 / Math.max(count, 1), 120);
    const totalW  = spacing * (count - 1);
    const startX  = (W - totalW) / 2;
    return played.map((card, i) => {
      const x   = startX + i * spacing;
      const obj = { x, y: yPos, card, nodes: [], edges: [], nodePhases: [] };
      if (card.pts) {
        const pts = card.pts;
        let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
        pts.forEach(([px,py])=>{if(px<mnX)mnX=px;if(py<mnY)mnY=py;if(px>mxX)mxX=px;if(py>mxY)mxY=py;});
        const sc  = Math.min(0.8/(mxX-mnX||1), 0.4/(mxY-mnY||1));
        const oX  = 0.5-(mnX+mxX)/2*sc;
        const oY  = 0.25-(mnY+mxY)/2*sc;
        obj.nodes = pts.map(([px,py]) => ({
          x: x + (px*sc + oX - 0.5)*80,
          y: yPos + (py*sc + oY - 0.25)*80,
          cycling: false, cycled: false, cycleProgress: 0,
        }));
        obj.nodePhases = pts.map(() => Math.random() * Math.PI * 2);
        const edgeSet = new Set();
        obj.nodes.forEach((p, ni) => {
          obj.nodes
            .map((q, j) => ({ j, d: Math.hypot(q.x-p.x, q.y-p.y) }))
            .filter(v => v.j !== ni)
            .sort((a, b) => a.d - b.d)
            .slice(0, 2)
            .forEach(({ j }) => {
              const k = Math.min(ni,j)+'-'+Math.max(ni,j);
              if (!edgeSet.has(k)) { edgeSet.add(k); obj.edges.push([ni, j]); }
            });
        });
      }
      return obj;
    });
  }

  const playerCards = layoutCards(S.playerPlayed, midY + (visibleBottom - midY) * 0.4);
  const enemyCards  = layoutCards(S.enemyPlayed,  visibleTop + (midY - visibleTop) * 0.6);
  const allCards    = [...playerCards, ...enemyCards];

  // ── Helpers ─────────────────────────────────────────────────
  function addLog(msg, type='sys') {
    const d = document.createElement('div');
    d.style.cssText = `font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.05em;line-height:1.5;color:${
      type==='p' ? 'rgba(212,175,55,.85)' : type==='e' ? 'rgba(224,85,85,.75)' : 'rgba(255,255,255,.4)'};`;
    d.textContent = msg;
    battleLogEl.appendChild(d);
    while (battleLogEl.children.length > 12) battleLogEl.removeChild(battleLogEl.firstChild);
  }

  function updateOverlayBars() {
    const ppPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
    const epPct = clamp(S.enemyAttn  / S.enemyMaxAttn * 100, 0, 100);
    const ps    = getAttnState(ppPct);
    const es    = getAttnState(epPct);
    const pf = document.getElementById('ovr-player-fill');   if (pf) pf.style.width  = ppPct + '%';
    const pc = document.getElementById('ovr-player-cursor'); if (pc) pc.style.left   = ppPct + '%';
    const pv = document.getElementById('ovr-player-val');    if (pv) pv.textContent  = Math.round(S.playerAttn) + ' / ' + S.maxAttn;
    const pl = document.getElementById('ovr-player-state');  if (pl) { pl.textContent = ps.label; pl.style.color = ps.col; }
    const ef = document.getElementById('ovr-enemy-fill');    if (ef) ef.style.width  = epPct + '%';
    const ec = document.getElementById('ovr-enemy-cursor');  if (ec) ec.style.left   = epPct + '%';
    const ev = document.getElementById('ovr-enemy-val');     if (ev) ev.textContent  = Math.round(S.enemyAttn) + ' / ' + S.enemyMaxAttn;
    const el2= document.getElementById('ovr-enemy-state');   if (el2){ el2.textContent = es.label; el2.style.color = es.col; }
  }

  function drawConstellation(cardData, alpha, timeSec, isEnemy) {
    if (!cardData.nodes.length) return;

    // Draw edges
    cardData.edges.forEach(([a, b]) => {
      const na = cardData.nodes[a], nb = cardData.nodes[b];
      const pulse = 0.12 + 0.06 * Math.sin(timeSec * 2 + (a + b) * 0.5);
      ctx.beginPath();
      ctx.strokeStyle = cardData.card.color + hexA(pulse * 255 * alpha);
      ctx.lineWidth   = 0.7;
      ctx.moveTo(na.x, na.y);
      ctx.lineTo(nb.x, nb.y);
      ctx.stroke();
    });

    // Draw nodes
    cardData.nodes.forEach((n) => {
      let sz = 4;
      let na = alpha * 0.75;
      let col = cardData.card.color;

      if (n.cycled) {
        sz  = 2.5;
        na  = alpha * 0.3;
        col = '#888888';
      } else if (n.cycling) {
        sz = 4 + Math.sin(n.cycleProgress * Math.PI) * 5;
        na = alpha * (0.5 + Math.sin(n.cycleProgress * Math.PI) * 0.5);
      }

      // Expand hex shorthand to full 6-char before appending alpha
      const fullCol = col.length === 4
        ? '#' + col[1]+col[1]+col[2]+col[2]+col[3]+col[3]
        : col;

      const g = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, sz * 3);
      g.addColorStop(0, fullCol + hexA(na * 140));
      g.addColorStop(1, 'transparent');
      ctx.fillStyle = g;
      ctx.beginPath(); ctx.arc(n.x, n.y, sz * 3, 0, Math.PI * 2); ctx.fill();

      ctx.beginPath(); ctx.arc(n.x, n.y, sz, 0, Math.PI * 2);
      ctx.fillStyle = fullCol + hexA(na * 255); ctx.fill();

      ctx.beginPath(); ctx.arc(n.x, n.y, sz * 0.35, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${na})`; ctx.fill();
    });

    // Card name label
    ctx.font      = `bold 10px 'Space Mono',monospace`;
    ctx.textAlign = 'center';
    ctx.fillStyle = `rgba(255,255,255,${alpha * 0.65})`;
    ctx.fillText(cardData.card.name.slice(0, 14), cardData.x, cardData.y + (isEnemy ? -62 : 62));
  }

  // ── THE FRAME LOOP ───────────────────────────────────────────
  let startTime = null;
  let rafId     = null;
  let finishing = false;

  function frame(ts) {
    if (!startTime) startTime = ts;
    const t       = ts - startTime;
    const timeSec = t / 1000;

    // Clear & background
    ctx.clearRect(0, 0, W, H);
    const bg = ctx.createRadialGradient(W/2, H/2, 0, W/2, H/2, Math.max(W, H) * 0.6);
    bg.addColorStop(0, 'rgba(25,10,50,0.08)');
    bg.addColorStop(1, 'rgba(3,3,14,0)');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, W, H);

    // Divider
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(255,255,255,.05)';
    ctx.lineWidth   = 1;
    ctx.moveTo(40, midY);
    ctx.lineTo(W - 40, midY);
    ctx.stroke();

    if (t < BT.charge) {
      // ── CHARGE PHASE: fade all cards in ──
      const prog = 1 - Math.pow(1 - t / BT.charge, 3);
      if (phaseLbl) { phaseLbl.textContent = 'Constellation Awakening...'; phaseLbl.style.color = '#D4AF37'; }
      allCards.forEach(c => drawConstellation(c, prog, timeSec, enemyCards.includes(c)));

    } else {
      // ── RESOLVE PHASE: all nodes ripple simultaneously ──
      if (phaseLbl) { phaseLbl.textContent = '⚡ AXIUM RESONANCE ⚡'; phaseLbl.style.color = '#D4AF37'; }

      let totalNodes    = 0;
      let finishedNodes = 0;

      allCards.forEach((cardData) => {
        const isEnemy = enemyCards.includes(cardData);
        totalNodes += cardData.nodes.length;

        cardData.nodes.forEach((node, nIdx) => {
          // All cards start simultaneously; only stagger within each card by node index
          const nodeStart    = BT.charge + nIdx * NODE_STAGGER;
          const nodeProgress = Math.min(1, Math.max(0, (t - nodeStart) / NODE_CYCLE));

          if (nodeProgress > 0 && nodeProgress < 1) {
            node.cycling       = true;
            node.cycleProgress = nodeProgress;
          } else if (nodeProgress >= 1 && !node.cycled) {
            node.cycled  = true;
            node.cycling = false;

            // Fire drain/log once per card (on first node completing)
            if (nIdx === 0) {
              if (!isEnemy) {
                const drain = Math.min(2, S.playerAttn - MIN_ATTN);
                if (drain > 0) { S.playerAttn -= drain; updateOverlayBars(); }
              } else {
                const drain = Math.min(2, S.enemyAttn);
                if (drain > 0) { S.enemyAttn -= drain; updateOverlayBars(); }
              }
              addLog(`${cardData.card.name} active`, isEnemy ? 'e' : 'p');
            }
          }

          if (node.cycled) finishedNodes++;
        });

        drawConstellation(cardData, cardData.nodes.every(n => n.cycled) ? 0.4 : 1.0, timeSec, isEnemy);
      });

      // ── All done — wrap up ──
      if (finishedNodes >= totalNodes && !finishing) {
        finishing = true;
        if (phaseLbl) {
          phaseLbl.textContent = 'Constellation Complete';
          phaseLbl.style.color = S.playerAttn >= S.enemyAttn ? '#86EFAC' : '#e05555';
        }
        setTimeout(() => {
          cancelAnimationFrame(rafId);
          document.getElementById('battle-overlay')?.remove();
          finaliseBattle();
        }, BT.done);
        return; // stop scheduling new frames after we're done
      }
    }

    rafId = requestAnimationFrame(frame);
  }

  // Kick off the loop
  rafId = requestAnimationFrame(frame);
}

// ═══════════════════════════════════════════════════════════════
// FINALISE
// ═══════════════════════════════════════════════════════════════
function finaliseBattle() {
  // Run the actual math
  const attnPct = clamp(S.playerAttn / S.maxAttn * 100, 0, 100);
  const stateId = getAttnState(attnPct).id;
  const result  = resolveBattle(S.playerPlayed, S.enemyPlayed, S.playerAttn, stateId, S.playerDeck);

  // Apply deltas
  S.playerAttn = clamp(S.playerAttn + result.playerDelta, MIN_ATTN, S.maxAttn);
  S.enemyAttn  = clamp(S.enemyAttn  - result.playerDelta * 0.5, 0, S.enemyMaxAttn);

  result.log.forEach(entry => log(entry.msg, entry.type));
  if (result.instantWin) { triggerWin(true); return; }

  updateBars();
  const pop   = el('resolve-pop');
  const state = getAttnState(clamp(S.playerAttn / S.maxAttn * 100, 0, 100));
  if (pop) { pop.textContent = state.label; pop.style.color = state.col; pop.classList.add('show'); }
  burst(window.innerWidth/2, window.innerHeight*0.5, state.col, 22);
  setTimeout(() => { if (pop) pop.classList.remove('show'); checkWinLose(); }, 1100);
}

// ═══════════════════════════════════════════════════════════════
// WIN / LOSE / NEXT TURN
// ═══════════════════════════════════════════════════════════════
function checkWinLose() {
  if (S.enemyAttn <= 0)                        { triggerWin(false); return; }
  if (S.playerAttn <= CHAPTER_CONFIG.loseAttn)  { triggerLose();     return; }
  if (S.playerAttn >= CHAPTER_CONFIG.winAttn)   { triggerWin(false); return; }
  if (S.playerPlayed.length >= 10) {
    const avg = S.playerPlayed.reduce((s,c) => s+(c.axiumScore||0), 0) / S.playerPlayed.length;
    if (avg >= 10) { triggerWin(true); return; }
  }
  // Single-turn design: if no win condition met, it's a loss
  triggerLose();
}

function triggerWin(perfect=false) {
  S.won = true; S.phase = 'done';
  el('out-title').textContent      = perfect ? 'The Axium' : CHAPTER_CONFIG.winTitle;
  el('out-title').style.color      = '#D4AF37';
  el('out-title').style.textShadow = '0 0 50px rgba(212,175,55,.4)';
  el('out-desc').textContent       = perfect ? 'Perfect constellation. The final boss awakens.' : CHAPTER_CONFIG.winDesc;
  el('out-btn').textContent        = CHAPTER_CONFIG.winBtn;
  el('out-btn').style.cssText      = 'background:linear-gradient(135deg,#AA8C2C,#D4AF37,#AA8C2C);color:#0a0a0a;border:none;padding:11px 34px;cursor:pointer;border-radius:2px;';
  setTimeout(() => el('outcome').classList.add('show'), 600);
  burst(window.innerWidth/2, window.innerHeight/2, '#D4AF37', 44);
  setTimeout(() => burst(window.innerWidth/2, window.innerHeight/2, '#86EFAC', 28), 500);
}

function triggerLose() {
  S.lost = true; S.phase = 'done';
  el('out-title').textContent      = CHAPTER_CONFIG.loseTitle;
  el('out-title').style.color      = '#e05555';
  el('out-title').style.textShadow = '0 0 50px rgba(224,85,85,.4)';
  el('out-desc').textContent       = CHAPTER_CONFIG.loseDesc;
  el('out-btn').textContent        = CHAPTER_CONFIG.loseBtn;
  el('out-btn').style.cssText      = 'background:rgba(224,85,85,.08);color:#e05555;border:1px solid rgba(224,85,85,.3);padding:11px 34px;cursor:pointer;border-radius:2px;';
  setTimeout(() => el('outcome').classList.add('show'), 400);
  burst(window.innerWidth/2, window.innerHeight/2, '#DC2626', 28);
}

function handleOutcome() { el('outcome').classList.remove('show'); if (S.won) openShop(); else resetChapter(); }

function nextTurn() {
  S.turn++; S.phase = 'build'; S.playerPlayed = []; S.enemyPlayed = [];
  const tn = el('turn-n'); if (tn) tn.textContent = S.turn;
  const pb = el('pass-btn'); if (pb) pb.disabled = false;
  recalcMaxAttn(); dealHand();
  log(`── Turn ${S.turn} ──`, 'sys');
}

function passAndEndTurn() {
  if (S.phase !== 'build') return;
  shiftPlayer(-6); log('Passed — attention −6', 'sys');
  S.playerPlayed = []; S.phase = 'battling';
  el('pass-btn').disabled = true;
  el('resolve-btn').classList.remove('show');
  setTimeout(() => checkWinLose(), 900);
}

function resetChapter() {
  const eDeckIds  = CHAPTER_CONFIG.enemyDeck || [];
  S.playerDeck    = shuffle([...PLAYER_CARDS]);
  S.enemyDeck     = eDeckIds.map(id => PLAYER_CARDS.find(c => c.id === id)).filter(Boolean);
  S.playerAttn    = CHAPTER_CONFIG.playerStart;
  S.enemyAttn     = CHAPTER_CONFIG.enemyStart || 80;
  S.enemyMaxAttn  = CHAPTER_CONFIG.enemyMax   || 100;
  S.turn          = 1; S.phase = 'build';
  S.playerHand    = []; S.playerPlayed = []; S.enemyHand = []; S.enemyPlayed = [];
  S.won = false; S.lost = false;
  recalcMaxAttn(); updateBars();
  el('turn-n').textContent = '1';
  el('pass-btn').disabled  = false;
  el('resolve-btn').classList.remove('show');
  const lg = el('log'); if (lg) lg.innerHTML = '';
  dealHand(); log('── Chapter reset ──', 'sys');
}

// ═══════════════════════════════════════════════════════════════
// SHOP
// ═══════════════════════════════════════════════════════════════
function openShop() {
  const npc    = getShopNPC(CHAPTER_CONFIG.shopChapter || CURRENT_CHAPTER);
  const offers = getShopOffers(CHAPTER_CONFIG.shopChapter || CURRENT_CHAPTER, 3).filter(Boolean);
  el('sh-npc-name').textContent = npc.name; el('sh-npc-role').textContent = npc.role;
  el('sh-speech').textContent   = npc.speeches[Math.floor(Math.random() * npc.speeches.length)];
  const cardsEl = el('sh-cards'); cardsEl.innerHTML = '';
  offers.forEach(card => {
    if (!card) return;
    const wrap = document.createElement('div'); wrap.className = 'sh-card-wrap';
    wrap.appendChild(makeCard(card, []));
    const desc = card.chunkDesc || card.shieldDesc || card.rechargeDesc || card.keywords || '';
    const lbl  = document.createElement('div'); lbl.className = 'sh-upgrade-lbl';
    lbl.textContent = desc.length > 52 ? desc.slice(0, 52) + '…' : desc;
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
// UTILS
// ═══════════════════════════════════════════════════════════════
function el(id)           { return document.getElementById(id); }
function clamp(v, mn, mx) { return Math.max(mn, Math.min(mx, v)); }

function log(msg, type='sys') {
  const logEl = el('log'); if (!logEl) return;
  const line = document.createElement('div'); line.className = `log-line ${type}`; line.textContent = msg;
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
    const p = document.createElement('div'); p.className = 'ptcl';
    const ang  = (Math.PI*2*i)/n + Math.random()*.5;
    const dist = 20 + Math.random()*65;
    const dur  = 380 + Math.random()*260;
    p.style.cssText = `left:${x}px;top:${y}px;width:${1.4+Math.random()*2.8}px;height:${1.4+Math.random()*2.8}px;background:${color};box-shadow:0 0 5px ${color};transition:transform ${dur}ms cubic-bezier(.22,1,.36,1),opacity ${dur}ms ease;position:fixed;border-radius:50%;pointer-events:none;z-index:900;`;
    document.body.appendChild(p);
    requestAnimationFrame(() => { p.style.transform=`translate(${Math.cos(ang)*dist}px,${Math.sin(ang)*dist}px) scale(0)`; p.style.opacity='0'; });
    setTimeout(() => p.remove(), dur);
  }
}

// Stars
(function(){
  const cv=el('stars'); if(!cv) return;
  const ctx=cv.getContext('2d');
  let W,H,stars=[];
  function resize(){
    W=cv.width=window.innerWidth; H=cv.height=window.innerHeight;
    stars=Array.from({length:140},()=>({x:Math.random()*W,y:Math.random()*H,r:.3+Math.random()*1.1,a:.07+Math.random()*.36,sp:.22+Math.random()*.55,ph:Math.random()*Math.PI*2}));
  }
  function draw(t){
    requestAnimationFrame(draw); ctx.fillStyle='#03030e'; ctx.fillRect(0,0,W,H);
    const g=ctx.createRadialGradient(W*.4,H*.35,0,W*.4,H*.35,W*.5);
    g.addColorStop(0,'rgba(40,20,70,.13)'); g.addColorStop(1,'rgba(3,3,14,0)');
    ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
    stars.forEach(s=>{ const tw=.4+.6*Math.abs(Math.sin(t*.0008*s.sp+s.ph)); ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2); ctx.fillStyle=`rgba(255,255,255,${s.a*tw})`; ctx.fill(); });
  }
  resize(); window.addEventListener('resize',resize); requestAnimationFrame(draw);
})();

(function(){
  if(!document.getElementById('game-keyframes')){
    const s=document.createElement('style'); s.id='game-keyframes';
    s.textContent='@keyframes logSlide{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}';
    document.head.appendChild(s);
  }
})();
