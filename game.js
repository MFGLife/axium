
// ═══════════════════════════════════════════════════════════════
// GAME STATE
// ═══════════════════════════════════════════════════════════════
const HAND_SIZE = 4;
const MIN_ATTN  = 5;
const MAX_ATTN  = 98;
const CURRENT_CHAPTER = 1;

const S = {
  playerAttn:       55,
  traumaCoherence:  80,
  turn:             1,
  phase:            'player',   // 'player' | 'resolving'
  playerHand:       [],
  playerDeck:       [],
  traumaDeck:       [],
  playerPlayed:     [null, null],
  traumaPlayed:     [null, null],
  extraDraw:        0,
  // Status flags
  shieldActive:     false,
  shieldCount:      0,    // how many hits left to block
  spaceSkip:        false,
  centeringActive:  false,
  monologueSkip:    false,
  monologueDouble:  false,
  attnFloorTurns:   0,
  attnFloor:        0,
  // Meta
  modalCard:        null,
  dragCard:         null,
  dragging:         false,
  won:              false,
  lost:             false,
};

// ═══════════════════════════════════════════════════════════════
// INIT
// ═══════════════════════════════════════════════════════════════
function startChapter() {
  document.getElementById('intro').style.display = 'none';

  // Build decks from cards.js
  S.playerDeck  = [...PLAYER_CARDS];
  S.traumaDeck  = getTraumaDeck(CURRENT_CHAPTER);
  S.playerAttn  = 55;
  S.traumaCoherence = 80;
  S.turn = 1;
  S.phase = 'player';

  initBars();
  initDropZones();
  dealHand();
  renderField();
  updateStatusStrip();

  log('── Chapter 1: Binding of Ego ──', 'sys');
  log('Drag cards to the field, then Resolve.', 'sys');
}

// ═══════════════════════════════════════════════════════════════
// BAR SYSTEM — uses ATTN_STATES from cards.js
// ═══════════════════════════════════════════════════════════════
function initBars() {
  buildPips('player-pips');
  buildPips('trauma-pips');
  updateBars();
}

function buildPips(id) {
  const el = document.getElementById(id);
  el.innerHTML = '';
  ATTN_STATES.forEach(s => {
    const pip = document.createElement('div');
    pip.className = 'bar-pip';
    pip.style.left = (s.pos * 100) + '%';
    pip.style.setProperty('--pip-col', s.col);
    el.appendChild(pip);
  });
}

function updateBars() {
  const pp = clamp(S.playerAttn, 0, 100);
  const tp = clamp(S.traumaCoherence, 0, 100);

  document.getElementById('player-fill').style.width  = pp + '%';
  document.getElementById('player-cursor').style.left = pp + '%';
  document.getElementById('trauma-fill').style.width  = tp + '%';
  document.getElementById('trauma-cursor').style.left = tp + '%';

  const ps = getAttnState(pp);
  const ts = getAttnState(tp);

  el('player-state-lbl').textContent = ps.label;
  el('player-state-lbl').style.color = ps.col;
  el('trauma-state-lbl').textContent = ts.label;
  el('trauma-state-lbl').style.color = ts.col;

  updatePips('player-pips', pp);
  updatePips('trauma-pips', tp);
}

function updatePips(id, val) {
  const pct = val / 100;
  document.querySelectorAll(`#${id} .bar-pip`).forEach((pip, i) => {
    pip.classList.toggle('lit', ATTN_STATES[i] && ATTN_STATES[i].pos <= pct);
  });
}

function shiftPlayer(delta) {
  let next = S.playerAttn + delta;
  // Apply floor if active
  if (S.attnFloorTurns > 0 && delta < 0) {
    next = Math.max(next, S.attnFloor);
  }
  S.playerAttn = clamp(next, MIN_ATTN, MAX_ATTN);
  updateBars();
}

function shiftTrauma(delta) {
  S.traumaCoherence = clamp(S.traumaCoherence + delta, 0, 100);
  updateBars();
}

// ═══════════════════════════════════════════════════════════════
// STATUS STRIP
// ═══════════════════════════════════════════════════════════════
function updateStatusStrip() {
  const strip = el('status-strip');
  strip.innerHTML = '';
  const pills = [];
  if (S.shieldActive)     pills.push({cls:'shield',   txt:'Shield Active'});
  if (S.spaceSkip)        pills.push({cls:'space',    txt:'Space Held'});
  if (S.centeringActive)  pills.push({cls:'centering',txt:'Centering Debuff'});
  if (S.attnFloorTurns>0) pills.push({cls:'shield',   txt:`Floor ×${S.attnFloorTurns}`});
  pills.forEach(p => {
    const d = document.createElement('div');
    d.className = `status-pill ${p.cls}`;
    d.textContent = p.txt;
    strip.appendChild(d);
  });
}

// ═══════════════════════════════════════════════════════════════
// CARD RENDERING
// ═══════════════════════════════════════════════════════════════
const cardAnimations = new Map();

// ── Card label: the small coloured tag above the name ──────────
// Major Arcana  → "Superego · VII"
// Court cards   → "Ego · Cups"
// Pip / ID cards→ "ID · Wands"   (or the traumaRole if set)
// Corruption    → "Corrupt"
function cardLabel(card) {
  if (card.corrupted)    return 'Corrupt';
  if (card.traumaRole)   return card.traumaRole;
  const layer = card.layer || '';
  if (layer === 'Superego') {
    const num = card.number !== undefined ? numberToRoman(card.number) : '';
    return num ? `Superego · ${num}` : 'Superego';
  }
  if (layer === 'Ego') {
    const suit = card.suit ? cap(card.suit) : '';
    return suit ? `Ego · ${suit}` : 'Ego';
  }
  if (layer === 'ID') {
    const suit = card.suit ? cap(card.suit) : '';
    return suit ? `ID · ${suit}` : 'ID';
  }
  return layer || 'Card';
}

// ── Card subtitle: the italic line in the modal ─────────────────
// Uses keywords if present; falls back to layer descriptor.
// Appends " · Corrupted" for corruption cards.
function cardSubtitle(card) {
  let base = '';
  if (card.keywords)   base = card.keywords;
  else if (card.upright) base = card.upright.slice(0, 60) + (card.upright.length > 60 ? '…' : '');
  else                   base = card.layer || '';
  return base + (card.corrupted ? ' · Corrupted' : '');
}

function cap(s) { return s ? s[0].toUpperCase() + s.slice(1) : s; }

function numberToRoman(n) {
  const map = [[21,'XXI'],[20,'XX'],[19,'XIX'],[18,'XVIII'],[17,'XVII'],[16,'XVI'],
               [15,'XV'],[14,'XIV'],[13,'XIII'],[12,'XII'],[11,'XI'],[10,'X'],
               [9,'IX'],[8,'VIII'],[7,'VII'],[6,'VI'],[5,'V'],[4,'IV'],
               [3,'III'],[2,'II'],[1,'I'],[0,'0']];
  const hit = map.find(([v]) => v === n);
  return hit ? hit[1] : String(n);
}

function makeCard(card, classes = []) {
  const isTrauma  = classes.includes('trauma-placed');
  const nameColor = isTrauma ? 'rgba(220,120,120,.95)' : '#fff';

  const el2 = document.createElement('div');
  el2.className = 'card ' + classes.join(' ');
  el2.dataset.id = card.id;
  if (card.corrupted) el2.classList.add('corrupted-card');

  const glow = document.createElement('div');
  glow.className = 'card-glow';
  glow.style.background = `radial-gradient(ellipse at 35% 25%, ${card.color}44, transparent 65%)`;
  el2.appendChild(glow);

  const top = document.createElement('div');
  top.className = 'card-top';
  top.innerHTML = `
    <div class="card-axium-lbl" style="color:${card.color}">${cardLabel(card)}</div>
    <div class="card-name" style="color:${nameColor}">${card.name}</div>
  `;
  el2.appendChild(top);

  const cvs = document.createElement('canvas');
  cvs.className = 'card-cvs';
  el2.appendChild(cvs);

  const cardType = card.type || 'compression';
  const cardInt  = card.intensity !== undefined ? card.intensity : card.traumaShift !== undefined ? '—' : '?';
  const bot = document.createElement('div');
  bot.className = 'card-bot';
  bot.innerHTML = `
    <span class="card-type-pip ${cardType}">${cardType.slice(0,4)}</span>
    <span class="card-intensity">${cardInt}</span>
  `;
  el2.appendChild(bot);

  if (card.corrupted) {
    const badge = document.createElement('div');
    badge.className = 'card-corrupt-badge';
    el2.appendChild(badge);
  }

  setTimeout(() => animateCardCanvas(cvs, card), 20);
  return el2;
}

function animateCardCanvas(canvas, card) {
  if (!canvas || !canvas.parentElement) return;
  const W = canvas.offsetWidth || 78;
  const H = canvas.offsetHeight || 34;
  if (W < 4 || H < 4) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');

  const pts = card.pts;
  if (!pts || !pts.length) return;

  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=6, sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc, oY=H/2-(mnY+mxY)/2*sc;
  const mapped = pts.map(([x,y])=>({x:x*sc+oX, y:y*sc+oY}));

  const edgeSet=new Set(), edges=[];
  mapped.forEach((p,i)=>{
    mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)}))
      .filter(v=>v.j!==i).sort((a,b)=>a.d-b.d)
      .slice(0,2).forEach(({j})=>{
        const k=Math.min(i,j)+'-'+Math.max(i,j);
        if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);}
      });
  });

  const phases = mapped.map((_,i)=>i*0.8+Math.random()*Math.PI*2);
  let t=0, rafId;

  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(rafId);cardAnimations.delete(canvas);return;}
    t+=0.015;
    ctx.clearRect(0,0,W,H);
    edges.forEach(([a,b])=>{
      const pulse=0.18+0.08*Math.sin(t*1.1+(a+b)*0.5);
      ctx.beginPath();
      ctx.strokeStyle=card.color+Math.round(pulse*255).toString(16).padStart(2,'0');
      ctx.lineWidth=0.7;
      ctx.moveTo(mapped[a].x,mapped[a].y);
      ctx.lineTo(mapped[b].x,mapped[b].y);
      ctx.stroke();
    });
    mapped.forEach((p,i)=>{
      const tw=0.55+0.45*Math.sin(t*0.95+phases[i]);
      const r=1.5+tw*0.65;
      ctx.beginPath();ctx.arc(p.x,p.y,r*2.6,0,Math.PI*2);
      ctx.fillStyle=card.color+Math.round(tw*32).toString(16).padStart(2,'0');ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fillStyle=card.color+'cc';ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,0.8,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,.88)';ctx.fill();
    });
    if(edges.length>0){
      const ei=Math.floor(t*0.5)%edges.length;
      const [ea,eb]=edges[ei];
      const prog=(t*0.5)%1;
      const lx=mapped[ea].x+(mapped[eb].x-mapped[ea].x)*prog;
      const ly=mapped[ea].y+(mapped[eb].y-mapped[ea].y)*prog;
      ctx.beginPath();ctx.arc(lx,ly,1.8,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,.9)';ctx.fill();
    }
    rafId=requestAnimationFrame(frame);
    cardAnimations.set(canvas,rafId);
  }
  if(cardAnimations.has(canvas)) cancelAnimationFrame(cardAnimations.get(canvas));
  frame();
}

function animateModalCanvas(card) {
  const canvas = document.getElementById('modal-canvas');
  const W = canvas.offsetWidth || 280;
  const H = 150;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const pts = card.pts;
  if (!pts) return;

  let mnX=1e9,mnY=1e9,mxX=-1e9,mxY=-1e9;
  pts.forEach(([x,y])=>{if(x<mnX)mnX=x;if(y<mnY)mnY=y;if(x>mxX)mxX=x;if(y>mxY)mxY=y;});
  const pad=22, sc=Math.min((W-pad*2)/(mxX-mnX||1),(H-pad*2)/(mxY-mnY||1));
  const oX=W/2-(mnX+mxX)/2*sc, oY=H/2-(mnY+mxY)/2*sc;
  const mapped=pts.map(([x,y])=>({x:x*sc+oX,y:y*sc+oY}));
  const edgeSet=new Set(), edges=[];
  mapped.forEach((p,i)=>{
    mapped.map((q,j)=>({j,d:Math.hypot(q.x-p.x,q.y-p.y)}))
      .filter(v=>v.j!==i).sort((a,b)=>a.d-b.d)
      .slice(0,2).forEach(({j})=>{
        const k=Math.min(i,j)+'-'+Math.max(i,j);
        if(!edgeSet.has(k)){edgeSet.add(k);edges.push([i,j]);}
      });
  });
  const phases=mapped.map((_,i)=>i*0.8+Math.random()*Math.PI*2);
  let t2=0, raf2;
  const key='modal-'+card.id;

  function frame(){
    if(!canvas.isConnected){cancelAnimationFrame(raf2);cardAnimations.delete(key);return;}
    t2+=0.018;
    ctx.clearRect(0,0,W,H);
    const bg=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*0.6);
    bg.addColorStop(0,card.color+'0a');bg.addColorStop(1,'rgba(3,3,14,0)');
    ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
    edges.forEach(([a,b])=>{
      const pulse=0.3+0.12*Math.sin(t2*1.1+(a+b)*0.5);
      const g=ctx.createLinearGradient(mapped[a].x,mapped[a].y,mapped[b].x,mapped[b].y);
      g.addColorStop(0,card.color+Math.round(pulse*255).toString(16).padStart(2,'0'));
      g.addColorStop(1,card.color+Math.round(pulse*200).toString(16).padStart(2,'0'));
      ctx.beginPath();ctx.strokeStyle=g;ctx.lineWidth=1.2;
      ctx.moveTo(mapped[a].x,mapped[a].y);ctx.lineTo(mapped[b].x,mapped[b].y);ctx.stroke();
    });
    mapped.forEach((p,i)=>{
      const tw=0.6+0.4*Math.sin(t2*0.95+phases[i]);
      const r=2.6+tw*1.3;
      ctx.beginPath();ctx.arc(p.x,p.y,r*3,0,Math.PI*2);
      ctx.fillStyle=card.color+Math.round(tw*26).toString(16).padStart(2,'0');ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);
      ctx.fillStyle=card.color+'dd';ctx.fill();
      ctx.beginPath();ctx.arc(p.x,p.y,1.1,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,.95)';ctx.fill();
      [[1,0],[0,1],[-1,0],[0,-1]].forEach(([dx,dy])=>{
        const sl=r*3+tw*4;
        const sg=ctx.createLinearGradient(p.x,p.y,p.x+dx*sl,p.y+dy*sl);
        sg.addColorStop(0,card.color+Math.round(tw*42).toString(16).padStart(2,'0'));
        sg.addColorStop(1,card.color+'00');
        ctx.beginPath();ctx.strokeStyle=sg;ctx.lineWidth=0.65;
        ctx.moveTo(p.x,p.y);ctx.lineTo(p.x+dx*sl,p.y+dy*sl);ctx.stroke();
      });
    });
    if(edges.length>0){
      const ei=Math.floor(t2*0.4)%edges.length;
      const [ea,eb]=edges[ei];
      const prog=(t2*0.4)%1;
      const lx=mapped[ea].x+(mapped[eb].x-mapped[ea].x)*prog;
      const ly=mapped[ea].y+(mapped[eb].y-mapped[ea].y)*prog;
      ctx.beginPath();ctx.arc(lx,ly,2.8,0,Math.PI*2);
      ctx.fillStyle='rgba(255,255,255,.95)';ctx.fill();
    }
    raf2=requestAnimationFrame(frame);
    cardAnimations.set(key,raf2);
  }
  if(cardAnimations.has(key)) cancelAnimationFrame(cardAnimations.get(key));
  frame();
}

// ═══════════════════════════════════════════════════════════════
// HAND & FIELD RENDER
// ═══════════════════════════════════════════════════════════════
function renderHand() {
  const handEl = document.getElementById('hand');
  handEl.innerHTML = '';
  S.playerHand.forEach(card => {
    const cardEl = makeCard(card, ['hand-card']);
    cardEl.addEventListener('click', e => { e.stopPropagation(); openModal(card); });
    attachDrag(cardEl, card);
    handEl.appendChild(cardEl);
  });
}

function renderField() {
  [0,1].forEach(i => {
    // Player slot
    const pSlot = document.getElementById(`player-slot-${i}`);
    const pOld  = pSlot.querySelector('.card');
    if (pOld) pOld.remove();
    pSlot.classList.toggle('has-card', !!S.playerPlayed[i]);
    if (S.playerPlayed[i]) {
      const cardEl = makeCard(S.playerPlayed[i], ['field-card','player-placed']);
      cardEl.addEventListener('click', e => { e.stopPropagation(); openModal(S.playerPlayed[i]); });
      pSlot.appendChild(cardEl);
    }
    // Trauma slot
    const tSlot = document.getElementById(`trauma-slot-${i}`);
    const tOld  = tSlot.querySelector('.card');
    if (tOld) tOld.remove();
    tSlot.classList.toggle('has-card', !!S.traumaPlayed[i]);
    if (S.traumaPlayed[i]) {
      const cardEl = makeCard(S.traumaPlayed[i], ['field-card','trauma-placed']);
      cardEl.addEventListener('click', e => { e.stopPropagation(); openModal(S.traumaPlayed[i]); });
      tSlot.appendChild(cardEl);
    }
  });
}

// ═══════════════════════════════════════════════════════════════
// DRAG SYSTEM
// ═══════════════════════════════════════════════════════════════
function attachDrag(cardEl, card) {
  function onStart(e) {
    e.preventDefault();
    const pt = e.touches ? e.touches[0] : e;
    S.dragCard = card;
    const ghost = document.getElementById('drag-ghost');
    ghost.innerHTML = '';
    const clone = makeCard(card, ['field-card','player-placed']);
    clone.style.width = '80px';
    ghost.style.width = '80px';
    ghost.style.height = '112px';
    ghost.appendChild(clone);
    ghost.style.left = (pt.clientX - 40) + 'px';
    ghost.style.top  = (pt.clientY - 56) + 'px';
    ghost.classList.add('show');
    cardEl.style.opacity = '.32';
  }
  function onMove(e) {
    if (!S.dragCard) return;
    e.preventDefault();
    const pt = e.touches ? e.touches[0] : e;
    const ghost = document.getElementById('drag-ghost');
    ghost.style.left = (pt.clientX - 40) + 'px';
    ghost.style.top  = (pt.clientY - 56) + 'px';
    document.querySelectorAll('.drop-zone.player-zone').forEach(z => {
      const r = z.getBoundingClientRect();
      z.classList.toggle('drag-over',
        pt.clientX >= r.left && pt.clientX <= r.right &&
        pt.clientY >= r.top  && pt.clientY <= r.bottom);
    });
    S.dragging = true;
  }
  function onEnd(e) {
    if (!S.dragCard) return;
    const pt = e.changedTouches ? e.changedTouches[0] : e;
    document.getElementById('drag-ghost').classList.remove('show');
    cardEl.style.opacity = '';
    document.querySelectorAll('.drop-zone.player-zone').forEach(z => {
      z.classList.remove('drag-over');
      const r = z.getBoundingClientRect();
      if (pt.clientX >= r.left && pt.clientX <= r.right &&
          pt.clientY >= r.top  && pt.clientY <= r.bottom) {
        placeCardInSlot(S.dragCard, parseInt(z.dataset.slot));
      }
    });
    S.dragCard = null; S.dragging = false;
  }
  cardEl.addEventListener('touchstart', onStart, {passive:false});
  cardEl.addEventListener('touchmove',  onMove,  {passive:false});
  cardEl.addEventListener('touchend',   onEnd,   {passive:false});
  cardEl.addEventListener('mousedown',  onStart);
  window.addEventListener('mousemove',  onMove);
  window.addEventListener('mouseup',    onEnd);
}

function placeCardInSlot(card, slotIdx) {
  if (S.phase !== 'player') return;
  if (S.playerPlayed[slotIdx]) { toast('Slot Occupied', 'Choose an empty slot'); return; }
  if (!S.playerHand.find(c => c.id === card.id)) return;

  // Anger debuff: only one card per turn
  const angState = getAttnState(S.playerAttn);
  if (angState.id === 'anger' && S.playerPlayed.filter(Boolean).length >= 1) {
    toast('Anger State', 'Only one card can be played — attention too narrow'); return;
  }

  S.playerPlayed[slotIdx] = card;
  S.playerHand = S.playerHand.filter(c => c.id !== card.id);
  renderHand();
  renderField();
  checkShowResolve();
  log(`You play ${card.name}`, 'p');
  burst(window.innerWidth/2, window.innerHeight * 0.65, card.color, 12);
}

function checkShowResolve() {
  const hasPlayed = S.playerPlayed.filter(Boolean).length > 0;
  document.getElementById('resolve-btn').classList.toggle('show', hasPlayed);
  document.getElementById('phase-msg').textContent = hasPlayed
    ? 'Resolve when ready — or play another card'
    : 'Drag a card to the field to play it';
}

// ═══════════════════════════════════════════════════════════════
// CARD MODAL
// ═══════════════════════════════════════════════════════════════
function openModal(card) {
  S.modalCard = card;
  el('modal-axium').textContent     = cardLabel(card);
  el('modal-axium').style.color     = card.color;
  el('modal-name').textContent      = card.name;
  el('modal-name').style.color      = card.color;
  el('modal-liturgy').textContent   = cardSubtitle(card);
  // effectDesc exists on player/ego cards; trauma cards use traumaDesc; ID cards use upright
  el('modal-effect').textContent    = card.effectDesc || card.traumaDesc || card.upright || '';
  el('modal-type').textContent      = card.type || 'compression';
  el('modal-int').textContent       = card.intensity !== undefined ? card.intensity + ' / 10' : '—';
  const shift = card.attnShift !== undefined ? card.attnShift : (card.traumaShift || 0);
  el('modal-shift').textContent     = (shift > 0 ? '+' : '') + shift;
  el('modal-shift').style.color     = shift > 0 ? '#86EFAC' : '#e05555';
  el('modal-exhaust').textContent   = (card.exhaustion || 0) + ' dmg';

  // Synergies from cards.js
  const myIds  = S.playerHand.map(c => c.id).concat(card.id);
  const active = SYNERGIES.filter(s => s.cards.includes(card.id) && s.cards.every(id => myIds.includes(id)));
  const potent = SYNERGIES.filter(s => s.cards.includes(card.id) && !active.includes(s));
  let synTxt = '';
  if (active.length) synTxt += '✦ ' + active.map(s=>s.name).join(' · ') + ' (ready) ';
  if (potent.length) synTxt += 'Synergies: ' + potent.map(s=>s.cards.filter(id=>id!==card.id).join('+')).join(' | ');
  el('modal-synergies').textContent = synTxt || (card.synergies ? 'Synergies: ' + card.synergies.join(' · ') : '');

  const isPlayerCard = S.playerHand.find(c => c.id === card.id);
  el('modal-play-btn').style.display = isPlayerCard ? 'block' : 'none';

  document.getElementById('card-modal').classList.add('show');
  setTimeout(() => animateModalCanvas(card), 50);
}

function closeModal(e) {
  if (e && e.target !== document.getElementById('card-modal') && !e.target.id?.includes('modal-close')) return;
  document.getElementById('card-modal').classList.remove('show');
  S.modalCard = null;
}

function playFromModal() {
  const card = S.modalCard;
  if (!card) return;
  closeModal();
  const slotIdx = !S.playerPlayed[0] ? 0 : !S.playerPlayed[1] ? 1 : -1;
  if (slotIdx === -1) { toast('Field Full', 'Both slots occupied'); return; }
  placeCardInSlot(card, slotIdx);
}

// ═══════════════════════════════════════════════════════════════
// TRAUMA AI — uses cards.js getTraumaDeck + strategy
// ═══════════════════════════════════════════════════════════════
function traumaChooseCards() {
  const deck  = shuffle([...S.traumaDeck]);
  const count = S.traumaCoherence > 60 ? 2 : 1;
  const hand  = deck.slice(0, count + 2);
  let chosen  = [];

  if (S.traumaCoherence < 35) {
    // Desperate — fire Collapse or heal
    const collapse = hand.find(c => c.id === 't_collapse');
    if (collapse) {
      chosen.push(collapse);
    } else {
      const healer = hand.find(c => (c.traumaHealing || 0) > 0);
      chosen.push(healer || hand[0]);
    }
  } else if (S.traumaCoherence < 55) {
    const healer   = hand.find(c => (c.traumaHealing || 0) > 0);
    const attacker = hand.find(c => c.attnShift < -8);
    if (healer)                     chosen.push(healer);
    if (attacker && chosen.length < count) chosen.push(attacker);
    if (!chosen.length)             chosen.push(hand[0]);
  } else {
    // Target player's current state weakness
    const state    = getAttnState(S.playerAttn);
    const targeted = hand.filter(c => c.attnTarget === state.id);
    chosen.push(targeted[0] || hand.find(c => c.type === 'compression') || hand[0]);
    if (chosen.length < count) {
      const second = hand.find(c => !chosen.includes(c) && ((c.traumaHealing||0) > 0 || c.attnShift < -5));
      if (second) chosen.push(second);
    }
  }

  S.traumaPlayed = [chosen[0] || null, chosen[1] || null];
  renderField();
  chosen.forEach(c => log(`Trauma plays ${c.name}`, 't'));
}

// ═══════════════════════════════════════════════════════════════
// RESOLVE ROUND
// ═══════════════════════════════════════════════════════════════
function resolveRound() {
  if (S.phase !== 'player') return;
  if (!S.playerPlayed.filter(Boolean).length) return;

  S.phase = 'resolving';
  el('resolve-btn').classList.remove('show');
  el('pass-btn').disabled = true;

  if (!S.monologueSkip) {
    traumaChooseCards();
  } else {
    S.monologueSkip   = false;
    S.monologueDouble = true;
    log('Trauma skips (monologue) — next hit doubles', 't');
  }

  setTimeout(() => applyAllEffects(), 700);
}

function applyAllEffects() {
  let playerNetShift  = 0;
  let traumaNetDamage = 0;
  const playedIds     = S.playerPlayed.filter(Boolean).map(c => c.id);

  // ── Check synergies from cards.js ──
  const activeSynergies = getSynergies(playedIds);
  activeSynergies.forEach(syn => {
    flashSynergy(syn);
    log(`✦ Synergy: ${syn.name}`, 'synerg');
  });

  // ── Player cards ──
  S.playerPlayed.filter(Boolean).forEach(card => {
    let shift = card.attnShift;

    // Centering debuff
    if (S.centeringActive && shift > 0) {
      shift = Math.max(0, shift - 3);
      if (card.id === 'mirror') { S.centeringActive = false; log('Mirror clears Centering', 's'); }
    }

    playerNetShift  += shift;
    traumaNetDamage += (card.exhaustion || 0);

    // Card-specific effects
    switch (card.id) {
      case 'shield':
        S.shieldActive = true; S.shieldCount = 1;
        log('Shield armed — next trauma blocked', 's'); break;
      case 'shield_plus':
        S.shieldActive = true; S.shieldCount = 2;
        log('Shield+ armed — 2 trauma hits blocked', 's'); break;
      case 'space':
        S.spaceSkip = true;
        log('Space: trauma skips next turn', 's'); break;
      case 'space_plus':
        S.spaceSkip = true;
        S.attnFloorTurns = 3;
        S.attnFloor = S.playerAttn;
        log('Space+: skip + attention floor set', 's'); break;
      case 'mirror':
      case 'mirror_plus':
        S.extraDraw += 1;
        if (S.traumaPlayed[0]) {
          const bonus = Math.abs(S.traumaPlayed[0].attnShift * (card.id==='mirror_plus' ? 1.0 : 0.5));
          playerNetShift += bonus;
          log(`Mirror copies ${bonus.toFixed(0)} attn from trauma`, 's');
        }
        if (card.id === 'mirror_plus' && S.traumaPlayed[0]) {
          // Negate the trauma card's shift too
          S.traumaPlayed[0] = { ...S.traumaPlayed[0], attnShift: 0 };
          log('Mirror+ negates trauma shift', 's');
        }
        break;
      case 'witness':
      case 'witness_plus':
        S.traumaPlayed.forEach(tc => { if (tc) tc.intensity = Math.max(1, tc.intensity - (card.id==='witness_plus'?4:3)); });
        log('Witness reduces trauma intensity', 's'); break;
      case 'truth':
      case 'truth_plus':
        S.traumaDeck = S.traumaDeck.filter(c => !c.tags?.includes('fabricate'));
        traumaNetDamage += card.id === 'truth_plus' ? 25 : 15;
        log('Truth destroys fabrication — extra trauma damage', 's'); break;
      case 'void':
        if (S.traumaCoherence < 50) { playerNetShift -= 4; log('Void risk — chaos reflected back', 'sys'); }
        break;
      case 'void_plus':
        // Risk removed in the + version
        break;
      case 'inversion':
      case 'inversion_plus': {
        const lastTrauma = S.traumaPlayed[0];
        if (lastTrauma) {
          const copy = Math.abs(lastTrauma.attnShift);
          playerNetShift += copy;
          log(`Inversion copies ${copy} from ${lastTrauma.name}`, 's');
          if (card.id === 'inversion_plus' && S.traumaPlayed[1]) {
            const copy2 = Math.abs(S.traumaPlayed[1].attnShift);
            playerNetShift += copy2;
            log(`Inversion+ copies second: +${copy2}`, 's');
          }
          S.traumaPlayed[0] = { ...lastTrauma, attnShift: 0 };
        }
        break;
      }
      case 'release':
      case 'release_plus': {
        const corruptIdx = S.playerDeck.findIndex(c => c.corrupted);
        if (corruptIdx >= 0) {
          const removed = S.playerDeck.splice(corruptIdx, 1)[0];
          if (card.id === 'release_plus') {
            // Remove ALL corruption
            let count = 1;
            while (true) {
              const i = S.playerDeck.findIndex(c => c.corrupted);
              if (i < 0) break;
              S.playerDeck.splice(i, 1); count++;
            }
            playerNetShift += count * 3;
            log(`Release+ cleared ${count} corruption cards, +${count*3} attn`, 's');
          } else {
            playerNetShift += 4;
            log(`Release removes ${removed.name}, +4 attn bonus`, 's');
          }
        } else {
          playerNetShift += card.id === 'release_plus' ? 0 : 4;
        }
        break;
      }
      case 'patient':
      case 'patient_plus':
        S.traumaPlayed.forEach(tc => { if (tc) tc.intensity = Math.max(1, tc.intensity - (card.id==='patient_plus'?3:2)); });
        log('Patience reduces trauma intensity', 's'); break;
    }
  });

  // ── Synergy overrides ──
  if (activeSynergies.some(s => s.id === 'fortified_ground')) {
    // Block ALL trauma shifts
    S.traumaPlayed = S.traumaPlayed.map(c => c ? {...c, attnShift:0} : null);
    toast('Fortified Ground', 'All trauma negated');
  }
  if (activeSynergies.some(s => s.id === 'clear_sight')) {
    if (S.traumaPlayed[0]) {
      playerNetShift += Math.abs(S.traumaPlayed[0].attnShift);
      S.traumaPlayed[0] = {...S.traumaPlayed[0], attnShift:0};
      log('Clear Sight: next trauma card used for you', 's');
    }
  }
  if (activeSynergies.some(s => s.id === 'shadow_flip')) {
    S.traumaPlayed.forEach(tc => {
      if (tc) { playerNetShift += Math.abs(tc.attnShift) * 2; tc.attnShift = 0; }
    });
    log('Shadow Flip: trauma confusion heals you at 2×', 's');
  }
  if (activeSynergies.some(s => s.id === 'full_exposure')) {
    traumaNetDamage += 20;
    S.traumaDeck = S.traumaDeck.filter(c => !c.tags?.includes('fabricate'));
    log('Full Exposure: fabrications destroyed +20 coherence dmg', 's');
  }
  if (activeSynergies.some(s => s.id === 'open_field')) {
    S.spaceSkip = true;
    S.attnFloorTurns = 2; S.attnFloor = S.playerAttn;
    log('Open Field: trauma skip + 2-turn floor', 's');
  }
  if (activeSynergies.some(s => s.id === 'patient_release')) {
    playerNetShift += 5;
    log('Steady Release: bonus +5 shift', 's');
  }
  if (activeSynergies.some(s => s.id === 'deep_mirror')) {
    S.traumaPlayed.forEach(tc => {
      if (tc) { shiftTrauma(tc.attnShift); tc.attnShift = 0; } // trauma takes own hit
    });
    log('Deep Mirror: trauma receives its own card', 's');
  }
  if (activeSynergies.some(s => s.id === 'constellation_complete')) {
    traumaNetDamage += 30;
    playerNetShift  += 15;
    toast('The Constellation', 'All three anchors aligned');
    log('✦✦ Constellation Complete — +30 trauma dmg, +15 attn', 'synerg');
  }
  if (activeSynergies.some(s => s.id === 'full_spectrum')) {
    S.spaceSkip = true;
    playerNetShift += 20;
    toast('Full Spectrum', 'Trauma stunned 2 turns');
    log('✦✦ Full Spectrum — trauma stunned, +20 attn', 'synerg');
  }

  // ── Trauma cards ──
  let traumaNetShift = 0;
  if (S.spaceSkip && !activeSynergies.some(s => ['open_field','full_spectrum'].includes(s.id))) {
    S.spaceSkip = false;
    log('Space holds — trauma turn skipped', 's');
    toast('Space Holds', 'Trauma skips this round');
  } else if (!S.spaceSkip) {
    S.traumaPlayed.filter(Boolean).forEach(card => {
      // Shield block
      if (S.shieldActive && S.shieldCount > 0) {
        S.shieldCount--;
        if (S.shieldCount <= 0) S.shieldActive = false;
        log(`Shield absorbs ${card.name}`, 's');
        if (card.traumaHealing) shiftTrauma(+card.traumaHealing);
        return;
      }

      let shift = card.attnShift;
      if (S.monologueDouble) { shift *= 2; S.monologueDouble = false; log('Monologue double fires!', 't'); }

      // Fragmented state: synergies blocked but no extra trauma penalty
      traumaNetShift += shift;

      if (card.traumaHealing) {
        shiftTrauma(+card.traumaHealing);
        log(`Trauma heals +${card.traumaHealing} coherence`, 't');
      }

      // Trauma special effects
      switch (card.id) {
        case 't_monologue':   S.monologueSkip = true; break;
        case 't_centering':   S.centeringActive = true; log('Centering: your cards -3 shift until Mirror', 't'); break;
        case 't_performance': S.extraDraw = Math.max(-1, S.extraDraw - 1); log('Performance: draw -1 next turn', 't'); break;
        case 't_collapse':
          if (S.traumaCoherence > 30) {
            // Collapse only fires at desperation threshold — ignore it
            traumaNetShift -= shift; // cancel
            log('Collapse: not desperate enough — fizzles', 'sys');
          }
          break;
      }
    });
    S.spaceSkip = false;
  }

  // ── Apply everything ──
  setTimeout(() => {
    const totalShift = playerNetShift + traumaNetShift;
    shiftPlayer(totalShift);
    shiftTrauma(-traumaNetDamage);

    if (S.attnFloorTurns > 0) S.attnFloorTurns--;
    updateStatusStrip();

    if (totalShift > 0) log(`Attention +${totalShift.toFixed(0)}`, 'p');
    else if (totalShift < 0) log(`Attention ${totalShift.toFixed(0)}`, 't');
    if (traumaNetDamage > 0) log(`Trauma −${traumaNetDamage} coherence`, 'p');

    // Resolution pop
    const pop   = document.getElementById('resolve-pop');
    const state = getAttnState(S.playerAttn);
    pop.textContent    = state.label;
    pop.style.color    = state.col;
    pop.classList.add('show');
    burst(window.innerWidth/2, window.innerHeight*0.5, state.col, 16);

    setTimeout(() => { pop.classList.remove('show'); checkWinLose(); }, 1000);
  }, 400);
}

// ═══════════════════════════════════════════════════════════════
// WIN / LOSE
// ═══════════════════════════════════════════════════════════════
function checkWinLose() {
  if (S.traumaCoherence <= 0)        { triggerWin(); return; }
  if (S.playerAttn <= MIN_ATTN + 3)  { triggerLose(); return; }
  if (S.playerAttn >= 92)            { triggerWin(); return; } // Enlightened
  setTimeout(() => nextTurn(), 600);
}

function triggerWin() {
  S.won = true;
  const o = document.getElementById('outcome');
  el('out-title').textContent        = 'Attention Held';
  el('out-title').style.color        = '#D4AF37';
  el('out-title').style.textShadow   = '0 0 50px rgba(212,175,55,.4)';
  el('out-desc').textContent         = 'The trauma found no purchase. Your attention held its center. Proceed to the shop.';
  const btn = el('out-btn');
  btn.textContent                    = 'Visit the Shop';
  btn.style.background               = 'linear-gradient(135deg,#AA8C2C,#D4AF37,#AA8C2C)';
  btn.style.color                    = '#0a0a0a';
  btn.style.border                   = 'none';
  setTimeout(() => o.classList.add('show'), 600);
  burst(window.innerWidth/2, window.innerHeight/2, '#D4AF37', 40);
  setTimeout(() => burst(window.innerWidth/2, window.innerHeight/2, '#86EFAC', 28), 500);
  log('✦ Chapter complete — trauma exhausted', 's');
}

function triggerLose() {
  S.lost = true;
  // Inject a corruption card
  const corrupt = CORRUPTION_CARDS[Math.floor(Math.random() * CORRUPTION_CARDS.length)];
  S.playerDeck.push({...corrupt});
  log(`Corruption enters deck: ${corrupt.name}`, 't');

  const o = document.getElementById('outcome');
  el('out-title').textContent        = 'Attention Lost';
  el('out-title').style.color        = '#e05555';
  el('out-title').style.textShadow   = '0 0 50px rgba(224,85,85,.4)';
  el('out-desc').textContent         = `The trauma found its hold. ${corrupt.name} enters your deck. The chapter resets — but the corruption stays.`;
  const btn = el('out-btn');
  btn.textContent                    = 'Try Again';
  btn.style.background               = 'rgba(224,85,85,.08)';
  btn.style.color                    = '#e05555';
  btn.style.border                   = '1px solid rgba(224,85,85,.3)';
  setTimeout(() => o.classList.add('show'), 400);
  burst(window.innerWidth/2, window.innerHeight/2, '#DC2626', 28);
}

function handleOutcome() {
  document.getElementById('outcome').classList.remove('show');
  if (S.won) openShop();
  else       resetChapter();
}

// ═══════════════════════════════════════════════════════════════
// TURN MANAGEMENT
// ═══════════════════════════════════════════════════════════════
function nextTurn() {
  S.turn++;
  S.phase        = 'player';
  S.playerPlayed = [null, null];
  S.traumaPlayed = [null, null];

  el('turn-n').textContent    = S.turn;
  el('phase-msg').textContent = 'Drag a card to the field to play it';
  el('pass-btn').disabled     = false;

  // Natural drift
  const pState = getAttnState(S.playerAttn);
  const pDrift = S.playerAttn < 50 ? +2 : S.playerAttn > 78 ? -1 : 0;
  const tDrift = S.traumaCoherence > 70 ? -3 : S.traumaCoherence > 40 ? -2 : -1;
  if (pDrift) shiftPlayer(pDrift);
  shiftTrauma(tDrift);

  updateStatusStrip();
  dealHand();
  renderField();
  checkShowResolve();
  log(`── Turn ${S.turn} ──`, 'sys');
}

function passAndEndTurn() {
  if (S.phase !== 'player') return;
  shiftPlayer(-5);
  log('Pass — attention drifts −5', 'sys');

  S.phase = 'resolving';
  el('pass-btn').disabled = true;
  el('resolve-btn').classList.remove('show');

  traumaChooseCards();
  setTimeout(() => {
    S.traumaPlayed.filter(Boolean).forEach(card => {
      if (S.shieldActive) {
        S.shieldCount--;
        if (S.shieldCount <= 0) S.shieldActive = false;
        log('Shield absorbs pass-turn trauma', 's'); return;
      }
      shiftPlayer(card.attnShift);
      if (card.traumaHealing) shiftTrauma(+card.traumaHealing);
      if (card.id === 't_centering') { S.centeringActive = true; }
      if (card.id === 't_monologue') { S.monologueSkip = true; }
      log(`Trauma: ${card.name} (${card.attnShift})`, 't');
    });
    updateStatusStrip();
    checkWinLose();
  }, 700);
}

function resetChapter() {
  // Keep corruption in deck
  const kept = S.playerDeck.filter(c => c.corrupted);
  S.playerDeck  = [...PLAYER_CARDS, ...kept];
  S.traumaDeck  = getTraumaDeck(CURRENT_CHAPTER);
  S.playerAttn  = 55;
  S.traumaCoherence = 80;
  S.turn = 1; S.phase = 'player';
  S.playerHand   = []; S.playerPlayed = [null,null]; S.traumaPlayed = [null,null];
  S.extraDraw    = 0;  S.shieldActive = false; S.shieldCount = 0;
  S.spaceSkip    = false; S.centeringActive = false;
  S.monologueSkip = false; S.monologueDouble = false;
  S.attnFloorTurns = 0; S.attnFloor = 0;
  S.won = false; S.lost = false;

  updateBars();
  updateStatusStrip();
  el('turn-n').textContent    = '1';
  el('phase-msg').textContent = 'Drag a card to the field to play it';
  el('resolve-btn').classList.remove('show');
  el('pass-btn').disabled     = false;
  el('log').innerHTML         = '';
  renderField();
  dealHand();
  if (kept.length) toast('Corruption Remains', `${kept.length} corruption card(s) in deck`);
  log('── Chapter reset ──', 'sys');
}

// ═══════════════════════════════════════════════════════════════
// SHOP — uses cards.js getShopOffers + getShopNPC
// ═══════════════════════════════════════════════════════════════
function openShop() {
  const npc     = getShopNPC(CURRENT_CHAPTER);
  const offers  = getShopOffers(CURRENT_CHAPTER).filter(Boolean);

  el('sh-npc-name').textContent  = npc.name;
  el('sh-npc-role').textContent  = npc.role;
  el('sh-speech').textContent    = npc.speeches[Math.floor(Math.random() * npc.speeches.length)];

  const cardsEl = el('sh-cards');
  cardsEl.innerHTML = '';
  offers.forEach(upg => {
    if (!upg) return;
    const wrap = document.createElement('div');
    wrap.className = 'sh-card-wrap';
    const cardEl = makeCard(upg, []);
    cardEl.style.width = '86px';
    wrap.appendChild(cardEl);
    const lbl = document.createElement('div');
    lbl.className   = 'sh-upgrade-lbl';
    lbl.textContent = upg.upgradeDesc || upg.effectDesc.slice(0, 40) + '…';
    wrap.appendChild(lbl);
    wrap.addEventListener('click', () => {
      acceptUpgrade(upg);
      document.getElementById('shop').classList.remove('show');
    });
    cardsEl.appendChild(wrap);
  });

  document.getElementById('shop').classList.add('show');
}

function acceptUpgrade(upg) {
  // Replace base card if upgrade, or push if new
  const baseId = upg.id.replace('_plus','');
  const idx    = S.playerDeck.findIndex(c => c.id === baseId);
  if (idx >= 0) S.playerDeck[idx] = upg;
  else          S.playerDeck.push(upg);
  toast('Card Acquired', upg.name);
  log(`Shop: ${upg.name} added to deck`, 's');
}

function skipShop() {
  document.getElementById('shop').classList.remove('show');
  log('Shop skipped', 'sys');
}

// ═══════════════════════════════════════════════════════════════
// SYNERGY FLASH
// ═══════════════════════════════════════════════════════════════
function flashSynergy(syn) {
  const flash = document.getElementById('synergy-flash');
  const msg   = document.getElementById('synergy-msg');
  flash.style.background = `radial-gradient(ellipse at center, ${syn.visual}22 0%, transparent 70%)`;
  el('synergy-msg-name').textContent = syn.name;
  el('synergy-msg-desc').textContent = syn.desc;
  el('synergy-msg-name').style.color = syn.visual;
  flash.classList.add('show');
  msg.classList.add('show');
  setTimeout(() => { flash.classList.remove('show'); msg.classList.remove('show'); }, 1600);
}

// ═══════════════════════════════════════════════════════════════
// DROP ZONES
// ═══════════════════════════════════════════════════════════════
function initDropZones() {
  [0,1].forEach(i => {
    const slot = document.getElementById(`player-slot-${i}`);
    slot.addEventListener('dragover',  e => { e.preventDefault(); slot.classList.add('drag-over'); });
    slot.addEventListener('dragleave', () => slot.classList.remove('drag-over'));
    slot.addEventListener('drop', e => {
      e.preventDefault();
      slot.classList.remove('drag-over');
      if (S.dragCard) placeCardInSlot(S.dragCard, i);
    });
  });
}

// ═══════════════════════════════════════════════════════════════
// UTILITIES
// ═══════════════════════════════════════════════════════════════
function el(id) { return document.getElementById(id); }
function clamp(v, mn, mx) { return Math.max(mn, Math.min(mx, v)); }

function dealHand() {
  // Fragmented state: draw -1
  const state   = getAttnState(S.playerAttn);
  const drawMod = (state.debuff?.drawMod || 0);
  const size    = Math.max(1, HAND_SIZE + S.extraDraw + drawMod);
  S.extraDraw   = 0;
  S.playerHand  = shuffle(S.playerDeck).slice(0, size);
  renderHand();
}

function log(msg, type='sys') {
  const logEl = el('log');
  const line  = document.createElement('div');
  line.className  = `log-line ${type}`;
  line.textContent = msg;
  logEl.appendChild(line);
  while (logEl.children.length > 14) logEl.removeChild(logEl.firstChild);
}

let toastTimer;
function toast(h, b) {
  el('toast-h').textContent = h;
  el('toast-b').textContent = b;
  const t = el('toast');
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2400);
}

function burst(x, y, color, n) {
  for (let i = 0; i < n; i++) {
    const p   = document.createElement('div');
    p.className = 'ptcl';
    const ang  = (Math.PI*2*i)/n + Math.random()*.5;
    const dist = 20 + Math.random()*65;
    const dur  = 360 + Math.random()*240;
    p.style.cssText = `left:${x}px;top:${y}px;width:${1.4+Math.random()*2.8}px;height:${1.4+Math.random()*2.8}px;background:${color};box-shadow:0 0 5px ${color};transition:transform ${dur}ms cubic-bezier(.22,1,.36,1),opacity ${dur}ms ease;`;
    document.body.appendChild(p);
    requestAnimationFrame(() => {
      p.style.transform = `translate(${Math.cos(ang)*dist}px,${Math.sin(ang)*dist}px) scale(0)`;
      p.style.opacity   = '0';
    });
    setTimeout(() => p.remove(), dur);
  }
}

// ═══════════════════════════════════════════════════════════════
// STAR BACKGROUND
// ═══════════════════════════════════════════════════════════════
(function(){
  const cv  = document.getElementById('stars');
  const ctx = cv.getContext('2d');
  let W, H, stars = [];
  function resize(){
    W = cv.width  = window.innerWidth;
    H = cv.height = window.innerHeight;
    stars = Array.from({length:160}, () => ({
      x:Math.random()*W, y:Math.random()*H,
      r:.3+Math.random()*1.1, a:.07+Math.random()*.38,
      sp:.22+Math.random()*.55, ph:Math.random()*Math.PI*2,
    }));
  }
  function draw(t){
    requestAnimationFrame(draw);
    ctx.fillStyle='#03030e'; ctx.fillRect(0,0,W,H);
    const g = ctx.createRadialGradient(W*.4,H*.35,0,W*.4,H*.35,W*.5);
    g.addColorStop(0,'rgba(40,20,70,.16)'); g.addColorStop(1,'rgba(3,3,14,0)');
    ctx.fillStyle=g; ctx.fillRect(0,0,W,H);
    stars.forEach(s=>{
      const tw = .4+.6*Math.abs(Math.sin(t*.0008*s.sp+s.ph));
      ctx.beginPath(); ctx.arc(s.x,s.y,s.r,0,Math.PI*2);
      ctx.fillStyle=`rgba(255,255,255,${s.a*tw})`; ctx.fill();
    });
  }
  resize(); window.addEventListener('resize', resize);
  requestAnimationFrame(draw);
})();
