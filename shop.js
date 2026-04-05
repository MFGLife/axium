/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — shop.js  (Shop, Personas, Deck Review, Seed Loader)
 * Depends on: cards.js, engine.js, ui.js
 * ═══════════════════════════════════════════════════════════════
 */

'use strict';

const PERSONA_ORDER = ['micheal', 'gabriel', 'ariel', 'seraphina'];
const CHAPTER_ID    = 1;   // overridden per chapter if needed

// ─────────────────────────────────────────────────────────
// INTRO / JOURNEY START
// ─────────────────────────────────────────────────────────
function beginJourney() {
  APP.save = AxiumSave.getOrCreate();
  const doneShops = APP.save.shopHistory.filter(h => h.chapter === CHAPTER_ID).length;
  if (doneShops >= 4) {
    APP.playerDeck = [...APP.save.deck];
    buildDeckReview();
    goTo('screen-deck-review');
  } else {
    APP.shopStep = 0;
    APP.shopPicks = [];
    APP.save.deck = [...STARTING_DECK_IDS];
    AxiumSave.set(APP.save);
    startShopStep(0);
    goTo('screen-shop');
  }
}

function showSeedLoad() { goTo('screen-seed-load'); }

function loadSeedInput() {
  const raw = gel('seed-input')?.value?.trim();
  if (!raw) return;
  const save = AxiumSeed.load(raw);
  const msg  = gel('sl-msg');
  if (save) {
    APP.save = save;
    if (msg) { msg.textContent = '✓ Deck restored!'; msg.style.color = '#86EFAC'; msg.classList.add('show'); }
    APP.playerDeck = [...save.deck];
    setTimeout(() => { buildDeckReview(); goTo('screen-deck-review'); }, 900);
  } else {
    if (msg) { msg.textContent = '✗ Invalid seed — check and try again.'; msg.style.color = '#e05555'; msg.classList.add('show'); }
    setTimeout(() => msg?.classList.remove('show'), 3000);
  }
}

// ─────────────────────────────────────────────────────────
// SHOP STEPS
// ─────────────────────────────────────────────────────────
function startShopStep(step) {
  APP.shopStep         = step;
  APP.shopSelectedCard = null;

  const personaId = PERSONA_ORDER[step];
  const persona   = PERSONAS[personaId];
  const save      = APP.save;

  // Progress bar
  document.getElementById('shop-progress-fill').style.width = ((step / 4) * 100) + '%';
  document.getElementById('screen-shop').style.setProperty('--persona-color', persona.color);
  document.getElementById('shop-progress-fill').style.background =
    `linear-gradient(90deg,${persona.color},rgba(255,255,255,.4))`;

  // Sidebar
  gel('persona-step-lbl').textContent    = `Shop ${step + 1} of 4`;
  gel('persona-name').textContent        = persona.name;
  gel('persona-name').style.color        = persona.color;
  gel('persona-title').textContent       = persona.subtitle + ' · ' + persona.title;
  gel('persona-tagline').textContent     = persona.tagline;
  gel('persona-tagline').style.color     = persona.color + 'aa';
  gel('persona-desc').textContent        = persona.description;
  gel('persona-mech-type').textContent   = persona.cardType;
  gel('persona-mech-type').style.color   = persona.color;
  gel('persona-mech-desc').textContent   = persona.mechDesc;
  gel('persona-glow-bg').style.background =
    `radial-gradient(ellipse at 30% 30%,${persona.color} 0%,transparent 65%)`;
  gel('persona-avatar-ring').style.borderColor = persona.color + '55';
  gel('persona-avatar-ring').style.boxShadow   = `0 0 20px ${persona.color}22`;

  const speeches = persona.speeches;
  gel('persona-speech-bubble').textContent      = speeches[Math.floor(Math.random() * speeches.length)];
  gel('persona-speech-bubble').style.borderLeftColor = persona.color + '44';

  // Progress dots
  PERSONA_ORDER.forEach((_, i) => {
    const dot = gel('pdot-' + i); if (!dot) return;
    dot.classList.remove('done', 'current');
    if (i < step) dot.classList.add('done');
    else if (i === step) dot.classList.add('current');
    dot.style.setProperty('--persona-color', persona.color);
  });

  // Avatar constellation
  animatePersonaAvatar(persona);

  // Card offers
  const offers  = AxiumShop.getOffers(personaId, CHAPTER_ID, save.deck, 4);
  APP.shopOffers = offers;
  renderShopCards(offers, persona);

  // Footer reset
  gel('shop-deck-count').innerHTML = `Deck: <span id="deck-size-lbl">${save.deck.length}</span>`;
  gel('shop-selection-msg').textContent = 'Select a card to continue';
  gel('shop-confirm-btn').classList.remove('ready');
  gel('shop-main-title').textContent = `${persona.name}'s Offering`;
  gel('shop-sub').textContent = 'Choose one card for your deck';
}

function confirmShopPick() {
  if (!APP.shopSelectedCard) return;
  const card     = APP.shopSelectedCard;
  APP.save       = AxiumSave.addCardToDeck(card.id, PERSONA_ORDER[APP.shopStep], CHAPTER_ID);
  const nextStep = APP.shopStep + 1;
  if (nextStep < 4) {
    toast('Added', card.name + ' added to your deck');
    startShopStep(nextStep);
  } else {
    toast('Deck Complete', 'Your constellation is ready');
    APP.playerDeck = [...APP.save.deck];
    buildDeckReview();
    setTimeout(() => goTo('screen-deck-review'), 600);
  }
}

function selectShopCard(card, el) {
  document.querySelectorAll('.shop-card').forEach(c => {
    c.classList.remove('selected');
    c.style.borderColor = (APP.shopOffers.find(o => o.id === c.dataset.id)?.color || '#fff') + '22';
    c.style.boxShadow = '';
  });
  APP.shopSelectedCard = card;
  el.classList.add('selected');
  el.style.borderColor = card.color + '66';
  el.style.boxShadow   = `0 0 22px ${card.color}33`;
  gel('shop-selection-msg').textContent = `"${card.name}" selected`;
  gel('shop-confirm-btn').classList.add('ready');
}

// ─────────────────────────────────────────────────────────
// RENDER SHOP CARDS
// ─────────────────────────────────────────────────────────
function renderShopCards(cards, persona) {
  const grid = gel('shop-card-grid');
  grid.innerHTML = '';
  APP.shopCanvasAnims.forEach((id, key) => { if (key.startsWith('shop-')) cancelAnimationFrame(id); });

  cards.forEach(card => {
    const el       = document.createElement('div');
    el.className   = 'shop-card';
    el.dataset.id  = card.id;
    const layerCol = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
    const mechStr  = card.layer === 'Superego'
      ? `Shield +${card.shieldVal || 0}`
      : card.layer === 'Ego'
        ? [(card.chunkFlat ? `+${card.chunkFlat} flat` : ''), (card.chunkPct ? `×${card.chunkPct}` : '')].filter(Boolean).join(' ')
        : `Recharge +${card.rechargeVal || 0}/stack`;

    el.innerHTML = `
      <div class="shop-card-glow" style="background:radial-gradient(ellipse at 35% 25%,${card.color}44,transparent 65%)"></div>
      <div class="shop-card-check">
        <svg viewBox="0 0 12 12" fill="none"><polyline points="2,6 5,9 10,3" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>
      </div>
      <div class="shop-card-top">
        <div class="shop-card-layer" style="color:${layerCol}">${cardLabel(card)}</div>
        <div class="shop-card-name" style="color:${card.color}">${card.name}</div>
      </div>
      <canvas class="shop-card-cvs" data-cid="${card.id}"></canvas>
      <div class="shop-card-bot">
        <span class="shop-card-type ${card.type || 'compression'}">${(card.type || 'comp').slice(0, 4)}</span>
        <span class="shop-card-axium" style="color:${card.color}">⬡${card.axiumScore || '?'}</span>
      </div>
      <div class="shop-card-mech" style="color:${card.color}88">${mechStr}</div>
    `;
    el.style.borderColor = card.color + '22';
    el.addEventListener('click', () => selectShopCard(card, el));
    grid.appendChild(el);

    setTimeout(() => {
      const cvs = el.querySelector('.shop-card-cvs');
      if (cvs) animateCardCanvas(cvs, card);
    }, 30);
  });
}

// ─────────────────────────────────────────────────────────
// PERSONA AVATAR CONSTELLATION
// ─────────────────────────────────────────────────────────
function animatePersonaAvatar(persona) {
  const canvas = gel('persona-avatar-canvas'); if (!canvas) return;
  const [R, G, B] = persona.colorRGB;
  const W = 80, H = 80;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const patterns = {
    micheal:   [[.5,.15],[.85,.5],[.5,.85],[.15,.5],[.5,.5],[.65,.35],[.65,.65],[.35,.65],[.35,.35]],
    gabriel:   [[.5,.1],[.9,.5],[.5,.9],[.1,.5],[.5,.3],[.7,.5],[.5,.7],[.3,.5]],
    ariel:     [[.5,.2],[.8,.4],[.8,.7],[.5,.9],[.2,.7],[.2,.4],[.5,.5]],
    seraphina: [[.5,.1],[.85,.35],[.75,.75],[.5,.9],[.25,.75],[.15,.35],[.5,.55],[.38,.38],[.62,.38]],
  };
  const pts = (patterns[persona.id] || patterns.micheal).map(([x, y]) => ({ x: x * W, y: y * H }));
  const edges = [], es = new Set();
  pts.forEach((p, i) => {
    pts.map((q, j) => ({ j, d: Math.hypot(q.x - p.x, q.y - p.y) }))
      .filter(v => v.j !== i).sort((a, b) => a.d - b.d).slice(0, 2)
      .forEach(({ j }) => {
        const k = Math.min(i, j) + '-' + Math.max(i, j);
        if (!es.has(k)) { es.add(k); edges.push([i, j]); }
      });
  });
  const phases = pts.map(() => Math.random() * Math.PI * 2);
  let t = 0;
  const key = 'avatar-' + persona.id;
  if (APP.shopCanvasAnims.has(key)) cancelAnimationFrame(APP.shopCanvasAnims.get(key));

  function frame() {
    if (!canvas.isConnected) { APP.shopCanvasAnims.delete(key); return; }
    t += .018; ctx.clearRect(0, 0, W, H);
    const bg = ctx.createRadialGradient(W / 2, H / 2, 0, W / 2, H / 2, W * .55);
    bg.addColorStop(0, `rgba(${R},${G},${B},.12)`); bg.addColorStop(1, 'rgba(2,2,10,.9)');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    edges.forEach(([a, b]) => {
      ctx.beginPath(); ctx.strokeStyle = `rgba(${R},${G},${B},.35)`; ctx.lineWidth = .8;
      ctx.moveTo(pts[a].x, pts[a].y); ctx.lineTo(pts[b].x, pts[b].y); ctx.stroke();
    });
    pts.forEach((p, i) => {
      const tw = .5 + .5 * Math.sin(t + phases[i]); const r = 1.2 + tw * .8;
      const g = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 3);
      g.addColorStop(0, `rgba(${R},${G},${B},${(.7 * tw).toFixed(2)})`); g.addColorStop(1, 'transparent');
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(p.x, p.y, r * 3, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(255,255,255,${(.8 + .2 * tw).toFixed(2)})`; ctx.fill();
    });
    APP.shopCanvasAnims.set(key, requestAnimationFrame(frame));
  }
  frame();
}

// ─────────────────────────────────────────────────────────
// DECK REVIEW
// ─────────────────────────────────────────────────────────
function buildDeckReview() {
  const grid = gel('deck-review-grid'); if (!grid) return;
  grid.innerHTML = '';
  const save = APP.save || AxiumSave.getOrCreate();
  save.deck.forEach(id => {
    const card = getCard(id); if (!card) return;
    const layerCol = card.layer === 'Superego' ? '#D4AF37' : card.layer === 'Ego' ? '#7EB8E8' : '#86EFAC';
    const chip = document.createElement('div');
    chip.className = 'dr-card-chip';
    chip.style.borderColor = card.color + '33';
    chip.innerHTML = `
      <div class="dr-chip-name" style="color:${card.color}">${card.name}</div>
      <div class="dr-chip-layer" style="color:${layerCol}">${cardLabel(card)}</div>
    `;
    grid.appendChild(chip);
  });
}

// ─────────────────────────────────────────────────────────
// OUTCOME HELPERS
// ─────────────────────────────────────────────────────────
function showOutcome(won, perfect, B) {
  const save = APP.save || AxiumSave.getOrCreate();
  gel('oc-stat-attn').textContent = Math.round(B.playerAttn);
  gel('oc-stat-attn').style.color = won ? '#86EFAC' : '#e05555';
  gel('oc-stat-deck').textContent = save.deck.length;
  gel('oc-stat-syns').textContent = B.synsFired;

  if (won) {
    gel('oc-state-lbl').textContent = 'Chapter I Complete';
    gel('oc-title').textContent     = perfect ? 'The Axium' : 'Attention Held';
    gel('oc-title').style.color     = '#D4AF37';
    gel('oc-desc').textContent      = perfect
      ? 'Perfect constellation. Your deck is forged in clarity.'
      : 'Your constellation held. The ego ran out of material.';
    gel('oc-primary-btn').textContent = 'Save & Continue';
    gel('oc-primary-btn').onclick     = () => toast('Deck Saved', 'Load your seed in Chapter 2 to continue');
  } else {
    gel('oc-state-lbl').textContent = 'Attention Lost';
    gel('oc-title').textContent     = 'Fragmented';
    gel('oc-title').style.color     = '#e05555';
    gel('oc-desc').textContent      = 'The ego outlasted you. Your deck is preserved — try again.';
    gel('oc-primary-btn').textContent = 'Try Again';
    gel('oc-primary-btn').onclick     = () => enterBattle();
  }

  if (typeof renderSeedWidget === 'function') renderSeedWidget('seed-container', save);
  goTo('screen-outcome');
}
