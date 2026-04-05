/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — battle.js  v2.0  (Pre-Simulation Engine)
 *
 * ARCHITECTURE:
 *   1. resolveRound()  — locks hand, runs PRE-SIMULATION synchronously
 *   2. showSynergyIntro() — displays found synergies (covers sim time)
 *   3. startBattlePlayback() — RAF loop replays the pre-computed timeline
 *
 * PRE-SIM produces a flat event array:
 *   { t, type, side, cardId, delta, newPlayerAttn, newEnemyAttn,
 *     newPlayerMax, newEnemyMax, label, color, progress }
 *
 * CARD MECHANICS (all computed in pre-sim, never live):
 *   ID upright      → +rechargeVal to own attn (+ accumulated flat bonus)
 *   ID reversed     → -rechargeVal from enemy attn
 *   Superego upright  → +shieldVal to own attn (+ flat bonus)
 *   Superego reversed → -shieldVal from enemy attn
 *   Superego capacity → +capacityVal to own maxAttn (re-applies each tick)
 *   Superego cap rev  → -capacityVal from enemy maxAttn
 *   Ego upright     → +chunkFlat to running flat bonus (all future gains boosted)
 *   Ego reversed    → -chunkFlat from enemy flat bonus (reduces their gains)
 *
 * TIER INTERVALS:
 *   Tier 1 → 8000ms  |  Tier 2 → 4000ms  |  Tier 3 → 2000ms
 *   Surge (t≥45s) → ÷10
 *
 * BATTLE DURATION: 60s max, early-end on threshold breach.
 * ═══════════════════════════════════════════════════════════════
 */

'use strict';

// ─────────────────────────────────────────────────────────
// BATTLE STATE
// ─────────────────────────────────────────────────────────
const B = {
  playerAttn:    55,
  enemyAttn:     80,
  playerMaxAttn: 100,
  enemyMaxAttn:  100,
  playerDeck:    [],
  playerHand:    [],
  playerPlayed:  [],
  enemyHand:     [],
  phase:         'build',   // 'build' | 'intro' | 'playing' | 'done'
  won:           false,
  lost:          false,
  synsFired:     0,
  // Pre-sim output
  timeline:      [],        // sorted array of sim events
  synergyList:   [],        // synergies found during pre-sim
};

// ─────────────────────────────────────────────────────────
// TIER → INTERVAL
// ─────────────────────────────────────────────────────────
function tierInterval(tier, surge) {
  const base = tier >= 3 ? 2000 : tier === 2 ? 4000 : 8000;
  return surge ? base / 10 : base;
}

// ─────────────────────────────────────────────────────────
// ENTER BATTLE
// ─────────────────────────────────────────────────────────
function enterBattle() {
  const save = AxiumSave.getOrCreate();

  B.playerDeck    = save.deck.map(id => getCard(id)).filter(Boolean);
  B.enemyHand     = buildEnemyHand();
  B.playerAttn    = CHAPTER_CONFIG.playerStart;
  B.enemyAttn     = CHAPTER_CONFIG.enemyStart;
  B.playerMaxAttn = CHAPTER_CONFIG.baseMaxAttn;
  B.enemyMaxAttn  = CHAPTER_CONFIG.enemyMax;
  B.phase         = 'build';
  B.playerPlayed  = [];
  B.won           = false;
  B.lost          = false;
  B.synsFired     = 0;
  B.timeline      = [];
  B.synergyList   = [];

  goTo('screen-battle', () => {
    initBattleBars();
    dealHand();
    if (typeof axcInit === 'function') axcInit();
    else renderBattleField(B, unstageCard);
    updateBattlePhaseUI(B);
  });
}

function buildEnemyHand() {
  const pool = CHAPTER_CONFIG.enemyDeck
    ? CHAPTER_CONFIG.enemyDeck.map(id => getCard(id)).filter(Boolean)
    : shuffle([...PLAYER_CARDS]).slice(0, CHAPTER_CONFIG.enemyHandSize || 4);
  return pool.slice(0, CHAPTER_CONFIG.enemyHandSize || 4).map(card => ({
    card,
    reversed: true,   // enemy always plays reversed → attacks player
  }));
}

function dealHand() {
  B.playerHand = [...B.playerDeck];
}

// ─────────────────────────────────────────────────────────
// INIT BARS
// ─────────────────────────────────────────────────────────
function initBattleBars() {
  buildPips('player-pips');
  buildPips('enemy-pips');
  _syncBarsDisplay();
}

function _syncBarsDisplay() {
  const ppPct = clamp(B.playerAttn / B.playerMaxAttn * 100, 0, 100);
  const epPct = clamp(B.enemyAttn  / B.enemyMaxAttn  * 100, 0, 100);
  setBar('player', ppPct, getAttnState(ppPct));
  setBar('enemy',  epPct, getAttnState(epPct));
  updatePips('player-pips', ppPct);
  updatePips('enemy-pips',  epPct);
}
// ─────────────────────────────────────────────────────────
function unstageCard(idx) {
  if (typeof axcUnstageCard === 'function') {
    axcUnstageCard(idx, B.playerPlayed[idx]?.card); return;
  }
  const entry = B.playerPlayed[idx]; if (!entry) return;
  B.playerPlayed.splice(idx, 1);
  renderBattleField(B, unstageCard);
  updateBattlePhaseUI(B);
}

function openHandPickerBattle() {
  if (typeof axcOpenHandPicker === 'function') axcOpenHandPicker();
  else openHandPicker(B);
}

function closeHandPickerBattle() {
  if (typeof axcCloseHandPicker === 'function') axcCloseHandPicker();
  else { closeHandPicker(); renderBattleField(B, unstageCard); updateBattlePhaseUI(B); }
}

function _setFieldLocked(locked) {
  const f = gel('battle-field');
  if (f) f.style.pointerEvents = locked ? 'none' : 'auto';
}

// ─────────────────────────────────────────────────────────
// RESOLVE — entry point from UI button
// ─────────────────────────────────────────────────────────
function resolveRound() {
  if (B.phase !== 'build') return;
  if (!B.playerPlayed.length) { toast('No Cards', 'Stage at least one card'); return; }
  B.phase = 'intro';

  if (typeof axcCloseHandPicker === 'function') axcCloseHandPicker();
  else closeHandPicker();

  const rb = gel('resolve-btn'); if (rb) rb.classList.remove('show');
  const pb = gel('pass-btn');    if (pb) pb.disabled = true;
  _setFieldLocked(true);

  // ── Run pre-simulation (synchronous, <1ms) ──
  const simResult = preSimulateBattle();
  B.timeline    = simResult.events;
  B.synergyList = simResult.synergies;

  // ── Show synergy intro (covers any render time) ──
  showSynergyIntro(B.synergyList, () => {
    B.phase = 'playing';
    startBattlePlayback();
  });
}

// ─────────────────────────────────────────────────────────
// PRE-SIMULATION
// Runs the full 60s battle in a tight loop.
// Returns { events[], synergies[] }
// ─────────────────────────────────────────────────────────
function preSimulateBattle() {
  const STEP        = 100;   // ms resolution
  const NORMAL_END  = 45000;
  const BATTLE_END  = 60000;
  const LOSE_THRESH = CHAPTER_CONFIG.loseAttn;
  // At time-expiry: player % of their max must exceed enemy % of their max by this margin.
  // e.g. 0.15 means player needs to be 15 percentage points ahead.
  const WIN_PCT_MARGIN = CHAPTER_CONFIG.winPctMargin ?? 0.15;

  const events = [];

  // Mutable sim state
  let pAttn    = B.playerAttn;
  let eAttn    = B.enemyAttn;
  let pMax     = B.playerMaxAttn;
  let eMax     = B.enemyMaxAttn;
  let pFlat    = 0;    // accumulated Ego chunk bonus for player gains
  let eDmgFlat = 0;    // accumulated enemy damage flat bonus (enemy Ego cards)

  // Build card list with per-card state
  const cards = [
    ...B.playerPlayed.map(e => ({
      id:       e.card.id,
      card:     e.card,
      reversed: e.reversed,
      side:     'player',
      tier:     e.card.tier || 1,
      nextFire: tierInterval(e.card.tier || 1, false),
      progress: 0,
    })),
    ...B.enemyHand.map(e => ({
      id:       e.card.id,
      card:     e.card,
      reversed: true,
      side:     'enemy',
      tier:     e.card.tier || 1,
      nextFire: tierInterval(e.card.tier || 1, false),
      progress: 0,
    })),
  ];

  // Detect synergies from player's upright cards
  const playerUprightIds = B.playerPlayed
    .filter(e => !e.reversed)
    .map(e => e.card.id);
  const synergies = typeof getSynergies === 'function'
    ? getSynergies(playerUprightIds)
    : [];

  // Apply one-time synergy attn bonuses at t=0
  synergies.forEach(syn => {
    if (syn.effect?.attnBoost) pAttn = clamp(pAttn + syn.effect.attnBoost, 0, pMax);
  });

  // ── Main simulation loop ──
  let battleEnded = false;
  for (let t = STEP; t <= BATTLE_END && !battleEnded; t += STEP) {
    const surge = t > NORMAL_END;
    if (t === NORMAL_END + STEP) {
      events.push({ t, type: 'surge_start' });
      // Recalculate next-fire for all cards into surge mode
      // Preserve proportional progress through current interval
      cards.forEach(c => {
        const oldInterval = tierInterval(c.tier, false);
        const newInterval = tierInterval(c.tier, true);
        const remaining   = c.nextFire - t + STEP;
        const pct         = clamp(1 - remaining / oldInterval, 0, 1);
        c.nextFire        = t + Math.round(newInterval * (1 - pct));
      });
    }

    // Emit charge-progress events for all cards
    cards.forEach(c => {
      const interval = tierInterval(c.tier, surge);
      const elapsed  = interval - (c.nextFire - t);
      c.progress     = clamp(elapsed / interval, 0, 1);
      events.push({
        t,
        type:     'card_charge',
        side:     c.side,
        cardId:   c.id,
        progress: c.progress,
      });
    });

    // Fire cards whose timer is up
    for (const c of cards) {
      if (battleEnded) break;
      if (t < c.nextFire) continue;

      const card     = c.card;
      const reversed = c.reversed;
      const layer    = card.layer;
      let delta      = 0;
      let label      = '';
      let color      = '#D4AF37';
      let targetSide = 'player'; // which attn value changes

      if (c.side === 'player') {
        if (!reversed) {
          // ── UPRIGHT: benefit player ──
          if (layer === 'ID') {
            delta      = (card.rechargeVal || 0) + pFlat;
            label      = `+${delta} Recharge`;
            color      = '#86EFAC';
            targetSide = 'player';
          } else if (layer === 'Superego') {
            if (card.capacityVal) {
              // Capacity: expand player max
              pMax  = clamp(pMax + card.capacityVal, 10, 300);
              delta = 0;
              label = `Max +${card.capacityVal}`;
              color = '#D4AF37';
            }
            const shieldGain = (card.shieldVal || 0) + pFlat;
            pAttn      = clamp(pAttn + shieldGain, 0, pMax);
            if (shieldGain) { delta = shieldGain; label = `+${shieldGain} Shield`; }
            targetSide = 'player';
          } else if (layer === 'Ego') {
            // Chunk: add to flat bonus, recalculate current gains aren't retroactive
            pFlat     += (card.chunkFlat || 0);
            delta      = card.chunkFlat || 0;
            label      = `+${delta} Chunk`;
            color      = '#7EB8E8';
            targetSide = 'player';
          }
        } else {
          // ── REVERSED: attack enemy ──
          if (layer === 'ID') {
            delta      = -((card.rechargeVal || 0) + pFlat);
            label      = `${delta} Drain`;
            color      = '#e05555';
            targetSide = 'enemy';
          } else if (layer === 'Superego') {
            if (card.capacityVal) {
              eMax  = clamp(eMax - card.capacityVal, 10, 300);
              eAttn = clamp(eAttn, 0, eMax);
              label = `Enemy Max −${card.capacityVal}`;
              color = '#e05555';
            }
            const debuff = -((card.shieldVal || 0) + pFlat);
            delta      = debuff;
            label      = label || `${delta} Debuff`;
            color      = '#e05555';
            targetSide = 'enemy';
          } else if (layer === 'Ego') {
            // Reversed Ego: reduce enemy damage flat bonus
            eDmgFlat   = Math.max(0, eDmgFlat - (card.chunkFlat || 0));
            delta      = -(card.chunkFlat || 0);
            label      = `Enemy Dmg −${card.chunkFlat || 0}`;
            color      = '#e05555';
            targetSide = 'enemy';
          }
        }
      } else {
        // ── ENEMY CARD — always attacks player ──
        if (layer === 'ID') {
          // Drain: base card drain + accumulated enemy damage flat
          delta      = -((card.rechargeVal || 0) + eDmgFlat);
          label      = `${delta} Drain`;
          color      = '#e05555';
          targetSide = 'player';
        } else if (layer === 'Superego') {
          // Enemy Superego deals its shieldVal as a flat drain hit — it does NOT
          // touch player max capacity (that value was designed for player upright use).
          delta      = -((card.shieldVal || 0) + eDmgFlat);
          label      = `${delta} Pressure`;
          color      = '#e05555';
          targetSide = 'player';
        } else if (layer === 'Ego') {
          // Enemy Ego grows the enemy damage flat bonus for all future hits
          eDmgFlat  += (card.chunkFlat || 0);
          delta      = -(card.chunkFlat || 0);
          label      = `Enemy Escalates −${card.chunkFlat || 0}`;
          color      = '#e05555';
          targetSide = 'player';
        }
      }

      // Apply delta
      if (targetSide === 'player') {
        pAttn = clamp(pAttn + delta, 0, pMax);
      } else {
        eAttn = clamp(eAttn + delta, 0, eMax);
      }

      // Record event
      events.push({
        t,
        type:          'card_trigger',
        side:          c.side,
        cardId:        c.id,
        reversed:      reversed,
        delta,
        label,
        color,
        targetSide,
        newPlayerAttn: pAttn,
        newEnemyAttn:  eAttn,
        newPlayerMax:  pMax,
        newEnemyMax:   eMax,
        pFlat,
        eDmgFlat,
      });

      // Reset card timer
      const nextInterval = tierInterval(c.tier, surge);
      c.nextFire = t + nextInterval;
      c.progress = 0;

      // Early KO: first side to hit 0 loses immediately
      if (eAttn <= 0) {
        events.push({ t, type: 'battle_end', winner: 'player', reason: 'enemy_zero' });
        battleEnded = true;
        break;
      }
      if (pAttn <= LOSE_THRESH) {
        events.push({ t, type: 'battle_end', winner: 'enemy', reason: 'player_lost' });
        battleEnded = true;
        break;
      }
    }
  }

  // Time expiry: compare each side's attention as a % of their own max.
  // Player wins only if they are ahead by WIN_PCT_MARGIN or more.
  if (!battleEnded) {
    const pPct = pAttn / pMax;
    const ePct = eAttn / eMax;
    const winner = (pPct - ePct) >= WIN_PCT_MARGIN ? 'player' : 'enemy';
    events.push({ t: BATTLE_END, type: 'battle_end', winner, reason: 'time', pPct, ePct });
  }

  return { events, synergies };
}

// ─────────────────────────────────────────────────────────
// SYNERGY INTRO SCREEN
// Shows all found synergies at once, then calls onDone.
// ─────────────────────────────────────────────────────────
function showSynergyIntro(synergies, onDone) {
  // Remove any existing intro
  document.getElementById('battle-synergy-intro')?.remove();

  const wrap = gel('battle-wrap');
  const intro = document.createElement('div');
  intro.id = 'battle-synergy-intro';
  intro.style.cssText = `
    position:absolute; inset:0; z-index:80;
    background:rgba(2,2,10,.96);
    display:flex; flex-direction:column;
    align-items:center; justify-content:center;
    padding:clamp(16px,4vw,32px); gap:clamp(10px,2.5vh,18px);
    animation: bsiIn .45s cubic-bezier(.22,1,.36,1) both;
  `;

  // Inject animation keyframe once
  if (!document.getElementById('bsi-styles')) {
    const st = document.createElement('style');
    st.id = 'bsi-styles';
    st.textContent = `
      @keyframes bsiIn  { from { opacity:0; transform:scale(.96) } to { opacity:1; transform:scale(1) } }
      @keyframes bsiOut { from { opacity:1; transform:scale(1) } to { opacity:0; transform:scale(1.03) } }
      @keyframes synCardIn { from { opacity:0; transform:translateY(14px) } to { opacity:1; transform:translateY(0) } }
      @keyframes surgeFlash {
        0%,100% { opacity:0 }
        10%,90% { opacity:1 }
        50%      { opacity:.7 }
      }
      .battle-surge-overlay {
        position:absolute; inset:0; pointer-events:none; z-index:70;
        background:radial-gradient(ellipse at center, rgba(212,175,55,.18) 0%, transparent 70%);
        animation: surgeFlash 1.2s ease-in-out infinite;
        display:none;
      }
      .battle-surge-overlay.active { display:block; }
    `;
    document.head.appendChild(st);
  }

  // Header
  const hdr = document.createElement('div');
  hdr.style.cssText = 'text-align:center;';
  hdr.innerHTML = `
    <div style="font-family:'Space Mono',monospace;font-size:8px;letter-spacing:.28em;text-transform:uppercase;color:rgba(212,175,55,.35);margin-bottom:6px;">Constellation Forming</div>
    <div style="font-family:'Cinzel',serif;font-size:clamp(16px,4vw,24px);font-weight:800;letter-spacing:.1em;color:#fff;">
      ${synergies.length ? `${synergies.length} ${synergies.length === 1 ? 'Synergy' : 'Synergies'} Detected` : 'No Synergies'}
    </div>
  `;
  intro.appendChild(hdr);

  if (synergies.length) {
    const grid = document.createElement('div');
    grid.style.cssText = `
      display:flex; flex-direction:column; gap:8px; width:100%;
      max-width:420px; max-height:55vh; overflow-y:auto;
      scrollbar-width:thin; scrollbar-color:rgba(212,175,55,.2) transparent;
    `;
    synergies.forEach((syn, i) => {
      const card = document.createElement('div');
      card.style.cssText = `
        padding:11px 14px; border-radius:8px;
        background:rgba(255,255,255,.03);
        border:1px solid ${syn.visual || 'rgba(212,175,55,.3)'};
        animation: synCardIn .4s cubic-bezier(.22,1,.36,1) ${i * 80}ms both;
      `;
      const cardNames = syn.cards.map(id => {
        const c = (typeof getCard === 'function') ? getCard(id) : null;
        return c ? c.name : id;
      }).join(' · ');
      card.innerHTML = `
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
          <div style="width:6px;height:6px;border-radius:50%;background:${syn.visual||'#D4AF37'};box-shadow:0 0 8px ${syn.visual||'#D4AF37'};flex-shrink:0;"></div>
          <span style="font-family:'Cinzel',serif;font-size:13px;font-weight:600;letter-spacing:.06em;color:${syn.visual||'#D4AF37'}">${syn.name}</span>
          ${syn.rare ? '<span style="font-family:Space Mono,monospace;font-size:6px;letter-spacing:.12em;text-transform:uppercase;padding:2px 7px;border-radius:10px;border:1px solid rgba(212,175,55,.4);color:#D4AF37;background:rgba(212,175,55,.08);">✦ Rare</span>' : ''}
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:7px;letter-spacing:.1em;text-transform:uppercase;color:rgba(255,255,255,.2);margin-bottom:4px;">${cardNames}</div>
        <div style="font-family:'Cormorant Garamond',serif;font-style:italic;font-size:12px;line-height:1.7;color:rgba(255,255,255,.45);">${syn.desc}</div>
      `;
      grid.appendChild(card);
    });
    intro.appendChild(grid);
  } else {
    const none = document.createElement('div');
    none.style.cssText = 'font-family:"Cormorant Garamond",serif;font-style:italic;font-size:14px;color:rgba(255,255,255,.22);text-align:center;';
    none.textContent = 'Cards will act independently.';
    intro.appendChild(none);
  }

  // Countdown bar
  const countdownWrap = document.createElement('div');
  countdownWrap.style.cssText = 'width:100%;max-width:420px;';
  countdownWrap.innerHTML = `
    <div style="font-family:'Space Mono',monospace;font-size:7px;letter-spacing:.2em;text-transform:uppercase;color:rgba(255,255,255,.18);text-align:center;margin-bottom:6px;" id="bsi-countdown-lbl">Beginning in 3…</div>
    <div style="height:2px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden;">
      <div id="bsi-countdown-bar" style="height:100%;background:linear-gradient(90deg,#D4AF37,rgba(255,255,255,.5));width:100%;transition:width .1s linear;"></div>
    </div>
  `;
  intro.appendChild(countdownWrap);

  wrap.appendChild(intro);

  // Countdown: 3 seconds
  const INTRO_DURATION = synergies.length > 0 ? 3500 : 2000;
  const startTime = performance.now();
  const labels = ['Beginning in 3…', 'Beginning in 2…', 'Beginning in 1…', 'Fight!'];

  function tickCountdown() {
    const elapsed = performance.now() - startTime;
    const pct     = clamp(1 - elapsed / INTRO_DURATION, 0, 1);
    const bar     = document.getElementById('bsi-countdown-bar');
    const lbl     = document.getElementById('bsi-countdown-lbl');
    if (bar) bar.style.width = (pct * 100) + '%';
    if (lbl) {
      const li = Math.min(3, Math.floor((elapsed / INTRO_DURATION) * 4));
      lbl.textContent = labels[li] || 'Fight!';
    }
    if (elapsed < INTRO_DURATION) {
      requestAnimationFrame(tickCountdown);
    } else {
      // Fade out intro
      intro.style.animation = 'bsiOut .35s cubic-bezier(.22,1,.36,1) forwards';
      setTimeout(() => {
        intro.remove();
        onDone();
      }, 350);
    }
  }
  requestAnimationFrame(tickCountdown);
}

// ─────────────────────────────────────────────────────────
// BATTLE PLAYBACK
// RAF loop that walks B.timeline and fires visual events.
// Charge is visualised by lighting constellation nodes on
// each field card — no separate ring canvas needed.
// ─────────────────────────────────────────────────────────
function startBattlePlayback() {
  const wrap = gel('battle-wrap');

  // Add surge overlay (hidden until surge)
  let surgeOverlay = wrap.querySelector('.battle-surge-overlay');
  if (!surgeOverlay) {
    surgeOverlay = document.createElement('div');
    surgeOverlay.className = 'battle-surge-overlay';
    wrap.appendChild(surgeOverlay);
  }

  // Build cardId → CardRenderer map from FieldZones via cintegration
  // axcGetFieldRenderer is exposed by cintegration.js
  const _getRenderer = window.axcGetFieldRenderer || (() => null);

  // Sort events by time
  const events  = [...B.timeline].sort((a, b) => a.t - b.t);
  let evtIdx    = 0;
  let startTs   = null;

  function frame(ts) {
    if (!startTs) startTs = ts;
    const elapsed = ts - startTs;

    // Process all events up to elapsed
    while (evtIdx < events.length && events[evtIdx].t <= elapsed) {
      _handlePlaybackEvent(events[evtIdx], surgeOverlay, _getRenderer);
      evtIdx++;
    }

    if (evtIdx < events.length) {
      requestAnimationFrame(frame);
    }
  }
  requestAnimationFrame(frame);
}

// ─────────────────────────────────────────────────────────
// HANDLE PLAYBACK EVENT — fires side effects for each event
// getRenderer(cardId) → CardRenderer | null  (from cintegration)
// ─────────────────────────────────────────────────────────
function _handlePlaybackEvent(evt, surgeOverlay, getRenderer) {
  switch (evt.type) {

    case 'card_charge': {
      // Update constellation charge on the field card
      const renderer = getRenderer(evt.cardId);
      if (renderer) renderer.setCharge(evt.progress);
      break;
    }

    case 'surge_start': {
      surgeOverlay.classList.add('active');
      toast('⚡ SURGE', 'All cards firing 10× faster');
      const pm = gel('b-phase-msg');
      if (pm) { pm.textContent = '⚡ SURGE PHASE'; pm.style.color = '#D4AF37'; }
      break;
    }

    case 'card_trigger': {
      // Update B state from pre-computed values
      B.playerAttn    = evt.newPlayerAttn;
      B.enemyAttn     = evt.newEnemyAttn;
      B.playerMaxAttn = evt.newPlayerMax;
      B.enemyMaxAttn  = evt.newEnemyMax;

      // Update bars
      const ppPct = clamp(B.playerAttn / B.playerMaxAttn * 100, 0, 100);
      const epPct = clamp(B.enemyAttn  / B.enemyMaxAttn  * 100, 0, 100);
      animateBarTo('player', B.playerAttn, B.playerMaxAttn);
      animateBarTo('enemy',  B.enemyAttn,  B.enemyMaxAttn);
      updatePips('player-pips', ppPct);
      updatePips('enemy-pips',  epPct);

      // Update phase message
      const pm = gel('b-phase-msg');
      if (pm) { pm.textContent = evt.label; pm.style.color = evt.color; }

      // Trigger constellation burst on the firing card, reset charge
      const renderer = getRenderer(evt.cardId);
      if (renderer) renderer.triggerBurst();

      // Card highlight flash + floating label over the card
      _flashCard(evt.side, evt.cardId, evt.color);
      _floatLabelOverCard(evt.side, evt.cardId, evt.label, evt.color);

      // Burst particles at bar
      const barEl = gel(evt.targetSide + '-bar-track');
      if (barEl) {
        const br = barEl.getBoundingClientRect();
        burst(br.left + br.width * (evt.targetSide === 'player' ? ppPct : epPct) / 100,
              br.top + br.height / 2,
              evt.color, 8);
      }

      // Update axium meter
      updateAxiumMeter(B.playerPlayed);
      break;
    }

    case 'battle_end': {
      const pop = gel('resolve-pop');
      let endLabel, endColor;
      if (evt.winner === 'player') {
        endLabel = evt.reason === 'enemy_zero' ? 'Enemy Collapsed' : 'Attention Held';
        endColor = '#86EFAC';
      } else {
        if (evt.reason === 'time') {
          const pPct = Math.round((evt.pPct ?? 0) * 100);
          const ePct = Math.round((evt.ePct ?? 0) * 100);
          endLabel = `Time — ${pPct}% vs ${ePct}%`;
        } else {
          endLabel = 'Attention Lost';
        }
        endColor = '#e05555';
      }
      if (pop) { pop.textContent = endLabel; pop.style.color = endColor; pop.classList.add('show'); }

      burst(window.innerWidth / 2, window.innerHeight * 0.5, endColor, 28);

      setTimeout(() => {
        if (pop) pop.classList.remove('show');
        finaliseBattle(evt.winner);
      }, 1200);
      break;
    }
  }
}

// ─────────────────────────────────────────────────────────
// CARD HIGHLIGHT FLASH
// ─────────────────────────────────────────────────────────
function _flashCard(side, cardId, color) {
  const fieldId = side === 'player' ? 'field-player' : 'field-enemy';
  const field   = gel(fieldId); if (!field) return;
  const cardEl  = field.querySelector(`[data-card-id="${cardId}"]`); if (!cardEl) return;
  const prev    = cardEl.style.boxShadow;
  cardEl.style.boxShadow  = `0 0 28px ${color}, 0 0 8px rgba(255,255,255,.3)`;
  cardEl.style.transform  = 'translateY(-6px) scale(1.06)';
  cardEl.style.transition = 'box-shadow .15s, transform .15s';
  setTimeout(() => {
    cardEl.style.boxShadow  = prev;
    cardEl.style.transform  = '';
  }, 400);
}

// ─────────────────────────────────────────────────────────
// FLOAT LABEL OVER CARD
// Spawns a rising delta label centred above the triggering card.
// ─────────────────────────────────────────────────────────
function _floatLabelOverCard(side, cardId, text, color) {
  if (!text) return;
  const fieldId = side === 'player' ? 'field-player' : 'field-enemy';
  const field   = gel(fieldId); if (!field) return;
  const cardEl  = field.querySelector(`[data-card-id="${cardId}"]`); if (!cardEl) return;

  const rect = cardEl.getBoundingClientRect();
  const el   = document.createElement('div');
  el.className = 'bar-float-lbl';
  el.textContent = text;
  el.style.cssText = `
    position: fixed;
    left: ${rect.left + rect.width / 2}px;
    top:  ${rect.top - 8}px;
    color: ${color};
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: .04em;
    white-space: nowrap;
    pointer-events: none;
    text-shadow: 0 0 10px ${color};
    z-index: 500;
    transform: translateX(-50%);
    animation: floatUp .9s ease-out forwards;
  `;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 950);
}

// ─────────────────────────────────────────────────────────
// FINALISE + WIN/LOSE
// ─────────────────────────────────────────────────────────
function finaliseBattle(winner) {
  B.phase = 'done';
  _syncBarsDisplay();
  // Remove surge overlay
  document.querySelector('.battle-surge-overlay')?.classList.remove('active');
  _setFieldLocked(false);

  if (winner === 'player') {
    B.won  = true;
    const perfect = B.playerPlayed.length >= 3 &&
      B.playerPlayed.every(e => (e.card?.axiumScore || 0) >= 9);
    APP.save = AxiumSave.recordBattle(CHAPTER_ID, true, B.playerAttn, B.enemyAttn);
    burst(window.innerWidth / 2, window.innerHeight / 2, '#D4AF37', 44);
    B.synsFired = B.synergyList.length;
    showOutcome(true, perfect, B);
  } else {
    B.lost = true;
    APP.save = AxiumSave.recordBattle(CHAPTER_ID, false, B.playerAttn, B.enemyAttn);
    burst(window.innerWidth / 2, window.innerHeight / 2, '#DC2626', 28);
    B.synsFired = B.synergyList.length;
    showOutcome(false, false, B);
  }
}

// ─────────────────────────────────────────────────────────
// UTILITY
// ─────────────────────────────────────────────────────────
function _hexRGBSimple(hex) {
  const h = hex.replace('#','');
  if (h.length === 3) return [parseInt(h[0]+h[0],16), parseInt(h[1]+h[1],16), parseInt(h[2]+h[2],16)];
  return [parseInt(h.slice(0,2),16), parseInt(h.slice(2,4),16), parseInt(h.slice(4,6),16)];
}

// passAndEndTurn removed — no longer relevant in real-time model.
// The battle runs its full sim once Resolve is pressed.

window.B = B;
