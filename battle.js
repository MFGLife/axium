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
// TIER INTERVAL — kept for surge transition reference only.
// The actual per-card fire rate is nodeFireInterval() in cards.js.
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
  B.enemyAttn     = CHAPTER_CONFIG.enemyStart;
  B.enemyMaxAttn  = CHAPTER_CONFIG.enemyMax;
  CHAPTER_CONFIG._stageApplied = false;  // allow stage def to override attn on buildEnemyHand
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
  // Use CHAPTER_CONFIG.stage if set (supports 10-stage progression).
  // Falls back to CHAPTER_CONFIG.enemyDeck for backward compatibility,
  // then falls back to stage 1 if neither is defined.
  const stage = CHAPTER_CONFIG.stage || 1;

  let pool;
  if (CHAPTER_CONFIG.enemyDeck) {
    // Legacy explicit deck — still respected for custom chapter configs
    pool = CHAPTER_CONFIG.enemyDeck.map(id => getCard(id)).filter(Boolean);
  } else if (typeof getStageEnemyDeck === 'function') {
    const stageDef = getStageEnemyDeck(stage);
    // Override starting attn/max from stage definition if not explicitly set
    if (!CHAPTER_CONFIG._stageApplied) {
      B.enemyAttn    = stageDef.enemyStart ?? B.enemyAttn;
      B.enemyMaxAttn = stageDef.enemyMax   ?? B.enemyMaxAttn;
      CHAPTER_CONFIG._stageApplied = true;
    }
    pool = stageDef.cards || [];
  } else {
    pool = shuffle([...PLAYER_CARDS]).slice(0, CHAPTER_CONFIG.enemyHandSize || 4);
  }

  const handSize = CHAPTER_CONFIG.enemyHandSize || pool.length || 4;
  return pool.slice(0, handSize).map(card => ({
    card,
    reversed: true,   // enemy always plays reversed — attacks player
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
// PRE-SIMULATION — Node Resonance Engine v4.0
// Runs the full 60s battle synchronously.
// Returns { events[], synergies[] }
//
// Fire speed = tierInterval / card.pts.length
// ID:        fires continuously — recharge (upright) or drain (reversed)
// Superego:  first fire = shield burst / enemy burst;
//            subsequent fires = capacity growth / capacity erosion
// Ego:       fires once — sets pMult/pFlat (upright) or eDivisor/eDmgFlat (reversed)
// ─────────────────────────────────────────────────────────
function preSimulateBattle() {
  const STEP        = 100;
  const NORMAL_END  = 45000;
  const BATTLE_END  = 60000;
  const LOSE_THRESH = CHAPTER_CONFIG.loseAttn;
  const WIN_PCT_MARGIN = CHAPTER_CONFIG.winPctMargin ?? 0.10;

  const events = [];

  // ── Mutable sim state ──────────────────────────────────
  let pAttn    = B.playerAttn;
  let eAttn    = B.enemyAttn;
  let pMax     = B.playerMaxAttn;
  let eMax     = B.enemyMaxAttn;
  let pMult    = 1.0;
  let pFlat    = 0;
  let eDivisor = 1.0;
  let eDmgFlat = 0;

  // ── Mechanic 3: Polarity Momentum ──────────────────────
  // reversalStack increments each time a reversed player ID fires,
  // decays by REVERSAL_DECAY on every non-reversed player ID fire.
  // Once stack ≥ threshold, a fragmentation debuff halves pMult.
  let reversalStack    = 0;
  let fragmented       = false;  // debuff applied once per battle

  // ── Mechanic 5: Surge Fragmentation ────────────────────
  // On the first fire cycle after surge, reversed player ID cards
  // split their drain 50/50 between enemy and player.
  // surgeSplitActive is true only for the first STEP after surge.
  let surgeSplitActive = false;
  let surgeFirstCycle  = true;   // flag: have we run the first surge STEP?

  // ── Build card entries ─────────────────────────────────
  const cards = [
    ...B.playerPlayed.map(e => {
      const interval = nodeFireInterval(e.card, false);
      return {
        id:         e.card.id,
        card:       e.card,
        reversed:   e.reversed,
        side:       'player',
        tier:       e.card.tier || 1,
        interval,
        nextFire:   interval,
        progress:   0,
        firstFired: false,
        fireCount:  0,   // Mechanic 1: Resonance Decay
        bleedCount: 0,   // Mechanic 6: Capacity Bleed
        isEgo:      e.card.layer === 'Ego',
      };
    }),
    ...B.enemyHand.map(e => {
      const interval = nodeFireInterval(e.card, false);
      return {
        id:         e.card.id,
        card:       e.card,
        reversed:   true,
        side:       'enemy',
        tier:       e.card.tier || 1,
        interval,
        nextFire:   interval,
        progress:   0,
        firstFired: false,
        fireCount:  0,
        bleedCount: 0,
        isEgo:      e.card.layer === 'Ego',
      };
    }),
  ];

  // ── Synergies at t=0 ───────────────────────────────────
  const { attnBoost, pMultBonus, activeSynergies } = getSynergyBonuses(B.playerPlayed);
  pAttn  = clamp(pAttn + attnBoost, 0, pMax);
  pMult *= pMultBonus;

  // Instant win synergy (The Axium)
  if (activeSynergies.some(s => s.effect?.instantWin)) {
    events.push({ t: 0, type: 'battle_end', winner: 'player', reason: 'instant_win' });
    return { events, synergies: activeSynergies };
  }

  if (attnBoost > 0) {
    events.push({
      t: 0, type: 'card_trigger', side: 'player', cardId: 'synergy',
      delta: attnBoost, label: `✦ +${attnBoost} Synergy`, color: '#D4AF37',
      targetSide: 'player',
      newPlayerAttn: pAttn, newEnemyAttn: eAttn,
      newPlayerMax: pMax,   newEnemyMax:  eMax,
    });
  }

  // ── Main loop ──────────────────────────────────────────
  let battleEnded = false;

  for (let t = STEP; t <= BATTLE_END && !battleEnded; t += STEP) {
    const surge = t > NORMAL_END;

    // Surge transition — recalibrate intervals and nextFire
    if (t === NORMAL_END + STEP) {
      events.push({ t, type: 'surge_start' });
      surgeSplitActive = true;   // Mechanic 5: first surge cycle
      surgeFirstCycle  = true;
      cards.forEach(c => {
        const oldInterval = nodeFireInterval(c.card, false);
        const newInterval = nodeFireInterval(c.card, true);
        c.interval = newInterval;
        const elapsed = oldInterval - (c.nextFire - t + STEP);
        const pct     = clamp(elapsed / oldInterval, 0, 1);
        c.nextFire    = t + Math.round(newInterval * (1 - pct));
      });
    }

    // Mechanic 5: clear surgeSplitActive after one full STEP
    if (surge && surgeFirstCycle && t > NORMAL_END + STEP) {
      surgeSplitActive = false;
      surgeFirstCycle  = false;
    }

    // Charge progress events for visual layer
    cards.forEach(c => {
      const elapsed = c.interval - (c.nextFire - t);
      c.progress    = clamp(elapsed / c.interval, 0, 1);
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

      // Ego fires once — park after first fire
      if (c.isEgo && c.firstFired) {
        c.nextFire = Infinity;
        continue;
      }

      // ── Build simCtx for the six mechanics ─────────────
      // Mechanic 4: Constellation Interference — same-suit suppression
      const suitSuppressCount = (c.side === 'player' && !c.reversed)
        ? getSuitSuppressCount(c.card, B.playerPlayed)
        : 0;

      const simCtx = {
        pAttn,
        pMax,
        eAttn,
        eMax,
        suitSuppressCount,                       // Mechanic 4
        surgeSplitActive: surge && surgeSplitActive && c.side === 'player' && c.reversed
                          && c.card.layer === 'ID', // Mechanic 5
        reversalMult: reversalStack,              // Mechanic 3
        enemyDrainMult: (typeof CHAPTER_CONFIG !== 'undefined' && CHAPTER_CONFIG.enemyDrainMult) || 1.0,
      };

      // Pass per-card fire/bleed counts for Mechanics 1 & 6
      const cWithCounts = Object.assign({}, c, {
        fireCount:  c.fireCount,
        bleedCount: c.bleedCount,
      });

      const result = computeCardFire(cWithCounts, pMult, pFlat, eDivisor, eDmgFlat, simCtx);
      const { delta, capacityDelta, splitDelta, label, color, targetSide, egoEffect, skipped } = result;

      // Threshold lock — card condition not met, advance timer and continue
      if (skipped) {
        c.nextFire = t + nodeFireInterval(c.card, surge);
        continue;
      }

      // ── Apply Ego modifier ──────────────────────────────
      if (egoEffect) {
        if (egoEffect.type === 'boost') {
          pMult = egoEffect.pMult;
          pFlat = egoEffect.pFlat;
        } else if (egoEffect.type === 'divide' || egoEffect.type === 'enemy_divide') {
          eDivisor = egoEffect.eDivisor;
          eDmgFlat = egoEffect.eDmgFlat;
        }
        c.firstFired = true;
        c.fireCount++;
        c.nextFire   = Infinity;

      } else {
        // ── Apply attn delta ────────────────────────────
        if (delta !== 0) {
          if (targetSide === 'player') {
            pAttn = clamp(pAttn + delta, 0, pMax);
          } else {
            eAttn = clamp(eAttn + delta, 0, eMax);
          }
        }

        // ── Mechanic 5: Surge Fragmentation backfire ────
        // splitDelta is the portion of drain that bounces to the
        // firing side's own attn (always negative = player takes damage).
        if (splitDelta && splitDelta !== 0) {
          pAttn = clamp(pAttn + splitDelta, 0, pMax);
        }

        // ── Apply capacity delta ─────────────────────────
        if (capacityDelta !== 0) {
          if (targetSide === 'player' && capacityDelta > 0) {
            pMax = clamp(pMax + capacityDelta, 10, 400);
          } else if (targetSide === 'player' && capacityDelta < 0) {
            pMax  = clamp(pMax + capacityDelta, 10, 400);
            pAttn = clamp(pAttn, 0, pMax);
          } else if (targetSide === 'enemy' && capacityDelta < 0) {
            eMax  = clamp(eMax + capacityDelta, 10, 400);
            eAttn = clamp(eAttn, 0, eMax);
          }
          // Mechanic 6: increment bleedCount on every Superego subsequent fire
          if (c.card.layer === 'Superego' && c.firstFired) c.bleedCount++;
        }

        // ── Mechanic 3: Polarity Momentum update ─────────
        if (c.side === 'player' && c.card.layer === 'ID') {
          if (c.reversed) {
            // Reversed player ID — increment reversal stack
            reversalStack = Math.min(reversalStack + 1, 10);
          } else {
            // Upright player ID — decay stack
            reversalStack = Math.max(0, reversalStack - REVERSAL_DECAY);
          }

          // Apply fragmentation debuff once when threshold crossed
          if (!fragmented && reversalStack >= REVERSAL_FRAG_THRESHOLD) {
            fragmented = true;
            pMult      = Math.max(0.1, pMult - REVERSAL_FRAG_PENALTY);
            events.push({
              t,
              type:   'card_trigger', side: 'player', cardId: c.id,
              delta:  0, label: '⚑ Fragmented', color: '#9333EA',
              targetSide: 'player',
              newPlayerAttn: pAttn, newEnemyAttn: eAttn,
              newPlayerMax:  pMax,  newEnemyMax:  eMax,
            });
          }
        }

        c.firstFired = true;
        c.fireCount++;

        // Reset timer
        c.nextFire = t + nodeFireInterval(c.card, surge);
        c.progress = 0;
      }

      // Record event
      const effectiveDelta = (splitDelta && splitDelta !== 0)
        ? delta + splitDelta  // show combined for label purposes
        : delta;
      if (effectiveDelta !== 0 || capacityDelta !== 0 || egoEffect) {
        events.push({
          t,
          type:          'card_trigger',
          side:          c.side,
          cardId:        c.id,
          reversed:      c.reversed,
          delta:         effectiveDelta,
          capacityDelta,
          label,
          color,
          targetSide,
          newPlayerAttn: pAttn,
          newEnemyAttn:  eAttn,
          newPlayerMax:  pMax,
          newEnemyMax:   eMax,
          pMult, pFlat, eDivisor, eDmgFlat,
        });
      }

      // Early KO checks
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

  // Time expiry — percentage margin decides winner
  if (!battleEnded) {
    const pPct   = pAttn / pMax;
    const ePct   = eAttn / eMax;
    const winner = (pPct - ePct) >= WIN_PCT_MARGIN ? 'player' : 'enemy';
    events.push({ t: BATTLE_END, type: 'battle_end', winner, reason: 'time', pPct, ePct });
  }

  return { events, synergies: activeSynergies };
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
