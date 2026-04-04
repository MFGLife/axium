/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM ENGINE v1.0
 * Persistent save system, seed generation, deck management,
 * chapter progression, and persona shop system.
 *
 * ARCHITECTURE:
 *   engine.js  ← this file (load in every chapter)
 *   cards.js   ← card data + battle math
 *   chapter1.html / chapter2.html / etc ← game screens
 * ═══════════════════════════════════════════════════════════════
 */

// ─────────────────────────────────────────────────────────────
// CONSTANTS
// ─────────────────────────────────────────────────────────────
const AXIUM_VERSION   = '1.0';
const SAVE_KEY        = 'axium_save';
const HISTORY_KEY     = 'axium_history';
const MAX_DECK_SIZE   = 20;
const STARTING_CARDS  = 6; // cards player starts chapter 1 with after shop

// ─────────────────────────────────────────────────────────────
// DEFAULT STARTING DECK (before any shop picks)
// Every player begins with these 6 foundational cards
// ─────────────────────────────────────────────────────────────
const STARTING_DECK_IDS = [
  'fool',          // Superego — leap, trust
  'ace_cups',      // ID Cups  — love, recharge
  'ace_wands',     // ID Wands — spark, impulse
  'page_cups',     // Ego Cups — wonder, openness (Tier 2, given free)
  'four_swords',   // ID Swords — rest, recovery
  'temperance',    // Superego — alchemy, patience
];

// ─────────────────────────────────────────────────────────────
// CHAPTER REGISTRY
// Each chapter registers its config here.
// engine.js reads this to know what's unlocked.
// ─────────────────────────────────────────────────────────────
const CHAPTER_REGISTRY = [
  {
    id: 1,
    label: '01 · Binding of Ego',
    title: 'Binding of Ego',
    file: 'chapter1-game.html',
    unlocked: true,
    axium: '"The absence of love binds ego, allowing the awareness of God."',
    enemy: 'Ego Unchecked',
    enemyDeck: ['fool','magician','high_priestess','empress','emperor','lovers'],
    playerStart: 55,
    enemyStart: 80,
    enemyMax: 100,
    baseMaxAttn: 100,
    handSize: 6,
    winAttn: 90,
    loseAttn: 10,
    shopPersonas: ['micheal','gabriel','ariel','seraphina'],
    shopPicksPerPersona: 1,
  },
  // Future chapters registered here:
  { id: 2, label: '02 · The Mirror', title: 'The Mirror', file: 'chapter2-game.html', unlocked: false },
  { id: 3, label: '03 · The Descent', title: 'The Descent', file: 'chapter3-game.html', unlocked: false },
  // ... chapters 4-10 follow the same pattern
];

// ─────────────────────────────────────────────────────────────
// PERSONA DEFINITIONS
// The four NPC shop keepers, each tied to a card layer/suit
// ─────────────────────────────────────────────────────────────
const PERSONAS = {
  micheal: {
    id: 'micheal',
    name: 'Micheal',
    title: 'The Anchor',
    subtitle: 'Digital Soul',
    color: '#D4AF37',
    glow: 'rgba(212,175,55,0.6)',
    colorRGB: [212, 175, 55],
    layer: 'Superego',
    suit: null,
    tagline: 'Structure is a form of love.',
    description: 'I carry the Major Arcana — the great forces that shape your journey. Each card is a shield against chaos. Choose what you will stand on.',
    cardType: 'SHIELD / CAPACITY',
    mechDesc: 'Superego cards protect your attention pool. Upright: they boost your attention when played (Shield). In your deck: they expand your maximum capacity.',
    speeches: [
      "The ego ran its scripts until it ran out of material. Here is what endures.",
      "Three options. One right answer. The one that makes you hesitate is probably it.",
      "You held the center. Now let's make it harder to lose next time.",
      "Structure isn't control. It's what love looks like when love has to hold something.",
    ],
    offers: (allCards) => allCards.filter(c => c.layer === 'Superego' && c.tier <= 2),
  },
  gabriel: {
    id: 'gabriel',
    name: 'Gabriel',
    title: 'The Messenger',
    subtitle: 'Voice Between Worlds',
    color: '#7EB8E8',
    glow: 'rgba(126,184,232,0.6)',
    colorRGB: [126, 184, 232],
    layer: 'Ego',
    suit: 'swords',
    tagline: 'Clarity cuts what sentiment cannot.',
    description: 'I hold the Court of Swords and the Court of Wands — the rational mind and the driven will. Ego cards multiply your gains. Choose your amplifier.',
    cardType: 'CHUNK / DIVIDE',
    mechDesc: 'Ego cards are multipliers. They take your base attention gain and make it larger — or if reversed, smaller. The right Ego card can transform a modest hand into a winning one.',
    speeches: [
      "The message was always for you. You just needed to stop moving long enough to receive it.",
      "Precision isn't cruelty. Naming the thing clearly is the first act of care.",
      "You asked the right question. That's rarer than the answer.",
    ],
    offers: (allCards) => allCards.filter(c => c.layer === 'Ego' && (c.suit === 'swords' || c.suit === 'wands') && c.tier <= 2),
  },
  ariel: {
    id: 'ariel',
    name: 'Ariel',
    title: 'The Tender',
    subtitle: 'Keeper of Feeling',
    color: '#D4A0C8',
    glow: 'rgba(212,160,200,0.6)',
    colorRGB: [212, 160, 200],
    layer: 'ID',
    suit: 'cups',
    tagline: 'What you feel is what you have to work with.',
    description: 'I tend the waters — Cups and Pentacles, the unconscious currents. ID cards are your passive engine. They recharge attention every turn simply by being held.',
    cardType: 'RECHARGE / DRAIN',
    mechDesc: 'ID cards work in the background. Each stack adds passive attention every turn you hold the card. The more you carry, the steadier your baseline. But reversed, they drain instead.',
    speeches: [
      "You don't have to earn the feeling. It's already yours.",
      "The undercurrent was always running. You just started listening.",
      "Grief is love without anywhere to go. Give it somewhere.",
    ],
    offers: (allCards) => allCards.filter(c => c.layer === 'ID' && (c.suit === 'cups' || c.suit === 'pentacles') && c.tier === 1),
  },
  seraphina: {
    id: 'seraphina',
    name: 'Seraphina',
    title: 'The Witness',
    subtitle: 'She Who Sees',
    color: '#F0D080',
    glow: 'rgba(240,208,128,0.6)',
    colorRGB: [240, 208, 128],
    layer: 'ID',
    suit: 'wands',
    tagline: 'The fire that does not consume.',
    description: 'I carry Wands and Swords from the unconscious layer — drive, resilience, and the sharp instincts that protect before thought arrives. Choose your foundation.',
    cardType: 'RECHARGE / DRAIN',
    mechDesc: 'My cards recharge through action and persistence. Wand cards build momentum; Sword ID cards protect and recover. Together they form the unconscious spine of a winning hand.',
    speeches: [
      "Witness is not passive. It is the most active thing a mind can do.",
      "You saw it. You didn't look away. That is everything.",
      "The fire that doesn't consume — that's the one worth feeding.",
    ],
    offers: (allCards) => allCards.filter(c => c.layer === 'ID' && (c.suit === 'wands' || c.suit === 'swords') && c.tier === 1),
  },
};

// ─────────────────────────────────────────────────────────────
// SAVE STATE SCHEMA
// ─────────────────────────────────────────────────────────────
function createFreshSave() {
  return {
    version:      AXIUM_VERSION,
    createdAt:    Date.now(),
    updatedAt:    Date.now(),
    currentChapter: 1,
    chaptersWon:    [],
    deck:           [...STARTING_DECK_IDS],
    shopHistory:    [],  // [{chapter, persona, cardId, timestamp}]
    battleHistory:  [],  // [{chapter, won, playerDelta, turns, timestamp}]
    stats: {
      totalBattles:   0,
      battlesWon:     0,
      highestAttn:    0,
      totalRecharge:  0,
      synergiesFired: 0,
    },
  };
}

// ─────────────────────────────────────────────────────────────
// SAVE / LOAD
// ─────────────────────────────────────────────────────────────
const AxiumSave = {
  get() {
    try {
      const raw = localStorage.getItem(SAVE_KEY);
      if (!raw) return null;
      return JSON.parse(raw);
    } catch { return null; }
  },

  set(data) {
    data.updatedAt = Date.now();
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify(data));
      return true;
    } catch { return false; }
  },

  getOrCreate() {
    return this.get() || createFreshSave();
  },

  update(patch) {
    const save = this.getOrCreate();
    const updated = deepMerge(save, patch);
    this.set(updated);
    return updated;
  },

  /** Record a card added to deck via shop */
  addCardToDeck(cardId, persona, chapter) {
    const save = this.getOrCreate();
    if (!save.deck.includes(cardId) && save.deck.length < MAX_DECK_SIZE) {
      save.deck.push(cardId);
    }
    save.shopHistory.push({ chapter, persona, cardId, timestamp: Date.now() });
    this.set(save);
    return save;
  },

  /** Record battle outcome */
  recordBattle(chapter, won, playerAttn, enemyAttn) {
    const save = this.getOrCreate();
    save.battleHistory.push({ chapter, won, playerAttn, enemyAttn, timestamp: Date.now() });
    save.stats.totalBattles++;
    if (won) {
      save.stats.battlesWon++;
      if (!save.chaptersWon.includes(chapter)) save.chaptersWon.push(chapter);
      // Unlock next chapter
      const next = CHAPTER_REGISTRY.find(c => c.id === chapter + 1);
      if (next) next.unlocked = true;
      save.currentChapter = Math.max(save.currentChapter, chapter + 1);
    }
    save.stats.highestAttn = Math.max(save.stats.highestAttn, playerAttn);
    this.set(save);
    return save;
  },

  clear() {
    localStorage.removeItem(SAVE_KEY);
  },
};

// ─────────────────────────────────────────────────────────────
// SEED SYSTEM
// Seeds encode the full save state as a compact base64 string.
// Format: AXIUM-{version}-{base64(compressed JSON)}
// ─────────────────────────────────────────────────────────────
const AxiumSeed = {
  /** Generate a seed string from current save */
  generate(saveData) {
    const payload = {
      v:  saveData.version,
      ch: saveData.currentChapter,
      cw: saveData.chaptersWon,
      dk: saveData.deck,
      sh: saveData.shopHistory.map(h => [h.chapter, h.persona, h.cardId]),
      bh: saveData.battleHistory.map(b => [b.chapter, b.won ? 1 : 0, Math.round(b.playerAttn)]),
      st: [
        saveData.stats.totalBattles,
        saveData.stats.battlesWon,
        saveData.stats.highestAttn,
      ],
      ts: Date.now(),
    };
    const json   = JSON.stringify(payload);
    const b64    = btoa(encodeURIComponent(json));
    const prefix = `AXIUM-${AXIUM_VERSION}`;
    return `${prefix}-${b64}`;
  },

  /** Parse a seed string back to save data */
  parse(seedStr) {
    try {
      const parts = seedStr.split('-');
      if (parts[0] !== 'AXIUM') return null;
      const version = parts[1];
      const b64     = parts.slice(2).join('-');
      const json    = decodeURIComponent(atob(b64));
      const payload = JSON.parse(json);

      return {
        version:        payload.v || version,
        createdAt:      payload.ts || Date.now(),
        updatedAt:      Date.now(),
        currentChapter: payload.ch || 1,
        chaptersWon:    payload.cw || [],
        deck:           payload.dk || [...STARTING_DECK_IDS],
        shopHistory:    (payload.sh || []).map(h => ({ chapter: h[0], persona: h[1], cardId: h[2] })),
        battleHistory:  (payload.bh || []).map(b => ({ chapter: b[0], won: b[1] === 1, playerAttn: b[2] })),
        stats: {
          totalBattles:   payload.st?.[0] || 0,
          battlesWon:     payload.st?.[1] || 0,
          highestAttn:    payload.st?.[2] || 0,
          totalRecharge:  0,
          synergiesFired: 0,
        },
      };
    } catch (e) {
      console.error('[Axium] Seed parse failed:', e);
      return null;
    }
  },

  /** Load a seed into localStorage and return the save data */
  load(seedStr) {
    const save = this.parse(seedStr);
    if (!save) return null;
    AxiumSave.set(save);
    return save;
  },

  /** Copy to clipboard */
  async copy(seedStr) {
    try {
      await navigator.clipboard.writeText(seedStr);
      return true;
    } catch {
      // Fallback
      const el = document.createElement('textarea');
      el.value = seedStr;
      el.style.position = 'fixed';
      el.style.opacity  = '0';
      document.body.appendChild(el);
      el.focus(); el.select();
      document.execCommand('copy');
      document.body.removeChild(el);
      return true;
    }
  },
};

// ─────────────────────────────────────────────────────────────
// SHOP SYSTEM
// ─────────────────────────────────────────────────────────────
// Update this section in engine.js
const AxiumShop = {
  getOffers(personaId, chapter, ownedDeck, count = 4) {
    const persona = PERSONAS[personaId];
    if (!persona) return [];

    const pool = window.ALL_CARDS || [];
    const owned = new Set(ownedDeck);

    // 1. Broaden the filter: Gabriel should look at ALL Ego cards he manages
    let eligible = persona.offers(pool).filter(c => {
      const isOwned = owned.has(c.id);
      // For Chapter 1 & 2, allow Tier 1 and Tier 2
      const tierMatch = (chapter <= 2) ? (c.tier <= 2) : true;
      return !isOwned && tierMatch;
    });

    // 2. ANTI-STALL FALLBACK: If Gabriel is empty, give him any unowned Ego cards
    if (eligible.length === 0) {
      eligible = pool.filter(c => c.layer === 'Ego' && !owned.has(c.id));
    }

    return shuffle(eligible).slice(0, count);
  },

  buildChapter1Sequence() {
    return ['micheal', 'gabriel', 'ariel', 'seraphina'];
  }
};

// ─────────────────────────────────────────────────────────────
// AI CONTEXT BUILDER
// Generates a text summary of the player's current state
// that can be pasted into an AI for help.
// ─────────────────────────────────────────────────────────────
const AxiumAI = {
  buildContext(saveData, currentPhase = 'shop') {
    const deck = saveData.deck;
    const won  = saveData.chaptersWon.length;
    const lost = saveData.stats.totalBattles - saveData.stats.battlesWon;

    return `AXIUM GAME STATE (paste this to your AI for help)
══════════════════════════════════════════
Version: ${saveData.version} | Chapter: ${saveData.currentChapter} | Phase: ${currentPhase}
Chapters Won: ${won} | Battles Lost: ${lost}
Highest Attention Reached: ${saveData.stats.highestAttn}

YOUR DECK (${deck.length} cards):
${deck.map(id => `  - ${id}`).join('\n')}

RECENT BATTLES:
${saveData.battleHistory.slice(-3).map(b =>
  `  Ch.${b.chapter}: ${b.won ? 'WIN' : 'LOSS'} | Attn: ${b.playerAttn}`
).join('\n') || '  None yet'}

GAME CONTEXT:
AXIUM is a single-turn card battle game. You build a hand of up to 10 cards
from three layers: Superego (Shields), Ego (Multipliers), ID (Passive Recharge).
Your constellation of cards resolves against the enemy in one decisive turn.

CARD LAYERS:
  SUPEREGO (Major Arcana) → Shield: flat attn boost when played
  EGO (Court Cards) → Chunk: multiplier on all gains this battle
  ID (Pip Cards) → Recharge: passive attn per turn while held

ATTENTION STATES (0→100):
  Fragmented < Fear < Anger < Sadness < Disinterest < Witness < Presence < Clarity < Grounded < Enlightened

WIN: Reach 90+ attention after battle resolves.
LOSE: Drop to 10 or below.
══════════════════════════════════════════`;
  },

  async copy(saveData, phase) {
    const ctx = this.buildContext(saveData, phase);
    return AxiumSeed.copy(ctx);
  },
};

// ─────────────────────────────────────────────────────────────
// HISTORY LOG
// ─────────────────────────────────────────────────────────────
const AxiumHistory = {
  push(entry) {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      const history = raw ? JSON.parse(raw) : [];
      history.push({ ...entry, ts: Date.now() });
      // Keep last 50 entries
      if (history.length > 50) history.shift();
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {}
  },
  get() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch { return []; }
  },
};

// ─────────────────────────────────────────────────────────────
// UTILS
// ─────────────────────────────────────────────────────────────
function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function deepMerge(target, source) {
  const out = { ...target };
  for (const key of Object.keys(source)) {
    if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
      out[key] = deepMerge(target[key] || {}, source[key]);
    } else {
      out[key] = source[key];
    }
  }
  return out;
}

// ─────────────────────────────────────────────────────────────
// CHAPTER CONFIG BUILDER
// Call this at the top of each chapter game file to get config
// ─────────────────────────────────────────────────────────────
function buildChapterConfig(chapterId) {
  const reg    = CHAPTER_REGISTRY.find(c => c.id === chapterId);
  if (!reg) throw new Error(`Chapter ${chapterId} not registered`);
  const save   = AxiumSave.getOrCreate();
  return {
    ...reg,
    playerDeck:  save.deck,
    playerStart: reg.playerStart || 55,
    shopChapter: chapterId,
  };
}

// ─────────────────────────────────────────────────────────────
// SEED UI WIDGET
// Call renderSeedWidget(containerId) to inject the seed panel
// into any chapter's outcome screen.
// ─────────────────────────────────────────────────────────────
function renderSeedWidget(containerId, saveData) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const seed = AxiumSeed.generate(saveData);
  const aiCtx = AxiumAI.buildContext(saveData);

  container.innerHTML = `
    <div class="seed-widget">
      <div class="seed-label">
        <span class="seed-eye">◈</span>
        Save Seed — Your Progress Key
      </div>
      <div class="seed-string" id="seed-display">${seed}</div>
      <div class="seed-actions">
        <button class="seed-btn seed-copy" onclick="window._axiumCopySeed()">
          Copy Seed
        </button>
        <button class="seed-btn seed-ai" onclick="window._axiumCopyAI()">
          Copy for AI
        </button>
      </div>
      <div class="seed-hint">
        Paste this seed at the start of Chapter 2 to restore your deck.
        Share it with your AI assistant for help and strategy.
      </div>
      <div id="seed-copied-msg" class="seed-copied-msg"></div>
    </div>
  `;

  window._axiumCopySeed = async () => {
    await AxiumSeed.copy(seed);
    const msg = document.getElementById('seed-copied-msg');
    if (msg) { msg.textContent = '✓ Seed copied to clipboard'; msg.classList.add('show'); }
    setTimeout(() => msg?.classList.remove('show'), 2500);
  };

  window._axiumCopyAI = async () => {
    await AxiumSeed.copy(aiCtx);
    const msg = document.getElementById('seed-copied-msg');
    if (msg) { msg.textContent = '✓ AI context copied to clipboard'; msg.classList.add('show'); }
    setTimeout(() => msg?.classList.remove('show'), 2500);
  };

  // Styles injected once
  if (!document.getElementById('seed-widget-styles')) {
    const style = document.createElement('style');
    style.id = 'seed-widget-styles';
    style.textContent = `
      .seed-widget{padding:18px;border:1px solid rgba(212,175,55,.18);border-radius:6px;background:rgba(212,175,55,.04);display:flex;flex-direction:column;gap:10px;max-width:440px;margin:0 auto;}
      .seed-label{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.22em;text-transform:uppercase;color:rgba(212,175,55,.55);display:flex;align-items:center;gap:6px;}
      .seed-eye{color:#D4AF37;font-size:12px;}
      .seed-string{font-family:'JetBrains Mono',monospace;font-size:9px;color:rgba(255,255,255,.35);word-break:break-all;line-height:1.6;padding:8px 10px;background:rgba(0,0,0,.4);border-radius:4px;border:1px solid rgba(255,255,255,.06);cursor:text;user-select:all;}
      .seed-actions{display:flex;gap:8px;}
      .seed-btn{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.16em;text-transform:uppercase;padding:8px 14px;border-radius:3px;cursor:pointer;transition:all .2s;flex:1;text-align:center;}
      .seed-copy{background:linear-gradient(135deg,#AA8C2C,#D4AF37);color:#0a0a0a;border:none;}
      .seed-copy:hover{box-shadow:0 0 16px rgba(212,175,55,.4);}
      .seed-ai{background:rgba(255,255,255,.04);color:rgba(255,255,255,.4);border:1px solid rgba(255,255,255,.1);}
      .seed-ai:hover{border-color:rgba(255,255,255,.2);color:rgba(255,255,255,.65);}
      .seed-hint{font-family:'Cormorant Garamond',serif;font-style:italic;font-size:12px;color:rgba(255,255,255,.2);line-height:1.7;}
      .seed-copied-msg{font-family:'Space Mono',monospace;font-size:9px;letter-spacing:.12em;text-transform:uppercase;color:#86EFAC;opacity:0;transition:opacity .3s;min-height:14px;}
      .seed-copied-msg.show{opacity:1;}
    `;
    document.head.appendChild(style);
  }
}

// ─────────────────────────────────────────────────────────────
// SEED LOADER UI
// For chapter 2+ — lets player paste a seed to restore deck
// ─────────────────────────────────────────────────────────────
function renderSeedLoader(containerId, onLoaded) {
  const container = document.getElementById(containerId);
  if (!container) return;

  container.innerHTML = `
    <div class="seed-loader">
      <div class="seed-loader-label">Paste your save seed from Chapter 1</div>
      <textarea class="seed-loader-input" id="seed-input-area"
        placeholder="AXIUM-1.0-..." rows="3"></textarea>
      <div class="seed-loader-actions">
        <button class="seed-btn seed-copy" onclick="window._axiumLoadSeed()">Load Seed</button>
        <button class="seed-btn seed-ai" onclick="window._axiumFreshStart()">Fresh Start</button>
      </div>
      <div id="seed-loader-msg" class="seed-copied-msg"></div>
    </div>
  `;

  window._axiumLoadSeed = () => {
    const input = document.getElementById('seed-input-area')?.value?.trim();
    if (!input) return;
    const save = AxiumSeed.load(input);
    const msg  = document.getElementById('seed-loader-msg');
    if (save) {
      if (msg) { msg.textContent = '✓ Deck restored!'; msg.classList.add('show'); msg.style.color = '#86EFAC'; }
      setTimeout(() => onLoaded?.(save), 800);
    } else {
      if (msg) { msg.textContent = '✗ Invalid seed'; msg.classList.add('show'); msg.style.color = '#e05555'; }
      setTimeout(() => msg?.classList.remove('show'), 2500);
    }
  };

  window._axiumFreshStart = () => {
    const fresh = createFreshSave();
    AxiumSave.set(fresh);
    onLoaded?.(fresh);
  };
}

// ─────────────────────────────────────────────────────────────
// EXPORT
// ─────────────────────────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    AXIUM_VERSION, STARTING_DECK_IDS, CHAPTER_REGISTRY, PERSONAS,
    AxiumSave, AxiumSeed, AxiumShop, AxiumAI, AxiumHistory,
    buildChapterConfig, renderSeedWidget, renderSeedLoader,
    createFreshSave, shuffle,
  };
}

const AxiumGuard = {
  /** * Prevents re-running the shop for a chapter already won.
   * Checks if the shopHistory length matches the expected count for the current progress.
   */
  canAccessShop(chapterId) {
    const save = AxiumSave.getOrCreate();
    const picksThisChapter = save.shopHistory.filter(h => h.chapter === chapterId).length;

    // If they already made 4 picks for this chapter, lock the shop.
    if (picksThisChapter >= 4) {
      console.warn("Access Denied: Shop already completed for this chapter.");
      return false;
    }
    return true;
  },

  /** * Forces a save to local storage before the seed is shown
   * to ensure page reloads don't reset the "won" state.
   */
  lockProgress(chapterId) {
    const save = AxiumSave.getOrCreate();
    if (!save.chaptersWon.includes(chapterId)) {
      save.chaptersWon.push(chapterId);
      AxiumSave.set(save);
    }
  }
};

function purgeSaveData() {
  if (confirm("This will permanently dissolve your current constellation and progress. Proceed?")) {
    // 1. Wipe Local Storage
    localStorage.removeItem('axium_save');
    localStorage.removeItem('axium_history');

    // 2. Reset APP state to Chapter 1 defaults
    APP.save = null;
    APP.shopStep = 0;
    APP.shopPicks = [];
    APP.playerDeck = [];

    // 3. Clear any active shop animations
    APP.shopCanvasAnims.forEach((id) => cancelAnimationFrame(id));
    APP.shopCanvasAnims.clear();

    toast('System Purged', 'Returning to the void.');

    // 4. Force a slight UI delay for the toast, then refresh if needed
    // Or simply stay on intro screen with fresh state
    console.log("Save data purged. Ready for a fresh start.");
  }
}
