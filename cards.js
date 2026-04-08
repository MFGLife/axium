/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — CARD MATRIX v4.0
 * "Battle for Attention" — Node Resonance Engine
 *
 * MECHANIC ARCHITECTURE v4.0
 * ───────────────────────────
 * BATTLE: Real-time simulation. All cards fire on repeating timers
 * derived from their constellation node count and tier.
 *
 * SPEED (node resonance):
 *   fireInterval = tierInterval(tier) / card.pts.length
 *   More nodes = faster warmup AND faster cycle = more powerful.
 *   Fool (7 nodes, T1) fires every ~1143ms.
 *   World (21 nodes, T1) fires every ~381ms.
 *   This is the primary differentiator between cards of the same tier.
 *
 * TIER BASE INTERVALS:
 *   Tier 1 → 8000ms  |  Tier 2 → 4000ms  |  Tier 3 → 2000ms
 *   Surge (t > 45s)  → ÷10 on all
 *
 * LAYER ROLES:
 *
 *   ID (40 Pip Cards) — The Pulse
 *     Fires continuously on its node-speed cycle.
 *     UPRIGHT  → +rechargeVal × pMult + pFlat  to own attn each fire.
 *     REVERSED → −drainVal  to enemy attn each fire.
 *                drainVal < rechargeVal by design. Reversed attacks raw —
 *                pMult does NOT apply. Aggression bypasses your own growth.
 *
 *   SUPEREGO (22 Major Arcana) — The Architecture
 *     Two-phase behavior:
 *     UPRIGHT — First fire:  +shieldVal burst to own attn (amplified by pMult).
 *               Subsequent:  +capacityVal × CAP_TICK_RATE added to pMax each fire.
 *                            If capacityVal ≤ 0, subsequent fires do nothing.
 *     REVERSED — First fire: −round(shieldVal × 0.6) burst to enemy attn.
 *                Subsequent: if capacityVal > 0, −capacityVal × CAP_TICK_RATE
 *                            subtracted from eMax each fire (collapsing their ceiling).
 *     Enemy Superego reversed targets pMax (not eMax) on subsequent ticks.
 *
 *   EGO (16 Court Cards) — The Multiplier
 *     Fires ONCE only on first tick. No subsequent fires.
 *     UPRIGHT  → pMult  *= (chunkPct || 1.0)
 *                pFlat  += (chunkFlat || 0)
 *                All future ID and Superego gains are now amplified.
 *                Kings (15 nodes, T2) fire multiplier later than Pages
 *                (9 nodes, T2) but deliver higher chunkPct — timing strategy.
 *     REVERSED → eDivisor  *= (chunkPct || 1.0)
 *                eDmgFlat  += (chunkFlat || 0)
 *                All enemy gains are divided and reduced going forward.
 *                Enemy Ego reversed divides player gains instead.
 *
 * SYNERGIES (at t=0, before any card fires):
 *   Active when all named cards are in playerPlayed upright.
 *   Apply attnBoost directly to pAttn.
 *   Apply synergyMult into pMult (stacks multiplicatively).
 *   Reward upright-heavy builds. Reversed cards do not trigger synergies.
 *
 * POLARITY TRADEOFF:
 *   Upright:  heal yourself + trigger synergies + benefit from pMult.
 *   Reversed: attack enemy directly, but lose all upright benefits.
 *             No pMult on reversed attacks — raw damage only.
 *
 * WIN/LOSE:
 *   Early KO: enemy attn hits 0 (player wins) or player attn ≤ loseAttn.
 *   Time expiry (60s): player wins if pAttn/pMax − eAttn/eMax ≥ winPctMargin.
 *
 * ENEMY DECK:
 *   All enemy cards play reversed. Stages 1–10 scale in tier mix, node
 *   density, and hand size. Enemy Ego reversed divides player gains.
 *   Enemy Superego reversed burst-damages player then slowly drains pMax.
 *
 * ═══════════════════════════════════════════════════════════════
 */

// ───────────────────────────────────────────────────────────────
// ATTENTION SPECTRUM  (0 = Fragmented → 100 = Enlightened)
// ───────────────────────────────────────────────────────────────
const ATTN_STATES = [
  {
    id:'fragmented', label:'Fragmented', pos:0.00, col:'#6B21A8',
    desc:'Attention shattered. Passive recharge halved. Capacity cards disabled.',
    debuff:{ passiveMod:0.5, capacityDisabled:true },
  },
  {
    id:'fear', label:'Fear', pos:0.10, col:'#9333EA',
    desc:'Attention collapses inward. Chunk multipliers reduced by 0.5x.',
    debuff:{ chunkMod:-0.5 },
  },
  {
    id:'anger', label:'Anger', pos:0.20, col:'#DC2626',
    desc:'Only 5 cards activate in battle instead of 10.',
    debuff:{ maxActiveCards:5 },
  },
  {
    id:'sadness', label:'Sadness', pos:0.30, col:'#1D4ED8',
    desc:'Recharge stacks accumulate at half rate.',
    debuff:{ rechargeMod:0.5 },
  },
  {
    id:'disinterest', label:'Disinterest', pos:0.42, col:'#374151',
    desc:'Synergy bonuses do not fire. Cards act in isolation.',
    debuff:{ synergiesBlocked:true },
  },
  {
    id:'witness', label:'Witness', pos:0.54, col:'#7EB8E8',
    desc:'Neutral baseline. All systems nominal.',
    debuff:null,
  },
  {
    id:'presence', label:'Presence', pos:0.65, col:'#D4AF37',
    desc:'Full presence. Each recharge stack gains +1 bonus.',
    buff:{ rechargeBonus:1 },
  },
  {
    id:'clarity', label:'Clarity', pos:0.76, col:'#86EFAC',
    desc:'Sees through distortion. Chunk multipliers gain +0.5x.',
    buff:{ chunkBonus:0.5 },
  },
  {
    id:'ground', label:'Grounded', pos:0.87, col:'#F0D080',
    desc:'Firmly held. Shield values doubled.',
    buff:{ shieldMult:2 },
  },
  {
    id:'enlightened', label:'Enlightened', pos:1.00, col:'#FFFFFF',
    desc:'Complete attention. Chapter win condition triggered.',
    buff:{ winTrigger:true },
  },
];

// ───────────────────────────────────────────────────────────────
// CARD GLOSSARY
// ───────────────────────────────────────────────────────────────
const CARD_GLOSSARY = {
  recharge:  'Passive attention gain that stacks each turn card is held.',
  drain:     'Passive attention loss. Reversed ID cards inject this.',
  shield:    'Temporary max-attention boost. Absorbs trauma above threshold.',
  capacity:  'Permanent max-attention increase while card is in your deck.',
  chunk:     'Multiplier on attention gain. Flat (+N) or percent (xN%).',
  divide:    'Divisor on attention. Occultism / conspiracy / misinformation.',
  study:     'Research bonus: amplifies the next card played.',
  synergy:   'Fires when named cards are all in the same hand.',
  axium:     'Perfect score. 10 axium cards = final boss unlock.',
  tier:      'Card power level 1-3. Tier 3 requires shop unlock.',
};

// ═══════════════════════════════════════════════════════════════
// SUPEREGO — 22 Major Arcana
// Layer mechanic: SHIELD (upright) / CAPACITY (in deck)
// shieldVal   = flat attention added when played in battle
// capacityVal = max-pool modifier while card sits in deck (+ expands, - shrinks)
// rechargeVal = small passive while held in hand
// ═══════════════════════════════════════════════════════════════
const PLAYER_CARDS = [

  // ── 0 · The Fool ─────────────────────────────────────────────
  {
    id:'fool', name:'The Fool',
    layer:'Superego', number:0,
    keywords:'leap · pure potential · trust',
    type:'decompression', axiumScore:7, tier:1, chapter:1,
    shieldVal:18,
    shieldDesc:'Leap clears the board. +18 attention. Discard remaining hand and redraw 4 cards at 0 cost this battle.',
    shieldEffect:{ attnBoost:18, redraw:4, redrawCost:0 },
    capacityVal:-8,
    capacityDesc:'The leap refused. Max pool shrinks by 8 while corrupted in deck.',
    rechargeVal:2, rechargeDesc:'Held potential: +2 passive while in hand.',
    reversedShift:-10,
    reversedDesc:'Paralysis injected. Max active cards reduced to 5 next battle.',
    color:'#86EFAC',
    pts:[[0.5,0.05],[0.871,0.29],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5]],
    tags:['redraw','trust','chaos'], synergies:['world','wheel_of_fortune'],
  },

  // ── I · The Magician ─────────────────────────────────────────
  {
    id:'magician', name:'The Magician',
    layer:'Superego', number:1,
    keywords:'will · manifestation · alignment',
    type:'both', axiumScore:8, tier:1, chapter:1,
    shieldVal:14,
    shieldDesc:'All tools present. +14 attention. Replay one card from discard at no cost.',
    shieldEffect:{ attnBoost:14, recycleDiscard:1 },
    capacityVal:6, capacityDesc:'+6 max pool while in deck — aligned will expands the vessel.',
    rechargeVal:3, rechargeDesc:'+3 passive — the will accumulates.',
    reversedShift:-9,
    reversedDesc:'Will fragmented. All chunk multipliers halved this battle.',
    color:'#D4AF37',
    pts:[[0.5,0.5],[0.618,0.5],[0.82,0.5],[0.5,0.618],[0.5,0.83],[0.382,0.5],[0.18,0.5],[0.5,0.382],[0.502,0.17]],
    tags:['recycle','will','alignment'], synergies:['high_priestess','strength'],
  },

  // ── II · The High Priestess ──────────────────────────────────
  {
    id:'high_priestess', name:'The High Priestess',
    layer:'Superego', number:2,
    keywords:'mystery · veil · knowing',
    type:'decompression', axiumScore:7, tier:1, chapter:1,
    shieldVal:12,
    shieldDesc:'Sit at the threshold. +12 attention. Reveal all trauma cards for this battle.',
    shieldEffect:{ attnBoost:12, revealTraumaHand:true },
    capacityVal:5, capacityDesc:'+5 max pool — the veil held open.',
    rechargeVal:2, rechargeDesc:'+2 passive — knowing accumulates quietly.',
    reversedShift:-11,
    reversedDesc:'Answer forced before silence speaks. Next draw randomised.',
    color:'#7EB8E8',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383],[0.274,0.274],[0.342,0.311],[0.5,0.27]],
    tags:['reveal','mystery'], synergies:['magician','moon'],
  },

  // ── III · The Empress ────────────────────────────────────────
  {
    id:'empress', name:'The Empress',
    layer:'Superego', number:3,
    keywords:'abundance · creation · nature',
    type:'decompression', axiumScore:7, tier:1, chapter:1,
    shieldVal:16,
    shieldDesc:'The earth does not hurry. +16 attention. Remove 1 drain stack from opponent.',
    shieldEffect:{ attnBoost:16, removeDrainFromOpponent:1 },
    capacityVal:8, capacityDesc:'+8 max pool — abundance expands the vessel.',
    rechargeVal:3, rechargeDesc:'+3 passive — growth given room.',
    reversedShift:-8,
    reversedDesc:'Growth forced. All recharge stacks halved this battle.',
    color:'#DCC0EC',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311]],
    tags:['cleanse','abundance'], synergies:['emperor','star'],
  },

  // ── IV · The Emperor ─────────────────────────────────────────
  {
    id:'emperor', name:'The Emperor',
    layer:'Superego', number:4,
    keywords:'structure · authority · law',
    type:'compression', axiumScore:8, tier:1, chapter:1,
    shieldVal:20,
    shieldDesc:'Order is a form of love. +20 attention. Lock current attention as floor for this battle.',
    shieldEffect:{ attnBoost:20, setFloor:true },
    capacityVal:10, capacityDesc:'+10 max pool — the foundation determines the ceiling.',
    rechargeVal:2, rechargeDesc:'+2 passive — structure accumulates steadily.',
    reversedShift:-13,
    reversedDesc:'Tyranny. All chunk multipliers disabled. Hand limit -2.',
    color:'#F0D080',
    pts:[[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.405,0.698],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tags:['floor','structure','authority'], synergies:['empress','justice'],
  },

  // ── V · The Hierophant ───────────────────────────────────────
  {
    id:'hierophant', name:'The Hierophant',
    layer:'Superego', number:5,
    keywords:'tradition · initiation · lineage',
    type:'decompression', axiumScore:6, tier:1, chapter:1,
    shieldVal:11,
    shieldDesc:'Teaching older than you. +11 attention. All synergies fire without needing both cards present.',
    shieldEffect:{ attnBoost:11, forceSynergies:true },
    capacityVal:5, capacityDesc:'+5 max pool — lineage expands what you can hold.',
    rechargeVal:2, rechargeDesc:'+2 passive — tradition accumulates.',
    reversedShift:-10,
    reversedDesc:"Tradition weaponised. Cards of type 'both' disabled this battle.",
    color:'#A0D0FF',
    pts:[[0.5,0.05],[0.825,0.175],[0.949,0.602],[0.7,0.914],[0.3,0.914],[0.051,0.602],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5]],
    tags:['synergy-boost','tradition'], synergies:['emperor','judgement'],
  },

  // ── VI · The Lovers ──────────────────────────────────────────
  {
    id:'lovers', name:'The Lovers',
    layer:'Superego', number:6,
    keywords:'union · sacred choice · alignment',
    type:'both', axiumScore:8, tier:1, chapter:1,
    shieldVal:17,
    shieldDesc:'The highest law. +17 attention. Select one card in hand to play free this battle.',
    shieldEffect:{ attnBoost:17, freePlay:1 },
    capacityVal:7, capacityDesc:'+7 max pool — sacred union expands the vessel.',
    rechargeVal:3, rechargeDesc:'+3 passive — chosen alignment builds energy.',
    reversedShift:-12,
    reversedDesc:'Chosen from obligation. Fabricates a false card that drains 8 on play.',
    color:'#F4A0C8',
    pts:[[0.5,0.05],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.817,0.725],[0.596,0.698],[0.5,0.95],[0.405,0.698],[0.16,0.747],[0.28,0.5],[0.136,0.29]],
    tags:['free-play','choice'], synergies:['chariot','star'],
  },

  // ── VII · The Chariot ────────────────────────────────────────
  {
    id:'chariot', name:'The Chariot',
    layer:'Superego', number:7,
    keywords:'will · victory · discipline',
    type:'compression', axiumScore:8, tier:1, chapter:1,
    shieldVal:22,
    shieldDesc:'Opposing forces harnessed. +22 attention. Next card played costs 0 recharge this battle.',
    shieldEffect:{ attnBoost:22, nextCardFree:true },
    capacityVal:9, capacityDesc:'+9 max pool — discipline held steady.',
    rechargeVal:4, rechargeDesc:'+4 passive — momentum compounds.',
    reversedShift:-14,
    reversedDesc:'Chaos wins. Decompression cards cost +3 recharge stacks.',
    color:'#C8A860',
    pts:[[0.7,0.086],[0.861,0.383],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.139,0.383],[0.3,0.086],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['momentum','will'], synergies:['emperor','strength'],
  },

  // ── VIII · Strength ──────────────────────────────────────────
  {
    id:'strength', name:'Strength',
    layer:'Superego', number:8,
    keywords:'courage · gentleness · mastery',
    type:'decompression', axiumScore:7, tier:1, chapter:1,
    shieldVal:14,
    shieldDesc:'The lion held by love. +14 attention. Reduce first trauma card to minimum intensity.',
    shieldEffect:{ attnBoost:14, reduceFirstTrauma:true },
    capacityVal:6, capacityDesc:'+6 max pool — mastery expands capacity.',
    rechargeVal:3, rechargeDesc:'+3 passive — gentleness accumulates more than force.',
    reversedShift:-11,
    reversedDesc:'Force where gentleness would work. Trauma gains +4 shift this battle.',
    color:'#FF9060',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.691,0.61],[0.72,0.5],[0.596,0.698],[0.5,0.72],[0.387,0.613],[0.764,0.618]],
    tags:['reduce','gentleness'], synergies:['magician','chariot'],
  },

  // ── IX · The Hermit ──────────────────────────────────────────
  {
    id:'hermit', name:'The Hermit',
    layer:'Superego', number:9,
    keywords:'solitude · lantern · inner light',
    type:'decompression', axiumScore:7, tier:1, chapter:1,
    shieldVal:13,
    shieldDesc:"The answer lives at your own corridor's end. Only fires when attention is below 50% — the lantern is for darkness. +13 attention. Skip trauma's first card this battle.",
    fireCondition:{ maxPct: 0.50 },
    shieldEffect:{ attnBoost:13, skipFirstTrauma:true },
    capacityVal:5, capacityDesc:'+5 max pool — solitude deepens the well.',
    rechargeVal:2, rechargeDesc:'+2 passive — the lantern burns quietly.',
    reversedShift:-9,
    reversedDesc:'Isolation as avoidance. Draw 1 fewer card at battle start.',
    color:'#C8D080',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61]],
    tags:['reveal','skip','solitude'], synergies:['high_priestess','moon'],
  },

  // ── X · Wheel of Fortune ─────────────────────────────────────
  {
    id:'wheel_of_fortune', name:'Wheel of Fortune',
    layer:'Superego', number:10,
    keywords:'fate · cycles · turning',
    type:'both', axiumScore:8, tier:1, chapter:1,
    shieldVal:16,
    shieldDesc:'The wheel turns. +16 attention. Shuffle discard into deck and draw 2 fresh cards mid-battle.',
    shieldEffect:{ attnBoost:16, reshuffleDraw:2 },
    capacityVal:7, capacityDesc:'+7 max pool — the cycle is also an expansion.',
    rechargeVal:3, rechargeDesc:'+3 passive — every rise seeds the next.',
    reversedShift:-12,
    reversedDesc:"Resistance to the turn. All trauma discards return recharged.",
    color:'#A060D0',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.342,0.311]],
    tags:['cycle','fate'], synergies:['fool','world'],
  },

  // ── XI · Justice ─────────────────────────────────────────────
  {
    id:'justice', name:'Justice',
    layer:'Superego', number:11,
    keywords:'truth · law · consequence',
    type:'compression', axiumScore:8, tier:1, chapter:1,
    shieldVal:18,
    shieldDesc:"The higher law is exacting. +18 attention. Mirror trauma's last drain value back as recharge.",
    shieldEffect:{ attnBoost:18, mirrorDrain:true },
    capacityVal:8, capacityDesc:'+8 max pool — law held in the body.',
    rechargeVal:3, rechargeDesc:'+3 passive — consequence accumulates.',
    reversedShift:-13,
    reversedDesc:'Verdict avoided. All chunk multipliers nullified this battle.',
    color:'#70C0A0',
    pts:[[0.5,0.05],[0.825,0.175],[0.949,0.602],[0.7,0.914],[0.3,0.914],[0.051,0.602],[0.136,0.29],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['mirror','truth','law'], synergies:['emperor','tower'],
  },

  // ── XII · The Hanged Man ─────────────────────────────────────
  {
    id:'hanged_man', name:'The Hanged Man',
    layer:'Superego', number:12,
    keywords:'surrender · suspension · vision',
    type:'decompression', axiumScore:9, tier:1, chapter:2,
    shieldVal:24,
    shieldDesc:'The view from upside down. +24 attention. Both sides lose their top card — what remains is revealed.',
    shieldEffect:{ attnBoost:24, mutualTopDiscard:true },
    capacityVal:10, capacityDesc:'+10 max pool — suspension deepens the vessel.',
    rechargeVal:4, rechargeDesc:'+4 passive — processed below strategy surfaces when ready.',
    reversedShift:-10,
    reversedDesc:'The surrender fought. All decompression costs double this battle.',
    color:'#60B0D8',
    pts:[[0.5,0.95],[0.136,0.29],[0.871,0.29],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['surrender','vision'], synergies:['hermit','moon'],
  },

  // ── XIII · Death ─────────────────────────────────────────────
  {
    id:'death', name:'Death',
    layer:'Superego', number:13,
    keywords:'transformation · ending · crossing',
    type:'both', axiumScore:9, tier:1, chapter:2,
    shieldVal:20,
    fireCondition:{ minPct: 0.70 },
    shieldDesc:'Nothing meant to continue ends here. Only transforms when you are strong enough to hold it (attn ≥ 70%). +20 attention. Remove all drain stacks permanently this battle.',
    shieldEffect:{ attnBoost:20, clearAllDrain:true },
    capacityVal:12, capacityDesc:'+12 max pool — old form dissolves, vessel grows.',
    rechargeVal:5, rechargeDesc:'+5 passive — transformation releases stored energy.',
    reversedShift:-15,
    reversedDesc:'Clinging to what Death claimed. Drain removal locked for 3 turns. Recharge capped at 3.',
    color:'#808090',
    pts:[[0.5,0.05],[0.584,0.297],[0.825,0.175],[0.72,0.5],[0.949,0.602],[0.613,0.613],[0.7,0.914],[0.5,0.72],[0.3,0.914],[0.387,0.613],[0.051,0.602],[0.28,0.5],[0.136,0.29],[0.416,0.297]],
    tags:['cleanse-drain','transformation'], synergies:['tower','judgement'],
  },

  // ── XIV · Temperance ─────────────────────────────────────────
  {
    id:'temperance', name:'Temperance',
    layer:'Superego', number:14,
    keywords:'alchemy · patience · blending',
    type:'decompression', axiumScore:8, tier:1, chapter:1,
    shieldVal:15,
    shieldDesc:'Pour between cups until two become one. +15 attention. Convert half your drain stacks into recharge.',
    shieldEffect:{ attnBoost:15, convertDrainToRecharge:0.5 },
    capacityVal:7, capacityDesc:'+7 max pool — the alchemical blend expands.',
    rechargeVal:3, rechargeDesc:'+3 passive — opposites held become fuel.',
    reversedShift:-10,
    reversedDesc:'Impatience. All held recharge stacks reset to 0.',
    color:'#80C8C0',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383],[0.274,0.274],[0.342,0.311],[0.5,0.27],[0.613,0.387]],
    tags:['convert','alchemy','patience'], synergies:['star','lovers'],
  },

  // ── XV · The Devil ───────────────────────────────────────────
  {
    id:'devil', name:'The Devil',
    layer:'Superego', number:15,
    keywords:'shadow · bondage · liberation',
    type:'compression', axiumScore:9, tier:1, chapter:2,
    shieldVal:25,
    shieldDesc:'The chains are yours to remove. +25 attention. Destroy all fabrication cards in hand.',
    shieldEffect:{ attnBoost:25, destroyFabrications:true },
    capacityVal:11, capacityDesc:'+11 max pool — named shadow expands conscious space.',
    rechargeVal:4, rechargeDesc:'+4 passive — liberation releases bound energy.',
    reversedShift:-14,
    reversedDesc:'Chains seen but not moved. Highest-shift card disabled this battle.',
    color:'#C06060',
    pts:[[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tags:['destroy-fabrications','shadow'], synergies:['tower','death'],
  },

  // ── XVI · The Tower ──────────────────────────────────────────
  {
    id:'tower', name:'The Tower',
    layer:'Superego', number:16,
    keywords:'revelation · collapse · liberation',
    type:'compression', axiumScore:8, tier:1, chapter:2,
    shieldVal:28,
    shieldDesc:'The lightning strikes what was never true. +28 attention. Clear all drain stacks from both sides.',
    shieldEffect:{ attnBoost:28, clearAllDrainBothSides:true },
    capacityVal:10, capacityDesc:'+10 max pool — what remains after clearing is real.',
    rechargeVal:3, rechargeDesc:'+3 passive — the cleared structure leaves more room.',
    reversedShift:-18,
    reversedDesc:'Refusing the lightning. Cannot play compression cards this battle.',
    color:'#E06040',
    pts:[[0.5,0.05],[0.613,0.387],[0.95,0.5],[0.613,0.613],[0.5,0.95],[0.387,0.613],[0.05,0.5],[0.387,0.387],[0.5,0.5],[0.382,0.5],[0.5,0.382]],
    tags:['max-clear','revelation'], synergies:['justice','death'],
  },

  // ── XVII · The Star ──────────────────────────────────────────
  {
    id:'star', name:'The Star',
    layer:'Superego', number:17,
    keywords:'hope · renewal · guidance',
    type:'decompression', axiumScore:9, tier:1, chapter:2,
    shieldVal:20,
    shieldDesc:'The wound becomes the opening. +20 attention. Restore 3 recharge stacks and hold a 12-point floor.',
    shieldEffect:{ attnBoost:20, restoreRechargeStacks:3, battleFloor:12 },
    capacityVal:12, capacityDesc:'+12 max pool — hope held expands what is possible.',
    rechargeVal:5, rechargeDesc:'+5 passive — the star burns steadily.',
    reversedShift:-11,
    reversedDesc:'Hope curdled to despair. All recharge stacks gain 0 bonus this battle.',
    color:'#A0C0FF',
    pts:[[0.5,0.05],[0.825,0.175],[0.949,0.398],[0.817,0.725],[0.609,0.906],[0.391,0.906],[0.16,0.747],[0.051,0.398],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.405,0.698],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tags:['restore','floor','hope'], synergies:['empress','temperance'],
  },

  // ── XVIII · The Moon ─────────────────────────────────────────
  {
    id:'moon', name:'The Moon',
    layer:'Superego', number:18,
    keywords:'illusion · mystery · cycles',
    type:'decompression', axiumScore:8, tier:1, chapter:2,
    shieldVal:16,
    shieldDesc:'Not everything in the dark is threat. +16 attention. Reveal all trauma and reduce each by 2 intensity.',
    shieldEffect:{ attnBoost:16, revealAll:true, reduceTraumaIntensity:2 },
    capacityVal:8, capacityDesc:'+8 max pool — the tidal cycle deepens capacity.',
    rechargeVal:3, rechargeDesc:'+3 passive — the moon lights enough for the next step.',
    reversedShift:-13,
    reversedDesc:'Anxiety colonises reason. All cards cost 3 attention to play.',
    color:'#8080C8',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311],[0.72,0.5]],
    tags:['reveal','illusion'], synergies:['high_priestess','hermit'],
  },

  // ── XIX · The Sun ────────────────────────────────────────────
  {
    id:'sun', name:'The Sun',
    layer:'Superego', number:19,
    keywords:'joy · consciousness · radiance',
    type:'decompression', axiumScore:10, tier:1, chapter:3,
    shieldVal:30,
    shieldDesc:'This is not earned — it simply is. +30 attention. All synergies fire twice. Recharge doubled.',
    shieldEffect:{ attnBoost:30, doubleSynergies:true, doubleRecharge:true },
    capacityVal:15, capacityDesc:'+15 max pool — radiance expands without limit.',
    rechargeVal:6, rechargeDesc:'+6 passive — the sun does not meter its output.',
    reversedShift:-10,
    reversedDesc:'Joy refused. All recharge stacks capped at 2.',
    color:'#FFD060',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.949,0.602],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.051,0.602],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.755,0.382],[0.691,0.61],[0.596,0.698],[0.5,0.72],[0.405,0.698],[0.309,0.61],[0.236,0.382],[0.342,0.311]],
    tags:['synergy-double','radiance'], synergies:['star','world'],
  },

  // ── XX · Judgement ───────────────────────────────────────────
  {
    id:'judgement', name:'Judgement',
    layer:'Superego', number:20,
    keywords:'calling · awakening · rebirth',
    type:'compression', axiumScore:10, tier:1, chapter:3,
    shieldVal:26,
    shieldDesc:'The trumpet has sounded. +26 attention. Remove all debuffs. Convert all drain to recharge.',
    shieldEffect:{ attnBoost:26, clearAllDebuffs:true, convertAllDrainToRecharge:true },
    capacityVal:14, capacityDesc:'+14 max pool — the awakening expands the possible.',
    rechargeVal:5, rechargeDesc:'+5 passive — the calling releases energy.',
    reversedShift:-16,
    reversedDesc:'The trumpet heard, the choice not to rise. Next 2 shield cards deal 0 boost.',
    color:'#E0C080',
    pts:[[0.5,0.5],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.416,0.297],[0.584,0.297],[0.95,0.5],[0.7,0.914],[0.3,0.914],[0.05,0.5],[0.3,0.086],[0.7,0.086],[0.5,0.382]],
    tags:['debuff-clear','rebirth'], synergies:['death','hierophant'],
  },

  // ── XXI · The World ──────────────────────────────────────────
  {
    id:'world', name:'The World',
    layer:'Superego', number:21,
    keywords:'completion · wholeness · arrival',
    type:'decompression', axiumScore:10, tier:1, chapter:3,
    shieldVal:35,
    fireCondition:{ minPct: 0.60 },
    shieldDesc:'The journey arrives where it began. Only completes when you are present (attn ≥ 60%). +35 attention. Win triggered if player attn >= 90.',
    shieldEffect:{ attnBoost:35, winCheck:true, winThreshold:90 },
    capacityVal:20, capacityDesc:'+20 max pool — the largest vessel in the deck.',
    rechargeVal:7, rechargeDesc:'+7 passive — wholeness is generative, not static.',
    reversedShift:-15,
    reversedDesc:'Completion withheld. Attention reads 15 below actual for 3 turns.',
    color:'#FFFFFF',
    pts:[[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5],[0.665,0.335],[0.691,0.61],[0.309,0.61],[0.342,0.311],[0.596,0.698],[0.405,0.698],[0.387,0.387],[0.613,0.387]],
    tags:['win-check','wholeness'], synergies:['fool','wheel_of_fortune'],
  },
];

// ═══════════════════════════════════════════════════════════════
// EGO — 16 Court Cards
// Layer mechanic: CHUNK (upright) / DIVIDE (reversed)
// chunkFlat : flat bonus added to all attention gains this battle
// chunkPct  : % multiplier on all attention gains (e.g. 1.5 = x1.5)
// studyMult : amplifies the NEXT card played by this multiplier
// divideMode: 'divide' = hard divisor | 'occult' = randomised 0.5-1.5x
// ═══════════════════════════════════════════════════════════════
const EGO_CARDS = [

  // ── CUPS ─────────────────────────────────────────────────────
  {
    id:'page_cups', name:'Page of Cups', layer:'Ego', suit:'cups',
    keywords:'wonder · openness · feeling',
    type:'decompression', axiumScore:5, tier:2, shopChapter:1,
    chunkFlat:4,
    chunkDesc:"Beginner's mind. +4 flat to all gains. Draw 2 extra cards this battle.",
    chunkEffect:{ flatBonus:4, extraDraw:2 },
    divideMode:'occult',
    divideDesc:'Over-receptivity. Next draw randomised across full deck.',
    color:'#7EB8E8',
    pts:[[0.5,0.27],[0.764,0.618],[0.236,0.618],[0.5,0.382],[0.613,0.387],[0.613,0.613],[0.5,0.618],[0.387,0.613],[0.387,0.387]],
    tags:['draw','openness'],
  },
  {
    id:'knight_cups', name:'Knight of Cups', layer:'Ego', suit:'cups',
    keywords:'romance · pursuit · idealism',
    type:'both', axiumScore:6, tier:2, shopChapter:1,
    chunkFlat:0, chunkPct:1.3,
    chunkDesc:'Ride toward beauty. x1.3 all gains when played alongside a decompression card.',
    chunkEffect:{ pctMultiplier:1.3, condition:'hasDecompression' },
    divideMode:'divide', divideVal:0.7,
    divideDesc:'Chasing an ideal. x0.7 on next gain.',
    color:'#60A0D8',
    pts:[[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.5,0.5],[0.765,0.563],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.235,0.563],[0.5,0.383],[0.5,0.95]],
    tags:['amplify','idealism'],
  },
  {
    id:'queen_cups', name:'Queen of Cups', layer:'Ego', suit:'cups',
    keywords:'empathy · intuition · depth',
    type:'decompression', axiumScore:7, tier:2, shopChapter:2,
    chunkFlat:6,
    chunkDesc:'The self holds space. +6 flat. Cancel all status debuffs on your attention.',
    chunkEffect:{ flatBonus:6, clearDebuffs:true },
    divideMode:'occult',
    divideDesc:'Empathy become self-erasure. Decompression cards cost +2 recharge stacks.',
    color:'#80A0D8',
    pts:[[0.502,0.17],[0.755,0.382],[0.817,0.725],[0.5,0.83],[0.236,0.618],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['debuff-clear','empathy'],
  },
  {
    id:'king_cups', name:'King of Cups', layer:'Ego', suit:'cups',
    keywords:'mastery · balance · diplomacy',
    type:'both', axiumScore:8, tier:2, shopChapter:3,
    chunkFlat:0, chunkPct:1.5,
    studyMult:2.0,
    chunkDesc:'Emotional authority. x1.5 all gains. All synergy exhaust deals x2 damage.',
    chunkEffect:{ pctMultiplier:1.5, synergyDouble:true },
    divideMode:'divide',
    divideDesc:'Feelings managed to numbness. Synergies deal 0 this battle.',
    color:'#4080C8',
    pts:[[0.5,0.5],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.416,0.297],[0.584,0.297],[0.95,0.5],[0.7,0.914],[0.3,0.914],[0.05,0.5],[0.3,0.086],[0.7,0.086],[0.5,0.382],[0.618,0.5]],
    tags:['synergy-double','authority'],
  },

  // ── SWORDS ─────────────────────────────────────────────────
  {
    id:'page_swords', name:'Page of Swords', layer:'Ego', suit:'swords',
    keywords:'curiosity · ideas · restless',
    type:'decompression', axiumScore:5, tier:2, shopChapter:1,
    chunkFlat:3,
    chunkDesc:'Questions in good faith protect. +3 flat. Reveal next 2 trauma, reduce each -1 intensity.',
    chunkEffect:{ flatBonus:3, revealReduce:{ count:2, reduction:1 } },
    divideMode:'occult',
    divideDesc:'Curiosity become cynicism. All reveal effects disabled.',
    color:'#C0C8A0',
    pts:[[0.502,0.17],[0.82,0.5],[0.5,0.83],[0.18,0.5],[0.5,0.382],[0.618,0.5],[0.613,0.613],[0.387,0.613],[0.382,0.5]],
    tags:['reveal','curiosity'],
  },
  {
    id:'knight_swords', name:'Knight of Swords', layer:'Ego', suit:'swords',
    keywords:'speed · ambition · reckless',
    type:'compression', axiumScore:6, tier:2, shopChapter:2,
    chunkFlat:8,
    chunkDesc:'Think once before the blade lands. +8 flat. Play two cards this battle — second costs x2 recharge.',
    chunkEffect:{ flatBonus:8, doublePlay:true, secondCardCost:2 },
    divideMode:'divide',
    divideDesc:'Speed outpaced judgment. Next 2 compression cards gain 0 flat.',
    color:'#C0C0B0',
    pts:[[0.502,0.17],[0.613,0.387],[0.82,0.5],[0.613,0.613],[0.5,0.83],[0.387,0.613],[0.18,0.5],[0.387,0.387],[0.5,0.5],[0.618,0.5],[0.5,0.382]],
    tags:['double-play','reckless'],
  },
  {
    id:'queen_swords', name:'Queen of Swords', layer:'Ego', suit:'swords',
    keywords:'clarity · boundary · discernment',
    type:'compression', axiumScore:7, tier:2, shopChapter:2,
    chunkFlat:0, chunkPct:1.4,
    chunkDesc:'Precision, not cruelty. x1.4 all gains. Destroy all fabrications in hand permanently.',
    chunkEffect:{ pctMultiplier:1.4, destroyFabrications:true },
    divideMode:'occult',
    divideDesc:"Clarity weaponised. Trauma's next 2 cards each gain +4 drain.",
    color:'#D0D0C8',
    pts:[[0.502,0.17],[0.726,0.274],[0.876,0.637],[0.715,0.83],[0.285,0.83],[0.11,0.548],[0.274,0.274],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['destroy-fabrications','clarity'],
  },
  {
    id:'king_swords', name:'King of Swords', layer:'Ego', suit:'swords',
    keywords:'authority · logic · judgment',
    type:'compression', axiumScore:8, tier:2, shopChapter:3,
    chunkFlat:0, chunkPct:1.6,
    studyMult:2.5,
    chunkDesc:'Logic as authority. x1.6 all gains. Study: next card played is amplified x2.5.',
    chunkEffect:{ pctMultiplier:1.6, study:2.5 },
    divideMode:'divide',
    divideDesc:'Judgment before all evidence. Synergies cannot fire this battle.',
    color:'#E0E0D8',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311],[0.72,0.5],[0.405,0.698],[0.5,0.27]],
    tags:['study','judgment'],
  },

  // ── WANDS ─────────────────────────────────────────────────────
  {
    id:'page_wands', name:'Page of Wands', layer:'Ego', suit:'wands',
    keywords:'adventure · enthusiasm · spark',
    type:'decompression', axiumScore:5, tier:2, shopChapter:1,
    chunkFlat:5,
    chunkDesc:'Follow the excitement. +5 flat. Play this card free (no recharge). Draw 1.',
    chunkEffect:{ flatBonus:5, selfFree:true, extraDraw:1 },
    divideMode:'occult',
    divideDesc:'Excitement scattered. Next draw randomised.',
    color:'#E08040',
    pts:[[0.871,0.29],[0.5,0.83],[0.136,0.29],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['free-play','spark'],
  },
  {
    id:'knight_wands', name:'Knight of Wands', layer:'Ego', suit:'wands',
    keywords:'passion · courage · momentum',
    type:'both', axiumScore:6, tier:2, shopChapter:2,
    chunkFlat:0, chunkPct:1.4,
    chunkDesc:'Channel the fire. x1.4 all gains. x2.8 if played as second card this battle.',
    chunkEffect:{ pctMultiplier:1.4, doubleIfSecond:true },
    divideMode:'divide',
    divideDesc:"Exhausted but won't admit it. Next compression card -5 flat.",
    color:'#E06820',
    pts:[[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.05],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['double-second','momentum'],
  },
  {
    id:'queen_wands', name:'Queen of Wands', layer:'Ego', suit:'wands',
    keywords:'charisma · will · presence',
    type:'both', axiumScore:7, tier:2, shopChapter:2,
    chunkFlat:7,
    chunkDesc:'Natural authority. +7 flat to ALL cards played this battle.',
    chunkEffect:{ flatBonus:7, appliedToAll:true },
    divideMode:'divide',
    divideDesc:'Presence become performance. All positive shifts reduced by 4.',
    color:'#E04820',
    pts:[[0.502,0.17],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.817,0.725],[0.596,0.698],[0.5,0.83],[0.405,0.698],[0.16,0.747],[0.28,0.5],[0.136,0.29],[0.416,0.297],[0.5,0.5]],
    tags:['all-boost','presence'],
  },
  {
    id:'king_wands', name:'King of Wands', layer:'Ego', suit:'wands',
    keywords:'vision · leadership · legacy',
    type:'compression', axiumScore:8, tier:2, shopChapter:3,
    chunkFlat:10, chunkPct:1.5,
    chunkDesc:'Vision made structure. +10 flat and x1.5. Compression cards gain +5 flat for 3 turns after.',
    chunkEffect:{ flatBonus:10, pctMultiplier:1.5, compressionBonus:{ flat:5, turns:3 } },
    divideMode:'divide',
    divideDesc:'Building for legacy, not purpose. Decompression cards cost double recharge.',
    color:'#E03000',
    pts:[[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['vision','sustained-chunk'],
  },

  // ── PENTACLES ─────────────────────────────────────────────────
  {
    id:'page_pents', name:'Page of Pentacles', layer:'Ego', suit:'pentacles',
    keywords:'study · diligence · beginning',
    type:'decompression', axiumScore:5, tier:2, shopChapter:1,
    chunkFlat:3, studyMult:1.5,
    chunkDesc:'The long view started carefully. +3 flat. Study: next card x1.5. Restore 2 discard cards.',
    chunkEffect:{ flatBonus:3, study:1.5, restoreDiscard:2 },
    divideMode:'occult',
    divideDesc:'Study stalled. Next draw returns itself at end of battle.',
    color:'#80C860',
    pts:[[0.05,0.236],[0.95,0.236],[0.95,0.764],[0.05,0.764],[0.391,0.199],[0.609,0.801],[0.5,0.5],[0.618,0.5],[0.391,0.801]],
    tags:['study','restore'],
  },
  {
    id:'knight_pents', name:'Knight of Pentacles', layer:'Ego', suit:'pentacles',
    keywords:'routine · reliability · patience',
    type:'decompression', axiumScore:6, tier:2, shopChapter:2,
    chunkFlat:0, chunkPct:1.0,
    chunkDesc:'Slow and steady. Gains +2 flat for each turn in your deck this run (max +10).',
    chunkEffect:{ scalingFlat:{ perTurn:2, max:10 } },
    divideMode:'divide',
    divideDesc:'Reliability become rigidity. Only 1 card can play this battle.',
    color:'#60A840',
    pts:[[0.726,0.274],[0.726,0.726],[0.274,0.726],[0.274,0.274],[0.5,0.27],[0.613,0.387],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['scaling','reliability'],
  },
  {
    id:'queen_pents', name:'Queen of Pentacles', layer:'Ego', suit:'pentacles',
    keywords:'nurturing · security · care',
    type:'decompression', axiumScore:7, tier:2, shopChapter:2,
    chunkFlat:6, chunkPct:1.2,
    chunkDesc:'Care and strength are not opposites. +6 flat and x1.2. Remove 1 drain + 1 corruption card.',
    chunkEffect:{ flatBonus:6, pctMultiplier:1.2, removeDrain:1, removeCorruption:1 },
    divideMode:'occult',
    divideDesc:'Nurturing become martyrdom. Lose 3 extra attn when trauma heals itself.',
    color:'#80B840',
    pts:[[0.73,0.18],[0.95,0.5],[0.715,0.83],[0.285,0.83],[0.11,0.548],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['cleanse','nurturing'],
  },
  {
    id:'king_pents', name:'King of Pentacles', layer:'Ego', suit:'pentacles',
    keywords:'mastery · wealth · discipline',
    type:'compression', axiumScore:8, tier:2, shopChapter:3,
    chunkFlat:8, chunkPct:1.6,
    studyMult:3.0,
    chunkDesc:'Mastery as discipline. +8 flat and x1.6. Study: next card x3.0. All cards gain +1 flat permanently.',
    chunkEffect:{ flatBonus:8, pctMultiplier:1.6, study:3.0, permanentFlat:1 },
    divideMode:'divide',
    divideDesc:'Mastery become cage. All gains capped at +8 for 3 turns.',
    color:'#60A820',
    pts:[[0.5,0.5],[0.5,0.382],[0.562,0.44],[0.618,0.5],[0.584,0.584],[0.5,0.618],[0.416,0.584],[0.382,0.5],[0.416,0.416],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.5,0.72],[0.28,0.5]],
    tags:['mastery','study','permanent-boost'],
  },
];

// ═══════════════════════════════════════════════════════════════
// ID — 40 Pip Cards
// Layer mechanic: RECHARGE (upright passive) / DRAIN (reversed passive)
// rechargeVal : attention gained per stack per turn while held
// drainVal    : attention lost per stack (trauma-played or corrupted)
// Stacks are additive across all cards in hand
// ═══════════════════════════════════════════════════════════════
const ID_CARDS = [

  // ── CUPS ─────────────────────────────────────────────────────
  { id:'ace_cups', name:'Ace of Cups', suit:'cups', layer:'ID',
    keywords:'love · awakening · grace', axiumScore:6, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive per turn. On play: +8 attn burst.',
    rechargeEffect:{ passive:4, onPlay:8 },
    drainVal:5, drainDesc:'-5 drain per turn (reversed). Trauma heals 10.',
    traumaHealing:10, color:'#4080C8',
    pts:[[0.502,0.17],[0.817,0.725],[0.16,0.747],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5]],
    tags:['recharge','overflow'] },

  { id:'two_cups', name:'Two of Cups', suit:'cups', layer:'ID',
    keywords:'connection · recognition · bond', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: synergy bonus +5.',
    rechargeEffect:{ passive:3, synergyBonus:5 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 8.',
    traumaHealing:8, color:'#5090C8',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383]],
    tags:['recharge','connection'] },

  { id:'three_cups', name:'Three of Cups', suit:'cups', layer:'ID',
    keywords:'celebration · reunion · warmth', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: +2 to all other recharge cards held.',
    rechargeEffect:{ passive:3, boostOtherRecharge:2 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 12.',
    traumaHealing:12, color:'#60A0C8',
    pts:[[0.755,0.382],[0.5,0.83],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['recharge','warmth'] },

  { id:'four_cups', name:'Four of Cups', suit:'cups', layer:'ID',
    keywords:'apathy · discontent · withdrawal', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:"+2 passive. On play: skip trauma's lowest card.",
    rechargeEffect:{ passive:2, skipTraumaLowest:true },
    drainVal:6, drainDesc:'-6 drain (reversed). Synergies blocked next turn.',
    traumaHealing:6, color:'#405080',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.5,0.382]],
    tags:['recharge','apathy'] },

  { id:'five_cups', name:'Five of Cups', suit:'cups', layer:'ID',
    keywords:'grief · loss · what remains', axiumScore:5, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: convert grief into +6 burst attn.',
    rechargeEffect:{ passive:2, griefConvert:6 },
    drainVal:7, drainDesc:'-7 drain (reversed). All player shifts -4 this battle.',
    traumaHealing:0, color:'#305070',
    pts:[[0.502,0.17],[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.27],[0.72,0.5],[0.613,0.613],[0.387,0.613],[0.28,0.5]],
    tags:['recharge','grief'] },

  { id:'six_cups', name:'Six of Cups', suit:'cups', layer:'ID',
    keywords:'nostalgia · memory · innocence', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: restore first card played this battle to hand.',
    rechargeEffect:{ passive:3, restoreFirst:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 7. Newest card disabled.',
    traumaHealing:7, color:'#507090',
    pts:[[0.502,0.17],[0.755,0.382],[0.817,0.725],[0.5,0.83],[0.236,0.618],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['recharge','nostalgia'] },

  { id:'seven_cups', name:'Seven of Cups', suit:'cups', layer:'ID',
    keywords:'desire · illusion · hunger', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: peak recharge card in hand gains +4.',
    rechargeEffect:{ passive:2, boostPeakRecharge:4 },
    drainVal:6, drainDesc:'-6 drain (reversed). Fabricates false card in hand.',
    traumaHealing:5, color:'#604080',
    pts:[[0.502,0.17],[0.613,0.387],[0.861,0.383],[0.618,0.5],[0.715,0.83],[0.5,0.618],[0.285,0.83],[0.382,0.5],[0.139,0.383],[0.387,0.387]],
    tags:['recharge','illusion'] },

  { id:'eight_cups', name:'Eight of Cups', suit:'cups', layer:'ID',
    keywords:'departure · meaning · move on', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: remove lowest drain stack permanently.',
    rechargeEffect:{ passive:3, removeLowestDrain:true },
    drainVal:5, drainDesc:'-5 drain (reversed). Highest card temporarily discarded.',
    traumaHealing:5, color:'#304060',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.691,0.61],[0.72,0.5],[0.596,0.698],[0.5,0.72],[0.387,0.613],[0.382,0.5]],
    tags:['recharge','departure'] },

  { id:'nine_cups', name:'Nine of Cups', suit:'cups', layer:'ID',
    keywords:'wish · satisfaction · pleasure', axiumScore:6, tier:1,
    rechargeVal:5, rechargeDesc:'+5 passive. Decays 8% per fire (satisfaction plateaus). On play: all recharge stacks fire simultaneously.',
    decayRate:0.08,
    rechargeEffect:{ passive:5, burstAllStacks:true },
    drainVal:3, drainDesc:'-3 drain (reversed). Trauma heals 15. Attn reads +10 above actual.',
    traumaHealing:15, color:'#7090C8',
    pts:[[0.5,0.05],[0.73,0.18],[0.861,0.383],[0.817,0.725],[0.609,0.906],[0.391,0.906],[0.16,0.747],[0.139,0.383],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5]],
    tags:['recharge','burst','satisfaction'] },

  { id:'ten_cups', name:'Ten of Cups', suit:'cups', layer:'ID',
    keywords:'bliss · family · home', axiumScore:7, tier:1,
    rechargeVal:6, rechargeDesc:'+6 passive (peak cups). Decays 6% per fire (bliss is not a steady state). On play: all recharge cards in hand gain +2.',
    decayRate:0.06,
    rechargeEffect:{ passive:6, boostAllRecharge:2 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 18. Synergies blocked.',
    traumaHealing:18, color:'#8090C8',
    pts:[[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.5,0.27],[0.236,0.382],[0.755,0.382],[0.5,0.5],[0.236,0.618],[0.764,0.618],[0.5,0.72],[0.5,0.95],[0.5,0.382],[0.387,0.387],[0.613,0.387]],
    tags:['recharge','home'] },

  // ── SWORDS ─────────────────────────────────────────────────
  { id:'ace_swords', name:'Ace of Swords', suit:'swords', layer:'ID',
    keywords:'clarity · cut · breakthrough', axiumScore:6, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: pierce all shields — bypass opponent shield values.',
    rechargeEffect:{ passive:3, pierceShields:true },
    drainVal:5, drainDesc:'-5 drain (reversed). Shield effects bypassed against player.',
    traumaHealing:0, color:'#C0C8D0',
    pts:[[0.5,0.5],[0.618,0.5],[0.5,0.618],[0.382,0.5],[0.5,0.382],[0.387,0.387],[0.387,0.613]],
    tags:['recharge','shield-pierce'] },

  { id:'two_swords', name:'Two of Swords', suit:'swords', layer:'ID',
    keywords:'stalemate · avoidance · blindfold', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: lock trauma attn at current for 2 turns.',
    rechargeEffect:{ passive:2, lockTraumaAttn:2 },
    drainVal:6, drainDesc:'-6 drain (reversed). Lock player attn — no upward movement.',
    traumaHealing:0, color:'#A0A8B0',
    pts:[[0.726,0.274],[0.726,0.726],[0.274,0.726],[0.274,0.274],[0.5,0.5],[0.382,0.5],[0.5,0.382],[0.5,0.618]],
    tags:['recharge','lock'] },

  { id:'three_swords', name:'Three of Swords', suit:'swords', layer:'ID',
    keywords:'heartbreak · grief · pain', axiumScore:5, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: absorb unblockable trauma hits.',
    rechargeEffect:{ passive:2, absorbUnblockable:true },
    drainVal:8, drainDesc:'-8 drain (reversed). Unblockable — only decompression cards absorb.',
    traumaHealing:0, color:'#8090A0',
    pts:[[0.502,0.17],[0.613,0.387],[0.82,0.5],[0.613,0.613],[0.5,0.83],[0.387,0.613],[0.11,0.548],[0.387,0.387],[0.5,0.5]],
    tags:['recharge','grief','unblockable'] },

  { id:'four_swords', name:'Four of Swords', suit:'swords', layer:'ID',
    keywords:'rest · recovery · recuperation', axiumScore:5, tier:1,
    rechargeVal:5, rechargeDesc:"+5 passive (peak swords). Only fires when your attention is between 25% and 75% — rest requires a window. On play: skip trauma's next turn.",
    fireCondition:{ minPct: 0.25, maxPct: 0.75 },
    rechargeEffect:{ passive:5, skipTraumaTurn:true },
    drainVal:3, drainDesc:'-3 drain (reversed). Trauma heals 14. Locks player pass next turn.',
    traumaHealing:14, color:'#708090',
    pts:[[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387],[0.5,0.382],[0.5,0.618]],
    tags:['recharge','rest','skip'] },

  { id:'five_swords', name:'Five of Swords', suit:'swords', layer:'ID',
    keywords:'conflict · hollow victory', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: deal 8 attn to trauma but cost player 3.',
    rechargeEffect:{ passive:2, tradeOff:{ deal:8, cost:3 } },
    drainVal:6, drainDesc:'-6 drain (reversed). Compression cards -5 flat this battle.',
    traumaHealing:0, color:'#9090A0',
    pts:[[0.5,0.05],[0.613,0.387],[0.861,0.383],[0.691,0.61],[0.715,0.83],[0.5,0.72],[0.285,0.83],[0.309,0.61],[0.139,0.383],[0.387,0.387]],
    tags:['recharge','conflict'] },

  { id:'six_swords', name:'Six of Swords', suit:'swords', layer:'ID',
    keywords:'transition · moving on · passage', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: move to next recharge threshold immediately.',
    rechargeEffect:{ passive:3, jumpThreshold:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 6. Last played card returned to deck.',
    traumaHealing:6, color:'#6080A0',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5]],
    tags:['recharge','transition'] },

  { id:'seven_swords', name:'Seven of Swords', suit:'swords', layer:'ID',
    keywords:'strategy · shadow · cunning', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: your recharge stacks are hidden from trauma AI.',
    rechargeEffect:{ passive:3, hideFromTrauma:true },
    drainVal:5, drainDesc:"-5 drain (reversed). Trauma's next card hidden from reveal effects.",
    traumaHealing:0, color:'#7080A0',
    pts:[[0.5,0.05],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.876,0.637],[0.613,0.613],[0.715,0.83],[0.5,0.72],[0.285,0.83],[0.387,0.613],[0.11,0.548],[0.28,0.5]],
    tags:['recharge','hidden'] },

  { id:'eight_swords', name:'Eight of Swords', suit:'swords', layer:'ID',
    keywords:'paralysis · fear · trapped', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: remove 1 lock status from your attention.',
    rechargeEffect:{ passive:2, removeLock:true },
    drainVal:7, drainDesc:'-7 drain (reversed). Hand locked at 2 for 2 turns.',
    traumaHealing:0, color:'#5060A0',
    pts:[[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['recharge','paralysis','remove-lock'] },

  { id:'nine_swords', name:'Nine of Swords', suit:'swords', layer:'ID',
    keywords:'anxiety · nightmare · dread', axiumScore:3, tier:1,
    rechargeVal:1, rechargeDesc:'+1 passive (lowest). Only fires when your attention is below 55%. On play: negate next trauma card entirely.',
    fireCondition:{ maxPct: 0.55 },
    rechargeEffect:{ passive:1, negateNextTrauma:true },
    drainVal:9, drainDesc:'-9 drain (reversed). Next decompression pre-negated.',
    traumaHealing:0, color:'#404090',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297]],
    tags:['recharge','anxiety','negate'] },

  { id:'ten_swords', name:'Ten of Swords', suit:'swords', layer:'ID',
    keywords:'endings · collapse · new dawn', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: max drain spike — converts to equal recharge next turn.',
    rechargeEffect:{ passive:3, drainThenRecharge:{ drain:10, recharge:10, delay:1 } },
    drainVal:10, drainDesc:'-10 drain (reversed). Only fires when trauma above 50%.',
    traumaHealing:0, color:'#202040',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.691,0.61],[0.596,0.698],[0.72,0.5],[0.5,0.72],[0.387,0.613],[0.405,0.698],[0.382,0.5],[0.309,0.61],[0.5,0.382]],
    tags:['recharge','collapse','convert'] },

  // ── WANDS ─────────────────────────────────────────────────────
  { id:'ace_wands', name:'Ace of Wands', suit:'wands', layer:'ID',
    keywords:'spark · fire · impulse', axiumScore:6, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: highest chunk card in hand fires x2.',
    rechargeEffect:{ passive:4, doubleTopChunk:true },
    drainVal:5, drainDesc:'-5 drain (reversed). Must play a card immediately — cannot pass.',
    traumaHealing:0, color:'#E06020',
    pts:[[0.5,0.05],[0.817,0.725],[0.16,0.747],[0.5,0.5],[0.382,0.5],[0.5,0.618],[0.5,0.382]],
    tags:['recharge','spark','amplify-chunk'] },

  { id:'two_wands', name:'Two of Wands', suit:'wands', layer:'ID',
    keywords:'planning · vision · decision', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: draw 1 extra and see it before committing.',
    rechargeEffect:{ passive:3, peekDraw:1 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 5. Start next turn without drawing.',
    traumaHealing:5, color:'#E05010',
    pts:[[0.05,0.236],[0.95,0.236],[0.95,0.764],[0.05,0.764],[0.391,0.199],[0.391,0.801],[0.5,0.5],[0.382,0.5],[0.609,0.199]],
    tags:['recharge','planning'] },

  { id:'three_wands', name:'Three of Wands', suit:'wands', layer:'ID',
    keywords:'ambition · horizon · momentum', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: next recharge stack builds double.',
    rechargeEffect:{ passive:3, doubleNextStack:true },
    drainVal:5, drainDesc:'-5 drain (reversed). Next attn gain reduced by 5.',
    traumaHealing:0, color:'#D05010',
    pts:[[0.5,0.95],[0.136,0.29],[0.871,0.29],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['recharge','ambition'] },

  { id:'four_wands', name:'Four of Wands', suit:'wands', layer:'ID',
    keywords:'celebration · homecoming · harvest', axiumScore:5, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: all recharge cards gain +1 passive for 2 turns.',
    rechargeEffect:{ passive:4, boostAll:{ bonus:1, turns:2 } },
    drainVal:3, drainDesc:'-3 drain (reversed). Trauma heals 10. Attn capped at Presence.',
    traumaHealing:10, color:'#C04010',
    pts:[[0.391,0.199],[0.609,0.199],[0.609,0.801],[0.391,0.801],[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29]],
    tags:['recharge','celebration'] },

  { id:'five_wands', name:'Five of Wands', suit:'wands', layer:'ID',
    keywords:'competition · conflict · friction', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: force trauma to play lowest-value card next.',
    rechargeEffect:{ passive:2, forceTraumaLowest:true },
    drainVal:6, drainDesc:'-6 drain (reversed). Player must play lowest-shift card first.',
    traumaHealing:0, color:'#B03008',
    pts:[[0.5,0.05],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.27,0.18],[0.5,0.27],[0.5,0.72]],
    tags:['recharge','conflict'] },

  { id:'six_wands', name:'Six of Wands', suit:'wands', layer:'ID',
    keywords:'victory · recognition · return', axiumScore:6, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: all passive stacks fire once immediately.',
    rechargeEffect:{ passive:4, burstAllPassive:true },
    drainVal:3, drainDesc:'-3 drain (reversed). Trauma heals 9. Attn gains capped at +8.',
    traumaHealing:9, color:'#A02808',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tags:['recharge','victory','burst'] },

  { id:'seven_wands', name:'Seven of Wands', suit:'wands', layer:'ID',
    keywords:'defense · resolve · holding ground', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: double your shield values this battle.',
    rechargeEffect:{ passive:3, doubleShields:true },
    drainVal:6, drainDesc:'-6 drain (reversed). Only 1 card per turn.',
    traumaHealing:0, color:'#902008',
    pts:[[0.5,0.05],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.27,0.18],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]],
    tags:['recharge','defense','shield-boost'] },

  { id:'eight_wands', name:'Eight of Wands', suit:'wands', layer:'ID',
    keywords:'speed · momentum · messages', axiumScore:5, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: all recharge cards in hand add their passive immediately.',
    rechargeEffect:{ passive:4, flushPassive:true },
    drainVal:5, drainDesc:'-5 drain (reversed). Trauma plays an extra card at 50% drain.',
    traumaHealing:0, color:'#801808',
    pts:[[0.05,0.05],[0.95,0.05],[0.95,0.95],[0.05,0.95],[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]],
    tags:['recharge','speed','flush'] },

  { id:'nine_wands', name:'Nine of Wands', suit:'wands', layer:'ID',
    keywords:'resilience · persistence · last push', axiumScore:5, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: all drain stacks reduced by 2.',
    rechargeEffect:{ passive:4, reduceDrain:2 },
    drainVal:5, drainDesc:'-5 drain (reversed). Trauma heals 6. Exhaust capacity -5.',
    traumaHealing:6, color:'#701008',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.5,0.72],[0.28,0.5]],
    tags:['recharge','resilience','reduce-drain'] },

  { id:'ten_wands', name:'Ten of Wands', suit:'wands', layer:'ID',
    keywords:'burden · overload · responsibility', axiumScore:4, tier:1,
    rechargeVal:2, decayRate:0.07, rechargeDesc:'+2 passive (fatigues 7% per fire — the burden weighs). On play: remove 2 corruption cards from deck.',
    rechargeEffect:{ passive:2, removeCorruption:2 },
    drainVal:8, drainDesc:'-8 drain (reversed). Hand limit 2, all costs +3 for 2 turns.',
    traumaHealing:0, color:'#600808',
    pts:[[0.27,0.18],[0.73,0.18],[0.73,0.82],[0.27,0.82],[0.5,0.05],[0.5,0.95],[0.382,0.5],[0.618,0.5],[0.5,0.382],[0.5,0.618],[0.387,0.387],[0.613,0.387],[0.613,0.613],[0.387,0.613]],
    tags:['recharge','burden'] },

  // ── PENTACLES ─────────────────────────────────────────────────
  { id:'ace_pents', name:'Ace of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'seed · material · opportunity', axiumScore:6, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: +6 attn AND block next drain stack.',
    rechargeEffect:{ passive:3, onPlay:6, blockNextDrain:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 8. No corruption removal this turn.',
    traumaHealing:8, color:'#90B850',
    pts:[[0.5,0.05],[0.82,0.5],[0.5,0.95],[0.18,0.5],[0.5,0.27],[0.691,0.5],[0.5,0.72],[0.309,0.5]],
    tags:['recharge','opportunity'] },

  { id:'two_pents', name:'Two of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'balance · juggling · adaptation', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: equalise drain and recharge to their average.',
    rechargeEffect:{ passive:3, balanceDrainRecharge:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Must play exactly 2 cards or lose 4 attn.',
    traumaHealing:6, color:'#80A840',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.387,0.613],[0.382,0.5],[0.387,0.387],[0.5,0.28],[0.613,0.387],[0.5,0.5]],
    tags:['recharge','balance'] },

  { id:'three_pents', name:'Three of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'craft · collaboration · mastery-in-progress', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: synergy effects fire +3 extra shift.',
    rechargeEffect:{ passive:3, synergyBonus:3 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 7. Synergy effects -4.',
    traumaHealing:7, color:'#70A030',
    pts:[[0.5,0.05],[0.817,0.725],[0.16,0.747],[0.5,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]],
    tags:['recharge','craft','synergy-boost'] },

  { id:'four_pents', name:'Four of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'holding · fear of loss · security', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: hold current attn — no drop for 1 turn.',
    rechargeEffect:{ passive:2, holdAttn:1 },
    drainVal:5, drainDesc:'-5 drain (reversed). Cannot discard cards this turn.',
    traumaHealing:0, color:'#60A020',
    pts:[[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5]],
    tags:['recharge','security'] },

  { id:'five_pents', name:'Five of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'hardship · exclusion · poverty', axiumScore:4, tier:1,
    rechargeVal:2, rechargeDesc:'+2 passive. On play: remove ceiling effects on your attn.',
    rechargeEffect:{ passive:2, removeCeiling:true },
    drainVal:7, drainDesc:'-7 drain (reversed). Attn cannot rise above Witness for 2 turns.',
    traumaHealing:0, color:'#509020',
    pts:[[0.502,0.17],[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.27],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tags:['recharge','hardship'] },

  { id:'six_pents', name:'Six of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'generosity · power · exchange', axiumScore:6, tier:1,
    rechargeVal:4, rechargeDesc:'+4 passive. On play: distribute 2 recharge stacks to each card in hand.',
    rechargeEffect:{ passive:4, distributeRecharge:2 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 12.',
    traumaHealing:12, color:'#408018',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382],[0.5,0.618]],
    tags:['recharge','generosity'] },

  { id:'seven_pents', name:'Seven of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'patience · investment · assessment', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: gain +1 recharge for each card held in hand.',
    rechargeEffect:{ passive:3, bonusPerHeld:1 },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 8. All passives -1 per turn.',
    traumaHealing:8, color:'#308010',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.602],[0.715,0.83],[0.285,0.83],[0.051,0.602],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382]],
    tags:['recharge','patience','scaling'] },

  { id:'eight_pents', name:'Eight of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'craft · repetition · skill', axiumScore:5, tier:1,
    rechargeVal:3, rechargeDesc:'+3 passive. On play: repeat the effect of the last recharge card played.',
    rechargeEffect:{ passive:3, repeatLastRecharge:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 5. Repeats last trauma at half.',
    traumaHealing:5, color:'#207010',
    pts:[[0.27,0.18],[0.73,0.18],[0.73,0.82],[0.27,0.82],[0.5,0.05],[0.5,0.95],[0.382,0.5],[0.618,0.5],[0.5,0.382],[0.5,0.618],[0.5,0.27],[0.5,0.72]],
    tags:['recharge','craft','repeat'] },

  { id:'nine_pents', name:'Nine of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'independence · abundance · self-sufficiency', axiumScore:6, tier:1,
    rechargeVal:5, rechargeDesc:'+5 passive. On play: +10 attn burst. Synergies fire even in isolation.',
    rechargeEffect:{ passive:5, onPlay:10, ignoreSynergyBlock:true },
    drainVal:4, drainDesc:'-4 drain (reversed). Trauma heals 12. Synergies blocked.',
    traumaHealing:12, color:'#106808',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5]],
    tags:['recharge','abundance','independence'] },

  { id:'ten_pents', name:'Ten of Pentacles', suit:'pentacles', layer:'ID',
    keywords:'legacy · wealth · rootedness', axiumScore:7, tier:1,
    rechargeVal:6, decayRate:0.05, rechargeDesc:'+6 passive (fatigues 5% per fire — legacy is generative but not infinite). On play: all held cards gain +1 recharge permanently.',
    rechargeEffect:{ passive:6, permanentRecharge:1 },
    drainVal:5, drainDesc:'-5 drain (reversed). Trauma heals 15. Adds 1 corruption to deck.',
    traumaHealing:15, color:'#006000',
    pts:[[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5],[0.5,0.5]],
    tags:['recharge','legacy','permanent'] },
];

// ═══════════════════════════════════════════════════════════════
// TRAUMA DECKS — Chapter 1 only for now
// Trauma plays ID cards REVERSED = DRAIN effects
// drainApplied = stacks applied to player this battle
// traumaHealing = trauma self-heals
// ═══════════════════════════════════════════════════════════════
const TRAUMA_DECKS = {

  chapter_1: [
    { ...ID_CARDS.find(c=>c.id==='ace_cups'),
      traumaRole:'The Approval Feed', activeEffect:'drain',
      drainApplied:5, traumaHealing:10,
      effectDesc:'False grace. Applies 5 drain stacks. Heals 10 trauma coherence.' },
    { ...ID_CARDS.find(c=>c.id==='three_cups'),
      traumaRole:'The Performance', activeEffect:'drain',
      drainApplied:4, traumaHealing:12,
      effectDesc:'Hollow warmth. Applies 4 drain. Heals 12. Player draw -1 next battle.' },
    { ...ID_CARDS.find(c=>c.id==='nine_cups'),
      traumaRole:'The Flattery', activeEffect:'drain',
      drainApplied:3, traumaHealing:15,
      effectDesc:'False satisfaction. Applies 3 drain. Heals 15. Attn reads +10 above actual.',
      special:{ falseAttnBonus:10, turns:2 } },
    { ...ID_CARDS.find(c=>c.id==='four_cups'),
      traumaRole:'The Centering', activeEffect:'drain',
      drainApplied:6, traumaHealing:6,
      effectDesc:'Apathy injection. Applies 6 drain stacks. Synergies blocked next battle.',
      special:{ synergiesBlocked:true } },
    { ...ID_CARDS.find(c=>c.id==='two_cups'),
      traumaRole:'The Recognition Loop', activeEffect:'drain',
      drainApplied:4, traumaHealing:8,
      effectDesc:'Surface mirror. Applies 4 drain. Heals 8. Player draw randomised.',
      special:{ scatterDraw:true } },
    { ...ID_CARDS.find(c=>c.id==='five_cups'),
      traumaRole:'The Withdrawal', activeEffect:'drain',
      drainApplied:3, traumaHealing:15,
      effectDesc:'Retreat to restore. Applies 3 drain. Heals 15.' },
    { ...ID_CARDS.find(c=>c.id==='ten_swords'),
      traumaRole:'The Collapse', activeEffect:'drain',
      drainApplied:10, traumaHealing:0,
      effectDesc:'Desperation. Applies 10 drain. Only fires when trauma coherence < 30%.',
      special:{ condition:'traumaCoherence < 30' } },
  ],
};

// ═══════════════════════════════════════════════════════════════
// SHOP OFFERINGS — Tier 2 Ego cards become available after win
// ═══════════════════════════════════════════════════════════════
const SHOP_OFFERINGS = {
  chapter_1: [
    EGO_CARDS.find(c=>c.id==='page_cups'),
    EGO_CARDS.find(c=>c.id==='page_swords'),
    EGO_CARDS.find(c=>c.id==='page_wands'),
    EGO_CARDS.find(c=>c.id==='page_pents'),
    EGO_CARDS.find(c=>c.id==='knight_cups'),
    EGO_CARDS.find(c=>c.id==='knight_wands'),
    EGO_CARDS.find(c=>c.id==='knight_pents'),
  ],
};

// ═══════════════════════════════════════════════════════════════
// SYNERGIES — fire when named cards are all present in the hand
// ═══════════════════════════════════════════════════════════════
const SYNERGIES = [

  // ── Two-card synergies ─────────────────────────────────────
  {
    id:'dual_light', name:'Dual Light',
    cards:['high_priestess','moon'],
    desc:'Reveal AND reduce the intensity of all trauma drain stacks by 3. The mystery and the cycle together illuminate what neither could alone.',
    effect:{ revealAll:true, reduceDrainStacks:3 },
    visual:'#8080C8',
  },
  {
    id:'sacred_law', name:'Sacred Law',
    cards:['emperor','justice'],
    desc:'Set an attention floor AND mirror trauma\'s last drain as recharge. Structure enforces consequence.',
    effect:{ attnFloor:3, mirrorLastDrain:true },
    visual:'#70C0A0',
  },
  {
    id:'the_liberation', name:'The Liberation',
    cards:['devil','tower'],
    desc:'Destroy all fabrications in hand AND clear all drain stacks from both sides. The named shadow and the fallen tower together.',
    effect:{ destroyFabrications:true, clearAllDrainBothSides:true },
    visual:'#E06040',
  },
  {
    id:'threshold_crossing', name:'Threshold Crossing',
    cards:['death','judgement'],
    desc:'Convert all drain stacks to recharge AND clear all debuffs. The ending and the calling heard simultaneously.',
    effect:{ convertAllDrainToRecharge:true, clearAllDebuffs:true },
    visual:'#808090',
  },
  {
    id:'the_alchemist', name:'The Alchemist',
    cards:['temperance','star'],
    desc:'Convert half drain to recharge AND restore 3 recharge stacks. Patience and hope in the same breath.',
    effect:{ convertDrainToRecharge:0.5, restoreRechargeStacks:3 },
    visual:'#A0C0FF',
  },
  {
    id:'radiant_cycle', name:'Radiant Cycle',
    cards:['sun','wheel_of_fortune'],
    desc:'All recharge values doubled AND all synergies fire twice this battle. Destiny reshuffled at the peak.',
    effect:{ doubleRecharge:true, doubleAllSynergies:true },
    visual:'#FFD060',
  },
  {
    id:'sovereign_will', name:'Sovereign Will',
    cards:['chariot','emperor'],
    desc:'Shield values doubled AND attention floor locked for 2 turns. Will disciplined by structure is unstoppable.',
    effect:{ doubleShields:true, attnFloor:2 },
    visual:'#C8A860',
  },
  {
    id:'the_mirror_revealed', name:'The Mirror Revealed',
    cards:['magician','high_priestess'],
    desc:'Replay one card from discard at no cost AND reveal all trauma drain stacks. Above and below, inside and out.',
    effect:{ recycleDiscard:1, revealTraumaHand:true },
    visual:'#7EB8E8',
  },
  {
    id:'deep_surrender', name:'Deep Surrender',
    cards:['hanged_man','hermit'],
    desc:'Skip trauma\'s top card AND freeze trauma drain regeneration for 2 turns. Stillness as the deepest strategy.',
    effect:{ skipFirstTrauma:true, traumaDrainFreeze:2 },
    visual:'#60B0D8',
  },
  {
    id:'sacred_union', name:'Sacred Union',
    cards:['lovers','temperance'],
    desc:'Free-play one card AND convert all drain stacks to recharge. The true choice and the alchemical pause.',
    effect:{ freePlay:1, convertAllDrainToRecharge:true },
    visual:'#F4A0C8',
  },
  {
    id:'fool_and_world', name:'The Full Circle',
    cards:['fool','world'],
    desc:'Redraw 4 cards at 0 cost AND attention jumps to Presence minimum. The beginning and the completion held simultaneously.',
    effect:{ redraw:4, attnFloorState:'presence' },
    visual:'#FFFFFF',
  },
  {
    id:'recharge_surge', name:'Recharge Surge',
    cards:['nine_cups','ten_cups'],
    desc:'All recharge stacks burst simultaneously AND each gives +2 extra. The satisfaction and the home fire together.',
    effect:{ burstAllStacks:true, rechargeBonus:2 },
    visual:'#8090C8',
  },
  {
    id:'the_anchor', name:'The Anchor',
    cards:['four_swords','seven_wands'],
    desc:'Skip trauma turn AND double shield values. Rest and resolve held simultaneously.',
    effect:{ skipFirstTrauma:true, doubleShields:true },
    visual:'#708090',
  },
  {
    id:'abundance_flow', name:'Abundance Flow',
    cards:['six_pents','nine_pents'],
    desc:'Distribute 4 recharge stacks to each card in hand AND synergies fire in isolation. Generosity and independence united.',
    effect:{ distributeRecharge:4, ignoreSynergyBlock:true },
    visual:'#408018',
  },

  // ── Three-card synergies ────────────────────────────────────
  {
    id:'the_great_work', name:'The Great Work',
    cards:['magician','emperor','world'],
    desc:'Will, structure, and completion. All chunk multipliers stack additively. Recharge triples. Player attention reaches Clarity minimum.',
    effect:{ stackChunks:true, tripleRecharge:true, attnFloorState:'clarity' },
    visual:'#FFFFFF',
    rare:true,
  },
  {
    id:'trinity_of_light', name:'Trinity of Light',
    cards:['high_priestess','star','sun'],
    desc:'Reveal, hope, and radiance. All drain stacks nullified. Recharge doubled. Attention floor at Presence for 3 turns.',
    effect:{ clearAllDrain:true, doubleRecharge:true, attnFloor:3, attnFloorState:'presence' },
    visual:'#FFD060',
    rare:true,
  },
  {
    id:'shadow_complete', name:'Shadow Complete',
    cards:['devil','death','tower'],
    desc:'The shadow named, the old form ended, the false structure fallen. All drain cleared. All fabrications destroyed. Recharge surges to maximum.',
    effect:{ clearAllDrainBothSides:true, destroyFabrications:true, burstRecharge:true },
    visual:'#E06040',
    rare:true,
  },
  {
    id:'perfect_hand', name:'The Axium',
    cards:['sun','judgement','world'],
    desc:'All three 10-axium cards held. Win condition triggered immediately. The full circle, fully conscious, fully arrived.',
    effect:{ instantWin:true },
    visual:'#FFFFFF',
    rare:true,
  },
];

// ═══════════════════════════════════════════════════════════════
// CHAPTER METADATA — Chapter 1 only
// ═══════════════════════════════════════════════════════════════
const CHAPTERS = [
  {
    id:1,
    label:'01 · The Unchecked Ego',
    title:'The Unchecked Ego',
    axium:'"The absence of love binds ego, allowing the awareness of God."',
    enemy:'Ego Unchecked',
    enemyRole:'The Performance',
    traumaDeck:'chapter_1',
    shopOffers:'chapter_1',
    traumaStart:80,
    playerStart:55,
    maxAttn:100,           // base max attention pool
    handSize:6,            // starting hand size (grows with capacity cards)
    winCondition:'Reach 90+ attention after the battle resolves.',
    loseCondition:'Attention drops to Fragmented (0-10) after the battle.',
    narrative:'The ego runs its scripts. Every performance, every approval loop — they are bids for your attention. Refuse them long enough, and they exhaust themselves.',
  },
];

// ───────────────────────────────────────────────────────────────
// SOUL SHOP NPCs
// ───────────────────────────────────────────────────────────────
const SHOP_NPCS = {
  chapter_1: {
    name:'Micheal',
    role:'Digital Soul · The Anchor',
    speeches:[
      "The trauma didn't break you — it ran out of material. Here is what's missing from your build.",
      "Three options. One right answer. I'm not going to tell you which one. But the one that makes you hesitate is probably it.",
      "You held the center. Now let's make it harder to lose next time.",
      "The ego ran its scripts. You stayed still. That's rarer than you think.",
    ],
  },
};

// ───────────────────────────────────────────────────────────────
// ALL CARDS — flat lookup pool
// ───────────────────────────────────────────────────────────────
const ALL_CARDS = [
  ...PLAYER_CARDS,
  ...EGO_CARDS,
  ...ID_CARDS,
  ...Object.values(TRAUMA_DECKS).flat(),
];

// ═══════════════════════════════════════════════════════════════
// NODE RESONANCE ENGINE — v4.0
// Single source of truth for all battle math.
// battle.js reads these constants and helpers directly.
// ═══════════════════════════════════════════════════════════════

/**
 * CAP_TICK_RATE — fraction of capacityVal applied to max pool
 * on each subsequent Superego fire after the initial shield burst.
 * With a 17-node Star (T1, interval ~470ms) and capacityVal 12,
 * each tick adds 12 × 0.08 = 0.96 to pMax — roughly +57 pMax over
 * the 60s battle if it fires continuously. Feels like slow growth.
 */
const CAP_TICK_RATE = 0.08;

/**
 * ─────────────────────────────────────────────────────────────────
 * NEW MECHANIC CONSTANTS  (v4.1 — Six Emergent Systems)
 * ─────────────────────────────────────────────────────────────────
 *
 * 1. RESONANCE DECAY
 *    Upright ID cards lose a fraction of their effective rechargeVal
 *    each time they fire, representing attention fatigue.
 *    Each card can define  decayRate: 0–1  (default 0 = no decay).
 *    Applied multiplicatively per fire count on that card.
 *    Effective recharge = rechargeVal × (1 - decayRate)^fireCount
 *    High-node cards reach their ceiling faster, forcing polarity
 *    decisions mid-battle.
 *
 * 2. THRESHOLD LOCKS
 *    A card may define  fireCondition: { minPct?, maxPct? }
 *    where pct is current playerAttn / playerMaxAttn (0–1).
 *    If the condition fails the card's timer still runs but the
 *    fire is skipped (no delta, no label) — it waits for the
 *    right window.  Enemy threshold cards check enemyAttn/eMax.
 *
 * 3. POLARITY MOMENTUM
 *    Tracked entirely in preSimulateBattle():
 *      reversalStack — incremented each time a reversed player ID
 *        card fires. Decays by REVERSAL_DECAY per non-reversed fire.
 *      Drain output multiplied by (1 + reversalStack × REVERSAL_MULT).
 *      Once reversalStack ≥ REVERSAL_FRAG_THRESHOLD a fragmentation
 *      debuff is applied: pMult is reduced by REVERSAL_FRAG_PENALTY
 *      for the rest of the battle.
 *
 * 4. CONSTELLATION INTERFERENCE
 *    Cards of the same suit staged together in playerPlayed suppress
 *    each other. For each additional same-suit card beyond the first,
 *    effective rechargeVal is multiplied by SUIT_SUPPRESS_MULT.
 *    Computed in computeCardFire via the suitSuppressCount parameter.
 *
 * 5. SURGE FRAGMENTATION
 *    On the first fire cycle after surge begins (t > 45 000ms),
 *    reversed player ID cards backfire: drain is split 50/50 between
 *    enemy and player. Tracked via the surgeSplitActive flag in the
 *    pre-sim loop; resets after the first full fire cycle.
 *
 * 6. CAPACITY BLEED
 *    Upright Superego cards accumulate a bleed counter each subsequent
 *    fire. Effective capacityDelta is reduced by
 *      bleedFactor × fireCount (clamped to 0).
 *    CAP_BLEED_RATE controls how fast the shield erodes.
 *    Enemy reversed Superego cards apply double bleed to the target's
 *    effective capacityVal.
 * ─────────────────────────────────────────────────────────────────
 */

/** Resonance Decay: default fractional decay per fire for ID upright cards. */
const DECAY_DEFAULT      = 0;     // base; per-card decayRate overrides
/** Polarity Momentum: drain multiplier per reversal stack level. */
const REVERSAL_MULT      = 0.12;
/** Polarity Momentum: how much the stack decays per non-reversed fire. */
const REVERSAL_DECAY     = 0.5;
/** Polarity Momentum: stack level that triggers the fragmentation debuff. */
const REVERSAL_FRAG_THRESHOLD = 4;
/** Polarity Momentum: pMult penalty applied when fragmented. */
const REVERSAL_FRAG_PENALTY   = 0.25;
/** Constellation Interference: rechargeVal multiplier per extra same-suit card. */
const SUIT_SUPPRESS_MULT = 0.82;
/** Capacity Bleed: capacityDelta reduction per subsequent Superego fire. */
const CAP_BLEED_RATE     = 0.06;

/**
 * nodeFireInterval(card, surge)
 * The time in ms between fires for this card.
 * Derived entirely from tier base interval and node count.
 * More nodes = smaller interval = faster firing.
 */
function nodeFireInterval(card, surge) {
  const tier     = card.tier || 1;
  const base     = tier >= 3 ? 2000 : tier === 2 ? 4000 : 8000;
  const interval = surge ? base / 10 : base;
  const nodes    = (card.pts && card.pts.length) ? card.pts.length : 7;
  return Math.round(interval / nodes);
}

/**
 * getSynergyBonuses(playerPlayedEntries)
 * Evaluate all active synergies from upright cards in playerPlayed.
 * Returns { attnBoost, pMultBonus, activeSynergies[] }
 *
 * attnBoost   — flat attn added to player at t=0.
 * pMultBonus  — multiplicative boost folded into pMult at battle start.
 * activeSynergies — array of matched SYNERGIES objects for display.
 */
function getSynergyBonuses(playerPlayedEntries) {
  const uprightIds = playerPlayedEntries
    .filter(e => !e.reversed)
    .map(e => e.card.id);

  const activeSynergies = SYNERGIES.filter(s =>
    s.cards.every(id => uprightIds.includes(id))
  );

  let attnBoost  = 0;
  let pMultBonus = 1.0;

  activeSynergies.forEach(syn => {
    if (syn.effect?.attnBoost)   attnBoost  += syn.effect.attnBoost;
    if (syn.effect?.synergyMult) pMultBonus *= syn.effect.synergyMult;
    // doubleRecharge handled in sim as pMult * 2 on ID cards — represented here
    if (syn.effect?.doubleRecharge) pMultBonus *= 2.0;
    // instantWin handled in sim loop
  });

  return { attnBoost, pMultBonus, activeSynergies };
}

/**
 * computeCardFire(c, pMult, pFlat, eDivisor, eDmgFlat, simCtx)
 * Pure function — given a card entry and current global modifiers,
 * returns what this card does when it fires.
 *
 * c = { card, reversed, side, firstFired, fireCount, bleedCount }
 *   fireCount  — how many times this card has fired (for Resonance Decay)
 *   bleedCount — how many subsequent Superego fires (for Capacity Bleed)
 *
 * simCtx = {
 *   pAttn, pMax,          — for Threshold Locks (player side)
 *   eAttn, eMax,          — for Threshold Locks (enemy side)
 *   suitSuppressCount,    — same-suit card count beyond 1 (Constellation Interference)
 *   surgeSplitActive,     — bool: first fire-cycle after surge (Surge Fragmentation)
 *   reversalMult,         — current drain multiplier from Polarity Momentum
 * }
 *
 * Returns { delta, capacityDelta, label, color, targetSide, egoEffect,
 *           splitDelta }
 *   splitDelta — extra delta applied to the SAME side as the firing card
 *                (Surge Fragmentation backfire: positive value = player takes damage)
 */
function computeCardFire(c, pMult, pFlat, eDivisor, eDmgFlat, simCtx) {
  const { card, reversed, side, firstFired } = c;
  const fireCount  = c.fireCount  || 0;
  const bleedCount = c.bleedCount || 0;
  const layer = card.layer;
  const ctx   = simCtx || {};

  let delta         = 0;
  let capacityDelta = 0;
  let splitDelta    = 0;   // Surge Fragmentation backfire
  let label         = '';
  let color         = '#D4AF37';
  let targetSide    = 'player';
  let egoEffect     = null;

  // ── THRESHOLD LOCK CHECK ─────────────────────────────────────
  // If the card has a fireCondition and it isn't met, skip the fire.
  if (card.fireCondition) {
    const fc    = card.fireCondition;
    const pPct  = ctx.pMax  ? (ctx.pAttn  / ctx.pMax)  : 0.5;
    const ePct  = ctx.eMax  ? (ctx.eAttn  / ctx.eMax)  : 0.5;
    const chkPct = (side === 'enemy') ? ePct : pPct;
    const minOk  = fc.minPct === undefined || chkPct >= fc.minPct;
    const maxOk  = fc.maxPct === undefined || chkPct <= fc.maxPct;
    if (!minOk || !maxOk) {
      // Condition not met — fire is skipped. Return zero delta with a dim label.
      return { delta: 0, capacityDelta: 0, splitDelta: 0,
               label: '', color, targetSide, egoEffect: null, skipped: true };
    }
  }

  // ── ID ───────────────────────────────────────────────────────
  if (layer === 'ID') {
    if (side === 'player' && !reversed) {
      // Upright player: amplified recharge
      // Mechanic 1: Resonance Decay — effective recharge shrinks each fire.
      const decayRate  = card.decayRate || DECAY_DEFAULT;
      const decayedRaw = (card.rechargeVal || 0) * Math.pow(1 - decayRate, fireCount);

      // Mechanic 4: Constellation Interference — same-suit suppression.
      const suppressed = decayedRaw * Math.pow(SUIT_SUPPRESS_MULT, ctx.suitSuppressCount || 0);

      delta      = Math.round(suppressed * pMult + pFlat);
      if (delta < 0) delta = 0; // decay can't flip to drain
      label      = `+${delta} Recharge` + (decayRate > 0 && fireCount > 0 ? ' ↓' : '');
      color      = '#86EFAC';
      targetSide = 'player';

    } else if (side === 'player' && reversed) {
      // Reversed player: raw drain on enemy, scaled by Polarity Momentum.
      const momentum  = 1 + ((ctx.reversalMult || 0) * REVERSAL_MULT);
      const rawDrain  = Math.round((card.drainVal || 0) * momentum);

      // Mechanic 5: Surge Fragmentation — on first surge fire cycle,
      // 50% of drain bounces back to the player.
      if (ctx.surgeSplitActive) {
        delta      = -Math.round(rawDrain * 0.5);
        splitDelta = -Math.round(rawDrain * 0.5);  // also hits player
        label      = `${delta} Drain ⚡split`;
        color      = '#e05555';
      } else {
        delta  = -rawDrain;
        label  = `${delta} Drain` + (momentum > 1.05 ? ' ⚑' : '');
        color  = '#e05555';
      }
      targetSide = 'enemy';

    } else {
      // Enemy (always reversed): raw drain on player.
      // NOTE: eDivisor/eDmgFlat are player-side Ego modifiers that divide
      // player's OWN gains — they do NOT soften raw enemy pressure.
      // enemyDrainMult scales ID drain pressure per chapter difficulty.
      const drainMult = ctx.enemyDrainMult || 1.0;
      delta      = -Math.round((card.drainVal || 0) * drainMult);
      label      = `${delta} Pressure`;
      color      = '#e05555';
      targetSide = 'player';
    }
  }

  // ── SUPEREGO ─────────────────────────────────────────────────
  else if (layer === 'Superego') {
    if (side === 'player' && !reversed) {
      if (!firstFired) {
        // First fire: shield burst (amplified by pMult)
        delta      = Math.round((card.shieldVal || 0) * pMult + pFlat);
        label      = `+${delta} Shield`;
        color      = '#D4AF37';
        targetSide = 'player';
      } else {
        // Subsequent fires: capacity growth on pMax
        // Mechanic 6: Capacity Bleed — capacityDelta shrinks each subsequent fire.
        const cv = card.capacityVal || 0;
        if (cv > 0) {
          const bleedReduction = bleedCount * CAP_BLEED_RATE;
          const effectiveCv    = Math.max(0, cv * (1 - bleedReduction));
          capacityDelta = effectiveCv * CAP_TICK_RATE;
          label         = capacityDelta > 0.01 ? `Max ↑` : `Max ~`;
          color         = '#D4AF37';
          targetSide    = 'player';
        }
        delta = 0;
      }
    } else if (side === 'player' && reversed) {
      if (!firstFired) {
        // First fire: burst damage on enemy (60% of shield, raw)
        delta      = -Math.round((card.shieldVal || 0) * 0.6);
        label      = `${delta} Shatter`;
        color      = '#e05555';
        targetSide = 'enemy';
      } else {
        // Subsequent fires: if capacityVal > 0, erode enemy max
        // Capacity Bleed also applies here (enemy erodes faster under pressure)
        const cv = card.capacityVal || 0;
        if (cv > 0) {
          const bleedReduction = bleedCount * CAP_BLEED_RATE;
          const effectiveCv    = Math.max(0, cv * (1 - bleedReduction));
          capacityDelta = -(effectiveCv * CAP_TICK_RATE);
          label         = `Enemy Max ↓`;
          color         = '#e05555';
          targetSide    = 'enemy';
        }
        delta = 0;
      }
    } else {
      // Enemy Superego reversed — targets player
      if (!firstFired) {
        // Burst damage on player — raw 60% of shieldVal, not softened by player's Ego modifiers.
        // eDivisor/eDmgFlat represent player's reversed Ego attacking enemy gains,
        // not a shield against enemy Superego bursts.
        delta      = -Math.round((card.shieldVal || 0) * 0.6);
        label      = `${delta} Pressure`;
        color      = '#e05555';
        targetSide = 'player';
      } else {
        // Subsequent fires: erode player max
        // Enemy reversed Superego applies double bleed rate.
        const cv = card.capacityVal || 0;
        if (cv > 0) {
          const bleedReduction = bleedCount * CAP_BLEED_RATE * 2; // double for enemy
          const effectiveCv    = Math.max(0, cv * (1 - bleedReduction));
          capacityDelta = -(effectiveCv * CAP_TICK_RATE);
          label         = `Your Max ↓`;
          color         = '#e05555';
          targetSide    = 'player';
        }
        delta = 0;
      }
    }
  }

  // ── EGO — fires once, sets global modifiers ──────────────────
  else if (layer === 'Ego') {
    if (!firstFired) {
      if (side === 'player' && !reversed) {
        // Upright: boost player multipliers going forward
        const newPMult = pMult * (card.chunkPct || 1.0);
        const newPFlat = pFlat + (card.chunkFlat || 0);
        egoEffect = { type: 'boost', pMult: newPMult, pFlat: newPFlat };
        label     = `Chunk ×${(card.chunkPct||1).toFixed(1)}${card.chunkFlat ? ` +${card.chunkFlat}` : ''}`;
        color     = '#7EB8E8';
        targetSide = 'player';
      } else if (side === 'player' && reversed) {
        // Reversed player: divide enemy gains
        const newEDiv  = eDivisor * (card.chunkPct || 1.0);
        const newEFlat = eDmgFlat + (card.chunkFlat || 0);
        egoEffect = { type: 'divide', eDivisor: newEDiv, eDmgFlat: newEFlat };
        label     = `Divide ÷${(card.chunkPct||1).toFixed(1)}`;
        color     = '#e05555';
        targetSide = 'enemy';
      } else {
        // Enemy Ego reversed: divide player gains
        const newEDiv  = eDivisor * (card.chunkPct || 1.0);
        const newEFlat = eDmgFlat + (card.chunkFlat || 0);
        egoEffect = { type: 'enemy_divide', eDivisor: newEDiv, eDmgFlat: newEFlat };
        label     = `Enemy Chunk ÷${(card.chunkPct||1).toFixed(1)}`;
        color     = '#e05555';
        targetSide = 'player';
      }
    }
    // After firstFired, Ego does nothing — fall through with delta=0
  }

  return { delta, capacityDelta, splitDelta, label, color, targetSide, egoEffect };
}

// ─────────────────────────────────────────────────────────────────
// SUIT SUPPRESS HELPER
// Returns how many additional same-suit cards share the field with
// the given card in playerPlayed (0 = no suppression, 1 = one extra, etc.)
// Called from preSimulateBattle() when building each card's simCtx.
// ─────────────────────────────────────────────────────────────────
function getSuitSuppressCount(card, playerPlayedEntries) {
  if (!card.suit) return 0; // Superego / suitless cards are never suppressed
  const suit = card.suit;
  let count = 0;
  for (const e of playerPlayedEntries) {
    const c = e.card || e;
    if (c.id !== card.id && c.suit === suit && !e.reversed) count++;
  }
  return count;
}

// ═══════════════════════════════════════════════════════════════
// STAGE ENEMY DECKS — 10 stages, escalating difficulty
//
// All enemy cards play reversed:
//   ID reversed    → drainVal hits player attn each fire
//   Superego rev   → shieldVal×0.6 burst then pMax erosion each tick
//   Ego reversed   → divides player gains (one-shot on first fire)
//
// Stage design principles:
//   Stage 1–3: Low-node ID only. Fast but small hits. Player learns timing.
//   Stage 4–5: First Superego cards appear. Burst damage introduced.
//   Stage 6–7: Ego cards enter. Player gains start getting divided.
//   Stage 8–9: High-node high-magnitude cards. Real pressure.
//   Stage 10:  Full optimised deck. Maximum node density. No mercy.
//
// enemyStart: enemy attn at battle start (player always starts lower)
// enemyMax:   enemy max attn pool
// handSize:   how many enemy cards deploy
// ═══════════════════════════════════════════════════════════════
const STAGE_ENEMY_DECKS = [

  // ── Stage 1 — The Echo Chamber ────────────────────────────────
  // Pure low-node ID. Fast drip, predictable rhythm. Learning stage.
  {
    stage:      1,
    name:       'The Echo Chamber',
    enemyStart: 60,
    enemyMax:   100,
    handSize:   4,
    ids: ['ace_cups','ace_swords','ace_wands','ace_pents'],
    // 4 × 7-node Tier 1 ID reversed
    // Fire every 8000/7 ≈ 1143ms. drain 5,5,5,4 per hit.
  },

  // ── Stage 2 — The Performance ─────────────────────────────────
  // More ID variety (mid-node), slightly harder hits.
  {
    stage:      2,
    name:       'The Performance',
    enemyStart: 65,
    enemyMax:   100,
    handSize:   5,
    ids: ['three_cups','five_cups','three_swords','four_wands','two_pents'],
    // Node range 8–10, drain 4–8. First card with drainVal 8 (three_swords).
  },

  // ── Stage 3 — The Approval Feed ───────────────────────────────
  // Higher-node ID. Slower but hits count more.
  {
    stage:      3,
    name:       'The Approval Feed',
    enemyStart: 65,
    enemyMax:   100,
    handSize:   5,
    ids: ['six_cups','seven_cups','six_swords','seven_wands','six_pents'],
    // Node range 10–12, drain 3–6. Rhythm shifts — gaps between hits lengthen.
  },

  // ── Stage 4 — The Wound ───────────────────────────────────────
  // First Superego card (Hermit reversed). Burst damage introduced.
  {
    stage:      4,
    name:       'The Wound',
    enemyStart: 70,
    enemyMax:   100,
    handSize:   6,
    ids: ['eight_cups','nine_swords','ten_cups','seven_swords','hermit','fool'],
    // hermit: 9 nodes, shieldVal 13 → burst −8 on first fire, then cap erosion.
    // fool:   7 nodes, shieldVal 18 → burst −11 very early. No cap erosion (capVal −8).
  },

  // ── Stage 5 — The Mirror ──────────────────────────────────────
  // Two Superego cards plus strong ID. Enemy max erosion begins.
  {
    stage:      5,
    name:       'The Mirror',
    enemyStart: 70,
    enemyMax:   105,
    handSize:   6,
    ids: ['ten_swords','nine_cups','magician','high_priestess','eight_swords','five_swords'],
    // magician (9n, shield 14, cap 6): burst −8, then slow pMax drain −0.48/fire.
    // high_priestess (11n, shield 12, cap 5): burst −7, then −0.4/fire.
  },

  // ── Stage 6 — The Rationaliser ────────────────────────────────
  // First Ego card (Page of Cups reversed). Player gains start dividing.
  {
    stage:      6,
    name:       'The Rationaliser',
    enemyStart: 75,
    enemyMax:   105,
    handSize:   7,
    ids: ['ten_wands','empress','chariot','strength','page_cups','nine_wands','eight_wands'],
    // page_cups (9n, T2, chunkPct 1.0, chunkFlat 4): divides player gains, +4 eDmgFlat.
    // chariot (13n, shield 22, cap 9): burst −13, ongoing cap erosion −0.72/fire.
    // empress (12n, shield 16, cap 8): burst −10, erosion −0.64/fire.
  },

  // ── Stage 7 — The Shadow ──────────────────────────────────────
  // Devil + Tower reversed — dangerous capacity collapse combo.
  {
    stage:      7,
    name:       'The Shadow',
    enemyStart: 75,
    enemyMax:   110,
    handSize:   7,
    ids: ['devil','tower','ten_cups','ten_wands','page_swords','nine_pents','eight_pents'],
    // devil (16n, shield 25, cap 11): burst −15, erosion −0.88/fire. Fastest T1 Superego.
    // tower (11n, shield 28, cap 10): burst −17, erosion −0.80/fire.
    // page_swords (9n, T2): divide + eDmgFlat 3.
  },

  // ── Stage 8 — The Void ────────────────────────────────────────
  // Star + Death reversed. High-node, high magnitude, Knight Ego.
  {
    stage:      8,
    name:       'The Void',
    enemyStart: 80,
    enemyMax:   110,
    handSize:   8,
    ids: ['star','death','knight_cups','knight_swords','ten_swords','nine_swords','ten_pents','nine_cups'],
    // star (18n, shield 20, cap 12): burst −12, fastest T1 outside sun/world/wheel.
    // death (14n, shield 20, cap 12): burst −12, erosion −0.96/fire.
    // knight_cups (11n, T2, chunkPct 1.3): ×1.3 eDivisor on player gains.
    // knight_swords (11n, T2, chunkFlat 8): +8 eDmgFlat — every enemy hit +8 raw.
  },

  // ── Stage 9 — The Collapse ────────────────────────────────────
  // Sun + Judgement reversed. Queen Ego. Maximum pressure.
  {
    stage:      9,
    name:       'The Collapse',
    enemyStart: 85,
    enemyMax:   115,
    handSize:   8,
    ids: ['sun','judgement','queen_cups','queen_swords','wheel_of_fortune','moon','ten_wands','nine_wands'],
    // sun (20n, shield 30, cap 15): burst −18 early (fast node), erosion −1.2/fire.
    // judgement (14n, shield 26, cap 14): burst −16, erosion −1.12/fire.
    // queen_cups (13n, T2, chunkPct 1.0, chunkFlat 6): +6 eDmgFlat.
    // queen_swords (13n, T2, chunkPct 1.4): ×1.4 eDivisor. Player gains crushed.
    // wheel_of_fortune (17n, shield 16, cap 7): fast erosion.
  },

  // ── Stage 10 — The Unchecked Ego ──────────────────────────────
  // Full deck. World reversed hits hardest (21 nodes, fastest T1).
  // King Ego cards. Complete optimised enemy build.
  {
    stage:      10,
    name:       'The Unchecked Ego',
    enemyStart: 90,
    enemyMax:   120,
    handSize:   8,
    ids: ['world','sun','king_cups','king_swords','devil','star','wheel_of_fortune','ten_pents'],
    // world (21n, shield 35, cap 20): burst −21 on first fire (fastest T1 card).
    //   Subsequent fires: −1.6 pMax per fire. Over 60s ≈ −94 pMax total.
    // king_cups (15n, T2, chunkPct 1.5): ×1.5 eDivisor. Player gains halved.
    // king_swords (15n, T2, chunkPct 1.6): stacks with cups → ×2.4 total divisor.
    // Combined Ego: player gains divided by up to 2.4 after both Kings fire.
  },
];

// ───────────────────────────────────────────────────────────────
// HELPERS
// ───────────────────────────────────────────────────────────────
function getCard(id) {
  return ALL_CARDS.find(c => c.id === id) || null;
}

function getTraumaDeck(chapter) {
  return TRAUMA_DECKS[`chapter_${chapter}`] || TRAUMA_DECKS.chapter_1;
}

function getShopOffers(chapter, count = 3) {
  const pool = (SHOP_OFFERINGS[`chapter_${chapter}`] || SHOP_OFFERINGS.chapter_1).filter(Boolean);
  return shuffle(pool).slice(0, count);
}

function getShopNPC(chapter) {
  return SHOP_NPCS[`chapter_${chapter}`] || SHOP_NPCS.chapter_1;
}

function getChapter(id) {
  return CHAPTERS.find(c => c.id === id) || CHAPTERS[0];
}

function getSynergies(cardIds) {
  return SYNERGIES.filter(s => s.cards.every(id => cardIds.includes(id)));
}

function getAttnState(val) {
  const pct = Math.max(0, Math.min(100, val)) / 100;
  let best  = ATTN_STATES[0];
  for (const s of ATTN_STATES) { if (pct >= s.pos) best = s; }
  return best;
}

function getStageEnemyDeck(stage) {
  const s = STAGE_ENEMY_DECKS.find(d => d.stage === stage);
  if (!s) return STAGE_ENEMY_DECKS[0];
  return {
    ...s,
    cards: s.ids.map(id => getCard(id)).filter(Boolean),
  };
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function clampAttn(val, min = 0, max = 100) {
  return Math.max(min, Math.min(max, val));
}

// ───────────────────────────────────────────────────────────────
// EXPORTS
// ───────────────────────────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ATTN_STATES, CARD_GLOSSARY, CAP_TICK_RATE,
    // New mechanic constants
    DECAY_DEFAULT, REVERSAL_MULT, REVERSAL_DECAY,
    REVERSAL_FRAG_THRESHOLD, REVERSAL_FRAG_PENALTY,
    SUIT_SUPPRESS_MULT, CAP_BLEED_RATE,
    PLAYER_CARDS, EGO_CARDS, ID_CARDS,
    TRAUMA_DECKS, SHOP_OFFERINGS, SHOP_NPCS, CHAPTERS,
    SYNERGIES, STAGE_ENEMY_DECKS, ALL_CARDS,
    // Core engine
    nodeFireInterval, getSynergyBonuses, computeCardFire, getSuitSuppressCount,
    // Helpers
    getCard, getTraumaDeck, getShopOffers, getShopNPC,
    getChapter, getSynergies, getAttnState, getStageEnemyDeck,
    shuffle, clampAttn,
  };
}

window.ALL_CARDS        = ALL_CARDS;
window.PLAYER_CARDS     = PLAYER_CARDS;
window.STAGE_ENEMY_DECKS = STAGE_ENEMY_DECKS;
