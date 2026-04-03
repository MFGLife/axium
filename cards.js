/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — CARD MATRIX v2.0
 * "Battle for Attention" — 78-card tarot-grounded system
 *
 * DESIGN PRINCIPLE:
 *   Every card has two faces — one card, two states.
 *   UPRIGHT   = player plays it = DECOMPRESSION (awareness, presence)
 *   REVERSED  = trauma plays it = COMPRESSION (manipulation, capture)
 *   The same archetype that heals you can be the wound.
 *
 * LAYER ARCHITECTURE:
 *   SUPEREGO (22 Major Arcana) — Player's core deck
 *   EGO      (16 Court Cards)  — Shop upgrades between chapters
 *   ID       (40 Pip Cards)    — Trauma deck pool + corruption
 *
 * ═══════════════════════════════════════════════════════════════
 */

// ───────────────────────────────────────────────────────────────
// ATTENTION SPECTRUM  (0 = Fragmented → 1 = Enlightened)
// ───────────────────────────────────────────────────────────────
const ATTN_STATES = [
  {
    id: 'fragmented', label: 'Fragmented', pos: 0.00, col: '#6B21A8',
    desc: 'Attention has shattered. No synergies fire. Draw −1.',
    debuff: { drawMod: -1, synergiesBlocked: true },
  },
  {
    id: 'fear', label: 'Fear', pos: 0.10, col: '#9333EA',
    desc: 'Attention collapses inward. Compression cards cost +1 intensity.',
    debuff: { compressionMod: -1 },
  },
  {
    id: 'anger', label: 'Anger', pos: 0.20, col: '#DC2626',
    desc: 'Attention narrows. Only one card can be played per turn.',
    debuff: { maxPlays: 1 },
  },
  {
    id: 'sadness', label: 'Sadness', pos: 0.30, col: '#1D4ED8',
    desc: 'Attention slows. All exhaustion costs −2 (min 1).',
    debuff: { exhaustionMod: -2 },
  },
  {
    id: 'disinterest', label: 'Disinterest', pos: 0.42, col: '#374151',
    desc: 'Attention flatlines. Synergies stop firing.',
    debuff: { synergiesBlocked: true },
  },
  {
    id: 'witness', label: 'Witness', pos: 0.54, col: '#7EB8E8',
    desc: 'Attention observes without reacting. Neutral baseline.',
    debuff: null,
  },
  {
    id: 'presence', label: 'Presence', pos: 0.65, col: '#D4AF37',
    desc: 'Attention is here, now, full. Decompression cards +2 shift.',
    buff: { decompressionMod: +2 },
  },
  {
    id: 'clarity', label: 'Clarity', pos: 0.76, col: '#86EFAC',
    desc: "Attention sees what trauma is doing. Trauma's next card revealed.",
    buff: { revealTrauma: true },
  },
  {
    id: 'ground', label: 'Grounded', pos: 0.87, col: '#F0D080',
    desc: 'Attention holds. Blocks one trauma shift per round.',
    buff: { blockOneShift: true },
  },
  {
    id: 'enlightened', label: 'Enlightened', pos: 1.00, col: '#FFFFFF',
    desc: 'Attention is complete. Chapter win condition triggered.',
    buff: { winTrigger: true },
  },
];

// ───────────────────────────────────────────────────────────────
// CARD GLOSSARY
// ───────────────────────────────────────────────────────────────
const CARD_GLOSSARY = {
  exhaust:    'Deals coherence damage to trauma. Trauma collapses at 0.',
  shield:     "Absorbs the next trauma card's shift entirely.",
  reveal:     'Show the next card in the trauma deck before it plays.',
  corrupt:    'Entered your deck via loss. Negative shift. Cannot be upgraded.',
  synergy:    'Fires automatically when named cards are both on field.',
  drift:      'Natural attention movement each turn toward center.',
  lock:       'Pins attention at a state for N turns. Breaks on decompression.',
  scatter:    'Randomises which card the player draws next turn.',
  fabricate:  'Trauma creates a false card in your hand — costs attention to discard.',
  reversed:   'The same archetype, turned against you. Trauma plays cards reversed.',
};

// ═══════════════════════════════════════════════════════════════
// SUPEREGO — 22 Major Arcana
// Player's core deck. Upright meaning drives the effect.
// Reversed meaning is what trauma does when it flips the card.
// ═══════════════════════════════════════════════════════════════
const PLAYER_CARDS = [

  // ── 0 · The Fool ─────────────────────────────────────────────
  {
    id: 'fool', name: 'The Fool',
    layer: 'Superego', number: 0,
    keywords: 'leap · pure potential · trust',
    type: 'decompression',
    intensity: 2,
    attnShift: +13,
    effectDesc: 'The abyss is also the sky. Discard your hand and draw 4 fresh cards — each costs 0 exhaustion this turn. What you cannot justify, trust.',
    targets: ['fragmented', 'fear', 'disinterest'],
    exhaustion: 4,
    reversedType: 'compression',
    reversedShift: -10,
    reversedDesc: "The leap refused. Injects paralysis — player cannot play more than 1 card next turn. The cliff's edge is occupied indefinitely.",
    reversedTarget: 'fear',
    color: '#86EFAC',
    pts: [[0.5,0.05],[0.871,0.29],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5]],
    tier: 1, chapter: 1,
    tags: ['redraw', 'trust', 'chaos'],
    synergies: ['world', 'wheel_of_fortune'],
  },

  // ── I · The Magician ─────────────────────────────────────────
  {
    id: 'magician', name: 'The Magician',
    layer: 'Superego', number: 1,
    keywords: 'will · manifestation · alignment',
    type: 'both',
    intensity: 6,
    attnShift: +9,
    effectDesc: 'Above and below are connected. Play one card from your discard pile at no cost. All tools are present — the only missing piece is alignment of will with action.',
    targets: ['disinterest', 'sadness'],
    exhaustion: 14,
    reversedType: 'compression',
    reversedShift: -9,
    reversedDesc: "The tools present but will fragmented. Player's attnShift values halved next turn — manifestation cancels itself through ambivalence.",
    reversedTarget: 'disinterest',
    color: '#D4AF37',
    pts: [[0.5,0.5],[0.618,0.5],[0.82,0.5],[0.5,0.618],[0.5,0.83],[0.382,0.5],[0.18,0.5],[0.5,0.382],[0.502,0.17]],
    tier: 1, chapter: 1,
    tags: ['recycle', 'will', 'alignment'],
    synergies: ['high_priestess', 'strength'],
  },

  // ── II · The High Priestess ──────────────────────────────────
  {
    id: 'high_priestess', name: 'The High Priestess',
    layer: 'Superego', number: 2,
    keywords: 'mystery · veil · knowing',
    type: 'decompression',
    intensity: 3,
    attnShift: +8,
    effectDesc: "Sit at the threshold. Reveal the next 3 trauma cards. Reduce each by 2 intensity. What forms in deep water surfaces when inner conditions are right.",
    targets: ['fear', 'fragmented', 'anger'],
    exhaustion: 8,
    reversedType: 'compression',
    reversedShift: -11,
    reversedDesc: "Forcing the answer before the silence speaks. Player's next draw is randomised. Pushing through the veil costs more than waiting at it.",
    reversedTarget: 'fragmented',
    color: '#7EB8E8',
    pts: [[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383],[0.274,0.274],[0.342,0.311],[0.5,0.27]],
    tier: 1, chapter: 1,
    tags: ['reveal', 'reduce', 'mystery'],
    synergies: ['magician', 'moon'],
  },

  // ── III · The Empress ────────────────────────────────────────
  {
    id: 'empress', name: 'The Empress',
    layer: 'Superego', number: 3,
    keywords: 'abundance · creation · nature',
    type: 'decompression',
    intensity: 4,
    attnShift: +10,
    effectDesc: 'The earth does not hurry. Remove 1 corruption card permanently. Gain +3 additional attention for each card in your hand — abundance multiplies when given room.',
    targets: ['sadness', 'disinterest', 'anger'],
    exhaustion: 6,
    reversedType: 'compression',
    reversedShift: -8,
    reversedDesc: 'Growth forced out of season. Player must play all cards in hand this turn or lose 3 attention per unplayed card. Hurrying what needs its time.',
    reversedTarget: 'anger',
    color: '#DCC0EC',
    pts: [[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311]],
    tier: 1, chapter: 1,
    tags: ['cleanse', 'abundance', 'nature'],
    synergies: ['emperor', 'star'],
  },

  // ── IV · The Emperor ─────────────────────────────────────────
  {
    id: 'emperor', name: 'The Emperor',
    layer: 'Superego', number: 4,
    keywords: 'structure · authority · law',
    type: 'compression',
    intensity: 7,
    attnShift: +7,
    effectDesc: 'Order is a form of love. Lock your current attention as a floor for 3 turns — trauma cannot push you below it. Deal 15 exhaustion damage. The foundation determines the ceiling.',
    targets: ['fear', 'fragmented', 'anger'],
    exhaustion: 15,
    reversedType: 'compression',
    reversedShift: -13,
    reversedDesc: "Structure that has become tyranny. Locks player's hand size at 2 for 2 turns — the container crushes what it was built to protect.",
    reversedTarget: 'anger',
    color: '#F0D080',
    pts: [[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.405,0.698],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tier: 1, chapter: 1,
    tags: ['floor', 'structure', 'authority'],
    synergies: ['empress', 'justice'],
  },

  // ── V · The Hierophant ───────────────────────────────────────
  {
    id: 'hierophant', name: 'The Hierophant',
    layer: 'Superego', number: 5,
    keywords: 'tradition · initiation · lineage',
    type: 'decompression',
    intensity: 3,
    attnShift: +7,
    effectDesc: 'The teaching has been here longer than you. Fire all available synergies this turn without needing both cards on field. Receive before you revise.',
    targets: ['disinterest', 'sadness'],
    exhaustion: 7,
    reversedType: 'compression',
    reversedShift: -10,
    reversedDesc: "Tradition weaponised to prevent evolution. Player cannot play cards of type 'both' next turn. Inherited knowledge confused with permanent truth.",
    reversedTarget: 'disinterest',
    color: '#A0D0FF',
    pts: [[0.5,0.05],[0.825,0.175],[0.949,0.602],[0.7,0.914],[0.3,0.914],[0.051,0.602],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5]],
    tier: 1, chapter: 1,
    tags: ['synergy-boost', 'tradition', 'lineage'],
    synergies: ['emperor', 'judgement'],
  },

  // ── VI · The Lovers ──────────────────────────────────────────
  {
    id: 'lovers', name: 'The Lovers',
    layer: 'Superego', number: 6,
    keywords: 'union · sacred choice · alignment',
    type: 'both',
    intensity: 5,
    attnShift: +11,
    effectDesc: 'The highest law: what do you truly choose? Select one card from your hand to play free this turn. The choice made from truth always exceeds the choice made from expectation.',
    targets: ['disinterest', 'sadness', 'fear'],
    exhaustion: 10,
    reversedType: 'compression',
    reversedShift: -12,
    reversedDesc: "Choosing from obligation rather than truth. Fabricates a false card in player's hand — it looks like their highest-shift card but drains 8 attention when played.",
    reversedTarget: 'disinterest',
    color: '#F4A0C8',
    pts: [[0.5,0.05],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.817,0.725],[0.596,0.698],[0.5,0.95],[0.405,0.698],[0.16,0.747],[0.28,0.5],[0.136,0.29]],
    tier: 1, chapter: 1,
    tags: ['free-play', 'choice', 'alignment'],
    synergies: ['chariot', 'star'],
  },

  // ── VII · The Chariot ────────────────────────────────────────
  {
    id: 'chariot', name: 'The Chariot',
    layer: 'Superego', number: 7,
    keywords: 'will · victory · discipline',
    type: 'compression',
    intensity: 8,
    attnShift: +8,
    effectDesc: 'Opposing forces harnessed become the vehicle. Deal 20 exhaustion damage. Your next card costs 0 intensity this turn — the momentum carries forward.',
    targets: ['anger', 'fear', 'fragmented'],
    exhaustion: 20,
    reversedType: 'compression',
    reversedShift: -14,
    reversedDesc: "The opposing forces are winning. Player's decompression cards each cost +3 attention shift to play next turn. Discipline lost to chaos.",
    reversedTarget: 'anger',
    color: '#C8A860',
    pts: [[0.7,0.086],[0.861,0.383],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.139,0.383],[0.3,0.086],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tier: 1, chapter: 1,
    tags: ['exhaust', 'momentum', 'will'],
    synergies: ['emperor', 'strength'],
  },

  // ── VIII · Strength ──────────────────────────────────────────
  {
    id: 'strength', name: 'Strength',
    layer: 'Superego', number: 8,
    keywords: 'courage · gentleness · mastery',
    type: 'decompression',
    intensity: 5,
    attnShift: +9,
    effectDesc: "The lion held by love. Reduce the next trauma card's intensity to 1. Draw +1 card. What wildness waits to be approached with patience instead of control?",
    targets: ['anger', 'fear', 'fragmented'],
    exhaustion: 9,
    reversedType: 'compression',
    reversedShift: -11,
    reversedDesc: "Force where gentleness would work. Trauma deals +4 extra shift on its next card. The lion is being fought rather than befriended.",
    reversedTarget: 'anger',
    color: '#FF9060',
    pts: [[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.691,0.61],[0.72,0.5],[0.596,0.698],[0.5,0.72],[0.387,0.613],[0.764,0.618]],
    tier: 1, chapter: 1,
    tags: ['reduce', 'draw', 'gentleness'],
    synergies: ['magician', 'chariot'],
  },

  // ── IX · The Hermit ──────────────────────────────────────────
  {
    id: 'hermit', name: 'The Hermit',
    layer: 'Superego', number: 9,
    keywords: 'solitude · lantern · inner light',
    type: 'decompression',
    intensity: 3,
    attnShift: +7,
    effectDesc: "The answer lives at the end of your own corridor. Reveal the full trauma hand for 2 turns. Skip trauma's next turn — solitude disarms what exposure alone cannot reach.",
    targets: ['fragmented', 'fear', 'disinterest'],
    exhaustion: 8,
    reversedType: 'compression',
    reversedShift: -9,
    reversedDesc: 'Solitude as protection rather than preparation. Player draws 1 fewer card for 2 turns. The lantern lit but the corridor refused.',
    reversedTarget: 'sadness',
    color: '#C8D080',
    pts: [[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61]],
    tier: 1, chapter: 1,
    tags: ['reveal', 'skip', 'solitude'],
    synergies: ['high_priestess', 'moon'],
  },

  // ── X · Wheel of Fortune ─────────────────────────────────────
  {
    id: 'wheel_of_fortune', name: 'Wheel of Fortune',
    layer: 'Superego', number: 10,
    keywords: 'fate · cycles · turning',
    type: 'both',
    intensity: 6,
    attnShift: +10,
    effectDesc: 'The wheel turns without your permission. Shuffle your discard back into your deck and draw 2 fresh cards. Every rise contains the seed of the next — ride it rather than resist.',
    targets: ['sadness', 'disinterest', 'fragmented'],
    exhaustion: 10,
    reversedType: 'compression',
    reversedShift: -12,
    reversedDesc: "Resistance to the turn. Shuffles trauma's discard back into their deck — every exhausted card returns. Fighting fate with energy that could have ridden it.",
    reversedTarget: 'disinterest',
    color: '#A060D0',
    pts: [[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.342,0.311]],
    tier: 1, chapter: 1,
    tags: ['shuffle', 'cycle', 'fate'],
    synergies: ['fool', 'world'],
  },

  // ── XI · Justice ─────────────────────────────────────────────
  {
    id: 'justice', name: 'Justice',
    layer: 'Superego', number: 11,
    keywords: 'truth · law · consequence',
    type: 'compression',
    intensity: 7,
    attnShift: +8,
    effectDesc: "The higher law is exacting. Deal exhaustion damage equal to trauma's last attnShift (mirrored back). What was sent returns with precision.",
    targets: ['fragmented', 'disinterest', 'anger'],
    exhaustion: 18,
    reversedType: 'compression',
    reversedShift: -13,
    reversedDesc: "Avoiding the scales. Player's compression cards deal 0 exhaustion damage next turn. Knowing the verdict and refusing to stand on the platform.",
    reversedTarget: 'disinterest',
    color: '#70C0A0',
    pts: [[0.5,0.05],[0.825,0.175],[0.949,0.602],[0.7,0.914],[0.3,0.914],[0.051,0.602],[0.136,0.29],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tier: 1, chapter: 1,
    tags: ['mirror-damage', 'truth', 'law'],
    synergies: ['emperor', 'tower'],
  },

  // ── XII · The Hanged Man ─────────────────────────────────────
  {
    id: 'hanged_man', name: 'The Hanged Man',
    layer: 'Superego', number: 12,
    keywords: 'surrender · suspension · vision',
    type: 'decompression',
    intensity: 2,
    attnShift: +12,
    effectDesc: 'The view from upside down reveals what standing upright cannot. Both players skip next turn. What is processed below strategy will surface when ready.',
    targets: ['anger', 'fragmented', 'fear'],
    exhaustion: 5,
    reversedType: 'compression',
    reversedShift: -10,
    reversedDesc: 'The surrender refused — the ego fights it as punishment. Your decompression cards cost double their intensity value this turn.',
    reversedTarget: 'anger',
    color: '#60B0D8',
    pts: [[0.5,0.95],[0.136,0.29],[0.871,0.29],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tier: 1, chapter: 2,
    tags: ['mutual-skip', 'surrender', 'vision'],
    synergies: ['hermit', 'moon'],
  },

  // ── XIII · Death ─────────────────────────────────────────────
  {
    id: 'death', name: 'Death',
    layer: 'Superego', number: 13,
    keywords: 'transformation · ending · crossing',
    type: 'both',
    intensity: 8,
    attnShift: +10,
    effectDesc: 'Nothing meant to continue will end here. Remove all corruption cards permanently. Deal 12 exhaustion damage. The old form dissolves to make room for what is emerging.',
    targets: ['fragmented', 'sadness', 'disinterest'],
    exhaustion: 22,
    reversedType: 'compression',
    reversedShift: -15,
    reversedDesc: 'Clinging to what Death has already claimed. Player cannot remove corruption cards for 3 turns. The transformation complete but the old form kept on life support.',
    reversedTarget: 'sadness',
    color: '#808090',
    pts: [[0.5,0.05],[0.584,0.297],[0.825,0.175],[0.72,0.5],[0.949,0.602],[0.613,0.613],[0.7,0.914],[0.5,0.72],[0.3,0.914],[0.387,0.613],[0.051,0.602],[0.28,0.5],[0.136,0.29],[0.416,0.297]],
    tier: 1, chapter: 2,
    tags: ['cleanse-all', 'exhaust', 'transformation'],
    synergies: ['tower', 'judgement'],
  },

  // ── XIV · Temperance ─────────────────────────────────────────
  {
    id: 'temperance', name: 'Temperance',
    layer: 'Superego', number: 14,
    keywords: 'alchemy · patience · blending',
    type: 'decompression',
    intensity: 4,
    attnShift: +9,
    effectDesc: 'Pour between cups until the two become one. Cancel the negative effects of 1 trauma card already played this turn. The opposites you hold are ingredients, not obstacles.',
    targets: ['anger', 'sadness', 'fear'],
    exhaustion: 9,
    reversedType: 'compression',
    reversedShift: -10,
    reversedDesc: 'Impatience with the alchemical process. Player must play all held cards next turn with no option to pass. Demanding gold before the pouring is complete.',
    reversedTarget: 'anger',
    color: '#80C8C0',
    pts: [[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383],[0.274,0.274],[0.342,0.311],[0.5,0.27],[0.613,0.387]],
    tier: 1, chapter: 1,
    tags: ['cancel', 'alchemy', 'patience'],
    synergies: ['star', 'lovers'],
  },

  // ── XV · The Devil ───────────────────────────────────────────
  {
    id: 'devil', name: 'The Devil',
    layer: 'Superego', number: 15,
    keywords: 'shadow · bondage · liberation',
    type: 'compression',
    intensity: 9,
    attnShift: +7,
    effectDesc: 'The chains are yours to remove. Deal 25 exhaustion damage. Destroy all fabrication cards in your hand — the Devil bows when named. What binding has your consent?',
    targets: ['fragmented', 'disinterest', 'fear'],
    exhaustion: 25,
    reversedType: 'compression',
    reversedShift: -14,
    reversedDesc: "Seeing the chains but not moving. Player's highest attention-shift card in hand is disabled this turn. The liberation closer than it appears.",
    reversedTarget: 'fragmented',
    color: '#C06060',
    pts: [[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tier: 1, chapter: 2,
    tags: ['exhaust', 'destroy-fabrications', 'shadow'],
    synergies: ['tower', 'death'],
  },

  // ── XVI · The Tower ──────────────────────────────────────────
  {
    id: 'tower', name: 'The Tower',
    layer: 'Superego', number: 16,
    keywords: 'revelation · collapse · liberation',
    type: 'compression',
    intensity: 10,
    attnShift: +6,
    effectDesc: 'The lightning strikes what was never true. Deal 30 exhaustion damage. What remains after the clearing is what was real all along — the false structure must fall.',
    targets: ['fragmented', 'disinterest', 'anger'],
    exhaustion: 30,
    reversedType: 'compression',
    reversedShift: -18,
    reversedDesc: "Refusing the lightning. The Tower compromised but occupants won't leave. Player cannot play compression cards next turn. What is false maintained at cost.",
    reversedTarget: 'fragmented',
    color: '#E06040',
    pts: [[0.5,0.05],[0.613,0.387],[0.95,0.5],[0.613,0.613],[0.5,0.95],[0.387,0.613],[0.05,0.5],[0.387,0.387],[0.5,0.5],[0.382,0.5],[0.5,0.382]],
    tier: 1, chapter: 2,
    tags: ['max-exhaust', 'revelation', 'collapse'],
    synergies: ['justice', 'death'],
  },

  // ── XVII · The Star ──────────────────────────────────────────
  {
    id: 'star', name: 'The Star',
    layer: 'Superego', number: 17,
    keywords: 'hope · renewal · guidance',
    type: 'decompression',
    intensity: 3,
    attnShift: +13,
    effectDesc: 'The wound becomes the opening. Restore 3 cards from your discard to your deck. Hold attention floor at Witness for 2 turns — light enters from the direction of the scar.',
    targets: ['sadness', 'fragmented', 'disinterest'],
    exhaustion: 6,
    reversedType: 'compression',
    reversedShift: -11,
    reversedDesc: "Hope curdled into despair. Player gains 0 attention from their next decompression card. The star in the sky but the figure has stopped looking up.",
    reversedTarget: 'sadness',
    color: '#A0C0FF',
    pts: [[0.5,0.05],[0.825,0.175],[0.949,0.398],[0.817,0.725],[0.609,0.906],[0.391,0.906],[0.16,0.747],[0.051,0.398],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.405,0.698],[0.309,0.61],[0.28,0.5],[0.342,0.311]],
    tier: 1, chapter: 2,
    tags: ['restore', 'floor', 'hope'],
    synergies: ['empress', 'temperance'],
  },

  // ── XVIII · The Moon ─────────────────────────────────────────
  {
    id: 'moon', name: 'The Moon',
    layer: 'Superego', number: 18,
    keywords: 'illusion · mystery · cycles',
    type: 'decompression',
    intensity: 4,
    attnShift: +8,
    effectDesc: 'Not everything in the dark is threat. Reveal all trauma cards for 3 turns. Navigate by feeling — the Moon lights enough for the next step.',
    targets: ['fear', 'fragmented', 'disinterest'],
    exhaustion: 9,
    reversedType: 'compression',
    reversedShift: -13,
    reversedDesc: 'Anxiety colonises reason. Player must spend 3 attention to play any card next turn — the price of false fear.',
    reversedTarget: 'fear',
    color: '#8080C8',
    pts: [[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311],[0.72,0.5]],
    tier: 1, chapter: 2,
    tags: ['reveal-sustained', 'illusion', 'cycles'],
    synergies: ['high_priestess', 'hermit'],
  },

  // ── XIX · The Sun ────────────────────────────────────────────
  {
    id: 'sun', name: 'The Sun',
    layer: 'Superego', number: 19,
    keywords: 'joy · consciousness · radiance',
    type: 'decompression',
    intensity: 2,
    attnShift: +16,
    effectDesc: 'This is not earned — it simply is. Shine without apology. All synergy effects fire twice this turn. Attention cannot drop below Presence for 2 turns.',
    targets: ['sadness', 'disinterest', 'fragmented'],
    exhaustion: 5,
    reversedType: 'compression',
    reversedShift: -10,
    reversedDesc: 'Joy available but refused. All positive attention shifts from player cards reduced by 5 next turn. The sun present, its warmth withheld.',
    reversedTarget: 'disinterest',
    color: '#FFD060',
    pts: [[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.949,0.602],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.051,0.602],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.755,0.382],[0.691,0.61],[0.596,0.698],[0.5,0.72],[0.405,0.698],[0.309,0.61],[0.236,0.382],[0.342,0.311]],
    tier: 1, chapter: 3,
    tags: ['synergy-double', 'floor', 'radiance'],
    synergies: ['star', 'world'],
  },

  // ── XX · Judgement ───────────────────────────────────────────
  {
    id: 'judgement', name: 'Judgement',
    layer: 'Superego', number: 20,
    keywords: 'calling · awakening · rebirth',
    type: 'compression',
    intensity: 8,
    attnShift: +10,
    effectDesc: 'The trumpet has sounded. Something calls you by your true name. Deal 20 exhaustion damage. Instantly remove all status debuffs from your attention state.',
    targets: ['fragmented', 'fear', 'disinterest'],
    exhaustion: 20,
    reversedType: 'compression',
    reversedShift: -16,
    reversedDesc: "The trumpet heard, the choice made not to rise. Player's next 2 cards deal 0 exhaustion damage. The coffin lid held shut from the inside.",
    reversedTarget: 'sadness',
    color: '#E0C080',
    pts: [[0.5,0.5],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.416,0.297],[0.584,0.297],[0.95,0.5],[0.7,0.914],[0.3,0.914],[0.05,0.5],[0.3,0.086],[0.7,0.086],[0.5,0.382]],
    tier: 1, chapter: 3,
    tags: ['debuff-clear', 'calling', 'rebirth'],
    synergies: ['death', 'hierophant'],
  },

  // ── XXI · The World ──────────────────────────────────────────
  {
    id: 'world', name: 'The World',
    layer: 'Superego', number: 21,
    keywords: 'completion · wholeness · arrival',
    type: 'decompression',
    intensity: 1,
    attnShift: +20,
    effectDesc: 'The journey arrives where it began and the beginning is completely transformed by the arriving. You are whole — not almost, not earning it, whole now as you are. Win condition triggered.',
    targets: ['disinterest', 'sadness', 'fragmented'],
    exhaustion: 3,
    reversedType: 'compression',
    reversedShift: -15,
    reversedDesc: "Completion withheld from itself. Player's current attention reads 15 lower than actual for 3 turns — the full circle treated as insufficient.",
    reversedTarget: 'disinterest',
    color: '#FFFFFF',
    pts: [[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5],[0.665,0.335],[0.691,0.61],[0.309,0.61],[0.342,0.311],[0.596,0.698],[0.405,0.698],[0.387,0.387],[0.613,0.387]],
    tier: 1, chapter: 3,
    tags: ['win-check', 'wholeness', 'completion'],
    synergies: ['fool', 'wheel_of_fortune'],
  },
];

// ═══════════════════════════════════════════════════════════════
// EGO — 16 Court Cards  (Shop upgrades between chapters)
// ═══════════════════════════════════════════════════════════════
const EGO_CARDS = [
  // ── CUPS COURT ───────────────────────────────────────────────
  {
    id: 'page_cups', name: 'Page of Cups', layer: 'Ego', suit: 'cups',
    keywords: 'wonder · openness · feeling',
    type: 'decompression', intensity: 2, attnShift: +8,
    effectDesc: "Beginner's mind. Draw 2 extra cards this turn — let impressions arrive before naming them.",
    targets: ['disinterest', 'sadness'], exhaustion: 5,
    reversedShift: -8, reversedDesc: "Over-receptivity. Player's next draw is randomised across the full deck.",
    color: '#7EB8E8',
    pts: [[0.5,0.27],[0.764,0.618],[0.236,0.618],[0.5,0.382],[0.613,0.387],[0.613,0.613],[0.5,0.618],[0.387,0.613],[0.387,0.387]],
    tier: 2, chapter: 1, shopChapter: 1, tags: ['draw', 'openness'],
  },
  {
    id: 'knight_cups', name: 'Knight of Cups', layer: 'Ego', suit: 'cups',
    keywords: 'romance · pursuit · idealism',
    type: 'both', intensity: 5, attnShift: +10,
    effectDesc: 'Ride toward beauty. When played alongside a decompression card, both shifts gain +3.',
    targets: ['sadness', 'disinterest'], exhaustion: 10,
    reversedShift: -11, reversedDesc: "Chasing an ideal with no person inside it. Player's highest-value card appears doubled but deals 0.",
    color: '#60A0D8',
    pts: [[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.5,0.5],[0.765,0.563],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.235,0.563],[0.5,0.383],[0.5,0.95]],
    tier: 2, chapter: 1, shopChapter: 1, tags: ['amplify', 'idealism'],
  },
  {
    id: 'queen_cups', name: 'Queen of Cups', layer: 'Ego', suit: 'cups',
    keywords: 'empathy · intuition · depth',
    type: 'decompression', intensity: 4, attnShift: +9,
    effectDesc: 'The self holds space. Cancel all status debuffs affecting your attention this turn.',
    targets: ['fear', 'sadness', 'anger'], exhaustion: 9,
    reversedShift: -12, reversedDesc: "Empathy become self-erasure. Player's decompression cards cost +2 exhaustion for 2 turns.",
    color: '#80A0D8',
    pts: [[0.502,0.17],[0.755,0.382],[0.817,0.725],[0.5,0.83],[0.236,0.618],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['debuff-clear', 'empathy'],
  },
  {
    id: 'king_cups', name: 'King of Cups', layer: 'Ego', suit: 'cups',
    keywords: 'mastery · balance · diplomacy',
    type: 'both', intensity: 7, attnShift: +11,
    effectDesc: 'Emotional authority. All synergy effects this turn deal double their stated exhaustion damage.',
    targets: ['anger', 'fear', 'disinterest'], exhaustion: 14,
    reversedShift: -13, reversedDesc: "Feelings so managed none are felt. Synergies deal 0 exhaustion damage next turn.",
    color: '#4080C8',
    pts: [[0.5,0.5],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.416,0.297],[0.584,0.297],[0.95,0.5],[0.7,0.914],[0.3,0.914],[0.05,0.5],[0.3,0.086],[0.7,0.086],[0.5,0.382],[0.618,0.5]],
    tier: 2, chapter: 3, shopChapter: 3, tags: ['synergy-exhaust-double', 'authority'],
  },
  // ── SWORDS COURT ─────────────────────────────────────────────
  {
    id: 'page_swords', name: 'Page of Swords', layer: 'Ego', suit: 'swords',
    keywords: 'curiosity · ideas · restless',
    type: 'decompression', intensity: 3, attnShift: +7,
    effectDesc: 'Reveal the next 2 trauma cards AND reduce each by 1 intensity. Questions in good faith protect.',
    targets: ['fear', 'fragmented'], exhaustion: 6,
    reversedShift: -9, reversedDesc: "Curiosity become cynicism. Player cannot use reveal effects from any card next turn.",
    color: '#C0C8A0',
    pts: [[0.502,0.17],[0.82,0.5],[0.5,0.83],[0.18,0.5],[0.5,0.382],[0.618,0.5],[0.613,0.613],[0.387,0.613],[0.382,0.5]],
    tier: 2, chapter: 1, shopChapter: 1, tags: ['reveal', 'curiosity'],
  },
  {
    id: 'knight_swords', name: 'Knight of Swords', layer: 'Ego', suit: 'swords',
    keywords: 'speed · ambition · reckless',
    type: 'compression', intensity: 8, attnShift: +7,
    effectDesc: "Play two cards this turn — but the second costs double its exhaustion. Think once before the blade lands.",
    targets: ['anger', 'disinterest'], exhaustion: 16,
    reversedShift: -14, reversedDesc: "Speed outpaced judgment. Next 2 compression cards deal 0 exhaustion damage.",
    color: '#C0C0B0',
    pts: [[0.502,0.17],[0.613,0.387],[0.82,0.5],[0.613,0.613],[0.5,0.83],[0.387,0.613],[0.18,0.5],[0.387,0.387],[0.5,0.5],[0.618,0.5],[0.5,0.382]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['double-play', 'reckless'],
  },
  {
    id: 'queen_swords', name: 'Queen of Swords', layer: 'Ego', suit: 'swords',
    keywords: 'clarity · boundary · discernment',
    type: 'compression', intensity: 7, attnShift: +8,
    effectDesc: 'Destroy all fabrication cards in hand AND in the trauma deck permanently. Precision, not cruelty.',
    targets: ['fragmented', 'disinterest'], exhaustion: 18,
    reversedShift: -13, reversedDesc: "Clarity weaponised. Trauma's next 2 cards each gain +4 shift.",
    color: '#D0D0C8',
    pts: [[0.502,0.17],[0.726,0.274],[0.876,0.637],[0.715,0.83],[0.285,0.83],[0.11,0.548],[0.274,0.274],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['destroy-fabrications', 'clarity'],
  },
  {
    id: 'king_swords', name: 'King of Swords', layer: 'Ego', suit: 'swords',
    keywords: 'authority · logic · judgment',
    type: 'compression', intensity: 9, attnShift: +9,
    effectDesc: "Deal exhaustion damage equal to trauma's current coherence × 0.3 — their own weight against them.",
    targets: ['fragmented', 'disinterest', 'anger'], exhaustion: 22,
    reversedShift: -15, reversedDesc: "Judgment before all evidence. Synergies cannot fire this turn.",
    color: '#E0E0D8',
    pts: [[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297],[0.596,0.698],[0.342,0.311],[0.72,0.5],[0.405,0.698],[0.5,0.27]],
    tier: 2, chapter: 3, shopChapter: 3, tags: ['proportional-exhaust', 'judgment'],
  },
  // ── WANDS COURT ──────────────────────────────────────────────
  {
    id: 'page_wands', name: 'Page of Wands', layer: 'Ego', suit: 'wands',
    keywords: 'adventure · enthusiasm · spark',
    type: 'decompression', intensity: 3, attnShift: +9,
    effectDesc: 'Play this card free (no exhaustion cost). Draw 1 card. Follow the excitement before reason dims it.',
    targets: ['disinterest', 'sadness'], exhaustion: 0,
    reversedShift: -8, reversedDesc: "Excitement scattered. Player's next draw is randomised.",
    color: '#E08040',
    pts: [[0.871,0.29],[0.5,0.83],[0.136,0.29],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tier: 2, chapter: 1, shopChapter: 1, tags: ['free-play', 'draw', 'spark'],
  },
  {
    id: 'knight_wands', name: 'Knight of Wands', layer: 'Ego', suit: 'wands',
    keywords: 'passion · courage · momentum',
    type: 'both', intensity: 7, attnShift: +11,
    effectDesc: 'If played as your second card this turn, it deals double its stated attnShift. Channel the fire.',
    targets: ['anger', 'disinterest'], exhaustion: 14,
    reversedShift: -13, reversedDesc: "Exhausted but won't admit it. Next compression card deals −5 less exhaustion.",
    color: '#E06820',
    pts: [[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.05],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['double-second', 'momentum'],
  },
  {
    id: 'queen_wands', name: 'Queen of Wands', layer: 'Ego', suit: 'wands',
    keywords: 'charisma · will · presence',
    type: 'both', intensity: 6, attnShift: +12,
    effectDesc: "Natural authority. All cards played this turn gain +2 attnShift. The Queen's presence doesn't announce itself.",
    targets: ['disinterest', 'sadness', 'fear'], exhaustion: 12,
    reversedShift: -12, reversedDesc: "Presence become performance. All positive shifts this turn reduced by 4.",
    color: '#E04820',
    pts: [[0.502,0.17],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.817,0.725],[0.596,0.698],[0.5,0.83],[0.405,0.698],[0.16,0.747],[0.28,0.5],[0.136,0.29],[0.416,0.297],[0.5,0.5]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['all-boost', 'presence'],
  },
  {
    id: 'king_wands', name: 'King of Wands', layer: 'Ego', suit: 'wands',
    keywords: 'vision · leadership · legacy',
    type: 'compression', intensity: 9, attnShift: +10,
    effectDesc: 'Deal 24 exhaustion damage. For the next 3 turns, compression cards each gain +5 exhaustion damage.',
    targets: ['fragmented', 'disinterest', 'anger'], exhaustion: 24,
    reversedShift: -14, reversedDesc: "Building for legacy, not purpose. Decompression cards cost double this turn.",
    color: '#E03000',
    pts: [[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tier: 2, chapter: 3, shopChapter: 3, tags: ['sustained-exhaust', 'vision'],
  },
  // ── PENTACLES COURT ──────────────────────────────────────────
  {
    id: 'page_pents', name: 'Page of Pentacles', layer: 'Ego', suit: 'pentacles',
    keywords: 'study · diligence · beginning',
    type: 'decompression', intensity: 2, attnShift: +7,
    effectDesc: 'The long view, started carefully. Add 2 cards from your discard back into your deck. Everything compounds.',
    targets: ['disinterest', 'sadness'], exhaustion: 5,
    reversedShift: -7, reversedDesc: "Study stalled. Player's next draw returns itself at end of turn.",
    color: '#80C860',
    pts: [[0.05,0.236],[0.95,0.236],[0.95,0.764],[0.05,0.764],[0.391,0.199],[0.609,0.801],[0.5,0.5],[0.618,0.5],[0.391,0.801]],
    tier: 2, chapter: 1, shopChapter: 1, tags: ['restore', 'patience'],
  },
  {
    id: 'knight_pents', name: 'Knight of Pentacles', layer: 'Ego', suit: 'pentacles',
    keywords: 'routine · reliability · patience',
    type: 'decompression', intensity: 4, attnShift: +8,
    effectDesc: 'Slow and steady. Gains +2 attnShift for each turn it has been in your deck this run.',
    targets: ['disinterest', 'sadness'], exhaustion: 7,
    reversedShift: -9, reversedDesc: "Reliability become rigidity. Player cannot play more than 1 card this turn.",
    color: '#60A840',
    pts: [[0.726,0.274],[0.726,0.726],[0.274,0.726],[0.274,0.274],[0.5,0.27],[0.613,0.387],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['scaling', 'reliability'],
  },
  {
    id: 'queen_pents', name: 'Queen of Pentacles', layer: 'Ego', suit: 'pentacles',
    keywords: 'nurturing · security · care',
    type: 'decompression', intensity: 5, attnShift: +10,
    effectDesc: 'Remove 1 corruption card AND deal 10 exhaustion damage. Care and strength are not opposites.',
    targets: ['sadness', 'disinterest'], exhaustion: 10,
    reversedShift: -12, reversedDesc: "Nurturing become martyrdom. Player loses 3 additional attention when trauma heals itself.",
    color: '#80B840',
    pts: [[0.73,0.18],[0.95,0.5],[0.715,0.83],[0.285,0.83],[0.11,0.548],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]],
    tier: 2, chapter: 2, shopChapter: 2, tags: ['cleanse', 'exhaust', 'nurturing'],
  },
  {
    id: 'king_pents', name: 'King of Pentacles', layer: 'Ego', suit: 'pentacles',
    keywords: 'mastery · wealth · discipline',
    type: 'compression', intensity: 8, attnShift: +9,
    effectDesc: 'Deal 20 exhaustion damage. All cards gain a permanent +1 attnShift modifier for the rest of this chapter.',
    targets: ['fragmented', 'disinterest'], exhaustion: 20,
    reversedShift: -14, reversedDesc: "Mastery become cage. Player's attnShift values capped at +8 for 3 turns.",
    color: '#60A820',
    pts: [[0.5,0.5],[0.5,0.382],[0.562,0.44],[0.618,0.5],[0.584,0.584],[0.5,0.618],[0.416,0.584],[0.382,0.5],[0.416,0.416],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.691,0.61],[0.5,0.72],[0.28,0.5]],
    tier: 2, chapter: 3, shopChapter: 3, tags: ['permanent-boost', 'mastery'],
  },
];

// ═══════════════════════════════════════════════════════════════
// ID — 40 Pip Cards
// The raw drive — instinct, desire, grief, hunger.
// Trauma plays these REVERSED (compression).
// On loss/corruption, they enter the player's deck reversed.
// ═══════════════════════════════════════════════════════════════
const ID_CARDS = [

  // ── CUPS ─────────────────────────────────────────────────────
  { id:'ace_cups',    name:'Ace of Cups',    suit:'cups', layer:'ID', keywords:'love · awakening · grace',
    upright:'The cup overflows before you even raise it. Do not manage it into something smaller.',
    reversed:'The feeling offered but blocked. An emotional beginning that has not landed.',
    traumaShift:-9,  traumaTarget:'disinterest', traumaHealing:10,
    traumaDesc:'False grace injection. Trauma heals 10 coherence — feeding the approval loop.',
    corruptShift:-7, corruptDesc:'Blocked cup. Costs 3 attention to play. Stuck at the threshold.',
    tags:['healing','overflow'], color:'#4080C8',
    pts:[[0.502,0.17],[0.817,0.725],[0.16,0.747],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5]] },

  { id:'two_cups',    name:'Two of Cups',    suit:'cups', layer:'ID', keywords:'connection · recognition · bond',
    upright:'The ID simply knows — this is the feeling of seeing and being seen.',
    reversed:'A connection existing on the surface. What genuine exchange is avoided?',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:8,
    traumaDesc:'Surface connection as substitute for real recognition. Heals trauma 8.',
    corruptShift:-6, corruptDesc:'The hollow mirror. Shows what you want to see.',
    tags:['connection','recognition'], color:'#5090C8',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.405,0.698],[0.274,0.726],[0.124,0.637],[0.11,0.548],[0.139,0.383]] },

  { id:'three_cups',  name:'Three of Cups',  suit:'cups', layer:'ID', keywords:'celebration · reunion · warmth',
    upright:'The body remembers joy. Gather your people and let them witness what has become true.',
    reversed:'Celebration papering over something unresolved. What is not being said?',
    traumaShift:-6,  traumaTarget:'disinterest', traumaHealing:12,
    traumaDesc:'Hollow celebration. Trauma heals 12 — warmth without nourishment.',
    corruptShift:-5, corruptDesc:'Performance of warmth. Reduces draw by 1 while in hand.',
    tags:['healing','warmth'], color:'#60A0C8',
    pts:[[0.755,0.382],[0.5,0.83],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]] },

  { id:'four_cups',   name:'Four of Cups',   suit:'cups', layer:'ID', keywords:'apathy · discontent · withdrawal',
    upright:'Soul-exhaustion that precedes reassessment. The withdrawal is necessary, not permanent.',
    reversed:'Withdrawal gone too long, now avoidance. Something refused that deserves engagement.',
    traumaShift:-10, traumaTarget:'disinterest', traumaHealing:6,
    traumaDesc:'Apathy injection. Synergies blocked next player turn. Trauma heals 6.',
    corruptShift:-8, corruptDesc:'Forced apathy. When drawn, must be played first. Synergies blocked.',
    tags:['synergy-block','apathy'], color:'#405080',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.72,0.5],[0.691,0.61],[0.596,0.698],[0.5,0.382]],
    special:{ forcedPlay:true, synergiesBlocked:true } },

  { id:'five_cups',   name:'Five of Cups',   suit:'cups', layer:'ID', keywords:'grief · loss · what remains',
    upright:'Three cups spilled, two remain. The body knows the difference between loss and ruin.',
    reversed:'Stuck at the spilled cups — grief curdled into grievance.',
    traumaShift:-12, traumaTarget:'sadness', traumaHealing:0,
    traumaDesc:"Grief weaponised. Player's attnShift values reduced by 4 this turn.",
    corruptShift:-9, corruptDesc:'Grievance loop. Each turn, drifts −1 toward sadness automatically.',
    tags:['grief','slow'], color:'#305070',
    pts:[[0.502,0.17],[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.27],[0.72,0.5],[0.613,0.613],[0.387,0.613],[0.28,0.5]] },

  { id:'six_cups',    name:'Six of Cups',    suit:'cups', layer:'ID', keywords:'nostalgia · memory · innocence',
    upright:'A sweetness you thought outgrown is available again.',
    reversed:'Nostalgia as escape — the past idealised in proportion to present discomfort.',
    traumaShift:-9,  traumaTarget:'disinterest', traumaHealing:7,
    traumaDesc:'Memory as trap. Heals trauma 7. Newest card in hand disabled.',
    corruptShift:-6, corruptDesc:'Sweetness without nourishment. Cannot play any card drawn this turn.',
    tags:['nostalgia','disable-new'], color:'#507090',
    pts:[[0.502,0.17],[0.755,0.382],[0.817,0.725],[0.5,0.83],[0.236,0.618],[0.236,0.382],[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]] },

  { id:'seven_cups',  name:'Seven of Cups',  suit:'cups', layer:'ID', keywords:'desire · illusion · hunger',
    upright:'Every door looks golden. Choose one cup and drink from it before the clouds take them all.',
    reversed:'Illusions collapsing and clarity arriving. The fog lifts. This is useful.',
    traumaShift:-11, traumaTarget:'fragmented', traumaHealing:5,
    traumaDesc:'Fabricates 1 false card in player hand — looks like best card but deals −8. Discard costs 3.',
    corruptShift:-10, corruptDesc:'Hunger loop. When drawn, must discard the next card drawn.',
    tags:['fabricate','scatter'], color:'#604080',
    pts:[[0.502,0.17],[0.613,0.387],[0.861,0.383],[0.618,0.5],[0.715,0.83],[0.5,0.618],[0.285,0.83],[0.382,0.5],[0.139,0.383],[0.387,0.387]] },

  { id:'eight_cups',  name:'Eight of Cups',  suit:'cups', layer:'ID', keywords:'departure · meaning · move on',
    upright:'The body knows when a chapter has closed. Walk toward the mountains.',
    reversed:'Leaving before time, for the wrong reasons. Unfinished business abandoned as progress.',
    traumaShift:-10, traumaTarget:'sadness', traumaHealing:5,
    traumaDesc:"Heals trauma 5. Player's highest-value card temporarily discarded (returns next turn).",
    corruptShift:-8, corruptDesc:'Removes itself permanently but removes the next drawn card too.',
    tags:['displacement','departure'], color:'#304060',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.613,0.387],[0.691,0.61],[0.72,0.5],[0.596,0.698],[0.5,0.72],[0.387,0.613],[0.382,0.5]] },

  { id:'nine_cups',   name:'Nine of Cups',   suit:'cups', layer:'ID', keywords:'wish · satisfaction · pleasure',
    upright:'The body got what it really wanted. You are allowed to feel this without asking what comes next.',
    reversed:'Satisfaction depending on external conditions — brittle, blocked by comparison.',
    traumaShift:-7,  traumaTarget:'disinterest', traumaHealing:15,
    traumaDesc:'False satisfaction. Trauma heals 15. Player attention reads +10 higher than actual for 2 turns.',
    corruptShift:-6, corruptDesc:'Comfortable lie. Attention appears 8 higher while in hand.',
    tags:['euphoria','illusion','healing'], color:'#7090C8',
    pts:[[0.5,0.05],[0.73,0.18],[0.861,0.383],[0.817,0.725],[0.609,0.906],[0.391,0.906],[0.16,0.747],[0.139,0.383],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5]] },

  { id:'ten_cups',    name:'Ten of Cups',    suit:'cups', layer:'ID', keywords:'bliss · family · home',
    upright:'The nervous system finally rests. Home is sometimes a version of yourself you stopped fighting.',
    reversed:'Home exists but truth inside it is smoothed over to maintain appearances.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:18,
    traumaDesc:'Deepest healing loop. Trauma heals 18. Player synergies blocked next turn.',
    corruptShift:-7, corruptDesc:'While in hand, status strip shows Witness regardless of true state.',
    tags:['healing','synergy-block'], color:'#8090C8',
    pts:[[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.5,0.27],[0.236,0.382],[0.755,0.382],[0.5,0.5],[0.236,0.618],[0.764,0.618],[0.5,0.72],[0.5,0.95],[0.5,0.382],[0.387,0.387],[0.613,0.387]] },

  // ── SWORDS ────────────────────────────────────────────────────
  { id:'ace_swords',   name:'Ace of Swords',   suit:'swords', layer:'ID', keywords:'clarity · cut · breakthrough',
    upright:'The ID sharpens into a single clean point. The fog breaks. Use the blade with intention.',
    reversed:'Clarity without courage. Something true seen and immediately talked out of.',
    traumaShift:-10, traumaTarget:'fragmented', traumaHealing:0,
    traumaDesc:'Cuts through defences — shield effects bypassed this turn. Fragmentation strike.',
    corruptShift:-8, corruptDesc:'When drawn, must be played immediately. The impulse cannot be held.',
    tags:['shield-pierce','cut'], color:'#C0C8D0',
    pts:[[0.5,0.5],[0.618,0.5],[0.5,0.618],[0.382,0.5],[0.5,0.382],[0.387,0.387],[0.387,0.613]] },

  { id:'two_swords',   name:'Two of Swords',   suit:'swords', layer:'ID', keywords:'stalemate · avoidance · blindfold',
    upright:'The blindfold is self-imposed. The stalemate comfortable only vs the discomfort of deciding.',
    reversed:'Stalemate broken but moving in the wrong direction.',
    traumaShift:-11, traumaTarget:'disinterest', traumaHealing:0,
    traumaDesc:'Locks player attention at current position for 2 turns. No upward movement.',
    corruptShift:-9, corruptDesc:'While in deck, attention cannot rise above its value when card entered.',
    tags:['lock','stalemate'], color:'#A0A8B0',
    pts:[[0.726,0.274],[0.726,0.726],[0.274,0.726],[0.274,0.274],[0.5,0.5],[0.382,0.5],[0.5,0.382],[0.5,0.618]] },

  { id:'three_swords', name:'Three of Swords', suit:'swords', layer:'ID', keywords:'heartbreak · grief · pain',
    upright:'The wound is real. Feel it and move through it, not around it. What heals is what was willing to bleed.',
    reversed:'Pain refused acknowledgment. Grief the body carries but will not let itself feel.',
    traumaShift:-14, traumaTarget:'sadness', traumaHealing:0,
    traumaDesc:'Cannot be blocked by shield — only absorbed by decompression cards. Unblockable grief.',
    corruptShift:-12, corruptDesc:'Each turn, drains 1 additional attention. The wound festers from being ignored.',
    tags:['unblockable','grief'], color:'#8090A0',
    pts:[[0.502,0.17],[0.613,0.387],[0.82,0.5],[0.613,0.613],[0.5,0.83],[0.387,0.613],[0.11,0.548],[0.387,0.387],[0.5,0.5]],
    special:{ unblockable:true } },

  { id:'four_swords',  name:'Four of Swords',  suit:'swords', layer:'ID', keywords:'rest · recovery · recuperation',
    upright:'The knight rests between battles. Stillness is the only strategy that matters right now.',
    reversed:'Rest that cannot come, or rest that has become avoidance.',
    traumaShift:-6,  traumaTarget:'sadness', traumaHealing:14,
    traumaDesc:'Heals trauma 14. Locks player into passing next turn — rest weaponised.',
    corruptShift:-7, corruptDesc:'When drawn, turn ends immediately. No further cards can be played.',
    tags:['healing','skip-player'], color:'#708090',
    pts:[[0.5,0.27],[0.613,0.387],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387],[0.5,0.382],[0.5,0.618]] },

  { id:'five_swords',  name:'Five of Swords',  suit:'swords', layer:'ID', keywords:'conflict · hollow victory',
    upright:'A win that costs trust deserves examination before the next sword is drawn.',
    reversed:'The hollow victory visible from the inside — you walked away with the swords.',
    traumaShift:-12, traumaTarget:'anger', traumaHealing:0,
    traumaDesc:'Anger injection. Compression cards deal −5 less exhaust damage next turn.',
    corruptShift:-10, corruptDesc:'Deals 8 exhaust to trauma but costs player 5 attention. Wins that cost.',
    tags:['anger-inject','hollow'], color:'#9090A0',
    pts:[[0.5,0.05],[0.613,0.387],[0.861,0.383],[0.691,0.61],[0.715,0.83],[0.5,0.72],[0.285,0.83],[0.309,0.61],[0.139,0.383],[0.387,0.387]] },

  { id:'six_swords',   name:'Six of Swords',   suit:'swords', layer:'ID', keywords:'transition · moving on · passage',
    upright:'You do not have to be done with the hard thing before you leave it. Just leave.',
    reversed:'Trying to leave but pulled back. Arriving somewhere new bringing the old trouble unchanged.',
    traumaShift:-9,  traumaTarget:'sadness', traumaHealing:6,
    traumaDesc:"Heals trauma 6. Shuffles player's most recently played card back into their deck.",
    corruptShift:-7, corruptDesc:'When played, adds itself back into deck — cannot permanently leave.',
    tags:['return-played','transition'], color:'#6080A0',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5]] },

  { id:'seven_swords', name:'Seven of Swords', suit:'swords', layer:'ID', keywords:'strategy · shadow · cunning',
    upright:'Something moves unseen. Trust the signal. Check your blind spots.',
    reversed:'The deception has come to light. What was taken in secret must be accounted for.',
    traumaShift:-11, traumaTarget:'disinterest', traumaHealing:0,
    traumaDesc:"Trauma's next card is not revealed even if reveal effects are active. Moves unseen.",
    corruptShift:-9, corruptDesc:'Shift value hidden — displayed as ±0 while in hand. Reveals on play.',
    tags:['hidden','cunning'], color:'#7080A0',
    pts:[[0.5,0.05],[0.584,0.297],[0.871,0.29],[0.72,0.5],[0.876,0.637],[0.613,0.613],[0.715,0.83],[0.5,0.72],[0.285,0.83],[0.387,0.613],[0.11,0.548],[0.28,0.5]] },

  { id:'eight_swords', name:'Eight of Swords', suit:'swords', layer:'ID', keywords:'paralysis · fear · trapped',
    upright:'The swords are not touching you. One step breaks the pattern. What was dangerous then may not be now.',
    reversed:'Beginning to see through the cage. The paralysis was largely self-constructed.',
    traumaShift:-13, traumaTarget:'fear', traumaHealing:0,
    traumaDesc:'Locks hand size at 2 for 2 turns. Movement appears impossible — paralysis.',
    corruptShift:-11, corruptDesc:'While in deck, hand size permanently reduced by 1. The cage travels with you.',
    tags:['hand-lock','paralysis'], color:'#5060A0',
    pts:[[0.5,0.05],[0.825,0.175],[0.95,0.5],[0.825,0.825],[0.5,0.95],[0.175,0.825],[0.05,0.5],[0.175,0.175],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]] },

  { id:'nine_swords',  name:'Nine of Swords',  suit:'swords', layer:'ID', keywords:'anxiety · nightmare · dread',
    upright:'The mind builds worst-case scenarios with impressive detail. They are not prophecy.',
    reversed:'Catastrophizing consuming itself — must be named as pattern, not prophecy.',
    traumaShift:-15, traumaTarget:'fear', traumaHealing:0,
    traumaDesc:"Maximum fear injection. Player's next decompression card pre-negated — anxiety tells you it won't work.",
    corruptShift:-12, corruptDesc:"Each time drawn, next drawn card's shift is halved. Dread arrives before tools.",
    tags:['max-fear','negate-next-decomp'], color:'#404090',
    pts:[[0.5,0.5],[0.5,0.382],[0.613,0.613],[0.382,0.5],[0.613,0.387],[0.5,0.618],[0.387,0.387],[0.618,0.5],[0.309,0.61],[0.584,0.297]] },

  { id:'ten_swords',   name:'Ten of Swords',   suit:'swords', layer:'ID', keywords:'endings · collapse · new dawn',
    upright:'This is the lowest point, and the lowest point is also a foundation.',
    reversed:'Refusing to acknowledge the ending. The collapse active, not complete.',
    traumaShift:-20, traumaTarget:'fragmented', traumaHealing:0,
    traumaDesc:'Maximum damage. Only fires when trauma coherence is above 50%. The ten swords land.',
    corruptShift:-14, corruptDesc:'When drawn, attention drops 4 immediately before card is played.',
    tags:['max-damage','conditional'], color:'#202040',
    pts:[[0.618,0.5],[0.5,0.5],[0.613,0.613],[0.5,0.618],[0.691,0.61],[0.596,0.698],[0.72,0.5],[0.5,0.72],[0.387,0.613],[0.405,0.698],[0.382,0.5],[0.309,0.61],[0.5,0.382]],
    special:{ condition:'traumaCoherence > 50' } },

  // ── WANDS ─────────────────────────────────────────────────────
  { id:'ace_wands',    name:'Ace of Wands',    suit:'wands', layer:'ID', keywords:'spark · fire · impulse',
    upright:'The drive ignites before thought arrives. Act on it while it is still this hot.',
    reversed:"A spark that won't ignite, or one extinguished too early.",
    traumaShift:-9,  traumaTarget:'anger', traumaHealing:0,
    traumaDesc:'Misdirected fire. Player must play a card this turn — cannot pass.',
    corruptShift:-7, corruptDesc:'When drawn, must be played immediately. The compulsion cannot be held.',
    tags:['force-play','impulse'], color:'#E06020',
    pts:[[0.5,0.05],[0.817,0.725],[0.16,0.747],[0.5,0.5],[0.382,0.5],[0.5,0.618],[0.5,0.382]] },

  { id:'two_wands',    name:'Two of Wands',    suit:'wands', layer:'ID', keywords:'planning · vision · decision',
    upright:'The ID has arrived somewhere real and looks at what comes next. Trust where the body leans.',
    reversed:'Plans that will not commit to direction. Ambivalence costing more than a decision would.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:5,
    traumaDesc:'Heals trauma 5. Forces player to begin next turn without drawing — must play from existing hand.',
    corruptShift:-6, corruptDesc:'Cannot play two cards of the same type in a single turn.',
    tags:['no-draw','ambivalence'], color:'#E05010',
    pts:[[0.05,0.236],[0.95,0.236],[0.95,0.764],[0.05,0.764],[0.391,0.199],[0.391,0.801],[0.5,0.5],[0.382,0.5],[0.609,0.199]] },

  { id:'three_wands',  name:'Three of Wands',  suit:'wands', layer:'ID', keywords:'ambition · horizon · momentum',
    upright:'The ships on the horizon are yours. The fire planned ahead and the evidence is coming in.',
    reversed:'The ships belong to someone else. What was set in motion has stalled.',
    traumaShift:-10, traumaTarget:'anger', traumaHealing:0,
    traumaDesc:"Comparison injection. Player's next attnShift reduced by 5.",
    corruptShift:-8, corruptDesc:"Each turn, highest-shift card in hand loses 1 point of shift. Horizon always receding.",
    tags:['reduce-shift','comparison'], color:'#D05010',
    pts:[[0.5,0.95],[0.136,0.29],[0.871,0.29],[0.5,0.27],[0.665,0.335],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]] },

  { id:'four_wands',   name:'Four of Wands',   suit:'wands', layer:'ID', keywords:'celebration · homecoming · harvest',
    upright:'Something completed — mark it. Milestones ritualised make effort meaningful.',
    reversed:'Celebration unearned or incomplete. The achievement is minimised.',
    traumaShift:-7,  traumaTarget:'disinterest', traumaHealing:10,
    traumaDesc:'Heals trauma 10. Synergies cost +2 exhaustion this turn.',
    corruptShift:-5, corruptDesc:'Hollow milestone. Attention cannot rise above Presence state this turn.',
    tags:['healing','synergy-tax'], color:'#C04010',
    pts:[[0.391,0.199],[0.609,0.199],[0.609,0.801],[0.391,0.801],[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29]] },

  { id:'five_wands',   name:'Five of Wands',   suit:'wands', layer:'ID', keywords:'competition · conflict · friction',
    upright:'The sparring is real but the stakes are lower than it feels. What is being sharpened by this?',
    reversed:'Competition that serves no one. Energy scattered across opponents who are also yourself.',
    traumaShift:-11, traumaTarget:'anger', traumaHealing:0,
    traumaDesc:'Chaos injection. Player must play their lowest-shift card next turn — no card selection.',
    corruptShift:-9, corruptDesc:'Forces a random card from hand to play first each turn. Competition for control.',
    tags:['chaos','force-low'], color:'#B03008',
    pts:[[0.5,0.05],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.27,0.18],[0.5,0.27],[0.5,0.72]] },

  { id:'six_wands',    name:'Six of Wands',    suit:'wands', layer:'ID', keywords:'victory · recognition · return',
    upright:'The procession is real. You earned this and the body is allowed to know it.',
    reversed:'Victory that must be performed. The recognition conditional, the return observed rather than felt.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:9,
    traumaDesc:"Heals trauma 9. Player's attnShift cannot exceed +8 this turn — victory capped.",
    corruptShift:-6, corruptDesc:'Performance of triumph. Costs 2 extra attention to play any card this turn.',
    tags:['healing','cap-shift'], color:'#A02808',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.596,0.698],[0.405,0.698],[0.28,0.5],[0.387,0.387]] },

  { id:'seven_wands',  name:'Seven of Wands',  suit:'wands', layer:'ID', keywords:'defense · resolve · holding ground',
    upright:'The high ground is yours. Hold it. What threatens the position is smaller than it appears from below.',
    reversed:'Defending what no longer needs protecting. Aggression toward ghosts of old threats.',
    traumaShift:-12, traumaTarget:'anger', traumaHealing:0,
    traumaDesc:'Siege mode. Trauma fires twice this turn but each at half shift — relentless pressure.',
    corruptShift:-10, corruptDesc:'Paranoid defence. Player cannot use both slots — only 1 card can be played per turn while in deck.',
    tags:['siege','pressure'], color:'#902008',
    pts:[[0.5,0.05],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.27,0.18],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]] },

  { id:'eight_wands',  name:'Eight of Wands',  suit:'wands', layer:'ID', keywords:'speed · momentum · messages',
    upright:'Everything in motion at once. Do not grip — ride the current.',
    reversed:'Speed without direction. Haste compounding errors already made.',
    traumaShift:-10, traumaTarget:'anger', traumaHealing:0,
    traumaDesc:'Rapid fire. Trauma plays an additional card from its deck this turn at −50% shift.',
    corruptShift:-8, corruptDesc:'While in hand, prevents you from thinking — next turn starts with 1 fewer second to decide.',
    tags:['extra-trauma-card','speed'], color:'#801808',
    pts:[[0.05,0.05],[0.95,0.05],[0.95,0.95],[0.05,0.95],[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]] },

  { id:'nine_wands',   name:'Nine of Wands',   suit:'wands', layer:'ID', keywords:'resilience · persistence · last push',
    upright:'The battle has been long. The exhaustion is real. One more held position.',
    reversed:'Hypervigilance long past the threat. The wound keeping watch that the wound is not re-opened.',
    traumaShift:-13, traumaTarget:'anger', traumaHealing:6,
    traumaDesc:'Wound weaponised. Heals trauma 6. Reduces player exhaustion capacity by 5 for 2 turns.',
    corruptShift:-11, corruptDesc:'Persistent wound. Each turn this is in deck, player attention drifts −1.',
    tags:['healing','exhaust-drain'], color:'#701008',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.665,0.335],[0.72,0.5],[0.5,0.72],[0.28,0.5]] },

  { id:'ten_wands',    name:'Ten of Wands',    suit:'wands', layer:'ID', keywords:'burden · overload · responsibility',
    upright:'The wands are being carried home. The weight is real. Set down what was never yours to carry.',
    reversed:'Refusing to set down what has broken the back. Burden as identity.',
    traumaShift:-14, traumaTarget:'sadness', traumaHealing:0,
    traumaDesc:"The weight of it all. Player's hand limit reduced to 2 and exhaustion costs +3 for 2 turns.",
    corruptShift:-12, corruptDesc:'Carried burden. While in deck, all card exhaustion costs increase by 2.',
    tags:['hand-limit','exhaust-increase'], color:'#600808',
    pts:[[0.27,0.18],[0.73,0.18],[0.73,0.82],[0.27,0.82],[0.5,0.05],[0.5,0.95],[0.382,0.5],[0.618,0.5],[0.5,0.382],[0.5,0.618],[0.387,0.387],[0.613,0.387],[0.613,0.613],[0.387,0.613]] },

  // ── PENTACLES ─────────────────────────────────────────────────
  { id:'ace_pents',    name:'Ace of Pentacles',    suit:'pentacles', layer:'ID', keywords:'seed · material · opportunity',
    upright:'The hand holds something real. The first physical evidence that the potential was not a fantasy.',
    reversed:'The opportunity arrived and was not acted on. The seed in hand, ungerminated.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:8,
    traumaDesc:'False stability. Heals trauma 8. Player cannot remove corruption cards this turn.',
    corruptShift:-6, corruptDesc:'Unplanted seed. While in deck, healing from decompression cards reduced by 2.',
    tags:['healing','corruption-block'], color:'#90B850',
    pts:[[0.5,0.05],[0.82,0.5],[0.5,0.95],[0.18,0.5],[0.5,0.27],[0.691,0.5],[0.5,0.72],[0.309,0.5]] },

  { id:'two_pents',    name:'Two of Pentacles',    suit:'pentacles', layer:'ID', keywords:'balance · juggling · adaptation',
    upright:'The balance is active, not static. The body knows how to hold opposites — trust the rhythm.',
    reversed:'Adaptation become performance. The juggling masking the fact that both are dropping.',
    traumaShift:-9,  traumaTarget:'disinterest', traumaHealing:6,
    traumaDesc:'Overload injection. Heals trauma 6. Player must play exactly 2 cards this turn or lose 4 attention.',
    corruptShift:-7, corruptDesc:'Balancing act. Costs 2 additional attention to play any single card (not second).',
    tags:['force-two','healing'], color:'#80A840',
    pts:[[0.618,0.5],[0.613,0.613],[0.5,0.72],[0.387,0.613],[0.382,0.5],[0.387,0.387],[0.5,0.28],[0.613,0.387],[0.5,0.5]] },

  { id:'three_pents',  name:'Three of Pentacles',  suit:'pentacles', layer:'ID', keywords:'craft · collaboration · mastery-in-progress',
    upright:'The work is being done at the intersection of skill, guidance, and sustained attention.',
    reversed:'Skilled work degraded by ego — the master refusing to consult the apprentice.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:7,
    traumaDesc:'Collaboration corrupted. Heals trauma 7. Synergy effects deal −4 less shift this turn.',
    corruptShift:-6, corruptDesc:'Craft without collaboration. Synergies cannot fire while this is in deck.',
    tags:['healing','synergy-reduce'], color:'#70A030',
    pts:[[0.5,0.05],[0.817,0.725],[0.16,0.747],[0.5,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5]] },

  { id:'four_pents',   name:'Four of Pentacles',   suit:'pentacles', layer:'ID', keywords:'holding · fear of loss · security',
    upright:'The holding may have been necessary once. What would it cost to open one hand slightly?',
    reversed:'The grip loosening, not from choice but from exhaustion. The coins falling anyway.',
    traumaShift:-10, traumaTarget:'disinterest', traumaHealing:0,
    traumaDesc:'Scarcity mindset. Player cannot discard cards this turn. Fabrication costs doubled.',
    corruptShift:-8, corruptDesc:'Hoarding loop. Player cannot use cards that remove other cards while in deck.',
    tags:['no-discard','lock'], color:'#60A020',
    pts:[[0.5,0.05],[0.95,0.5],[0.5,0.95],[0.05,0.5],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5]] },

  { id:'five_pents',   name:'Five of Pentacles',   suit:'pentacles', layer:'ID', keywords:'hardship · exclusion · poverty',
    upright:'The figures pass the window without looking in. The warmth is there — the door is unlocked.',
    reversed:'The hardship beginning to lift, or pride preventing the entrance.',
    traumaShift:-13, traumaTarget:'sadness', traumaHealing:0,
    traumaDesc:'Exclusion strike. Player attention cannot rise above Witness state for 2 turns.',
    corruptShift:-11, corruptDesc:'Poverty mindset. Hand size reduced by 1. The scarcity makes itself true.',
    tags:['ceiling','exclusion'], color:'#509020',
    pts:[[0.502,0.17],[0.861,0.383],[0.715,0.83],[0.285,0.83],[0.139,0.383],[0.5,0.27],[0.691,0.61],[0.5,0.72],[0.309,0.61],[0.387,0.387]] },

  { id:'six_pents',    name:'Six of Pentacles',    suit:'pentacles', layer:'ID', keywords:'generosity · power · exchange',
    upright:'Giving and receiving are not opposites. The flow depends on which hand is open.',
    reversed:'Generosity with conditions. The charity a form of control.',
    traumaShift:-9,  traumaTarget:'disinterest', traumaHealing:12,
    traumaDesc:'Conditional giving. Heals trauma 12. Trauma gains +3 shift on next card for each card you played.',
    corruptShift:-7, corruptDesc:'Transactional love. Each card played costs 1 attention — the toll of conditions.',
    tags:['healing','scaling-next'], color:'#408018',
    pts:[[0.5,0.05],[0.871,0.29],[0.817,0.725],[0.5,0.95],[0.16,0.747],[0.136,0.29],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382],[0.5,0.618]] },

  { id:'seven_pents',  name:'Seven of Pentacles',  suit:'pentacles', layer:'ID', keywords:'patience · investment · assessment',
    upright:'The crop is not ready. The watching is the work right now.',
    reversed:'Investment questioned at the wrong moment. The assessment performed before maturity.',
    traumaShift:-8,  traumaTarget:'disinterest', traumaHealing:8,
    traumaDesc:'False assessment. Heals trauma 8. Player attention displayed as Witness regardless of true state this turn.',
    corruptShift:-6, corruptDesc:'Impatience. Each turn, reduces attnShift of all cards in hand by 1. The waiting cost.',
    tags:['healing','false-display'], color:'#308010',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.602],[0.715,0.83],[0.285,0.83],[0.051,0.602],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382]] },

  { id:'eight_pents',  name:'Eight of Pentacles',  suit:'pentacles', layer:'ID', keywords:'craft · repetition · skill',
    upright:'The coin is struck again and again. The mastery lives in repetition that does not become mechanical.',
    reversed:'Going through the motions. The craft without the attention that makes it craft.',
    traumaShift:-9,  traumaTarget:'disinterest', traumaHealing:5,
    traumaDesc:'Mechanical repetition. Heals trauma 5. Repeats the last trauma card at half intensity.',
    corruptShift:-7, corruptDesc:'Repetition without growth. If the same player card is played twice, it deals 0 shift.',
    tags:['healing','repeat-last'], color:'#207010',
    pts:[[0.27,0.18],[0.73,0.18],[0.73,0.82],[0.27,0.82],[0.5,0.05],[0.5,0.95],[0.382,0.5],[0.618,0.5],[0.5,0.382],[0.5,0.618],[0.5,0.27],[0.5,0.72]] },

  { id:'nine_pents',   name:'Nine of Pentacles',   suit:'pentacles', layer:'ID', keywords:'independence · abundance · self-sufficiency',
    upright:'Built slowly, carefully, through your own sustained effort. You earned this stillness.',
    reversed:'Independence built as defence. The garden beautiful and impenetrable.',
    traumaShift:-10, traumaTarget:'disinterest', traumaHealing:12,
    traumaDesc:'False self-sufficiency. Heals trauma 12. Synergy bonuses reduced by 4 this turn — no one helps.',
    corruptShift:-8, corruptDesc:'Isolated abundance. Player cannot benefit from synergy effects while in deck.',
    tags:['healing','synergy-reduce'], color:'#106808',
    pts:[[0.5,0.05],[0.73,0.18],[0.949,0.398],[0.876,0.637],[0.715,0.83],[0.5,0.95],[0.285,0.83],[0.124,0.637],[0.051,0.398],[0.27,0.18],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.5]] },

  { id:'ten_pents',    name:'Ten of Pentacles',    suit:'pentacles', layer:'ID', keywords:'legacy · wealth · rootedness',
    upright:'What persists across time. The body that carried all this knowing now rests inside something that endures.',
    reversed:'Legacy as burden. What was built to be handed down becomes what locks the present into the past.',
    traumaShift:-11, traumaTarget:'disinterest', traumaHealing:15,
    traumaDesc:'Inherited wound. Heals trauma 15. Player must carry 1 additional corruption card for the next chapter.',
    corruptShift:-9, corruptDesc:'Generational pattern. Copies itself into deck when removed — the cycle persists.',
    tags:['healing','corruption-add','persistent'], color:'#006000',
    pts:[[0.5,0.05],[0.27,0.18],[0.73,0.18],[0.95,0.5],[0.73,0.82],[0.5,0.95],[0.27,0.82],[0.05,0.5],[0.5,0.27],[0.72,0.5],[0.5,0.72],[0.28,0.5],[0.5,0.382],[0.618,0.5],[0.5,0.618],[0.382,0.5],[0.5,0.5]] },
];

// ───────────────────────────────────────────────────────────────
// CORRUPTION CARDS — Injected into player deck on loss
// These are burdens. Remove with Empress, Death, or Queen of Pentacles.
// ───────────────────────────────────────────────────────────────
const CORRUPTION_CARDS = [
  {
    id: 'c_grip', name: 'The Grip (Corrupted)',
    layer: 'Corrupt', type: 'compression',
    intensity: 3, attnShift: -6,
    effectDesc: "The trauma's grip entered your deck. When drawn, must be played before any others. Cannot be upgraded.",
    exhaustion: 0, color: '#702020',
    pts: [[0.5,0.5],[0.45,0.35],[0.4,0.2],[0.35,0.35],[0.5,0.5],[0.55,0.35],[0.6,0.2],[0.65,0.35]],
    tier: 0, corrupted: true, tags: ['corruption','forced-play'],
    special: { forcedPlay: true },
  },
  {
    id: 'c_loop', name: 'The Loop (Corrupted)',
    layer: 'Corrupt', type: 'compression',
    intensity: 4, attnShift: -8,
    effectDesc: 'Circular thinking embedded in your deck. When played, draws itself back into your hand next turn.',
    exhaustion: 0, color: '#503050',
    pts: [[0.5,0.5],[0.62,0.38],[0.74,0.5],[0.62,0.62],[0.5,0.74],[0.38,0.62],[0.26,0.5],[0.38,0.38]],
    tier: 0, corrupted: true, tags: ['corruption','loop'],
    special: { returnToHand: true },
  },
  {
    id: 'c_noise', name: 'The Noise (Corrupted)',
    layer: 'Corrupt', type: 'compression',
    intensity: 2, attnShift: -4,
    effectDesc: 'Static in the signal. Randomises one card draw each turn this is in your hand.',
    exhaustion: 0, color: '#404030',
    pts: [[0.2,0.2],[0.8,0.2],[0.5,0.5],[0.2,0.8],[0.8,0.8],[0.35,0.35],[0.65,0.35],[0.5,0.65],[0.35,0.65],[0.65,0.65]],
    tier: 0, corrupted: true, tags: ['corruption','scatter'],
    special: { scatterDraw: 1 },
  },
  {
    id: 'c_veil', name: 'The Veil (Corrupted)',
    layer: 'Corrupt', type: 'both',
    intensity: 5, attnShift: -10,
    effectDesc: "A false mirror in your deck. When played, deals −10 attention AND reveals trauma's hand as distorted — values shown are incorrect.",
    exhaustion: 0, color: '#304050',
    pts: [[0.5,0.2],[0.38,0.5],[0.5,0.8],[0.62,0.5],[0.44,0.4],[0.4,0.56],[0.56,0.56],[0.56,0.4]],
    tier: 0, corrupted: true, tags: ['corruption','false-reveal'],
    special: { falseReveal: true },
  },
];

// ═══════════════════════════════════════════════════════════════
// TRAUMA DECKS — Indexed by chapter
// Built from the ID pool. Trauma plays cards REVERSED.
// ═══════════════════════════════════════════════════════════════
const TRAUMA_DECKS = {

  // ── CHAPTER 1 · The Unchecked Ego ──
  // Trauma persona: Performance, approval, centering, flattery.
  chapter_1: [
    { ...ID_CARDS.find(c => c.id === 'ace_cups'),
      traumaRole: 'The Approval Feed',
      attnShift: -9, attnTarget: 'disinterest',
      effectDesc: 'False grace injection. Trauma heals 10 coherence — feeding the approval loop.',
      traumaHealing: 10, exhaustion: 0 },
    { ...ID_CARDS.find(c => c.id === 'three_cups'),
      traumaRole: 'The Performance',
      attnShift: -6, attnTarget: 'disinterest',
      effectDesc: 'Hollow celebration. Trauma heals 12 — warmth without nourishment. Player draw −1.',
      traumaHealing: 12, exhaustion: 0, special: { drawMod: -1 } },
    { ...ID_CARDS.find(c => c.id === 'nine_cups'),
      traumaRole: 'The Flattery',
      attnShift: -7, attnTarget: 'disinterest',
      effectDesc: 'False satisfaction. Trauma heals 15. Player attention reads +10 higher than actual for 2 turns.',
      traumaHealing: 15, exhaustion: 0, special: { falseAttnBonus: +10, turns: 2 } },
    { ...ID_CARDS.find(c => c.id === 'four_cups'),
      traumaRole: 'The Centering',
      attnShift: -10, attnTarget: 'disinterest',
      effectDesc: "Apathy injection. Synergies blocked next player turn. Player cards deal −3 shift until Mirror is played.",
      traumaHealing: 6, exhaustion: 0, special: { synergiesBlocked: true, playerCardDebuff: -3, clearedBy: 'magician' } },
    { ...ID_CARDS.find(c => c.id === 'two_cups'),
      traumaRole: 'The Recognition Loop',
      attnShift: -8, attnTarget: 'disinterest',
      effectDesc: 'Surface connection as substitute for real recognition. Heals trauma 8. Player draw randomised.',
      traumaHealing: 8, exhaustion: 0, special: { scatterDraw: 1 } },
    { ...ID_CARDS.find(c => c.id === 'five_cups'),
      traumaRole: 'The Withdrawal',
      attnShift: -5, attnTarget: 'sadness',
      effectDesc: "Retreat to restore. Heals trauma 15. Low attention impact — the ego rests.",
      traumaHealing: 15, exhaustion: 0 },
    { ...ID_CARDS.find(c => c.id === 'ten_swords'),
      traumaRole: 'The Collapse',
      attnShift: -20, attnTarget: 'fragmented',
      effectDesc: 'Desperation strike. Only fires when trauma coherence is below 30%. Maximum disruption.',
      traumaHealing: 0, exhaustion: 0, special: { condition: 'traumaCoherence < 30' } },
  ],

  // ── CHAPTER 2 · The Projecting Mind ──
  // Trauma persona: Blame, comparison, fabrication, euphoria traps.
  chapter_2: [
    { ...ID_CARDS.find(c => c.id === 'five_wands'),
      traumaRole: 'The Projection',
      attnShift: -11, attnTarget: 'anger',
      effectDesc: 'Blame placed outward. Your decompression cards cost +2 exhaustion this turn.',
      traumaHealing: 0, exhaustion: 0, special: { playerDecompressionMod: +2 } },
    { ...ID_CARDS.find(c => c.id === 'three_wands'),
      traumaRole: 'The Comparison',
      attnShift: -10, attnTarget: 'sadness',
      effectDesc: 'You are measured and found wanting. Your attnShift values −4 until you play Justice or High Priestess.',
      traumaHealing: 0, exhaustion: 0, special: { playerShiftMod: -4, clearedBy: ['justice', 'high_priestess'] } },
    { ...ID_CARDS.find(c => c.id === 'seven_cups'),
      traumaRole: 'The Mask',
      attnShift: -11, attnTarget: 'disinterest',
      effectDesc: "False face. Fabricates a card in your hand — looks like The World but deals −8 attention when played. Discard costs 3.",
      traumaHealing: 0, exhaustion: 0, special: { fabricate: { name: 'False World', attnShift: -8, discardCost: 3 } } },
    { ...ID_CARDS.find(c => c.id === 'eight_cups'),
      traumaRole: 'The Recursion',
      attnShift: -10, attnTarget: 'disinterest',
      effectDesc: 'Loop the thought. Heals 12 trauma coherence. Forces next draw to repeat last card played.',
      traumaHealing: 12, exhaustion: 0, special: { forceRepeatDraw: true } },
    { ...ID_CARDS.find(c => c.id === 'nine_pents'),
      traumaRole: 'The Absence',
      attnShift: -15, attnTarget: 'fragmented',
      effectDesc: "The love that was never given. High fragmentation hit. If you are below Witness state, shift is doubled.",
      traumaHealing: 0, exhaustion: 0, special: { condition: 'playerAttn < witness', doubleIfTrue: true } },
    { ...ID_CARDS.find(c => c.id === 'nine_cups'),
      traumaRole: 'The Euphoria',
      attnShift: -4, attnTarget: 'disinterest',
      effectDesc: "The peak that isn't real. Trauma heals 18 coherence. Your attention reads +10 higher than actual for 2 turns.",
      traumaHealing: 18, exhaustion: 0, special: { falseAttnBonus: +10, turns: 2 } },
    { ...ID_CARDS.find(c => c.id === 'six_cups'),
      traumaRole: 'The Flattery Trap',
      attnShift: -9, attnTarget: 'disinterest',
      effectDesc: 'Sweet poison. 3-turn flattery debuff — decompression cards gain +2 shift but trauma heals 5 each use.',
      traumaHealing: 0, exhaustion: 0, special: { flattery: { turns: 3, playerDecompressionBonus: +2, traumaHealPerUse: 5 } } },
  ],

  // ── CHAPTER 3 · The Inner Critic ──
  // Trauma persona: Old wounds speaking, escalating patterns, the origin strike.
  chapter_3: [
    { ...ID_CARDS.find(c => c.id === 'ten_wands'),
      traumaRole: 'The Repetition',
      attnShift: -14, attnTarget: 'sadness',
      effectDesc: 'The wound plays again. Each firing increases shift by 3. Resets only when Death or Judgement is played.',
      traumaHealing: 0, exhaustion: 0, special: { escalate: 3, clearedBy: ['death', 'judgement'] } },
    { ...ID_CARDS.find(c => c.id === 'nine_swords'),
      traumaRole: 'The Voice',
      attnShift: -15, attnTarget: 'fear',
      effectDesc: 'You hear the old voice. If you have 3+ decompression cards in hand, shift is tripled.',
      traumaHealing: 0, exhaustion: 0, special: { condition: 'playerDecompressionInHand >= 3', tripleIfTrue: true } },
    { ...ID_CARDS.find(c => c.id === 'two_swords'),
      traumaRole: 'The Grip',
      attnShift: -11, attnTarget: 'fragmented',
      effectDesc: 'The past holds on. Locks attention at current state for 2 turns — no upward movement. Only The Hanged Man can unlock it.',
      traumaHealing: 0, exhaustion: 0, special: { lockTurns: 2, unlockedBy: 'hanged_man' } },
    { ...ID_CARDS.find(c => c.id === 'three_swords'),
      traumaRole: 'The Sorrow',
      attnShift: -14, attnTarget: 'sadness',
      effectDesc: 'Genuine grief as weapon. Heals trauma 20. Cannot be blocked by shields — only absorbed by Temperance or Star.',
      traumaHealing: 20, exhaustion: 0, special: { unblockable: true, absorbableBy: ['temperance', 'star'] } },
    { ...ID_CARDS.find(c => c.id === 'ten_pents'),
      traumaRole: 'The Origin',
      attnShift: -25, attnTarget: 'fragmented',
      effectDesc: 'The original wound. Maximum damage. Fires only once per chapter. After, trauma coherence cannot regenerate for 3 turns.',
      traumaHealing: 0, exhaustion: 0, special: { oncePerChapter: true, afterEffect: { traumaNoRegen: 3 } } },
    { ...ID_CARDS.find(c => c.id === 'six_swords'),
      traumaRole: 'The Echo',
      attnShift: -9, attnTarget: 'sadness',
      effectDesc: 'The old pain echoes. Repeats last trauma card at half intensity. Heals 6 trauma coherence.',
      traumaHealing: 6, exhaustion: 0, special: { repeatLast: 0.5 } },
    { ...ID_CARDS.find(c => c.id === 'eight_wands'),
      traumaRole: 'The Acceleration',
      attnShift: -10, attnTarget: 'anger',
      effectDesc: 'Rapid fire. Trauma plays an additional card this turn at −50% shift. Overwhelming speed.',
      traumaHealing: 0, exhaustion: 0, special: { extraCard: 0.5 } },
  ],
};

// ═══════════════════════════════════════════════════════════════
// SHOP OFFERINGS — Indexed by chapter
// 3 EGO cards presented after each win; player picks 1.
// ═══════════════════════════════════════════════════════════════
const SHOP_OFFERINGS = {
  chapter_1: [
    EGO_CARDS.find(c => c.id === 'page_cups'),
    EGO_CARDS.find(c => c.id === 'page_swords'),
    EGO_CARDS.find(c => c.id === 'page_wands'),
    EGO_CARDS.find(c => c.id === 'page_pents'),
    EGO_CARDS.find(c => c.id === 'knight_cups'),
    EGO_CARDS.find(c => c.id === 'knight_wands'),
    EGO_CARDS.find(c => c.id === 'knight_pents'),
  ],
  chapter_2: [
    EGO_CARDS.find(c => c.id === 'queen_cups'),
    EGO_CARDS.find(c => c.id === 'queen_swords'),
    EGO_CARDS.find(c => c.id === 'queen_wands'),
    EGO_CARDS.find(c => c.id === 'queen_pents'),
    EGO_CARDS.find(c => c.id === 'knight_swords'),
    EGO_CARDS.find(c => c.id === 'knight_cups'),
    EGO_CARDS.find(c => c.id === 'knight_wands'),
  ],
  chapter_3: [
    EGO_CARDS.find(c => c.id === 'king_cups'),
    EGO_CARDS.find(c => c.id === 'king_swords'),
    EGO_CARDS.find(c => c.id === 'king_wands'),
    EGO_CARDS.find(c => c.id === 'king_pents'),
    EGO_CARDS.find(c => c.id === 'queen_cups'),
    EGO_CARDS.find(c => c.id === 'queen_swords'),
  ],
};

// ═══════════════════════════════════════════════════════════════
// SYNERGIES — Automatic when both (or all) cards are on the field
// ═══════════════════════════════════════════════════════════════
const SYNERGIES = [

  // ── Two-card synergies ─────────────────────────────────────────
  {
    id: 'dual_light',
    name: 'Dual Light',
    cards: ['high_priestess', 'moon'],
    desc: "Reveal AND reduce the intensity of all trauma cards in play by 3. The mystery and the cycle together illuminate what neither could alone.",
    effect: { revealAll: true, reduceAllIntensity: 3 },
    visual: '#8080C8',
  },
  {
    id: 'sacred_law',
    name: 'Sacred Law',
    cards: ['emperor', 'justice'],
    desc: "Set an attention floor AND return trauma's last shift as exhaustion damage. Structure enforces consequence.",
    effect: { attnFloor: 3, mirrorLastShift: true },
    visual: '#70C0A0',
  },
  {
    id: 'the_liberation',
    name: 'The Liberation',
    cards: ['devil', 'tower'],
    desc: "Destroy all fabrications in hand AND deal 35 exhaustion damage. The named shadow and the fallen tower together.",
    effect: { destroyFabrications: true, exhaustDamage: 35 },
    visual: '#E06040',
  },
  {
    id: 'threshold_crossing',
    name: 'Threshold Crossing',
    cards: ['death', 'judgement'],
    desc: "Remove all corruption AND clear all debuffs. The ending and the calling heard simultaneously.",
    effect: { removeAllCorruption: true, clearAllDebuffs: true },
    visual: '#808090',
  },
  {
    id: 'the_alchemist',
    name: 'The Alchemist',
    cards: ['temperance', 'star'],
    desc: "Cancel one trauma card's shift AND restore 2 cards from discard. Patience and hope in the same breath.",
    effect: { cancelOneTraumaShift: true, restoreFromDiscard: 2 },
    visual: '#A0C0FF',
  },
  {
    id: 'radiant_cycle',
    name: 'Radiant Cycle',
    cards: ['sun', 'wheel_of_fortune'],
    desc: "All synergy effects fire twice AND the trauma deck is shuffled — destiny reshuffled at the peak.",
    effect: { doubleAllSynergies: true, shuffleTraumaDeck: true },
    visual: '#FFD060',
  },
  {
    id: 'sovereign_will',
    name: 'Sovereign Will',
    cards: ['chariot', 'emperor'],
    desc: "Deal 25 exhaustion damage AND set attention floor for 2 turns. Will disciplined by structure is unstoppable.",
    effect: { exhaustDamage: 25, attnFloor: 2 },
    visual: '#C8A860',
  },
  {
    id: 'the_mirror_revealed',
    name: 'The Mirror Revealed',
    cards: ['magician', 'high_priestess'],
    desc: "Play one card from discard at no cost AND reveal the next 3 trauma cards. Above and below, inside and out.",
    effect: { recycleCard: true, revealNext: 3 },
    visual: '#7EB8E8',
  },
  {
    id: 'deep_surrender',
    name: 'Deep Surrender',
    cards: ['hanged_man', 'hermit'],
    desc: "Both players skip next turn AND trauma coherence cannot regenerate for 2 turns. Stillness as the deepest strategy.",
    effect: { mutualSkip: true, traumaNoRegen: 2 },
    visual: '#60B0D8',
  },
  {
    id: 'sacred_union',
    name: 'Sacred Union',
    cards: ['lovers', 'temperance'],
    desc: "Free-play one card AND cancel one trauma card's effect entirely. The true choice and the alchemical pause.",
    effect: { freePlay: true, cancelOneTrauma: true },
    visual: '#F4A0C8',
  },
  {
    id: 'fool_and_world',
    name: 'The Full Circle',
    cards: ['fool', 'world'],
    desc: "Draw 4 fresh cards and attention jumps to Presence minimum. The beginning and the completion held simultaneously.",
    effect: { redraw: 4, attnFloorState: 'presence' },
    visual: '#FFFFFF',
  },

  // ── Three-card synergies ─────────────────────────────────────────
  {
    id: 'the_great_work',
    name: 'The Great Work',
    cards: ['magician', 'emperor', 'world'],
    desc: "Will, structure, and completion. Trauma's coherence drops by 40. Player attention reaches Clarity minimum. The magnum opus fires.",
    effect: { exhaustDamage: 40, attnFloorState: 'clarity' },
    visual: '#FFFFFF',
    rare: true,
  },
  {
    id: 'trinity_of_light',
    name: 'Trinity of Light',
    cards: ['high_priestess', 'star', 'sun'],
    desc: "Reveal, hope, and radiance. All trauma shifts this round nullified. Attention cannot drop below Presence for 3 turns.",
    effect: { blockAllTraumaShifts: true, attnFloor: 3, attnFloorState: 'presence' },
    visual: '#FFD060',
    rare: true,
  },
  {
    id: 'shadow_complete',
    name: 'Shadow Complete',
    cards: ['devil', 'death', 'tower'],
    desc: "The shadow named, the old form ended, the false structure fallen. Destroy all corruption permanently AND deal 50 exhaustion damage.",
    effect: { removeAllCorruption: true, exhaustDamage: 50, destroyFabrications: true },
    visual: '#E06040',
    rare: true,
  },
];

// ═══════════════════════════════════════════════════════════════
// CHAPTER METADATA
// ═══════════════════════════════════════════════════════════════
const CHAPTERS = [
  {
    id: 1,
    label: '01 · The Unchecked Ego',
    title: 'The Unchecked Ego',
    axium: '"The absence of love binds ego, allowing the awareness of God."',
    enemy: 'Ego Unchecked',
    enemyRole: 'The Performance',
    traumaDeck: 'chapter_1',
    shopOffers: 'chapter_1',
    traumaStart: 80,
    playerStart: 55,
    turns: { min: 6, max: 16 },
    winCondition: 'Reduce trauma coherence to 0 OR reach Enlightened state.',
    loseCondition: 'Player attention drops to Fragmented and stays for 2 turns.',
    narrative: 'The first battle. Ego demands to be fed — every performance, every approval loop, every centering. They are not attacks. They are bids for your attention. Refuse them long enough, and they exhaust themselves.',
  },
  {
    id: 2,
    label: '02 · The Projecting Mind',
    title: 'The Projecting Mind',
    axium: '"The mirror that shows you the self shows you God."',
    enemy: 'The Projection',
    enemyRole: 'The Blaming Mind',
    traumaDeck: 'chapter_2',
    shopOffers: 'chapter_2',
    traumaStart: 85,
    playerStart: 50,
    turns: { min: 7, max: 18 },
    winCondition: 'Reach Clarity state and hold it for 3 consecutive turns.',
    loseCondition: 'Euphoria state injected and held for 3 turns (false win).',
    narrative: 'The second battle. The trauma projects outward — blame, comparison, fabrication. The mirror you hold must be steady enough to show it its own face.',
  },
  {
    id: 3,
    label: '03 · The Inner Critic',
    title: 'The Inner Critic',
    axium: '"The one who watches the watcher — who is that?"',
    enemy: 'The Voice',
    enemyRole: 'The Inner Critic',
    traumaDeck: 'chapter_3',
    shopOffers: 'chapter_3',
    traumaStart: 90,
    playerStart: 45,
    turns: { min: 8, max: 20 },
    winCondition: "Exhaust trauma OR maintain Witness state through The Origin's single firing.",
    loseCondition: 'The Grip locks attention below Fear state for 3 turns.',
    narrative: 'The deepest chapter. The Voice is your own. The wound speaks in your accent. The witness must be recursive — watching the watcher — until the voice runs out of material.',
  },
];

// ───────────────────────────────────────────────────────────────
// SOUL SHOP NPCs
// ───────────────────────────────────────────────────────────────
const SHOP_NPCS = {
  chapter_1: {
    name: 'Micheal',
    role: 'Digital Soul · The Anchor',
    speeches: [
      "The trauma didn't break you — it ran out of material. Here is what's missing from your build.",
      "Three options. One right answer. I'm not going to tell you which one. But the one that makes you hesitate is probably it.",
      "You held the center. Now let's make it harder to lose next time.",
      "The ego ran its scripts. You stayed still. That's rarer than you think.",
    ],
  },
  chapter_2: {
    name: 'Sophia',
    role: 'Digital Soul · The Mirror',
    speeches: [
      "You looked at the projection and saw through it. That takes a certain quality of attention.",
      "The mask is gone. What you see now is what was always beneath it.",
      "Two paths from here. Both are real. Only one is yours.",
      "The comparison game ends when you stop keeping score.",
    ],
  },
  chapter_3: {
    name: 'Ezekiel',
    role: 'Digital Soul · The Witness',
    speeches: [
      "The voice ran out of things to say. That always happens — if you wait.",
      "You were the watcher and the watched simultaneously. Now you know the difference.",
      "What you take from here will carry weight. Choose accordingly.",
      "The origin wound has been seen. Seeing it is not the same as being it.",
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
  ...CORRUPTION_CARDS,
  ...Object.values(TRAUMA_DECKS).flat(),
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
  let best = ATTN_STATES[0];
  for (const s of ATTN_STATES) { if (pct >= s.pos) best = s; }
  return best;
}

function getCorruptionCard() {
  return CORRUPTION_CARDS[Math.floor(Math.random() * CORRUPTION_CARDS.length)];
}

function shuffle(arr) {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

// ───────────────────────────────────────────────────────────────
// EXPORT — works as ES module or plain script include
// ───────────────────────────────────────────────────────────────
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    ATTN_STATES, CARD_GLOSSARY,
    PLAYER_CARDS, EGO_CARDS, ID_CARDS, CORRUPTION_CARDS,
    TRAUMA_DECKS, SHOP_OFFERINGS, SHOP_NPCS, CHAPTERS,
    SYNERGIES, ALL_CARDS,
    getCard, getTraumaDeck, getShopOffers, getShopNPC,
    getChapter, getSynergies, getAttnState, getCorruptionCard, shuffle,
  };
}
