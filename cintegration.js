/**
 * ═══════════════════════════════════════════════════════════════
 * AXIUM — CARD SYSTEM INTEGRATION  v2.0
 * Load order: cards → engine → ui → shop → csystem → cintegration → battle
 *
 * Wires csystem.js (CardRenderer, HandPickerModal, FieldZone) into
 * the battle screen. Patches renderBattleField, openHandPickerBattle,
 * closeHandPickerBattle, buildDeckReview, and animateShopCardCanvas.
 * ═══════════════════════════════════════════════════════════════
 */

'use strict';

// ─────────────────────────────────────────────────────────────
// MODULE STATE
// ─────────────────────────────────────────────────────────────
let _axcPicker = null;   // HandPickerModal
let _axcFieldP = null;   // FieldZone — player
let _axcFieldE = null;   // FieldZone — enemy

// ─────────────────────────────────────────────────────────────
// INIT — called from enterBattle() after the screen is shown
// ─────────────────────────────────────────────────────────────
function axcInit() {
  // Clean up previous instances
  if (_axcPicker) { _axcPicker.destroy(); _axcPicker = null; }
  if (_axcFieldP) { _axcFieldP.destroy(); _axcFieldP = null; }
  if (_axcFieldE) { _axcFieldE.destroy(); _axcFieldE = null; }

  // Hide legacy hand-picker HTML (still in DOM for compatibility)
  const oldPicker = document.getElementById('hand-picker');
  if (oldPicker) oldPicker.style.display = 'none';

  // Create FieldZones
  const pfEl = document.getElementById('field-player');
  const efEl = document.getElementById('field-enemy');
  if (pfEl) _axcFieldP = new FieldZone(pfEl, false);
  if (efEl) _axcFieldE = new FieldZone(efEl, true);

  // Create HandPickerModal inside battle-wrap
  const battleWrap = document.getElementById('battle-wrap');
  if (battleWrap) {
    _axcPicker = new HandPickerModal(battleWrap, B.playerHand, B.playerPlayed, {
      maxStaged:      10,
      getSynergies:   (ids) => typeof getSynergies === 'function' ? getSynergies(ids) : [],

      // onStagedChange receives the live playerPlayed array (already {card,reversed} entries).
      // Assign it directly to B.playerPlayed — the reversed flags are set by the polarity pills.
      onStagedChange: (played) => {
        B.playerPlayed = played;
        axcRenderField();
        updateBattlePhaseUI(B);
      },

      onResolve: () => { resolveRound(); },

      onClose: () => {
        axcRenderField();
        updateBattlePhaseUI(B);
      },
    });
  }

  axcRenderField();
}

// ─────────────────────────────────────────────────────────────
// OPEN / CLOSE HAND PICKER
// Called by HTML buttons and by battle.js openHandPickerBattle()
// ─────────────────────────────────────────────────────────────
function axcOpenHandPicker() {
  if (!_axcPicker) axcInit();
  // Sync latest hand (in case it was re-dealt)
  if (_axcPicker) {
    _axcPicker.hand = B.playerHand;
    _axcPicker.open();
  }
}

function axcCloseHandPicker() {
  _axcPicker?.close();
  axcRenderField();
  updateBattlePhaseUI(B);
}

// ─────────────────────────────────────────────────────────────
// RENDER FIELD
// Draws real cards via FieldZone + injects polarity badges.
// ─────────────────────────────────────────────────────────────
function axcRenderField() {
  if (_axcFieldP) {
    const cards = B.playerPlayed.map(e => e.card || e);
    _axcFieldP.render(cards, (idx, card) => axcUnstageCard(idx, card));
    // After FieldZone builds DOM, overlay polarity toggle badges
    _injectPolarityBadges();
  }
  if (_axcFieldE) {
    _axcFieldE.render((B.enemyHand || []).map(e => e.card || e));
  }
}

// Add ↑ NRM / ↓ REV badges below each player field card
function _injectPolarityBadges() {
  const pf = document.getElementById('field-player');
  if (!pf) return;
  // FieldZone wraps each card in a position:relative div
  const wraps = pf.querySelectorAll('div[style*="position:relative"]');
  wraps.forEach((wrap, i) => {
    const entry = B.playerPlayed[i];
    if (!entry) return;
    // Don't double-add
    if (wrap.querySelector('.axc-pol-badge')) return;

    const badge = document.createElement('div');
    badge.className = 'axc-pol-badge';
    const isRev = entry.reversed;
    badge.style.cssText = `
      position:absolute; bottom:-8px; left:50%; transform:translateX(-50%);
      font-family:'Space Mono',monospace; font-size:6.5px; letter-spacing:.06em;
      padding:2px 7px; border-radius:10px; border:1px solid; cursor:pointer;
      z-index:25; transition:all .18s; white-space:nowrap; user-select:none;
      ${isRev
        ? 'color:rgba(224,85,85,.9);border-color:rgba(224,85,85,.5);background:rgba(224,85,85,.12);'
        : 'color:rgba(212,175,55,.85);border-color:rgba(212,175,55,.4);background:rgba(212,175,55,.1);'}
    `;
    badge.textContent = isRev ? '↓ REV' : '↑ NRM';
    badge.title = isRev ? 'Reversed — tap to set Normal' : 'Normal — tap to set Reversed';
    badge.addEventListener('click', e => {
      e.stopPropagation();
      entry.reversed = !entry.reversed;
      axcRenderField();
      updateBattlePhaseUI(B);
    });
    wrap.appendChild(badge);
  });
}

// ─────────────────────────────────────────────────────────────
// UNSTAGE CARD
// ─────────────────────────────────────────────────────────────
function axcUnstageCard(idx, card) {
  // Remove from B.playerPlayed
  if (idx >= 0 && idx < B.playerPlayed.length) {
    B.playerPlayed.splice(idx, 1);
  } else if (card) {
    const fi = B.playerPlayed.findIndex(e => (e.card || e).id === card.id);
    if (fi >= 0) B.playerPlayed.splice(fi, 1);
  }
  // Deselect in picker
  if (_axcPicker && card) {
    const r = _axcPicker._renderers?.get(card.id);
    if (r) r.setStaged(false);
    _axcPicker._updateCount?.();
  }
  axcRenderField();
  updateBattlePhaseUI(B);
}

// ─────────────────────────────────────────────────────────────
// PATCH window.dealHand — sync picker after re-deal
// ─────────────────────────────────────────────────────────────
const _origDealHand = window.dealHand;
window.dealHand = function () {
  if (typeof _origDealHand === 'function') _origDealHand();
  if (_axcPicker) _axcPicker.hand = B.playerHand;
  axcRenderField();
};

// ─────────────────────────────────────────────────────────────
// PATCH window.renderBattleField — legacy callers
// ─────────────────────────────────────────────────────────────
window.renderBattleField = function (_B, _onUnstage) {
  axcRenderField();
  updateBattlePhaseUI(B);
};

// ─────────────────────────────────────────────────────────────
// axcGetFieldRenderer(cardId) → CardRenderer | null
// Used by battle.js to drive constellation charge on field cards.
// Looks in both player and enemy FieldZone renderer maps.
// ─────────────────────────────────────────────────────────────
window.axcGetFieldRenderer = function(cardId) {
  if (_axcFieldP && _axcFieldP._renderers.has(cardId)) return _axcFieldP._renderers.get(cardId);
  if (_axcFieldE && _axcFieldE._renderers.has(cardId)) return _axcFieldE._renderers.get(cardId);
  return null;
};

// ─────────────────────────────────────────────────────────────
// PATCH hand picker functions for HTML buttons + battle.js
// These must be on window so HTML onclick= and battle.js can find them.
// battle.js also declares openHandPickerBattle / closeHandPickerBattle
// as named functions — those will delegate to axcOpenHandPicker via
// the typeof check, so no conflict.
// ─────────────────────────────────────────────────────────────
window.openHandPickerBattle  = axcOpenHandPicker;
window.closeHandPickerBattle = axcCloseHandPicker;
// Also patch the ui.js names in case anything calls them directly
window.openHandPicker  = function (_B) { axcOpenHandPicker(); };
window.closeHandPicker = function ()   { axcCloseHandPicker(); };

// ─────────────────────────────────────────────────────────────
// SHOP — upgrade card canvases to ConstellationAnim + particles
// ─────────────────────────────────────────────────────────────
window.animateShopCardCanvas = function (canvas, card) {
  if (!canvas || !card) return;
  requestAnimationFrame(() => {
    if (!canvas.offsetWidth) return;
    const anim = new ConstellationAnim(canvas, card, {
      speed: 0.016, nodeSize: 1.6, edgeAlpha: 0.3, glowMult: 3,
    });
    anim.start();
    if (APP?.shopCanvasAnims) {
      const key = 'shop-' + card.id;
      const old = APP.shopCanvasAnims.get(key);
      if (old?.stop) old.stop();
      APP.shopCanvasAnims.set(key, anim);
    }
  });
};

// Auto-enhance shop cards when grid changes
function _enhanceShopCards() {
  document.querySelectorAll('.shop-card').forEach(cardEl => {
    const cid  = cardEl.dataset.id; if (!cid) return;
    const card = typeof getCard === 'function' ? getCard(cid) : null; if (!card) return;
    if (cardEl.querySelector('.axc-shop-ptcl')) return; // already done

    const ptcl = document.createElement('canvas');
    ptcl.className = 'axc-shop-ptcl';
    ptcl.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:2;border-radius:10px;';
    cardEl.style.position = 'relative';
    cardEl.insertBefore(ptcl, cardEl.firstChild);

    requestAnimationFrame(() => {
      if (!ptcl.offsetWidth) return;
      const pa = new CardParticles(ptcl, card);
      pa.start();
      if (APP?.shopCanvasAnims) APP.shopCanvasAnims.set('shop-ptcl-' + card.id, pa);
    });
  });
}

if (typeof MutationObserver !== 'undefined') {
  const shopObs = new MutationObserver(_enhanceShopCards);
  requestAnimationFrame(() => {
    const grid = document.getElementById('shop-card-grid');
    if (grid) shopObs.observe(grid, { childList: true });
  });
}

// ─────────────────────────────────────────────────────────────
// DECK REVIEW — full mini cards instead of text chips
// ─────────────────────────────────────────────────────────────
window.buildDeckReview = function () {
  const grid = document.getElementById('deck-review-grid'); if (!grid) return;
  grid.innerHTML = '';
  grid.style.cssText += ';grid-template-columns:repeat(auto-fill,minmax(88px,1fr));gap:10px;padding:10px;';

  const save = APP?.save || AxiumSave.getOrCreate();
  save.deck.forEach((id, i) => {
    const card = typeof getCard === 'function' ? getCard(id) : null; if (!card) return;
    const renderer = new CardRenderer(card, { size: 'sm', showTooltip: true, animDelay: i * 45 });
    renderer.el.style.cursor = 'default';
    renderer.el.addEventListener('contextmenu', e => { e.preventDefault(); openExpandModal(card); });
    let pt;
    renderer.el.addEventListener('pointerdown', () => { pt = setTimeout(() => openExpandModal(card), 600); });
    renderer.el.addEventListener('pointerup',   () => clearTimeout(pt));
    renderer.el.addEventListener('pointermove', () => clearTimeout(pt));
    grid.appendChild(renderer.el);
  });
};
