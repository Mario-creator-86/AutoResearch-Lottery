// --- State ---
let experiments = [];
let sortCol = 0, sortAsc = false;  // Default sort by experiment number descending (most recent first)
let chart = null;
let evtSource = null;
let lotteries = [];
const experimentsCache = {};
let lastChartDataHash = '';
let currentLottery = 'multi_multi_pl';
let currentLotteryViewToken = 0;
const LOTTERY_RULE_FALLBACKS = {
  multi_multi_pl: { ticket_size: 10, min_pick_count: 1, max_pick_count: 10, numbers_drawn: 20, bonus_count: 0, bonus_range: null, min_bonus_pick_count: 0, max_bonus_pick_count: 0, number_range: 80, baseline: 2.5, name: 'Multi Multi (Poland)' },
  lotto_pl: { ticket_size: 6, min_pick_count: 6, max_pick_count: 12, numbers_drawn: 6, bonus_count: 0, bonus_range: null, min_bonus_pick_count: 0, max_bonus_pick_count: 0, number_range: 49, baseline: 0.734694, name: 'Lotto (Poland)' },
  mini_lotto_pl: { ticket_size: 5, min_pick_count: 5, max_pick_count: 12, numbers_drawn: 5, bonus_count: 0, bonus_range: null, min_bonus_pick_count: 0, max_bonus_pick_count: 0, number_range: 42, baseline: 0.595238, name: 'Mini Lotto (Poland)' },
  loto_6_49_ro: { ticket_size: 6, min_pick_count: 6, max_pick_count: 10, numbers_drawn: 6, bonus_count: 0, bonus_range: null, min_bonus_pick_count: 0, max_bonus_pick_count: 0, number_range: 49, baseline: 0.734694, name: 'Loto 6/49 (Romania)' },
  loto_5_40_ro: { ticket_size: 5, min_pick_count: 5, max_pick_count: 10, numbers_drawn: 6, bonus_count: 0, bonus_range: null, min_bonus_pick_count: 0, max_bonus_pick_count: 0, number_range: 40, baseline: 0.75, name: 'Loto 5/40 (Romania)' },
  joker_ro: { ticket_size: 5, min_pick_count: 5, max_pick_count: 10, numbers_drawn: 5, bonus_count: 1, bonus_range: 20, min_bonus_pick_count: 1, max_bonus_pick_count: 1, number_range: 45, baseline: 0.555556, name: 'Joker (Romania)' },
  eurojackpot: { ticket_size: 5, min_pick_count: 5, max_pick_count: 12, numbers_drawn: 5, bonus_count: 2, bonus_range: 12, min_bonus_pick_count: 2, max_bonus_pick_count: 12, number_range: 50, baseline: 0.5, name: 'EuroJackpot' },
  powerball_us: { ticket_size: 5, min_pick_count: 5, max_pick_count: 5, numbers_drawn: 5, bonus_count: 1, bonus_range: 26, min_bonus_pick_count: 1, max_bonus_pick_count: 1, number_range: 69, baseline: 0.362319, name: 'Powerball (US)' },
  mega_millions_us: { ticket_size: 5, min_pick_count: 5, max_pick_count: 5, numbers_drawn: 5, bonus_count: 1, bonus_range: 24, min_bonus_pick_count: 1, max_bonus_pick_count: 1, number_range: 70, baseline: 0.357143, name: 'Mega Millions (US)' },
};
let currentLotteryMeta = {
  baseline: 2.5,
  name: 'Multi Multi (Poland)',
  ticket_size: 10,
  numbers_drawn: 20,
  min_pick_count: 1,
  max_pick_count: 10,
  number_range: 80,
  bonus_count: 0,
  bonus_range: null,
  min_bonus_pick_count: 0,
  max_bonus_pick_count: 0,
};
let currentMainTab = 'strategy';
let historyState = { groups: [], today_count: 0, today_remaining: 20, today_signatures: [] };
let lastGeneratedPayload = null;
let lastGeneratedLotteryId = null;

function normalizeLotteryMeta(meta) {
  const fallback = LOTTERY_RULE_FALLBACKS[meta?.id] || {};
  const ticketSize = meta?.ticket_size ?? fallback.ticket_size ?? 5;
  const numbersDrawn = meta?.numbers_drawn ?? fallback.numbers_drawn ?? 0;
  const numberRange = meta?.number_range ?? fallback.number_range ?? 1;
  const baseline = meta?.baseline ?? fallback.baseline ?? ((ticketSize * numbersDrawn) / Math.max(numberRange, 1));
  return {
    ...fallback,
    ...meta,
    baseline,
    ticket_size: ticketSize,
    numbers_drawn: numbersDrawn,
    min_pick_count: meta?.min_pick_count ?? fallback.min_pick_count ?? ticketSize ?? 1,
    max_pick_count: meta?.max_pick_count ?? fallback.max_pick_count ?? ticketSize ?? 10,
    bonus_count: meta?.bonus_count ?? fallback.bonus_count ?? 0,
    bonus_range: meta?.bonus_range ?? fallback.bonus_range ?? null,
    min_bonus_pick_count: meta?.min_bonus_pick_count ?? fallback.min_bonus_pick_count ?? meta?.bonus_count ?? 0,
    max_bonus_pick_count: meta?.max_bonus_pick_count ?? fallback.max_bonus_pick_count ?? meta?.bonus_count ?? 0,
  };
}

function generatorRuleText(meta) {
  const simpleMain = `${meta.ticket_size} din ${meta.number_range}`;
  const systemMain = meta.max_pick_count > meta.ticket_size || meta.min_pick_count < meta.ticket_size
    ? `${meta.min_pick_count}-${meta.max_pick_count} din ${meta.number_range}`
    : null;
  if (!meta.bonus_count || !meta.bonus_range) {
    return systemMain
      ? `<strong>Reguli:</strong> Simplu ${simpleMain}. Sistem ${systemMain}. Max 20 bilete.`
      : `<strong>Reguli:</strong> ${simpleMain}. Max 20 bilete.`;
  }
  const simpleBonus = `${meta.bonus_count} din ${meta.bonus_range}`;
  const systemBonus = meta.max_bonus_pick_count > meta.bonus_count || meta.min_bonus_pick_count < meta.bonus_count
    ? `${meta.min_bonus_pick_count}-${meta.max_bonus_pick_count} din ${meta.bonus_range}`
    : null;
  if (systemMain || systemBonus) {
    return `<strong>Reguli:</strong> Simplu ${simpleMain} + ${simpleBonus}. Sistem ${meta.min_pick_count}-${meta.max_pick_count} din ${meta.number_range} + ${meta.min_bonus_pick_count}-${meta.max_bonus_pick_count} din ${meta.bonus_range}. Max 20 bilete.`;
  }
  return `<strong>Reguli:</strong> ${simpleMain} + ${simpleBonus}. Max 20 bilete.`;
}

function multiMultiMinimumHits(pickedCount) {
  const thresholds = {
    1: 1,
    2: 2,
    3: 2,
    4: 2,
    5: 3,
    6: 3,
    7: 3,
    8: 4,
    9: 4,
    10: 4,
  };
  return thresholds[pickedCount] || 99;
}

function minimumWinHint(meta) {
  if (meta.id === 'multi_multi_pl') {
    const picked = parseInt(document.getElementById('nums-per-ticket')?.value || meta.ticket_size, 10);
    return `Min castig: ${multiMultiMinimumHits(picked)} hit${multiMultiMinimumHits(picked) === 1 ? '' : 's'} la ${picked} numere jucate`;
  }
  if (meta.id === 'lotto_pl' || meta.id === 'mini_lotto_pl') return 'Min castig: 3 hits';
  if (meta.id === 'loto_6_49_ro') return 'Min castig: 3 hits';
  if (meta.id === 'loto_5_40_ro') return 'Min castig: 4 hits';
  if (meta.id === 'joker_ro') return 'Min castig: 3+0 sau 1+1';
  if (meta.id === 'eurojackpot') return 'Min castig: 3+0 sau 2+1 sau 1+2';
  if (meta.id === 'powerball_us') return 'Min castig: Powerball';
  if (meta.id === 'mega_millions_us') return 'Min castig: Mega Ball';
  return 'Min castig conform regulilor loteriei';
}

function refreshGeneratorRules() {
  document.getElementById('generator-rules').innerHTML = `${generatorRuleText(currentLotteryMeta)}<br><span>${minimumWinHint(currentLotteryMeta)}</span>`;
}

function updateNumsPerTicketLabel(value) {
  document.getElementById('nums-per-ticket-val').textContent = value;
  refreshGeneratorRules();
}

function updateBonusPerTicketLabel(value) {
  document.getElementById('bonus-per-ticket-val').textContent = value;
  refreshGeneratorRules();
}

function syncRangePresentation(sliderId, valueId, fixedId) {
  const slider = document.getElementById(sliderId);
  const value = document.getElementById(valueId);
  const fixed = document.getElementById(fixedId);
  if (!slider || !value || !fixed) return;
  const isFixed = String(slider.min) === String(slider.max);
  slider.style.display = isFixed ? 'none' : '';
  value.style.display = isFixed ? 'none' : '';
  fixed.style.display = isFixed ? 'inline-flex' : 'none';
  fixed.textContent = isFixed ? `Fix ${slider.value}` : '';
}

async function fetchJsonWithRetry(url, options = {}, attempts = 2, delayMs = 900) {
  let lastError = null;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const resp = await fetch(url, options);
      const data = await resp.json().catch(() => ({}));
      return { resp, data };
    } catch (error) {
      lastError = error;
      if (attempt < attempts) {
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }
  throw lastError || new Error('Request failed');
}

function evaluateTicketOutcome(lotteryId, mainHits, bonusHits, mainCount) {
  if (lotteryId === 'multi_multi_pl') {
    const minHits = multiMultiMinimumHits(mainCount);
    return {
      won: mainHits >= minHits,
      reason: `${mainHits} hit${mainHits === 1 ? '' : 's'} / min ${minHits}`,
    };
  }
  if (lotteryId === 'lotto_pl' || lotteryId === 'mini_lotto_pl') {
    return { won: mainHits >= 3, reason: `${mainHits} hit${mainHits === 1 ? '' : 's'} / min 3` };
  }
  if (lotteryId === 'loto_6_49_ro') {
    return { won: mainHits >= 3, reason: `${mainHits} hit${mainHits === 1 ? '' : 's'} / min 3` };
  }
  if (lotteryId === 'loto_5_40_ro') {
    return { won: mainHits >= 4, reason: `${mainHits} hit${mainHits === 1 ? '' : 's'} / min 4` };
  }
  if (lotteryId === 'joker_ro') {
    return { won: mainHits >= 3 || (mainHits >= 1 && bonusHits >= 1), reason: `${mainHits}+${bonusHits} / min 3+0 or 1+1` };
  }
  if (lotteryId === 'eurojackpot') {
    const won = mainHits >= 3 || (mainHits >= 2 && bonusHits >= 1) || (mainHits >= 1 && bonusHits >= 2);
    return { won, reason: `${mainHits}+${bonusHits} / min 3+0, 2+1, 1+2` };
  }
  if (lotteryId === 'powerball_us') {
    return { won: bonusHits >= 1, reason: bonusHits >= 1 ? 'Powerball hit' : 'No Powerball' };
  }
  if (lotteryId === 'mega_millions_us') {
    return { won: bonusHits >= 1, reason: bonusHits >= 1 ? 'Mega Ball hit' : 'No Mega Ball' };
  }
  if (lotteryId === 'kaskada_pl') {
    return { won: mainHits >= 8, reason: `${mainHits} hit${mainHits === 1 ? '' : 's'} / min 8` };
  }
  return { won: false, reason: `${mainHits} hit${mainHits === 1 ? '' : 's'}${bonusHits ? `, ${bonusHits} bonus` : ''}` };
}

function apiPath(path) {
  return `/api/${currentLottery}${path}`;
}

function apiPathFor(lotteryId, path) {
  return `/api/${lotteryId}${path}`;
}

function isActiveLotteryView(lotteryId, viewToken) {
  return lotteryId === currentLottery && viewToken === currentLotteryViewToken;
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function ticketSignature(main, bonus = []) {
  return `${[...main].sort((a, b) => a - b).join(',')}|${[...bonus].sort((a, b) => a - b).join(',')}`;
}

function ticketSaveKey(main, bonus = [], drawDate = '', drawId = '', drawTime = '') {
  return `${drawDate || ''}|${drawId || ''}|${drawTime || ''}|${ticketSignature(main, bonus)}`;
}

// --- Draw time selector ---
let _selectedDrawTime = null;

function updateDrawTimeSelect(times) {
  const control = document.getElementById('draw-time-control');
  const select = document.getElementById('draw-time-select');
  if (!control || !select) return;
  if (!times || times.length <= 1) {
    control.style.display = 'none';
    _selectedDrawTime = times && times.length === 1 ? times[0] : null;
    return;
  }
  control.style.display = '';
  select.innerHTML = times.map((t, i) => `<option value="${t}" ${i === 0 ? 'selected' : ''}>${t}</option>`).join('');
  _selectedDrawTime = times[0];
}

function onDrawTimeChange() {
  const select = document.getElementById('draw-time-select');
  _selectedDrawTime = select ? select.value : null;
}

function getSelectedDrawTime() {
  // Priority: user-selected draw time > first available from payload > last draw time
  if (_selectedDrawTime) return _selectedDrawTime;
  if (lastGeneratedPayload?.next_draw_times?.length) return lastGeneratedPayload.next_draw_times[0];
  return lastGeneratedPayload?.last_draw_time || null;
}

function switchMainTab(tab) {
  currentMainTab = tab;
  const strategyView = document.getElementById('strategy-view');
  const historyView = document.getElementById('history-view');
  const aiView = document.getElementById('ai-view');
  const strategyTab = document.getElementById('tab-strategy');
  const historyTab = document.getElementById('tab-history');
  const aiTab = document.getElementById('tab-ai');
  const chartCard = document.getElementById('chart-card');
  const tableCard = document.getElementById('table-card');
  const walkforwardPanel = document.getElementById('walkforward-panel');
  const strategyActive = tab === 'strategy';
  const aiActive = tab === 'ai';
  strategyView.style.display = strategyActive ? '' : 'none';
  historyView.style.display = (!strategyActive && !aiActive) ? '' : 'none';
  if (aiView) aiView.style.display = aiActive ? '' : 'none';
  if (chartCard) chartCard.style.display = strategyActive ? '' : 'none';
  if (tableCard) tableCard.style.display = strategyActive ? '' : 'none';
  if (walkforwardPanel) walkforwardPanel.style.display = strategyActive ? '' : 'none';
  strategyTab.classList.toggle('active', strategyActive);
  historyTab.classList.toggle('active', !strategyActive && !aiActive);
  if (aiTab) aiTab.classList.toggle('active', aiActive);
  if (!strategyActive && !aiActive) {
    loadHistory(currentLottery, currentLotteryViewToken).catch(err => console.error('Failed to load history:', err));
    loadAITicketHistory(currentLottery, currentLotteryViewToken);
  }
  if (aiActive) {
    loadAITickets(currentLottery);
    loadLedger();
    loadAIPending();
  }
}

// Debounce flag for loadHistory to prevent rapid successive calls
let loadHistoryDebounceTimer = null;

async function loadHistory(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  if (loadHistoryDebounceTimer) {
    clearTimeout(loadHistoryDebounceTimer);
  }

  return new Promise((resolve) => {
    loadHistoryDebounceTimer = setTimeout(async () => {
      loadHistoryDebounceTimer = null;

      const now = Date.now();
      if (now - lastHistoryFetchTime < 5000) {
        if (currentMainTab !== 'history') {
          resolve();
          return;
        }
      }

      const groupsEl = document.getElementById('history-groups');
      const showLoading = !historyState || !historyState.groups || historyState.groups.length === 0;
      if (showLoading && groupsEl) {
        groupsEl.innerHTML = '<div class="history-loading">Se încarcă History...</div>';
      }

      try {
        const { data: payload } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/history'), {}, 3, 500);
        if (!isActiveLotteryView(lotteryId, viewToken)) {
          resolve();
          return;
        }

        lastHistoryFetchTime = Date.now();
        let newDataHash = null;
        try {
          newDataHash = JSON.stringify(payload.groups || []);
        } catch (e) {
          newDataHash = "hash_failed";
        }

        if (!historyState || newDataHash !== lastHistoryHash) {
          historyState = payload;
          renderHistory();
          lastHistoryHash = newDataHash;

          if (lastGeneratedPayload && lastGeneratedLotteryId === lotteryId) {
            renderGeneratedTickets(lastGeneratedPayload);
          }
        } else {
          // Data unchanged — make sure loading spinner is cleared
          renderHistory();
        }

        resolve();
      } catch (e) {
        if (isActiveLotteryView(lotteryId, viewToken)) {
          console.error('Failed to load history:', e);
          if (groupsEl) groupsEl.innerHTML = '<div class="history-error">Eroare la încărcarea History. Încearcă din nou.</div>';
        }
        resolve();
      }
    }, 300);
  });
}

async function refreshHistoryManually() {
  const refreshBtn = document.getElementById('history-refresh-btn');
  if (!refreshBtn) return;

  const originalContent = refreshBtn.innerHTML;
  refreshBtn.disabled = true;
  refreshBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin"><path d="M23 4v6h-6"></path><path d="M1 20v-6h6"></path><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>';

  try {
    const response = await fetch(`${apiPathFor(currentLottery, '/history')}?bypass_cache=true`);
    const payload = await response.json();
    historyState = payload;
    renderHistory();
    refreshBtn.style.color = '#4fb650';

    setTimeout(() => {
      refreshBtn.innerHTML = originalContent;
      refreshBtn.style.color = '';
      refreshBtn.disabled = false;
    }, 1000);
  } catch (error) {
    console.error('Manual refresh failed:', error);
    const groupsEl = document.getElementById('history-groups');
    if (groupsEl) {
      groupsEl.innerHTML = '<div class="history-error">Manual refresh failed. Please try again.</div>';
    }
    refreshBtn.style.color = '#f85149';

    setTimeout(() => {
      refreshBtn.innerHTML = originalContent;
      refreshBtn.style.color = '';
      refreshBtn.disabled = false;
    }, 2000);
  }
}

// --- Toast notification ---
function showToast(message, type = 'info') {
  const existing = document.getElementById('toast-container');
  if (!existing) {
    const container = document.createElement('div');
    container.id = 'toast-container';
    container.style.cssText = 'position:fixed;bottom:20px;right:20px;z-index:9999;display:flex;flex-direction:column;gap:8px;pointer-events:none;';
    document.body.appendChild(container);
  }
  const container = document.getElementById('toast-container');
  const toast = document.createElement('div');
  const colors = {
    success: 'background:rgba(63,185,80,0.95);color:#fff;',
    error: 'background:rgba(248,81,73,0.95);color:#fff;',
    info: 'background:rgba(88,166,255,0.95);color:#fff;',
  };
  toast.style.cssText = `${colors[type] || colors.info}padding:10px 18px;border-radius:8px;font-size:14px;font-weight:600;font-family:var(--mono);box-shadow:0 4px 12px rgba(0,0,0,0.3);opacity:0;transform:translateY(10px);transition:opacity 0.3s,transform 0.3s;pointer-events:auto;`;
  toast.textContent = message;
  container.appendChild(toast);
  requestAnimationFrame(() => { toast.style.opacity = '1'; toast.style.transform = 'translateY(0)'; });
  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(10px)';
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

// --- Scrape draw results from official sources ---
async function scrapeDrawResults() {
  const btn = document.getElementById('scrape-draws-btn');
  const statusDot = document.getElementById('scrape-status');
  if (!btn || btn.disabled) return;

  const originalHTML = btn.innerHTML;
  btn.disabled = true;
  if (statusDot) { statusDot.style.display = 'block'; statusDot.style.background = '#f0ad4e'; }

  try {
    const resp = await fetch(apiPathFor(currentLottery, '/scrape-draws'), { method: 'POST' });
    const data = await resp.json();

    if (!resp.ok) {
      throw new Error(data.error || 'Scraping failed');
    }

    // Show result
    if (statusDot) { statusDot.style.background = '#4fb650'; }
    const msg = data.new_draws > 0
      ? `Updated! ${data.new_draws} new draw(s) added.`
      : 'No new draws found.';
    btn.title = msg;

    // Refresh history to reflect updated draws
    await refreshHistoryManually();

    setTimeout(() => {
      btn.innerHTML = originalHTML;
      btn.title = 'Update draw results from official sources';
      btn.disabled = false;
      if (statusDot) { statusDot.style.display = 'none'; }
    }, 3000);
  } catch (error) {
    console.error('Scrape failed:', error);
    if (statusDot) { statusDot.style.background = '#f85149'; }
    btn.title = `Error: ${error.message}`;
    setTimeout(() => {
      btn.innerHTML = originalHTML;
      btn.title = 'Update draw results from official sources';
      btn.disabled = false;
      if (statusDot) { statusDot.style.display = 'none'; }
    }, 3000);
  }
}

// --- Load best strategy for badge ---
let _cachedBestStrategy = {};
async function loadBestStrategy(lotteryId = currentLottery) {
  try {
    const { data } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/best-strategy'), {}, 3, 500);
    _cachedBestStrategy[lotteryId] = data;
    const badge = document.getElementById('strategy-badge');
    if (badge && data && data.name) {
      badge.textContent = 'Strategy: ' + data.name;
    }
    // linia „Ultima extragere / Următoarea / Numere" — la încărcare și la
    // schimbarea loteriei, nu doar după apăsarea „Generează Numere"
    if (data) renderLastDrawLine(data);
    return data;
  } catch (e) {
    console.error('Failed to load best strategy:', e);
    return null;
  }
}

function renderLastDrawLine(data) {
  const lastDrawEl = document.getElementById('last-draw-date');
  if (lastDrawEl) lastDrawEl.textContent = data.last_draw_date || '--';
  const nextDrawEl = document.getElementById('next-draw-date');
  if (nextDrawEl) nextDrawEl.textContent = data.next_draw_date || '--';
  const numsEl = document.getElementById('last-draw-nums');
  if (numsEl) {
    const main = data.last_draw_numbers || [];
    const bonus = data.last_draw_bonus || [];
    const pad = n => String(n).padStart(2, '0');
    numsEl.textContent = main.length
      ? main.map(pad).join(' · ') + (bonus.length ? `  (+bonus: ${bonus.map(pad).join(', ')})` : '')
      : '--';
  }
}

// --- Walkforward Validation ---
let _wfData = null;

async function loadWalkforwardResults() {
  loadRealityCheck();
  try {
    const { resp, data } = await fetchJsonWithRetry(apiPath('/walkforward'));
    if (resp.ok) {
      _wfData = data;
      renderWalkforwardPanel(data);
    } else {
      _wfData = null;
      renderWalkforwardPanel(null);
    }
  } catch (e) {
    renderWalkforwardPanel(null);
  }
}

const REALITY_COLORS = {
  significant: 'var(--green)',
  suggestive: 'var(--yellow)',
  noise: 'var(--text2)',
  insufficient: 'var(--text2)',
};
const REALITY_LABELS = {
  significant: 'EDGE REAL',
  suggestive: 'SUGESTIV',
  noise: 'ZGOMOT',
  insufficient: 'DATE PUȚINE',
};

async function loadRealityCheck(lotteryId = currentLottery) {
  const content = document.getElementById('reality-content');
  if (!content) return;
  try {
    const { resp, data } = await fetchJsonWithRetry(apiPath('/reality-check'));
    if (lotteryId !== currentLottery) return;
    if (!resp.ok) {
      content.innerHTML = `<div style="color:var(--red);padding:4px 0;">${escapeHtml(data.error || 'Reality check indisponibil')}</div>`;
      return;
    }
    renderRealityCheck(data);
  } catch (e) {
    content.innerHTML = `<div style="color:var(--text2);padding:4px 0;">Reality check indisponibil: ${escapeHtml(e.message)}</div>`;
  }
}

function realityTile(title, stats, mainValue, sub) {
  const verdict = (stats && stats.verdict) || 'insufficient';
  const color = REALITY_COLORS[verdict] || 'var(--text2)';
  return `
    <div style="background:var(--bg3);border-radius:8px;padding:10px 12px;">
      <div style="display:flex;justify-content:space-between;gap:6px;align-items:center;">
        <span style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;">${escapeHtml(title)}</span>
        <span style="font-size:9px;font-weight:800;color:${color};border:1px solid ${color};border-radius:6px;padding:1px 5px;">${escapeHtml((stats && stats.label) || REALITY_LABELS[verdict] || verdict)}</span>
      </div>
      <div style="font-size:18px;font-weight:800;color:var(--blue);margin-top:4px;">${mainValue}</div>
      <div style="font-size:10px;color:var(--text2);line-height:1.4;">${sub}</div>
    </div>`;
}

function renderRealityCheck(data) {
  const content = document.getElementById('reality-content');
  if (!content || !data) return;
  const fmt = (v, d = 4) => (typeof v === 'number' ? v.toFixed(d) : '--');
  const signed = (v, d = 2) => (typeof v === 'number' ? `${v >= 0 ? '+' : ''}${v.toFixed(d)}` : '--');
  const tiles = [];

  const bt = data.backtest;
  if (bt) {
    tiles.push(realityTile('Backtest (best)', bt, fmt(bt.metric),
      `aleator ${fmt(bt.baseline)} · edge ${signed(bt.edge_pct)}%<br>z=${signed(bt.z)} vs max. zgomot ≈${fmt(bt.noise_max_z, 2)}<br>` +
      `${bt.n_experiments} config. testate · p ajustat ${fmt(bt.p_adjusted, 3)}`));
  }
  const wf = data.walkforward;
  if (wf) {
    tiles.push(realityTile('Walkforward', wf, fmt(wf.metric),
      `z=${signed(wf.z)} pe ${wf.n_draws} extrageri<br><span title="${escapeHtml(wf.caveat || '')}">⚠ parametri aleși pe tot istoricul</span>`));
  }
  const ft = data.forward_test;
  if (ft && !ft.error) {
    tiles.push(realityTile('Forward test (înghețat)', ft,
      ft.n_draws ? fmt(ft.metric) : '--',
      ft.n_draws
        ? `aleator ${fmt(ft.baseline)} · z=${signed(ft.z)}<br>${ft.n_draws} extrageri după ${escapeHtml(ft.frozen_after || '')}<br>singurul test fără selecție`
        : `parametri înghețați ${escapeHtml(ft.frozen_after || '—')}<br>se evaluează pe extragerile următoare`));
  }
  const pr = data.ai_tickets_prospective || {};
  tiles.push(realityTile('Bilete AI reale', pr,
    pr.draws_checked ? fmt(pr.mean_hits_per_ticket, 3) : '--',
    pr.draws_checked
      ? `lovituri/bilet vs aleator ${fmt(pr.baseline, 3)}<br>${pr.draws_checked} extrageri verificate · z=${signed(pr.z)}`
      : 'încă nicio extragere verificată'));
  const pf = data.latest_portfolio;
  if (pf) {
    const gain = (pf.p_any_prize_after - pf.p_any_prize_before) * 100;
    tiles.push(realityTile('Portofoliu bilete', { verdict: gain > 0 ? 'significant' : 'insufficient' },
      `${fmt(pf.p_any_prize_after * 100, 1)}%`,
      `șansa ca măcar un bilet să prindă premiu (≥${pf.min_prize_hits} numere)<br>` +
      `înainte ${fmt(pf.p_any_prize_before * 100, 1)}% · suprapunere ${fmt(pf.mean_overlap_before, 2)}→${fmt(pf.mean_overlap_after, 2)}`));
  }

  const scan = data.anomaly_scan;
  if (scan && scan.tests && scan.strongest_test) {
    const strongest = scan.tests[scan.strongest_test] || {};
    tiles.push(realityTile('Anomalii în extrageri', scan,
      `p=${fmt(scan.p_adjusted, 3)}`,
      `7 teste exacte pe ${scan.n_draws} extrageri<br>cel mai puternic: ${escapeHtml(scan.strongest_test)} (p brut ${fmt(strongest.p_value, 4)})<br>` +
      (scan.exploitable
        ? `abatere stabilă: ${(scan.most_frequent || []).map(item => item.number).join(', ')}`
        : `plafon dacă ar fi reală: ${signed(scan.face_value_edge_pct)}% lovituri`)));
  }
  const advice = data.play_advice;
  if (advice) {
    const adviceVerdict = { favorable: 'significant', near: 'suggestive' }[advice.verdict] || 'noise';
    const money = v => (typeof v === 'number' ? v.toLocaleString('ro-RO', { maximumFractionDigits: 0 }) : '--');
    const adviceLabel = { favorable: 'PESTE PRAG', near: 'APROAPE DE PRAG', fixed: 'PREMII FIXE', unknown: 'REPORT NECUNOSCUT' }[advice.verdict] || 'SUB COST';
    tiles.push(realityTile('Valoarea biletului', { verdict: adviceVerdict, label: adviceLabel },
      `${fmt(advice.current_return_pct ?? advice.typical_return_pct, 0)}%`,
      `din preț se întoarce în medie${advice.current_jackpot ? ` la reportul de ${money(advice.current_jackpot)} ${escapeHtml(advice.currency)}` : ' (tipic)'}<br>` +
      (advice.breakeven_jackpot
        ? `prag de rentabilitate: ${money(advice.breakeven_jackpot)} ${escapeHtml(advice.currency)}<br>șansă premiu mare 1 la ${money(advice.jackpot_odds)}`
        : 'premii fixe — nu există moment mai bun de joc')));
  }

  content.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin-bottom:10px;">${tiles.join('')}</div>
    <div style="font-size:11px;color:var(--text2);line-height:1.5;">
      ${escapeHtml((bt && bt.verdict_text) ? `Backtest: ${bt.verdict_text}.` : '')}
      Extragerile sunt independente și numerele ies uniform, deci nicio strategie nu crește șansa unui bilet.
      Ce se poate îmbunătăți real: <strong>acoperirea</strong> portofoliului (bilete puțin suprapuse) și
      evitarea combinațiilor <strong>populare</strong> (premii împărțite la mai puțini câștigători).
    </div>`;
}

async function runWalkforward() {
  const btn = document.getElementById('wf-run-btn');
  const content = document.getElementById('walkforward-content');
  if (btn) { btn.disabled = true; btn.textContent = 'Running...'; }
  content.innerHTML = '<div style="color:var(--yellow);padding:4px 0;">Running walkforward validation (5 folds)... This may take 30-60 seconds.</div>';
  try {
    const { resp, data } = await fetchJsonWithRetry(apiPath('/run-walkforward'), { method: 'POST' }, 1, 0);
    if (resp.ok) {
      _wfData = data;
      renderWalkforwardPanel(data);
    } else {
      content.innerHTML = `<div style="color:var(--red);padding:4px 0;">Error: ${escapeHtml(data.error || 'Unknown error')}</div>`;
    }
  } catch (e) {
    content.innerHTML = `<div style="color:var(--red);padding:4px 0;">Failed: ${escapeHtml(e.message)}</div>`;
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Run Walkforward'; }
  }
}

function renderWalkforwardPanel(data) {
  const content = document.getElementById('walkforward-content');
  if (!content) return;
  if (!data || !data.wf_score) {
    content.innerHTML = '<div style="color:var(--text2);padding:4px 0;">No walkforward data. Click <strong>Run Walkforward</strong> to validate current strategy.</div>';
    return;
  }

  const gapPct = data.backtest_score > 0 ? ((data.gap / data.backtest_score) * 100).toFixed(2) : '0.00';
  const gapColor = data.gap > 0.05 ? 'var(--red)' : data.gap > 0.01 ? 'var(--yellow)' : 'var(--green)';
  const stabilityColor = data.stability >= 0.9 ? 'var(--green)' : data.stability >= 0.7 ? 'var(--yellow)' : 'var(--red)';
  const edgeColor = data.baseline_edge > 0 ? 'var(--green)' : 'var(--red)';
  const folds = data.folds || [];

  content.innerHTML = `
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-bottom:14px;">
      <div style="background:var(--bg3);border-radius:8px;padding:10px 12px;">
        <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;">WF Score</div>
        <div style="font-size:18px;font-weight:800;color:var(--blue);margin-top:4px;">${data.wf_score.toFixed(6)}</div>
        <div style="font-size:10px;color:var(--text2);">vs backtest ${data.backtest_score.toFixed(6)}</div>
      </div>
      <div style="background:var(--bg3);border-radius:8px;padding:10px 12px;">
        <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;">Gap (Overfit)</div>
        <div style="font-size:18px;font-weight:800;color:${gapColor};margin-top:4px;">${data.gap >= 0 ? '+' : ''}${data.gap.toFixed(6)}</div>
        <div style="font-size:10px;color:${gapColor};">${gapPct}% ${data.gap > 0.01 ? 'overfitting' : 'healthy'}</div>
      </div>
      <div style="background:var(--bg3);border-radius:8px;padding:10px 12px;">
        <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;">Stability</div>
        <div style="font-size:18px;font-weight:800;color:${stabilityColor};margin-top:4px;">${(data.stability * 100).toFixed(1)}%</div>
        <div style="font-size:10px;color:${stabilityColor};">${data.stability >= 0.9 ? 'Very stable' : data.stability >= 0.7 ? 'Moderate' : 'Unstable'}</div>
      </div>
      <div style="background:var(--bg3);border-radius:8px;padding:10px 12px;">
        <div style="font-size:10px;color:var(--text2);text-transform:uppercase;letter-spacing:0.5px;">vs Baseline</div>
        <div style="font-size:18px;font-weight:800;color:${edgeColor};margin-top:4px;">${data.baseline_edge >= 0 ? '+' : ''}${data.baseline_edge.toFixed(6)}</div>
        <div style="font-size:10px;color:${edgeColor};">edge ${data.baseline_edge > 0 ? 'above' : 'below'} random</div>
      </div>
    </div>
    <div style="font-size:11px;color:var(--text2);margin-bottom:6px;">
      Strategy: <span style="color:var(--blue)">${escapeHtml(data.strategy || '--')}</span>
      &nbsp;|&nbsp; ${folds.length} folds &nbsp;|&nbsp; ${data.elapsed_seconds ? data.elapsed_seconds.toFixed(1) + 's' : '--'}
      &nbsp;|&nbsp; ${data.timestamp ? new Date(data.timestamp).toLocaleString() : '--'}
    </div>
    <details>
      <summary style="cursor:pointer;font-size:12px;color:var(--text2);font-weight:700;">Per-Fold Breakdown</summary>
      <table style="width:100%;border-collapse:collapse;margin-top:8px;font-size:11px;">
        <thead>
          <tr style="color:var(--text2);border-bottom:1px solid var(--border);">
            <th style="text-align:left;padding:4px 8px;">Fold</th>
            <th style="text-align:right;padding:4px 8px;">Train</th>
            <th style="text-align:right;padding:4px 8px;">Test</th>
            <th style="text-align:right;padding:4px 8px;">Score</th>
            <th style="text-align:right;padding:4px 8px;">Baseline</th>
            <th style="text-align:right;padding:4px 8px;">Edge</th>
            <th style="text-align:right;padding:4px 8px;">2+ Rate</th>
            <th style="text-align:right;padding:4px 8px;">3+ Rate</th>
          </tr>
        </thead>
        <tbody>
          ${folds.map(f => {
            const edge = f.score - f.baseline;
            const edgeC = edge > 0 ? 'var(--green)' : 'var(--red)';
            return `<tr style="border-bottom:1px solid rgba(48,54,61,0.5);">
              <td style="padding:4px 8px;font-weight:700;">Fold ${f.fold}</td>
              <td style="padding:4px 8px;text-align:right;">${f.train_size}</td>
              <td style="padding:4px 8px;text-align:right;">${f.test_size}</td>
              <td style="padding:4px 8px;text-align:right;font-weight:700;color:var(--blue);">${f.score.toFixed(6)}</td>
              <td style="padding:4px 8px;text-align:right;color:var(--text2);">${f.baseline.toFixed(6)}</td>
              <td style="padding:4px 8px;text-align:right;color:${edgeC};font-weight:700;">${edge >= 0 ? '+' : ''}${edge.toFixed(6)}</td>
              <td style="padding:4px 8px;text-align:right;">${(f.hit_rate_2plus * 100).toFixed(1)}%</td>
              <td style="padding:4px 8px;text-align:right;">${(f.hit_rate_3plus * 100).toFixed(1)}%</td>
            </tr>`;
          }).join('')}
        </tbody>
      </table>
    </details>
  `;
}

async function saveGeneratedTicket(index) {
  if (!lastGeneratedPayload || lastGeneratedLotteryId !== currentLottery) return;
  const ticket = lastGeneratedPayload.tickets?.[index];
  if (!ticket) return;
  const btn = document.getElementById(`save-ticket-${index}`);
  if (btn && btn.disabled) return;
  if (btn) {
    btn.disabled = true;
    btn.textContent = '···';
    btn.title = 'Salvare...';
  }
  try {
    const { resp, data } = await fetchJsonWithRetry(apiPath('/history'), {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        main: ticket.main || [],
        bonus: ticket.bonus || [],
        strategy: lastGeneratedPayload.strategy || null,
        metric: lastGeneratedPayload.metric ?? null,
        target_draw_date: lastGeneratedPayload.next_draw_iso_date || lastGeneratedPayload.last_draw_iso_date || null,
        target_draw_id: lastGeneratedPayload.last_draw_id || null,
        target_draw_time: getSelectedDrawTime(),
      }),
    });
    if (!resp.ok) {
      if (btn) {
        btn.textContent = '+';
        btn.title = 'Salvează';
        btn.disabled = false;
      }
      showToast(data.error || 'Eroare la salvare', 'error');
      return;
    }
    // Success — mark button as saved and show confirmation
    if (btn) {
      btn.textContent = '✓';
      btn.title = 'Salvat!';
      btn.classList.add('saved');
      btn.disabled = true;
    }
    showToast(`Bilet salvat pentru ${lastGeneratedPayload.next_draw_date || lastGeneratedPayload.last_draw_date || 'extragerea următoare'}`, 'success');
    await loadHistory();
  } catch (e) {
    if (btn) {
      btn.textContent = '+';
      btn.title = 'Salvează';
      btn.disabled = false;
    }
    showToast(`Eroare: ${e.message}`, 'error');
  }
}

async function deleteHistoryTicket(ticketId) {
  try {
    const { resp, data } = await fetchJsonWithRetry(apiPath(`/history/${ticketId}`), { method: 'DELETE' });
    if (!resp.ok) {
      alert(data.error || 'Failed to delete ticket');
      return;
    }
    lastHistoryFetchTime = 0;
    const { data: payload } = await fetchJsonWithRetry(
      apiPath(`/history?bypass_cache=true&_t=${Date.now()}`), {}, 3, 500
    );
    if (payload) {
      historyState = payload;
      lastHistoryHash = null;
      renderHistory();
      if (lastGeneratedPayload && lastGeneratedLotteryId === currentLottery) {
        renderGeneratedTickets(lastGeneratedPayload);
      }
    }
  } catch (e) {
    alert(`Error deleting ticket: ${e.message}`);
  }
}

function renderGeneratedTickets(data) {
  const lastDraw = new Set(data.last_draw_numbers || []);
  const lastBonusDraw = new Set(data.last_draw_bonus || []);
  const todaySaveKeys = new Set(historyState.today_save_keys || []);
  const container = document.getElementById('tickets-display');
  const summaryEl = document.getElementById('tickets-summary');

  const evaluated = data.tickets.map((ticket, i) => {
    const main = ticket.main || ticket;
    const bonus = ticket.bonus || [];
    const hits = main.filter(n => lastDraw.has(n)).length;
    const bonusHits = bonus.filter(n => lastBonusDraw.has(n)).length;
    const outcome = evaluateTicketOutcome(currentLotteryMeta.id, hits, bonusHits, main.length);
    return { ticket, originalIndex: i, main, bonus, hits, bonusHits, outcome };
  });

  evaluated.sort((a, b) => {
    if (a.outcome.won !== b.outcome.won) return a.outcome.won ? -1 : 1;
    return (b.hits + b.bonusHits) - (a.hits + a.bonusHits);
  });

  const winCount = evaluated.filter(e => e.outcome.won).length;
  const totalCount = evaluated.length;
  if (totalCount > 0) {
    summaryEl.style.display = '';
    if (winCount > 0) {
      summaryEl.style.background = 'rgba(63,185,80,0.12)';
      summaryEl.style.border = '1px solid rgba(63,185,80,0.3)';
      summaryEl.style.color = 'var(--green)';
      summaryEl.innerHTML = `${winCount} castigatoare din ${totalCount} bilete`;
    } else {
      summaryEl.style.background = 'rgba(248,81,73,0.08)';
      summaryEl.style.border = '1px solid rgba(248,81,73,0.2)';
      summaryEl.style.color = 'var(--red)';
      summaryEl.innerHTML = `0 castigatoare din ${totalCount} bilete`;
    }
  } else {
    summaryEl.style.display = 'none';
  }

  container.innerHTML = evaluated.map(({ ticket, originalIndex: i, main, bonus, hits, bonusHits, outcome }) => {
    const statusClass = outcome.won ? 'win' : 'miss';
    const statusMark = outcome.won ? '✓' : 'X';
    const nums = main.map(n => {
      const hit = lastDraw.has(n);
      const color = hit ? 'var(--green)' : 'var(--red)';
      const weight = hit ? '800' : '400';
      return `<span style="color:${color};font-weight:${weight}">${n}</span>`;
    }).join(' ');
    const bonusHtml = bonus.length ? bonus.map(n => {
      const hit = lastBonusDraw.has(n);
      const color = hit ? 'var(--green)' : 'var(--red)';
      const weight = hit ? '800' : '400';
      return `<span style="color:${color};font-weight:${weight}">${n}</span>`;
    }).join(' ') : '';
    const bonusLabelColor = bonusHits > 0 ? 'var(--green)' : 'var(--text2)';
    const bonusLabelWeight = bonusHits > 0 ? '700' : '400';
    const hitParts = [];
    if (hits > 0) hitParts.push(`${hits} hit${hits > 1 ? 's' : ''}`);
    if (bonusHits > 0) hitParts.push(`${bonusHits} bonus`);
    const hitsLabel = hitParts.length ? hitParts.join(', ') : '0 hits';
    const saved = todaySaveKeys.has(ticketSaveKey(main, bonus, data.next_draw_iso_date || data.last_draw_iso_date, data.last_draw_id, getSelectedDrawTime()));
    return `<div class="ticket-card ${statusClass}">
      <div class="ticket-head">
        <span class="ticket-index">${i + 1}.</span>
        <div class="ticket-actions">
          <button id="save-ticket-${i}" class="icon-btn ${saved ? 'saved' : ''}" type="button" onclick="saveGeneratedTicket(${i})" ${saved ? 'disabled' : ''} title="${saved ? 'Salvat' : 'Salvează'}">${saved ? '✓' : '+'}</button>
          <span class="ticket-status ${statusClass}">${statusMark}</span>
        </div>
      </div>
      <div class="ticket-lines">
        <div class="ticket-line">${nums}</div>
        ${bonus.length ? `<div class="ticket-line"><span class="ticket-bonus-label" style="color:${bonusLabelColor};font-weight:${bonusLabelWeight}">Bonus</span>${bonusHtml}</div>` : ''}
        </div>
      <div class="ticket-foot">
        <span class="ticket-outcome ${statusClass}">${hitsLabel}</span>
        <span class="ticket-reason">${outcome.reason}</span>
      </div>
    </div>`;
  }).join('');
}

function renderHistory() {
  const capacity = document.getElementById('history-capacity');
  const summary = document.getElementById('history-summary');
  const empty = document.getElementById('history-empty');
  const groupsEl = document.getElementById('history-groups');
  const drawCapacity = historyState.capacity || 20;
  capacity.textContent = `${drawCapacity} / draw`;
  const groups = historyState.groups || [];
  const totalTickets = groups.reduce((sum, g) => sum + (g.total_tickets || 0), 0);
  summary.innerHTML = `<strong>${escapeHtml(currentLotteryMeta.name)}</strong>: ${groups.length} extrageri, ${totalTickets} bilete total. Limita: ${drawCapacity} bilete/extragere.`;
  empty.style.display = groups.length ? 'none' : '';
  groupsEl.innerHTML = groups.map(group => {
    const drawMain = (group.draw_numbers || []).join(', ');
    const drawBonus = (group.draw_bonus || []).length ? ` + ${(group.draw_bonus || []).join(', ')}` : '';
    const drawMetaBits = [];
    if (group.target_draw_id) drawMetaBits.push(`#${group.target_draw_id}`);
    if (group.target_draw_time) drawMetaBits.push(group.target_draw_time);
    const drawMeta = drawMetaBits.length ? ` (${drawMetaBits.join(' | ')})` : '';
    const drawLine = group.draw_available
      ? `<strong>Extragere ${escapeHtml(group.target_draw_date || group.saved_date)}${escapeHtml(drawMeta)}:</strong> ${escapeHtml(drawMain)}${escapeHtml(drawBonus)}`
      : `<strong>Extragere ${escapeHtml(group.target_draw_date || group.saved_date)}${escapeHtml(drawMeta)}:</strong> Waiting for draw`;
    const ticketCards = (group.tickets || []).map((ticket, idx) => {
      const main = ticket.main || [];
      const bonus = ticket.bonus || [];
      const mainSet = new Set(group.draw_numbers || []);
      const bonusSet = new Set(group.draw_bonus || []);
      const statusClass = ticket.won === true ? 'win' : (ticket.won === false ? 'miss' : '');
      const statusMark = ticket.won === true ? '✓' : (ticket.won === false ? 'X' : '·');
      const nums = main.map(n => {
        const hit = mainSet.has(n);
        const color = !group.draw_available ? 'var(--text)' : (hit ? 'var(--green)' : 'var(--red)');
        const weight = hit ? '800' : '400';
        return `<span style="color:${color};font-weight:${weight}">${n}</span>`;
      }).join(' ');
      const bonusHtml = bonus.length ? bonus.map(n => {
        const hit = bonusSet.has(n);
        const color = !group.draw_available ? 'var(--blue)' : (hit ? 'var(--green)' : 'var(--red)');
        const weight = hit ? '800' : '400';
        return `<span style="color:${color};font-weight:${weight}">${n}</span>`;
      }).join(' ') : '';
      const bonusLabelColor = !group.draw_available ? 'var(--text2)' : ((ticket.bonus_hits || 0) > 0 ? 'var(--green)' : 'var(--text2)');
      const bonusLabelWeight = (ticket.bonus_hits || 0) > 0 ? '700' : '400';
      const hitsLabel = group.draw_available
        ? `${ticket.main_hits || 0} hit${(ticket.main_hits || 0) === 1 ? '' : 's'}${ticket.bonus_hits ? `, ${ticket.bonus_hits} bonus` : ''}`
        : 'Pending';
      return `<div class="ticket-card ${statusClass}">
        <div class="ticket-head">
          <span class="ticket-index">${idx + 1}.</span>
          <div class="ticket-actions">
            ${String(ticket.id || '').startsWith('ai|') ? '' : `<button class="icon-btn danger" type="button" onclick="deleteHistoryTicket('${ticket.id}')" title="Delete from History">×</button>`}
            <span class="ticket-status ${statusClass}">${statusMark}</span>
          </div>
        </div>
        <div class="ticket-lines">
          <div class="ticket-line">${nums}</div>
          ${bonus.length ? `<div class="ticket-line"><span class="ticket-bonus-label" style="color:${bonusLabelColor};font-weight:${bonusLabelWeight}">Bonus</span>${bonusHtml}</div>` : ''}
          <div class="ticket-line" style="color:var(--text2);font-size:11px;">Saved ${escapeHtml(ticket.saved_at || '')}${ticket.strategy ? ` | ${escapeHtml(ticket.strategy)}` : ''}</div>
        </div>
        <div class="ticket-foot">
          <span class="ticket-outcome ${statusClass}">${escapeHtml(hitsLabel)}</span>
          <span class="ticket-reason">${escapeHtml(ticket.reason || '--')}</span>
        </div>
      </div>`;
    }).join('');
    return `<div class="history-group-card">
      <div class="status-header">
        <span class="status-label">History ${escapeHtml(group.target_draw_date || group.saved_date)}${escapeHtml(drawMeta)}</span>
        <div class="d-flex align-center gap-sm">
          ${group.source === 'ai' ? '<span class="status-badge badge-running">AI</span>' : ''}
          <span class="status-badge ${group.draw_available ? 'badge-finished' : 'badge-stopped'}">${group.total_tickets} / ${group.draw_capacity || drawCapacity} bilete</span>
        </div>
      </div>
      <div class="history-group-meta">
        <span class="history-chip" style="${(group.winning_ticket_count || 0) > 0 ? 'background:rgba(63,185,80,0.15);border-color:rgba(63,185,80,0.4);color:var(--green)' : ''}">${group.winning_ticket_count || 0} castigatoare</span>
        <span class="history-chip">${group.matched_ticket_count || 0} cu hit</span>
        <span class="history-chip">${group.draw_available ? 'draw loaded' : 'pending draw'}</span>
      </div>
      <div class="history-draw-line">${drawLine}</div>
      <div class="tickets-grid">${ticketCards}</div>
    </div>`;
  }).join('');
}

// --- Formatting ---
function fmtDuration(seconds) {
  if (!seconds || seconds <= 0) return '--';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function fmtETA(seconds) {
  if (!seconds || seconds <= 0) return 'Done';
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  if (m > 60) return `~${Math.round(m / 6)}h ${m % 60}m`;
  return m > 0 ? `~${m}m ${s}s` : `~${s}s`;
}

function statusBadgeClass(status) {
  return {
    running: 'badge-running',
    finished: 'badge-finished',
    crashed: 'badge-crashed',
    stopped: 'badge-stopped',
    stalled: 'badge-stalled',
    idle: 'badge-idle',
  }[status] || 'badge-idle';
}

function sidebarProgressPercent(summary, snapshot) {
  if (snapshot?.status === 'running' || snapshot?.status === 'stalled' || snapshot?.status === 'crashed') {
    return Math.max(0, Math.min(100, snapshot.percent || 0));
  }
  if (snapshot?.status === 'finished') return 100;
  if (summary?.status === 'finished') return 100;
  return 0;
}

function sidebarFoot(summary, snapshot) {
  const status = snapshot?.status || summary?.status || 'idle';
  if (status === 'running' || status === 'stalled' || status === 'crashed') {
    return {
      left: `${snapshot?.tested || 0}/${snapshot?.total || '?'}`,
      right: snapshot?.eta_seconds ? fmtETA(snapshot.eta_seconds) : (status === 'crashed' ? 'Stopped' : '--'),
    };
  }
  if (status === 'finished') {
    return { left: `Done`, right: `${summary?.experiment_count || 0} exp` };
  }
  return {
    left: `${summary?.experiment_count || 0} exp`,
    right: summary?.best_metric ? summary.best_metric.toFixed(3) : '--',
  };
}

function renderLotteryRail() {
  const rail = document.getElementById('lottery-rail');
  if (!rail) return;
  rail.innerHTML = lotteries.map(summary => {
    const status = summary.status || 'idle';
    const pct = sidebarProgressPercent(summary, undefined);
    const foot = sidebarFoot(summary, undefined);
    const active = summary.id === currentLottery ? 'active' : '';
    return `
      <button class="lottery-item ${active}" onclick="onLotteryChange('${summary.id}')">
        <div class="lottery-item-head">
          <span class="lottery-item-name">${summary.name}</span>
          <span class="lottery-item-status ${statusBadgeClass(status)}">${status}</span>
        </div>
        <div class="lottery-item-meta">
          <span>Best ${summary.best_metric ? summary.best_metric.toFixed(3) : '--'}</span>
          <span>${summary.draw_count || 0} draws</span>
        </div>
        <div class="lottery-mini-wrap">
          <div class="lottery-mini-bar">
            <div class="lottery-mini-fill status-${status}" style="width:${pct}%"></div>
          </div>
        </div>
        <div class="lottery-item-foot">
          <span>${foot.left}</span>
          <span>${foot.right}</span>
        </div>
      </button>
    `;
  }).join('');
}


async function loadLotteries() {
  const resp = await fetch('/api/lotteries');
  lotteries = (await resp.json()).map(normalizeLotteryMeta);
  const hashLottery = location.hash ? location.hash.slice(1) : '';
  const initial = lotteries.find(l => l.id === hashLottery)?.id || lotteries[0]?.id || 'multi_multi_pl';
  renderLotteryRail();
  await onLotteryChange(initial, false);
}

function resetLotteryViewForSwitch() {
  showAllExperiments = false;
  experiments = experimentsCache[currentLottery] || [];
  historyState = { groups: [], today_count: 0, today_remaining: 20, today_signatures: [] };
  renderTable();
  renderChart();
  const summary = lotteries.find(l => l.id === currentLottery);
  if (summary && summary.status) {
    updateStatus({ status: summary.status });
  } else {
    document.getElementById('strategy-badge').textContent = 'Strategy: loading...';
    document.getElementById('recent-list').innerHTML = '';
    document.getElementById('stat-tested').textContent = '-- / --';
    document.getElementById('stat-best').textContent = '--';
    document.getElementById('stat-best-config').textContent = '';
    document.getElementById('stat-eta').textContent = '--';
    document.getElementById('stat-elapsed').textContent = '--';
    const fill = document.getElementById('progress-fill');
    fill.style.width = '0%';
    fill.textContent = '...';
    const badge = document.getElementById('status-badge');
    badge.className = 'status-badge badge-idle';
    badge.textContent = 'LOADING';
    document.getElementById('status-running').style.display = 'none';
    document.getElementById('status-finished').style.display = 'none';
    const idleMsg = document.getElementById('status-idle-msg');
    idleMsg.style.display = 'block';
    idleMsg.innerHTML = 'Loading lottery data...';
  }
}

async function onLotteryChange(lotteryId, updateHash = true) {
  const viewToken = ++currentLotteryViewToken;
  currentLottery = lotteryId;
  currentLotteryMeta = normalizeLotteryMeta(lotteries.find(l => l.id === lotteryId) || { id: lotteryId });
  lastGeneratedPayload = null;
  lastGeneratedLotteryId = null;
  if (evtSource) {
    evtSource.close();
    evtSource = null;
  }
  document.getElementById('tickets-display').innerHTML = '';
  document.getElementById('tickets-summary').style.display = 'none';
  document.getElementById('last-draw-date').textContent = '--';
  document.getElementById('last-draw-nums').textContent = '--';
  document.getElementById('lottery-label').textContent = `${currentLotteryMeta.name} Experiment`;
  document.getElementById('active-lottery-chip').textContent = currentLotteryMeta.name;
  updateDrawTimeSelect(currentLotteryMeta.draw_times_local || []);
  const numsSlider = document.getElementById('nums-per-ticket');
  numsSlider.min = currentLotteryMeta.min_pick_count || currentLotteryMeta.ticket_size || 1;
  numsSlider.max = currentLotteryMeta.max_pick_count || currentLotteryMeta.ticket_size || 10;
  numsSlider.value = currentLotteryMeta.ticket_size || numsSlider.value;
  updateNumsPerTicketLabel(numsSlider.value);
  syncRangePresentation('nums-per-ticket', 'nums-per-ticket-val', 'nums-per-ticket-fixed');
  const bonusControl = document.getElementById('bonus-control');
  const bonusSlider = document.getElementById('bonus-per-ticket');
  if ((currentLotteryMeta.max_bonus_pick_count || 0) > 0) {
    bonusControl.style.display = 'flex';
    bonusSlider.min = currentLotteryMeta.min_bonus_pick_count || currentLotteryMeta.bonus_count || 0;
    bonusSlider.max = currentLotteryMeta.max_bonus_pick_count || currentLotteryMeta.bonus_count || 0;
    bonusSlider.value = currentLotteryMeta.bonus_count || currentLotteryMeta.min_bonus_pick_count || 0;
    updateBonusPerTicketLabel(bonusSlider.value);
    syncRangePresentation('bonus-per-ticket', 'bonus-per-ticket-val', 'bonus-per-ticket-fixed');
  } else {
    bonusControl.style.display = 'none';
    bonusSlider.value = 0;
    updateBonusPerTicketLabel('0');
    syncRangePresentation('bonus-per-ticket', 'bonus-per-ticket-val', 'bonus-per-ticket-fixed');
  }
  if (updateHash) location.hash = lotteryId;
  resetLotteryViewForSwitch();
  renderLotteryRail();
  const progressPromise = refreshProgress(lotteryId, viewToken);
  const experimentsPromise = loadExperiments(lotteryId, viewToken);
  const historyPromise = loadHistory(lotteryId, viewToken);
  const strategyPromise = loadBestStrategy(lotteryId);
  const aiPromise = loadAIPredictions(lotteryId);
  loadAITicketHistory(lotteryId, viewToken);
  // Walkforward + Reality Check (funcția exista, dar nu era apelată nicăieri)
  loadWalkforwardResults();
  connectSSE(lotteryId, viewToken);
  checkLoopStatus(lotteryId, viewToken);
  await progressPromise;
  await Promise.allSettled([experimentsPromise, historyPromise, strategyPromise, aiPromise]);
}

async function refreshProgress(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  try {
    const { data } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/progress'), {}, 3, 500);
    const summary = lotteries.find(item => item.id === lotteryId);
    if (summary && data?.status) summary.status = data.status;
    renderLotteryRail();
    if (!isActiveLotteryView(lotteryId, viewToken)) return;
    updateStatus(data);
  } catch (e) {
    if (isActiveLotteryView(lotteryId, viewToken)) {
      console.error('Failed to load progress:', e);
    }
  }
}

// --- Load experiments ---
async function loadExperiments(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  if (experimentsCache[lotteryId]) {
    experiments = experimentsCache[lotteryId];
    renderTable();
    renderChart(true);
  }
  try {
    const { data: payload } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/experiments'), {}, 3, 500);
    if (!isActiveLotteryView(lotteryId, viewToken)) return;
    experimentsCache[lotteryId] = payload;
    experiments = payload;
    renderTable();
    renderChart();
  } catch (e) {
    if (isActiveLotteryView(lotteryId, viewToken)) {
      console.error('Failed to load experiments:', e);
    }
  }
}

function collapseConsecutiveDiscards(rows) {
  const collapsed = [];
  for (const row of rows) {
    const isDiscard = row.status === 'DISCARD';
    const last = collapsed[collapsed.length - 1];
    if (isDiscard && last && last._collapsed_discard) {
      last._count += 1;
      const expNum = Number(row.experiment_number);
      last._min_exp = Math.min(last._min_exp, expNum);
      last._max_exp = Math.max(last._max_exp, expNum);
      last.experiment_number = `${last._min_exp}-${last._max_exp}`;
      const metric = Number(row.average_hits || 0);
      if (metric > last._best_metric) last._best_metric = metric;
      last.average_hits = last._best_metric;
      last.strategy = `${last._count} discarded experiments`;
      last._descriptions.push(row.description || '');
      last.description = `${last._count} consecutive DISCARD experiments (best: ${last._best_metric.toFixed(6)}).\n` +
        last._descriptions.filter(Boolean).join('\n');
      last.elapsed_s = (last.elapsed_s || 0) + (row.elapsed_s || 0);
      last.hit_rate_2_plus = 0;
      last.hit_rate_3_plus = 0;
      last.portfolio_avg = 0;
      continue;
    }
    if (isDiscard) {
      const expNum = Number(row.experiment_number);
      collapsed.push({
        ...row,
        _collapsed_discard: true,
        _count: 1,
        _min_exp: expNum,
        _max_exp: expNum,
        _best_metric: Number(row.average_hits || 0),
        _descriptions: [row.description || ''],
      });
      continue;
    }
    collapsed.push(row);
  }
  return collapsed;
}

// --- Render table ---
const MAX_TABLE_ROWS = 15;
let showAllExperiments = false;

function renderTable() {
  const tbody = document.getElementById('exp-table-body');
  const sorted = [...experiments];
  const keys = ['experiment_number', 'status', 'average_hits', 'strategy',
                'hit_rate_2_plus', 'hit_rate_3_plus', 'portfolio_avg', 'elapsed_s'];
  sorted.sort((a, b) => {
    let va = a[keys[sortCol]], vb = b[keys[sortCol]];
    if (typeof va === 'string') va = va.toLowerCase();
    if (typeof vb === 'string') vb = vb.toLowerCase();
    if (va < vb) return sortAsc ? -1 : 1;
    if (va > vb) return sortAsc ? 1 : -1;
    return 0;
  });

  const collapseEnabled = document.getElementById('toggle-collapse')?.checked ?? true;
  const displayRows = (sortCol === 0 && !sortAsc && collapseEnabled) ? collapseConsecutiveDiscards(sorted) : sorted;
  const visibleRows = showAllExperiments ? displayRows : displayRows.slice(0, MAX_TABLE_ROWS);

  tbody.innerHTML = visibleRows.map(exp => {
    const isKeep = exp.status === 'KEEP';
    const rowClass = isKeep ? 'row-keep' : 'row-discard';
    const chipClass = isKeep ? 'chip-keep' : 'chip-discard';
    return `
      <tr class="${rowClass} expandable" onclick="toggleDesc(this)">
        <td><span class="expand-icon">&#9654;</span>${exp.experiment_number}</td>
        <td><span class="status-chip ${chipClass}">${exp.status}</span></td>
        <td class="metric-cell">${Number(exp.average_hits).toFixed(6)}</td>
        <td>${exp.strategy || '--'}</td>
        <td>${exp.hit_rate_2_plus ? (exp.hit_rate_2_plus * 100).toFixed(1) + '%' : '--'}</td>
        <td>${exp.hit_rate_3_plus ? (exp.hit_rate_3_plus * 100).toFixed(1) + '%' : '--'}</td>
        <td>${Number(exp.portfolio_avg).toFixed(4)}</td>
        <td>${fmtDuration(exp.elapsed_s)}</td>
      </tr>
      <tr class="desc-row" style="display:none">
        <td colspan="8">${exp.description || 'No description'}<br><code style="color:var(--text2)">${exp.commit || ''}</code></td>
      </tr>`;
  }).join('');

  // Show/hide "load more" button
  const moreBtn = document.getElementById('exp-show-more');
  if (moreBtn) {
    if (displayRows.length > MAX_TABLE_ROWS && !showAllExperiments) {
      moreBtn.style.display = '';
      moreBtn.textContent = `Show all ${displayRows.length} experiments`;
    } else {
      moreBtn.style.display = 'none';
    }
  }
}

function toggleDesc(row) {
  const descRow = row.nextElementSibling;
  const icon = row.querySelector('.expand-icon');
  if (descRow.style.display === 'none') {
    descRow.style.display = 'table-row';
    icon.classList.add('open');
  } else {
    descRow.style.display = 'none';
    icon.classList.remove('open');
  }
}

function sortTable(col) {
  if (sortCol === col) sortAsc = !sortAsc;
  else { sortCol = col; sortAsc = true; }
  showAllExperiments = false;
  renderTable();
}

function toggleShowAll() {
  showAllExperiments = true;
  renderTable();
}

// --- Render chart ---
function renderChart(skipDataHashCheck = false) {
  // Seria cronologică completă; graficul se concentrează pe ce contează:
  // linia „Running Best" vs. „Baseline", cu punctele de îmbunătățire evidente.
  const series = [...experiments].sort((a, b) => a.experiment_number - b.experiment_number);
  const labels = series.map(e => `Exp${e.experiment_number}`);
  const baseline = currentLotteryMeta.baseline || 0;

  // linia running best (pași) + punctele de îmbunătățire
  const bestLine = [];
  let currentBest = 0;
  series.forEach(e => {
    if (e.status === 'KEEP' && e.average_hits > currentBest) currentBest = e.average_hits;
    bestLine.push(currentBest || null);
  });
  const improvementPoints = [];
  let trackBest = 0;
  series.forEach(e => {
    if (e.status === 'KEEP' && e.average_hits > trackBest) {
      trackBest = e.average_hits;
      improvementPoints.push(e.average_hits);
    } else {
      improvementPoints.push(null);
    }
  });
  const nNewBests = improvementPoints.filter(v => v !== null).length;
  // discards la VALOAREA PROPRIE: norul de căutare, discret (nu bandă plată la best)
  const discardPoints = series.map(e => e.status === 'DISCARD' ? e.average_hits : null);

  // fereastră Y focusată pe intervalul baseline ↔ best; discarzile mult mai
  // mici ies din cadru (mai puțin zgomot vizual)
  const focusValues = [
    ...improvementPoints.filter(v => Number.isFinite(v)),
    ...bestLine.filter(v => Number.isFinite(v)),
  ];
  if (baseline > 0) focusValues.push(baseline);
  let yMin = 0, yMax = 1;
  if (focusValues.length) {
    const lo = Math.min(...focusValues);
    const hi = Math.max(...focusValues);
    const span = Math.max(hi - lo, hi * 0.03, 0.002);
    yMin = Math.max(0, lo - span * 0.45);
    yMax = hi + span * 0.12;
  }
  const dataSpan = yMax - yMin;
  const tickDecimals = dataSpan < 0.005 ? 6 : dataSpan < 0.03 ? 5 : dataSpan < 0.2 ? 4 : 3;

  const edgeBadge = document.getElementById('edge-badge');
  if (edgeBadge && focusValues.length) {
    const bestVal = Math.max(0, ...bestLine.filter(Number.isFinite));
    if (baseline > 0 && bestVal > baseline) {
      const edgePct = ((bestVal - baseline) / baseline * 100).toFixed(2);
      edgeBadge.textContent = `Best ${bestVal.toFixed(3)} vs baseline ${baseline.toFixed(3)} · edge +${edgePct}% · ${series.length} experimente · ${nNewBests} new bests`;
    } else if (baseline > 0) {
      edgeBadge.textContent = `Best ${bestVal.toFixed(3)} vs baseline ${baseline.toFixed(3)} · încă sub baseline · ${series.length} experimente`;
    } else {
      edgeBadge.textContent = `${series.length} experimente · ${nNewBests} new bests`;
    }
  }

  const dataHash = JSON.stringify({ labels, improvementPoints, discardPoints, bestLine, baseline, yMin, yMax });
  if (!skipDataHashCheck && dataHash === lastChartDataHash && chart) return;
  lastChartDataHash = dataHash;

  const datasets = [
    {
      label: 'Running Best',
      data: bestLine,
      borderColor: '#3fb950',
      borderWidth: 2.5,
      pointStyle: 'line',
      pointRadius: 0,
      stepped: true,
      fill: { target: 'origin', above: 'rgba(63,185,80,0.06)' },
      tension: 0,
      order: 1,
    },
    {
      label: `Baseline (${Number(baseline || 0).toFixed(3)})`,
      data: series.map(() => baseline || 0),
      borderColor: 'rgba(210,153,34,0.75)',
      borderWidth: 1.5,
      borderDash: [6, 4],
      pointStyle: 'line',
      pointRadius: 0,
      fill: false,
      order: 4,
    },
    {
      label: 'New Best',
      data: improvementPoints,
      pointBackgroundColor: '#3fb950',
      pointBorderColor: '#0f1a12',
      pointStyle: 'circle',
      pointRadius: 6,
      pointHoverRadius: 8,
      showLine: false,
      borderWidth: 1,
      fill: false,
      order: 2,
    },
    {
      label: 'Experiments discarded',
      data: discardPoints,
      pointBackgroundColor: 'rgba(248,81,73,0.35)',
      pointBorderColor: 'rgba(248,81,73,0.35)',
      pointStyle: 'circle',
      pointRadius: 2,
      pointHoverRadius: 4,
      showLine: false,
      borderWidth: 0,
      fill: false,
      order: 3,
    },
  ];

  const tooltipCfg = {
    callbacks: {
      label: ctx => {
        const exp = series[ctx.dataIndex];
        const value = ctx.parsed.y?.toFixed(6) || '--';
        if (!exp) return `${ctx.dataset.label}: ${value}`;
        const parts = [`${ctx.dataset.label}: ${value}`];
        if (exp.strategy) parts.push(`strategie: ${exp.strategy}`);
        if (exp.elapsed_seconds) parts.push(`durată: ${Number(exp.elapsed_seconds).toFixed(1)}s`);
        const edge = currentLotteryMeta.baseline ? ((exp.average_hits - currentLotteryMeta.baseline) / currentLotteryMeta.baseline * 100) : null;
        if (edge !== null && Number.isFinite(edge)) parts.push(`edge vs baseline: ${edge >= 0 ? '+' : ''}${edge.toFixed(2)}%`);
        return parts;
      },
      title: items => items.length ? `Experiment ${items[0].label.replace('Exp', '#')} (${series[items[0].dataIndex]?.status || ''})` : '',
    }
  };

  if (!chart) {
    const ctx = document.getElementById('evolution-chart').getContext('2d');
    chart = new Chart(ctx, {
      type: 'line',
      data: { labels, datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'nearest', axis: 'x', intersect: false },
        plugins: {
          legend: {
            labels: {
              color: '#8b949e',
              font: { family: "'Consolas', monospace", size: 11 },
              usePointStyle: true,
            }
          },
          tooltip: tooltipCfg,
        },
        scales: {
          x: {
            ticks: { color: '#8b949e', autoSkip: true, maxTicksLimit: 12, maxRotation: 0, minRotation: 0 },
            grid: { color: 'rgba(48,54,61,0.5)' },
          },
          y: {
            ticks: { color: '#8b949e', callback: v => Number(v).toFixed(tickDecimals) },
            grid: { color: 'rgba(48,54,61,0.5)' },
            min: yMin,
            max: yMax,
          }
        }
      }
    });
  } else {
    // dacă structura dataset-urilor s-a schimbat (update de cod), recreează
    if (chart.data.datasets.length !== datasets.length) {
      chart.destroy();
      chart = null;
      lastChartDataHash = null;
      renderChart(true);
      return;
    }
    chart.data.labels = labels;
    chart.data.datasets.forEach((ds, i) => {
      Object.assign(ds, datasets[i]);
      ds.data = datasets[i].data;
    });
    chart.options.scales.y.min = yMin;
    chart.options.scales.y.max = yMax;
    chart.options.scales.y.ticks.callback = v => Number(v).toFixed(tickDecimals);
    chart.options.scales.x.ticks.autoSkip = true;
    chart.options.scales.x.ticks.maxTicksLimit = 12;
    chart.options.scales.x.ticks.maxRotation = 0;
    chart.options.plugins.tooltip.callbacks = tooltipCfg.callbacks;
    chart.update('none');
  }
}

// --- SSE for live progress ---
function connectSSE(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  if (evtSource) evtSource.close();
  evtSource = new EventSource(apiPathFor(lotteryId, '/progress/stream'));
  const source = evtSource;

  evtSource.onopen = () => {
    if (!isActiveLotteryView(lotteryId, viewToken) || evtSource !== source) return;
    document.getElementById('connection-status').textContent = 'Connected';
    document.getElementById('connection-status').style.color = 'var(--green)';
  };

  evtSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      const summary = lotteries.find(item => item.id === lotteryId);
      if (summary && data?.status) summary.status = data.status;
      renderLotteryRail();
      if (!isActiveLotteryView(lotteryId, viewToken) || evtSource !== source) return;
      updateStatus(data);
    } catch (e) {
      if (isActiveLotteryView(lotteryId, viewToken) && evtSource === source) {
        console.error('SSE parse error:', e);
      }
    }
  };

  evtSource.onerror = () => {
    if (!isActiveLotteryView(lotteryId, viewToken) || evtSource !== source) return;
    document.getElementById('connection-status').textContent = 'Reconnecting...';
    document.getElementById('connection-status').style.color = 'var(--yellow)';
  };
}

function updateStatus(data) {
  const badge = document.getElementById('status-badge');
  const idleMsg = document.getElementById('status-idle-msg');
  const runningDiv = document.getElementById('status-running');
  const finishedDiv = document.getElementById('status-finished');
  const btnStart = document.getElementById('btn-start');
  const btnStop = document.getElementById('btn-stop');
  const btnLoop = document.getElementById('btn-loop');

  idleMsg.style.display = 'none';
  runningDiv.style.display = 'none';
  finishedDiv.style.display = 'none';

  if (data.status === 'running') {
    badge.className = 'status-badge badge-running';
    const loopInfo = data.loop_info;
    if (data.waiting_for_next_iteration && data.source === 'auto_loop' && loopInfo) {
      badge.textContent = `AUTO LOOP iter${loopInfo.iteration} (waiting for next iteration)`;
    } else if (data.source === 'auto_loop' && loopInfo) {
      badge.textContent = `AUTO LOOP iter${loopInfo.iteration} (r=${loopInfo.radius.toFixed(2)}, streak=${loopInfo.no_improve_streak})`;
    } else if (data.source === 'external') {
      badge.textContent = `RUNNING (External PID ${data.pid || '?'})`;
    } else {
      badge.textContent = `RUNNING (${data.experiment_id || '?'})`;
    }
    runningDiv.style.display = 'block';
    btnStart.disabled = true;
    btnLoop.disabled = data.source !== 'auto_loop';
    btnStop.disabled = false;
    _startLiveRefresh();

    const pct = data.percent || 0;
    const fill = document.getElementById('progress-fill');
    fill.style.width = pct + '%';
    fill.textContent = data.waiting_for_next_iteration ? 'waiting' : (Math.round(pct) + '%');

    document.getElementById('stat-tested').textContent = data.waiting_for_next_iteration
      ? '-- / --'
      : `${data.tested || 0} / ${data.total || '?'}`;
    document.getElementById('stat-eta').textContent = data.waiting_for_next_iteration
      ? 'starting soon'
      : fmtETA(data.eta_seconds);
    document.getElementById('stat-elapsed').textContent = fmtDuration(data.elapsed_seconds);

    if (data.best_so_far && data.best_so_far.average_hits) {
      document.getElementById('stat-best').textContent = data.best_so_far.average_hits.toFixed(6);
      document.getElementById('stat-best-config').textContent = data.best_so_far.config || '';
    } else {
      document.getElementById('stat-best').textContent = '--';
      document.getElementById('stat-best-config').textContent = '';
    }

    const recentEl = document.getElementById('recent-list');
    if (data.recent && data.recent.length > 0) {
      const items = [...data.recent].reverse();
      recentEl.innerHTML = items.map(r =>
        `<div class="recent-item">
          <span>${r.config}</span>
          <span class="recent-metric">${r.average_hits.toFixed(6)}</span>
        </div>`
      ).join('');
    } else if (data.waiting_for_next_iteration) {
      recentEl.innerHTML = '<div class="recent-item"><span>Loop active. Waiting for next iteration...</span><span class="recent-metric">--</span></div>';
    }

  } else if (data.status === 'finished') {
    badge.className = 'status-badge badge-finished';
    badge.textContent = 'FINISHED';
    finishedDiv.style.display = 'block';
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnLoop.disabled = false;
    _stopLiveRefresh();

    if (data.result) {
      document.getElementById('finished-metric').textContent =
        data.result.metric ? data.result.metric.toFixed(6) : '--';
      document.getElementById('finished-config').textContent =
        data.result.best_config || data.result.strategy || '--';
      document.getElementById('finished-total').textContent =
        data.result.total_tested || '--';
    }
    loadExperiments();

  } else if (data.status === 'crashed') {
    badge.className = 'status-badge badge-crashed';
    badge.textContent = 'CRASHED';
    idleMsg.style.display = 'block';
    idleMsg.innerHTML = `<span style="color:var(--red)">Experiment crashed. ${data.error || ''}</span>`;
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnLoop.disabled = false;
    _stopLiveRefresh();

  } else if (data.status === 'stopped') {
    badge.className = 'status-badge badge-stopped';
    badge.textContent = 'STOPPED';
    idleMsg.style.display = 'block';
    idleMsg.innerHTML = 'Experiment stopped. Click <b>Start</b> or <b>Auto Loop</b> to resume.';
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnLoop.disabled = false;
    _stopLiveRefresh();

  } else if (data.status === 'stalled') {
    badge.className = 'status-badge badge-stalled';
    badge.textContent = 'STALLED';
    idleMsg.style.display = 'block';
    idleMsg.innerHTML = `<span style="color:var(--yellow)">Experiment is stalled. ${data.error || ''}</span>`;
    btnStart.disabled = true;
    btnStop.disabled = false;
    btnLoop.disabled = true;
    _stopLiveRefresh();

  } else {
    badge.className = 'status-badge badge-idle';
    badge.textContent = 'IDLE';
    idleMsg.style.display = 'block';
    idleMsg.innerHTML = 'No experiment running. Click <b>Start</b> to begin.';
    btnStart.disabled = false;
    btnStop.disabled = true;
    btnLoop.disabled = false;
    _stopLiveRefresh();
  }
}

// --- Generate Numbers ---
async function generateNumbers() {
  const ticketCount = parseInt(document.getElementById('ticket-slider').value);
  const numsPerTicket = parseInt(document.getElementById('nums-per-ticket').value);
  const bonusControl = document.getElementById('bonus-control');
  const bonusPerTicket = bonusControl.style.display === 'none'
    ? 0
    : parseInt(document.getElementById('bonus-per-ticket').value);
  const btn = document.getElementById('btn-generate');
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = 'Generez...';
  try {
    const params = new URLSearchParams({
      ticket_count: String(ticketCount),
      nums_per_ticket: String(numsPerTicket),
    });
    if (bonusPerTicket > 0) params.set('bonus_per_ticket', String(bonusPerTicket));
    const { resp, data } = await fetchJsonWithRetry(`${apiPath('/generate')}?${params.toString()}`);
    if (!resp.ok) {
      alert(data.error || 'Error generating numbers');
      return;
    }
    const modeLabel = data.mode === 'system' ? ' [system]' : '';
    document.getElementById('strategy-badge').textContent = 'Strategy: ' + (data.strategy || 'unknown') + modeLabel;
    renderLastDrawLine(data);
    lastGeneratedPayload = data;
    lastGeneratedLotteryId = currentLottery;
    renderGeneratedTickets(data);
  } catch (e) {
    const statusText = document.getElementById('connection-status')?.textContent || 'Disconnected';
    alert(`Error: ${e.message}. Status conexiune: ${statusText}`);
  }
  if (btn) {
    btn.disabled = false;
    btn.textContent = 'Generează';
  }
}

// --- Auto Loop ---
let _loopRunning = false;

async function toggleLoop() {
  const btn = document.getElementById('btn-loop');
  if (_loopRunning) {
    btn.disabled = true;
    btn.textContent = 'Stopping...';
    try {
      const resp = await fetch(apiPath('/loop/stop'), { method: 'POST' });
      if (resp.ok) {
        _loopRunning = false;
        btn.textContent = 'Auto Loop';
        btn.style.background = 'var(--blue)';
      } else {
        _loopRunning = false;
        btn.textContent = 'Auto Loop';
        btn.style.background = 'var(--blue)';
      }
    } catch (e) {
      _loopRunning = false;
      btn.textContent = 'Auto Loop';
      btn.style.background = 'var(--blue)';
    }
    btn.disabled = false;
  } else {
    btn.disabled = true;
    btn.textContent = 'Starting...';
    try {
      const resp = await fetch(apiPath('/loop/start'), { method: 'POST' });
      const data = await resp.json();
      if (resp.ok) {
        _loopRunning = true;
        btn.textContent = 'Stop Loop';
        btn.style.background = 'var(--red)';
      } else {
        alert(data.error || 'Failed to start loop');
      }
    } catch (e) {
      alert('Error: ' + e.message);
    }
    btn.disabled = false;
  }
}

async function checkLoopStatus(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  const btn = document.getElementById('btn-loop');
  try {
    const { data } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/loop/status'), {}, 3, 500);
    if (!isActiveLotteryView(lotteryId, viewToken)) return;
    if (data.running) {
      _loopRunning = true;
      btn.textContent = 'Stop Loop';
      btn.style.background = 'var(--red)';
    } else {
      _loopRunning = false;
      btn.textContent = 'Auto Loop';
      btn.style.background = 'var(--blue)';
    }
  } catch (e) {
    if (!isActiveLotteryView(lotteryId, viewToken)) return;
    _loopRunning = false;
    btn.textContent = 'Auto Loop';
    btn.style.background = 'var(--blue)';
  }
}

// --- Actions ---
let _startPending = false;
async function startExperiment() {
  if (_startPending) return;
  _startPending = true;
  const btn = document.getElementById('btn-start');
  const btnLoop = document.getElementById('btn-loop');
  btn.disabled = true;
  btnLoop.disabled = true;
  btn.textContent = 'Starting...';
  try {
    const resp = await fetch(apiPath('/run'), { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      alert(data.error || 'Failed to start');
      btn.disabled = false;
      btnLoop.disabled = false;
    }
  } catch (e) {
    alert('Error: ' + e.message);
    btn.disabled = false;
    btnLoop.disabled = false;
  }
  btn.textContent = 'Start';
  _startPending = false;
}

async function stopExperiment() {
  const btn = document.getElementById('btn-stop');
  btn.disabled = true;
  btn.textContent = 'Stopping...';
  try {
    const resp = await fetch(apiPath('/stop'), { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      await refreshProgress();
    }
  } catch (e) {
    alert('Error: ' + e.message);
  }
  btn.textContent = 'Stop';
  btn.disabled = false;
}

// --- Init ---
let _refreshInterval = null;

function _startLiveRefresh() {
  if (_refreshInterval) return;
  _refreshInterval = setInterval(() => {
    loadExperiments(currentLottery, currentLotteryViewToken);
  }, 15000);
}

function _stopLiveRefresh() {
  if (_refreshInterval) {
    clearInterval(_refreshInterval);
    _refreshInterval = null;
  }
}

// Debounce flags
let isHistoryLoading = false;
let lastHistoryHash = null;
let lastHistoryFetchTime = 0;
const MIN_HISTORY_REFRESH_INTERVAL = 15000;

// Periodic refresh — experiments only when running, history only when tab active
setInterval(() => {
  // Only refresh experiments when an experiment is running
  const statusBadge = document.getElementById('status-badge');
  const isRunning = statusBadge && (statusBadge.textContent === 'RUNNING' || statusBadge.textContent === 'LOOPING');
  if (isRunning) {
    loadExperiments(currentLottery, currentLotteryViewToken).catch(() => {});
  }

  // Refresh history only when on the history tab
  if (currentMainTab === 'history' && !isHistoryLoading) {
    const now = Date.now();
    if (now - lastHistoryFetchTime >= MIN_HISTORY_REFRESH_INTERVAL) {
      isHistoryLoading = true;
      loadHistory(currentLottery, currentLotteryViewToken)
        .then(() => { lastHistoryFetchTime = now; })
        .catch(() => {})
        .finally(() => { isHistoryLoading = false; });
      loadAITicketHistory(currentLottery, currentLotteryViewToken);
    }
  }
}, 15000);

// Start live refresh
_startLiveRefresh();

// Hash change handler
window.addEventListener('hashchange', () => {
  const lotteryId = location.hash ? location.hash.slice(1) : currentLottery;
  if (lotteryId && lotteryId !== currentLottery && lotteries.some(l => l.id === lotteryId)) {
    onLotteryChange(lotteryId, false);
  }
});

// ===================== AI (Ollama) Panel =====================

let aiStatusCache = null;
let aiStatusFetchedAt = 0;
let aiAnalysisInFlight = false;

async function getAIStatus() {
  const now = Date.now();
  if (aiStatusCache && now - aiStatusFetchedAt < 60000) return aiStatusCache;
  try {
    const resp = await fetch('/api/ai/status');
    aiStatusCache = await resp.json();
    aiStatusFetchedAt = now;
  } catch (e) {
    aiStatusCache = { available: false };
  }
  return aiStatusCache;
}

function renderAIPanel(prediction) {
  const empty = document.getElementById('ai-empty');
  const details = document.getElementById('ai-details');
  const modelBadge = document.getElementById('ai-model-badge');
  if (!prediction) {
    if (empty) empty.style.display = 'block';
    if (details) details.style.display = 'none';
    return;
  }
  if (empty) empty.style.display = 'none';
  if (details) details.style.display = 'block';
  if (modelBadge && prediction.model) {
    modelBadge.textContent = `AI: ${prediction.model}`;
    modelBadge.className = 'status-badge badge-finished';
  }

  const confEl = document.getElementById('ai-confidence');
  if (confEl) {
    confEl.textContent = (prediction.confidence || 'low').toUpperCase();
    confEl.style.color = prediction.confidence === 'high' ? 'var(--success)'
      : prediction.confidence === 'medium' ? 'var(--warning)' : 'var(--text-tertiary)';
  }
  const nextEl = document.getElementById('ai-next-draw');
  if (nextEl) nextEl.textContent = prediction.next_draw_date || '--';
  const genEl = document.getElementById('ai-generated-at');
  if (genEl) genEl.textContent = prediction.generated_at || '--';

  const numsEl = document.getElementById('ai-numbers');
  if (numsEl) {
    const nums = prediction.recommended_numbers || [];
    numsEl.innerHTML = nums.map((n, i) =>
      `<span class="ai-ball ${i < 6 ? 'ai-ball-top' : ''}" title="rank #${i + 1}">${n}</span>`
    ).join('');
  }
  const analysisEl = document.getElementById('ai-analysis');
  if (analysisEl) analysisEl.textContent = prediction.analysis || '';
  const noiseBlock = document.getElementById('ai-noise-block');
  const noiseEl = document.getElementById('ai-noise');
  if (noiseBlock && noiseEl) {
    if (prediction.noise_assessment) {
      noiseBlock.style.display = 'block';
      noiseEl.textContent = prediction.noise_assessment;
    } else {
      noiseBlock.style.display = 'none';
    }
  }
  const suggBlock = document.getElementById('ai-suggestions-block');
  const suggEl = document.getElementById('ai-suggestions');
  if (suggBlock && suggEl) {
    const suggestions = prediction.strategy_suggestions || [];
    if (suggestions.length) {
      suggBlock.style.display = 'block';
      suggEl.innerHTML = suggestions.map(s =>
        `<div>&bull; <strong>${escapeHtml(s.param)}</strong> &rarr; ${escapeHtml(String(s.suggested))} — ${escapeHtml(s.reason || '')}</div>`
      ).join('');
    } else {
      suggBlock.style.display = 'none';
    }
  }
}

async function loadAIPredictions(lotteryId = currentLottery) {
  try {
    const [statusResp, predResp] = await Promise.all([
      getAIStatus(),
      fetch(apiPathFor(lotteryId, '/ai-predictions')),
    ]);
    const predData = await predResp.json();
    const modelBadge = document.getElementById('ai-model-badge');
    if (modelBadge && !predData?.prediction) {
      if (statusResp.available) {
        modelBadge.textContent = `AI: ${statusResp.resolved_model || 'gata'}`;
        modelBadge.className = 'status-badge badge-finished';
      } else {
        modelBadge.textContent = 'AI: Ollama offline';
        modelBadge.className = 'status-badge badge-idle';
      }
    }
    renderAIPanel(predData?.prediction || null);
  } catch (e) {
    console.error('Failed to load AI predictions:', e);
  }
}

async function runAIAnalysis() {
  if (aiAnalysisInFlight) return;
  const btn = document.getElementById('ai-analyze-btn');
  aiAnalysisInFlight = true;
  if (btn) { btn.disabled = true; btn.textContent = 'AI analizează...'; }
  try {
    const resp = await fetch(apiPathFor(currentLottery, '/ai-analyze'), { method: 'POST' });
    const data = await resp.json();
    if (!resp.ok) {
      alert(`AI analysis failed: ${data.error || resp.statusText}`);
    } else {
      renderAIPanel(data.prediction);
    }
  } catch (e) {
    alert(`AI analysis failed: ${e.message}`);
  } finally {
    aiAnalysisInFlight = false;
    if (btn) { btn.disabled = false; btn.textContent = 'Analizează cu AI'; }
  }
}

// --- Boot ---
loadLotteries();
// ===================== Agent Activity Widget (dreapta-sus) =====================

const AA_STATE_LABELS = {
  coding: 'scrie cod',
  researching: 'caută online',
  thinking: 'gândește',
  working: 'lucrează',
  idle: 'în așteptare',
};

function aaSinceText(sinceTs) {
  if (!sinceTs) return '';
  const secs = Math.max(0, Math.floor(Date.now() / 1000 - sinceTs));
  if (secs < 60) return `de ${secs}s`;
  const mins = Math.floor(secs / 60);
  if (mins < 60) return `de ${mins}m`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `de ${hours}h ${mins % 60}m`;
  return `de ${Math.floor(hours / 24)}z`;
}

function renderAgentActivity(data) {
  const dot = document.getElementById('aa-dot');
  const ring = document.getElementById('aa-ring');
  const stateEl = document.getElementById('aa-state');
  const taskEl = document.getElementById('aa-task');
  const sinceEl = document.getElementById('aa-since');
  const mini = document.getElementById('aa-mini');
  if (!dot || !data) return;

  const state = data.state || 'idle';
  dot.className = `aa-dot aa-${state}`;
  if (ring) {
    ring.className = `aa-ring aa-ring-${state}`;
    ring.style.display = (state === 'idle') ? 'none' : '';
  }
  if (stateEl) stateEl.textContent = `agent ${AA_STATE_LABELS[state] || state}`;
  if (taskEl) taskEl.textContent = data.task || '';
  if (sinceEl) sinceEl.textContent = aaSinceText(data.since);
  if (mini) {
    const parts = [];
    if (typeof data.loops_running === 'number' && data.loops_running > 0) {
      parts.push(`<span title="bucle de optimizare active">⚙ ${data.loops_running}</span>`);
    }
    (data.details || []).slice(0, 4).forEach(d => {
      if (d.state === 'coding') parts.push('<span title="AI coder">⌨</span>');
      else if (d.state === 'thinking') parts.push('<span title="analiză AI">✦</span>');
      else if (d.state === 'researching') parts.push('<span title="cercetare web">🌐</span>');
    });
    mini.innerHTML = parts.join(' ');
  }
}

async function pollAgentActivity() {
  try {
    const resp = await fetch('/api/agent-activity');
    if (resp.ok) renderAgentActivity(await resp.json());
  } catch (e) { /* server offline — UI-ul rămâne pe ultima stare */ }
}

setInterval(pollAgentActivity, 15000);
pollAgentActivity();

// ===================== AI Tickets panel =====================

function ticketOutcomeClass(hitCount, pickCount) {
  if (hitCount >= Math.max(3, Math.ceil(pickCount / 2))) return 'ticket-great';
  if (hitCount >= 2) return 'ticket-good';
  return 'ticket-none';
}

function renderAITickets(data) {
  const list = document.getElementById('ai-tickets-list');
  const badge = document.getElementById('ai-tickets-model');
  if (!list) return;
  const entries = data.entries || [];
  if (badge) {
    const model = entries[0]?.model;
    badge.textContent = model ? `model: ${model}` : '--';
  }
  if (!entries.length) {
    list.innerHTML = '<div style="color:var(--text-tertiary);font-size:13px;">Încă nu există bilete AI pentru această loterie. Agentul ai_ticket_agent le generează zilnic înaintea extragerii.</div>';
    renderAITicketsChart([]);
    return;
  }
  renderAITicketsChart(entries.filter(e => e.status === 'checked').slice(0, 30).reverse());
  list.innerHTML = entries.slice(0, 8).map(e => {
    const checked = e.status === 'checked';
    const drawNums = checked ? (e.draw_numbers || []) : null;
    const ticketsHtml = (e.tickets || []).map((t, i) => {
      const res = checked && e.results ? e.results[i] : null;
      const cls = res ? ticketOutcomeClass(res.hit_count, (t.main || []).length) : 'ticket-pending';
      const numsHtml = (t.main || []).map(n => {
        const hit = res && (res.hits || []).includes(n);
        return `<span class="ai-ball ${hit ? 'ai-ball-top' : ''}">${n}</span>`;
      }).join('');
      const bonusHtml = (t.bonus && t.bonus.length) ? ` <span style="color:var(--text-tertiary);font-size:11px;">+ bonus:</span> ` +
        t.bonus.map(n => {
          const hit = res && (res.bonus_hits || []).includes(n);
          return `<span class="ai-ball ${hit ? 'ai-ball-top' : ''}" style="min-width:26px;height:26px;font-size:11px;">${n}</span>`;
        }).join('') : '';
      const resBadge = res
        ? `<span class="status-badge ${res.hit_count > 0 ? 'badge-finished' : 'badge-idle'}">${res.hit_count} hits</span>`
        : `<span class="status-badge badge-idle">în aștept</span>`;
      return `<div class="ai-ticket-row ${cls}">
        <div class="ai-ticket-head">${resBadge} <span class="ai-ticket-why">${escapeHtml(t.why || '')}</span></div>
        <div class="ai-ticket-nums">${numsHtml}${bonusHtml}</div>
      </div>`;
    }).join('');
    const drawLine = checked
      ? `<span style="color:var(--success);">Numere câștigătoare (${e.draw_date}): ${drawNums.join(', ')}</span>`
      : `<span>Extragere: <strong>${e.draw_date}</strong></span>`;
    return `<div class="ai-ticket-entry">
      <div class="ai-ticket-date">${drawLine} <span style="color:var(--text-tertiary);font-size:11px;margin-left:var(--space-sm);">${escapeHtml(e.created_at || '')}</span></div>
      ${e.reasoning ? `<div class="ai-ticket-reasoning">${escapeHtml(e.reasoning)}</div>` : ''}
      ${ticketsHtml}
    </div>`;
  }).join('');
}

// ===================== AI Ticket History (tab History) =====================

function fmtMoneyRO(amount, currency) {
  if (!amount || !currency) return '—';
  const sign = amount < 0 ? '−' : '';
  return `${sign}≈ ${Math.abs(amount).toLocaleString('ro-RO', { maximumFractionDigits: 2 })} ${currency}`;
}

function aiHistSlotLabel(e) {
  if (e.draw_index && e.n_draw_times > 1) return `#${e.draw_index} (${e.draw_time || '?'})`;
  if (e.draw_time) return e.draw_time;
  return '—';
}

async function loadAITicketHistory(lotteryId = currentLottery, viewToken = currentLotteryViewToken) {
  if (currentMainTab !== 'history') return;
  try {
    const { data } = await fetchJsonWithRetry(apiPathFor(lotteryId, '/ai-ticket-history'), {}, 3, 500);
    if (!isActiveLotteryView(lotteryId, viewToken)) return;
    renderAITicketProgress(data.summary || {});
    renderAITicketHistoryTable(data.entries || []);
  } catch (e) {
    const body = document.getElementById('ai-hist-table-body');
    if (body && isActiveLotteryView(lotteryId, viewToken)) {
      body.innerHTML = '<tr><td colspan="8" style="color:var(--danger);">Eroare la încărcarea istoricului biletelor AI.</td></tr>';
    }
  }
}

function renderAITicketProgress(s) {
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  const cur = s.currency || '';
  set('ai-hist-draws', `${s.draws_played ?? 0}${s.draws_in_play ? ` (+${s.draws_in_play} în joc)` : ''}`);
  set('ai-hist-tickets', `${s.tickets_played ?? 0}${s.tickets_in_play ? ` (+${s.tickets_in_play} în joc)` : ''}`);
  set('ai-hist-hits', `${s.total_hits ?? 0} · cel mai bun bilet: ${s.best_hit ?? 0}`);
  set('ai-hist-winnings', fmtMoneyRO(s.estimated_winnings, cur));
  set('ai-hist-cost', fmtMoneyRO(s.estimated_cost, cur));
  const bal = document.getElementById('ai-hist-balance');
  if (bal) {
    bal.textContent = fmtMoneyRO(s.balance, cur);
    bal.style.color = (s.balance ?? 0) > 0 ? 'var(--success)' : ((s.balance ?? 0) < 0 ? 'var(--danger)' : '');
  }
  const priceEl = document.getElementById('ai-hist-price');
  if (priceEl) priceEl.textContent = s.price_per_ticket ? `≈ ${s.price_per_ticket} ${cur} / bilet` : 'preț nedefinit';

  const dist = document.getElementById('ai-hist-distribution');
  if (dist) {
    const parts = Object.entries(s.distribution || {});
    if (!parts.length) {
      dist.innerHTML = 'Din toate biletele jucate: încă nu există extrageri verificate.';
    } else {
      const total = parts.reduce((a, [, n]) => a + n, 0);
      dist.innerHTML = '<strong style="color:var(--text-secondary);">Din toate biletele jucate:</strong> ' +
        parts.map(([hits, n]) => {
          const pct = total ? Math.round(100 * n / total) : 0;
          return `<span class="status-badge ${hits > 0 ? 'badge-finished' : 'badge-idle'}" style="margin-right:6px;">${hits} lovituri: ${n} bilete (${pct}%)</span>`;
        }).join(' ');
    }
  }
}

function renderAITicketHistoryTable(entries) {
  const body = document.getElementById('ai-hist-table-body');
  if (!body) return;
  if (!entries.length) {
    body.innerHTML = '<tr><td colspan="8" style="color:var(--text-tertiary);">Încă nu există bilete AI. Agentul le generează în ziua extragerii.</td></tr>';
    return;
  }
  const dash = '—';
  body.innerHTML = entries.map(e => {
    const checked = e.status === 'checked';
    const cancelled = e.status === 'cancelled';
    const pad = arr => (arr || []).map(n => String(n).padStart(2, '0')).join(' · ');
    const nums = checked ? pad(e.draw_numbers) : `<span style="color:var(--text-tertiary);">în aștept</span>`;
    const bonus = checked && e.draw_bonus && e.draw_bonus.length
      ? ` <span style="color:var(--text-tertiary);">+bonus ${e.draw_bonus.join(', ')}</span>` : '';
    const best = checked && e.best_ticket
      ? `${pad(e.best_ticket)} <strong>(${e.best_hits} lovituri)</strong>` : dash;
    const prize = checked ? fmtMoneyRO(e.estimated_prize && e.estimated_prize.amount, e.estimated_prize && e.estimated_prize.currency) : dash;
    const statusBadge = cancelled
      ? '<span class="status-badge badge-idle">anulat</span>'
      : checked ? '<span class="status-badge badge-finished">verificat</span>'
                : '<span class="status-badge badge-idle">în joc</span>';
    const hits = checked ? `<strong>${e.total_hits ?? 0}</strong>` : dash;
    return `<tr>
      <td>${e.draw_date || dash}</td>
      <td>${aiHistSlotLabel(e)}</td>
      <td>${nums}${bonus}</td>
      <td>${e.n_tickets ?? 0}</td>
      <td>${hits}</td>
      <td style="font-size:12px;">${best}</td>
      <td>${prize}</td>
      <td>${statusBadge}</td>
    </tr>`;
  }).join('');
}

let aiTicketsChart = null;

function renderAITicketsChart(checkedEntries) {
  const canvas = document.getElementById('ai-tickets-chart');
  if (!canvas || typeof Chart === 'undefined') return;
  const labels = checkedEntries.map(e => e.draw_date || '');
  const totals = checkedEntries.map(e => (e.results || []).reduce((s, r) => s + (r.hit_count || 0), 0));
  const counts = checkedEntries.map(e => (e.results || []).length);
  if (aiTicketsChart) {
    aiTicketsChart.data.labels = labels;
    aiTicketsChart.data.datasets[0].data = totals;
    aiTicketsChart.data.datasets[1].data = counts;
    aiTicketsChart.update();
    return;
  }
  aiTicketsChart = new Chart(canvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels,
      datasets: [
        {
          label: 'Hits totale bilete AI',
          data: totals,
          backgroundColor: 'rgba(63,185,80,0.55)',
          borderColor: '#3fb950',
          borderWidth: 1,
        },
        {
          label: 'Nr. bilete jucate',
          data: counts,
          backgroundColor: 'rgba(139,148,158,0.35)',
          borderColor: '#8b949e',
          borderWidth: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { labels: { color: '#8b949e', font: { size: 10 }, usePointStyle: true } },
        tooltip: { callbacks: { title: items => items.length ? `Extragere ${items[0].label}` : '' } },
      },
      scales: {
        x: { ticks: { color: '#8b949e', font: { size: 9 } }, grid: { display: false } },
        y: { beginAtZero: true, ticks: { color: '#8b949e', precision: 0 }, grid: { color: 'rgba(48,54,61,0.5)' } },
      },
    },
  });
}

async function loadAITickets(lotteryId = currentLottery) {
  try {
    const resp = await fetch(apiPathFor(lotteryId, '/ai-tickets'));
    renderAITickets(await resp.json());
  } catch (e) {
    console.error('Failed to load AI tickets:', e);
  }
}

// ===================== Decision Ledger panel =====================

const LEDGER_TYPE_LABELS = {
  ai_analysis: ['✦', 'Analiză AI'],
  ai_tickets: ['🎟', 'Bilete propuse'],
  ticket_result: ['✅', 'Rezultat extragere'],
  strategy_code: ['⌨', 'Strategie nouă (cod)'],
  web_research: ['🌐', 'Cercetare web'],
  self_improvement: ['🧠', 'Auto-îmbunătățire'],
  ai_action: ['🎬', 'Acțiune AI'],
  system: ['•', 'Sistem'],
};

function renderAIPending(actions) {
  const box = document.getElementById('ai-pending-list');
  const wrap = document.getElementById('ai-pending-block');
  if (!box || !wrap) return;
  if (!actions || !actions.length) {
    wrap.style.display = 'none';
    box.innerHTML = '';
    return;
  }
  wrap.style.display = 'block';
  box.innerHTML = actions.map(a => `
    <div class="ledger-entry ai-pending-entry">
      <div class="ledger-head"><span class="ledger-icon">🎬</span>
        <span class="ledger-label">${escapeHtml(a.type)}</span>
        <span class="ledger-ts">${escapeHtml(a.created_at || '')}</span></div>
      <div class="ledger-action"><em>acțiune:</em> <code>${escapeHtml(JSON.stringify(a.params || {}))}</code></div>
      ${a.why ? `<div class="ledger-why"><em>motiv:</em> ${escapeHtml(a.why)}</div>` : ''}
      <div class="ai-pending-btns">
        <button class="btn btn-success" onclick="decideAIAction('${a.id}', 'approve')">✅ Aprobă</button>
        <button class="btn btn-danger" onclick="decideAIAction('${a.id}', 'reject')">❌ Refuză</button>
      </div>
    </div>`).join('');
}

async function decideAIAction(actionId, verdict) {
  try {
    await fetch(`/api/ai-actions/${actionId}/${verdict}`, { method: 'POST' });
  } catch (e) { console.error('action failed', e); }
  loadAIPending();
  loadLedger();
}

async function loadAIPending() {
  try {
    const resp = await fetch('/api/ai-actions');
    const data = await resp.json();
    renderAIPending(data.pending || []);
  } catch (e) { console.error('failed to load pending actions', e); }
}

function ledgerEntryHtml(e) {
  const [icon, label] = LEDGER_TYPE_LABELS[e.type] || LEDGER_TYPE_LABELS.system;
  const payload = e.payload || {};
  let extra = '';
  if (e.type === 'strategy_code') {
    const m = payload.metrics || {};
    extra = `<div class="ledger-metrics">average_hits: <strong>${m.average_hits ?? '?'}</strong>` +
      (payload.promoted ? ' <span class="status-badge badge-finished">PROMOVATĂ în auto-loop</span>'
                        : ' <span class="status-badge badge-idle">arhivată</span>') + `</div>`;
  }
  if (e.type === 'ticket_result' && payload.results) {
    extra = `<div class="ledger-metrics">hits/bilet: ${payload.results.map(r => r.hit_count).join(', ')} · extragere: ${(payload.draw_numbers || []).join(', ')}</div>`;
  }
  if (e.type === 'ai_analysis' && (payload.recommended_numbers || []).length) {
    extra = `<div class="ledger-metrics">top: ${(payload.recommended_numbers || []).slice(0, 10).join(', ')}</div>`;
  }
  return `<div class="ledger-entry">
    <div class="ledger-head"><span class="ledger-icon">${icon}</span>
      <span class="ledger-label">${label}</span>
      <span class="ledger-lottery">${escapeHtml(e.lottery || '')}</span>
      <span class="ledger-ts">${escapeHtml((e.ts || '').replace('T', ' '))}</span></div>
    ${e.thought ? `<div class="ledger-thought"><em>gândire:</em> ${escapeHtml(e.thought)}</div>` : ''}
    ${e.action ? `<div class="ledger-action"><em>acțiune:</em> ${escapeHtml(e.action)}</div>` : ''}
    ${e.why ? `<div class="ledger-why"><em>motiv:</em> ${escapeHtml(e.why)}</div>` : ''}
    ${extra}
  </div>`;
}

async function loadLedger() {
  const list = document.getElementById('ledger-list');
  if (!list) return;
  const filter = document.getElementById('ledger-type-filter')?.value || '';
  try {
    const [globalResp, localResp] = await Promise.all([
      fetch(`/api/ledger?limit=40${filter ? '&entry_type=' + filter : ''}`),
      fetch(apiPathFor(currentLottery, `/ledger?limit=30${filter ? '&entry_type=' + filter : ''}`)),
    ]);
    const globalData = await globalResp.json();
    const localData = await localResp.json();
    const seen = new Set();
    const merged = [];
    for (const e of [...(localData.entries || []), ...(globalData.entries || [])]) {
      if (seen.has(e.id)) continue;
      seen.add(e.id);
      merged.push(e);
    }
    merged.sort((a, b) => (b.ts || '').localeCompare(a.ts || ''));
    if (!merged.length) {
      list.innerHTML = '<div style="color:var(--text-tertiary);font-size:13px;">Ledger-ul e încă gol. Agentul AI îl completează la fiecare analiză, set de bilete sau strategie nouă.</div>';
      return;
    }
    list.innerHTML = merged.slice(0, 60).map(ledgerEntryHtml).join('');
  } catch (e) {
    console.error('Failed to load ledger:', e);
  }
}
