/**
 * Autoia Widget v2 — Asistente ABA embebido para TheTigerBets
 *
 * Novedades v2:
 * - 4to tab "Historial": predicciones pasadas con estado correcto/incorrecto
 * - Tarjeta de Edge visual: modelo vs mercado con diferencia resaltada
 * - Auto-refresh: re-consulta prediccion cada 60s con el panel abierto
 * - Timestamp "Actualizado hace X seg" en header
 *
 * Integracion de 2 lineas:
 *   <script src="/autoia/autoia_client.js"></script>
 *   <script src="/autoia/widget.js"></script>
 *   <script>AutoiaWidget.init({ host: 'localhost', port: 8765 });</script>
 */

(function(global) {
  'use strict';

  const CSS = `
    #autoia-widget * { box-sizing: border-box; font-family: 'Segoe UI', Arial, sans-serif; }

    #autoia-fab {
      position: fixed; bottom: 24px; right: 24px; z-index: 9999;
      width: 58px; height: 58px; border-radius: 50%;
      background: linear-gradient(135deg, #1a1f35 0%, #2a3560 100%);
      border: 2px solid #4a90e2; cursor: pointer;
      display: flex; align-items: center; justify-content: center;
      box-shadow: 0 4px 20px rgba(74,144,226,0.4);
      transition: transform 0.2s, box-shadow 0.2s;
    }
    #autoia-fab:hover { transform: scale(1.1); box-shadow: 0 6px 28px rgba(74,144,226,0.6); }
    #autoia-fab .fab-icon { font-size: 22px; }
    #autoia-fab .fab-badge {
      position: absolute; top: -4px; right: -4px;
      background: #3ddc84; color: #000; font-size: 10px; font-weight: bold;
      border-radius: 10px; padding: 1px 5px; display: none;
    }

    #autoia-panel {
      position: fixed; bottom: 94px; right: 24px; z-index: 9998;
      width: 370px; max-height: 560px;
      background: #0d1117; border: 1px solid #2a3560;
      border-radius: 14px; overflow: hidden;
      box-shadow: 0 8px 40px rgba(0,0,0,0.7);
      display: none; flex-direction: column;
      animation: autoia-slide-up 0.2s ease;
    }
    @keyframes autoia-slide-up {
      from { opacity:0; transform: translateY(12px); }
      to   { opacity:1; transform: translateY(0); }
    }
    #autoia-panel.open { display: flex; }

    .autoia-header {
      background: linear-gradient(135deg, #1a1f35 0%, #1e2a50 100%);
      padding: 10px 14px; display: flex; align-items: center; gap: 10px;
      border-bottom: 1px solid #2a3560;
    }
    .autoia-header .logo { font-size: 20px; }
    .autoia-header-text { flex: 1; }
    .autoia-header .title { color: #e8eaf0; font-size: 13px; font-weight: 700; }
    .autoia-header .updated { color: #556; font-size: 10px; margin-top: 1px; }
    .autoia-header .refresh-spin { display: inline-block; animation: autoia-spin 1s linear infinite; }
    @keyframes autoia-spin { to { transform: rotate(360deg); } }
    .autoia-close {
      color: #7888aa; cursor: pointer; font-size: 18px; line-height: 1;
      padding: 2px 6px; border-radius: 4px;
    }
    .autoia-close:hover { color: #e8eaf0; background: rgba(255,255,255,0.07); }

    .autoia-tabs { display: flex; background: #0d1117; border-bottom: 1px solid #1e2a50; }
    .autoia-tab {
      flex: 1; padding: 8px 2px; text-align: center; font-size: 10px; font-weight: 600;
      color: #445; cursor: pointer; transition: color 0.15s;
      border-bottom: 2px solid transparent; white-space: nowrap;
    }
    .autoia-tab:hover { color: #aab; }
    .autoia-tab.active { color: #4a90e2; border-bottom-color: #4a90e2; }

    .autoia-body { flex: 1; overflow-y: auto; padding: 12px; }
    .autoia-body::-webkit-scrollbar { width: 4px; }
    .autoia-body::-webkit-scrollbar-track { background: transparent; }
    .autoia-body::-webkit-scrollbar-thumb { background: #2a3560; border-radius: 2px; }

    .autoia-card {
      background: #131820; border: 1px solid #1e2a50; border-radius: 10px;
      padding: 11px; margin-bottom: 9px;
    }
    .autoia-card-title {
      color: #7888aa; font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.8px; margin-bottom: 7px;
    }

    .pred-match { color: #c8d0e8; font-size: 12px; font-weight: 600; margin-bottom: 3px; }
    .pred-winner { font-size: 17px; font-weight: 800; margin: 5px 0 3px; }
    .pred-conf { font-size: 11px; color: #7888aa; margin-bottom: 7px; }

    .prob-row { display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }
    .prob-label { color: #9aaac8; font-size: 11px; min-width: 88px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .prob-bar-wrap { flex: 1; height: 8px; background: #1e2a50; border-radius: 4px; overflow: hidden; position: relative; }
    .prob-bar { height: 100%; border-radius: 4px; transition: width 0.4s; }
    .prob-bar.market { position: absolute; top: 0; left: 0; height: 100%; background: rgba(255,255,255,0.1); border-radius: 4px; }
    .prob-pct { color: #9aaac8; font-size: 11px; min-width: 34px; text-align: right; }
    .prob-edge { font-size: 10px; font-weight: 700; min-width: 40px; text-align: right; }

    .edge-card { border-color: #3ddc84 !important; }
    .edge-card .autoia-card-title { color: #3ddc84; }

    .mo-row { display: flex; align-items: flex-start; gap: 7px; padding: 6px 0; border-bottom: 1px solid #1a2040; }
    .mo-row:last-child { border-bottom: none; }
    .mo-badge { font-size: 10px; font-weight: 800; padding: 2px 5px; border-radius: 4px; flex-shrink: 0; margin-top: 1px; }
    .mo-badge.EO { background: rgba(61,220,132,0.15); color: #3ddc84; }
    .mo-badge.AO { background: rgba(220,80,80,0.15); color: #e05252; }
    .mo-desc { color: #c8d0e8; font-size: 11px; line-height: 1.4; flex: 1; }
    .mo-strength { font-size: 10px; color: #556; flex-shrink: 0; }

    .factor-row { display: flex; gap: 7px; margin-bottom: 5px; align-items: flex-start; }
    .factor-icon { font-size: 12px; flex-shrink: 0; margin-top: 1px; }
    .factor-text { color: #c8d0e8; font-size: 11px; line-height: 1.4; }

    .aba-term { display: flex; gap: 8px; margin-bottom: 5px; align-items: center; }
    .aba-abbr { min-width: 28px; height: 22px; border-radius: 4px; font-size: 10px; font-weight: 800; display: flex; align-items: center; justify-content: center; }
    .aba-desc { color: #8899bb; font-size: 10px; }

    /* Historial tab */
    .hist-row {
      display: flex; align-items: center; gap: 8px;
      padding: 8px 0; border-bottom: 1px solid #1a2040; cursor: pointer;
    }
    .hist-row:last-child { border-bottom: none; }
    .hist-row:hover { background: rgba(74,144,226,0.04); border-radius: 6px; }
    .hist-badge { width: 20px; height: 20px; border-radius: 50%; flex-shrink: 0; display: flex; align-items: center; justify-content: center; font-size: 10px; font-weight: 800; }
    .hist-badge.ok  { background: rgba(61,220,132,0.2); color: #3ddc84; }
    .hist-badge.fail{ background: rgba(220,80,80,0.2);  color: #e05252; }
    .hist-badge.pend{ background: rgba(100,120,180,0.2);color: #4a90e2; }
    .hist-info { flex: 1; min-width: 0; }
    .hist-subject { color: #c8d0e8; font-size: 11px; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .hist-pred { color: #7888aa; font-size: 10px; }
    .hist-conf { font-size: 11px; font-weight: 700; flex-shrink: 0; }

    .accuracy-row { display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
    .accuracy-label { color: #9aaac8; font-size: 11px; min-width: 60px; }
    .accuracy-bar-wrap { flex: 1; height: 8px; background: #1e2a50; border-radius: 4px; overflow: hidden; }
    .accuracy-bar { height: 100%; border-radius: 4px; }
    .accuracy-pct { font-size: 11px; font-weight: 700; min-width: 34px; text-align: right; }

    .autoia-input-area { padding: 9px 12px; border-top: 1px solid #1e2a50; background: #0d1117; }
    .autoia-input-row { display: flex; gap: 7px; }
    .autoia-input {
      flex: 1; background: #131820; border: 1px solid #2a3560;
      border-radius: 8px; padding: 7px 10px; color: #c8d0e8; font-size: 12px; outline: none;
    }
    .autoia-input:focus { border-color: #4a90e2; }
    .autoia-send { background: #4a90e2; color: #fff; border: none; border-radius: 8px; padding: 7px 12px; cursor: pointer; font-size: 13px; font-weight: 600; }
    .autoia-send:hover { background: #5aa0f2; }
    .autoia-input-hint { color: #334; font-size: 10px; margin-top: 4px; }

    .autoia-empty { color: #445; font-size: 12px; text-align: center; padding: 24px 0; }
    .autoia-loading { color: #4a90e2; font-size: 12px; text-align: center; padding: 16px 0; }
    .autoia-error { color: #e05252; font-size: 11px; text-align: center; padding: 12px; background: rgba(220,80,80,0.07); border-radius: 8px; }
    .small-btn {
      background: none; border: 1px solid #2a3560; border-radius: 5px;
      color: #7888aa; font-size: 10px; padding: 3px 8px; cursor: pointer;
    }
    .small-btn:hover { border-color: #4a90e2; color: #4a90e2; }
  `;

  class AutoiaWidget {

    static init(config = {}) {
      const w = new AutoiaWidget(config);
      w.mount();
      AutoiaWidget._instance = w;
      return w;
    }

    constructor(config) {
      this.client      = new AutoiaClient(config);
      this.tab         = 'pred';
      this.open        = false;
      this.loading     = false;
      this.refreshing  = false;
      this.prediction  = null;
      this.lastMatch   = null;
      this._lastUpdate = null;
      this._refreshTimer = null;
      this._historyLocal = [];   // cache local de predicciones
      this._autoDetect = config.autoDetect !== false;
      this._refreshInterval = config.refreshInterval || 60000; // 60s
    }

    mount() {
      if (!document.getElementById('autoia-styles')) {
        const s = document.createElement('style');
        s.id = 'autoia-styles';
        s.textContent = CSS;
        document.head.appendChild(s);
      }

      document.body.insertAdjacentHTML('beforeend', `
        <div id="autoia-widget">
          <div id="autoia-fab" title="Autoia ABA — Motor de prediccion">
            <span class="fab-icon">🐯</span>
            <span class="fab-badge" id="autoia-badge">!</span>
          </div>
          <div id="autoia-panel">
            <div class="autoia-header">
              <span class="logo">🐯</span>
              <div class="autoia-header-text">
                <div class="title">Autoia ABA</div>
                <div class="updated" id="autoia-updated">Motor de prediccion conductual</div>
              </div>
              <span class="autoia-close" id="autoia-close-btn">✕</span>
            </div>
            <div class="autoia-tabs">
              <div class="autoia-tab active" data-tab="pred">Prediccion</div>
              <div class="autoia-tab" data-tab="aba">Marco ABA</div>
              <div class="autoia-tab" data-tab="hist">Historial</div>
              <div class="autoia-tab" data-tab="input">Analizar</div>
            </div>
            <div class="autoia-body" id="autoia-body">
              <div class="autoia-empty">Ingresa un partido para analizar</div>
            </div>
            <div class="autoia-input-area">
              <div class="autoia-input-row">
                <input class="autoia-input" id="autoia-input"
                  placeholder="Real Madrid vs Barcelona | 1.85 / 3.60 / 4.20" />
                <button class="autoia-send" id="autoia-analyze-btn">▶</button>
              </div>
              <div class="autoia-input-hint">Formato: Equipo A vs Equipo B | local / empate / visitante</div>
            </div>
          </div>
        </div>
      `);

      document.getElementById('autoia-fab').addEventListener('click', () => this.toggle());
      document.getElementById('autoia-close-btn').addEventListener('click', () => this.close());
      document.querySelectorAll('.autoia-tab').forEach(t =>
        t.addEventListener('click', () => {
          this.tab = t.dataset.tab;
          this._renderTabs();
          this._renderBody();
        })
      );
      document.getElementById('autoia-analyze-btn').addEventListener('click', () => this._analyzeInput());
      document.getElementById('autoia-input').addEventListener('keydown', e => {
        if (e.key === 'Enter') this._analyzeInput();
      });

      if (this._autoDetect) setTimeout(() => this._autoDetectMatch(), 1500);
    }

    // ── Panel control ──────────────────────────────────────────────────────

    toggle() { this.open ? this.close() : this.open_panel(); }

    open_panel() {
      this.open = true;
      document.getElementById('autoia-panel').classList.add('open');
      document.getElementById('autoia-badge').style.display = 'none';
      if (!this.prediction && this.lastMatch) {
        this._loadPrediction(this.lastMatch);
      }
      this._startAutoRefresh();
    }

    close() {
      this.open = false;
      document.getElementById('autoia-panel').classList.remove('open');
      this._stopAutoRefresh();
    }

    // ── Auto-refresh ───────────────────────────────────────────────────────

    _startAutoRefresh() {
      this._stopAutoRefresh();
      if (!this.lastMatch) return;
      this._refreshTimer = setInterval(() => {
        if (this.open && this.lastMatch && !this.loading) {
          this._silentRefresh();
        }
      }, this._refreshInterval);
    }

    _stopAutoRefresh() {
      if (this._refreshTimer) {
        clearInterval(this._refreshTimer);
        this._refreshTimer = null;
      }
    }

    async _silentRefresh() {
      this.refreshing = true;
      this._updateTimestamp('Actualizando...');
      try {
        let pred;
        const m = this.lastMatch;
        if (m.odds) {
          pred = await this.client.predictBetting(m.home, m.away, m.odds.home, m.odds.draw, m.odds.away);
        } else {
          pred = await this.client.predictSports(m.home, m.away);
        }
        // Invalidar cache para forzar datos frescos
        this.client._cache.clear();
        this.prediction = pred;
        this._lastUpdate = Date.now();
        this._renderBody();
      } catch(e) { /* silencioso */ }
      this.refreshing = false;
      this._updateTimestamp();
    }

    _updateTimestamp(override) {
      const el = document.getElementById('autoia-updated');
      if (!el) return;
      if (override) { el.textContent = override; return; }
      if (this._lastUpdate) {
        const sec = Math.round((Date.now() - this._lastUpdate) / 1000);
        el.textContent = sec < 10 ? 'Actualizado ahora' : `Actualizado hace ${sec}s`;
      }
    }

    // ── Auto-detect ────────────────────────────────────────────────────────

    _autoDetectMatch() {
      const text    = document.body.innerText || '';
      const pattern = /([A-Z][a-zA-Z\s\.]{2,25})\s+(?:vs\.?|contra|v\.?)\s+([A-Z][a-zA-Z\s\.]{2,25})/gi;
      const matches = [...text.matchAll(pattern)];
      if (!matches.length) return;
      const [, home, away] = matches[0];
      const h = home.trim(), a = away.trim();
      if (h.length < 3 || a.length < 3) return;
      this.lastMatch = { home: h, away: a };
      document.getElementById('autoia-input').value = `${h} vs ${a}`;
      document.getElementById('autoia-badge').style.display = 'block';
    }

    // ── Input parser ───────────────────────────────────────────────────────

    _parseInput(text) {
      const parts   = text.split('|');
      const vsMatch = parts[0].trim().match(/^(.+?)\s+(?:vs\.?|contra|v\.?)\s+(.+)$/i);
      if (!vsMatch) return null;
      const home = vsMatch[1].trim();
      const away = vsMatch[2].trim();
      let odds = null;
      if (parts[1]) {
        const nums = parts[1].match(/[\d]+[.,][\d]+/g);
        if (nums && nums.length >= 2) {
          const parse = s => parseFloat(s.replace(',', '.'));
          odds = { home: parse(nums[0]), draw: parse(nums[1]), away: parse(nums[2] || nums[1]) };
        }
      }
      return { home, away, odds };
    }

    async _analyzeInput() {
      const raw    = document.getElementById('autoia-input').value.trim();
      if (!raw) return;
      const parsed = this._parseInput(raw);
      if (!parsed) {
        this._renderError('Formato: "Real Madrid vs Barcelona" o agregar cuotas: "... | 1.85 / 3.60 / 4.20"');
        return;
      }
      this.lastMatch = parsed;
      await this._loadPrediction(parsed);
      this._startAutoRefresh();
    }

    async _loadPrediction({ home, away, odds }) {
      this.loading = true;
      this._renderBody();
      try {
        this.client._cache.clear();
        const pred = odds
          ? await this.client.predictBetting(home, away, odds.home, odds.draw, odds.away)
          : await this.client.predictSports(home, away);
        this.prediction = pred;
        this._lastUpdate = Date.now();
        this._historyLocal.unshift({ ...pred, _ts: Date.now() });
        if (this._historyLocal.length > 30) this._historyLocal.pop();
      } catch(e) {
        this.prediction = null;
        this._renderError('No se pudo conectar con Autoia. Ejecuta: python predict_main.py --serve');
      }
      this.loading = false;
      this._updateTimestamp();
      this._renderBody();
    }

    // ── Render ─────────────────────────────────────────────────────────────

    _renderTabs() {
      document.querySelectorAll('.autoia-tab').forEach(t =>
        t.classList.toggle('active', t.dataset.tab === this.tab)
      );
    }

    _renderBody() {
      const body = document.getElementById('autoia-body');
      if (!body) return;
      if (this.loading) {
        body.innerHTML = '<div class="autoia-loading">⟳ Analizando conducta de la masa...</div>';
        return;
      }
      if (this.tab === 'pred')  body.innerHTML = this._htmlPrediction();
      if (this.tab === 'aba')   body.innerHTML = this._htmlABA();
      if (this.tab === 'hist')  body.innerHTML = this._htmlHistory();
      if (this.tab === 'input') body.innerHTML = this._htmlInput();
    }

    // ── Tab: Prediccion ────────────────────────────────────────────────────

    _htmlPrediction() {
      if (!this.prediction) return '<div class="autoia-empty">Ingresa un partido arriba para analizar</div>';
      const p    = this.prediction;
      const conf = p.confidence || 0;
      const cc   = AutoiaClient.confidenceColor(conf);
      const probs = p.probs || p.probabilities || {};
      const odds  = this.lastMatch?.odds;

      let html = `<div class="autoia-card">
        <div class="autoia-card-title">Prediccion ABA</div>
        <div class="pred-match">${p.subject || (this.lastMatch?.home + ' vs ' + this.lastMatch?.away)}</div>
        <div class="pred-winner" style="color:${cc}">${p.prediction || '—'}</div>
        <div class="pred-conf">Confianza: <strong style="color:${cc}">${(conf*100).toFixed(0)}%</strong>
          ${this.refreshing ? '<span class="refresh-spin">↻</span>' : ''}
        </div>
      `;

      // Barras de probabilidad con cuota implicita del mercado superpuesta
      let marketProbs = {};
      if (odds) {
        const totRaw = 1/odds.home + 1/odds.draw + 1/odds.away;
        marketProbs = {
          [this.lastMatch.home]: (1/odds.home)/totRaw,
          'draw':                (1/odds.draw)/totRaw,
          [this.lastMatch.away]: (1/odds.away)/totRaw,
        };
      }

      Object.entries(probs).forEach(([label, pval]) => {
        const pct      = (pval * 100).toFixed(1);
        const isWinner = label === p.prediction;
        const mktVal   = marketProbs[label] || 0;
        const mktPct   = (mktVal * 100).toFixed(0);
        const edge     = odds ? ((pval - mktVal) * 100) : 0;
        const edgeCol  = edge > 2 ? '#3ddc84' : edge < -2 ? '#e05252' : '#556';
        html += `
          <div class="prob-row">
            <div class="prob-label" title="${label}">${label}</div>
            <div class="prob-bar-wrap">
              ${mktVal ? `<div class="prob-bar market" style="width:${mktPct}%"></div>` : ''}
              <div class="prob-bar" style="width:${pct}%;background:${isWinner ? cc : '#2a3560'}"></div>
            </div>
            <div class="prob-pct">${pct}%</div>
            ${odds ? `<div class="prob-edge" style="color:${edgeCol}">${edge>0?'+':''}${edge.toFixed(0)}%</div>` : ''}
          </div>
        `;
      });
      html += '</div>';

      // Tarjeta de edge (value bet) si hay cuotas
      if (odds) {
        html += this._htmlEdgeCard(probs, odds);
      }

      // Factores
      const factors = p.factors || [];
      const risks   = p.risks   || [];
      if (factors.length || risks.length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">Factores ABA</div>';
        factors.forEach(f => html += `<div class="factor-row"><span class="factor-icon" style="color:#3ddc84">+</span><div class="factor-text">${f}</div></div>`);
        risks.forEach(r   => html += `<div class="factor-row"><span class="factor-icon" style="color:#e05252">!</span><div class="factor-text">${r}</div></div>`);
        html += '</div>';
      }

      return html;
    }

    _htmlEdgeCard(probs, odds) {
      const home = this.lastMatch?.home || '';
      const away = this.lastMatch?.away || '';
      const oddsMap = { [home]: odds.home, draw: odds.draw, [away]: odds.away };
      const probMap = probs;

      let bestLabel = '', bestEV = -Infinity;
      Object.entries(probMap).forEach(([label, p]) => {
        const o  = oddsMap[label];
        if (!o) return;
        const ev = p * o - 1;
        if (ev > bestEV) { bestEV = ev; bestLabel = label; }
      });

      if (bestEV > 0.02) {
        return `
          <div class="autoia-card edge-card">
            <div class="autoia-card-title">Value Bet Detectado</div>
            <div style="color:#3ddc84;font-size:16px;font-weight:800;margin-bottom:4px">${bestLabel.toUpperCase()}</div>
            <div style="display:flex;gap:16px;align-items:center">
              <div>
                <div style="color:#556;font-size:9px">CUOTA</div>
                <div style="color:#c8d0e8;font-size:14px;font-weight:700">${(oddsMap[bestLabel]||0).toFixed(2)}</div>
              </div>
              <div>
                <div style="color:#556;font-size:9px">PROB. MODELO</div>
                <div style="color:#4a90e2;font-size:14px;font-weight:700">${((probMap[bestLabel]||0)*100).toFixed(1)}%</div>
              </div>
              <div>
                <div style="color:#556;font-size:9px">EV</div>
                <div style="color:#3ddc84;font-size:14px;font-weight:700">+${(bestEV*100).toFixed(1)}%</div>
              </div>
            </div>
            <div style="color:#556;font-size:10px;margin-top:6px">
              Modelo sobrevalua esta opcion respecto al mercado
            </div>
          </div>
        `;
      } else if (bestEV < -0.05) {
        return `
          <div class="autoia-card" style="border-color:#e05252">
            <div class="autoia-card-title" style="color:#e05252">Mercado Sobrevaluado</div>
            <div style="color:#e05252;font-size:12px">
              Las cuotas actuales no ofrecen valor segun el modelo ABA.<br>
              <span style="color:#556">EV mejor opcion: ${(bestEV*100).toFixed(1)}%</span>
            </div>
          </div>
        `;
      }
      return '';
    }

    // ── Tab: ABA ───────────────────────────────────────────────────────────

    _htmlABA() {
      const p   = this.prediction;
      const aba = p?.aba || {};
      const mos = aba.mos || [];

      let html = `
        <div class="autoia-card">
          <div class="autoia-card-title">Contingencia de 4 Terminos</div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a3a2a;color:#3ddc84">MO</div><div class="aba-desc">Operacion Motivadora (EO sube valor / AO lo baja)</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a2a40;color:#4a90e2">SD</div><div class="aba-desc">Estimulo discriminativo (cuota, noticia, racha)</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#2a2a1a;color:#f0c832">R</div><div class="aba-desc">Respuesta de la masa (apuesta, inversion, apoyo)</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a2a1a;color:#3ddc84">C</div><div class="aba-desc">Consecuencia (ganancia o perdida)</div></div>
        </div>
      `;

      if (!p) return html + '<div class="autoia-empty" style="font-size:11px">Analiza un partido para ver los MOs</div>';

      if (mos.length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">MOs Activos</div>';
        mos.forEach(mo => {
          const t = mo.mo_type || mo.type || '?';
          const s = typeof mo.strength === 'number' ? (mo.strength*100).toFixed(0)+'%' : '';
          html += `
            <div class="mo-row">
              <div class="mo-badge ${t}">${t}</div>
              <div class="mo-desc">${mo.description || mo.source || ''}</div>
              <div class="mo-strength">${s}</div>
            </div>
          `;
        });
        html += '</div>';
      }

      const fn = aba.function;
      if (fn) {
        const fnColors = { tangible:'#3ddc84', escape:'#e05252', attention:'#f0c832', automatic:'#9a80e0' };
        html += `
          <div class="autoia-card">
            <div class="autoia-card-title">Funcion Conductual</div>
            <div style="color:${fnColors[fn]||'#9aaac8'};font-size:13px;font-weight:700;margin-bottom:4px">${fn.toUpperCase()}</div>
            <div style="color:#9aaac8;font-size:11px">${AutoiaClient.functionLabel(fn)}</div>
          </div>
        `;
      }

      const dist = aba.distribution || {};
      if (Object.keys(dist).length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">Matching Law — Distribucion conductual</div>';
        Object.entries(dist).forEach(([k, v]) => {
          const pct = (v * 100).toFixed(1);
          html += `
            <div class="prob-row">
              <div class="prob-label">${k}</div>
              <div class="prob-bar-wrap"><div class="prob-bar" style="width:${pct}%;background:#4a90e2"></div></div>
              <div class="prob-pct">${pct}%</div>
            </div>
          `;
        });
        html += '</div>';
      }

      return html;
    }

    // ── Tab: Historial ─────────────────────────────────────────────────────

    _htmlHistory() {
      let html = '';

      // Cargar historial del servidor en background la primera vez
      if (this._historyLocal.length === 0) {
        this.client.recentPredictions(20).then(data => {
          if (data.predictions) {
            this._historyLocal = data.predictions.map(p => ({ ...p, _fromServer: true }));
            if (this.tab === 'hist') this._renderBody();
          }
        }).catch(() => {});
        return '<div class="autoia-loading">⟳ Cargando historial...</div>';
      }

      // Precision del motor
      this.client.accuracy().then(acc => {
        const el = document.getElementById('autoia-acc-section');
        if (!el || !acc) return;
        const domains = { sports: acc.sports, market: acc.market, masses: acc.masses, betting: acc.betting };
        let accHtml = '';
        Object.entries(domains).forEach(([d, v]) => {
          if (!v || !v.total) return;
          const a = v.accuracy || 0;
          const c = AutoiaClient.confidenceColor(a);
          accHtml += `
            <div class="accuracy-row">
              <div class="accuracy-label">${d}</div>
              <div class="accuracy-bar-wrap"><div class="accuracy-bar" style="width:${(a*100).toFixed(0)}%;background:${c}"></div></div>
              <div class="accuracy-pct" style="color:${c}">${(a*100).toFixed(0)}%</div>
            </div>
          `;
        });
        el.innerHTML = accHtml || '<div style="color:#445;font-size:11px">Sin datos de precision todavia</div>';
      }).catch(() => {});

      html += `
        <div class="autoia-card">
          <div class="autoia-card-title">Precision del motor</div>
          <div id="autoia-acc-section"><div style="color:#445;font-size:11px">Cargando...</div></div>
        </div>
        <div class="autoia-card">
          <div class="autoia-card-title">Predicciones recientes</div>
      `;

      const items = this._historyLocal.slice(0, 15);
      if (!items.length) {
        html += '<div class="autoia-empty" style="font-size:11px;padding:12px 0">Sin predicciones todavia</div>';
      } else {
        items.forEach((p, i) => {
          const correct = p.was_correct;
          const badgeClass = correct === true ? 'ok' : correct === false ? 'fail' : 'pend';
          const badgeIcon  = correct === true ? '✓'  : correct === false ? '✗'  : '?';
          const conf = p.confidence || 0;
          const cc   = AutoiaClient.confidenceColor(conf);
          const pred = p.prediction || p.predicted_outcome || '?';
          const subj = p.subject || '?';
          const dom  = (p.domain || '').toUpperCase().slice(0,3);
          html += `
            <div class="hist-row" onclick="AutoiaWidget._instance._histClick(${i})">
              <div class="hist-badge ${badgeClass}">${badgeIcon}</div>
              <div class="hist-info">
                <div class="hist-subject">${subj}</div>
                <div class="hist-pred">${dom} · ${pred}</div>
              </div>
              <div class="hist-conf" style="color:${cc}">${(conf*100).toFixed(0)}%</div>
            </div>
          `;
        });
      }
      html += '</div>';

      // Boton para registrar resultado de la ultima prediccion
      if (this.lastMatch) {
        html += `
          <div class="autoia-card">
            <div class="autoia-card-title">Registrar resultado real</div>
            <div style="color:#9aaac8;font-size:11px;margin-bottom:7px">
              Partido: <strong style="color:#c8d0e8">${this.lastMatch.home} vs ${this.lastMatch.away}</strong>
            </div>
            <div style="display:flex;gap:6px">
              <button class="small-btn" onclick="AutoiaWidget._instance._registerResult('${this.lastMatch.home}')">
                ${this.lastMatch.home.split(' ').pop()} gana
              </button>
              <button class="small-btn" onclick="AutoiaWidget._instance._registerResult('draw')">
                Empate
              </button>
              <button class="small-btn" onclick="AutoiaWidget._instance._registerResult('${this.lastMatch.away}')">
                ${this.lastMatch.away.split(' ').pop()} gana
              </button>
            </div>
          </div>
        `;
      }

      return html;
    }

    async _histClick(i) {
      const p = this._historyLocal[i];
      if (!p) return;
      const parts = (p.subject || '').split(' vs ');
      if (parts.length === 2) {
        document.getElementById('autoia-input').value = p.subject;
        this.tab = 'pred';
        this._renderTabs();
        this.lastMatch = { home: parts[0].trim(), away: parts[1].trim() };
        await this._loadPrediction(this.lastMatch);
      }
    }

    async _registerResult(outcome) {
      if (!this.lastMatch) return;
      const subject = `${this.lastMatch.home} vs ${this.lastMatch.away}`;
      await this.client.recordOutcome(subject, 'sports', outcome);
      // Marcar la primera prediccion pendiente como resuelta en local
      const pending = this._historyLocal.find(p => p.subject === subject && p.was_correct == null);
      if (pending) {
        pending.was_correct = (pending.prediction === outcome || pending.predicted_outcome === outcome);
        pending.actual_outcome = outcome;
      }
      this._renderBody();
    }

    // ── Tab: Analizar ──────────────────────────────────────────────────────

    _htmlInput() {
      return `
        <div class="autoia-card">
          <div class="autoia-card-title">Formatos de entrada</div>
          <div style="color:#9aaac8;font-size:11px;line-height:1.7">
            <strong style="color:#4a90e2">Solo partido:</strong><br>
            &nbsp;Real Madrid vs Barcelona<br>
            <strong style="color:#4a90e2">Con cuotas:</strong><br>
            &nbsp;Real Madrid vs Barcelona | 1.85 / 3.60 / 4.20<br>
            <strong style="color:#4a90e2">Titular de noticia:</strong><br>
            &nbsp;Lewandowski baja por lesion 3 semanas<br>
            <span style="color:#3ddc84">El motor detecta EOs y AOs automaticamente.</span>
          </div>
        </div>
        <div class="autoia-card">
          <div class="autoia-card-title">Enviar noticia como MO</div>
          <textarea id="autoia-news-txt" style="width:100%;background:#0d1117;border:1px solid #2a3560;border-radius:6px;padding:7px;color:#c8d0e8;font-size:11px;font-family:inherit;resize:vertical;min-height:60px;outline:none" placeholder="Pega titulares, tweets o noticias sobre el partido..."></textarea>
          <div style="display:flex;gap:6px;margin-top:7px">
            <button class="small-btn" style="flex:1" onclick="AutoiaWidget._instance._sendNewsText('EO')">▲ Enviar como EO</button>
            <button class="small-btn" style="flex:1" onclick="AutoiaWidget._instance._sendNewsText('AO')">▼ Enviar como AO</button>
            <button class="small-btn" style="flex:1" onclick="AutoiaWidget._instance._sendNewsText('auto')">Auto-detectar</button>
          </div>
          <div id="autoia-news-status" style="font-size:10px;margin-top:5px;color:#445"></div>
        </div>
      `;
    }

    async _sendNewsText(moType) {
      const txt = document.getElementById('autoia-news-txt')?.value.trim();
      if (!txt) return;
      const status = document.getElementById('autoia-news-status');
      if (status) status.textContent = '⟳ Enviando...';
      try {
        if (moType === 'auto') {
          await this.client.analyzeSentiment([txt], 'sports');
        } else {
          await this.client.sendMO({
            moType, description: txt, domain: 'sports',
            target: this.lastMatch?.home || 'general', strength: 0.6, durationH: 24
          });
        }
        if (status) {
          status.style.color = '#3ddc84';
          status.textContent = '✓ MO enviado al motor ABA';
        }
        // Refrescar prediccion despues de nuevo MO
        if (this.lastMatch) setTimeout(() => this._silentRefresh(), 500);
      } catch(e) {
        if (status) { status.style.color = '#e05252'; status.textContent = '✕ Error al enviar'; }
      }
    }

    _renderError(msg) {
      document.getElementById('autoia-body').innerHTML =
        `<div class="autoia-error">${msg}</div>`;
    }
  }

  AutoiaWidget._instance = null;
  global.AutoiaWidget = AutoiaWidget;

})(typeof window !== 'undefined' ? window : global);
