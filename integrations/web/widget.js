/**
 * Autoia Widget — Asistente ABA embebido para TheTigerBets
 *
 * Inserta una burbuja flotante que muestra analisis ABA en tiempo real.
 * Se adapta automaticamente a los partidos/cuotas que detecta en la pagina.
 *
 * Integracion de 2 lineas:
 *   <script src="/autoia/widget.js"></script>
 *   <script>AutoiaWidget.init({ host: 'localhost', port: 8765 });</script>
 */

(function(global) {
  'use strict';

  // ── Estilos del widget ─────────────────────────────────────────────────
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
      width: 360px; max-height: 520px;
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
      padding: 12px 16px; display: flex; align-items: center; gap: 10px;
      border-bottom: 1px solid #2a3560;
    }
    .autoia-header .logo { font-size: 20px; }
    .autoia-header .title { color: #e8eaf0; font-size: 14px; font-weight: 700; flex: 1; }
    .autoia-header .subtitle { color: #7888aa; font-size: 10px; }
    .autoia-close {
      color: #7888aa; cursor: pointer; font-size: 18px; line-height: 1;
      padding: 2px 6px; border-radius: 4px;
    }
    .autoia-close:hover { color: #e8eaf0; background: rgba(255,255,255,0.07); }

    .autoia-tabs {
      display: flex; background: #0d1117; border-bottom: 1px solid #1e2a50;
    }
    .autoia-tab {
      flex: 1; padding: 9px 4px; text-align: center; font-size: 11px; font-weight: 600;
      color: #556; cursor: pointer; transition: color 0.15s;
      border-bottom: 2px solid transparent;
    }
    .autoia-tab:hover { color: #aab; }
    .autoia-tab.active { color: #4a90e2; border-bottom-color: #4a90e2; }

    .autoia-body { flex: 1; overflow-y: auto; padding: 14px; }
    .autoia-body::-webkit-scrollbar { width: 4px; }
    .autoia-body::-webkit-scrollbar-track { background: transparent; }
    .autoia-body::-webkit-scrollbar-thumb { background: #2a3560; border-radius: 2px; }

    .autoia-card {
      background: #131820; border: 1px solid #1e2a50; border-radius: 10px;
      padding: 12px; margin-bottom: 10px;
    }
    .autoia-card-title {
      color: #7888aa; font-size: 10px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.8px; margin-bottom: 8px;
    }

    .pred-match { color: #c8d0e8; font-size: 13px; font-weight: 600; margin-bottom: 4px; }
    .pred-winner {
      font-size: 18px; font-weight: 800; margin: 6px 0 4px;
    }
    .pred-conf { font-size: 11px; color: #7888aa; margin-bottom: 8px; }

    .prob-row { display: flex; align-items: center; gap: 8px; margin-bottom: 5px; }
    .prob-label { color: #9aaac8; font-size: 11px; min-width: 90px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .prob-bar-wrap { flex: 1; height: 8px; background: #1e2a50; border-radius: 4px; overflow: hidden; }
    .prob-bar { height: 100%; border-radius: 4px; transition: width 0.4s; }
    .prob-pct { color: #9aaac8; font-size: 11px; min-width: 36px; text-align: right; }

    .mo-row {
      display: flex; align-items: flex-start; gap: 8px;
      padding: 7px 0; border-bottom: 1px solid #1a2040;
    }
    .mo-row:last-child { border-bottom: none; }
    .mo-badge {
      font-size: 10px; font-weight: 800; padding: 2px 6px;
      border-radius: 4px; flex-shrink: 0; margin-top: 1px;
    }
    .mo-badge.EO { background: rgba(61,220,132,0.15); color: #3ddc84; }
    .mo-badge.AO { background: rgba(220,80,80,0.15);  color: #e05252; }
    .mo-desc { color: #c8d0e8; font-size: 12px; line-height: 1.4; }

    .factor-row { display: flex; gap: 8px; margin-bottom: 5px; align-items: flex-start; }
    .factor-icon { font-size: 12px; flex-shrink: 0; margin-top: 1px; }
    .factor-text { color: #c8d0e8; font-size: 11px; line-height: 1.4; }

    .aba-term { display: flex; gap: 8px; margin-bottom: 5px; align-items: center; }
    .aba-abbr {
      min-width: 28px; height: 22px; border-radius: 4px;
      font-size: 10px; font-weight: 800; display: flex; align-items: center; justify-content: center;
    }
    .aba-desc { color: #8899bb; font-size: 11px; }

    .autoia-input-area {
      padding: 10px 14px; border-top: 1px solid #1e2a50;
      background: #0d1117;
    }
    .autoia-input-row { display: flex; gap: 8px; }
    .autoia-input {
      flex: 1; background: #131820; border: 1px solid #2a3560;
      border-radius: 8px; padding: 8px 12px; color: #c8d0e8;
      font-size: 12px; outline: none;
    }
    .autoia-input:focus { border-color: #4a90e2; }
    .autoia-send {
      background: #4a90e2; color: #fff; border: none; border-radius: 8px;
      padding: 8px 14px; cursor: pointer; font-size: 13px; font-weight: 600;
    }
    .autoia-send:hover { background: #5aa0f2; }

    .autoia-empty { color: #556; font-size: 12px; text-align: center; padding: 20px 0; }
    .autoia-loading { color: #4a90e2; font-size: 12px; text-align: center; padding: 16px 0; }
    .autoia-error { color: #e05252; font-size: 12px; text-align: center; padding: 12px; }
  `;

  // ── Widget ─────────────────────────────────────────────────────────────
  class AutoiaWidget {

    static init(config = {}) {
      const w = new AutoiaWidget(config);
      w.mount();
      return w;
    }

    constructor(config) {
      this.client  = new AutoiaClient(config);
      this.tab     = 'pred';   // pred | aba | input
      this.open    = false;
      this.loading = false;
      this.prediction = null;
      this.lastMatch  = null;
      this.inputText  = '';
      this._autoDetect = config.autoDetect !== false;
    }

    mount() {
      // Inyectar CSS
      if (!document.getElementById('autoia-styles')) {
        const style = document.createElement('style');
        style.id = 'autoia-styles';
        style.textContent = CSS;
        document.head.appendChild(style);
      }

      // Crear DOM
      document.body.insertAdjacentHTML('beforeend', `
        <div id="autoia-widget">
          <div id="autoia-fab" title="Autoia ABA Assistant">
            <span class="fab-icon">🐯</span>
            <span class="fab-badge" id="autoia-badge">!</span>
          </div>
          <div id="autoia-panel">
            <div class="autoia-header">
              <span class="logo">🐯</span>
              <div>
                <div class="title">Autoia ABA</div>
                <div class="subtitle">Motor de prediccion conductual</div>
              </div>
              <span class="autoia-close" id="autoia-close-btn">✕</span>
            </div>
            <div class="autoia-tabs">
              <div class="autoia-tab active" data-tab="pred">Prediccion</div>
              <div class="autoia-tab" data-tab="aba">Marco ABA</div>
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
            </div>
          </div>
        </div>
      `);

      // Eventos
      document.getElementById('autoia-fab')
        .addEventListener('click', () => this.toggle());
      document.getElementById('autoia-close-btn')
        .addEventListener('click', () => this.close());
      document.querySelectorAll('.autoia-tab')
        .forEach(t => t.addEventListener('click', () => {
          this.tab = t.dataset.tab;
          this._renderTabs();
          this._renderBody();
        }));
      document.getElementById('autoia-analyze-btn')
        .addEventListener('click', () => this._analyzeInput());
      document.getElementById('autoia-input')
        .addEventListener('keydown', e => {
          if (e.key === 'Enter') this._analyzeInput();
        });

      // Auto-detectar partido en pagina si esta en TheTigerBets
      if (this._autoDetect) {
        setTimeout(() => this._autoDetectMatch(), 1500);
      }
    }

    toggle() {
      this.open ? this.close() : this.open_panel();
    }
    open_panel() {
      this.open = true;
      document.getElementById('autoia-panel').classList.add('open');
      document.getElementById('autoia-badge').style.display = 'none';
      if (!this.prediction && this.lastMatch) this._loadPrediction(this.lastMatch);
    }
    close() {
      this.open = false;
      document.getElementById('autoia-panel').classList.remove('open');
    }

    // ── Auto-deteccion ─────────────────────────────────────────────────

    _autoDetectMatch() {
      // Busca patrones como "Real Madrid vs Barcelona" en el DOM
      const text = document.body.innerText || '';
      const vsPattern = /([A-Z][a-zA-Z\s\.]{2,25})\s+(?:vs\.?|contra|v\.?)\s+([A-Z][a-zA-Z\s\.]{2,25})/gi;
      const matches = [...text.matchAll(vsPattern)];
      if (matches.length === 0) return;

      const [, home, away] = matches[0];
      const cleanHome = home.trim();
      const cleanAway = away.trim();
      if (cleanHome.length < 3 || cleanAway.length < 3) return;

      this.lastMatch = { home: cleanHome, away: cleanAway };
      document.getElementById('autoia-input').value = `${cleanHome} vs ${cleanAway}`;

      // Mostrar badge en FAB
      document.getElementById('autoia-badge').style.display = 'block';
    }

    // ── Input parser ───────────────────────────────────────────────────

    _parseInput(text) {
      // Formato: "Real Madrid vs Barcelona" o "Real Madrid vs Barcelona | 1.85 / 3.60 / 4.20"
      const parts  = text.split('|');
      const match  = parts[0].trim();
      const vsMatch = match.match(/^(.+?)\s+(?:vs\.?|contra|v\.?)\s+(.+)$/i);
      if (!vsMatch) return null;

      const home = vsMatch[1].trim();
      const away = vsMatch[2].trim();

      // Cuotas opcionales: 1.85 / 3.60 / 4.20
      let odds = null;
      if (parts[1]) {
        const nums = parts[1].match(/[\d]+\.[\d]+/g);
        if (nums && nums.length >= 2) {
          odds = {
            home: parseFloat(nums[0]),
            draw: parseFloat(nums[1]),
            away: parseFloat(nums[2] || nums[1]),
          };
        }
      }
      return { home, away, odds };
    }

    async _analyzeInput() {
      const raw = document.getElementById('autoia-input').value.trim();
      if (!raw) return;
      const parsed = this._parseInput(raw);
      if (!parsed) {
        this._renderError('Formato: "Real Madrid vs Barcelona" o "... | 1.85 / 3.60 / 4.20"');
        return;
      }
      this.lastMatch = parsed;
      await this._loadPrediction(parsed);
    }

    async _loadPrediction({ home, away, odds }) {
      this.loading = true;
      this._renderBody();

      try {
        let pred;
        if (odds) {
          pred = await this.client.predictBetting(
            home, away, odds.home, odds.draw, odds.away
          );
        } else {
          pred = await this.client.predictSports(home, away);
        }
        this.prediction = pred;
        this.prediction._home = home;
        this.prediction._away = away;
        this.prediction._odds = odds;
      } catch (e) {
        this.prediction = null;
        this._renderError('No se pudo conectar con Autoia. ¿Esta corriendo? python predict_main.py --serve');
      }

      this.loading = false;
      this._renderBody();
    }

    // ── Render ─────────────────────────────────────────────────────────

    _renderTabs() {
      document.querySelectorAll('.autoia-tab').forEach(t => {
        t.classList.toggle('active', t.dataset.tab === this.tab);
      });
    }

    _renderBody() {
      const body = document.getElementById('autoia-body');
      if (this.loading) {
        body.innerHTML = '<div class="autoia-loading">⟳ Analizando con ABA...</div>';
        return;
      }
      if (!this.prediction) return;

      if (this.tab === 'pred')  body.innerHTML = this._htmlPrediction();
      if (this.tab === 'aba')   body.innerHTML = this._htmlABA();
      if (this.tab === 'input') body.innerHTML = this._htmlInput();
    }

    _htmlPrediction() {
      const p = this.prediction;
      const conf = p.confidence || 0;
      const confColor = AutoiaClient.confidenceColor(conf);
      const probs = p.probs || p.probabilities || {};

      let html = `
        <div class="autoia-card">
          <div class="autoia-card-title">Prediccion</div>
          <div class="pred-match">${p.subject || (this.lastMatch?.home + ' vs ' + this.lastMatch?.away)}</div>
          <div class="pred-winner" style="color:${confColor}">${p.prediction || '—'}</div>
          <div class="pred-conf">Confianza: <strong style="color:${confColor}">${(conf*100).toFixed(0)}%</strong></div>
      `;

      // Barras de probabilidad
      Object.entries(probs).forEach(([label, pval]) => {
        const pct  = (pval * 100).toFixed(1);
        const isWinner = label === p.prediction;
        html += `
          <div class="prob-row">
            <div class="prob-label" title="${label}">${label}</div>
            <div class="prob-bar-wrap">
              <div class="prob-bar" style="width:${pct}%;background:${isWinner ? confColor : '#2a3560'}"></div>
            </div>
            <div class="prob-pct">${pct}%</div>
          </div>
        `;
      });
      html += '</div>';

      // Factores clave
      const factors   = p.factors || [];
      const risks     = p.risks || [];
      if (factors.length || risks.length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">Factores</div>';
        factors.forEach(f => html += `<div class="factor-row"><span class="factor-icon" style="color:#3ddc84">+</span><div class="factor-text">${f}</div></div>`);
        risks.forEach(r => html += `<div class="factor-row"><span class="factor-icon" style="color:#e05252">!</span><div class="factor-text">${r}</div></div>`);
        html += '</div>';
      }

      // Apuesta con valor (si aplica)
      const bet = p.aba?.edge;
      if (bet && bet !== 'house' && p.domain === 'betting') {
        const ev = (this.prediction.probabilities?.['value_' + bet] || 0) * 100;
        html += `
          <div class="autoia-card" style="border-color:#3ddc84">
            <div class="autoia-card-title" style="color:#3ddc84">Value Bet Detectado</div>
            <div style="color:#3ddc84;font-size:16px;font-weight:800;">${bet.toUpperCase()}</div>
            <div style="color:#9aaac8;font-size:11px">EV estimado: +${ev.toFixed(1)}%</div>
          </div>
        `;
      }

      return html;
    }

    _htmlABA() {
      const p   = this.prediction;
      const aba = p.aba || {};
      const mos = aba.mos || [];

      let html = `
        <div class="autoia-card">
          <div class="autoia-card-title">Contingencia de 4 Terminos</div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a3a2a;color:#3ddc84">MO</div><div class="aba-desc">Operacion Motivadora activa</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a2a40;color:#4a90e2">SD</div><div class="aba-desc">Estimulo discriminativo (cuota, noticia)</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#2a2a1a;color:#f0c832">R</div><div class="aba-desc">Respuesta conductual de la masa</div></div>
          <div class="aba-term"><div class="aba-abbr" style="background:#1a2a1a;color:#3ddc84">C</div><div class="aba-desc">Consecuencia (ganancia o perdida)</div></div>
        </div>
      `;

      if (mos.length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">MOs Activos</div>';
        mos.forEach(mo => {
          const t = mo.mo_type || mo.type || '?';
          html += `
            <div class="mo-row">
              <div class="mo-badge ${t}">${t}</div>
              <div class="mo-desc">${mo.description || mo.source || ''}</div>
            </div>
          `;
        });
        html += '</div>';
      }

      const fn = aba.function;
      if (fn) {
        html += `
          <div class="autoia-card">
            <div class="autoia-card-title">Funcion Conductual</div>
            <div style="color:#f0c832;font-size:13px;font-weight:700;margin-bottom:4px">${fn.toUpperCase()}</div>
            <div style="color:#9aaac8;font-size:11px">${AutoiaClient.functionLabel(fn)}</div>
          </div>
        `;
      }

      const dist = aba.distribution || {};
      if (Object.keys(dist).length) {
        html += '<div class="autoia-card"><div class="autoia-card-title">Matching Law</div>';
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

    _htmlInput() {
      return `
        <div class="autoia-card">
          <div class="autoia-card-title">Enviar Datos al Motor</div>
          <div style="color:#9aaac8;font-size:11px;line-height:1.6">
            Usa el campo de abajo para analizar noticias o titulares:<br><br>
            <strong style="color:#4a90e2">Partido:</strong> Real Madrid vs Barcelona<br>
            <strong style="color:#4a90e2">Con cuotas:</strong> Real Madrid vs Barcelona | 1.85 / 3.60 / 4.20<br>
            <strong style="color:#4a90e2">Titular:</strong> Lewandowski lesionado, baja 3 semanas<br><br>
            <span style="color:#3ddc84">El motor ABA detecta automaticamente EOs y AOs.</span>
          </div>
        </div>
        <div class="autoia-card">
          <div class="autoia-card-title">Registrar Resultado Real</div>
          <div style="color:#9aaac8;font-size:11px;margin-bottom:8px">
            Despues del partido, registra el resultado para que el motor aprenda:
          </div>
          <div style="display:flex;gap:6px;margin-bottom:6px">
            <input id="autoia-outcome-subject" class="autoia-input" style="flex:2"
              placeholder="Real Madrid vs Barcelona" value="${this.lastMatch ? this.lastMatch.home + ' vs ' + this.lastMatch.away : ''}" />
            <input id="autoia-outcome-result" class="autoia-input" style="flex:1"
              placeholder="Real Madrid" />
          </div>
          <button class="autoia-send" style="width:100%;font-size:12px"
            onclick="AutoiaWidget._instance._sendOutcome()">
            Registrar resultado
          </button>
        </div>
      `;
    }

    async _sendOutcome() {
      const subject = document.getElementById('autoia-outcome-subject')?.value.trim();
      const outcome = document.getElementById('autoia-outcome-result')?.value.trim();
      if (!subject || !outcome) return;
      await this.client.recordOutcome(subject, 'sports', outcome);
      alert(`✓ Resultado registrado: ${outcome}\nEl motor ABA aprendera de esto.`);
    }

    _renderError(msg) {
      document.getElementById('autoia-body').innerHTML =
        `<div class="autoia-error">${msg}</div>`;
    }
  }

  // Registro global para callbacks inline
  AutoiaWidget._instance = null;
  const _origInit = AutoiaWidget.init.bind(AutoiaWidget);
  AutoiaWidget.init = function(config) {
    const w = _origInit(config);
    AutoiaWidget._instance = w;
    return w;
  };

  global.AutoiaWidget = AutoiaWidget;

})(typeof window !== 'undefined' ? window : global);
