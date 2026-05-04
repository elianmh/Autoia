/**
 * Autoia ABA Client — SDK JavaScript para TheTigerBets
 *
 * Conecta tu web directamente con el motor ABA de Autoia.
 * Sin dependencias externas — vanilla JS puro.
 *
 * Uso rapido:
 *   const autoia = new AutoiaClient({ host: 'localhost', port: 8765 });
 *   const pred   = await autoia.predictSports('Real Madrid', 'Barcelona');
 *   console.log(pred.prediction, pred.confidence);
 */

class AutoiaClient {
  /**
   * @param {Object} config
   * @param {string} config.host    - Host del servidor Autoia (default: localhost)
   * @param {number} config.port    - Puerto (default: 8765)
   * @param {boolean} config.debug  - Logs en consola
   */
  constructor(config = {}) {
    this.host    = config.host  || 'localhost';
    this.port    = config.port  || 8765;
    this.debug   = config.debug || false;
    this.baseUrl = `http://${this.host}:${this.port}`;
    this._cache  = new Map();
    this._cacheTTL = config.cacheTTL || 30000; // 30s
  }

  // ── Core HTTP ────────────────────────────────────────────────────────────

  async _get(path, params = {}) {
    const qs  = new URLSearchParams(params).toString();
    const url = `${this.baseUrl}${path}${qs ? '?' + qs : ''}`;

    // Cache GET requests
    const cached = this._cache.get(url);
    if (cached && Date.now() - cached.ts < this._cacheTTL) return cached.data;

    const res  = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);

    this._cache.set(url, { ts: Date.now(), data });
    if (this.debug) console.log('[Autoia] GET', path, data);
    return data;
  }

  async _post(path, body = {}) {
    const url = `${this.baseUrl}${path}`;
    const res  = await fetch(url, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    if (this.debug) console.log('[Autoia] POST', path, data);
    return data;
  }

  // ── Status ───────────────────────────────────────────────────────────────

  /** Verifica si el servidor Autoia esta activo */
  async status() {
    return this._get('/status');
  }

  /** Precision historica del motor por dominio */
  async accuracy() {
    return this._get('/accuracy');
  }

  // ── Predicciones ─────────────────────────────────────────────────────────

  /**
   * Prediccion de partido deportivo con analisis ABA completo.
   * @param {string} home     - Equipo local
   * @param {string} away     - Equipo visitante
   * @returns {PredictionResult}
   */
  async predictSports(home, away) {
    return this._get('/predict', { domain: 'sports', home, away });
  }

  /**
   * Prediccion de mercado financiero.
   * @param {string} symbol - Ej: "BTC-USD", "AAPL", "^GSPC"
   */
  async predictMarket(symbol) {
    return this._get('/predict', { domain: 'market', symbol });
  }

  /**
   * Prediccion de tendencia de masas sobre un topico.
   * @param {string} topic - Ej: "bitcoin", "real madrid", "elecciones"
   */
  async predictMasses(topic) {
    return this._get('/predict', { domain: 'masses', topic });
  }

  /**
   * Analisis de valor en apuesta (value bet).
   * @param {string} home      - Equipo local
   * @param {string} away      - Equipo visitante
   * @param {number} homeOdds  - Cuota local (decimal europeo)
   * @param {number} drawOdds  - Cuota empate
   * @param {number} awayOdds  - Cuota visitante
   */
  async predictBetting(home, away, homeOdds, drawOdds, awayOdds) {
    return this._get('/predict', {
      domain:     'betting',
      home, away,
      home_odds:  homeOdds,
      draw_odds:  drawOdds,
      away_odds:  awayOdds,
    });
  }

  /** Ultimas N predicciones generadas */
  async recentPredictions(n = 10) {
    return this._get('/predictions', { n });
  }

  /** Lista todos los equipos registrados con stats y MOs activos */
  async getTeams() {
    return this._get('/teams');
  }

  /**
   * Analisis completo de un partido especifico.
   * Retorna prediccion + stats detalladas de ambos equipos.
   */
  async getMatchAnalysis(home, away) {
    const h = encodeURIComponent(home);
    const a = encodeURIComponent(away);
    return this._get(`/match/${h}/${a}`);
  }

  /** Lista todos los MOs activos con fuerza y tiempo restante */
  async getActiveMOs() {
    return this._get('/mo/active');
  }

  // ── Envio de datos ───────────────────────────────────────────────────────

  /**
   * Envia cuotas de un partido al motor ABA.
   * Ideal para alimentar cuotas directamente desde TheTigerBets.
   */
  async sendOdds({ home, away, homeOdds, drawOdds, awayOdds, league = '' }) {
    return this._post('/data', {
      source:    'thetiger_bets',
      domain:    'betting',
      data_type: 'odds',
      payload:   {
        home_team:  home,
        away_team:  away,
        home_odds:  homeOdds,
        draw_odds:  drawOdds,
        away_odds:  awayOdds,
        league,
      },
    });
  }

  /**
   * Envia estadisticas de un equipo al motor.
   */
  async sendTeamStats({ team, league, result, goalsFor, goalsAgainst, injuries = [], keyReturns = [] }) {
    return this._post('/data', {
      source:    'thetiger_bets',
      domain:    'sports',
      data_type: 'team_stats',
      payload:   {
        team, league, result,
        goals_for:      goalsFor,
        goals_against:  goalsAgainst,
        injuries,
        key_returns:    keyReturns,
      },
    });
  }

  /**
   * Registra una Operacion Motivadora (noticia, evento, lesion).
   * @param {string} moType     - "EO" (activa, hype) o "AO" (suprime, lesion)
   * @param {string} description - Descripcion del evento
   * @param {string} domain     - "sports"|"market"|"masses"|"betting"
   * @param {number} strength   - Intensidad 0.0-1.0
   * @param {number} durationH  - Horas que dura el efecto
   */
  async sendMO({ moType, description, domain = 'sports', target = '', strength = 0.6, durationH = 24 }) {
    return this._post('/mo', {
      mo_type:     moType,
      description,
      domain,
      target,
      strength,
      duration_h:  durationH,
    });
  }

  /**
   * Analiza texto para detectar sentimiento y MOs automaticamente.
   * Envia articulos, noticias, titulares, tweets sobre un partido o mercado.
   */
  async analyzeSentiment(texts, domain = 'masses') {
    const arr = Array.isArray(texts) ? texts : [texts];
    return this._post('/sentiment', { texts: arr, domain });
  }

  /**
   * Registra el resultado real de un evento para que el motor aprenda.
   * Llamar SIEMPRE despues de conocer el resultado.
   * @param {string} subject  - "Real Madrid vs Barcelona"
   * @param {string} domain   - "sports"|"market"|"betting"
   * @param {string} outcome  - El resultado real: "Real Madrid", "up", "draw", etc.
   */
  async recordOutcome(subject, domain, outcome) {
    return this._post('/outcome', { subject, domain, outcome });
  }

  /**
   * Registra el resultado de un equipo (W/D/L) para actualizar momentum.
   */
  async sendTeamResult({ team, result, league = '', goalsFor = 0, goalsAgainst = 0 }) {
    return this._post('/team/result', {
      team, result, league,
      goals_for:      goalsFor,
      goals_against:  goalsAgainst,
    });
  }

  /**
   * Registra multiples datos en una sola peticion (mas eficiente).
   */
  async sendBatch(items) {
    return this._post('/data/batch', { items });
  }

  // ── Utilidades ───────────────────────────────────────────────────────────

  /** Formatea confianza como porcentaje con color CSS */
  static confidenceColor(confidence) {
    if (confidence > 0.65) return '#3ddc84';   // verde
    if (confidence > 0.45) return '#f0c832';   // amarillo
    return '#e05252';                           // rojo
  }

  /** Descripcion ABA de la funcion conductual */
  static functionLabel(fn) {
    return {
      tangible:  'Busca ganancia monetaria',
      escape:    'Evita perdida / miedo',
      attention: 'Busca validacion social / FOMO',
      automatic: 'Habito / inercia / VR schedule',
    }[fn] || fn;
  }

  /** Icono de MO */
  static moIcon(moType) {
    return moType === 'EO' ? '▲' : '▼';
  }
}

// Exportar para uso como modulo o script directo
if (typeof module !== 'undefined' && module.exports) {
  module.exports = AutoiaClient;
} else {
  window.AutoiaClient = AutoiaClient;
}
