"""
API REST para integracion con apps externas.

Expone endpoints HTTP para que apps externas puedan:
- Enviar datos al motor ABA (POST /data)
- Consultar predicciones (GET /predict)
- Registrar MOs (POST /mo)
- Ver estado del sistema (GET /status)
- Recibir webhooks (POST /webhook/register)

Solo usa stdlib: http.server + json + urllib.
Sin dependencias externas.

Arrancar:
    server = APIServer(bus, engine, port=8765)
    server.start()   # en hilo separado
    server.stop()
"""

import json
import time
import threading
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Callable

from .base_plugin import DataPoint, PredictionResult

logger = logging.getLogger("autoia.integrations.api")


class ABARequestHandler(BaseHTTPRequestHandler):
    """Handler HTTP para la API REST."""

    # Silenciar logs del servidor HTTP
    def log_message(self, fmt, *args):
        logger.debug(f"HTTP: {fmt % args}")

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, msg: str, status: int = 400):
        self._send_json({"error": msg, "status": status}, status)

    def _read_body(self) -> Optional[dict]:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw.decode("utf-8"))
        except Exception as e:
            logger.warning(f"Error leyendo body: {e}")
            return None

    def do_OPTIONS(self):
        """CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-API-Key")
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")
        params = parse_qs(parsed.query)

        bus    = self.server.bus
        engine = self.server.engine

        # ── GET /status ───────────────────────────────────────────────
        if path == "/status":
            self._send_json({
                "status":    "ok",
                "version":   "1.0",
                "timestamp": time.time(),
                "bus":       bus.get_status() if bus else {},
            })

        # ── GET /predict ─────────────────────────────────────────────
        # ?domain=sports&home=Real+Madrid&away=Barcelona
        elif path == "/predict":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            domain = params.get("domain", ["sports"])[0]
            try:
                if domain == "sports":
                    home = params.get("home", [""])[0]
                    away = params.get("away", [""])[0]
                    if not (home and away):
                        return self._send_error("Parametros requeridos: home, away")
                    pred = engine.predict_sports_match(home, away)
                elif domain == "market":
                    symbol = params.get("symbol", ["BTC-USD"])[0]
                    pred = engine.predict_market(symbol)
                elif domain == "masses":
                    topic = params.get("topic", ["general"])[0]
                    pred = engine.predict_mass_trend(topic)
                elif domain == "betting":
                    home = params.get("home", [""])[0]
                    away = params.get("away", [""])[0]
                    if not (home and away):
                        return self._send_error("Parametros requeridos: home, away")
                    pred = engine.predict_betting_value(
                        home, away,
                        float(params.get("home_odds", [2.0])[0]),
                        float(params.get("draw_odds", [3.5])[0]),
                        float(params.get("away_odds", [3.0])[0]),
                    )
                else:
                    return self._send_error(f"Dominio desconocido: {domain}")
                self._send_json(pred.to_dict())
            except Exception as e:
                self._send_error(str(e), 500)

        # ── GET /predictions ──────────────────────────────────────────
        # Ultimas N predicciones
        elif path == "/predictions":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            n = int(params.get("n", [10])[0])
            self._send_json({"predictions": engine.get_recent_predictions(n)})

        # ── GET /accuracy ─────────────────────────────────────────────
        elif path == "/accuracy":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            self._send_json(engine.get_global_accuracy())

        # ── GET /plugins ──────────────────────────────────────────────
        elif path == "/plugins":
            self._send_json({"plugins": bus.list_plugins() if bus else []})

        else:
            self._send_error(f"Ruta no encontrada: {path}", 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path.rstrip("/")

        bus    = self.server.bus
        engine = self.server.engine

        body = self._read_body()
        if body is None:
            return self._send_error("JSON invalido en el body")

        # ── POST /data ────────────────────────────────────────────────
        # Enviar un DataPoint al motor ABA
        if path == "/data":
            try:
                dp = DataPoint(
                    source    = body.get("source", "external"),
                    domain    = body.get("domain", "general"),
                    data_type = body.get("data_type", "raw"),
                    payload   = body.get("payload", body),
                    confidence= float(body.get("confidence", 1.0)),
                    tags      = body.get("tags", []),
                )
                if bus:
                    bus.process_data_point(dp)
                else:
                    logger.info(f"DataPoint recibido (sin bus): {dp.source}/{dp.data_type}")
                self._send_json({"ok": True, "processed": dp.to_dict()})
            except Exception as e:
                self._send_error(str(e), 500)

        # ── POST /data/batch ──────────────────────────────────────────
        # Enviar multiples DataPoints de una vez
        elif path == "/data/batch":
            items = body.get("items", [])
            processed = 0
            errors = []
            for item in items:
                try:
                    dp = DataPoint(
                        source    = item.get("source", "external"),
                        domain    = item.get("domain", "general"),
                        data_type = item.get("data_type", "raw"),
                        payload   = item.get("payload", item),
                        confidence= float(item.get("confidence", 1.0)),
                        tags      = item.get("tags", []),
                    )
                    if bus:
                        bus.process_data_point(dp)
                    processed += 1
                except Exception as e:
                    errors.append(str(e))
            self._send_json({"ok": True, "processed": processed, "errors": errors})

        # ── POST /mo ──────────────────────────────────────────────────
        # Registrar una Operacion Motivadora
        elif path == "/mo":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            try:
                engine.add_mo(
                    domain      = body.get("domain", "general"),
                    source      = body.get("source", "external"),
                    description = body.get("description", ""),
                    mo_type     = body.get("mo_type", "EO"),
                    target      = body.get("target", "general"),
                    strength    = float(body.get("strength", 0.5)),
                    duration_h  = float(body.get("duration_h", 4.0)),
                )
                self._send_json({"ok": True, "mo_registered": True})
            except Exception as e:
                self._send_error(str(e), 500)

        # ── POST /outcome ─────────────────────────────────────────────
        # Registrar resultado real para aprendizaje
        elif path == "/outcome":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            try:
                engine.record_outcome(
                    subject = body.get("subject", ""),
                    domain  = body.get("domain", ""),
                    outcome = body.get("outcome", ""),
                )
                self._send_json({"ok": True, "outcome_recorded": True})
            except Exception as e:
                self._send_error(str(e), 500)

        # ── POST /sentiment ───────────────────────────────────────────
        # Analizar texto y añadir al sentimiento masivo
        elif path == "/sentiment":
            if not engine:
                return self._send_error("Motor ABA no disponible", 503)
            texts = body.get("texts", [body.get("text", "")])
            domain = body.get("domain", "masses")
            results = []
            for text in texts:
                if text:
                    signal = engine.sentiment._analyze_with_patterns(text, "external_api")
                    results.append({
                        "score":    signal.score,
                        "mo_type":  signal.mo_type,
                        "function": signal.function,
                        "summary":  signal.summary,
                    })
            self._send_json({"ok": True, "signals": results,
                             "aggregate": engine.sentiment.get_aggregate_sentiment()})

        # ── POST /webhook/register ────────────────────────────────────
        # Registrar URL para recibir predicciones automaticamente
        elif path == "/webhook/register":
            url      = body.get("url", "")
            domains  = body.get("domains", [])
            secret   = body.get("secret", "")
            if not url:
                return self._send_error("Campo 'url' requerido")
            if bus:
                # Crear plugin de webhook dinamicamente
                from .plugins.webhook_plugin import WebhookPlugin
                plugin = WebhookPlugin(url=url, domains=domains, secret=secret)
                bus.register(plugin, auto_connect=True)
                self._send_json({"ok": True, "webhook_id": plugin.name,
                                  "url": url, "domains": domains})
            else:
                self._send_error("Bus no disponible", 503)

        else:
            self._send_error(f"Ruta no encontrada: {path}", 404)


class APIServer:
    """
    Servidor HTTP REST para integracion con apps externas.

    Endpoints:
        GET  /status               — Estado del sistema
        GET  /predict              — Generar prediccion
        GET  /predictions          — Ultimas predicciones
        GET  /accuracy             — Precision del motor
        GET  /plugins              — Plugins registrados
        POST /data                 — Enviar DataPoint
        POST /data/batch           — Enviar multiples DataPoints
        POST /mo                   — Registrar MO (EO/AO)
        POST /outcome              — Registrar resultado real
        POST /sentiment            — Analizar texto
        POST /webhook/register     — Registrar URL de webhook
    """

    def __init__(self, bus=None, engine=None, host: str = "0.0.0.0", port: int = 8765):
        self.host   = host
        self.port   = port
        self.bus    = bus
        self.engine = engine
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self, blocking: bool = False):
        """
        Inicia el servidor HTTP.
        blocking=False: inicia en hilo daemon (recomendado).
        blocking=True:  bloquea hasta Ctrl+C.
        """
        self._server = HTTPServer((self.host, self.port), ABARequestHandler)
        self._server.bus    = self.bus
        self._server.engine = self.engine

        logger.info(f"API REST iniciada en http://{self.host}:{self.port}")
        self._log_endpoints()

        if blocking:
            try:
                self._server.serve_forever()
            except KeyboardInterrupt:
                self.stop()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever,
                daemon=True,
                name="autoia-api-server",
            )
            self._thread.start()

    def stop(self):
        if self._server:
            self._server.shutdown()
            logger.info("API REST detenida")

    def _log_endpoints(self):
        base = f"http://localhost:{self.port}"
        logger.info(f"  GET  {base}/status")
        logger.info(f"  GET  {base}/predict?domain=sports&home=X&away=Y")
        logger.info(f"  GET  {base}/predictions?n=10")
        logger.info(f"  POST {base}/data       -- DataPoint JSON")
        logger.info(f"  POST {base}/mo         -- MO (EO/AO)")
        logger.info(f"  POST {base}/outcome    -- resultado real")
        logger.info(f"  POST {base}/sentiment  -- texto a analizar")
