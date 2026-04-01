"""
Integration Bus — Bus central de integracion.

Gestiona todos los plugins conectados y los sincroniza con el motor ABA.
Ejecuta el fetch de cada plugin en su propio hilo.
Distribuye las predicciones del motor de vuelta a los plugins suscritos.

Uso:
    bus = IntegrationBus(engine)
    bus.register(MiAppPlugin(config={...}))
    bus.start()          # inicia polling en background
    bus.stop()           # detiene todo
"""

import time
import threading
import logging
from typing import Dict, List, Optional, Callable, Type

from .base_plugin import BasePlugin, DataPoint, PredictionResult

logger = logging.getLogger("autoia.integrations.bus")


class IntegrationBus:
    """
    Bus de integracion central.

    Responsabilidades:
    - Registrar y gestionar plugins
    - Hacer polling de datos (fetch) periodicamente
    - Convertir DataPoints en señales ABA para el motor
    - Enviar predicciones de vuelta a los plugins
    - Exponer API REST para integraciones HTTP
    """

    def __init__(self, engine=None, poll_interval: float = 30.0):
        """
        Args:
            engine:        PredictionEngine de Autoia
            poll_interval: segundos entre fetches automaticos
        """
        self.engine        = engine
        self.poll_interval = poll_interval
        self.plugins:      Dict[str, BasePlugin] = {}
        self._running      = False
        self._threads:     List[threading.Thread] = []
        self._lock         = threading.Lock()
        self._data_queue:  List[DataPoint] = []
        self._pred_callbacks: List[Callable] = []
        self._stats = {"total_data_points": 0, "total_predictions": 0}

    # ── Plugin Management ─────────────────────────────────────────────────

    def register(self, plugin: BasePlugin, auto_connect: bool = True) -> bool:
        """
        Registra un plugin en el bus.
        Si auto_connect=True, llama a plugin.connect() inmediatamente.
        """
        with self._lock:
            if plugin.name in self.plugins:
                logger.warning(f"Plugin '{plugin.name}' ya registrado — reemplazando")
            self.plugins[plugin.name] = plugin

        if auto_connect:
            try:
                ok = plugin.connect()
                if ok:
                    logger.info(f"Plugin '{plugin.name}' conectado OK")
                else:
                    logger.warning(f"Plugin '{plugin.name}' fallo al conectar")
                    plugin.enabled = False
                return ok
            except Exception as e:
                logger.error(f"Plugin '{plugin.name}' error al conectar: {e}")
                plugin.enabled = False
                return False

        logger.info(f"Plugin '{plugin.name}' registrado (sin auto-connect)")
        return True

    def unregister(self, name: str):
        """Desconecta y elimina un plugin."""
        plugin = self.plugins.pop(name, None)
        if plugin:
            try:
                plugin.disconnect()
            except Exception:
                pass
            logger.info(f"Plugin '{name}' desregistrado")

    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        return self.plugins.get(name)

    def list_plugins(self) -> List[Dict]:
        return [p.health_check() for p in self.plugins.values()]

    # ── Data Processing ───────────────────────────────────────────────────

    def process_data_point(self, dp: DataPoint):
        """
        Convierte un DataPoint en señal ABA y la inyecta en el motor.
        Este es el corazon de la integracion: traduce cualquier formato externo
        al lenguaje ABA del motor.
        """
        if not self.engine:
            with self._lock:
                self._data_queue.append(dp)
            return

        try:
            if dp.data_type == "sentiment":
                text = dp.payload.get("text", "")
                if text:
                    self.engine.analyze_text_mo(text, domain=dp.domain)

            elif dp.data_type == "odds":
                home = dp.payload.get("home_team", "")
                away = dp.payload.get("away_team", "")
                if home and away:
                    self.engine.betting.add_manual_odds(
                        home, away,
                        dp.payload.get("home_odds", 2.0),
                        dp.payload.get("draw_odds", 3.5),
                        dp.payload.get("away_odds", 3.0),
                    )

            elif dp.data_type == "team_stats":
                team = dp.payload.get("team", "")
                if team:
                    t = self.engine.sports.add_team(team,
                        dp.payload.get("league", ""))
                    if "injuries" in dp.payload:
                        t.injuries = dp.payload["injuries"]
                    if "key_returns" in dp.payload:
                        t.key_returns = dp.payload["key_returns"]
                    if "result" in dp.payload:
                        self.engine.sports.update_team_result(
                            team,
                            dp.payload["result"],
                            dp.payload.get("goals_for", 0),
                            dp.payload.get("goals_against", 0),
                        )

            elif dp.data_type == "mo_signal":
                self.engine.add_mo(
                    domain=dp.domain,
                    source=dp.source,
                    description=dp.payload.get("description", ""),
                    mo_type=dp.payload.get("mo_type", "EO"),
                    target=dp.payload.get("target", "general"),
                    strength=dp.payload.get("strength", 0.5),
                    duration_h=dp.payload.get("duration_h", 4.0),
                )

            elif dp.data_type == "price":
                symbol = dp.payload.get("symbol", "")
                if symbol:
                    self.engine.market.add_symbol(symbol)

            elif dp.data_type == "raw":
                # Datos crudos: pasar texto para analisis de sentimiento
                text = str(dp.payload.get("text", dp.payload))
                self.engine.analyze_text_mo(text, domain=dp.domain)

            self._stats["total_data_points"] += 1
            logger.debug(f"DataPoint procesado: {dp.source}/{dp.data_type} "
                         f"dominio={dp.domain}")

        except Exception as e:
            logger.error(f"Error procesando DataPoint de {dp.source}: {e}")

    def push_prediction_to_plugins(self, result: PredictionResult,
                                    domains: Optional[List[str]] = None):
        """
        Envia una prediccion a todos los plugins interesados en ese dominio.
        """
        for plugin in self.plugins.values():
            if not plugin.enabled or not plugin.connected:
                continue
            if domains and result.domain not in (plugin.supported_domains or [result.domain]):
                continue
            try:
                plugin.push_prediction(result)
                self._stats["total_predictions"] += 1
            except Exception as e:
                logger.error(f"Error enviando prediccion a {plugin.name}: {e}")

    # ── Polling loop ──────────────────────────────────────────────────────

    def _poll_plugin(self, plugin: BasePlugin):
        """Loop de polling para un plugin especifico."""
        logger.info(f"Polling iniciado para plugin '{plugin.name}'")
        while self._running and plugin.enabled:
            try:
                data_points = plugin.fetch()
                plugin._record_fetch()
                for dp in (data_points or []):
                    self.process_data_point(dp)
            except Exception as e:
                plugin._record_error()
                logger.error(f"Error fetch plugin '{plugin.name}': {e}")
            time.sleep(self.poll_interval)
        logger.info(f"Polling detenido para plugin '{plugin.name}'")

    def start(self):
        """Inicia el polling de todos los plugins en hilos separados."""
        if self._running:
            return
        self._running = True
        for plugin in self.plugins.values():
            if plugin.enabled:
                t = threading.Thread(
                    target=self._poll_plugin,
                    args=(plugin,),
                    daemon=True,
                    name=f"poll-{plugin.name}",
                )
                t.start()
                self._threads.append(t)
        logger.info(f"IntegrationBus iniciado con {len(self._threads)} plugins")

    def stop(self):
        """Detiene el bus y desconecta todos los plugins."""
        self._running = False
        for plugin in self.plugins.values():
            try:
                plugin.disconnect()
            except Exception:
                pass
        logger.info("IntegrationBus detenido")

    def get_status(self) -> Dict:
        """Estado completo del bus."""
        return {
            "running":   self._running,
            "plugins":   self.list_plugins(),
            "stats":     self._stats,
            "queue_len": len(self._data_queue),
        }
