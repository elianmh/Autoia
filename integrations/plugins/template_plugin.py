"""
PLANTILLA para conectar una nueva app externa.

Copia este archivo, renombra la clase y completa los metodos marcados con TODO.
Luego registra el plugin en el bus:

    from integrations.plugins.mi_app_plugin import MiAppPlugin
    bus.register(MiAppPlugin(config={"api_key": "...", "url": "..."}))

El bus llamara a fetch() cada `poll_interval` segundos.
"""

import time
import logging
from typing import List, Dict, Optional

from ..base_plugin import BasePlugin, DataPoint, PredictionResult

logger = logging.getLogger("autoia.integrations.template")


class TemplatePlugin(BasePlugin):
    """
    Plugin plantilla — renombrar y adaptar para la app especifica.
    """

    # ── Metadata — CAMBIAR ESTOS VALORES ─────────────────────────────────
    name               = "template_app"
    version            = "1.0"
    description        = "Plugin plantilla para app externa"
    supported_domains  = ["sports", "market", "masses", "betting"]
    author             = ""

    def __init__(self, config: Dict = None):
        super().__init__(config)
        # TODO: extraer credenciales/config
        self._api_key  = self.config.get("api_key", "")
        self._base_url = self.config.get("url", "")
        self._session  = None

    # ── Conexion ──────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        TODO: establecer conexion con la app externa.
        Ejemplos:
        - Autenticar con API key
        - Abrir conexion WebSocket
        - Inicializar SDK de la app
        """
        try:
            # self._session = requests.Session()
            # self._session.headers["Authorization"] = f"Bearer {self._api_key}"
            self.connected = True
            logger.info(f"[{self.name}] Conectado OK")
            return True
        except Exception as e:
            logger.error(f"[{self.name}] Error al conectar: {e}")
            return False

    def disconnect(self):
        """TODO: cerrar conexion."""
        self.connected = False

    # ── Fetch de datos ────────────────────────────────────────────────────

    def fetch(self) -> List[DataPoint]:
        """
        TODO: obtener datos de la app externa y convertirlos a DataPoints.

        Tipos de DataPoint disponibles:
            "sentiment"   — texto para analisis de sentimiento ABA
            "odds"        — cuotas de apuesta
            "team_stats"  — estadisticas de equipo deportivo
            "mo_signal"   — señal de MO directa (EO o AO)
            "price"       — precio de instrumento financiero
            "raw"         — texto crudo para analisis general

        Ejemplo de cada tipo:
        """
        self._record_fetch()
        data_points = []

        # ── Ejemplo: sentimiento ──────────────────────────────────────
        # data_points.append(DataPoint(
        #     source="template_app",
        #     domain="masses",
        #     data_type="sentiment",
        #     payload={"text": "Texto obtenido de la app externa"},
        #     confidence=0.8,
        # ))

        # ── Ejemplo: cuotas ───────────────────────────────────────────
        # data_points.append(DataPoint(
        #     source="template_app",
        #     domain="betting",
        #     data_type="odds",
        #     payload={
        #         "home_team":  "Equipo A",
        #         "away_team":  "Equipo B",
        #         "home_odds":  1.90,
        #         "draw_odds":  3.40,
        #         "away_odds":  4.00,
        #     },
        # ))

        # ── Ejemplo: estadisticas de equipo ───────────────────────────
        # data_points.append(DataPoint(
        #     source="template_app",
        #     domain="sports",
        #     data_type="team_stats",
        #     payload={
        #         "team":        "Equipo A",
        #         "league":      "La Liga",
        #         "result":      "W",
        #         "goals_for":   2,
        #         "goals_against": 0,
        #         "injuries":    ["Jugador X"],
        #     },
        # ))

        # ── Ejemplo: MO directo ───────────────────────────────────────
        # data_points.append(DataPoint(
        #     source="template_app",
        #     domain="sports",
        #     data_type="mo_signal",
        #     payload={
        #         "mo_type":     "AO",
        #         "description": "Jugador estrella lesionado",
        #         "target":      "Equipo A",
        #         "strength":    0.8,
        #         "duration_h":  48,
        #     },
        # ))

        return data_points

    # ── Push de predicciones ──────────────────────────────────────────────

    def push_prediction(self, result: PredictionResult) -> bool:
        """
        TODO: enviar la prediccion de Autoia a la app externa.
        Se llama automaticamente cuando el motor genera una prediccion
        en un dominio que este plugin soporta.
        """
        self._record_push()
        logger.debug(f"[{self.name}] Prediccion: {result.domain} -> {result.prediction} "
                     f"({result.confidence:.0%})")

        # TODO: enviar a la app externa
        # Ejemplo:
        # response = self._session.post(
        #     f"{self._base_url}/predictions",
        #     json=result.to_dict(),
        # )
        # return response.ok

        return True

    # ── Health check ──────────────────────────────────────────────────────

    def health_check(self) -> Dict:
        base = super().health_check()
        base.update({
            "api_key_set": bool(self._api_key),
            "base_url":    self._base_url,
        })
        return base
