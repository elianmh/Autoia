"""
Plugin base para integracion con apps externas.

Para conectar una nueva app de analisis:

    from integrations.base_plugin import BasePlugin, DataPoint

    class MiAppPlugin(BasePlugin):
        name = "mi_app"
        version = "1.0"
        supported_domains = ["sports", "market"]

        def fetch(self) -> List[DataPoint]:
            # Obtener datos de la app externa
            ...

        def push_prediction(self, prediction: dict) -> bool:
            # Enviar prediccion de vuelta a la app
            ...
"""

import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger("autoia.integrations.plugin")


@dataclass
class DataPoint:
    """
    Unidad de dato que un plugin entrega al motor ABA.
    Formato universal: cualquier app externa debe convertir sus datos a este formato.
    """
    source:    str           # nombre del plugin que lo genera
    domain:    str           # "sports" | "market" | "masses" | "betting"
    data_type: str           # "odds" | "price" | "sentiment" | "stats" | "event"
    payload:   Dict          # datos especificos del tipo
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0  # confianza en el dato (0-1)
    tags:      List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "source":     self.source,
            "domain":     self.domain,
            "data_type":  self.data_type,
            "payload":    self.payload,
            "timestamp":  self.timestamp,
            "confidence": self.confidence,
            "tags":       self.tags,
        }


@dataclass
class PredictionResult:
    """
    Resultado de prediccion que el motor envia de vuelta al plugin.
    """
    domain:     str
    subject:    str
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    aba_summary: Dict
    timestamp:  float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "domain":       self.domain,
            "subject":      self.subject,
            "prediction":   self.prediction,
            "confidence":   self.confidence,
            "probabilities": self.probabilities,
            "aba_summary":  self.aba_summary,
            "timestamp":    self.timestamp,
        }


class BasePlugin(ABC):
    """
    Clase base para todos los plugins de integracion.

    Ciclo de vida:
        1. __init__: configuracion inicial
        2. connect(): establece conexion con la app externa
        3. fetch(): obtiene datos en formato DataPoint
        4. push_prediction(): envia predicciones de vuelta
        5. disconnect(): cierra la conexion

    El IntegrationBus llama a fetch() periodicamente y a
    push_prediction() despues de cada prediccion.
    """

    # ── Metadata del plugin (sobrescribir en subclase) ────────────────────
    name:               str = "base"
    version:            str = "1.0"
    description:        str = ""
    supported_domains:  List[str] = []
    author:             str = ""

    def __init__(self, config: Dict = None):
        self.config    = config or {}
        self.connected = False
        self.enabled   = True
        self._callbacks: List[Callable[[DataPoint], None]] = []
        self._stats = {
            "fetches":     0,
            "data_points": 0,
            "pushes":      0,
            "errors":      0,
        }
        self.logger = logging.getLogger(f"autoia.integrations.{self.name}")

    # ── Interface obligatoria ─────────────────────────────────────────────

    @abstractmethod
    def fetch(self) -> List[DataPoint]:
        """
        Obtiene datos de la app externa.
        Se llama periodicamente por el bus.
        Retorna lista de DataPoints listos para el motor ABA.
        """
        ...

    # ── Interface opcional ────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Establece conexion con la app externa.
        Retorna True si la conexion fue exitosa.
        """
        self.connected = True
        return True

    def disconnect(self):
        """Cierra la conexion con la app externa."""
        self.connected = False

    def push_prediction(self, result: PredictionResult) -> bool:
        """
        Envia una prediccion del motor ABA de vuelta a la app externa.
        Retorna True si se envio correctamente.
        """
        return True

    def health_check(self) -> Dict:
        """
        Verifica el estado de la conexion.
        Retorna dict con status y metricas.
        """
        return {
            "plugin":    self.name,
            "connected": self.connected,
            "enabled":   self.enabled,
            "stats":     self._stats,
        }

    def on_data(self, callback: Callable[[DataPoint], None]):
        """Registra callback para cuando llegan nuevos datos."""
        self._callbacks.append(callback)

    def _emit(self, dp: DataPoint):
        """Emite un DataPoint a todos los callbacks registrados."""
        self._stats["data_points"] += 1
        for cb in self._callbacks:
            try:
                cb(dp)
            except Exception as e:
                self.logger.error(f"Callback error: {e}")

    def _record_fetch(self):
        self._stats["fetches"] += 1

    def _record_push(self):
        self._stats["pushes"] += 1

    def _record_error(self):
        self._stats["errors"] += 1

    def __repr__(self):
        return f"<Plugin:{self.name} v{self.version} connected={self.connected}>"
