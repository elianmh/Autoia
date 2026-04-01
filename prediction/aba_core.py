"""
Marco ABA (Applied Behavior Analysis) para prediccion.

Conceptos implementados:
- Contingencia de 4 terminos: MO -> SD -> R -> C
- FBA: hipotesis de funcion a partir de datos descriptivos
- FA: testeo controlado (A/B) para confirmar funcion
- Funciones: tangible, escape, atencion, automatico
- MOs: EO (establishing) y AO (abolishing)
- Reforzamiento intermitente: patrones VR, FR, VI, FI
- Impulso conductual (behavioral momentum)
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from collections import deque

logger = logging.getLogger("autoia.prediction.aba")


# ─── Enums ────────────────────────────────────────────────────────────────────

class BehaviorFunction(Enum):
    """Las 4 funciones del comportamiento."""
    TANGIBLE   = "tangible"    # Busca ganancia material (precio, puntos, dinero)
    ESCAPE     = "escape"      # Huye de algo negativo (riesgo, perdida, presion)
    ATTENTION  = "attention"   # Busca validacion social (likes, tendencias, FOMO)
    AUTOMATIC  = "automatic"   # Habito/inercia (trading algoritmico, rituales deportivos)


class MOType(Enum):
    """Tipos de Operacion Motivadora."""
    ESTABLISHING = "EO"   # Aumenta el valor del reforzador (hype, escasez, miedo)
    ABOLISHING   = "AO"   # Disminuye el valor (saciacion, malas noticias, saturacion)


class ReinforcementSchedule(Enum):
    """Programas de reforzamiento."""
    CONTINUOUS  = "CRF"  # Cada respuesta es reforzada
    FIXED_RATIO = "FR"   # Cada N respuestas
    VAR_RATIO   = "VR"   # Promedio de N respuestas (mas resistente a extincion)
    FIXED_INT   = "FI"   # Cada N segundos
    VAR_INT     = "VI"   # Promedio de N segundos


# ─── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class ABCEvent:
    """
    Evento Antecedente-Conducta-Consecuencia.
    La unidad basica de analisis en ABA.
    """
    timestamp:   float
    antecedent:  str           # Que ocurrio ANTES (noticia, dato, señal)
    behavior:    str           # Que hizo la masa/mercado (compro, vendio, apoyo)
    consequence: str           # Que obtuvo (ganancia, perdida, retiro)
    domain:      str           # "market" | "sports" | "masses" | "betting"
    magnitude:   float = 0.0   # Intensidad de la conducta (-1.0 a 1.0)
    function:    Optional[BehaviorFunction] = None
    tags:        List[str] = field(default_factory=list)

    def is_reinforced(self) -> bool:
        """El comportamiento fue seguido de consecuencia positiva."""
        return self.magnitude > 0.05

    def is_punished(self) -> bool:
        """El comportamiento fue seguido de consecuencia negativa."""
        return self.magnitude < -0.05


@dataclass
class MotivatingOperation:
    """
    Operacion Motivadora: altera el valor de reforzadores y la probabilidad de conducta.
    Es lo que hace que la masa QUIERA algo en este momento.
    """
    mo_type:     MOType
    source:      str        # "news" | "sentiment" | "injury" | "earnings" | "rumor"
    description: str
    target:      str        # A que conducta afecta
    strength:    float      # 0.0 a 1.0
    timestamp:   float = field(default_factory=time.time)
    duration_h:  float = 4.0   # Horas que dura el efecto

    @property
    def is_active(self) -> bool:
        age_h = (time.time() - self.timestamp) / 3600
        return age_h < self.duration_h

    @property
    def current_strength(self) -> float:
        """Fuerza actual con decaimiento temporal."""
        if not self.is_active:
            return 0.0
        age_h = (time.time() - self.timestamp) / 3600
        decay = 1.0 - (age_h / self.duration_h)
        return self.strength * decay

    def to_dict(self) -> dict:
        return {
            "mo_type":     self.mo_type.value,
            "source":      self.source,
            "description": self.description,
            "target":      self.target,
            "strength":    round(self.current_strength, 3),
        }


@dataclass
class FunctionalHypothesis:
    """
    Hipotesis de funcion generada por FBA.
    Dice: 'Cuando ocurre X, la masa hace Y para obtener Z (funcion)'.
    """
    domain:      str
    antecedent:  str
    behavior:    str
    function:    BehaviorFunction
    confidence:  float      # 0.0 a 1.0
    evidence:    List[str] = field(default_factory=list)
    n_confirmed: int = 0
    n_rejected:  int = 0

    def update(self, confirmed: bool):
        if confirmed:
            self.n_confirmed += 1
            self.confidence = min(0.99,
                self.confidence + 0.05 * (1 - self.confidence))
        else:
            self.n_rejected += 1
            self.confidence = max(0.01,
                self.confidence - 0.08 * self.confidence)

    @property
    def reliability(self) -> float:
        total = self.n_confirmed + self.n_rejected
        if total == 0:
            return 0.5
        return self.n_confirmed / total


# ─── FBA Engine ───────────────────────────────────────────────────────────────

class FBAEngine:
    """
    Functional Behavior Assessment Engine.

    Analiza patrones ABC para generar hipotesis de funcion.
    Proceso:
    1. Registro indirecto: suma ABCs por dominio
    2. Registro descriptivo: calcula correlaciones A->C
    3. Genera hipotesis de funcion para cada patron frecuente
    """

    FUNCTION_INDICATORS = {
        BehaviorFunction.TANGIBLE: [
            "precio sube", "ganancia", "rentabilidad", "bonus",
            "recompensa", "dividendo", "gol", "victoria", "punto"
        ],
        BehaviorFunction.ESCAPE: [
            "riesgo", "perdida", "caida", "lesion", "derrota",
            "multa", "recesion", "panico", "venta masiva", "crash"
        ],
        BehaviorFunction.ATTENTION: [
            "viral", "trending", "fomo", "social", "influencer",
            "multitud", "seguidor", "apoyo", "comunidad", "rumor"
        ],
        BehaviorFunction.AUTOMATIC: [
            "habito", "patron", "historico", "siempre", "rutina",
            "algoritmo", "cierre", "apertura", "estacional"
        ],
    }

    def __init__(self):
        self.abc_log:     List[ABCEvent] = []
        self.hypotheses:  Dict[str, FunctionalHypothesis] = {}
        self.max_log      = 10000

    def log_event(self, event: ABCEvent):
        """Registra un evento ABC y actualiza hipotesis."""
        self.abc_log.append(event)
        if len(self.abc_log) > self.max_log:
            self.abc_log.pop(0)
        self._infer_function(event)

    def _infer_function(self, event: ABCEvent):
        """Infiere la funcion del comportamiento a partir del contexto."""
        text = f"{event.antecedent} {event.behavior} {event.consequence}".lower()
        scores = {}
        for func, keywords in self.FUNCTION_INDICATORS.items():
            score = sum(1 for kw in keywords if kw in text)
            if score > 0:
                scores[func] = score

        if not scores:
            return

        best_func = max(scores, key=scores.get)
        event.function = best_func

        # Actualizar o crear hipotesis
        key = f"{event.domain}:{event.antecedent[:30]}"
        if key in self.hypotheses:
            self.hypotheses[key].update(event.is_reinforced())
        else:
            self.hypotheses[key] = FunctionalHypothesis(
                domain=event.domain,
                antecedent=event.antecedent[:60],
                behavior=event.behavior[:60],
                function=best_func,
                confidence=0.5,
                evidence=[event.consequence[:40]],
            )

    def get_top_hypotheses(self, domain: str = None,
                           min_confidence: float = 0.6,
                           n: int = 10) -> List[FunctionalHypothesis]:
        """Retorna las hipotesis mas confiables."""
        hyps = list(self.hypotheses.values())
        if domain:
            hyps = [h for h in hyps if h.domain == domain]
        hyps = [h for h in hyps if h.confidence >= min_confidence]
        return sorted(hyps, key=lambda h: -h.confidence)[:n]

    def run_functional_analysis(self, antecedent_type: str,
                                 domain: str) -> Dict[str, float]:
        """
        FA simplificado: agrupa ABCs por tipo de antecedente
        y calcula la probabilidad de cada funcion.
        Equivale al 'testeo controlado' del analisis funcional.
        """
        relevant = [e for e in self.abc_log
                    if domain in e.domain
                    and antecedent_type.lower() in e.antecedent.lower()
                    and e.function is not None]

        if not relevant:
            return {f.value: 0.25 for f in BehaviorFunction}

        counts = {f: 0 for f in BehaviorFunction}
        for ev in relevant:
            counts[ev.function] += 1

        total = sum(counts.values()) or 1
        return {f.value: counts[f] / total for f in BehaviorFunction}

    def get_stats(self) -> Dict:
        return {
            "total_abc":     len(self.abc_log),
            "hypotheses":    len(self.hypotheses),
            "high_conf":     sum(1 for h in self.hypotheses.values()
                                  if h.confidence > 0.7),
            "by_function":   {
                f.value: sum(1 for h in self.hypotheses.values()
                             if h.function == f)
                for f in BehaviorFunction
            }
        }
