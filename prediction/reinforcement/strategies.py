"""
Estrategias de prediccion que se refuerzan o castigan segun sus resultados.

Cada estrategia es una funcion que genera una prediccion para un dominio.
El sistema registra aciertos/fallos y ajusta los pesos (Matching Law).

Dominios soportados:
- sports:   prediccion de partidos deportivos
- market:   direccion de precio (up/down)
- masses:   tendencia del sentimiento masivo
- betting:  valor esperado de apuesta
"""

import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from collections import deque

logger = logging.getLogger("autoia.prediction.strategies")


@dataclass
class Prediction:
    """Una prediccion concreta de una estrategia."""
    strategy_id:  str
    domain:       str
    prediction:   str          # texto de la prediccion
    confidence:   float        # 0 a 1
    probability:  Dict[str, float] = field(default_factory=dict)  # mapa resultado->prob
    timestamp:    float = field(default_factory=time.time)
    metadata:     Dict = field(default_factory=dict)

    # Resultado real (se llena post-evento)
    actual_outcome: Optional[str] = None
    was_correct:    Optional[bool] = None
    magnitude:      float = 0.0  # fuerza del acierto (0-1) o castigo (-1 a 0)

    def evaluate(self, actual: str) -> float:
        """
        Evalua la prediccion contra el resultado real.
        Retorna magnitud del reforzamiento/castigo.
        """
        self.actual_outcome = actual
        if actual not in self.probability:
            self.was_correct = (self.prediction == actual)
            self.magnitude = 0.5 if self.was_correct else -0.3
            return self.magnitude

        # Puntuacion log-probability: premia probabilidades bien calibradas
        prob_assigned = self.probability.get(actual, 0.01)
        log_score = math.log(max(0.01, prob_assigned))  # entre -inf y 0
        # Normalizar a [-1, 0.5]
        normalized = max(-1.0, log_score / 4)  # log(0.01)/4 = -1
        if actual == self.prediction:
            self.was_correct = True
            self.magnitude = max(0.1, normalized + 1.0)  # [0.1, 1.0]
        else:
            self.was_correct = False
            self.magnitude = min(-0.1, normalized)

        return self.magnitude


@dataclass
class Strategy:
    """
    Una estrategia de prediccion con historial de rendimiento.
    El peso (matching_weight) se ajusta segun la Ley de Igualacion.
    """
    strategy_id:    str
    domain:         str
    name:           str
    description:    str
    matching_weight: float = 1.0  # peso en la distribucion de Matching Law
    prediction_history: deque = field(default_factory=lambda: deque(maxlen=100))

    @property
    def accuracy(self) -> float:
        done = [p for p in self.prediction_history if p.was_correct is not None]
        if not done:
            return 0.5
        return sum(1 for p in done if p.was_correct) / len(done)

    @property
    def avg_magnitude(self) -> float:
        done = [p for p in self.prediction_history if p.magnitude != 0.0]
        if not done:
            return 0.0
        return sum(p.magnitude for p in done) / len(done)

    @property
    def reinforcement_rate(self) -> float:
        """
        Tasa de reforzamiento reciente (ponderada temporalmente).
        Es el valor central para la Matching Law.
        """
        now = time.time()
        done = [p for p in self.prediction_history if p.was_correct is not None]
        if not done:
            return 0.5
        weighted_sum = 0.0
        weight_total = 0.0
        for pred in done:
            age_h = (now - pred.timestamp) / 3600
            weight = math.exp(-0.05 * age_h)  # decae en ~20h
            weighted_sum += pred.magnitude * weight
            weight_total += weight
        rate = weighted_sum / weight_total if weight_total > 0 else 0.0
        return max(0.01, rate)  # nunca negativo para Matching Law

    @property
    def resistance_to_extinction(self) -> float:
        """Impulso conductual de esta estrategia."""
        n = len([p for p in self.prediction_history if p.was_correct])
        if n == 0:
            return 0.0
        return math.log(1 + n) * max(0, self.avg_magnitude)

    def add_prediction(self, pred: Prediction):
        self.prediction_history.append(pred)

    def reinforce(self, magnitude: float):
        """Actualiza el peso de Matching Law tras un resultado."""
        if magnitude > 0:
            self.matching_weight = min(5.0, self.matching_weight * (1 + magnitude * 0.15))
        else:
            self.matching_weight = max(0.1, self.matching_weight * (1 + magnitude * 0.10))
        logger.debug(f"Strategy {self.name}: weight={self.matching_weight:.3f} "
                     f"after magnitude={magnitude:+.3f}")


# ─── Estrategias concretas por dominio ────────────────────────────────────────

def make_sports_strategies() -> List[Strategy]:
    """Genera las estrategias base para prediccion deportiva."""
    return [
        Strategy(
            strategy_id="sports_matching_law",
            domain="sports",
            name="Matching Law",
            description="Predice basado en tasa de reforzamiento (victorias recientes)",
            matching_weight=1.5,
        ),
        Strategy(
            strategy_id="sports_mo_dominance",
            domain="sports",
            name="MO Dominance",
            description="Predice segun Operaciones Motivadoras activas (lesiones, racha)",
            matching_weight=1.2,
        ),
        Strategy(
            strategy_id="sports_home_bias",
            domain="sports",
            name="Home Advantage",
            description="Sesgo historico hacia local (~12% ventaja en futbol)",
            matching_weight=1.0,
        ),
        Strategy(
            strategy_id="sports_momentum",
            domain="sports",
            name="Behavioral Momentum",
            description="Predice continuacion de racha actual (impulso conductual)",
            matching_weight=0.8,
        ),
        Strategy(
            strategy_id="sports_contrarian",
            domain="sports",
            name="Contrarian",
            description="Apuesta contra el consenso masivo cuando hay sobreconfianza",
            matching_weight=0.6,
        ),
    ]


def make_market_strategies() -> List[Strategy]:
    """Estrategias para mercados financieros."""
    return [
        Strategy(
            strategy_id="market_trend_following",
            domain="market",
            name="Trend Following",
            description="Sigue la tendencia: MACD + momentum. Conducta reforzada = continua",
            matching_weight=1.5,
        ),
        Strategy(
            strategy_id="market_mean_reversion",
            domain="market",
            name="Mean Reversion",
            description="Contra-tendencia: RSI extremo + sobreextension (saciacion/privacion)",
            matching_weight=1.0,
        ),
        Strategy(
            strategy_id="market_volume_breakout",
            domain="market",
            name="Volume Anomaly",
            description="Detecta activacion masiva por volumen anomalo (MO colectivo)",
            matching_weight=1.2,
        ),
        Strategy(
            strategy_id="market_behavioral_contrast",
            domain="market",
            name="Behavioral Contrast",
            description="Detecta contraste conductual: rebote en alternativas tras extincion",
            matching_weight=0.9,
        ),
    ]


def make_masses_strategies() -> List[Strategy]:
    """Estrategias para prediccion de masas."""
    return [
        Strategy(
            strategy_id="masses_sentiment_momentum",
            domain="masses",
            name="Sentiment Momentum",
            description="Sentimiento positivo creciente predice continuacion de tendencia",
            matching_weight=1.3,
        ),
        Strategy(
            strategy_id="masses_fomo_detection",
            domain="masses",
            name="FOMO Detection",
            description="FOMO = EO masivo poderoso que acelera conducta de masa",
            matching_weight=1.1,
        ),
        Strategy(
            strategy_id="masses_panic_reversal",
            domain="masses",
            name="Panic Reversal",
            description="Panico extremo (AO) suele preceder rebote (contraste conductual)",
            matching_weight=0.8,
        ),
    ]


def make_betting_strategies() -> List[Strategy]:
    """Estrategias para analisis de apuestas."""
    return [
        Strategy(
            strategy_id="betting_value_hunter",
            domain="betting",
            name="Value Hunter",
            description="Busca cuotas donde probabilidad implicita < probabilidad real",
            matching_weight=1.4,
        ),
        Strategy(
            strategy_id="betting_steam_follower",
            domain="betting",
            name="Steam Follower",
            description="Sigue movimientos bruscos de cuota (dinero inteligente = MO)",
            matching_weight=1.2,
        ),
        Strategy(
            strategy_id="betting_fade_public",
            domain="betting",
            name="Fade the Public",
            description="Apuesta contra el consenso masivo: cuota inflada por sesgo",
            matching_weight=0.9,
        ),
    ]
