"""
Ley de Igualacion (Matching Law) + Impulso Conductual (Behavioral Momentum).

La masa distribuye su conducta (dinero, atencion, apuestas) en proporcion
a la tasa de reforzamiento de cada opcion.

Formula: B1/(B1+B2) = R1/(R1+R2)
donde B = conducta, R = tasa de reforzamiento

Aplicaciones:
- Mercados:  la inversion fluye hacia donde mas retorno ha dado historicamente
- Apuestas:  el dinero va al equipo con mayor tasa de victorias recientes
- Deportes:  las jugadas se repiten en proporcion a los exitos pasados
- Masas:     la atencion se distribuye segun reforzamiento social reciente
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("autoia.prediction.matching")


@dataclass
class BehaviorOption:
    """
    Una opcion de conducta con su historial de reforzamiento.
    Ej: 'comprar AAPL', 'apostar equipo A', 'seguir tendencia X'
    """
    name:        str
    domain:      str
    # Historial de resultados: (timestamp, magnitude)
    reinforcement_history: deque = field(
        default_factory=lambda: deque(maxlen=200)
    )
    punishments:   int   = 0
    reinforcements: int  = 0
    total_magnitude: float = 0.0

    def add_outcome(self, magnitude: float):
        """Registra el resultado de una ocurrencia de esta conducta."""
        self.reinforcement_history.append((time.time(), magnitude))
        self.total_magnitude += magnitude
        if magnitude > 0:
            self.reinforcements += 1
        elif magnitude < 0:
            self.punishments += 1

    @property
    def reinforcement_rate(self) -> float:
        """
        Tasa de reforzamiento reciente (ultimas N ocurrencias).
        Usa promedio ponderado: mas peso a resultados recientes.
        """
        if not self.reinforcement_history:
            return 0.0
        now = time.time()
        weighted_sum = 0.0
        weight_total = 0.0
        for ts, mag in self.reinforcement_history:
            # Peso exponencial: mas reciente = mas peso
            age_hours = (now - ts) / 3600
            weight = math.exp(-0.1 * age_hours)  # decae en ~10h
            weighted_sum += mag * weight
            weight_total += weight
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    @property
    def resistance_to_extinction(self) -> float:
        """
        Impulso conductual: cuanto tarda la masa en abandonar esta opcion.
        Mayor historia de reforzamiento = mayor resistencia.
        Formula: log(1 + total_reinforcements) * avg_magnitude
        """
        if self.reinforcements == 0:
            return 0.0
        avg_mag = (self.total_magnitude / max(1, self.reinforcements + self.punishments))
        return math.log(1 + self.reinforcements) * max(0, avg_mag)

    @property
    def extinction_risk(self) -> float:
        """
        Probabilidad de que la masa abandone esta opcion pronto.
        Alta si hay muchos castigos recientes y baja historia.
        """
        recent = list(self.reinforcement_history)[-20:]
        if not recent:
            return 0.5
        recent_neg = sum(1 for _, m in recent if m < -0.02)
        recent_pos = sum(1 for _, m in recent if m > 0.02)
        ratio = recent_neg / max(1, recent_neg + recent_pos)
        # Moderar por impulso conductual
        return ratio * (1.0 - min(0.8, self.resistance_to_extinction * 0.1))


class MatchingLawEngine:
    """
    Calcula la distribucion de conducta segun la Ley de Igualacion.

    Predice:
    1. Donde fluira el dinero/atencion en el proximo periodo
    2. Cuando ocurrira un cambio (switching) de opcion
    3. Que tan fuerte es el impulso de la tendencia actual
    4. Contraste conductual: rebote tras extincion de una opcion
    """

    def __init__(self):
        self.options: Dict[str, BehaviorOption] = {}
        self.switching_history: List[Dict] = []

    def register_option(self, name: str, domain: str) -> BehaviorOption:
        key = f"{domain}:{name}"
        if key not in self.options:
            self.options[key] = BehaviorOption(name=name, domain=domain)
        return self.options[key]

    def record_outcome(self, name: str, domain: str, magnitude: float):
        """Registra el resultado de una conducta (+ = reforzado, - = castigado)."""
        opt = self.register_option(name, domain)
        opt.add_outcome(magnitude)

    def get_matching_distribution(self, domain: str) -> Dict[str, float]:
        """
        Calcula la distribucion de conducta segun la Ley de Igualacion.
        Retorna: {opcion: proporcion_esperada_de_conducta}
        """
        domain_opts = {k: v for k, v in self.options.items()
                       if v.domain == domain and v.reinforcement_rate > 0}

        if not domain_opts:
            return {}

        rates = {k: max(0, v.reinforcement_rate)
                 for k, v in domain_opts.items()}
        total_rate = sum(rates.values())

        if total_rate == 0:
            n = len(rates)
            return {k: 1/n for k in rates}

        return {k: r / total_rate for k, r in rates.items()}

    def detect_behavioral_contrast(self, domain: str) -> List[Dict]:
        """
        Detecta contraste conductual: cuando una opcion entra en extincion,
        las alternativas reciben un rebote de conducta extra.
        Esto causa movimientos de precio inesperadamente fuertes.
        """
        contrasts = []
        domain_opts = {k: v for k, v in self.options.items()
                       if v.domain == domain}

        for key, opt in domain_opts.items():
            if opt.extinction_risk > 0.65:
                # Esta opcion esta en extincion -> las alternativas rebotaran
                alternatives = {k: v for k, v in domain_opts.items()
                                if k != key and v.resistance_to_extinction > 0.5}
                for alt_key, alt_opt in alternatives.items():
                    contrasts.append({
                        "extinguishing": opt.name,
                        "beneficiary":   alt_opt.name,
                        "contrast_strength": opt.extinction_risk * alt_opt.resistance_to_extinction,
                        "expected_increase_pct": opt.extinction_risk * 15,  # % estimado
                    })

        return sorted(contrasts, key=lambda x: -x["contrast_strength"])[:5]

    def calculate_momentum(self, name: str, domain: str) -> float:
        """
        Calcula el impulso conductual de una opcion.
        Resultado: 0.0 (sin impulso) a 10.0 (impulso muy alto)
        Traduce: cuantos 'disruptores' (malas noticias) son necesarios
        para cambiar la conducta de la masa.
        """
        key = f"{domain}:{name}"
        if key not in self.options:
            return 0.0
        opt = self.options[key]
        return min(10.0, opt.resistance_to_extinction)

    def predict_switching(self, domain: str) -> Optional[Dict]:
        """
        Predice si la masa esta a punto de cambiar de opcion (switching).
        Ocurre cuando: alto castigo reciente + existe alternativa con buen impulso.
        """
        domain_opts = {k: v for k, v in self.options.items()
                       if v.domain == domain}
        if len(domain_opts) < 2:
            return None

        # Opcion en riesgo de extincion
        at_risk = [(k, v) for k, v in domain_opts.items()
                   if v.extinction_risk > 0.6]
        if not at_risk:
            return None

        riskiest_key, riskiest = max(at_risk, key=lambda x: x[1].extinction_risk)

        # Alternativa con mayor impulso
        alternatives = [(k, v) for k, v in domain_opts.items()
                        if k != riskiest_key
                        and v.resistance_to_extinction > 0.2]
        if not alternatives:
            return None

        best_alt_key, best_alt = max(alternatives,
                                     key=lambda x: x[1].resistance_to_extinction)

        prob = riskiest.extinction_risk * (
            best_alt.resistance_to_extinction / max(1,
            riskiest.resistance_to_extinction + best_alt.resistance_to_extinction)
        )

        return {
            "from_option":      riskiest.name,
            "to_option":        best_alt.name,
            "switch_probability": round(prob, 3),
            "momentum_from":    round(riskiest.resistance_to_extinction, 2),
            "momentum_to":      round(best_alt.resistance_to_extinction, 2),
            "trigger":          "extinction" if riskiest.extinction_risk > 0.75 else "contrast",
        }

    def get_domain_summary(self, domain: str) -> Dict:
        """Resumen del estado conductual de un dominio."""
        domain_opts = {k: v for k, v in self.options.items()
                       if v.domain == domain}
        if not domain_opts:
            return {}

        dist = self.get_matching_distribution(domain)
        switch = self.predict_switching(domain)
        contrasts = self.detect_behavioral_contrast(domain)

        leader = max(dist, key=dist.get) if dist else None
        leader_name = self.options[leader].name if leader else None

        return {
            "leader":              leader_name,
            "distribution":        {self.options[k].name: round(v, 3)
                                    for k, v in dist.items()},
            "predicted_switch":    switch,
            "behavioral_contrasts": contrasts[:2],
            "total_options":       len(domain_opts),
            "avg_momentum":        round(
                sum(v.resistance_to_extinction for v in domain_opts.values())
                / max(1, len(domain_opts)), 2
            ),
        }
