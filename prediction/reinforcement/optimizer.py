"""
Optimizador de estrategias de prediccion usando Matching Law.

El optimizador distribuye las predicciones entre estrategias disponibles
en proporcion a su tasa de reforzamiento (Matching Law).
Aprende que estrategias funcionan mejor para cada dominio y condicion.

Principios ABA:
- Las estrategias con mayor tasa de reforzamiento reciben mas "conducta"
  (mas peso en el ensamble de predicciones)
- Estrategias sin historial parten de un peso uniforme (exploracion)
- Cuando una estrategia entra en extincion, las alternativas reciben impulso
  extra (contraste conductual)
"""

import time
import math
import logging
from typing import Dict, List, Optional, Tuple
from collections import deque

from .strategies import Strategy, Prediction, make_sports_strategies, \
    make_market_strategies, make_masses_strategies, make_betting_strategies

logger = logging.getLogger("autoia.prediction.optimizer")


class StrategyOptimizer:
    """
    Gestor de estrategias con aprendizaje por reforzamiento.

    Implementa:
    1. Matching Law: distribucion proporcional a tasa de reforzamiento
    2. Behavioral Momentum: estrategias con impulso son difíciles de abandonar
    3. Behavioral Contrast: si una estrategia falla, otras reciben boost
    4. Extinction detection: desactiva estrategias no rentables
    """

    def __init__(self, domain: str = "general"):
        self.domain     = domain
        self.strategies: Dict[str, Strategy] = {}
        self._load_default_strategies(domain)
        self.outcome_log: deque = deque(maxlen=500)

    def _load_default_strategies(self, domain: str):
        """Carga estrategias por defecto segun el dominio."""
        from .strategies import (make_sports_strategies, make_market_strategies,
                                  make_masses_strategies, make_betting_strategies)
        loaders = {
            "sports":  make_sports_strategies,
            "market":  make_market_strategies,
            "masses":  make_masses_strategies,
            "betting": make_betting_strategies,
        }
        fn = loaders.get(domain)
        if fn:
            for s in fn():
                self.strategies[s.strategy_id] = s

    def add_strategy(self, strategy: Strategy):
        self.strategies[strategy.strategy_id] = strategy

    # ── Matching Law distribution ──────────────────────────────────────────

    def get_matching_weights(self) -> Dict[str, float]:
        """
        Distribucion de Matching Law sobre las estrategias.
        Retorna: {strategy_id: proporcion}
        """
        if not self.strategies:
            return {}

        rates = {sid: max(0.01, s.reinforcement_rate)
                 for sid, s in self.strategies.items()}
        total = sum(rates.values())
        return {sid: r / total for sid, r in rates.items()}

    def get_ensemble_prediction(self, predictions: Dict[str, Prediction]) -> Dict:
        """
        Combina predicciones de multiples estrategias usando Matching Law.
        predictions: {strategy_id: Prediction}
        Retorna: prediccion ensamblada con confianza ponderada.
        """
        weights = self.get_matching_weights()

        # Acumular probabilidades ponderadas
        prob_accumulator: Dict[str, float] = {}
        total_weight = 0.0

        for sid, pred in predictions.items():
            w = weights.get(sid, 0.5)
            total_weight += w
            for outcome, prob in pred.probability.items():
                if outcome not in prob_accumulator:
                    prob_accumulator[outcome] = 0.0
                prob_accumulator[outcome] += prob * w

        if not prob_accumulator or total_weight == 0:
            return {}

        # Normalizar
        ensemble_probs = {k: v / total_weight for k, v in prob_accumulator.items()}

        # Prediccion final
        best_outcome = max(ensemble_probs, key=ensemble_probs.get)
        confidence = ensemble_probs[best_outcome]

        # Calcular acuerdo entre estrategias (consenso)
        n_agree = sum(1 for p in predictions.values() if p.prediction == best_outcome)
        consensus = n_agree / max(1, len(predictions))

        return {
            "prediction":  best_outcome,
            "confidence":  round(confidence, 3),
            "consensus":   round(consensus, 3),
            "probabilities": {k: round(v, 3) for k, v in ensemble_probs.items()},
            "top_strategy": self._get_top_strategy(),
            "n_strategies": len(predictions),
        }

    def _get_top_strategy(self) -> Optional[str]:
        """Estrategia con mayor peso actual."""
        if not self.strategies:
            return None
        return max(self.strategies, key=lambda sid: self.strategies[sid].matching_weight)

    # ── Outcome recording and learning ────────────────────────────────────

    def record_outcome(self, strategy_id: str, prediction: Prediction,
                       actual_outcome: str):
        """
        Registra el resultado real y actualiza el peso de la estrategia.
        Esta es la funcion de aprendizaje central.
        """
        if strategy_id not in self.strategies:
            return

        strategy = self.strategies[strategy_id]
        magnitude = prediction.evaluate(actual_outcome)
        strategy.reinforce(magnitude)
        strategy.add_prediction(prediction)

        self.outcome_log.append({
            "timestamp":   time.time(),
            "strategy_id": strategy_id,
            "prediction":  prediction.prediction,
            "actual":      actual_outcome,
            "correct":     prediction.was_correct,
            "magnitude":   magnitude,
        })

        logger.info(f"[{self.domain}] {strategy.name}: "
                    f"{'OK' if prediction.was_correct else 'FAIL'} "
                    f"mag={magnitude:+.3f} weight={strategy.matching_weight:.3f}")

        # Contraste conductual: si falla, boost a alternativas
        if magnitude < -0.2:
            self._apply_behavioral_contrast(strategy_id, abs(magnitude))

    def record_ensemble_outcome(self, predictions: Dict[str, Prediction],
                                 actual_outcome: str):
        """Registra resultado para todas las estrategias del ensemble."""
        for sid, pred in predictions.items():
            self.record_outcome(sid, pred, actual_outcome)

    def _apply_behavioral_contrast(self, failing_sid: str, contrast_strength: float):
        """
        Contraste conductual: estrategias alternativas reciben un boost
        cuando la estrategia dominante falla (extinction effect).
        """
        for sid, strategy in self.strategies.items():
            if sid == failing_sid:
                continue
            # Boost proporcional al impulso de la estrategia alternativa
            boost = contrast_strength * 0.05 * min(1.0, strategy.resistance_to_extinction)
            strategy.matching_weight = min(5.0, strategy.matching_weight + boost)
            if boost > 0.01:
                logger.debug(f"Contrast boost {strategy.name}: +{boost:.4f}")

    # ── Analysis and reporting ─────────────────────────────────────────────

    def get_performance_summary(self) -> Dict:
        """Resumen de rendimiento de todas las estrategias."""
        summary = {}
        for sid, s in self.strategies.items():
            summary[s.name] = {
                "accuracy":    round(s.accuracy, 3),
                "weight":      round(s.matching_weight, 3),
                "rr":          round(s.reinforcement_rate, 3),
                "momentum":    round(s.resistance_to_extinction, 3),
                "n_total":     len(s.prediction_history),
                "n_correct":   sum(1 for p in s.prediction_history if p.was_correct),
            }
        return summary

    def get_matching_distribution_summary(self) -> Dict:
        """
        Distribucion actual con interpretacion ABA.
        """
        dist = self.get_matching_weights()
        leader_sid = max(dist, key=dist.get) if dist else None
        leader = self.strategies.get(leader_sid)

        # Detectar switching inminente
        sorted_by_weight = sorted(self.strategies.values(),
                                   key=lambda s: s.matching_weight, reverse=True)
        switch_risk = 0.0
        if len(sorted_by_weight) >= 2:
            top, second = sorted_by_weight[0], sorted_by_weight[1]
            # Si la segunda estrategia esta creciendo y la primera bajando
            if (second.reinforcement_rate > top.reinforcement_rate * 0.9
                    and top.matching_weight > second.matching_weight):
                switch_risk = second.reinforcement_rate / (
                    top.reinforcement_rate + second.reinforcement_rate)

        return {
            "distribution": {self.strategies[k].name: round(v, 3)
                             for k, v in dist.items()},
            "leader":        leader.name if leader else None,
            "switch_risk":   round(switch_risk, 3),
            "domain":        self.domain,
        }

    def reset_weights(self):
        """Reinicia todos los pesos (para experimentos)."""
        for s in self.strategies.values():
            s.matching_weight = 1.0
        logger.info(f"Weights reset for domain {self.domain}")
