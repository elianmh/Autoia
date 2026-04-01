"""
Motor de Prediccion ABA Unificado.

Integra todos los colectores y el marco ABA en un sistema coherente.
Para cada dominio (deportes, mercados, masas, apuestas) genera predicciones
combinando:

1. FBA (Functional Behavior Assessment) — funcion de la conducta masiva
2. Matching Law — distribucion de conducta proporcional a reforzamiento
3. Operaciones Motivadoras — ajuste de valor del reforzador
4. Impulso Conductual — resistencia a cambios de tendencia
5. Contraste Conductual — rebotes tras extincion de una opcion
6. Analisis de ensamble — combinacion ponderada de estrategias

Dominio-agnostico: el mismo motor sirve para futbol, acciones, criptos,
tendencias de masas o mercados de apuestas.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import deque

from .aba_core import FBAEngine, ABCEvent, MotivatingOperation, BehaviorFunction, MOType
from .matching_law import MatchingLawEngine, BehaviorOption
from .collectors.sentiment import SentimentCollector, SentimentSignal
from .collectors.market import MarketCollector, MarketSnapshot
from .collectors.sports import SportsCollector, MatchPrediction, TeamStats
from .collectors.betting import BettingCollector, OddsSnapshot
from .reinforcement.optimizer import StrategyOptimizer
from .reinforcement.strategies import Prediction

logger = logging.getLogger("autoia.prediction.engine")


@dataclass
class UnifiedPrediction:
    """
    Prediccion unificada que integra todos los dominios ABA.
    Esta es la salida final del motor.
    """
    domain:          str
    subject:         str        # "BTC-USD", "Real Madrid vs Barcelona", etc.
    timestamp:       float = field(default_factory=time.time)

    # Prediccion central
    predicted_outcome: str  = "incierto"
    confidence:        float = 0.5
    probabilities:     Dict[str, float] = field(default_factory=dict)

    # ABA Analysis
    dominant_function:  str = ""     # TANGIBLE/ESCAPE/ATTENTION/AUTOMATIC
    active_mos:         List[Dict] = field(default_factory=list)  # MOs activos
    behavioral_edge:    str = ""     # quien tiene la ventaja conductual
    matching_distribution: Dict[str, float] = field(default_factory=dict)
    momentum_score:     float = 0.0
    contrast_risk:      List[Dict] = field(default_factory=list)  # riesgos de contraste

    # Fuentes de datos
    sentiment_score:    float = 0.0   # -1 a 1
    market_signal:      str = ""
    sports_prediction:  Optional[Dict] = None
    betting_analysis:   Optional[Dict] = None

    # Estrategias
    strategy_ensemble:  Dict = field(default_factory=dict)
    key_factors:        List[str] = field(default_factory=list)
    risk_factors:       List[str] = field(default_factory=list)

    # Post-resultado
    actual_outcome:     Optional[str] = None
    was_correct:        Optional[bool] = None

    def to_dict(self) -> Dict:
        return {
            "domain":     self.domain,
            "subject":    self.subject,
            "prediction": self.predicted_outcome,
            "confidence": self.confidence,
            "probs":      self.probabilities,
            "aba": {
                "function":    self.dominant_function,
                "mos":         self.active_mos,
                "edge":        self.behavioral_edge,
                "momentum":    self.momentum_score,
                "distribution": self.matching_distribution,
            },
            "signals": {
                "sentiment": self.sentiment_score,
                "market":    self.market_signal,
            },
            "factors":    self.key_factors,
            "risks":      self.risk_factors,
            "timestamp":  self.timestamp,
        }


class PredictionEngine:
    """
    Motor central de prediccion ABA.

    Conecta todos los subsistemas:
    - FBA para identificar la funcion de la conducta masiva
    - Matching Law para distribuir probabilidades
    - Colectores para datos de cada dominio
    - Optimizador para aprender que estrategias funcionan
    """

    DOMAINS = ["sports", "market", "masses", "betting"]

    def __init__(self, orchestrator=None,
                 market_symbols: Optional[List[str]] = None,
                 sports_sport: str = "football",
                 betting_api_key: Optional[str] = None):
        """
        Args:
            orchestrator:    Ollama orchestrator (opcional) para sentimiento via LLM
            market_symbols:  Simbolos a seguir (ej: ["BTC-USD", "AAPL", "^GSPC"])
            sports_sport:    Deporte principal
            betting_api_key: API key de The Odds API (opcional)
        """
        self.orchestrator = orchestrator

        # ── Subsistemas ABA ──────────────────────────────────────────────
        self.fba = FBAEngine()
        self.matching_law = MatchingLawEngine()
        self.active_mos: List[MotivatingOperation] = []

        # ── Colectores ───────────────────────────────────────────────────
        self.sentiment = SentimentCollector(
            orchestrator=orchestrator, domain="masses"
        )
        self.market = MarketCollector(symbols=market_symbols or [])
        self.sports = SportsCollector(sport=sports_sport)
        self.betting = BettingCollector(api_key=betting_api_key, sport=sports_sport)

        # ── Optimizadores por dominio ────────────────────────────────────
        self.optimizers: Dict[str, StrategyOptimizer] = {
            d: StrategyOptimizer(domain=d) for d in self.DOMAINS
        }

        # ── Historial ────────────────────────────────────────────────────
        self.predictions: deque = deque(maxlen=1000)
        self._lock = threading.Lock()

        logger.info("PredictionEngine ABA inicializado")

    # ── MO Management ─────────────────────────────────────────────────────

    def add_mo(self, domain: str, source: str, description: str,
               mo_type: str = "EO", target: str = "general",
               strength: float = 0.5, duration_h: float = 4.0):
        """
        Registra una Operacion Motivadora activa.
        EO: aumenta el valor del reforzador (mercado sube, equipo gana, etc.)
        AO: disminuye el valor del reforzador (crisis, lesion, malas noticias)
        """
        mo = MotivatingOperation(
            mo_type=MOType.ESTABLISHING if mo_type == "EO" else MOType.ABOLISHING,
            source=source,
            description=description,
            target=target,
            strength=strength,
            timestamp=time.time(),
            duration_h=duration_h,
        )
        self.active_mos.append(mo)
        logger.info(f"MO [{mo_type}] en {domain}: {description[:60]}")

    def analyze_text_mo(self, text: str, domain: str = "general",
                         callback: Optional[Callable] = None):
        """
        Analiza texto para detectar MOs automaticamente.
        Registra tanto la señal de sentimiento como el evento ABC.
        """
        def _on_signal(signal: SentimentSignal):
            if signal:
                # Registrar como ABCEvent
                event = ABCEvent(
                    timestamp=time.time(),
                    antecedent="texto_externo",
                    behavior="apostar/invertir/seguir_tendencia",
                    consequence="ganancia_esperada" if signal.score > 0 else "perdida_esperada",
                    domain=domain,
                    magnitude=signal.score,
                    function=BehaviorFunction(signal.function)
                    if signal.function in BehaviorFunction._value2member_map_
                    else BehaviorFunction.AUTOMATIC,
                )
                self.fba.log_event(event)
                if callback:
                    callback(signal)

        self.sentiment.analyze_text(text, source=domain, callback=_on_signal)

    # ── Core Prediction Methods ────────────────────────────────────────────

    def predict_sports_match(self, home: str, away: str,
                              extra_mos: Optional[List[Dict]] = None) -> UnifiedPrediction:
        """
        Prediccion completa de partido deportivo con analisis ABA.

        Args:
            home: equipo local
            away: equipo visitante
            extra_mos: MOs adicionales [{"source": ..., "type": "EO"|"AO", "description": ...}]
        """
        # Aplicar MOs adicionales si se proveen
        if extra_mos:
            for mo_data in extra_mos:
                home_team = self.sports.teams.get(home)
                away_team = self.sports.teams.get(away)
                if mo_data.get("team") == home and home_team:
                    if mo_data["type"] == "EO":
                        home_team.key_returns.append(mo_data["description"])
                    else:
                        home_team.injuries.append(mo_data["description"])
                elif mo_data.get("team") == away and away_team:
                    if mo_data["type"] == "EO":
                        away_team.key_returns.append(mo_data["description"])
                    else:
                        away_team.injuries.append(mo_data["description"])

        # Prediccion base del colector
        match_pred = self.sports.predict_match(home, away)

        # Registrar en Matching Law
        self.matching_law.record_outcome(home, "sports", match_pred.home_win_prob)
        self.matching_law.record_outcome(away, "sports", match_pred.away_win_prob)

        # Cuotas si disponibles
        event_id = f"{home}_vs_{away}".lower().replace(" ", "_")
        betting_analysis = self.betting.analyze_market_sentiment(event_id)

        # Sentimiento agregado
        sentiment = self.sentiment.get_aggregate_sentiment(24.0)

        # Hipotesis FBA del dominio deportivo
        top_hypo = self.fba.get_top_hypotheses("sports", min_confidence=0.4, n=1)
        dominant_func = top_hypo[0].function.value if top_hypo else "tangible"

        # Ajustar confianza por consenso de cuotas
        confidence = match_pred.confidence
        if betting_analysis:
            odds_prob = betting_analysis.get("implied_probs", {})
            predicted_side = ("home" if match_pred.predicted_winner == home else
                              "draw" if match_pred.predicted_winner == "empate" else "away")
            if predicted_side in odds_prob:
                market_prob = odds_prob[predicted_side]
                # Promedio ponderado: modelo 60%, mercado 40%
                confidence = round(confidence * 0.6 + market_prob * 0.4, 3)

        # Factores clave
        key_factors = list(match_pred.key_factors)
        risk_factors = list(match_pred.risk_factors)
        if sentiment.get("mo_type") == "EO":
            key_factors.append(f"Sentimiento masivo positivo ({sentiment['score']:+.2f})")
        if betting_analysis and betting_analysis.get("steam_move"):
            sm = betting_analysis["steam_move"]
            key_factors.append(f"Steam move hacia {sm['direction']} ({sm['magnitude']}%)")

        # MOs activos
        active_mos = [mo.to_dict() for mo in self.active_mos
                      if mo.current_strength > 0.1]

        pred = UnifiedPrediction(
            domain="sports",
            subject=f"{home} vs {away}",
            predicted_outcome=match_pred.predicted_winner,
            confidence=confidence,
            probabilities={
                home:    match_pred.home_win_prob,
                "draw":  match_pred.draw_prob,
                away:    match_pred.away_win_prob,
            },
            dominant_function=dominant_func,
            active_mos=active_mos[:3],
            behavioral_edge=match_pred.behavioral_edge,
            matching_distribution=match_pred.matching_law_dist,
            momentum_score=round(
                (self.sports.teams.get(home, TeamStats(home,"","")).behavioral_momentum +
                 self.sports.teams.get(away, TeamStats(away,"","")).behavioral_momentum) / 2, 2
            ),
            sentiment_score=sentiment.get("score", 0.0),
            sports_prediction=match_pred.__dict__,
            betting_analysis=betting_analysis or {},
            key_factors=key_factors,
            risk_factors=risk_factors,
        )
        with self._lock:
            self.predictions.append(pred)
        return pred

    def predict_market(self, symbol: str) -> UnifiedPrediction:
        """
        Prediccion de mercado financiero con analisis ABA.
        Combina: RSI, MACD, volumen anómalo, sentimiento, Matching Law.
        """
        snap = self.market.fetch(symbol)

        if snap is None:
            return UnifiedPrediction(
                domain="market", subject=symbol,
                predicted_outcome="sin_datos", confidence=0.0,
            )

        # Señales base
        signals = snap.behavioral_signal.split(" | ")
        up_signals = sum(1 for s in signals if s in
                         ("REFUERZO_POSITIVO", "MOMENTUM_ALCISTA", "PRIVACION"))
        down_signals = sum(1 for s in signals if s in
                           ("CASTIGO", "MOMENTUM_BAJISTA", "SACIACION"))

        # Matching Law sobre historial del simbolo
        self.matching_law.record_outcome(symbol, "market",
                                          snap.change_pct / 10)  # normalizado
        ml_dist = self.matching_law.get_matching_distribution("market")

        # Probabilidades
        if up_signals > down_signals:
            up_prob   = 0.4 + min(0.35, up_signals * 0.08)
            down_prob = max(0.1, 0.4 - up_signals * 0.06)
        elif down_signals > up_signals:
            up_prob   = max(0.1, 0.4 - down_signals * 0.06)
            down_prob = 0.4 + min(0.35, down_signals * 0.08)
        else:
            up_prob = down_prob = 0.35
        flat_prob = max(0.05, 1.0 - up_prob - down_prob)

        total = up_prob + down_prob + flat_prob
        up_prob   /= total
        down_prob /= total
        flat_prob /= total

        predicted = "up" if up_prob > down_prob else "down"
        confidence = max(up_prob, down_prob)

        # Factores
        key_factors, risk_factors = [], []
        if "VOLUMEN_ANOMALO" in signals:
            key_factors.append(f"Volumen anomalo (activacion masiva) en {symbol}")
        if "PRIVACION" in signals:
            key_factors.append(f"RSI oversold: privacion activa EO de compra")
        if "SACIACION" in signals:
            risk_factors.append(f"RSI overbought: saciacion puede reducir demanda")
        if snap.macd_signal == "bearish":
            risk_factors.append(f"MACD bajista: momentum conductual negativo")

        # MOs activos
        active_mos = [mo.to_dict() for mo in self.active_mos
                      if mo.current_strength > 0.1]

        # Sentimiento
        sentiment = self.sentiment.get_aggregate_sentiment(4.0)  # 4h para mercado

        pred = UnifiedPrediction(
            domain="market",
            subject=symbol,
            predicted_outcome=predicted,
            confidence=round(confidence, 3),
            probabilities={"up": round(up_prob,3), "flat": round(flat_prob,3),
                           "down": round(down_prob,3)},
            dominant_function="tangible",
            active_mos=active_mos[:3],
            behavioral_edge=predicted,
            matching_distribution={k.split(":")[1]: round(v, 3)
                                    for k, v in ml_dist.items() if ":" in k},
            momentum_score=round(abs(snap.change_pct) * 0.3, 2),
            market_signal=snap.behavioral_signal,
            sentiment_score=sentiment.get("score", 0.0),
            key_factors=key_factors,
            risk_factors=risk_factors,
        )
        with self._lock:
            self.predictions.append(pred)
        return pred

    def predict_mass_trend(self, topic: str,
                            texts: Optional[List[str]] = None) -> UnifiedPrediction:
        """
        Prediccion de tendencia masiva para un topico.
        Analiza textos, agrega sentimiento, detecta MOs colectivos.
        """
        # Analizar textos si se proveen
        if texts:
            for text in texts[:10]:  # limitar a 10
                self.analyze_text_mo(text, domain="masses")

        # Agregado de sentimiento
        sentiment = self.sentiment.get_aggregate_sentiment(24.0)
        score = sentiment.get("score", 0.0)
        trend = sentiment.get("trend", "estable")
        mo_type = sentiment.get("mo_type", "neutral")

        # Matching Law sobre topico
        self.matching_law.record_outcome(topic, "masses", score)
        ml_dist = self.matching_law.get_matching_distribution("masses")
        contrast = self.matching_law.detect_behavioral_contrast("masses")
        switch   = self.matching_law.predict_switching("masses")

        # Prediccion de tendencia
        if score > 0.3:
            predicted, confidence = "creciente", 0.5 + score * 0.3
        elif score < -0.3:
            predicted, confidence = "decreciente", 0.5 + abs(score) * 0.3
        else:
            predicted, confidence = "estable", 0.6

        probs = {
            "creciente":  max(0.1, 0.33 + score * 0.4),
            "estable":    max(0.1, 0.34 - abs(score) * 0.2),
            "decreciente": max(0.1, 0.33 - score * 0.4),
        }
        total = sum(probs.values())
        probs = {k: round(v/total, 3) for k, v in probs.items()}

        # Factores
        key_factors, risk_factors = [], []
        if mo_type == "EO" and trend == "acelerando":
            key_factors.append(f"EO masivo acelerando: FOMO activo en {topic}")
        if contrast:
            risk_factors.append(f"Contraste conductual: {contrast[0]['extinguishing']} "
                                 f"en extincion -> rebote en alternativas")
        if switch:
            risk_factors.append(f"Switch inminente: masa abandonando "
                                 f"{switch['from_option']} -> {switch['to_option']}")

        pred = UnifiedPrediction(
            domain="masses",
            subject=topic,
            predicted_outcome=predicted,
            confidence=round(min(0.95, confidence), 3),
            probabilities=probs,
            dominant_function=sentiment.get("function", "attention"),
            active_mos=[{"type": mo_type, "score": score, "trend": trend}],
            behavioral_edge=predicted,
            matching_distribution={k.split(":")[1]: round(v, 3)
                                    for k, v in ml_dist.items() if ":" in k},
            momentum_score=round(abs(score) * 2, 2),
            contrast_risk=contrast[:2],
            sentiment_score=score,
            key_factors=key_factors,
            risk_factors=risk_factors,
        )
        with self._lock:
            self.predictions.append(pred)
        return pred

    def predict_betting_value(self, home: str, away: str,
                               home_odds: float, draw_odds: float,
                               away_odds: float) -> UnifiedPrediction:
        """
        Analiza si hay valor en las cuotas disponibles.
        Compara probabilidades del modelo vs cuotas del mercado.
        """
        # Registrar cuotas
        event_id = f"{home}_vs_{away}".lower().replace(" ", "_")
        snap = self.betting.add_manual_odds(
            home, away, home_odds, draw_odds, away_odds
        )

        # Prediccion del modelo deportivo para el mismo partido
        sports_pred = self.predict_sports_match(home, away)
        model_probs = {
            "home": sports_pred.probabilities.get(home, 0.33),
            "draw": sports_pred.probabilities.get("draw", 0.33),
            "away": sports_pred.probabilities.get(away, 0.33),
        }

        # Probabilidades implicitas del mercado
        market_probs = snap.implied_probs

        # Calcular Edge: valor esperado de cada apuesta
        edges = {
            "home": model_probs["home"] * home_odds - 1,
            "draw": model_probs["draw"] * draw_odds - 1,
            "away": model_probs["away"] * away_odds - 1,
        }

        # Mejor apuesta (mayor EV)
        best_bet = max(edges, key=edges.get)
        best_ev  = edges[best_bet]
        has_value = best_ev > 0.03  # >3% de EV positivo

        key_factors, risk_factors = [], []
        if has_value:
            key_factors.append(f"Value bet detectado: {best_bet} EV={best_ev:+.1%}")
        if snap.house_margin > 7:
            risk_factors.append(f"Margen alto ({snap.house_margin:.1f}%): AO severa")

        steam = snap.steam_move
        if steam:
            key_factors.append(f"Steam move: {steam['interpretation']}")

        # VR analysis
        vr = self.betting.get_vr_schedule_analysis()

        betting_analysis = {
            "event": f"{home} vs {away}",
            "model_probs": model_probs,
            "market_probs": market_probs,
            "edges": {k: round(v, 4) for k, v in edges.items()},
            "best_bet": best_bet,
            "best_ev": round(best_ev, 4),
            "has_value": has_value,
            "house_margin": snap.house_margin,
            "steam_move": steam,
            "vr_warning": vr["chasing_losses"],
        }

        pred = UnifiedPrediction(
            domain="betting",
            subject=f"{home} vs {away}",
            predicted_outcome=f"value_{best_bet}" if has_value else "no_value",
            confidence=round(min(0.9, 0.5 + best_ev * 2), 3),
            probabilities={
                "value_home": max(0, edges["home"]),
                "value_draw": max(0, edges["draw"]),
                "value_away": max(0, edges["away"]),
            },
            dominant_function="tangible",
            behavioral_edge=best_bet if has_value else "house",
            matching_distribution=model_probs,
            momentum_score=sports_pred.momentum_score,
            sentiment_score=sports_pred.sentiment_score,
            sports_prediction=sports_pred.__dict__,
            betting_analysis=betting_analysis,
            key_factors=key_factors,
            risk_factors=risk_factors,
        )
        with self._lock:
            self.predictions.append(pred)
        return pred

    # ── Outcome Registration ───────────────────────────────────────────────

    def record_outcome(self, subject: str, domain: str, outcome: str):
        """
        Registra el resultado real para aprendizaje.
        Actualiza: Matching Law, FBA, optimizadores, historial de precision.
        """
        with self._lock:
            relevant = [p for p in self.predictions
                        if p.subject == subject and p.domain == domain
                        and p.actual_outcome is None]
        if not relevant:
            return

        pred = relevant[-1]
        pred.actual_outcome = outcome

        # ¿Acerto?
        if domain == "sports":
            # outcome: nombre del ganador o "empate"/"draw"
            pred.was_correct = (pred.predicted_outcome == outcome)
            magnitude = 0.5 if pred.was_correct else -0.3
            # Actualizar historial del colector de deportes
            # El mapping requiere saber home/away — extraer del subject
            parts = pred.subject.split(" vs ")
            if len(parts) == 2:
                home, away = parts
                result_map = {
                    home:     "home",
                    "empate": "draw",
                    "draw":   "draw",
                    away:     "away",
                }
                raw_outcome = result_map.get(outcome, "draw")
                self.sports.record_outcome(home, away, raw_outcome)

        elif domain == "market":
            pred.was_correct = (pred.predicted_outcome == outcome)
            magnitude = 0.5 if pred.was_correct else -0.25
            self.matching_law.record_outcome(subject, "market",
                                              magnitude * 0.5)

        elif domain == "masses":
            pred.was_correct = (pred.predicted_outcome == outcome)
            magnitude = 0.4 if pred.was_correct else -0.2

        elif domain == "betting":
            pred.was_correct = ("value" in pred.predicted_outcome and pred.was_correct)
            magnitude = 0.6 if pred.was_correct else -0.35

        else:
            pred.was_correct = (pred.predicted_outcome == outcome)
            magnitude = 0.3 if pred.was_correct else -0.2

        # Registrar en Matching Law global
        self.matching_law.record_outcome(subject, domain, magnitude)

        logger.info(f"Outcome [{domain}] {subject}: {outcome} "
                    f"-> {'CORRECT' if pred.was_correct else 'WRONG'} "
                    f"(mag={magnitude:+.2f})")

    # ── Global Statistics ──────────────────────────────────────────────────

    def get_global_accuracy(self) -> Dict:
        """Precision global del motor por dominio."""
        stats = {}
        for domain in self.DOMAINS:
            domain_preds = [p for p in self.predictions
                            if p.domain == domain and p.was_correct is not None]
            if not domain_preds:
                stats[domain] = {"total": 0, "correct": 0, "accuracy": 0.0}
                continue
            correct = sum(1 for p in domain_preds if p.was_correct)
            stats[domain] = {
                "total":    len(domain_preds),
                "correct":  correct,
                "accuracy": round(correct / len(domain_preds), 3),
            }
        return stats

    def get_full_analysis(self, domain: str) -> Dict:
        """Analisis completo del estado ABA de un dominio."""
        return {
            "fba":          {h.behavior: h.__dict__
                             for h in self.fba.get_top_hypotheses(domain)},
            "matching_law": self.matching_law.get_domain_summary(domain),
            "active_mos":   [mo.to_dict() for mo in self.active_mos
                             if mo.current_strength > 0.1],
            "optimizer":    self.optimizers[domain].get_performance_summary()
                             if domain in self.optimizers else {},
            "sentiment":    self.sentiment.get_aggregate_sentiment(),
        }

    def get_recent_predictions(self, n: int = 10) -> List[Dict]:
        """Ultimas N predicciones en formato dict."""
        with self._lock:
            recent = list(self.predictions)[-n:]
        return [p.to_dict() for p in reversed(recent)]
