"""
Recolector de datos deportivos.

Mide las variables que determinan el comportamiento de masas y apostadores.
Fuentes: APIs publicas gratuitas + Wikipedia para contexto historico.

Variables ABA clave:
- Forma reciente (tasa de reforzamiento del equipo)
- Lesiones/bajas (Operacion de Abolicion: reduce valor del reforzador)
- Sentimiento de hinchada (Atencion social como funcion)
- Historial de enfrentamientos (historia de reforzamiento)
- Cuotas de apuestas (proxy del "consenso" de la masa)
"""

import json
import time
import logging
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("autoia.prediction.sports")


@dataclass
class TeamStats:
    """Estadisticas de un equipo con enfoque conductual."""
    name:           str
    league:         str
    sport:          str

    # Forma reciente (tasa de reforzamiento)
    recent_results: List[str] = field(default_factory=list)  # W/D/L
    goals_scored:   List[int] = field(default_factory=list)
    goals_conceded: List[int] = field(default_factory=list)

    # MOs activos
    injuries:       List[str] = field(default_factory=list)  # AO
    suspensions:    List[str] = field(default_factory=list)  # AO
    key_returns:    List[str] = field(default_factory=list)  # EO

    # Sentimiento
    fan_sentiment:  float = 0.0   # -1 a 1
    media_coverage: float = 0.0   # 0 a 1 (cantidad de atencion)

    # Contexto
    home_advantage: bool  = True
    rest_days:      int   = 3
    travel_fatigue: float = 0.0   # 0 a 1

    @property
    def win_rate(self) -> float:
        """Tasa de victorias en los ultimos N partidos."""
        if not self.recent_results:
            return 0.5
        wins = self.recent_results.count("W")
        return wins / len(self.recent_results)

    @property
    def reinforcement_rate(self) -> float:
        """
        Tasa de reforzamiento: combina victorias, goles y clean sheets.
        Es lo que 'refuerza' la conducta de apostar por este equipo.
        """
        if not self.recent_results:
            return 0.5
        n = len(self.recent_results)
        wins   = self.recent_results.count("W") / n
        draws  = self.recent_results.count("D") / n * 0.3
        goals  = sum(self.goals_scored[-n:]) / max(1, n) / 3.0  # normalizado a 3 goles
        return min(1.0, (wins + draws + goals) / 2.3)

    @property
    def mo_score(self) -> float:
        """
        Score neto de Operaciones Motivadoras.
        + = EOs dominan (el equipo es mas atractivo de lo normal)
        - = AOs dominan (el equipo es menos atractivo de lo normal)
        """
        ao_count = len(self.injuries) + len(self.suspensions)
        eo_count = len(self.key_returns)
        if self.home_advantage:
            eo_count += 1
        if self.rest_days >= 5:
            eo_count += 1
        elif self.rest_days <= 2:
            ao_count += 1
        ao_count += self.travel_fatigue > 0.6
        net = (eo_count - ao_count) / max(1, eo_count + ao_count)
        return round(net, 2)

    @property
    def form_string(self) -> str:
        return "".join(self.recent_results[-5:])

    @property
    def behavioral_momentum(self) -> float:
        """
        Impulso conductual: cuanto tarda la masa en cambiar de opinion.
        Series de victorias = alto impulso.
        """
        if not self.recent_results:
            return 0.0
        streak = 0
        for r in reversed(self.recent_results):
            if r == "W":
                streak += 1
            elif r == "D":
                streak += 0.3
            else:
                break
        return min(5.0, streak * 0.8 + self.win_rate * 2.0)


@dataclass
class MatchPrediction:
    """Prediccion de un partido con analisis ABA completo."""
    home_team:        str
    away_team:        str
    sport:            str
    match_time:       Optional[float] = None

    # Prediccion
    home_win_prob:    float = 0.33
    draw_prob:        float = 0.33
    away_win_prob:    float = 0.33
    predicted_winner: str  = "incierto"
    confidence:       float = 0.5

    # Analisis ABA
    dominant_mo:      str  = ""   # MO que mas influye
    behavioral_edge:  str  = ""   # Que equipo tiene ventaja conductual
    matching_law_dist: Dict = field(default_factory=dict)  # distribucion segun ML
    functional_analysis: Dict = field(default_factory=dict)

    # Factores clave
    key_factors:      List[str] = field(default_factory=list)
    risk_factors:     List[str] = field(default_factory=list)

    timestamp:        float = field(default_factory=time.time)
    outcome:          Optional[str] = None    # se llena post-partido
    was_correct:      Optional[bool] = None


class SportsCollector:
    """
    Recolecta y analiza datos deportivos con enfoque ABA.

    Puede usarse con cualquier deporte ajustando los parametros.
    Sin API: usa heuristicas + datos manuales.
    Con API: integra football-data.org, ESPN, etc.
    """

    def __init__(self, sport: str = "football",
                 api_key: Optional[str] = None):
        self.sport     = sport
        self.api_key   = api_key
        self.teams:    Dict[str, TeamStats] = {}
        self.history:  deque = deque(maxlen=500)  # historial de predicciones

    def add_team(self, name: str, league: str = "general") -> TeamStats:
        if name not in self.teams:
            self.teams[name] = TeamStats(name=name, league=league, sport=self.sport)
        return self.teams[name]

    def update_team_result(self, team_name: str, result: str,
                           goals_for: int = 0, goals_against: int = 0):
        """Actualiza el historial de un equipo tras un partido."""
        team = self.add_team(team_name)
        team.recent_results.append(result)
        team.goals_scored.append(goals_for)
        team.goals_conceded.append(goals_against)
        # Mantener solo los ultimos 10 partidos
        if len(team.recent_results) > 10:
            team.recent_results.pop(0)
            team.goals_scored.pop(0)
            team.goals_conceded.pop(0)

    def predict_match(self, home: str, away: str) -> MatchPrediction:
        """
        Genera prediccion de partido usando:
        1. Tasas de reforzamiento (Matching Law)
        2. Operaciones Motivadoras (MOs)
        3. Impulso conductual
        4. Analisis funcional del historial
        """
        home_team = self.teams.get(home) or self.add_team(home)
        away_team = self.teams.get(away) or self.add_team(away)

        # ── Paso 1: Matching Law ──────────────────────────────────────────
        home_rr = home_team.reinforcement_rate
        away_rr = away_team.reinforcement_rate
        total_rr = home_rr + away_rr

        if total_rr > 0:
            home_base = home_rr / total_rr
            away_base = away_rr / total_rr
        else:
            home_base = away_base = 0.5

        # ── Paso 2: Ajuste por MOs ────────────────────────────────────────
        home_mo = home_team.mo_score
        away_mo = away_team.mo_score

        home_adj = home_base * (1 + home_mo * 0.3)
        away_adj = away_base * (1 + away_mo * 0.3)

        # ── Paso 3: Home advantage como EO permanente ─────────────────────
        if home_team.home_advantage:
            home_adj *= 1.12  # ~12% ventaja local historica en futbol

        # ── Paso 4: Impulso conductual ────────────────────────────────────
        home_momentum = home_team.behavioral_momentum
        away_momentum = away_team.behavioral_momentum
        momentum_factor = (home_momentum - away_momentum) * 0.05
        home_adj += momentum_factor
        away_adj -= momentum_factor

        # ── Paso 5: Normalizar y calcular probabilidad de empate ──────────
        total_adj = home_adj + away_adj
        if total_adj == 0:
            home_win, away_win = 0.4, 0.3
        else:
            home_win = home_adj / total_adj * 0.75  # 75% del espacio (sin empate)
            away_win = away_adj / total_adj * 0.75

        draw_prob = max(0.10, min(0.35,
            0.30 - abs(home_win - away_win) * 0.5
        ))
        # Renormalizar
        total = home_win + away_win + draw_prob
        home_win  /= total
        away_win  /= total
        draw_prob /= total

        # ── Paso 6: Prediccion final ──────────────────────────────────────
        probs = {"home": home_win, "draw": draw_prob, "away": away_win}
        predicted = max(probs, key=probs.get)
        winner_name = home if predicted == "home" else (
            away if predicted == "away" else "empate")
        confidence = probs[predicted]

        # ── Paso 7: Factores clave ────────────────────────────────────────
        key_factors = []
        risk_factors = []

        if home_team.behavioral_momentum > 3:
            key_factors.append(f"{home} con racha positiva (momentum={home_momentum:.1f})")
        if away_team.behavioral_momentum > 3:
            key_factors.append(f"{away} con racha positiva (momentum={away_momentum:.1f})")
        if home_team.injuries:
            risk_factors.append(f"Lesiones {home}: {', '.join(home_team.injuries[:2])}")
        if away_team.injuries:
            risk_factors.append(f"Lesiones {away}: {', '.join(away_team.injuries[:2])}")
        if home_team.home_advantage:
            key_factors.append(f"Ventaja local para {home} (EO activo)")
        if abs(home_mo - away_mo) > 0.3:
            dom = home if home_mo > away_mo else away
            key_factors.append(f"MOs favorables para {dom}")

        # Dominant MO
        all_mo = home_mo + away_mo
        dom_mo = "EO" if all_mo > 0 else ("AO" if all_mo < 0 else "neutral")

        pred = MatchPrediction(
            home_team=home,
            away_team=away,
            sport=self.sport,
            home_win_prob=round(home_win, 3),
            draw_prob=round(draw_prob, 3),
            away_win_prob=round(away_win, 3),
            predicted_winner=winner_name,
            confidence=round(confidence, 3),
            dominant_mo=dom_mo,
            behavioral_edge=home if home_adj > away_adj else away,
            matching_law_dist={
                home: round(home_base, 3),
                away: round(away_base, 3),
            },
            key_factors=key_factors,
            risk_factors=risk_factors,
        )
        self.history.append(pred)
        return pred

    def record_outcome(self, home: str, away: str, outcome: str):
        """
        Registra el resultado real para medir precision y actualizar el modelo.
        outcome: "home" | "draw" | "away"
        """
        # Encontrar prediccion correspondiente
        for pred in reversed(list(self.history)):
            if pred.home_team == home and pred.away_team == away and pred.outcome is None:
                pred.outcome = outcome
                pred.was_correct = (pred.predicted_winner == (
                    home if outcome == "home" else
                    away if outcome == "away" else "empate"
                ))
                break

        # Actualizar estadisticas de equipos
        goals = {"home": (1, 0), "draw": (0, 0), "away": (0, 1)}
        gf, gc = goals.get(outcome, (0, 0))
        result_map = {"home": ("W", "L"), "draw": ("D", "D"), "away": ("L", "W")}
        hr, ar = result_map.get(outcome, ("D", "D"))
        self.update_team_result(home, hr, gf, gc)
        self.update_team_result(away, ar, gc, gf)

    def get_accuracy(self) -> Dict:
        """Precision historica del modelo."""
        done = [p for p in self.history if p.was_correct is not None]
        if not done:
            return {"total": 0, "correct": 0, "accuracy": 0.0}
        correct = sum(1 for p in done if p.was_correct)
        return {
            "total":    len(done),
            "correct":  correct,
            "accuracy": round(correct / len(done), 3),
            "avg_confidence": round(sum(p.confidence for p in done) / len(done), 3),
        }
