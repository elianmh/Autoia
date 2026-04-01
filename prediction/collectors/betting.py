"""
Recolector de datos de apuestas.

Analiza las cuotas de casas de apuestas como proxy del comportamiento masivo.
Las cuotas reflejan:
1. El consenso de la masa (Matching Law aplicada)
2. El margen de la casa (reforzador del operador)
3. Patrones de chasing losses (VR schedule en apostadores)
4. Movimientos de cuota = MOs que afectan la conducta de apuesta

ABA en apuestas:
- VR (Variable Ratio): el reforzamiento impredecible genera la mayor tasa y resistencia
- Chasing losses: conducta mantenida por reforzamiento negativo (evitar sensacion de perdida)
- House edge: AO sistematica que hace la conducta no rentable a largo plazo
- Steam moves: MOs masivos que mueven cuotas rapidamente
"""

import time
import json
import math
import logging
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("autoia.prediction.betting")


@dataclass
class OddsSnapshot:
    """Cuotas actuales de un evento."""
    event_id:     str
    home_team:    str
    away_team:    str
    sport:        str
    league:       str

    # Cuotas decimales (europeas)
    home_odds:    float = 2.0
    draw_odds:    float = 3.5
    away_odds:    float = 3.0

    # Timestamp y fuente
    timestamp:    float = field(default_factory=time.time)
    source:       str   = "manual"

    # Movimiento de cuota (Steam Move)
    home_prev:    float = 0.0   # cuota anterior
    away_prev:    float = 0.0
    draw_prev:    float = 0.0

    @property
    def house_margin(self) -> float:
        """
        Margen de la casa (vig/juice/overround).
        1/home + 1/draw + 1/away > 1.0
        El exceso es el margen. Tipico: 4-8%.
        """
        if 0 in (self.home_odds, self.draw_odds, self.away_odds):
            return 0.0
        total = 1/self.home_odds + 1/self.draw_odds + 1/self.away_odds
        return round((total - 1.0) * 100, 2)  # en %

    @property
    def implied_probs(self) -> Dict[str, float]:
        """
        Probabilidades implicitas (sin ajuste por margen).
        Refleja la opinion de la masa.
        """
        if 0 in (self.home_odds, self.draw_odds, self.away_odds):
            return {"home": 0.33, "draw": 0.33, "away": 0.33}
        raw_home = 1 / self.home_odds
        raw_draw = 1 / self.draw_odds
        raw_away = 1 / self.away_odds
        total = raw_home + raw_draw + raw_away
        return {
            "home": round(raw_home / total, 3),
            "draw": round(raw_draw / total, 3),
            "away": round(raw_away / total, 3),
        }

    @property
    def fair_probs(self) -> Dict[str, float]:
        """Probabilidades justas ajustadas por el margen de la casa."""
        margin_factor = 1 + self.house_margin / 100
        implied = self.implied_probs
        return {k: round(v / margin_factor, 3) for k, v in implied.items()}

    @property
    def steam_move(self) -> Optional[Dict]:
        """
        Detecta un Steam Move: movimiento brusco de cuota.
        Indica que dinero 'inteligente' o informacion nueva ha llegado.
        """
        if not self.home_prev:
            return None
        home_shift = (self.home_odds - self.home_prev) / self.home_prev
        away_shift = (self.away_odds - self.away_prev) / self.away_prev
        threshold = 0.05  # 5% de movimiento
        if abs(home_shift) > threshold or abs(away_shift) > threshold:
            direction = "home" if home_shift < -threshold else "away"
            magnitude = max(abs(home_shift), abs(away_shift))
            return {
                "direction":  direction,
                "magnitude":  round(magnitude * 100, 1),  # %
                "mo_type":    "EO",  # steam move siempre activa conducta
                "interpretation": f"Dinero masivo entrando en {direction}",
            }
        return None

    @property
    def value_bet(self) -> Optional[Dict]:
        """
        Detecta value bet: cuota que paga mas de la probabilidad real.
        Valor esperado positivo para el apostador (raro).
        """
        # Sin estimacion externa de probabilidad real, no podemos calcular
        # Se llena externamente comparando con prediccion del modelo
        return None


@dataclass
class BettingPattern:
    """
    Patron conductual de un apostador/mercado.
    Detecta VR schedule, chasing, y otras conductas.
    """
    entity_id:    str   # id del apostador o mercado
    bet_history:  List[Dict] = field(default_factory=list)  # {amount, odds, result, timestamp}

    @property
    def vr_schedule_strength(self) -> float:
        """
        Fuerza del efecto VR (Variable Ratio).
        Alta si los reforzamientos son impredecibles pero frecuentes.
        Predice: alta resistencia a extincion, alta tasa de conducta.
        """
        if len(self.bet_history) < 10:
            return 0.0
        results = [b.get("result", "loss") for b in self.bet_history[-20:]]
        wins = results.count("win")
        if wins == 0:
            return 0.0
        win_rate = wins / len(results)
        # VR optimo: ~30-50% de reforzamiento (maximo engagement)
        # Demasiado alto = FR, demasiado bajo = extincion
        vr_peak = 0.4
        vr_strength = 1.0 - abs(win_rate - vr_peak) / vr_peak
        return round(max(0, vr_strength), 3)

    @property
    def chasing_loss_score(self) -> float:
        """
        Detecta patron de 'chasing losses'.
        Señal: apuestas mayores despues de perdidas consecutivas.
        Reforzado negativamente: reducir sensacion de perdida acumulada.
        """
        if len(self.bet_history) < 5:
            return 0.0
        recent = self.bet_history[-10:]
        chasing_count = 0
        for i in range(1, len(recent)):
            prev, curr = recent[i-1], recent[i]
            if (prev.get("result") == "loss" and
                    curr.get("amount", 0) > prev.get("amount", 0) * 1.2):
                chasing_count += 1
        return round(chasing_count / max(1, len(recent) - 1), 3)

    @property
    def expected_value_awareness(self) -> float:
        """
        Mide si el apostador selecciona apuestas con valor esperado positivo.
        Alto: apostador racional. Bajo: conducta impulsiva.
        """
        if not self.bet_history:
            return 0.5
        ev_positive = sum(1 for b in self.bet_history
                         if b.get("ev", -1) > 0)
        return round(ev_positive / len(self.bet_history), 3)


class BettingCollector:
    """
    Analiza el mercado de apuestas desde perspectiva ABA.

    Fuentes:
    1. Odds manuales / API externa (si se provee key)
    2. The Odds API (gratuita con limite)
    3. Analisis de patrones sin datos en tiempo real

    ABA aplicado:
    - Las cuotas son el SD: discriminan cuando la masa apuesta
    - Steam moves = MOs que activan conducta masiva
    - House margin = AO sistematica (hace la conducta no rentable)
    - VR schedule = por que el apostador no puede parar
    """

    ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"

    def __init__(self, api_key: Optional[str] = None,
                 sport: str = "soccer_spain_la_liga"):
        self.api_key   = api_key
        self.sport     = sport
        self.events:   Dict[str, OddsSnapshot] = {}
        self.patterns: Dict[str, BettingPattern] = {}
        self.history:  deque = deque(maxlen=1000)
        self._cache_ttl = 300  # 5 min
        self._last_fetch: float = 0

    def add_manual_odds(self, home: str, away: str,
                        home_odds: float, draw_odds: float, away_odds: float,
                        league: str = "general",
                        home_prev: float = 0.0, away_prev: float = 0.0,
                        draw_prev: float = 0.0) -> OddsSnapshot:
        """Agrega cuotas manuales (sin API)."""
        event_id = f"{home}_vs_{away}".lower().replace(" ", "_")
        snap = OddsSnapshot(
            event_id=event_id,
            home_team=home,
            away_team=away,
            sport=self.sport,
            league=league,
            home_odds=home_odds,
            draw_odds=draw_odds,
            away_odds=away_odds,
            home_prev=home_prev,
            away_prev=away_prev,
            draw_prev=draw_prev,
            source="manual",
        )
        self.events[event_id] = snap
        self.history.append(snap)
        logger.info(f"Odds {home} vs {away}: {home_odds}/{draw_odds}/{away_odds} "
                    f"margin={snap.house_margin:.1f}%")
        return snap

    def fetch_live_odds(self, force: bool = False) -> List[OddsSnapshot]:
        """
        Obtiene cuotas en vivo de The Odds API.
        Requiere API key. Sin key retorna lista vacia.
        """
        if not self.api_key:
            logger.info("No API key para The Odds API — usando datos manuales")
            return list(self.events.values())

        now = time.time()
        if not force and (now - self._last_fetch) < self._cache_ttl:
            return list(self.events.values())

        try:
            params = urllib.parse.urlencode({
                "apiKey": self.api_key,
                "regions": "eu",
                "markets": "h2h",
                "oddsFormat": "decimal",
            })
            url = f"{self.ODDS_API_BASE}/{self.sport}/odds?{params}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                events_data = json.loads(resp.read().decode("utf-8"))

            new_snaps = []
            for ev in events_data:
                bookmakers = ev.get("bookmakers", [])
                if not bookmakers:
                    continue

                # Promedio de todos los bookmakers disponibles
                home_odds_list, draw_odds_list, away_odds_list = [], [], []
                home_name = ev.get("home_team", "")
                away_name = ev.get("away_team", "")

                for bk in bookmakers:
                    for market in bk.get("markets", []):
                        if market.get("key") != "h2h":
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == home_name:
                                home_odds_list.append(outcome.get("price", 2.0))
                            elif outcome.get("name") == away_name:
                                away_odds_list.append(outcome.get("price", 2.0))
                            elif outcome.get("name") == "Draw":
                                draw_odds_list.append(outcome.get("price", 3.5))

                if not (home_odds_list and away_odds_list):
                    continue

                # Cuota media del mercado
                avg_home = sum(home_odds_list) / len(home_odds_list)
                avg_away = sum(away_odds_list) / len(away_odds_list)
                avg_draw = (sum(draw_odds_list) / len(draw_odds_list)
                            if draw_odds_list else 3.5)

                # Comparar con datos anteriores para detectar steam moves
                event_id = ev.get("id", f"{home_name}_vs_{away_name}")
                prev = self.events.get(event_id)

                snap = OddsSnapshot(
                    event_id=event_id,
                    home_team=home_name,
                    away_team=away_name,
                    sport=self.sport,
                    league=ev.get("sport_key", ""),
                    home_odds=round(avg_home, 3),
                    draw_odds=round(avg_draw, 3),
                    away_odds=round(avg_away, 3),
                    home_prev=prev.home_odds if prev else 0.0,
                    away_prev=prev.away_odds if prev else 0.0,
                    draw_prev=prev.draw_odds if prev else 0.0,
                    source="odds_api",
                )
                self.events[event_id] = snap
                self.history.append(snap)
                new_snaps.append(snap)

            self._last_fetch = now
            logger.info(f"Obtenidas {len(new_snaps)} cuotas en vivo")
            return new_snaps

        except Exception as e:
            logger.warning(f"Error fetching odds: {e}")
            return list(self.events.values())

    def analyze_market_sentiment(self, event_id: str) -> Dict:
        """
        Analiza el sentimiento del mercado de apuestas.
        Las cuotas bajas = la masa apuesta mas = EO activo para ese equipo.
        """
        snap = self.events.get(event_id)
        if not snap:
            return {}

        implied = snap.implied_probs
        steam   = snap.steam_move

        # El lado con menor cuota tiene mas dinero masivo -> EO dominante
        dominant = max(implied, key=implied.get)
        dominant_name = (snap.home_team if dominant == "home" else
                         snap.away_team if dominant == "away" else "empate")

        # Fuerza del consenso masivo (0 = equilibrado, 1 = unanime)
        probs = list(implied.values())
        consensus = max(probs) - sum(p for p in probs if p != max(probs)) / max(1, len(probs)-1)

        # Valor esperado para apostador promedio
        # VE = (prob_real * cuota) - 1 (negativo siempre por margen)
        ve_home = implied["home"] * snap.home_odds - 1
        ve_away = implied["away"] * snap.away_odds - 1
        # El EV real es negativo por el margen, pero comparamos relativamente

        analysis = {
            "event":       f"{snap.home_team} vs {snap.away_team}",
            "house_margin": snap.house_margin,
            "implied_probs": implied,
            "mass_consensus": {
                "dominant":  dominant_name,
                "strength":  round(consensus, 3),
                "mo_type":   "EO" if consensus > 0.15 else "neutral",
            },
            "steam_move": steam,
            "aba_interpretation": self._aba_interpret(snap),
            "house_edge_aba": {
                "type":  "AO",
                "description": f"Margen {snap.house_margin:.1f}% reduce valor de reforzador monetario",
                "long_term_effect": "extinction" if snap.house_margin > 6 else "gradual",
            },
        }
        return analysis

    def _aba_interpret(self, snap: OddsSnapshot) -> str:
        """Interpreta las cuotas en lenguaje ABA."""
        parts = []
        steam = snap.steam_move
        if steam:
            parts.append(f"STEAM_MOVE hacia {steam['direction']} "
                         f"({steam['magnitude']:.0f}%) — EO masivo activo")

        implied = snap.implied_probs
        max_side = max(implied, key=implied.get)
        if implied[max_side] > 0.55:
            parts.append(f"CONSENSO_MASIVO: masa apuesta por {max_side} "
                         f"(SD claro, tasa de conducta alta)")

        if snap.house_margin > 8:
            parts.append(f"AO_ALTA: margen {snap.house_margin:.1f}% "
                         f"suprime valor del reforzador")
        elif snap.house_margin < 4:
            parts.append(f"MERCADO_EFICIENTE: margen bajo, cuotas competitivas")

        return " | ".join(parts) if parts else "NEUTRO"

    def get_vr_schedule_analysis(self) -> Dict:
        """
        Analiza el efecto del VR schedule en las apuestas deportivas.
        Explica por que los apostadores no pueden parar a pesar de perder.
        """
        return {
            "schedule_type": "VR (Variable Ratio)",
            "why_addictive": [
                "Reforzamiento impredecible genera mayor tasa de conducta que FR/FI",
                "Alta resistencia a extincion: el apostador sigue aunque pierda seguido",
                "Cada nueva apuesta puede ser 'la ganadora' — nunca se sabe cuando",
                "Neurobiologicamente equivalente a tragamonedas",
            ],
            "aba_mechanism": {
                "MO": "Privacion de dinero/emocion actua como EO temporal",
                "SD": "Cuotas atractivas + historial de victorias pasadas",
                "R":  "Apostar (conducta operante)",
                "C":  "Ganancia ocasional (VR+) o perdida (castigo inconsistente)",
            },
            "chasing_losses": {
                "function":      "Escape/Avoidance",
                "description":   "Apostar para recuperar = reforzamiento negativo",
                "why_persistent": "Evitar la sensacion de perdida es mas poderoso que buscar ganancia",
            },
            "house_edge_over_time": {
                "monthly_expected_loss": "5-15% del total apostado",
                "extinction_curve":      "gradual (VR retrasa extincion) pero inevitable",
            },
        }

    def get_event_prediction(self, event_id: str) -> Optional[Dict]:
        """
        Genera prediccion ABA para un evento basada en cuotas.
        Considera: cuotas implicitas, steam moves, margen de casa.
        """
        snap = self.events.get(event_id)
        if not snap:
            return None

        implied = snap.fair_probs  # ajustadas por margen
        steam   = snap.steam_move

        # Si hay steam move, ajustar hacia donde va el dinero inteligente
        if steam and steam["magnitude"] > 8:
            shift = min(0.08, steam["magnitude"] / 100 * 0.5)
            if steam["direction"] == "home":
                implied["home"] = min(0.95, implied["home"] + shift)
                implied["away"] = max(0.02, implied["away"] - shift)
            else:
                implied["away"] = min(0.95, implied["away"] + shift)
                implied["home"] = max(0.02, implied["home"] - shift)

            # Renormalizar
            total = sum(implied.values())
            implied = {k: round(v/total, 3) for k, v in implied.items()}

        predicted = max(implied, key=implied.get)
        winner_name = (snap.home_team if predicted == "home" else
                       snap.away_team if predicted == "away" else "empate")

        return {
            "event_id":   event_id,
            "home_team":  snap.home_team,
            "away_team":  snap.away_team,
            "probs":      implied,
            "predicted":  winner_name,
            "confidence": implied[predicted],
            "steam_boost": steam is not None,
            "source":     "odds_analysis",
        }

    def record_result(self, event_id: str, outcome: str):
        """Registra resultado real para medir precision del analisis de cuotas."""
        snap = self.events.get(event_id)
        if not snap:
            return
        # En una extension futura: comparar con prediccion y actualizar accuracy
        logger.info(f"Resultado {event_id}: {outcome}")
