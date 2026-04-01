"""
Dashboard pygame para el sistema de prediccion ABA.

Muestra en tiempo real:
- Panel izquierdo: dominios activos y predicciones
- Panel central: distribucion Matching Law (barras)
- Panel derecho: MOs activos, factores clave, sentimiento
- Barra inferior: historial de aciertos por dominio

Controles:
  [1-4]   Cambiar dominio (Sports/Market/Masses/Betting)
  [R]     Refrescar datos
  [Q/Esc] Salir
  [Enter] Añadir prediccion manual (modo demo)
"""

import sys
import time
import math
import threading
import logging
from typing import Dict, List, Optional, Tuple

try:
    import pygame
    PYGAME_OK = True
except ImportError:
    PYGAME_OK = False

from ..engine import PredictionEngine, UnifiedPrediction

logger = logging.getLogger("autoia.prediction.dashboard")

# ── Paleta de colores ────────────────────────────────────────────────────────
C = {
    "bg":        (10,  12,  20),
    "panel":     (18,  22,  35),
    "border":    (35,  45,  65),
    "header":    (25,  30,  50),
    "text":      (200, 210, 230),
    "text_dim":  (100, 115, 140),
    "accent":    (80,  160, 255),
    "green":     (60,  200, 100),
    "red":       (220,  70,  70),
    "yellow":    (240, 200,  60),
    "orange":    (240, 140,  40),
    "purple":    (160, 100, 240),
    "eo":        (80,  200, 120),   # EO color
    "ao":        (220,  80,  80),   # AO color
    "neutral":   (140, 140, 160),
    "sports":    (80,  160, 255),
    "market":    (60,  200, 100),
    "masses":    (240, 200,  60),
    "betting":   (240, 140,  40),
}

DOMAIN_COLORS = {
    "sports":  C["sports"],
    "market":  C["market"],
    "masses":  C["masses"],
    "betting": C["betting"],
}


class Dashboard:
    """
    Dashboard de prediccion ABA.
    Renderiza en pygame las predicciones y el estado ABA del sistema.
    """

    W, H = 1280, 720

    def __init__(self, engine: PredictionEngine):
        if not PYGAME_OK:
            raise RuntimeError("pygame no disponible. Instala: pip install pygame")

        self.engine = engine
        self.running = False
        self.domain = "sports"
        self.current_pred: Optional[UnifiedPrediction] = None
        self.font_cache: Dict[int, "pygame.font.Font"] = {}
        self._input_text = ""
        self._input_active = False
        self._demo_queue: List[dict] = []

    # ── Font helper ──────────────────────────────────────────────────────────

    def font(self, size: int = 14, bold: bool = False) -> "pygame.font.Font":
        key = (size, bold)
        if key not in self.font_cache:
            for name in (["Segoe UI", "Calibri", "Arial", "Liberation Sans"]
                         if not bold else ["Segoe UI Bold", "Arial Bold", "Arial"]):
                try:
                    self.font_cache[key] = pygame.font.SysFont(name, size, bold=bold)
                    break
                except Exception:
                    continue
            if key not in self.font_cache:
                self.font_cache[key] = pygame.font.SysFont(None, size)
        return self.font_cache[key]

    def txt(self, surf, text: str, pos: Tuple[int, int],
            color=None, size: int = 14, bold: bool = False,
            anchor: str = "topleft") -> "pygame.Rect":
        color = color or C["text"]
        f = self.font(size, bold)
        s = f.render(str(text), True, color)
        r = s.get_rect(**{anchor: pos})
        surf.blit(s, r)
        return r

    # ── Main Loop ────────────────────────────────────────────────────────────

    def run(self):
        """Inicia el dashboard."""
        pygame.init()
        flags = pygame.RESIZABLE
        try:
            flags |= pygame.SCALED
        except AttributeError:
            pass

        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("Autoia — ABA Prediction Engine")
        self.clock = pygame.time.Clock()
        self.running = True

        # Demo: cargar predicciones de ejemplo si no hay datos
        self._load_demo_data()

        while self.running:
            self._handle_events()
            self._render()
            self.clock.tick(30)  # 30 FPS suficiente para dashboard

        pygame.quit()

    def _load_demo_data(self):
        """Carga datos de demo para mostrar el sistema funcionando."""
        # Equipos deportivos con historial
        engine = self.engine

        # Real Madrid
        rm = engine.sports.add_team("Real Madrid", "La Liga")
        for r in ["W", "W", "D", "W", "W", "L", "W", "W", "D", "W"]:
            engine.sports.update_team_result("Real Madrid", r, 2, 1)
        rm.home_advantage = True
        rm.rest_days = 4

        # Barcelona
        barca = engine.sports.add_team("Barcelona", "La Liga")
        for r in ["L", "W", "W", "D", "L", "W", "W", "L", "W", "D"]:
            engine.sports.update_team_result("Barcelona", r, 1, 1)
        barca.injuries = ["Lewandowski", "Pedri"]
        barca.rest_days = 2

        # Prediccion demo
        self.current_pred = engine.predict_sports_match("Real Madrid", "Barcelona")

        # Mercado demo (sin fetch real)
        engine.market.add_symbol("BTC-USD")

        # Sentimiento demo
        for text in [
            "Real Madrid en racha historica, Mbappe imparable, equipo solido",
            "Barcelona con bajas graves, crisis interna, presion sobre el entrenador",
            "El Clasico mas esperado de la temporada, millones de espectadores",
        ]:
            engine.sentiment.analyze_text(text, source="demo")

        # MOs demo
        engine.add_mo("sports", "lesion_estrella", "Lewandowski baja confirmada",
                      "AO", "Barcelona", 0.8, 48)
        engine.add_mo("sports", "racha_positiva", "Real Madrid 8 victorias seguidas",
                      "EO", "Real Madrid", 0.9, 24)

    def _handle_events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE or ev.key == pygame.K_q:
                    self.running = False
                elif ev.key == pygame.K_1:
                    self.domain = "sports"
                    self._refresh_demo_pred()
                elif ev.key == pygame.K_2:
                    self.domain = "market"
                elif ev.key == pygame.K_3:
                    self.domain = "masses"
                elif ev.key == pygame.K_4:
                    self.domain = "betting"
                elif ev.key == pygame.K_r:
                    self._refresh_demo_pred()

    def _refresh_demo_pred(self):
        if self.domain == "sports":
            self.current_pred = self.engine.predict_sports_match(
                "Real Madrid", "Barcelona")
        elif self.domain == "market":
            pred = self.engine.predict_market("BTC-USD")
            # Si no hay datos reales, usar sentimiento como proxy
            if pred.confidence == 0:
                pred.confidence = 0.55
                pred.predicted_outcome = "up"
                pred.probabilities = {"up": 0.52, "flat": 0.18, "down": 0.30}
                pred.market_signal = "MOMENTUM_ALCISTA | PRIVACION"
                pred.key_factors = ["RSI bajo: privacion activa EO de compra",
                                     "Sentimiento masivo positivo"]
            self.current_pred = pred
        elif self.domain == "masses":
            self.current_pred = self.engine.predict_mass_trend(
                "criptomonedas",
                ["Bitcoin rally, FOMO masivo, todos quieren comprar",
                 "Instituciones acumulando, halving pronto"]
            )
        elif self.domain == "betting":
            self.current_pred = self.engine.predict_betting_value(
                "Real Madrid", "Barcelona",
                home_odds=1.85, draw_odds=3.60, away_odds=4.20
            )

    # ── Rendering ────────────────────────────────────────────────────────────

    def _render(self):
        self.screen.fill(C["bg"])

        # Layout
        pw = 300  # left panel width
        rw = 280  # right panel width
        cx = pw    # center x
        cw = self.W - pw - rw  # center width
        bh = 60    # bottom bar height
        ch = self.H - 50 - bh  # content height (below header)

        self._render_header(0, 0, self.W, 50)
        self._render_left_panel(0, 50, pw, ch)
        self._render_center(cx, 50, cw, ch)
        self._render_right_panel(cx + cw, 50, rw, ch)
        self._render_bottom_bar(0, self.H - bh, self.W, bh)

        pygame.display.flip()

    def _render_header(self, x, y, w, h):
        pygame.draw.rect(self.screen, C["header"], (x, y, w, h))
        pygame.draw.line(self.screen, C["border"], (x, y+h-1), (x+w, y+h-1))
        self.txt(self.screen, "AUTOIA — ABA PREDICTION ENGINE",
                 (x + 20, y + h//2), C["accent"], 18, bold=True, anchor="midleft")

        # Domain tabs
        domains = ["1:SPORTS", "2:MARKET", "3:MASSES", "4:BETTING"]
        tx = w - 30
        for d in reversed(domains):
            key, label = d.split(":")
            active = label.lower() == self.domain
            color = DOMAIN_COLORS.get(self.domain if active else label.lower(), C["text_dim"])
            fg = color if active else C["text_dim"]
            text = f"[{key}] {label}"
            r = self.txt(self.screen, text, (tx, y + h//2), fg, 13,
                         bold=active, anchor="midright")
            if active:
                pygame.draw.line(self.screen, color,
                                 (r.left, y+h-2), (r.right, y+h-2), 2)
            tx = r.left - 20

    def _render_left_panel(self, x, y, w, h):
        pygame.draw.rect(self.screen, C["panel"], (x, y, w, h))
        pygame.draw.line(self.screen, C["border"], (x+w-1, y), (x+w-1, y+h))
        cy = y + 12

        self.txt(self.screen, "PREDICCION ACTUAL", (x+12, cy), C["accent"], 12, bold=True)
        cy += 24
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10

        if self.current_pred is None:
            self.txt(self.screen, "Sin datos — presiona [1-4]", (x+12, cy), C["text_dim"], 12)
            return

        p = self.current_pred

        # Subject
        self.txt(self.screen, p.subject, (x+12, cy), C["text"], 13, bold=True)
        cy += 20
        self.txt(self.screen, p.domain.upper(), (x+12, cy),
                 DOMAIN_COLORS.get(p.domain, C["text_dim"]), 11)
        cy += 22

        # Predicted outcome
        pygame.draw.rect(self.screen, C["border"], (x+10, cy, w-20, 36), 0)
        pygame.draw.rect(self.screen, DOMAIN_COLORS.get(p.domain, C["accent"]),
                         (x+10, cy, w-20, 36), 1)
        self.txt(self.screen, p.predicted_outcome.upper(),
                 (x + w//2, cy + 10), C["text"], 15, bold=True, anchor="midtop")
        cy += 44

        # Confidence bar
        self.txt(self.screen, "Confianza", (x+12, cy), C["text_dim"], 11)
        cy += 14
        bar_w = w - 24
        conf = p.confidence
        conf_color = (C["green"] if conf > 0.65 else
                      C["yellow"] if conf > 0.45 else C["red"])
        pygame.draw.rect(self.screen, C["border"], (x+12, cy, bar_w, 8))
        pygame.draw.rect(self.screen, conf_color, (x+12, cy, int(bar_w * conf), 8))
        self.txt(self.screen, f"{conf:.0%}", (x+12+bar_w+4, cy-1), conf_color, 11)
        cy += 18

        # Probabilities
        cy += 6
        self.txt(self.screen, "Probabilidades (Matching Law)", (x+12, cy), C["text_dim"], 11)
        cy += 16
        for outcome, prob in p.probabilities.items():
            label = str(outcome)[:16]
            bar_full = w - 24 - 60
            bar_fill = int(bar_full * prob)
            col = (C["green"] if prob == max(p.probabilities.values()) else C["text_dim"])
            pygame.draw.rect(self.screen, C["border"], (x+12, cy, bar_full, 12))
            pygame.draw.rect(self.screen, col, (x+12, cy, bar_fill, 12))
            self.txt(self.screen, label, (x+12, cy-1), C["text"], 10)
            self.txt(self.screen, f"{prob:.1%}",
                     (x+12+bar_full+4, cy-1), col, 10)
            cy += 16

        # Behavioral edge
        cy += 8
        edge_color = C["green"] if p.behavioral_edge else C["text_dim"]
        self.txt(self.screen, f"Ventaja conductual:", (x+12, cy), C["text_dim"], 11)
        cy += 15
        self.txt(self.screen, p.behavioral_edge or "—", (x+20, cy), edge_color, 13, bold=True)
        cy += 22

        # Momentum
        self.txt(self.screen, f"Momentum: {p.momentum_score:.1f}/5.0",
                 (x+12, cy), C["text_dim"], 11)
        cy += 15
        m_bar = min(1.0, p.momentum_score / 5.0)
        pygame.draw.rect(self.screen, C["border"], (x+12, cy, w-24, 6))
        pygame.draw.rect(self.screen, C["orange"],
                         (x+12, cy, int((w-24)*m_bar), 6))
        cy += 14

        # Sentiment
        cy += 6
        sent_color = (C["green"] if p.sentiment_score > 0.1 else
                      C["red"] if p.sentiment_score < -0.1 else C["neutral"])
        self.txt(self.screen, f"Sentimiento masivo: {p.sentiment_score:+.2f}",
                 (x+12, cy), sent_color, 11)

    def _render_center(self, x, y, w, h):
        pygame.draw.rect(self.screen, C["bg"], (x, y, w, h))
        cy = y + 12

        self.txt(self.screen, "ANALISIS ABA — DISTRIBUCION CONDUCTUAL",
                 (x + w//2, cy), C["text"], 13, bold=True, anchor="midtop")
        cy += 30

        if self.current_pred is None:
            return

        p = self.current_pred

        # ── Matching Law distribution ──────────────────────────────────
        self.txt(self.screen, "Ley de Igualacion (tasa de reforzamiento por opcion)",
                 (x+16, cy), C["text_dim"], 11)
        cy += 20

        ml = p.matching_distribution
        if ml:
            max_val = max(ml.values()) if ml else 1
            bh = 18
            for name, val in ml.items():
                bar_w = int((w - 200) * val / max(0.01, max_val))
                col = DOMAIN_COLORS.get(p.domain, C["accent"])
                pygame.draw.rect(self.screen, col, (x+120, cy, bar_w, bh-2))
                self.txt(self.screen, str(name)[:14],
                         (x+16, cy+1), C["text"], 11)
                self.txt(self.screen, f"{val:.1%}",
                         (x+120+bar_w+6, cy+1), col, 11)
                cy += bh + 2
        else:
            self.txt(self.screen, "Sin distribucion disponible", (x+16, cy), C["text_dim"], 11)
        cy += 16

        # ── MOs activos ───────────────────────────────────────────────
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10
        self.txt(self.screen, "Operaciones Motivadoras Activas",
                 (x+16, cy), C["text"], 12, bold=True)
        cy += 20

        if p.active_mos:
            for mo in p.active_mos[:5]:
                mo_type = mo.get("mo_type", mo.get("type", "?"))
                color = C["eo"] if mo_type == "EO" else C["ao"]
                badge = f"[{mo_type}]"
                self.txt(self.screen, badge, (x+16, cy), color, 11, bold=True)
                desc = (mo.get("description", mo.get("source", ""))
                        or str(mo.get("score", "?")))
                self.txt(self.screen, str(desc)[:50], (x+60, cy), C["text"], 11)
                strength = mo.get("strength", mo.get("score", 0))
                if isinstance(strength, (int, float)):
                    s_w = int(80 * abs(float(strength)))
                    pygame.draw.rect(self.screen, color,
                                     (x+w-100, cy, s_w, 10))
                cy += 16
        else:
            self.txt(self.screen, "Sin MOs activos detectados",
                     (x+16, cy), C["text_dim"], 11)
            cy += 16
        cy += 10

        # ── Funcion conductual dominante ───────────────────────────────
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10
        self.txt(self.screen, "Funcion Conductual Dominante (FBA)",
                 (x+16, cy), C["text"], 12, bold=True)
        cy += 20
        func = p.dominant_function or "automatic"
        func_descs = {
            "tangible":  "TANGIBLE: conducta mantenida por ganancia monetaria directa",
            "escape":    "ESCAPE: conducta para evitar perdida (miedo/panico)",
            "attention": "ATTENTION: conducta para obtener reconocimiento social",
            "automatic": "AUTOMATICA: conducta impulsiva sin funcion social clara",
        }
        func_colors = {
            "tangible":  C["green"],
            "escape":    C["red"],
            "attention": C["yellow"],
            "automatic": C["purple"],
        }
        fcol = func_colors.get(func, C["text"])
        self.txt(self.screen, func_descs.get(func, func),
                 (x+16, cy), fcol, 12)
        cy += 26

        # Implicacion ABA
        impl = {
            "tangible":  "SD clave: precio, cuota, retorno esperado",
            "escape":    "AO activa: miedo reduce valor del reforzador positivo",
            "attention": "EO activo: visibilidad social amplifica conducta",
            "automatic": "VR schedule: impredecibilidad genera alta tasa de conducta",
        }
        self.txt(self.screen, "-> " + impl.get(func, ""),
                 (x+16, cy), C["text_dim"], 11)
        cy += 24

        # ── Contraste conductual / switching ─────────────────────────
        if p.contrast_risk:
            pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
            cy += 10
            self.txt(self.screen, "Riesgos de Contraste Conductual",
                     (x+16, cy), C["orange"], 12, bold=True)
            cy += 20
            for cr in p.contrast_risk[:2]:
                msg = str(cr) if isinstance(cr, str) else (
                    f"{cr.get('extinguishing','')} -> {cr.get('beneficiary','')}: "
                    f"+{cr.get('expected_increase_pct',0):.0f}% esperado"
                    if isinstance(cr, dict) else str(cr)
                )
                self.txt(self.screen, msg[:70], (x+16, cy), C["orange"], 11)
                cy += 16

        # ── Key factors ───────────────────────────────────────────────
        cy += 6
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10
        self.txt(self.screen, "Factores Clave",
                 (x+16, cy), C["green"], 12, bold=True)
        cy += 18
        for f in (p.key_factors or [])[:5]:
            self.txt(self.screen, "+ " + str(f)[:65], (x+16, cy), C["text"], 11)
            cy += 14

        if p.risk_factors:
            cy += 6
            self.txt(self.screen, "Factores de Riesgo",
                     (x+16, cy), C["red"], 12, bold=True)
            cy += 18
            for rf in p.risk_factors[:4]:
                self.txt(self.screen, "! " + str(rf)[:65], (x+16, cy), C["red"], 11)
                cy += 14

    def _render_right_panel(self, x, y, w, h):
        pygame.draw.rect(self.screen, C["panel"], (x, y, w, h))
        pygame.draw.line(self.screen, C["border"], (x, y), (x, y+h))
        cy = y + 12

        self.txt(self.screen, "MARCO ABA", (x+12, cy), C["accent"], 12, bold=True)
        cy += 24
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10

        # 4-term contingency diagram
        self.txt(self.screen, "Contingencia de 4 Terminos", (x+12, cy), C["text"], 11)
        cy += 18
        terms = [
            ("MO", "Operacion Motivadora", C["eo"]),
            ("SD", "Estimulo Discriminativo", C["accent"]),
            ("R",  "Respuesta (conducta)", C["yellow"]),
            ("C",  "Consecuencia", C["green"]),
        ]
        for abbr, name, col in terms:
            pygame.draw.rect(self.screen, col, (x+12, cy, 28, 18))
            self.txt(self.screen, abbr, (x+26, cy+1), C["bg"], 10, bold=True, anchor="midtop")
            self.txt(self.screen, name, (x+46, cy+2), C["text_dim"], 10)
            cy += 22
            if abbr != "C":
                pygame.draw.line(self.screen, C["border"],
                                 (x+25, cy-2), (x+25, cy+2))
                cy += 4
        cy += 10

        # Accuracy panel
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10
        self.txt(self.screen, "Precision del Motor", (x+12, cy), C["text"], 11, bold=True)
        cy += 20

        accuracy = self.engine.get_global_accuracy()
        sports_acc = self.engine.sports.get_accuracy()

        domains_acc = {
            "sports":  sports_acc.get("accuracy", 0.0),
            "market":  accuracy.get("market", {}).get("accuracy", 0.0),
            "masses":  accuracy.get("masses", {}).get("accuracy", 0.0),
            "betting": accuracy.get("betting", {}).get("accuracy", 0.0),
        }
        for dom, acc in domains_acc.items():
            col = DOMAIN_COLORS.get(dom, C["text"])
            self.txt(self.screen, dom.capitalize(), (x+12, cy), col, 11)
            bar_w = w - 110
            pygame.draw.rect(self.screen, C["border"], (x+70, cy, bar_w, 10))
            if acc > 0:
                pygame.draw.rect(self.screen, col,
                                 (x+70, cy, int(bar_w * acc), 10))
            self.txt(self.screen, f"{acc:.0%}", (x+70+bar_w+4, cy-1), col, 10)
            cy += 18
        cy += 6

        # Matching Law explanation
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 10
        self.txt(self.screen, "Ley de Igualacion", (x+12, cy), C["text"], 11, bold=True)
        cy += 18
        lines = [
            "B1/(B1+B2) = R1/(R1+R2)",
            "La masa distribuye su conducta",
            "(dinero, atencion, apuestas)",
            "en proporcion a la tasa de",
            "reforzamiento de cada opcion.",
        ]
        for line in lines:
            col = C["accent"] if "B1" in line else C["text_dim"]
            size = 12 if "B1" in line else 10
            self.txt(self.screen, line, (x+12, cy), col, size)
            cy += 13
        cy += 8

        # VR Warning
        if self.domain == "betting":
            pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
            cy += 10
            self.txt(self.screen, "ALERTA VR SCHEDULE", (x+12, cy), C["red"], 11, bold=True)
            cy += 18
            vr_lines = [
                "Las apuestas operan en VR:",
                "- Mayor tasa de conducta",
                "- Alta resistencia extincion",
                "- Chasing losses = escape",
                "Margen casa = AO sistematica",
            ]
            for line in vr_lines:
                self.txt(self.screen, line, (x+12, cy), C["red"], 10)
                cy += 13

        # Controls
        cy = y + h - 100
        pygame.draw.line(self.screen, C["border"], (x+10, cy), (x+w-10, cy))
        cy += 8
        self.txt(self.screen, "Controles", (x+12, cy), C["text_dim"], 10, bold=True)
        cy += 14
        for ctrl in ["[1-4] Cambiar dominio", "[R] Refrescar", "[Q] Salir"]:
            self.txt(self.screen, ctrl, (x+12, cy), C["text_dim"], 10)
            cy += 12

    def _render_bottom_bar(self, x, y, w, h):
        pygame.draw.rect(self.screen, C["header"], (x, y, w, h))
        pygame.draw.line(self.screen, C["border"], (x, y), (x+w, y))

        cx = x + 12
        self.txt(self.screen, "HISTORIAL:", (cx, y + h//2),
                 C["text_dim"], 11, anchor="midleft")
        cx += 70

        recents = self.engine.get_recent_predictions(6)
        for pred in recents:
            correct = pred.get("was_correct")
            col = (C["green"] if correct == True else
                   C["red"] if correct == False else C["text_dim"])
            label = f"{pred['domain'][0].upper()}:{pred['prediction'][:10]}"
            r = self.txt(self.screen, label, (cx, y + h//2), col, 10, anchor="midleft")
            cx = r.right + 14
            if cx > w - 100:
                break

        # Timestamp
        self.txt(self.screen, time.strftime("%H:%M:%S"),
                 (w - 10, y + h//2), C["text_dim"], 11, anchor="midright")
