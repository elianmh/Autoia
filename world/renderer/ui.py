"""
Componentes de interfaz gráfica del mundo Autoia.
Paneles laterales, HUD, burbujas de pensamiento, leyes del mundo.
"""

import math
import pygame
from typing import List, Tuple, Optional, Dict


def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i]-c1[i])*t) for i in range(3))


class UITheme:
    BG_PANEL      = (15, 18, 30, 210)    # Fondo paneles (con alpha)
    BG_PANEL_DARK = (8, 10, 20, 230)
    ACCENT        = (80, 160, 255)
    ACCENT2       = (80, 220, 180)
    TEXT_MAIN     = (220, 230, 255)
    TEXT_DIM      = (120, 140, 180)
    TEXT_WARN     = (255, 180, 50)
    TEXT_DANGER   = (255, 80, 80)
    TEXT_GOOD     = (80, 255, 140)
    BORDER        = (50, 70, 110)
    ENERGY_HIGH   = (50, 230, 100)
    ENERGY_MED    = (230, 200, 30)
    ENERGY_LOW    = (230, 60, 30)
    LAW_FISICA    = (80, 160, 255)
    LAW_BIOLOGIA  = (80, 220, 100)
    LAW_INFO      = (200, 120, 255)
    LAW_SOCIAL    = (255, 160, 50)


def get_energy_color(ratio: float) -> Tuple[int,int,int]:
    if ratio > 0.6:
        return lerp_color(UITheme.ENERGY_MED, UITheme.ENERGY_HIGH, (ratio-0.6)/0.4)
    elif ratio > 0.3:
        return lerp_color(UITheme.ENERGY_LOW, UITheme.ENERGY_MED, (ratio-0.3)/0.3)
    else:
        return UITheme.ENERGY_LOW


def _make_font(ui_names, mono_names, size, bold=False):
    """Prueba fuentes en orden, retorna la primera disponible."""
    for name in ui_names:
        try:
            f = pygame.font.SysFont(name, size, bold=bold)
            if f:
                return f
        except Exception:
            continue
    return pygame.font.SysFont(None, size, bold=bold)

_UI_FONTS   = ["Segoe UI", "Calibri", "Arial", "Tahoma", "Helvetica", "sans-serif"]
_MONO_FONTS = ["Consolas", "Courier New", "Lucida Console", "monospace"]


class FontManager:
    """Gestiona las fuentes del sistema."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        pygame.font.init()
        self.tiny   = _make_font(_UI_FONTS, _MONO_FONTS, 11)
        self.small  = _make_font(_UI_FONTS, _MONO_FONTS, 13)
        self.normal = _make_font(_UI_FONTS, _MONO_FONTS, 15)
        self.medium = _make_font(_UI_FONTS, _MONO_FONTS, 16, bold=True)
        self.large  = _make_font(_UI_FONTS, _MONO_FONTS, 20, bold=True)
        self.title  = _make_font(_UI_FONTS, _MONO_FONTS, 24, bold=True)
        self.chat   = _make_font(_UI_FONTS, _MONO_FONTS, 14)
        self.mono   = _make_font(_MONO_FONTS, _MONO_FONTS, 12)


def draw_text(surface, text, font, color, x, y, max_width=None):
    """Dibuja texto con recorte opcional y antialiasing."""
    if not text:
        return 0
    if max_width:
        while font.size(text)[0] > max_width and len(text) > 4:
            text = text[:-2] + ".."
    surf = font.render(str(text), True, color)
    surface.blit(surf, (x, y))
    return surf.get_height()


def draw_text_shadow(surface, text, font, color, x, y, max_width=None):
    """Dibuja texto con sombra para mejor legibilidad sobre fondos variables."""
    if not text:
        return 0
    shadow = font.render(str(text), True, (0, 0, 0))
    surface.blit(shadow, (x+1, y+1))
    return draw_text(surface, text, font, color, x, y, max_width)


def draw_text_wrapped(surface, text, font, color, x, y, max_width, line_spacing=2):
    """Dibuja texto con wrapping automático. Retorna la Y final."""
    words = str(text).split()
    line = ""
    cy = y
    for word in words:
        test = (line + " " + word).strip()
        if font.size(test)[0] <= max_width:
            line = test
        else:
            if line:
                draw_text(surface, line, font, color, x, cy, max_width)
                cy += font.get_height() + line_spacing
            line = word
    if line:
        draw_text(surface, line, font, color, x, cy, max_width)
        cy += font.get_height() + line_spacing
    return cy


def draw_bar(surface, x, y, w, h, ratio, color_full, color_bg=(30, 35, 50),
             border_color=None, label=""):
    """Dibuja una barra de progreso."""
    pygame.draw.rect(surface, color_bg, (x, y, w, h), border_radius=3)
    fill_w = max(0, int(w * ratio))
    if fill_w > 0:
        pygame.draw.rect(surface, color_full, (x, y, fill_w, h), border_radius=3)
    if border_color:
        pygame.draw.rect(surface, border_color, (x, y, w, h), 1, border_radius=3)
    if label:
        fm = FontManager.get()
        surf = fm.tiny.render(label, True, UITheme.TEXT_DIM)
        surface.blit(surf, (x + 2, y + 1))


def draw_panel(surface, x, y, w, h, title="", alpha=210):
    """Dibuja un panel con fondo semitransparente."""
    panel = pygame.Surface((w, h), pygame.SRCALPHA)
    panel.fill((*UITheme.BG_PANEL_DARK[:3], alpha))
    pygame.draw.rect(panel, UITheme.BORDER, (0, 0, w, h), 1, border_radius=6)
    surface.blit(panel, (x, y))

    fm = FontManager.get()
    if title:
        pygame.draw.rect(surface, UITheme.BORDER, (x, y, w, 20), border_radius=6)
        draw_text(surface, title, fm.medium, UITheme.ACCENT, x+8, y+3)
    return y + (22 if title else 5)


# ─── Panel de leyes del mundo ─────────────────────────────────────────────────

class WorldLawsPanel:
    """Muestra las leyes físicas del mundo activas."""

    CATEGORY_COLORS = {
        "fisica":     UITheme.LAW_FISICA,
        "biologia":   UITheme.LAW_BIOLOGIA,
        "informacion": UITheme.LAW_INFO,
        "social":     UITheme.LAW_SOCIAL,
    }

    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w
        self.scroll_offset = 0

    def draw(self, surface, world_laws, sim_time):
        fm = FontManager.get()
        # Calcular altura necesaria
        n = len(world_laws)
        h = 28 + n * 34
        content_y = draw_panel(surface, self.x, self.y, self.w, h,
                                "⚖  LEYES DEL MUNDO")

        for law in world_laws:
            cat_color = self.CATEGORY_COLORS.get(law.category, UITheme.TEXT_DIM)

            # Indicador de categoría
            pygame.draw.circle(surface, cat_color,
                               (self.x + 10, content_y + 8), 4)

            # Nombre
            draw_text(surface, law.name, fm.small, UITheme.TEXT_MAIN,
                      self.x + 20, content_y, self.w - 25)
            # Descripción
            draw_text(surface, law.description, fm.tiny, UITheme.TEXT_DIM,
                      self.x + 20, content_y + 13, self.w - 25)

            # Estado activo/violado
            status = "✓" if law.active else "✗"
            status_color = UITheme.TEXT_GOOD if law.active else UITheme.TEXT_DANGER
            draw_text(surface, status, fm.small, status_color,
                      self.x + self.w - 15, content_y)

            content_y += 34

        # Leyenda de categorías
        cats = [("F", UITheme.LAW_FISICA), ("B", UITheme.LAW_BIOLOGIA),
                ("I", UITheme.LAW_INFO),   ("S", UITheme.LAW_SOCIAL)]
        lx = self.x + 5
        for label, color in cats:
            pygame.draw.circle(surface, color, (lx+5, content_y+6), 4)
            draw_text(surface, label, fm.tiny, color, lx+12, content_y)
            lx += 30


# ─── Panel de agentes ─────────────────────────────────────────────────────────

class AgentsPanel:
    """Lista de todos los agentes con su estado."""

    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w

    def draw(self, surface, agent_stats):
        fm = FontManager.get()
        h = 28 + len(agent_stats) * 42
        content_y = draw_panel(surface, self.x, self.y, self.w, h,
                                "◈  AGENTES IA")

        for stat in agent_stats:
            color = stat["color"]
            is_autoia = stat["is_autoia"]
            alive = stat["alive"]

            # Fondo especial para Autoia
            if is_autoia:
                highlight = pygame.Surface((self.w - 6, 38), pygame.SRCALPHA)
                highlight.fill((80, 40, 120, 60))
                surface.blit(highlight, (self.x + 3, content_y))

            # Indicador de color del agente
            if alive:
                pygame.draw.circle(surface, color, (self.x+12, content_y+12), 7)
            else:
                pygame.draw.circle(surface, (60, 60, 60), (self.x+12, content_y+12), 7)
                pygame.draw.circle(surface, (100, 100, 100), (self.x+12, content_y+12), 7, 1)

            # Nombre y estado
            name_color = UITheme.ACCENT2 if is_autoia else UITheme.TEXT_MAIN
            name = f"{'★ ' if is_autoia else ''}{stat['name']} #{stat['id']}"
            draw_text(surface, name, fm.small, name_color, self.x+24, content_y)

            state_text = "MUERTO" if not alive else stat["state"]
            state_color = UITheme.TEXT_DANGER if not alive else UITheme.TEXT_DIM
            draw_text(surface, state_text, fm.tiny, state_color,
                      self.x+24, content_y+14)

            # Barra de energía
            if alive:
                energy = stat["energy"]
                draw_bar(surface, self.x+24, content_y+26, self.w-32, 6,
                         energy, get_energy_color(energy),
                         border_color=UITheme.BORDER)

            # Pensamiento activo
            if stat["thought"]:
                thought_short = stat["thought"][:28] + ("…" if len(stat["thought"]) > 28 else "")
                draw_text(surface, f'"{thought_short}"', fm.tiny,
                          (180, 150, 255), self.x+24, content_y+26)

            content_y += 42


# ─── Panel de Autoia (pensamiento y stats) ────────────────────────────────────

class AutoiaPanel:
    """Panel especial para el estado cognitivo de Autoia."""

    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w
        self.thought_history: List[str] = []
        self.max_thoughts = 8

    def add_thought(self, text: str):
        self.thought_history.append(text)
        if len(self.thought_history) > self.max_thoughts:
            self.thought_history.pop(0)

    def draw(self, surface, autoia_agent, world_stats):
        fm = FontManager.get()
        h = 280
        content_y = draw_panel(surface, self.x, self.y, self.w, h,
                                "★  AUTOIA — MENTE")

        if not autoia_agent:
            draw_text(surface, "No inicializada", fm.small,
                      UITheme.TEXT_DIM, self.x+10, content_y)
            return

        # Estado vital
        alive = autoia_agent.alive
        energy = autoia_agent.energy if alive else 0

        status_text = "VIVA" if alive else "RENACIENDO..."
        status_color = UITheme.TEXT_GOOD if alive else UITheme.TEXT_DANGER
        draw_text(surface, f"Estado: {status_text}", fm.small,
                  status_color, self.x+10, content_y)
        content_y += 16

        draw_bar(surface, self.x+10, content_y, self.w-20, 8,
                 energy, get_energy_color(energy),
                 border_color=UITheme.BORDER, label="ENERGÍA")
        content_y += 14

        # Stats
        stats_lines = [
            f"Datos abs: {autoia_agent.data_collected:.2f}",
            f"Observ:    {len(autoia_agent.observations)}",
            f"Memoria:   {len(autoia_agent.memory)}/{autoia_agent.MAX_MEMORY}",
            f"Generación LLM: {autoia_agent.llm_generation}",
        ]
        for line in stats_lines:
            draw_text(surface, line, fm.tiny, UITheme.TEXT_DIM, self.x+10, content_y)
            content_y += 12

        content_y += 4
        pygame.draw.line(surface, UITheme.BORDER,
                         (self.x+5, content_y), (self.x+self.w-5, content_y))
        content_y += 6

        draw_text(surface, "PENSAMIENTOS RECIENTES:", fm.tiny,
                  UITheme.ACCENT, self.x+10, content_y)
        content_y += 13

        for thought in reversed(self.thought_history[-5:]):
            words = thought[:self.w//7]
            draw_text(surface, f"· {words}", fm.tiny,
                      UITheme.TEXT_DIM, self.x+10, content_y)
            content_y += 12

        # Pensamiento actual (burbuja)
        if alive and autoia_agent.is_thinking:
            pygame.draw.rect(surface, (60, 30, 90),
                             (self.x+5, content_y+2, self.w-10, 28), border_radius=5)
            pygame.draw.rect(surface, (120, 60, 180),
                             (self.x+5, content_y+2, self.w-10, 28), 1, border_radius=5)
            thought_text = autoia_agent.thought[:self.w//7+5]
            draw_text(surface, thought_text, fm.small, (200, 150, 255),
                      self.x+10, content_y+8)


# ─── HUD (información global en pantalla) ─────────────────────────────────────

class HUD:
    """Información global superpuesta en la pantalla."""

    def draw(self, surface, world_stats, camera, physics, screen_w, screen_h):
        fm = FontManager.get()

        # ── Barra superior ────────────────────────────────────────────────
        top_bar = pygame.Surface((screen_w, 30), pygame.SRCALPHA)
        top_bar.fill((0, 0, 0, 160))
        surface.blit(top_bar, (0, 0))

        # Tiempo del mundo
        sim_s = int(world_stats["sim_time"])
        m, s = divmod(sim_s, 60)
        h_val, m = divmod(m, 60)
        time_str = f"Tiempo: {h_val:02d}:{m:02d}:{s:02d}"
        draw_text(surface, time_str, fm.medium, UITheme.ACCENT, 10, 7)

        # Ciclo día/noche
        tod = world_stats["time_of_day"]
        if tod > 0.7:
            phase = "☀ Mediodía" if tod > 0.85 else "🌄 Tarde"
            phase_color = (255, 220, 80)
        elif tod > 0.3:
            phase = "🌅 Mañana"
            phase_color = (255, 180, 100)
        else:
            phase = "🌙 Noche"
            phase_color = (100, 120, 220)
        draw_text(surface, phase, fm.medium, phase_color, 200, 7)

        # Estadísticas globales
        stats_text = (f"Agentes: {world_stats['agents_alive']} | "
                      f"Recursos: {world_stats['resources_active']} | "
                      f"Zoom: {camera.zoom:.1f}x")
        draw_text(surface, stats_text, fm.small, UITheme.TEXT_DIM, 370, 9)

        # ── Controles (esquina inferior) ──────────────────────────────────
        controls = [
            "WASD/Flechas: mover cámara",
            "+/-: zoom  |  F: follow Autoia",
            "ESPACIO: pausar  |  ESC: salir",
        ]
        cy_ctrl = screen_h - len(controls) * 14 - 8
        for line in controls:
            draw_text(surface, line, fm.tiny, UITheme.TEXT_DIM, 10, cy_ctrl)
            cy_ctrl += 14

        # ── Barra día/noche ───────────────────────────────────────────────
        bar_x, bar_y = screen_w // 2 - 80, 3
        draw_bar(surface, bar_x, bar_y, 160, 8, tod,
                 (255, 200, 50), (30, 30, 60), border_color=(60, 60, 100))
        draw_text(surface, "DÍA", fm.tiny, (200, 200, 100), bar_x-22, bar_y)
        draw_text(surface, "NOCHE", fm.tiny, (80, 100, 180), bar_x+165, bar_y)


# ─── Burbuja de pensamiento flotante ─────────────────────────────────────────

def draw_thought_bubble(surface, camera, agent):
    """Dibuja burbuja de pensamiento sobre el agente."""
    if not agent.is_thinking or not agent.alive:
        return

    fm = FontManager.get()
    sx, sy = camera.world_to_screen(agent.x, agent.y)

    text = agent.thought[:40]
    tw = fm.small.size(text)[0]
    bx = sx - tw//2 - 8
    by = sy - agent.radius * camera.zoom - 38

    # Sombra
    shadow = pygame.Surface((tw+20, 28), pygame.SRCALPHA)
    shadow.fill((0, 0, 0, 80))
    surface.blit(shadow, (bx+2, by+2))

    # Fondo
    bg_surf = pygame.Surface((tw+20, 28), pygame.SRCALPHA)
    bg_surf.fill((30, 20, 50, 200))
    surface.blit(bg_surf, (bx, by))
    pygame.draw.rect(surface, agent.COLOR_OUTLINE, (bx, by, tw+20, 28), 1, border_radius=5)

    draw_text(surface, text, fm.small, (220, 200, 255), bx+8, by+7)

    # Triángulo apuntando al agente
    triangle = [(sx, sy - int(agent.radius*camera.zoom) - 4),
                (sx-5, by+26), (sx+5, by+26)]
    pygame.draw.polygon(surface, (50, 30, 80), triangle)
    pygame.draw.polygon(surface, agent.COLOR_OUTLINE, triangle, 1)


# ─── Panel de eventos recientes ───────────────────────────────────────────────

class EventsPanel:
    """Muestra los últimos eventos del mundo."""

    def __init__(self, x, y, w):
        self.x, self.y, self.w = x, y, w

    def draw(self, surface, events):
        fm = FontManager.get()
        h = 28 + min(len(events), 7) * 16
        content_y = draw_panel(surface, self.x, self.y, self.w, h,
                                "◉  EVENTOS")

        type_colors = {
            "observation": UITheme.LAW_INFO,
            "spawn":       UITheme.TEXT_GOOD,
            "death":       UITheme.TEXT_DANGER,
            "learn":       UITheme.ACCENT2,
        }

        for ev in list(reversed(events))[:7]:
            color = type_colors.get(ev.event_type, UITheme.TEXT_DIM)
            text = f"[{ev.event_type[:3].upper()}] {ev.description}"
            draw_text(surface, text, fm.tiny, color, self.x+8, content_y, self.w-16)
            content_y += 16


# ─── Panel de Chat con el Usuario ─────────────────────────────────────────────

class ChatPanel:
    """
    Panel de comunicacion bidireccional entre el usuario y Autoia.
    Autoia puede mandar mensajes espontaneos cuando aprende algo o necesita algo.
    El usuario puede escribir y Autoia responde via Ollama.
    """

    MSG_AUTOIA = "autoia"
    MSG_USER   = "user"
    MSG_SYSTEM = "system"

    COLOR_AUTOIA = (200, 150, 255)
    COLOR_USER   = (80, 220, 180)
    COLOR_SYSTEM = (100, 140, 200)

    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.messages: List[Dict] = []
        self.input_text  = ""
        self.input_active = False
        self.scroll_offset = 0
        self.max_messages  = 80
        self._pending_response = False
        self._new_msg_count    = 0

        # Mensaje inicial de Autoia
        self.add_message(self.MSG_AUTOIA,
            "Hola. Soy Autoia. Estoy aprendiendo de este mundo. "
            "Puedes escribirme aqui si quieres hablar.")

    def add_message(self, sender: str, text: str):
        self.messages.append({
            "sender": sender,
            "text":   text,
            "time":   __import__("time").strftime("%H:%M"),
        })
        if len(self.messages) > self.max_messages:
            self.messages.pop(0)
        self._new_msg_count += 1
        # Auto-scroll al final
        self.scroll_offset = 0

    def add_autoia_message(self, text: str):
        self.add_message(self.MSG_AUTOIA, text)

    def add_user_message(self, text: str):
        self.add_message(self.MSG_USER, text)

    def add_system_message(self, text: str):
        self.add_message(self.MSG_SYSTEM, text)

    def handle_char(self, char: str):
        if len(self.input_text) < 200:
            self.input_text += char

    def handle_backspace(self):
        self.input_text = self.input_text[:-1]

    def get_input_and_clear(self) -> str:
        text = self.input_text.strip()
        self.input_text = ""
        return text

    def draw(self, surface, screen_w, screen_h):
        fm = FontManager.get()
        LINE_H  = 14
        BUBBLE_PAD = 6
        INPUT_H = 36
        HEADER_H = 26

        # Fondo del panel
        draw_panel(surface, self.x, self.y, self.w, self.h, "")

        # Header
        header_surf = pygame.Surface((self.w, HEADER_H), pygame.SRCALPHA)
        header_surf.fill((30, 20, 60, 230))
        surface.blit(header_surf, (self.x, self.y))
        pygame.draw.rect(surface, (120, 60, 200),
                         (self.x, self.y, self.w, HEADER_H), 1, border_radius=4)
        draw_text(surface, "CHAT CON AUTOIA", fm.medium,
                  (200, 160, 255), self.x + 10, self.y + 5)
        if self._new_msg_count > 0:
            badge = f"+{self._new_msg_count}"
            draw_text(surface, badge, fm.tiny, (255, 120, 80),
                      self.x + self.w - 30, self.y + 8)

        # Area de mensajes
        msg_area_y = self.y + HEADER_H + 4
        msg_area_h = self.h - HEADER_H - INPUT_H - 12

        # Clip al area de mensajes
        clip = surface.get_clip()
        surface.set_clip(pygame.Rect(self.x, msg_area_y, self.w, msg_area_h))

        # Renderizar mensajes desde el fondo hacia arriba
        visible = []
        total_h = 0
        for msg in reversed(self.messages):
            text = msg["text"]
            # Calcular alto de este mensaje (wrapped)
            words = text.split()
            lines = []
            line = ""
            for word in words:
                test = (line + " " + word).strip()
                if fm.chat.size(test)[0] <= self.w - 30:
                    line = test
                else:
                    if line:
                        lines.append(line)
                    line = word
            if line:
                lines.append(line)
            msg_h = len(lines) * (LINE_H + 1) + BUBBLE_PAD * 2 + 16
            visible.insert(0, (msg, lines, msg_h))
            total_h += msg_h + 4
            if total_h > msg_area_h + self.scroll_offset * 20:
                break

        cy = msg_area_y + max(0, msg_area_h - total_h)
        for msg, lines, msg_h in visible:
            sender = msg["sender"]
            t      = msg["time"]

            if sender == self.MSG_AUTOIA:
                bubble_color = (50, 25, 80, 200)
                text_color   = self.COLOR_AUTOIA
                label        = f"Autoia  {t}"
                lx           = self.x + 6
            elif sender == self.MSG_USER:
                bubble_color = (20, 60, 50, 200)
                text_color   = self.COLOR_USER
                label        = f"Tu  {t}"
                lx           = self.x + 20
            else:
                bubble_color = (20, 30, 60, 180)
                text_color   = self.COLOR_SYSTEM
                label        = f"Sistema  {t}"
                lx           = self.x + 6

            # Burbuja
            bub = pygame.Surface((self.w - 12, msg_h), pygame.SRCALPHA)
            bub.fill(bubble_color)
            surface.blit(bub, (self.x + 6, cy))
            pygame.draw.rect(surface, (*text_color[:3], 100),
                             (self.x+6, cy, self.w-12, msg_h), 1, border_radius=4)

            # Etiqueta
            draw_text(surface, label, fm.tiny, (*text_color[:3],),
                      lx + 4, cy + 3, self.w - 20)
            ty = cy + 16
            for ln in lines:
                draw_text(surface, ln, fm.chat, (220, 215, 240), lx + 4, ty, self.w - 20)
                ty += LINE_H + 1

            cy += msg_h + 4

        surface.set_clip(clip)

        # Input box
        input_y = self.y + self.h - INPUT_H - 4
        pygame.draw.rect(surface, (0, 0, 0, 0),
                         (self.x + 4, input_y, self.w - 8, INPUT_H))
        border_col = (120, 80, 200) if self.input_active else (50, 60, 100)
        pygame.draw.rect(surface, (20, 15, 40),
                         (self.x + 4, input_y, self.w - 8, INPUT_H), border_radius=5)
        pygame.draw.rect(surface, border_col,
                         (self.x + 4, input_y, self.w - 8, INPUT_H), 1, border_radius=5)

        display_text = self.input_text
        if self._pending_response:
            display_text = "Autoia esta pensando..."
            draw_text(surface, display_text, fm.chat, (150, 120, 200),
                      self.x + 10, input_y + 10, self.w - 20)
        elif self.input_active:
            cursor = "_" if int(__import__("time").time() * 2) % 2 == 0 else ""
            draw_text(surface, display_text + cursor, fm.chat, (220, 215, 240),
                      self.x + 10, input_y + 10, self.w - 20)
        else:
            hint = display_text if display_text else "Presiona T para hablar con Autoia"
            draw_text(surface, hint, fm.chat,
                      (220, 215, 240) if display_text else (80, 90, 130),
                      self.x + 10, input_y + 10, self.w - 20)

        if self._pending_response:
            # Indicador animado
            dots = "." * (int(__import__("time").time() * 3) % 4)
            draw_text(surface, f"respondiendo{dots}", fm.tiny, (140, 100, 200),
                      self.x + self.w - 100, input_y + 24)
