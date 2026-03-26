"""
Aplicación pygame principal del mundo visual de Autoia.
Renderiza el mundo, los agentes y la interfaz gráfica completa.
"""

import sys
import math
import random
import pygame
import pygame.gfxdraw
import logging
from typing import Optional

from world.world_sim import WorldSimulation
from world.physics import TERRAIN_PROPS, TerrainType
from world.renderer.camera import Camera
from world.renderer.ui import (
    FontManager, UITheme, HUD, WorldLawsPanel,
    AgentsPanel, AutoiaPanel, EventsPanel,
    draw_thought_bubble, draw_text, draw_panel, lerp_color
)

logger = logging.getLogger("autoia.renderer")


# ─── Constantes de pantalla ────────────────────────────────────────────────────

SCREEN_W     = 1400
SCREEN_H     = 820
VIEWPORT_W   = 950        # Área del mundo
VIEWPORT_H   = SCREEN_H
PANEL_LEFT_W = 0          # Sin panel izquierdo
PANEL_RIGHT_W = SCREEN_W - VIEWPORT_W
TARGET_FPS   = 60


class WorldApp:
    """
    Aplicación principal del mundo visual.
    Gestiona la ventana, el loop de juego, el renderizado y los inputs.
    """

    def __init__(self, world: WorldSimulation, autoia_agent=None):
        pygame.init()
        pygame.display.set_caption("Autoia — Mundo de IAs con Leyes Físicas")

        self.screen = pygame.display.set_mode(
            (SCREEN_W, SCREEN_H), pygame.RESIZABLE
        )
        self.clock      = pygame.time.Clock()
        self.running    = True
        self.paused     = False
        self.sim_speed  = 1.0   # Multiplicador de velocidad

        self.world      = world
        self.autoia     = autoia_agent
        self.fonts      = FontManager.get()

        # Cámara
        self.camera = Camera(VIEWPORT_W, VIEWPORT_H,
                             world.pixel_w, world.pixel_h)
        if autoia_agent:
            self.camera.follow(autoia_agent)
        else:
            self.camera.cx = world.pixel_w / 2
            self.camera.cy = world.pixel_h / 2

        self.follow_autoia = (autoia_agent is not None)

        # Superficie del viewport (área del mundo)
        self.viewport_surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        # Superficie del terreno (precalculada, actualiza solo si cambia)
        self.terrain_surf  = None
        self._build_terrain_surface()

        # UI Components
        px = VIEWPORT_W + 8
        self.hud            = HUD()
        self.laws_panel     = WorldLawsPanel(px, 35, PANEL_RIGHT_W - 12)
        self.autoia_panel   = AutoiaPanel(px, 35, PANEL_RIGHT_W - 12)
        self.agents_panel   = AgentsPanel(px, 35, PANEL_RIGHT_W - 12)
        self.events_panel   = EventsPanel(px, 35, PANEL_RIGHT_W - 12)

        # Estado de qué panel mostrar a la derecha
        self.right_panel_mode = "autoia"  # "autoia", "agents", "laws", "events"
        self.panel_tabs       = ["autoia", "agents", "laws", "events"]

        # Control de teclado
        self.keys_held = set()

        # Efectos de luz dinámica
        self.light_surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)

        logger.info("WorldApp iniciada")

    def _build_terrain_surface(self):
        """Pre-renderiza el terreno completo (se actualiza con scroll/zoom)."""
        ts = self.world.terrain.tile_size
        gw = self.world.terrain.grid_w
        gh = self.world.terrain.grid_h

        full_w = gw * ts
        full_h = gh * ts
        self.terrain_surf = pygame.Surface((full_w, full_h))

        for gy in range(gh):
            for gx in range(gw):
                color = self.world.terrain.get_tile_color(gx, gy)
                rect = (gx*ts, gy*ts, ts, ts)
                pygame.draw.rect(self.terrain_surf, color, rect)

                # Borde sutil entre tiles
                terrain = self.world.terrain.grid[gy][gx]
                if terrain not in (TerrainType.WATER, TerrainType.FIRE):
                    darker = tuple(max(0, c-20) for c in color)
                    pygame.draw.rect(self.terrain_surf, darker, rect, 1)

        # Dibujar marcadores de nodos de datos
        for gy in range(gh):
            for gx in range(gw):
                if self.world.terrain.grid[gy][gx] == TerrainType.DATA:
                    cx = gx*ts + ts//2
                    cy = gy*ts + ts//2
                    pygame.draw.circle(self.terrain_surf, (120, 255, 240), (cx, cy), 3)

    def run(self):
        """Loop principal."""
        while self.running:
            dt = self.clock.tick(TARGET_FPS) / 1000.0
            dt = min(dt, 0.05)

            self._handle_events()
            self._handle_keys(dt)

            if not self.paused:
                self.world.step(dt * self.sim_speed)

            self.camera.update(dt)
            self._render()
            pygame.display.flip()

        pygame.quit()

    # ─── Input ────────────────────────────────────────────────────────────────

    def _handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            elif event.type == pygame.KEYDOWN:
                self.keys_held.add(event.key)
                self._handle_keydown(event.key)

            elif event.type == pygame.KEYUP:
                self.keys_held.discard(event.key)

            elif event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    self.camera.zoom_in()
                else:
                    self.camera.zoom_out()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    self._handle_click(event.pos)

            elif event.type == pygame.VIDEORESIZE:
                pass

    def _handle_keydown(self, key):
        if key == pygame.K_ESCAPE:
            self.running = False
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key == pygame.K_f:
            self.follow_autoia = not self.follow_autoia
            if self.follow_autoia and self.autoia:
                self.camera.follow(self.autoia)
            else:
                self.camera.follow_target = None
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.camera.zoom_in()
        elif key in (pygame.K_MINUS,):
            self.camera.zoom_out()
        elif key == pygame.K_TAB:
            # Rotar panel derecho
            idx = self.panel_tabs.index(self.right_panel_mode)
            self.right_panel_mode = self.panel_tabs[(idx+1) % len(self.panel_tabs)]
        elif key == pygame.K_1:
            self.right_panel_mode = "autoia"
        elif key == pygame.K_2:
            self.right_panel_mode = "agents"
        elif key == pygame.K_3:
            self.right_panel_mode = "laws"
        elif key == pygame.K_4:
            self.right_panel_mode = "events"
        elif key in (pygame.K_COMMA, pygame.K_LESS):
            self.sim_speed = max(0.25, self.sim_speed / 2)
        elif key in (pygame.K_PERIOD, pygame.K_GREATER):
            self.sim_speed = min(4.0, self.sim_speed * 2)

    def _handle_keys(self, dt: float):
        """Scroll de cámara con teclas."""
        speed = 200 / self.camera.zoom
        dx, dy = 0, 0
        if pygame.K_LEFT  in self.keys_held or pygame.K_a in self.keys_held: dx -= speed * dt
        if pygame.K_RIGHT in self.keys_held or pygame.K_d in self.keys_held: dx += speed * dt
        if pygame.K_UP    in self.keys_held or pygame.K_w in self.keys_held: dy -= speed * dt
        if pygame.K_DOWN  in self.keys_held or pygame.K_s in self.keys_held: dy += speed * dt
        if dx or dy:
            self.camera.follow_target = None
            self.follow_autoia = False
            self.camera.scroll(dx * self.camera.zoom, dy * self.camera.zoom)

    def _handle_click(self, pos):
        """Click en un agente lo selecciona."""
        mx, my = pos
        if mx < VIEWPORT_W:
            wx, wy = self.camera.screen_to_world(mx, my)
            for agent in self.world.agents:
                if agent.alive:
                    dx = agent.x - wx
                    dy = agent.y - wy
                    if math.sqrt(dx*dx+dy*dy) < agent.radius + 5:
                        logger.info(f"Seleccionado: {agent.AGENT_NAME} #{agent.agent_id}")
                        if hasattr(agent, 'is_autoia'):
                            self.right_panel_mode = "autoia"
                        break

    # ─── Renderizado ──────────────────────────────────────────────────────────

    def _render(self):
        self.screen.fill((10, 12, 22))
        self._render_viewport()
        self._render_right_panel()
        self._render_panel_tabs()
        self.hud.draw(self.screen, self.world.get_world_state_summary(),
                      self.camera, self.world.physics, SCREEN_W, SCREEN_H)
        self._render_speed_indicator()

    def _render_viewport(self):
        """Renderiza el mundo en el viewport."""
        # Fondo
        sky = self.world.physics.sky_color
        self.viewport_surf.fill(sky)

        # Terreno escalado y desplazado
        zoom = self.camera.zoom
        # Calcular qué parte del terreno es visible
        wx1, wy1, wx2, wy2 = self.camera.get_visible_rect()
        crop_x = max(0, int(wx1))
        crop_y = max(0, int(wy1))
        crop_w = min(self.world.pixel_w, int(wx2)) - crop_x
        crop_h = min(self.world.pixel_h, int(wy2)) - crop_y

        if crop_w > 0 and crop_h > 0:
            terrain_crop = self.terrain_surf.subsurface(
                pygame.Rect(crop_x, crop_y,
                            min(crop_w, self.world.pixel_w - crop_x),
                            min(crop_h, self.world.pixel_h - crop_y))
            )
            scaled_w = int(terrain_crop.get_width() * zoom)
            scaled_h = int(terrain_crop.get_height() * zoom)
            if scaled_w > 0 and scaled_h > 0:
                terrain_scaled = pygame.transform.scale(terrain_crop, (scaled_w, scaled_h))
                # Posición en viewport
                sx, sy = self.camera.world_to_screen(crop_x, crop_y)
                self.viewport_surf.blit(terrain_scaled, (sx, sy))

        # Efecto de oscuridad nocturna
        ambient = self.world.physics.ambient_light
        if ambient < 0.95:
            darkness = int((1 - ambient) * 200)
            dark_overlay = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
            dark_overlay.fill((0, 5, 20, darkness))
            self.viewport_surf.blit(dark_overlay, (0, 0))

        # Partículas
        self._render_particles()

        # Recursos
        self._render_resources()

        # Rastros de agentes
        self._render_trails()

        # Agentes
        self._render_agents()

        # Burbujas de pensamiento
        self._render_thought_bubbles()

        # Visión de Autoia
        if self.autoia and self.autoia.alive:
            self._render_vision_circle(self.autoia)

        # Blit del viewport a la pantalla principal
        self.screen.blit(self.viewport_surf, (0, 0))

    def _render_particles(self):
        for p in self.world.particles.particles:
            if not self.camera.is_visible(p.x, p.y):
                continue
            sx, sy = self.camera.world_to_screen(p.x, p.y)
            size = max(1, int(p.current_size * self.camera.zoom))
            alpha = p.alpha
            color = (*p.color, alpha)
            try:
                surf = pygame.Surface((size*2+2, size*2+2), pygame.SRCALPHA)
                pygame.draw.circle(surf, color, (size+1, size+1), max(1, size))
                self.viewport_surf.blit(surf, (sx-size-1, sy-size-1))
            except Exception:
                pass

    def _render_resources(self):
        """Dibuja recursos de energía con efecto de brillo."""
        for r in self.world.resources:
            if not self.camera.is_visible(r.x, r.y, margin=20):
                continue
            sx, sy = self.camera.world_to_screen(r.x, r.y)
            radius = max(3, int(r.radius * self.camera.zoom))

            if not r.active:
                # Recurso agotado
                pygame.draw.circle(self.viewport_surf, (40, 40, 40), (sx, sy), radius)
                pygame.draw.circle(self.viewport_surf, (60, 60, 60), (sx, sy), radius, 1)
                continue

            # Halo de brillo
            glow_r = int(radius * (1.5 + 0.3 * r.glow_intensity))
            try:
                glow_surf = pygame.Surface((glow_r*2+2, glow_r*2+2), pygame.SRCALPHA)
                glow_alpha = int(60 + 40 * r.glow_intensity)
                glow_color = (*r.color, glow_alpha)
                pygame.draw.circle(glow_surf, glow_color,
                                   (glow_r+1, glow_r+1), glow_r)
                self.viewport_surf.blit(glow_surf, (sx-glow_r-1, sy-glow_r-1))
            except Exception:
                pass

            # Núcleo
            pygame.draw.circle(self.viewport_surf, r.color, (sx, sy), radius)
            bright = tuple(min(255, c+80) for c in r.color)
            pygame.draw.circle(self.viewport_surf, bright, (sx, sy), max(2, radius//2))
            pygame.draw.circle(self.viewport_surf, (255, 255, 255), (sx, sy), radius, 1)

    def _render_trails(self):
        """Dibuja rastros de movimiento de agentes."""
        for agent in self.world.agents:
            if not agent.alive or len(agent.trail) < 2:
                continue
            trail_list = list(agent.trail)
            for i in range(1, len(trail_list)):
                ratio = i / len(trail_list)
                alpha = int(120 * ratio)
                color = agent.COLOR_BODY
                sx1, sy1 = self.camera.world_to_screen(*trail_list[i-1])
                sx2, sy2 = self.camera.world_to_screen(*trail_list[i])
                try:
                    line_surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H), pygame.SRCALPHA)
                    pygame.draw.line(line_surf, (*color, alpha), (sx1, sy1), (sx2, sy2),
                                     max(1, int(self.camera.zoom)))
                    self.viewport_surf.blit(line_surf, (0, 0))
                except Exception:
                    pass

    def _render_agents(self):
        """Dibuja todos los agentes."""
        for agent in self.world.agents:
            sx, sy = self.camera.world_to_screen(agent.x, agent.y)
            radius = max(3, int(agent.radius * self.camera.zoom))

            if not self.camera.is_visible(agent.x, agent.y, margin=30):
                continue

            if not agent.alive:
                # Agente muerto: X gris
                pygame.draw.circle(self.viewport_surf, (50, 50, 50), (sx, sy), radius)
                size = max(4, radius)
                pygame.draw.line(self.viewport_surf, (120, 50, 50),
                                 (sx-size, sy-size), (sx+size, sy+size), 2)
                pygame.draw.line(self.viewport_surf, (120, 50, 50),
                                 (sx+size, sy-size), (sx-size, sy+size), 2)
                continue

            is_autoia = getattr(agent, 'is_autoia', False)

            # Aura especial de Autoia
            if is_autoia:
                self._render_autoia_aura(agent, sx, sy)

            # Cuerpo principal
            pygame.draw.circle(self.viewport_surf, agent.COLOR_BODY, (sx, sy), radius)

            # Highlight interior
            bright = tuple(min(255, c+60) for c in agent.COLOR_BODY)
            pygame.draw.circle(self.viewport_surf, bright,
                               (sx - radius//4, sy - radius//4), max(2, radius//3))

            # Contorno
            pygame.draw.circle(self.viewport_surf, agent.COLOR_OUTLINE, (sx, sy), radius, 2)

            # Barra de energía
            bar_w = radius * 2 + 4
            bar_h = 3
            bx = sx - bar_w//2
            by = sy - radius - 8
            pygame.draw.rect(self.viewport_surf, (30, 30, 30), (bx, by, bar_w, bar_h))
            energy_w = max(0, int(bar_w * agent.energy))
            energy_color = self._energy_color(agent.energy)
            if energy_w > 0:
                pygame.draw.rect(self.viewport_surf, energy_color,
                                 (bx, by, energy_w, bar_h))

            # Nombre (si hay zoom suficiente)
            if self.camera.zoom > 0.8:
                name_surf = self.fonts.tiny.render(
                    agent.AGENT_NAME[:6], True, agent.COLOR_OUTLINE
                )
                self.viewport_surf.blit(name_surf,
                                         (sx - name_surf.get_width()//2, sy + radius + 2))

            # Indicador de estado (pequeño ícono)
            self._render_state_icon(agent, sx, sy, radius)

    def _render_autoia_aura(self, agent, sx, sy):
        """Aura pulsante especial para Autoia."""
        pulse = getattr(agent, 'pulse_alpha', 80)
        aura_r = getattr(agent, 'aura_size', agent.radius + 10)
        aura_r_px = max(5, int(aura_r * self.camera.zoom))

        try:
            aura_surf = pygame.Surface((aura_r_px*2+4, aura_r_px*2+4), pygame.SRCALPHA)
            pygame.draw.circle(aura_surf, (140, 60, 220, pulse),
                               (aura_r_px+2, aura_r_px+2), aura_r_px)
            self.viewport_surf.blit(aura_surf, (sx-aura_r_px-2, sy-aura_r_px-2))
        except Exception:
            pass

    def _render_state_icon(self, agent, sx, sy, radius):
        """Pequeño ícono según el estado del agente."""
        state_icons = {
            "exploring":      ("·", UITheme.TEXT_DIM),
            "mapping":        ("◎", (100, 200, 255)),
            "collecting":     ("▲", (100, 255, 100)),
            "chasing":        ("▶", (255, 100, 50)),
            "patrolling":     ("↻", (100, 150, 255)),
            "studying":       ("◈", (80, 220, 220)),
            "seeking_data":   ("⬟", (180, 100, 255)),
            "absorbing_data": ("★", (220, 180, 255)),
            "seeking_energy": ("⚡", (255, 220, 50)),
            "low_energy":     ("!", (255, 80, 30)),
            "insight":        ("✦", (255, 240, 80)),
        }
        state = getattr(agent, 'state', getattr(agent, 'current_goal', ''))
        if state in state_icons and self.camera.zoom > 0.7:
            icon, color = state_icons[state]
            icon_surf = self.fonts.tiny.render(icon, True, color)
            self.viewport_surf.blit(icon_surf, (sx + radius - 2, sy - radius - 2))

    def _render_thought_bubbles(self):
        """Dibuja burbujas de pensamiento sobre los agentes."""
        for agent in self.world.agents:
            if agent.is_thinking and agent.alive:
                draw_thought_bubble(self.viewport_surf, self.camera, agent)

    def _render_vision_circle(self, agent):
        """Dibuja el radio de visión de Autoia."""
        vision = self.world.physics.get_vision_range(agent)
        sx, sy = self.camera.world_to_screen(agent.x, agent.y)
        vr = int(vision * self.camera.zoom)
        try:
            vision_surf = pygame.Surface((vr*2+2, vr*2+2), pygame.SRCALPHA)
            pygame.draw.circle(vision_surf, (140, 60, 220, 20),
                               (vr+1, vr+1), vr)
            pygame.draw.circle(vision_surf, (140, 60, 220, 60),
                               (vr+1, vr+1), vr, 1)
            self.viewport_surf.blit(vision_surf, (sx-vr-1, sy-vr-1))
        except Exception:
            pass

    def _energy_color(self, ratio: float):
        if ratio > 0.6:
            return lerp_color((230, 200, 30), (50, 230, 100), (ratio-0.6)/0.4)
        elif ratio > 0.3:
            return lerp_color((230, 60, 30), (230, 200, 30), (ratio-0.3)/0.3)
        return (230, 60, 30)

    # ─── Panel derecho ────────────────────────────────────────────────────────

    def _render_panel_tabs(self):
        """Pestañas de navegación del panel derecho."""
        px = VIEWPORT_W
        tab_labels = ["1:Autoia", "2:Agentes", "3:Leyes", "4:Eventos"]
        tab_w = PANEL_RIGHT_W // len(tab_labels)

        for i, (mode, label) in enumerate(zip(self.panel_tabs, tab_labels)):
            tx = px + i * tab_w
            is_active = (mode == self.right_panel_mode)
            bg_color = UITheme.ACCENT if is_active else (25, 30, 50)
            pygame.draw.rect(self.screen, bg_color, (tx, 0, tab_w, 32))
            pygame.draw.rect(self.screen, UITheme.BORDER, (tx, 0, tab_w, 32), 1)
            text_color = (255, 255, 255) if is_active else UITheme.TEXT_DIM
            text_surf = self.fonts.tiny.render(label, True, text_color)
            self.screen.blit(text_surf, (tx + tab_w//2 - text_surf.get_width()//2, 10))

    def _render_right_panel(self):
        """Renderiza el panel derecho según el modo activo."""
        # Fondo del panel
        pygame.draw.rect(self.screen, (10, 13, 25),
                         (VIEWPORT_W, 0, PANEL_RIGHT_W, SCREEN_H))
        pygame.draw.line(self.screen, UITheme.BORDER,
                         (VIEWPORT_W, 0), (VIEWPORT_W, SCREEN_H), 2)

        world_stats = self.world.get_world_state_summary()
        mode = self.right_panel_mode

        if mode == "autoia":
            self.autoia_panel.x = VIEWPORT_W + 6
            self.autoia_panel.y = 38
            self.autoia_panel.w = PANEL_RIGHT_W - 12
            if self.autoia and self.autoia.is_thinking:
                self.autoia_panel.add_thought(self.autoia.thought)
            self.autoia_panel.draw(self.screen, self.autoia, world_stats)

            # Abajo: info del LLM
            if self.autoia:
                info_y = 330
                fm = self.fonts
                draw_text(self.screen, "── LLM ENGINE ──", fm.small,
                          UITheme.ACCENT, VIEWPORT_W + 10, info_y)
                info_y += 18
                llm_info = self.autoia.get_current_generation_text()
                draw_text(self.screen, llm_info, fm.tiny,
                          UITheme.TEXT_DIM, VIEWPORT_W + 10, info_y)
                info_y += 16

                # Observaciones recientes
                draw_text(self.screen, "── OBSERVACIONES ──", fm.small,
                          UITheme.ACCENT, VIEWPORT_W + 10, info_y)
                info_y += 16
                obs_list = list(self.autoia.observations)[-6:]
                for obs in reversed(obs_list):
                    words = obs[:32]
                    draw_text(self.screen, f"· {words}", fm.tiny,
                              UITheme.TEXT_DIM, VIEWPORT_W + 10, info_y)
                    info_y += 12
                    if info_y > SCREEN_H - 60:
                        break

        elif mode == "agents":
            agent_stats = self.world.get_agent_stats()
            # Panel con scroll
            panel_surf = pygame.Surface((PANEL_RIGHT_W-12, SCREEN_H-50), pygame.SRCALPHA)
            self.agents_panel.x = 0
            self.agents_panel.y = 0
            self.agents_panel.w = PANEL_RIGHT_W - 12
            self.agents_panel.draw(panel_surf, agent_stats)
            self.screen.blit(panel_surf, (VIEWPORT_W+6, 38))

        elif mode == "laws":
            self.laws_panel.x = VIEWPORT_W + 6
            self.laws_panel.y = 38
            self.laws_panel.w = PANEL_RIGHT_W - 12
            self.laws_panel.draw(self.screen, self.world.world_laws,
                                  self.world.sim_time)

        elif mode == "events":
            events = self.world.get_recent_events(12)
            self.events_panel.x = VIEWPORT_W + 6
            self.events_panel.y = 38
            self.events_panel.w = PANEL_RIGHT_W - 12
            self.events_panel.draw(self.screen, events)

            # Stats del mundo
            stats_y = 200
            draw_panel(self.screen, VIEWPORT_W+6, stats_y, PANEL_RIGHT_W-12, 180,
                       "◈  STATS GLOBALES")
            stats_y += 55
            fm = self.fonts
            lines = [
                f"Tiempo: {world_stats['sim_time']:.1f}s",
                f"Ticks:  {world_stats['tick']}",
                f"Agentes vivos: {world_stats['agents_alive']}/{len(self.world.agents)}",
                f"Energía media: {world_stats['avg_energy']*100:.1f}%",
                f"Recursos: {world_stats['resources_active']}/{len(self.world.resources)}",
                f"Vel. sim: {self.sim_speed:.1f}x",
                f"FPS: {self.clock.get_fps():.0f}",
            ]
            for line in lines:
                draw_text(self.screen, line, fm.small, UITheme.TEXT_MAIN,
                          VIEWPORT_W+14, stats_y)
                stats_y += 17

    def _render_speed_indicator(self):
        """Indicador de velocidad y pausa."""
        fm = self.fonts
        if self.paused:
            surf = fm.large.render("⏸ PAUSADO", True, (255, 200, 50))
        else:
            surf = fm.small.render(f"⏩ {self.sim_speed:.1f}x", True, UITheme.TEXT_DIM)
        self.screen.blit(surf, (VIEWPORT_W - surf.get_width() - 10,
                                SCREEN_H - surf.get_height() - 10))
