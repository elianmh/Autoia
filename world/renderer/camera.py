"""
Sistema de cámara con zoom y scroll para el mundo 2D.
"""

import math


class Camera:
    """
    Cámara que sigue a Autoia con zoom suave y scroll.
    """

    def __init__(self, viewport_w: int, viewport_h: int,
                 world_w: int, world_h: int):
        self.vp_w = viewport_w
        self.vp_h = viewport_h
        self.world_w = world_w
        self.world_h = world_h

        # Posición del centro de la cámara en coordenadas mundo
        self.cx = world_w / 2
        self.cy = world_h / 2

        # Zoom
        self.zoom = 1.0
        self.target_zoom = 1.0
        self.min_zoom = 0.3
        self.max_zoom = 2.5

        # Suavidad de seguimiento
        self.follow_speed = 4.0
        self.zoom_speed = 5.0

        # Objetivo a seguir
        self.follow_target = None

    def follow(self, target):
        """Establece el objetivo que la cámara sigue."""
        self.follow_target = target

    def zoom_in(self):
        self.target_zoom = min(self.max_zoom, self.target_zoom * 1.15)

    def zoom_out(self):
        self.target_zoom = max(self.min_zoom, self.target_zoom / 1.15)

    def scroll(self, dx: float, dy: float):
        """Desplazamiento manual de la cámara."""
        self.follow_target = None
        self.cx += dx / self.zoom
        self.cy += dy / self.zoom
        self._clamp()

    def update(self, dt: float):
        """Actualiza la posición suave de la cámara."""
        # Zoom suave
        diff = self.target_zoom - self.zoom
        self.zoom += diff * self.zoom_speed * dt

        # Seguir objetivo
        if self.follow_target and self.follow_target.alive:
            tx, ty = self.follow_target.x, self.follow_target.y
            self.cx += (tx - self.cx) * self.follow_speed * dt
            self.cy += (ty - self.cy) * self.follow_speed * dt

        self._clamp()

    def _clamp(self):
        """Limita la cámara para no salir del mundo."""
        half_vp_w = (self.vp_w / 2) / self.zoom
        half_vp_h = (self.vp_h / 2) / self.zoom
        self.cx = max(half_vp_w, min(self.world_w - half_vp_w, self.cx))
        self.cy = max(half_vp_h, min(self.world_h - half_vp_h, self.cy))

    def world_to_screen(self, wx: float, wy: float):
        """Convierte coordenadas mundo a coordenadas pantalla."""
        sx = (wx - self.cx) * self.zoom + self.vp_w / 2
        sy = (wy - self.cy) * self.zoom + self.vp_h / 2
        return int(sx), int(sy)

    def screen_to_world(self, sx: float, sy: float):
        """Convierte coordenadas pantalla a coordenadas mundo."""
        wx = (sx - self.vp_w / 2) / self.zoom + self.cx
        wy = (sy - self.vp_h / 2) / self.zoom + self.cy
        return wx, wy

    def is_visible(self, wx: float, wy: float, margin: float = 50) -> bool:
        """Verifica si un punto del mundo es visible en la pantalla."""
        sx, sy = self.world_to_screen(wx, wy)
        return (-margin <= sx <= self.vp_w + margin and
                -margin <= sy <= self.vp_h + margin)

    def get_visible_rect(self):
        """Retorna el rectángulo del mundo visible (wx1, wy1, wx2, wy2)."""
        wx1, wy1 = self.screen_to_world(0, 0)
        wx2, wy2 = self.screen_to_world(self.vp_w, self.vp_h)
        return wx1, wy1, wx2, wy2
