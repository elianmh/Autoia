"""
Mapa del mundo: grid de terreno generado proceduralmente.
"""

import random
import math
from typing import Tuple, List
from world.physics import TerrainType, TERRAIN_PROPS


class TerrainGrid:
    """
    Grid 2D de terreno generado proceduralmente con noise.
    Contiene distintas zonas con sus propiedades físicas.
    """

    def __init__(self, tile_size: int, grid_w: int, grid_h: int, seed: int = 42):
        self.tile_size = tile_size
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.pixel_w = tile_size * grid_w
        self.pixel_h = tile_size * grid_h
        self.grid: List[List[TerrainType]] = []
        self.seed = seed
        random.seed(seed)
        self._generate()

    def _noise(self, x: float, y: float, scale: float = 1.0) -> float:
        """Pseudo-noise 2D simple para generación del mundo."""
        x, y = x * scale, y * scale
        xi, yi = int(x), int(y)
        xf, yf = x - xi, y - yi
        # Interpolación bilineal de valores hash
        def h(a, b): return (math.sin(a * 127.1 + b * 311.7) * 43758.5453) % 1
        v00 = h(xi,   yi)
        v10 = h(xi+1, yi)
        v01 = h(xi,   yi+1)
        v11 = h(xi+1, yi+1)
        # Smoothstep
        sx = xf * xf * (3 - 2 * xf)
        sy = yf * yf * (3 - 2 * yf)
        return v00 + sx*(v10-v00) + sy*(v01-v00) + sx*sy*(v00-v10-v01+v11)

    def _generate(self):
        """Genera el mapa usando capas de noise."""
        self.grid = []
        cx, cy = self.grid_w // 2, self.grid_h // 2

        for gy in range(self.grid_h):
            row = []
            for gx in range(self.grid_w):
                # Distancia al centro
                dx = (gx - cx) / self.grid_w
                dy = (gy - cy) / self.grid_h
                dist_center = math.sqrt(dx*dx + dy*dy)

                # Múltiples capas de noise
                n1 = self._noise(gx, gy, 0.08)
                n2 = self._noise(gx + 100, gy + 100, 0.15)
                n3 = self._noise(gx + 200, gy + 200, 0.25)
                combined = n1 * 0.5 + n2 * 0.3 + n3 * 0.2

                # Bordes del mundo → agua
                if dist_center > 0.42:
                    terrain = TerrainType.WATER
                elif combined < 0.25:
                    terrain = TerrainType.WATER
                elif combined < 0.35:
                    terrain = TerrainType.MUD
                elif combined < 0.45:
                    terrain = TerrainType.GRASS
                elif combined < 0.60:
                    terrain = TerrainType.GRASS
                elif combined < 0.70:
                    terrain = TerrainType.STONE
                elif combined < 0.78:
                    terrain = TerrainType.STONE
                else:
                    terrain = TerrainType.GRASS

                row.append(terrain)
            self.grid.append(row)

        # Añadir zonas especiales
        self._place_special_zones()

    def _place_special_zones(self):
        """Coloca zonas especiales: fuego, hielo, oscuridad, datos, energía."""
        # Zonas de fuego (esquinas)
        self._place_zone(3, 3, 4, TerrainType.FIRE)
        self._place_zone(self.grid_w-5, 3, 4, TerrainType.FIRE)

        # Zonas de hielo
        self._place_zone(self.grid_w//2, 5, 5, TerrainType.ICE)

        # Zonas de oscuridad
        self._place_zone(8, self.grid_h-8, 5, TerrainType.DARK)
        self._place_zone(self.grid_w-8, self.grid_h-8, 5, TerrainType.DARK)

        # Nodos de datos (para Autoia)
        data_positions = [
            (self.grid_w//4, self.grid_h//4),
            (3*self.grid_w//4, self.grid_h//4),
            (self.grid_w//2, self.grid_h//2),
            (self.grid_w//4, 3*self.grid_h//4),
            (3*self.grid_w//4, 3*self.grid_h//4),
        ]
        for gx, gy in data_positions:
            self._place_zone(gx, gy, 2, TerrainType.DATA)

        # Fuentes de energía
        energy_positions = [
            (self.grid_w//3, self.grid_h//2),
            (2*self.grid_w//3, self.grid_h//2),
            (self.grid_w//2, self.grid_h//3),
            (self.grid_w//2, 2*self.grid_h//3),
        ]
        for gx, gy in energy_positions:
            self._place_zone(gx, gy, 2, TerrainType.ENERGY)

    def _place_zone(self, cx: int, cy: int, radius: int, terrain: TerrainType):
        """Coloca una zona circular de un tipo de terreno."""
        for dy in range(-radius, radius+1):
            for dx in range(-radius, radius+1):
                if dx*dx + dy*dy <= radius*radius:
                    gx, gy = cx+dx, cy+dy
                    if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                        # No sobreescribir agua en bordes
                        if self.grid[gy][gx] != TerrainType.WATER or terrain == TerrainType.WATER:
                            self.grid[gy][gx] = terrain

    def get_terrain_at(self, px: float, py: float) -> TerrainType:
        """Obtiene el tipo de terreno en coordenadas pixel."""
        gx = int(px / self.tile_size)
        gy = int(py / self.tile_size)
        gx = max(0, min(self.grid_w-1, gx))
        gy = max(0, min(self.grid_h-1, gy))
        return self.grid[gy][gx]

    def get_tile_color(self, gx: int, gy: int) -> Tuple[int, int, int]:
        terrain = self.grid[gy][gx]
        return TERRAIN_PROPS[terrain].color

    def is_walkable_at(self, px: float, py: float) -> bool:
        terrain = self.get_terrain_at(px, py)
        return TERRAIN_PROPS[terrain].walkable

    def find_walkable_spawn(self) -> Tuple[float, float]:
        """Encuentra una posición aleatoria transitable."""
        for _ in range(200):
            gx = random.randint(5, self.grid_w-5)
            gy = random.randint(5, self.grid_h-5)
            if TERRAIN_PROPS[self.grid[gy][gx]].walkable:
                return (gx * self.tile_size + self.tile_size//2,
                        gy * self.tile_size + self.tile_size//2)
        # Fallback al centro
        return (self.pixel_w / 2, self.pixel_h / 2)

    def get_data_richness_at(self, px: float, py: float) -> float:
        terrain = self.get_terrain_at(px, py)
        return TERRAIN_PROPS[terrain].data_richness
