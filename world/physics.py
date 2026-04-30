"""
Motor de leyes físicas del mundo.
Define las reglas que gobiernan la simulación: qué puede y qué no puede ocurrir.
Inspirado en las leyes físicas reales pero en 2D.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
from enum import Enum, auto


# ─── Constantes físicas del mundo ─────────────────────────────────────────────

class PhysicsConstants:
    GRAVITY          = 0.0          # Sin gravedad vertical (top-down view)
    MAX_SPEED        = 180.0        # píxeles/segundo — límite de velocidad
    FRICTION_NORMAL  = 0.88         # Factor de fricción en terreno normal
    FRICTION_ICE     = 0.98         # Casi sin fricción en hielo
    FRICTION_MUD     = 0.60         # Mucho rozamiento en barro
    FRICTION_WATER   = 0.0          # No se puede caminar en agua
    ENERGY_DRAIN     = 0.02         # Energía perdida por segundo al moverse
    ENERGY_IDLE_DRAIN= 0.005        # Energía perdida por segundo en reposo
    COLLISION_BOUNCE = 0.3          # Coeficiente de rebote entre agentes
    VISION_RANGE_DAY = 220.0        # Radio de visión de día (px)
    VISION_RANGE_NIGHT = 90.0       # Radio de visión de noche (px)
    DAY_CYCLE_SECONDS = 60.0        # Duración de un ciclo día/noche en segundos


class TerrainType(Enum):
    """Tipos de terreno con sus propiedades físicas."""
    GRASS   = auto()   # Normal
    WATER   = auto()   # Intransitable
    STONE   = auto()   # Normal pero sin recursos
    FIRE    = auto()   # Daña agentes, no transitable
    ICE     = auto()   # Transitable, casi sin fricción
    MUD     = auto()   # Transitable, muy lento
    DARK    = auto()   # Transitable, visión reducida
    DATA    = auto()   # Nodo de datos: Autoia aprende aquí
    ENERGY  = auto()   # Fuente de energía: recarga agentes


@dataclass
class TerrainProperties:
    walkable: bool
    friction: float
    vision_mult: float      # Multiplicador de visión (0=ciega, 1=normal)
    damage_per_sec: float   # Daño por segundo al estar aquí
    energy_regen: float     # Regeneración de energía por segundo
    data_richness: float    # Riqueza de datos para Autoia (0-1)
    color: Tuple[int,int,int]
    name: str


TERRAIN_PROPS: Dict[TerrainType, TerrainProperties] = {
    TerrainType.GRASS: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_NORMAL,
        vision_mult=1.0, damage_per_sec=0.0, energy_regen=0.005,
        data_richness=0.2, color=(60, 140, 50), name="Hierba"
    ),
    TerrainType.WATER: TerrainProperties(
        walkable=False, friction=0.0,
        vision_mult=1.0, damage_per_sec=0.0, energy_regen=0.0,
        data_richness=0.1, color=(30, 90, 200), name="Agua"
    ),
    TerrainType.STONE: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_NORMAL,
        vision_mult=1.0, damage_per_sec=0.0, energy_regen=0.0,
        data_richness=0.1, color=(110, 110, 110), name="Piedra"
    ),
    TerrainType.FIRE: TerrainProperties(
        walkable=False, friction=0.0,
        vision_mult=1.2, damage_per_sec=0.3, energy_regen=0.0,
        data_richness=0.05, color=(220, 70, 10), name="Fuego"
    ),
    TerrainType.ICE: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_ICE,
        vision_mult=1.1, damage_per_sec=0.01, energy_regen=0.0,
        data_richness=0.15, color=(180, 220, 255), name="Hielo"
    ),
    TerrainType.MUD: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_MUD,
        vision_mult=0.8, damage_per_sec=0.0, energy_regen=0.002,
        data_richness=0.3, color=(100, 70, 40), name="Barro"
    ),
    TerrainType.DARK: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_NORMAL,
        vision_mult=0.35, damage_per_sec=0.0, energy_regen=0.0,
        data_richness=0.5, color=(30, 30, 50), name="Zona oscura"
    ),
    TerrainType.DATA: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_NORMAL,
        vision_mult=1.2, damage_per_sec=0.0, energy_regen=0.01,
        data_richness=1.0, color=(80, 200, 200), name="Nodo de datos"
    ),
    TerrainType.ENERGY: TerrainProperties(
        walkable=True, friction=PhysicsConstants.FRICTION_NORMAL,
        vision_mult=1.1, damage_per_sec=0.0, energy_regen=0.12,
        data_richness=0.4, color=(255, 210, 50), name="Fuente de energía"
    ),
}


# ─── Leyes del mundo (visibles en UI) ─────────────────────────────────────────

@dataclass
class WorldLaw:
    """Una ley que rige el mundo. Puede ser activa o violada."""
    name: str
    description: str
    category: str          # "fisica", "biologia", "social", "informacion"
    active: bool = True
    violation_count: int = 0


WORLD_LAWS: List[WorldLaw] = [
    # Leyes físicas
    WorldLaw("Conservación de energía",
             "Ningún agente puede crear energía de la nada",
             "fisica"),
    WorldLaw("Velocidad máxima",
             f"Ningún agente puede superar {PhysicsConstants.MAX_SPEED:.0f} px/s",
             "fisica"),
    WorldLaw("Colisión sólida",
             "Dos cuerpos sólidos no pueden ocupar el mismo espacio",
             "fisica"),
    WorldLaw("Fricción cinética",
             "El movimiento en superficies reduce la velocidad gradualmente",
             "fisica"),
    WorldLaw("Visión limitada",
             "Los agentes solo ven dentro de su radio de visión",
             "fisica"),
    # Leyes biológicas
    WorldLaw("Metabolismo",
             "Los agentes consumen energía al existir y moverse",
             "biologia"),
    WorldLaw("Muerte por inanición",
             "Un agente sin energía muere y reaparece",
             "biologia"),
    WorldLaw("Curación lenta",
             "Los agentes se recuperan lentamente en terreno seguro",
             "biologia"),
    # Leyes informacionales
    WorldLaw("Aprendizaje local",
             "Autoia solo aprende de lo que observa en su radio de visión",
             "informacion"),
    WorldLaw("Memoria limitada",
             "Los agentes recuerdan solo los últimos N eventos",
             "informacion"),
    WorldLaw("Incertidumbre",
             "Los agentes no conocen el estado exacto de otros",
             "informacion"),
    # Leyes sociales
    WorldLaw("No telepatía",
             "Los agentes no pueden comunicarse a distancia",
             "social"),
    WorldLaw("Territorio libre",
             "Ningún agente puede bloquear permanentemente un área",
             "social"),
]


# ─── Motor de física ───────────────────────────────────────────────────────────

class PhysicsEngine:
    """
    Aplica las leyes físicas a todas las entidades del mundo.
    """

    def __init__(self, world_width: int, world_height: int):
        self.world_w = world_width
        self.world_h = world_height
        self.time_of_day = 0.0      # 0.0=amanecer, 0.5=mediodía, 1.0=noche
        self.day_cycle = 0.0        # Acumulador en segundos

    def update(self, dt: float, entities: list, terrain_grid):
        """Aplica física a todas las entidades."""
        self.day_cycle += dt
        self.time_of_day = (math.sin(2 * math.pi * self.day_cycle /
                                     PhysicsConstants.DAY_CYCLE_SECONDS) + 1) / 2

        for entity in entities:
            if not entity.alive:
                continue
            self._apply_movement(entity, dt, terrain_grid)
            self._apply_terrain_effects(entity, dt, terrain_grid)
            self._apply_energy_drain(entity, dt)
            self._check_bounds(entity)

        # Resolver colisiones entre agentes
        self._resolve_collisions(entities)

    def _apply_movement(self, entity, dt: float, terrain_grid):
        """Aplica velocidad y fricción al agente."""
        terrain = terrain_grid.get_terrain_at(entity.x, entity.y)
        props = TERRAIN_PROPS[terrain]

        if not props.walkable:
            # No puede estar en terreno intransitable
            entity.vx = 0
            entity.vy = 0
            return

        # Aplicar velocidad
        new_x = entity.x + entity.vx * dt
        new_y = entity.y + entity.vy * dt

        # Verificar si el nuevo punto es transitable
        new_terrain = terrain_grid.get_terrain_at(new_x, new_y)
        new_props = TERRAIN_PROPS[new_terrain]

        if new_props.walkable:
            entity.x = new_x
            entity.y = new_y
        else:
            # Rebote contra obstáculo
            entity.vx *= -PhysicsConstants.COLLISION_BOUNCE
            entity.vy *= -PhysicsConstants.COLLISION_BOUNCE

        # Fricción
        friction = props.friction
        entity.vx *= friction
        entity.vy *= friction

        # Límite de velocidad (ley física universal)
        speed = math.sqrt(entity.vx**2 + entity.vy**2)
        if speed > PhysicsConstants.MAX_SPEED:
            factor = PhysicsConstants.MAX_SPEED / speed
            entity.vx *= factor
            entity.vy *= factor

    def _apply_terrain_effects(self, entity, dt: float, terrain_grid):
        """Aplica efectos del terreno al agente."""
        terrain = terrain_grid.get_terrain_at(entity.x, entity.y)
        props = TERRAIN_PROPS[terrain]

        # Daño
        entity.energy -= props.damage_per_sec * dt

        # Regeneración de energía
        entity.energy = min(1.0, entity.energy + props.energy_regen * dt)

        # Notificar al agente el terreno actual
        entity.current_terrain = terrain

        # Datos para Autoia
        if hasattr(entity, 'is_autoia') and entity.is_autoia:
            entity.data_collected += props.data_richness * dt

    def _apply_energy_drain(self, entity, dt: float):
        """Los agentes consumen energía al existir (metabolismo)."""
        speed = math.sqrt(entity.vx**2 + entity.vy**2)
        if speed > 5:
            entity.energy -= PhysicsConstants.ENERGY_DRAIN * dt
        else:
            entity.energy -= PhysicsConstants.ENERGY_IDLE_DRAIN * dt

        # Muerte por inanición
        if entity.energy <= 0:
            entity.energy = 0
            entity.die()

    def _check_bounds(self, entity):
        """Mantiene entidades dentro del mundo."""
        margin = entity.radius
        entity.x = max(margin, min(self.world_w - margin, entity.x))
        entity.y = max(margin, min(self.world_h - margin, entity.y))
        if entity.x <= margin or entity.x >= self.world_w - margin:
            entity.vx *= -PhysicsConstants.COLLISION_BOUNCE
        if entity.y <= margin or entity.y >= self.world_h - margin:
            entity.vy *= -PhysicsConstants.COLLISION_BOUNCE

    def _resolve_collisions(self, entities: list):
        """Resuelve colisiones entre agentes (círculo vs círculo)."""
        alive = [e for e in entities if e.alive]
        for i, a in enumerate(alive):
            for b in alive[i+1:]:
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.sqrt(dx**2 + dy**2)
                min_dist = a.radius + b.radius

                if dist < min_dist and dist > 0:
                    # Separación
                    overlap = (min_dist - dist) / 2
                    nx, ny = dx/dist, dy/dist
                    a.x -= nx * overlap
                    a.y -= ny * overlap
                    b.x += nx * overlap
                    b.y += ny * overlap

                    # Intercambio de momentum
                    rel_vx = b.vx - a.vx
                    rel_vy = b.vy - a.vy
                    dot = rel_vx * nx + rel_vy * ny
                    if dot < 0:
                        impulse = PhysicsConstants.COLLISION_BOUNCE * dot
                        a.vx += impulse * nx
                        a.vy += impulse * ny
                        b.vx -= impulse * nx
                        b.vy -= impulse * ny

    def get_vision_range(self, entity) -> float:
        """Radio de visión según momento del día."""
        base = (PhysicsConstants.VISION_RANGE_DAY * self.time_of_day +
                PhysicsConstants.VISION_RANGE_NIGHT * (1 - self.time_of_day))
        # Modificar por terreno
        if hasattr(entity, 'current_terrain') and entity.current_terrain:
            props = TERRAIN_PROPS[entity.current_terrain]
            base *= props.vision_mult
        return max(50.0, base)

    @property
    def is_day(self) -> bool:
        return self.time_of_day > 0.5

    @property
    def sky_color(self) -> Tuple[int, int, int]:
        """Color del cielo según hora del día."""
        day = (25, 35, 80)      # Noche
        night = (100, 160, 255) # Día
        t = self.time_of_day
        return tuple(int(day[i] + (night[i]-day[i])*t) for i in range(3))

    @property
    def ambient_light(self) -> float:
        """Iluminación ambiental (0=oscuro, 1=plena luz)."""
        return 0.2 + 0.8 * self.time_of_day
