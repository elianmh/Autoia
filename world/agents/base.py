"""
Agente base del mundo. Todos los AIs heredan de esto.
"""

import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from world.physics import TerrainType, PhysicsConstants


@dataclass
class Memory:
    """Un recuerdo del agente."""
    event: str
    x: float
    y: float
    timestamp: float
    importance: float = 1.0


class BaseAgent:
    """
    Agente base con física, energía, visión y memoria.
    Todos los AIs del mundo son instancias de subclases de esto.
    """

    # Colores por defecto (sobreescritos en subclases)
    COLOR_BODY    = (200, 200, 200)
    COLOR_OUTLINE = (255, 255, 255)
    COLOR_ENERGY  = (0, 200, 100)
    AGENT_NAME    = "Agente"
    PERSONALITY   = "neutro"
    RADIUS        = 10

    MAX_MEMORY    = 40        # Ley: memoria limitada
    RESPAWN_DELAY = 3.0       # Segundos antes de resucitar

    def __init__(self, agent_id: int, x: float, y: float, terrain_grid):
        self.agent_id   = agent_id
        self.x          = float(x)
        self.y          = float(y)
        self.vx         = 0.0
        self.vy         = 0.0
        self.radius     = self.RADIUS
        self.energy     = 1.0          # 0-1
        self.health     = 1.0          # 0-1
        self.alive      = True
        self.terrain_grid = terrain_grid
        self.current_terrain = TerrainType.GRASS
        self.data_collected = 0.0

        # Memoria (ley: limitada)
        self.memory: deque = deque(maxlen=self.MAX_MEMORY)
        # Estado y comportamiento
        self.state        = "exploring"
        self.target_x     = x
        self.target_y     = y
        self.age          = 0.0         # Segundos de vida
        self.respawn_timer= 0.0
        self.death_count  = 0

        # Lo que el agente "piensa" (visible en UI)
        self.thought      = ""
        self.thought_timer= 0.0

        # Historial de posiciones (para rastro visual)
        self.trail: deque = deque(maxlen=20)
        self.trail_timer  = 0.0

        # Estadísticas
        self.energy_collected  = 0.0
        self.distance_traveled = 0.0
        self.interactions      = 0
        self._prev_x = x
        self._prev_y = y

    # ─── Ciclo de vida ────────────────────────────────────────────────────────

    def update(self, dt: float, world_state: Dict, physics):
        """Actualiza el agente. Sobreescribir en subclases."""
        if not self.alive:
            self.respawn_timer -= dt
            if self.respawn_timer <= 0:
                self._respawn()
            return

        self.age += dt

        # Rastro
        self.trail_timer += dt
        if self.trail_timer > 0.12:
            self.trail.append((self.x, self.y))
            self.trail_timer = 0

        # Actualizar distancia
        dx = self.x - self._prev_x
        dy = self.y - self._prev_y
        self.distance_traveled += math.sqrt(dx*dx + dy*dy)
        self._prev_x, self._prev_y = self.x, self.y

        # Tick de pensamiento
        self.thought_timer -= dt
        if self.thought_timer <= 0:
            self.thought_timer = 0

        # Comportamiento específico del agente
        self.behave(dt, world_state, physics)

    def behave(self, dt: float, world_state: Dict, physics):
        """Lógica de comportamiento. Sobreescribir en subclases."""
        self._wander(dt)

    def die(self):
        """El agente muere. Respawnea tras un delay."""
        if not self.alive:
            return
        self.alive = False
        self.death_count += 1
        self.respawn_timer = self.RESPAWN_DELAY
        self.vx = 0
        self.vy = 0
        self.remember(f"Morí en ({self.x:.0f},{self.y:.0f})", importance=2.0)

    def _respawn(self):
        """Reaparece en una posición walkable aleatoria."""
        spawn = self.terrain_grid.find_walkable_spawn()
        self.x, self.y = spawn
        self.energy = 0.5
        self.health = 0.8
        self.alive = True
        self.state = "exploring"
        self.set_thought("Renací...")

    # ─── Movimiento ───────────────────────────────────────────────────────────

    def move_toward(self, tx: float, ty: float, speed: float = 80.0):
        """Mueve el agente hacia un objetivo con aceleración."""
        dx = tx - self.x
        dy = ty - self.y
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 5:
            self.vx += (dx/dist) * speed * 0.15
            self.vy += (dy/dist) * speed * 0.15

    def _wander(self, dt: float):
        """Movimiento aleatorio de exploración."""
        dist_to_target = math.sqrt((self.target_x-self.x)**2 + (self.target_y-self.y)**2)
        if dist_to_target < 20:
            # Elegir nuevo objetivo
            self.target_x = self.x + random.uniform(-150, 150)
            self.target_y = self.y + random.uniform(-150, 150)
            # Asegurar que sea transitable
            for _ in range(10):
                tx, ty = self.target_x, self.target_y
                tx = max(20, min(self.terrain_grid.pixel_w-20, tx))
                ty = max(20, min(self.terrain_grid.pixel_h-20, ty))
                if self.terrain_grid.is_walkable_at(tx, ty):
                    self.target_x, self.target_y = tx, ty
                    break
                self.target_x = self.x + random.uniform(-150, 150)
                self.target_y = self.y + random.uniform(-150, 150)
        self.move_toward(self.target_x, self.target_y, speed=60)

    def flee_from(self, ex: float, ey: float, speed: float = 100.0):
        """Huye de un punto."""
        dx = self.x - ex
        dy = self.y - ey
        dist = math.sqrt(dx*dx + dy*dy)
        if dist > 0:
            self.vx += (dx/dist) * speed * 0.2
            self.vy += (dy/dist) * speed * 0.2

    # ─── Percepción ───────────────────────────────────────────────────────────

    def can_see(self, other_x: float, other_y: float, vision_range: float) -> bool:
        """Ley: visión limitada al radio de visión."""
        dx = other_x - self.x
        dy = other_y - self.y
        return (dx*dx + dy*dy) <= vision_range*vision_range

    def get_visible_agents(self, all_agents: list, vision_range: float) -> list:
        """Retorna agentes dentro del radio de visión."""
        return [
            a for a in all_agents
            if a is not self and a.alive and self.can_see(a.x, a.y, vision_range)
        ]

    def get_nearby_resources(self, resources: list, vision_range: float) -> list:
        return [
            r for r in resources
            if r.active and self.can_see(r.x, r.y, vision_range)
        ]

    # ─── Memoria ──────────────────────────────────────────────────────────────

    def remember(self, event: str, importance: float = 1.0):
        """Añade un recuerdo (ley: memoria limitada = maxlen)."""
        self.memory.append(Memory(
            event=event, x=self.x, y=self.y,
            timestamp=time.time(), importance=importance
        ))

    def get_recent_memories(self, n: int = 5) -> List[str]:
        memories = list(self.memory)[-n:]
        return [m.event for m in reversed(memories)]

    # ─── UI ───────────────────────────────────────────────────────────────────

    def set_thought(self, text: str, duration: float = 3.0):
        self.thought = text
        self.thought_timer = duration

    @property
    def is_thinking(self) -> bool:
        return self.thought_timer > 0 and bool(self.thought)

    def get_status_text(self) -> List[str]:
        return [
            f"{self.AGENT_NAME} #{self.agent_id}",
            f"Estado: {self.state}",
            f"Energía: {self.energy*100:.0f}%",
            f"Muertes: {self.death_count}",
            f"Dist: {self.distance_traveled/1000:.1f}km",
        ]
