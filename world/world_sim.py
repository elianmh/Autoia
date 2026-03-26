"""
Simulación del mundo completo.
Gestiona el estado global, el ciclo de actualización y los eventos.
"""

import math
import random
import time
import logging
from typing import List, Dict, Optional, Tuple
from collections import deque

from world.physics import PhysicsEngine, TerrainType, TERRAIN_PROPS, WORLD_LAWS
from world.terrain import TerrainGrid
from world.entities import Resource, ParticleSystem
from world.agents.base import BaseAgent
from world.agents.npc import (
    ExplorerAgent, CollectorAgent, GuardianAgent,
    WandererAgent, PredatorAgent, ScholarAgent
)

logger = logging.getLogger("autoia.world")


class WorldEvent:
    """Un evento que ocurrió en el mundo."""
    def __init__(self, event_type: str, description: str, x: float, y: float):
        self.event_type  = event_type
        self.description = description
        self.x           = x
        self.y           = y
        self.timestamp   = time.time()


class WorldSimulation:
    """
    El mundo completo con todos sus componentes.
    Actualiza física, agentes, recursos y genera eventos observables.
    """

    # Tamaño del mundo
    TILE_SIZE = 18
    GRID_W    = 72
    GRID_H    = 54

    def __init__(self, seed: int = 42):
        self.seed = seed

        # Dimensiones en píxeles
        self.pixel_w = self.TILE_SIZE * self.GRID_W
        self.pixel_h = self.TILE_SIZE * self.GRID_H

        # Componentes
        self.terrain   = TerrainGrid(self.TILE_SIZE, self.GRID_W, self.GRID_H, seed)
        self.physics   = PhysicsEngine(self.pixel_w, self.pixel_h)
        self.particles = ParticleSystem()

        # Agentes y recursos
        self.agents:    List[BaseAgent] = []
        self.resources: List[Resource]  = []

        # Eventos recientes (para Autoia)
        self.events: deque = deque(maxlen=50)

        # Estadísticas
        self.sim_time   = 0.0
        self.tick_count = 0
        self.world_laws = WORLD_LAWS

        # Autoia reference (se inyecta desde fuera)
        self.autoia_agent = None

        self._setup()

    def _setup(self):
        """Inicializa agentes y recursos."""
        # Recursos de energía distribuidos por el mapa
        resource_positions = [
            # Zona central
            (self.pixel_w*0.50, self.pixel_h*0.50),
            (self.pixel_w*0.45, self.pixel_h*0.45),
            (self.pixel_w*0.55, self.pixel_h*0.45),
            # Cuadrantes
            (self.pixel_w*0.25, self.pixel_h*0.30),
            (self.pixel_w*0.75, self.pixel_h*0.30),
            (self.pixel_w*0.25, self.pixel_h*0.70),
            (self.pixel_w*0.75, self.pixel_h*0.70),
            # Bordes accesibles
            (self.pixel_w*0.50, self.pixel_h*0.20),
            (self.pixel_w*0.50, self.pixel_h*0.80),
            (self.pixel_w*0.20, self.pixel_h*0.50),
            (self.pixel_w*0.80, self.pixel_h*0.50),
            # Extra
            (self.pixel_w*0.35, self.pixel_h*0.35),
            (self.pixel_w*0.65, self.pixel_h*0.65),
            (self.pixel_w*0.35, self.pixel_h*0.65),
            (self.pixel_w*0.65, self.pixel_h*0.35),
        ]

        for i, (rx, ry) in enumerate(resource_positions):
            # Ajustar a posición walkable más cercana
            if not self.terrain.is_walkable_at(rx, ry):
                rx, ry = self.terrain.find_walkable_spawn()
            self.resources.append(Resource(
                i, rx, ry,
                max_energy=random.uniform(0.6, 1.2),
                regen_rate=random.uniform(0.02, 0.08)
            ))

        # Crear agentes NPC en posiciones aleatorias
        agent_classes = [
            ExplorerAgent, CollectorAgent, GuardianAgent,
            WandererAgent, PredatorAgent, ScholarAgent
        ]

        agent_id = 0
        for cls in agent_classes:
            for _ in range(2):  # 2 de cada tipo = 12 NPCs total
                x, y = self.terrain.find_walkable_spawn()
                agent = cls(agent_id, x, y, self.terrain)
                agent.energy = random.uniform(0.5, 1.0)
                self.agents.append(agent)
                agent_id += 1

        logger.info(f"Mundo creado: {self.pixel_w}x{self.pixel_h}px, "
                    f"{len(self.agents)} agentes, {len(self.resources)} recursos")

    def add_autoia(self, autoia_agent):
        """Inyecta el agente Autoia en el mundo."""
        self.autoia_agent = autoia_agent
        self.agents.append(autoia_agent)
        self.log_event("spawn", "Autoia ha entrado al mundo", autoia_agent.x, autoia_agent.y)

    # ─── Ciclo principal ───────────────────────────────────────────────────────

    def step(self, dt: float):
        """Un paso de simulación completo."""
        dt = min(dt, 0.05)  # Cap en 50ms para estabilidad

        self.sim_time   += dt
        self.tick_count += 1

        # 1. Actualizar recursos
        for r in self.resources:
            r.update(dt)

        # 2. Estado del mundo para agentes
        world_state = {
            "agents":    self.agents,
            "resources": self.resources,
            "time":      self.sim_time,
            "is_day":    self.physics.is_day,
        }

        # 3. Actualizar agentes (comportamiento)
        for agent in self.agents:
            agent.update(dt, world_state, self.physics)

        # 4. Aplicar física
        self.physics.update(dt, self.agents, self.terrain)

        # 5. Detectar y emitir eventos
        self._detect_events(dt)

        # 6. Actualizar partículas
        self.particles.update(dt)

        # 7. Autoia observa y aprende
        if self.autoia_agent and self.autoia_agent.alive:
            self._update_autoia_observations(dt)

    def _detect_events(self, dt: float):
        """Detecta eventos significativos y genera partículas."""
        for agent in self.agents:
            if not agent.alive:
                continue

            # Agente recogiendo recurso
            for r in self.resources:
                if r.active and agent.alive:
                    dist = math.sqrt((r.x-agent.x)**2 + (r.y-agent.y)**2)
                    if dist < agent.radius + r.radius + 5:
                        # Solo colectores y vagabundos recogen activamente
                        from world.agents.npc import CollectorAgent, WandererAgent
                        if isinstance(agent, (CollectorAgent, WandererAgent)):
                            gained = r.collect(0.15 * dt)
                            if gained > 0:
                                agent.energy = min(1.0, agent.energy + gained)
                                if random.random() < 0.02:
                                    self.particles.emit_energy(r.x, r.y, agent.COLOR_BODY)

        # Datos para Autoia
        if self.autoia_agent and self.autoia_agent.alive:
            terrain_here = self.terrain.get_terrain_at(
                self.autoia_agent.x, self.autoia_agent.y
            )
            if terrain_here == TerrainType.DATA:
                if random.random() < 0.1:
                    self.particles.emit_data(self.autoia_agent.x, self.autoia_agent.y)

    def _update_autoia_observations(self, dt: float):
        """Autoia observa el mundo y acumula experiencias."""
        vision = self.physics.get_vision_range(self.autoia_agent)
        visible = self.autoia_agent.get_visible_agents(self.agents, vision)

        # Registrar observaciones como eventos
        if self.tick_count % 120 == 0 and visible:  # Cada ~2 segundos
            obs = self._describe_scene(visible, vision)
            self.autoia_agent.add_observation(obs)
            self.log_event("observation", obs,
                           self.autoia_agent.x, self.autoia_agent.y)

    def _describe_scene(self, visible_agents: list, vision_range: float) -> str:
        """Genera descripción textual de lo que Autoia observa."""
        terrain = self.terrain.get_terrain_at(
            self.autoia_agent.x, self.autoia_agent.y
        )
        terrain_name = TERRAIN_PROPS[terrain].name
        parts = [f"Estoy en {terrain_name}."]

        if visible_agents:
            agent_names = [a.AGENT_NAME for a in visible_agents[:4]]
            parts.append(f"Veo a: {', '.join(agent_names)}.")

        # Describir actividades visibles
        for a in visible_agents[:3]:
            parts.append(f"El {a.AGENT_NAME} está en estado '{a.state}'.")

        # Hora del día
        parts.append("Es de día." if self.physics.is_day else "Es de noche.")

        return " ".join(parts)

    def log_event(self, event_type: str, description: str, x: float, y: float):
        self.events.append(WorldEvent(event_type, description, x, y))

    # ─── Consultas ────────────────────────────────────────────────────────────

    def get_world_state_summary(self) -> Dict:
        """Resumen del estado del mundo para el UI."""
        alive = sum(1 for a in self.agents if a.alive)
        avg_energy = (sum(a.energy for a in self.agents if a.alive) /
                      max(alive, 1))
        return {
            "sim_time":    self.sim_time,
            "tick":        self.tick_count,
            "agents_alive": alive,
            "avg_energy":  avg_energy,
            "is_day":      self.physics.is_day,
            "time_of_day": self.physics.time_of_day,
            "resources_active": sum(1 for r in self.resources if r.active),
        }

    def get_recent_events(self, n: int = 8) -> List[WorldEvent]:
        return list(self.events)[-n:]

    def get_agent_stats(self) -> List[Dict]:
        stats = []
        for a in self.agents:
            stats.append({
                "id":      a.agent_id,
                "name":    a.AGENT_NAME,
                "alive":   a.alive,
                "energy":  a.energy,
                "state":   a.state,
                "deaths":  a.death_count,
                "color":   a.COLOR_BODY,
                "thought": a.thought if a.is_thinking else "",
                "is_autoia": getattr(a, 'is_autoia', False),
            })
        return stats
