"""
Agentes NPC del mundo: cada uno con personalidad y comportamiento distinto.
Todos siguen las mismas leyes físicas que Autoia.
"""

import math
import random
from typing import Dict
from world.agents.base import BaseAgent
from world.physics import TerrainType, TERRAIN_PROPS


class ExplorerAgent(BaseAgent):
    """
    EXPLORADOR — Rojo
    Personalidad: curioso, intrépido. Siempre busca zonas nuevas que no ha visitado.
    Visita cada rincón del mapa. Comparte posiciones de recursos si los encuentra.
    """
    COLOR_BODY    = (200, 50, 50)
    COLOR_OUTLINE = (255, 120, 120)
    COLOR_ENERGY  = (255, 80, 80)
    AGENT_NAME    = "Explorador"
    PERSONALITY   = "Curioso e intrépido. Mapea el mundo."
    RADIUS        = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_zones = set()
        self.exploration_grid_size = 60  # px por celda de exploración
        self.state = "mapping"

    def behave(self, dt, world_state, physics):
        self.start_action("explore")
        vision = physics.get_vision_range(self)

        # Marcar zona visitada
        cx = int(self.x / self.exploration_grid_size)
        cy = int(self.y / self.exploration_grid_size)
        self.visited_zones.add((cx, cy))

        # Buscar energía si está baja
        if self.energy < 0.25:
            self.state = "seeking_energy"
            self.start_action("seek_energy")
            resources = self.get_nearby_resources(world_state.get("resources", []), vision)
            energy_sources = [r for r in resources if r.energy_value > 0]
            if energy_sources:
                closest = min(energy_sources, key=lambda r: (r.x-self.x)**2+(r.y-self.y)**2)
                self.move_toward(closest.x, closest.y,
                                 speed=110 * self.get_skill("seek_energy"))
                self.set_thought("Busco energia!")
                self.learn_from_outcome(dt)
                return

        # Explorar zonas no visitadas
        self.state = "mapping"
        best_target = None
        best_score = -1
        explore_range = 300 * self.get_skill("explore")

        for _ in range(20):
            tx = random.uniform(20, self.terrain_grid.pixel_w-20)
            ty = random.uniform(20, self.terrain_grid.pixel_h-20)
            gcx = int(tx / self.exploration_grid_size)
            gcy = int(ty / self.exploration_grid_size)
            if (gcx, gcy) not in self.visited_zones:
                dx, dy = tx-self.x, ty-self.y
                dist = math.sqrt(dx*dx+dy*dy)
                if self.terrain_grid.is_walkable_at(tx, ty) and dist < explore_range:
                    score = 1.0 / (1 + dist/100)
                    if score > best_score:
                        best_score = score
                        best_target = (tx, ty)

        if best_target:
            self.target_x, self.target_y = best_target
            self.move_toward(self.target_x, self.target_y, speed=95)
        else:
            self._wander(dt)

        self.learn_from_outcome(dt)

        if random.random() < 0.005:
            exp_rounded = round(self.experience, 1)
            self.set_thought(f"Zonas: {len(self.visited_zones)} | XP:{exp_rounded}")


class CollectorAgent(BaseAgent):
    """
    RECOLECTOR — Verde
    Personalidad: eficiente, metódico. Maximiza la recolección de energía.
    Siempre va a la fuente de energía más cercana.
    """
    COLOR_BODY    = (40, 180, 70)
    COLOR_OUTLINE = (100, 255, 120)
    COLOR_ENERGY  = (100, 255, 100)
    AGENT_NAME    = "Recolector"
    PERSONALITY   = "Eficiente y metódico. Acumula recursos."
    RADIUS        = 11

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collected_total = 0.0
        self.state = "seeking_resource"

    def behave(self, dt, world_state, physics):
        vision = physics.get_vision_range(self)
        resources = self.get_nearby_resources(world_state.get("resources", []), vision * 1.3)

        if resources:
            # Ir a la fuente más rica cercana
            best = max(resources, key=lambda r: r.energy_value / max(1,
                math.sqrt((r.x-self.x)**2+(r.y-self.y)**2)))
            self.state = "collecting"
            self.target_x, self.target_y = best.x, best.y
            self.move_toward(best.x, best.y, speed=90)

            dist = math.sqrt((best.x-self.x)**2+(best.y-self.y)**2)
            if dist < self.radius + best.radius:
                gained = best.collect(0.3 * dt)
                self.energy = min(1.0, self.energy + gained)
                self.collected_total += gained
                if gained > 0 and random.random() < 0.02:
                    self.set_thought(f"Recolectando... total: {self.collected_total:.1f}")
        else:
            self.state = "searching"
            # Ir hacia zonas de energía conocidas en el mapa
            energy_zones = [(self.terrain_grid.pixel_w//3,   self.terrain_grid.pixel_h//2),
                            (2*self.terrain_grid.pixel_w//3, self.terrain_grid.pixel_h//2),
                            (self.terrain_grid.pixel_w//2,   self.terrain_grid.pixel_h//3),
                            (self.terrain_grid.pixel_w//2,   2*self.terrain_grid.pixel_h//3)]
            best_zone = min(energy_zones, key=lambda z: (z[0]-self.x)**2+(z[1]-self.y)**2)
            self.move_toward(best_zone[0], best_zone[1], speed=75)


class GuardianAgent(BaseAgent):
    """
    GUARDIÁN — Azul
    Personalidad: protector, territorial. Defiende una zona central.
    Alerta a otros agentes de peligros cercanos.
    """
    COLOR_BODY    = (50, 80, 220)
    COLOR_OUTLINE = (100, 150, 255)
    COLOR_ENERGY  = (80, 120, 255)
    AGENT_NAME    = "Guardián"
    PERSONALITY   = "Protector y territorial. Defiende zonas clave."
    RADIUS        = 12

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Territorio a defender (nodo de datos central)
        self.territory_x = self.terrain_grid.pixel_w // 2
        self.territory_y = self.terrain_grid.pixel_h // 2
        self.territory_radius = 120.0
        self.alerts_issued = 0
        self.state = "patrolling"
        # Punto de patrulla actual
        self.patrol_angle = random.uniform(0, 2*math.pi)

    def behave(self, dt, world_state, physics):
        vision = physics.get_vision_range(self)
        visible = self.get_visible_agents(world_state.get("agents", []), vision)

        # ¿Hay intrusos en el territorio?
        intruders = [
            a for a in visible
            if not isinstance(a, GuardianAgent) and
            math.sqrt((a.x-self.territory_x)**2+(a.y-self.territory_y)**2) < self.territory_radius
        ]

        if intruders:
            self.state = "intercepting"
            closest = min(intruders, key=lambda a: (a.x-self.x)**2+(a.y-self.y)**2)
            self.move_toward(closest.x, closest.y, speed=100)
            self.alerts_issued += 1
            if random.random() < 0.01:
                self.set_thought(f"¡Intruso detectado! Alerta #{self.alerts_issued}")
        else:
            # Patrullar el perímetro del territorio
            self.state = "patrolling"
            self.patrol_angle += 0.5 * dt
            px = self.territory_x + math.cos(self.patrol_angle) * self.territory_radius * 0.8
            py = self.territory_y + math.sin(self.patrol_angle) * self.territory_radius * 0.8

            # Asegurar que sea transitable
            if self.terrain_grid.is_walkable_at(px, py):
                self.move_toward(px, py, speed=65)
            else:
                self.patrol_angle += math.pi * 0.1

        # Buscar energía si baja
        if self.energy < 0.2:
            self.state = "low_energy"
            resources = self.get_nearby_resources(world_state.get("resources", []), vision)
            if resources:
                closest = min(resources, key=lambda r: (r.x-self.x)**2+(r.y-self.y)**2)
                self.move_toward(closest.x, closest.y, speed=100)


class WandererAgent(BaseAgent):
    """
    VAGABUNDO — Amarillo
    Personalidad: impredecible, caótico. Movimiento casi aleatorio.
    A veces tiene destellos de insight y va directo a un recurso.
    """
    COLOR_BODY    = (220, 200, 30)
    COLOR_OUTLINE = (255, 240, 80)
    COLOR_ENERGY  = (255, 220, 0)
    AGENT_NAME    = "Vagabundo"
    PERSONALITY   = "Impredecible y caótico. Movimiento aleatorio con intuición."
    RADIUS        = 9

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.insight_timer = 0.0
        self.insight_target = None
        self.state = "wandering"

    def behave(self, dt, world_state, physics):
        vision = physics.get_vision_range(self)

        # Destellos de insight
        self.insight_timer -= dt
        if self.insight_timer <= 0:
            self.insight_timer = random.uniform(5, 20)
            if random.random() < 0.4:
                resources = self.get_nearby_resources(world_state.get("resources", []), vision * 2)
                if resources:
                    self.insight_target = random.choice(resources)
                    self.state = "insight"
                    self.set_thought("¡Insight! Sé a dónde ir...")
                else:
                    self.insight_target = None
                    self.state = "wandering"
            else:
                self.insight_target = None
                self.state = "wandering"

        if self.insight_target and self.state == "insight":
            if self.insight_target.active:
                self.move_toward(self.insight_target.x, self.insight_target.y, speed=110)
                dist = math.sqrt((self.insight_target.x-self.x)**2+
                                 (self.insight_target.y-self.y)**2)
                if dist < self.radius + self.insight_target.radius:
                    gained = self.insight_target.collect(0.4 * dt)
                    self.energy = min(1.0, self.energy + gained)
            else:
                self.insight_target = None
                self.state = "wandering"
        else:
            # Movimiento caótico
            if random.random() < 0.05:
                self.vx += random.uniform(-50, 50)
                self.vy += random.uniform(-50, 50)
            self._wander(dt)


class PredatorAgent(BaseAgent):
    """
    DEPREDADOR — Naranja
    Personalidad: agresivo, estratégico. Persigue agentes con poca energía.
    Al "atrapar" a otro agente le roba energía.
    """
    COLOR_BODY    = (220, 120, 20)
    COLOR_OUTLINE = (255, 160, 60)
    COLOR_ENERGY  = (255, 100, 0)
    AGENT_NAME    = "Depredador"
    PERSONALITY   = "Agresivo y estratégico. Caza a los más débiles."
    RADIUS        = 11

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prey = None
        self.energy_stolen = 0.0
        self.state = "hunting"

    def behave(self, dt, world_state, physics):
        vision = physics.get_vision_range(self)
        visible = self.get_visible_agents(world_state.get("agents", []), vision)

        # Filtrar presas válidas (excluir guardianes y otros depredadores)
        prey_candidates = [
            a for a in visible
            if not isinstance(a, (PredatorAgent, GuardianAgent))
            and a.energy < 0.6
        ]

        if prey_candidates:
            # Elegir la presa más débil más cercana
            self.prey = min(prey_candidates,
                            key=lambda a: a.energy + (math.sqrt((a.x-self.x)**2+(a.y-self.y)**2)/500))
            self.state = "chasing"
            self.move_toward(self.prey.x, self.prey.y, speed=120)

            # ¿Capturó la presa?
            dist = math.sqrt((self.prey.x-self.x)**2+(self.prey.y-self.y)**2)
            if dist < self.radius + self.prey.radius + 5:
                stolen = min(0.08 * dt, self.prey.energy)
                self.prey.energy -= stolen
                self.energy = min(1.0, self.energy + stolen * 0.5)
                self.energy_stolen += stolen
                if random.random() < 0.02:
                    self.set_thought(f"¡Cazado! Robé {self.energy_stolen:.1f} energía")
                    self.prey.set_thought("¡Me atacan!")
        else:
            # Buscar energía si está baja, o explorar
            self.prey = None
            if self.energy < 0.3:
                self.state = "low_energy"
                resources = self.get_nearby_resources(world_state.get("resources", []), vision)
                if resources:
                    closest = min(resources, key=lambda r: (r.x-self.x)**2+(r.y-self.y)**2)
                    self.move_toward(closest.x, closest.y, speed=100)
                    return
            self.state = "hunting"
            self._wander(dt)


class ScholarAgent(BaseAgent):
    """
    ESTUDIOSO — Cian
    Personalidad: intelectual, metódico. Va a los nodos de datos y los "estudia".
    Comparte conocimiento con Autoia cuando está cerca.
    """
    COLOR_BODY    = (20, 190, 190)
    COLOR_OUTLINE = (80, 240, 240)
    COLOR_ENERGY  = (0, 220, 220)
    AGENT_NAME    = "Estudioso"
    PERSONALITY   = "Intelectual. Busca nodos de datos y comparte conocimiento."
    RADIUS        = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge = 0.0
        self.data_nodes = []   # Posiciones de nodos de datos conocidas
        self.state = "seeking_knowledge"
        self._discover_data_nodes()

    def _discover_data_nodes(self):
        """Conoce las posiciones de los nodos de datos del mapa."""
        grid = self.terrain_grid
        from world.physics import TerrainType, TERRAIN_PROPS
        for gy in range(grid.grid_h):
            for gx in range(grid.grid_w):
                if grid.grid[gy][gx] == TerrainType.DATA:
                    px = gx * grid.tile_size + grid.tile_size // 2
                    py = gy * grid.tile_size + grid.tile_size // 2
                    self.data_nodes.append((px, py))
        # Reducir a centros únicos
        self.data_nodes = self.data_nodes[::6][:8]

    def behave(self, dt, world_state, physics):
        vision = physics.get_vision_range(self)

        # Ir a nodo de datos
        if self.data_nodes:
            current_terrain = self.terrain_grid.get_terrain_at(self.x, self.y)
            from world.physics import TerrainType
            if current_terrain == TerrainType.DATA:
                # Estudiar aquí
                self.state = "studying"
                gained = 0.1 * dt
                self.knowledge += gained
                self.data_collected += gained
                if random.random() < 0.015:
                    self.set_thought(f"Estudiando... conocimiento: {self.knowledge:.2f}")
                # Moverse poco (estudiando)
                self.vx *= 0.5
                self.vy *= 0.5
            else:
                self.state = "seeking_knowledge"
                closest = min(self.data_nodes,
                              key=lambda n: (n[0]-self.x)**2+(n[1]-self.y)**2)
                self.move_toward(closest[0], closest[1], speed=80)

        # Energía baja
        if self.energy < 0.2:
            self.state = "seeking_energy"
            resources = self.get_nearby_resources(world_state.get("resources", []), vision)
            if resources:
                closest = min(resources, key=lambda r: (r.x-self.x)**2+(r.y-self.y)**2)
                self.move_toward(closest.x, closest.y, speed=95)
