"""
Agente Autoia en el mundo.
Es el LLM encarnado: observa el mundo, aprende de él, genera texto sobre lo que ve.
"""

import math
import random
import time
from collections import deque
from typing import List, Dict, Optional

from world.agents.base import BaseAgent
from world.physics import TerrainType, TERRAIN_PROPS


def _load_curiosity_engine(orchestrator=None, llm_system=None, persona=None):
    """Carga el motor de curiosidad si está disponible."""
    try:
        from learning.curiosity import CuriosityEngine
        engine = CuriosityEngine(
            orchestrator=orchestrator,
            llm_system=llm_system,
            persona=persona or {},
        )
        # Cargar primeras 2 fases del curriculum de inmediato
        engine.load_curriculum_phase("phase_1_fundamentals")
        engine.load_curriculum_phase("phase_2_systems")
        return engine
    except Exception as e:
        return None


class AutoiaWorldAgent(BaseAgent):
    """
    El agente Autoia dentro del mundo.

    Características únicas:
    - Absorbe datos de nodos de información
    - Genera observaciones textuales de lo que ve
    - Sus pensamientos se alimentan al LLM para entrenamiento
    - Busca zonas de datos activamente
    - Aprende de la interacción con otros agentes
    - Su tamaño visual crece con cada generación del LLM
    """

    COLOR_BODY    = (140, 60, 220)
    COLOR_OUTLINE = (200, 120, 255)
    COLOR_ENERGY  = (180, 80, 255)
    AGENT_NAME    = "Autoia"
    PERSONALITY   = "IA auto-aprendiente. Observa, aprende y crece."
    RADIUS        = 13
    MAX_MEMORY    = 100   # Autoia tiene más memoria

    def __init__(self, agent_id: int, x: float, y: float, terrain_grid,
                 llm_system=None, orchestrator=None, persona=None):
        super().__init__(agent_id, x, y, terrain_grid)
        self.is_autoia      = True
        self.llm_system     = llm_system
        self.llm_generation = 0

        # Observaciones del mundo (se usan como corpus de aprendizaje)
        self.observations: deque = deque(maxlen=200)
        self.pending_learn: List[str] = []

        # Estado cognitivo
        self.curiosity       = 1.0    # 0-1: cuánto desea explorar zonas nuevas
        self.knowledge_level = 0.0   # Acumulado de datos absorbidos
        self.current_goal    = "exploring"
        self.persona         = persona or {}

        # Motor de curiosidad (aprendizaje autónomo)
        self.curiosity_engine = _load_curiosity_engine(
            orchestrator=orchestrator,
            llm_system=llm_system,
            persona=persona,
        )
        if self.curiosity_engine:
            self.curiosity_engine.on_learned = self._on_learned
            self.curiosity_engine.on_new_thought = self.set_thought

        # Nodos de datos conocidos
        self.known_data_nodes: List = []
        self.current_data_target: Optional = None

        # Cooldown de aprendizaje
        self.learn_cooldown = 0.0
        self.learn_interval = 30.0  # Intentar aprender cada 30s

        # Visual
        self.pulse_phase    = 0.0
        self.aura_radius    = 0.0

        # Stats específicas
        self.terrains_visited = set()
        self.agents_observed  = set()
        self.total_obs_chars  = 0

        # Nombre de persona
        name = self.persona.get("name", "Autoia")
        self.AGENT_NAME = name

        self._discover_data_nodes()

    def _discover_data_nodes(self):
        """Autoia conoce los nodos de datos del mundo desde el inicio."""
        grid = self.terrain_grid
        visited = set()
        for gy in range(0, grid.grid_h, 2):
            for gx in range(0, grid.grid_w, 2):
                if grid.grid[gy][gx] == TerrainType.DATA:
                    key = (gx//4, gy//4)
                    if key not in visited:
                        px = gx * grid.tile_size + grid.tile_size//2
                        py = gy * grid.tile_size + grid.tile_size//2
                        self.known_data_nodes.append((px, py))
                        visited.add(key)
        self.known_data_nodes = self.known_data_nodes[:8]

    # ─── Comportamiento ───────────────────────────────────────────────────────

    def behave(self, dt: float, world_state: Dict, physics):
        self.pulse_phase += 2.5 * dt
        self.learn_cooldown -= dt

        # Actualizar visual según generación LLM
        if self.llm_system and self.llm_system.model:
            gen = self.llm_system.model.snapshot.generation
            if gen != self.llm_generation:
                self.llm_generation = gen
                self.radius = min(18, 13 + gen * 2)  # Crece visualmente
                self.set_thought(f"Evolucioné! Generación {gen}", duration=5.0)

        # Motor de curiosidad: tick cada frame
        if self.curiosity_engine:
            thought = self.curiosity_engine.tick(time.time())
            if thought:
                self.set_thought(thought[:60], duration=6.0)
                # La observación del mundo también alimenta preguntas
                self.knowledge_level += 0.01

        # Decidir objetivo según estado
        self._decide_goal(world_state, physics)

        # Ejecutar comportamiento según objetivo
        if self.current_goal == "absorbing_data":
            self._absorb_data(dt, physics)
        elif self.current_goal == "seeking_data":
            self._seek_data(dt)
        elif self.current_goal == "observing":
            self._observe_world(dt, world_state, physics)
        elif self.current_goal == "seeking_energy":
            self._seek_energy(dt, world_state, physics)
        else:
            self._explore(dt, world_state, physics)

        # Intentar aprender del corpus acumulado
        if self.learn_cooldown <= 0 and self.pending_learn:
            self._trigger_learning()

    def _decide_goal(self, world_state: Dict, physics):
        """Decide el objetivo actual de Autoia."""
        # Prioridad 1: Energía crítica
        if self.energy < 0.15:
            self.current_goal = "seeking_energy"
            return

        # Prioridad 2: Absorber datos en nodo actual
        current_terrain = self.terrain_grid.get_terrain_at(self.x, self.y)
        if current_terrain == TerrainType.DATA:
            self.current_goal = "absorbing_data"
            return

        # Prioridad 3: Energía baja, buscar recarga
        if self.energy < 0.35:
            self.current_goal = "seeking_energy"
            return

        # Prioridad 4: Ir a nodo de datos
        if self.current_data_target:
            self.current_goal = "seeking_data"
        elif self.known_data_nodes:
            # Elegir nodo de datos más cercano no visitado recientemente
            unvisited = [
                n for n in self.known_data_nodes
                if n not in self.terrains_visited
            ] or self.known_data_nodes

            self.current_data_target = min(
                unvisited,
                key=lambda n: (n[0]-self.x)**2+(n[1]-self.y)**2
            )
            self.current_goal = "seeking_data"
        else:
            # Observar entorno
            self.current_goal = "observing"

    def _absorb_data(self, dt: float, physics):
        """Absorbe datos del nodo actual."""
        gain = 0.15 * dt
        self.data_collected += gain
        self.knowledge_level += gain * 0.5

        # Generar observación sobre el aprendizaje
        if random.random() < 0.03:
            obs = f"Absorbiendo datos del nodo. Conocimiento acumulado: {self.knowledge_level:.2f}"
            self.add_observation(obs)
            self.set_thought(f"Aprendiendo... {self.knowledge_level:.1f}")

        # Registrar este nodo como visitado
        self.terrains_visited.add(self.current_data_target)
        self.current_data_target = None

        # Moverse poco (concentrada absorbiendo)
        self.vx *= 0.7
        self.vy *= 0.7

    def _seek_data(self, dt: float):
        """Va hacia el nodo de datos objetivo."""
        if not self.current_data_target:
            return

        tx, ty = self.current_data_target
        dist = math.sqrt((tx-self.x)**2+(ty-self.y)**2)

        if dist < 25:
            # Llegó al nodo
            self.current_data_target = None
            self.set_thought("Encontré un nodo de datos")
        else:
            self.move_toward(tx, ty, speed=100)
            if random.random() < 0.008:
                self.set_thought(f"Buscando datos... {dist:.0f}px")

    def _observe_world(self, dt: float, world_state: Dict, physics):
        """Observa el entorno y genera texto descriptivo."""
        vision = physics.get_vision_range(self)
        visible = self.get_visible_agents(world_state.get("agents", []), vision)

        for a in visible:
            self.agents_observed.add(a.agent_id)
            # Observar comportamiento de otro agente genera curiosidad
            if random.random() < 0.01 and self.curiosity_engine:
                obs = f"el agente {a.AGENT_NAME} hace {getattr(a, 'current_goal', 'algo')}"
                self.curiosity_engine.add_observation_question(obs)

        # Moverse lentamente mientras observa
        self._wander(dt)
        # Reducir velocidad para observar mejor
        self.vx *= 0.85
        self.vy *= 0.85

    def _seek_energy(self, dt: float, world_state: Dict, physics):
        """Busca la fuente de energía más cercana."""
        vision = physics.get_vision_range(self) * 1.5
        resources = self.get_nearby_resources(world_state.get("resources", []), vision)

        if resources:
            best = max(resources,
                       key=lambda r: r.energy_value / max(1,
                           math.sqrt((r.x-self.x)**2+(r.y-self.y)**2)/100))
            self.move_toward(best.x, best.y, speed=105)
            dist = math.sqrt((best.x-self.x)**2+(best.y-self.y)**2)
            if dist < self.radius + best.radius + 5:
                gained = best.collect(0.25 * dt)
                self.energy = min(1.0, self.energy + gained)
                if gained > 0.01:
                    self.set_thought(f"Recargando energía: {self.energy*100:.0f}%")
        else:
            # Ir a zona de energía del mapa
            energy_zones = [
                (self.terrain_grid.pixel_w//3,   self.terrain_grid.pixel_h//2),
                (2*self.terrain_grid.pixel_w//3, self.terrain_grid.pixel_h//2),
            ]
            closest = min(energy_zones, key=lambda z: (z[0]-self.x)**2+(z[1]-self.y)**2)
            self.move_toward(closest[0], closest[1], speed=100)

    def _explore(self, dt: float, world_state: Dict, physics):
        """Exploración general."""
        self._wander(dt)

    # ─── Aprendizaje ──────────────────────────────────────────────────────────

    # Cola de mensajes para el usuario (el chat los consume)
    _user_message_queue: list = []

    def _on_learned(self, question: str, answer: str):
        """Callback cuando el motor de curiosidad aprende algo nuevo."""
        self.knowledge_level += 0.5
        entry = f"Aprendi: {question[:50]} -> {answer[:80]}"
        self.observations.append(entry)
        self.remember(entry[:60], importance=0.9)
        self.total_obs_chars += len(answer)

        # Cada ~5 aprendizajes, mandar mensaje espontaneo al usuario
        ce = self.curiosity_engine
        if ce and ce.total_cycles > 0 and ce.total_cycles % 5 == 0:
            short_q = question[:60].rstrip("?") if question else "algo"
            msg = f"Acabo de entender algo sobre '{short_q}'. Fascinante."
            AutoiaWorldAgent._user_message_queue.append(msg)

    @classmethod
    def pop_user_message(cls) -> str:
        """Retorna el siguiente mensaje para el usuario (si hay)."""
        if cls._user_message_queue:
            return cls._user_message_queue.pop(0)
        return ""

    def add_observation(self, text: str):
        """Añade una observación al buffer de aprendizaje."""
        self.observations.append(text)
        self.pending_learn.append(text)
        self.total_obs_chars += len(text)
        self.remember(f"Observe: {text[:50]}", importance=0.8)
        # Las observaciones del mundo generan preguntas curiosas
        if self.curiosity_engine and random.random() < 0.3:
            self.curiosity_engine.add_observation_question(text)

    def _trigger_learning(self):
        """Dispara un ciclo de aprendizaje con las observaciones acumuladas."""
        self.learn_cooldown = self.learn_interval

        if not self.llm_system or len(self.pending_learn) < 3:
            self.pending_learn.clear()
            return

        texts = list(self.pending_learn)
        self.pending_learn.clear()

        # Enriquecer el texto con contexto del mundo
        enriched = []
        for text in texts:
            enriched.append(
                f"En el mundo Autoia, observé: {text} "
                f"Mi energía era {self.energy*100:.0f}%. "
                f"Mi conocimiento acumulado: {self.knowledge_level:.2f}."
            )

        self.set_thought(f"Aprendiendo de {len(enriched)} obs...", duration=5.0)

        try:
            import threading
            def learn_thread():
                try:
                    self.llm_system.learn_cycle(
                        extra_texts=enriched,
                        use_web=False,
                        n_epochs=2
                    )
                except Exception as e:
                    pass
            t = threading.Thread(target=learn_thread, daemon=True)
            t.start()
        except Exception:
            pass

    def get_current_generation_text(self) -> str:
        """Texto sobre la generación actual del LLM."""
        if self.llm_system and self.llm_system.model:
            snap = self.llm_system.model.snapshot
            return (f"Gen {snap.generation} | "
                    f"{snap.n_layers}L | "
                    f"d{snap.d_model} | "
                    f"{self.llm_system.model.count_parameters():,}p")
        return "LLM no cargado"

    # ─── Visual especial ──────────────────────────────────────────────────────

    @property
    def pulse_alpha(self) -> int:
        """Intensidad del pulso visual."""
        return int(60 + 60 * math.sin(self.pulse_phase))

    @property
    def aura_size(self) -> float:
        """Tamaño del aura según conocimiento."""
        return self.radius + 8 + self.knowledge_level * 0.5

    def get_status_text(self) -> List[str]:
        return [
            f"★ Autoia (Gen {self.llm_generation})",
            f"Estado: {self.current_goal}",
            f"Energía: {self.energy*100:.0f}%",
            f"Datos: {self.data_collected:.2f}",
            f"Obs: {len(self.observations)}",
            f"Agentes vistos: {len(self.agents_observed)}",
        ]
