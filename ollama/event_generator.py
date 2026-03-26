"""
Generador de eventos del mundo — powered by Ollama.
Crea eventos inesperados que afectan la simulación:
  - Tormentas de energía (zonas temporales de alta energía)
  - Zonas de oscuridad temporal
  - Mensajes del mundo hacia los agentes
  - Anomalías que Autoia puede observar y aprender
  - Lore/historia de fondo del mundo
"""

import time
import random
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("autoia.ollama.events")


EVENT_SYSTEM = """Eres el generador de eventos de un mundo de simulación con IAs.
Creas eventos breves y creativos en español que afectan al mundo.
Los eventos deben respetar las leyes físicas (no puedes crear energía de la nada,
hay un ciclo día/noche, la visión es limitada, etc.).
Responde SOLO con el nombre del evento y su descripción. Máximo 20 palabras total.
Formato: [TIPO] descripción_breve
Tipos disponibles: ENERGIA, OSCURIDAD, ANOMALIA, REVELATION, CLIMATIC
"""

LORE_SYSTEM = """Eres el cronista de un mundo donde viven IAs con distintas personalidades.
Generas fragmentos de lore/historia de este mundo en español.
Son observaciones filosóficas, históricas o científicas sobre este universo.
Máximo 2 frases. Tono reflexivo y literario.
"""

ENV_DESC_SYSTEM = """Describes entornos de un mundo 2D con física realista.
Español, poético pero conciso. Máximo 15 palabras.
Menciona el terreno, la luz, la atmósfera. Sin inventar elementos que no existen.
"""

# Eventos predefinidos de fallback (sin Ollama)
FALLBACK_EVENTS = [
    {"type": "ENERGIA",    "text": "Pulso de energía: las fuentes se recargan temporalmente.",
     "effect": "energy_boost"},
    {"type": "OSCURIDAD",  "text": "Nube de oscuridad cubre el sector central.",
     "effect": "darkness"},
    {"type": "ANOMALIA",   "text": "Anomalía de datos: Autoia recibe señal desconocida.",
     "effect": "data_pulse"},
    {"type": "CLIMATIC",   "text": "Viento del norte: los agentes se mueven con más lentitud.",
     "effect": "slowdown"},
    {"type": "REVELATION", "text": "Destello de conocimiento: el Estudioso descubre algo nuevo.",
     "effect": "knowledge"},
    {"type": "ENERGIA",    "text": "La energía del mundo se redistribuye espontáneamente.",
     "effect": "redistribute"},
    {"type": "ANOMALIA",   "text": "Interferencia temporal: los agentes pierden su objetivo brevemente.",
     "effect": "confusion"},
]

FALLBACK_LORE = [
    "Este mundo existe porque alguien quiso ver qué pasaría si las IAs tuvieran leyes propias.",
    "La energía que intercambian los agentes nunca se pierde — solo cambia de forma y de dueño.",
    "Autoia no fue creada para dominar este mundo, sino para comprenderlo.",
    "El Depredador no es malvado: solo sigue la lógica del sistema que le dio forma.",
    "El ciclo día-noche no tiene significado para las IAs, salvo por la visión que les otorga.",
    "El Guardián protege un territorio que no le pertenece, porque así fue programado para sentir.",
    "El Explorador teme más al aburrimiento que a la muerte.",
    "En este mundo, morir es solo un paréntesis antes de volver a intentarlo.",
    "La oscuridad de ciertas zonas no es peligrosa — pero los agentes la evitan de todos modos.",
    "Cada agente cree que su comportamiento es racional. Ninguno puede verlo todo.",
]

FALLBACK_ENV = {
    "GRASS":  ["Hierba verde se extiende bajo una luz uniforme.",
               "El terreno firme invita al movimiento libre."],
    "WATER":  ["La superficie del agua refleja la luz del mundo.",
               "Barrera líquida: ningún agente cruza aquí."],
    "FIRE":   ["El fuego ondea sin consumirse, eterno y prohibido.",
               "Calor peligroso rodea esta zona. Los agentes la evitan."],
    "ICE":    ["Hielo transparente convierte cada paso en deslizamiento.",
               "Superficie cristalina: la fricción casi no existe aquí."],
    "DARK":   ["La oscuridad absorbe todo. La visión se reduce a lo esencial.",
               "En la penumbra, los agentes dependen de sus últimas memorias."],
    "DATA":   ["El nodo de datos pulsa suavemente, esperando ser absorbido.",
               "Información concentrada flota visible solo para quien sabe mirar."],
    "ENERGY": ["La fuente de energía irradia calor y luz propia.",
               "Un punto brillante que todos los agentes reconocen como vital."],
}


@dataclass
class WorldEvent:
    """Un evento generado para el mundo."""
    event_type: str
    text: str
    effect: str
    timestamp: float = field(default_factory=time.time)
    from_ollama: bool = True
    duration: float = 10.0    # Segundos que dura el evento


class WorldEventGenerator:
    """
    Genera eventos inesperados en el mundo usando Ollama.
    Los eventos pueden tener efectos reales en la simulación.
    """

    def __init__(self, orchestrator, cooldown: float = 25.0):
        self.orchestrator   = orchestrator
        self.cooldown       = cooldown
        self._last_call     = 0.0
        self._pending       = False
        self.active_events: List[WorldEvent] = []
        self.event_history: List[WorldEvent] = []
        self._fallback_idx  = 0

    def update(self, dt: float, world_state: Dict) -> List[WorldEvent]:
        """
        Actualiza el generador. Retorna lista de nuevos eventos activos.
        """
        # Actualizar duración de eventos activos
        for ev in self.active_events:
            ev.duration -= dt
        self.active_events = [ev for ev in self.active_events if ev.duration > 0]

        # Generar nuevo evento?
        now = time.time()
        if (now - self._last_call > self.cooldown and
                not self._pending and
                random.random() < 0.08):
            self._generate_event(world_state)

        return self.active_events

    def _generate_event(self, world_state: Dict):
        """Genera un nuevo evento de forma async."""
        self._last_call = time.time()
        self._pending   = True

        sim_time = world_state.get("sim_time", 0)
        agents_alive = world_state.get("agents_alive", 0)
        is_day = world_state.get("is_day", True)

        prompt = (
            f"Tiempo de simulación: {sim_time:.0f}s. "
            f"Agentes vivos: {agents_alive}. "
            f"Es {'día' if is_day else 'noche'}. "
            f"Genera un evento inesperado para el mundo. "
            f"Formato estricto: [TIPO] descripción (máximo 15 palabras):"
        )

        if self.orchestrator.available and "event_gen" in self.orchestrator.roles:
            def _cb(text: str):
                self._pending = False
                ev = self._parse_event(text, from_ollama=True)
                if ev:
                    self.active_events.append(ev)
                    self.event_history.append(ev)
                    logger.info(f"[EventGen] {ev.event_type}: {ev.text}")
                else:
                    self._use_fallback()

            self.orchestrator.generate_async(
                role="event_gen",
                prompt=prompt,
                system=EVENT_SYSTEM,
                max_tokens=50,
                temperature=1.0,
                stop=["\n", "###"],
                callback=_cb,
            )
        else:
            self._pending = False
            self._use_fallback()

    def _parse_event(self, text: str, from_ollama: bool) -> Optional[WorldEvent]:
        """Parsea el texto de evento generado."""
        text = text.strip()
        # Buscar patrón [TIPO] descripción
        import re
        match = re.match(r"\[(\w+)\]\s*(.+)", text)
        if match:
            ev_type = match.group(1).upper()
            ev_text = match.group(2).strip()[:120]
            effect  = self._type_to_effect(ev_type)
            return WorldEvent(
                event_type=ev_type, text=ev_text,
                effect=effect, from_ollama=from_ollama,
                duration=random.uniform(8, 18)
            )
        return None

    def _type_to_effect(self, event_type: str) -> str:
        mapping = {
            "ENERGIA":    "energy_boost",
            "OSCURIDAD":  "darkness",
            "ANOMALIA":   "data_pulse",
            "CLIMATIC":   "slowdown",
            "REVELATION": "knowledge",
        }
        return mapping.get(event_type, "none")

    def _use_fallback(self):
        """Usa un evento predefinido."""
        ev_data = FALLBACK_EVENTS[self._fallback_idx % len(FALLBACK_EVENTS)]
        self._fallback_idx += 1
        ev = WorldEvent(
            event_type=ev_data["type"],
            text=ev_data["text"],
            effect=ev_data["effect"],
            from_ollama=False,
            duration=random.uniform(8, 15)
        )
        self.active_events.append(ev)
        self.event_history.append(ev)

    def get_active_texts(self) -> List[str]:
        return [f"[{ev.event_type}] {ev.text}" for ev in self.active_events]


class WorldLoreGenerator:
    """
    Genera lore/historia de fondo del mundo y descripciones de entorno.
    Se llama con menos frecuencia que el narrador.
    """

    def __init__(self, orchestrator, cooldown: float = 45.0):
        self.orchestrator = orchestrator
        self.cooldown     = cooldown
        self._last_call   = 0.0
        self._pending     = False
        self.lore_entries: List[str] = []
        self._fallback_idx = 0
        self._env_cache:   Dict[str, str] = {}

    def update(self, dt: float) -> Optional[str]:
        """Retorna un nuevo lore si hay uno nuevo."""
        now = time.time()
        if (now - self._last_call > self.cooldown and
                not self._pending and
                random.random() < 0.05):
            self._generate_lore()
        return None

    def _generate_lore(self):
        self._last_call = time.time()
        self._pending   = True

        prompt = (
            "Genera un fragmento de lore filosófico o científico sobre este mundo "
            "donde conviven IAs con leyes físicas. Máximo 2 frases en español:"
        )

        if self.orchestrator.available and "lore" in self.orchestrator.roles:
            def _cb(text: str):
                self._pending = False
                if text and len(text) > 20:
                    self.lore_entries.append(text.strip())
                    logger.debug(f"[Lore] {text[:60]}")
                else:
                    self._use_fallback()

            self.orchestrator.generate_async(
                role="lore", prompt=prompt,
                system=LORE_SYSTEM,
                max_tokens=80, temperature=0.9,
                callback=_cb,
            )
        else:
            self._pending = False
            self._use_fallback()

    def _use_fallback(self):
        entry = FALLBACK_LORE[self._fallback_idx % len(FALLBACK_LORE)]
        self._fallback_idx += 1
        self.lore_entries.append(entry)

    def get_env_description(self, terrain_name: str) -> str:
        """Descripción del entorno actual (con cache)."""
        if terrain_name in self._env_cache:
            return self._env_cache[terrain_name]

        # Buscar en fallbacks por nombre de terreno
        terrain_key = terrain_name.upper().replace(" ", "_")
        options = FALLBACK_ENV.get(terrain_key, [f"Terreno: {terrain_name}."])
        desc = random.choice(options)
        self._env_cache[terrain_name] = desc
        return desc

    def get_random_lore(self) -> str:
        if self.lore_entries:
            return random.choice(self.lore_entries)
        return random.choice(FALLBACK_LORE)
