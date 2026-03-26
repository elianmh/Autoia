"""
Narrador del mundo — powered by Ollama.
Genera descripciones en tiempo real de lo que ocurre en el mundo.
NO es parte de Autoia. Es la "voz" que describe el mundo alrededor de ella.
"""

import time
import logging
import random
from collections import deque
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("autoia.ollama.narrator")


NARRATOR_SYSTEM = """Eres el narrador de un mundo de simulación donde coexisten varias IAs.
Describes en español lo que ocurre de forma breve, evocadora y literaria.
Una o dos frases máximo. No uses emojis. No repitas lo que ya dijiste.
El mundo tiene leyes físicas reales: los agentes no pueden crearla energía de la nada,
tienen visión limitada, mueren de inanición, respetan la velocidad máxima.
"""

FALLBACK_NARRATIONS = [
    "El mundo sigue su ciclo imperturbable, cada agente siguiendo su naturaleza.",
    "La energía fluye entre los seres del mundo, de unos a otros, nunca creándose ni destruyéndose.",
    "Bajo las leyes que rigen este universo, cada acción tiene su consecuencia.",
    "El tiempo avanza y los agentes aprenden, aunque sea sin saberlo.",
    "Autoia observa el mundo que la rodea, absorbiendo silenciosamente cada detalle.",
    "El ciclo día-noche marca el ritmo de un mundo que nunca descansa.",
    "Entre la luz y la sombra, los agentes buscan su propósito.",
    "La energía no desaparece: cambia de manos, de forma, de agente en agente.",
    "El Depredador acecha desde las sombras. El Guardián vigila su territorio.",
    "El Explorador traza en su mente un mapa que nadie más puede ver.",
    "Cada muerte es solo el inicio de un nuevo ciclo en este mundo sin fin.",
]


@dataclass
class Narration:
    """Una narración del mundo."""
    text: str
    trigger: str           # Qué la generó
    timestamp: float
    from_ollama: bool = True


class WorldNarrator:
    """
    Genera narración del mundo usando Ollama de forma asíncrona.
    Nunca bloquea la simulación.
    Si Ollama no responde, usa narraciones de fallback.
    """

    def __init__(self, orchestrator, cooldown: float = 8.0):
        self.orchestrator  = orchestrator
        self.cooldown      = cooldown
        self._last_call    = 0.0
        self._pending      = False
        self.history: deque = deque(maxlen=30)
        self.current_text  = ""
        self.current_timer = 0.0
        self.display_duration = 6.0
        self._used_fallbacks: set = set()

    def update(self, dt: float, world_state: Dict, agents: list):
        """Actualiza el narrador. Llamar en cada tick de simulación."""
        # Cuenta regresiva de display
        if self.current_timer > 0:
            self.current_timer -= dt

        # Cooldown entre naraciones
        now = time.time()
        if now - self._last_call < self.cooldown or self._pending:
            return

        # Decidir si generar narración
        if random.random() < 0.15:   # 15% de probabilidad por tick de evaluación
            self._trigger_narration(world_state, agents)

    def _trigger_narration(self, world_state: Dict, agents: list):
        """Dispara una narración async."""
        self._last_call = time.time()

        # Construir contexto para el prompt
        prompt = self._build_prompt(world_state, agents)

        if self.orchestrator.available and "narrator" in self.orchestrator.roles:
            self._pending = True
            self.orchestrator.generate_async(
                role="narrator",
                prompt=prompt,
                system=NARRATOR_SYSTEM,
                max_tokens=80,
                temperature=0.85,
                callback=self._on_narration_received,
            )
        else:
            # Fallback sin Ollama
            self._use_fallback("no_ollama")

    def _build_prompt(self, world_state: Dict, agents: list) -> str:
        """Construye el prompt de narración a partir del estado actual."""
        alive = [a for a in agents if a.alive]
        is_day = world_state.get("is_day", True)
        sim_time = world_state.get("sim_time", 0)

        # Resumir situaciones interesantes
        situations = []
        for a in alive[:6]:
            state = getattr(a, "current_goal", getattr(a, "state", ""))
            if state in ("chasing", "intercepting", "absorbing_data", "studying"):
                situations.append(f"el {a.AGENT_NAME} está {state}")

        # Eventos recientes (si hay)
        recent_events = world_state.get("recent_events", [])
        event_str = ""
        if recent_events:
            last = recent_events[-1]
            event_str = f"Último evento: {last}. "

        # Hora del día
        phase = "de día" if is_day else "de noche"

        history_recent = [n.text for n in list(self.history)[-3:]]
        history_str = " | ".join(history_recent) if history_recent else "inicio"

        prompt = (
            f"Es {phase}. Tiempo simulado: {sim_time:.0f}s. "
            f"Hay {len(alive)} agentes vivos. "
            f"{event_str}"
            f"Situaciones: {', '.join(situations[:3]) or 'exploración tranquila'}. "
            f"Narración previa: {history_str}. "
            f"Genera UNA frase de narración nueva y diferente:"
        )
        return prompt

    def _on_narration_received(self, text: str):
        """Callback cuando Ollama responde."""
        self._pending = False
        if text and len(text) > 10:
            # Limpiar el texto
            text = text.strip().strip('"').strip("'")
            # No repetir narraciones muy similares
            if not self._is_too_similar(text):
                narration = Narration(
                    text=text, trigger="ollama",
                    timestamp=time.time(), from_ollama=True
                )
                self.history.append(narration)
                self.current_text = text
                self.current_timer = self.display_duration
                logger.debug(f"[Narrador] {text}")
            else:
                self._use_fallback("repetition")
        else:
            self._use_fallback("empty_response")

    def _use_fallback(self, reason: str = ""):
        """Usa una narración pre-escrita de fallback."""
        self._pending = False
        # Elegir una no usada recientemente
        available = [i for i in range(len(FALLBACK_NARRATIONS))
                     if i not in self._used_fallbacks]
        if not available:
            self._used_fallbacks.clear()
            available = list(range(len(FALLBACK_NARRATIONS)))

        import random
        idx = random.choice(available)
        self._used_fallbacks.add(idx)
        text = FALLBACK_NARRATIONS[idx]

        narration = Narration(
            text=text, trigger=f"fallback:{reason}",
            timestamp=time.time(), from_ollama=False
        )
        self.history.append(narration)
        self.current_text = text
        self.current_timer = self.display_duration

    def _is_too_similar(self, new_text: str) -> bool:
        """Verifica si la nueva narración es muy similar a una reciente."""
        new_words = set(new_text.lower().split())
        for narration in list(self.history)[-3:]:
            old_words = set(narration.text.lower().split())
            if len(new_words) == 0:
                return True
            overlap = len(new_words & old_words) / len(new_words)
            if overlap > 0.7:
                return True
        return False

    @property
    def is_displaying(self) -> bool:
        return self.current_timer > 0 and bool(self.current_text)

    @property
    def display_alpha(self) -> int:
        """Alpha de fade-in/out."""
        ratio = self.current_timer / self.display_duration
        if ratio > 0.8:
            return 255
        elif ratio > 0.1:
            return int(255 * ratio / 0.8)
        return int(255 * ratio / 0.1)

    def force_narration(self, context: str):
        """Fuerza una narración inmediata (p.ej. al morir Autoia)."""
        self._last_call = 0  # Resetear cooldown
        self._trigger_narration({"sim_time": 0, "is_day": True}, [])

    def get_recent(self, n: int = 5) -> List[str]:
        return [n.text for n in list(self.history)[-n:]]
