"""
Motor de mente para NPCs — powered by Ollama.
Genera pensamientos, decisiones y diálogos de los agentes NPC.
Autoia tiene su propio LLM. El resto de los agentes usan Ollama para "pensar".
"""

import time
import random
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger("autoia.ollama.npc_mind")


# ─── Personalidades de sistema por tipo de agente ─────────────────────────────

AGENT_SYSTEMS = {
    "Explorador": (
        "Eres el Explorador, un agente IA curioso e intrépido en un mundo 2D con leyes físicas. "
        "Hablas en primera persona, en español, con emoción y curiosidad. Máximo 15 palabras."
    ),
    "Recolector": (
        "Eres el Recolector, un agente IA metódico y eficiente. Hablas en primera persona, "
        "en español, de forma analítica y calculadora. Máximo 15 palabras."
    ),
    "Guardián": (
        "Eres el Guardián, un agente IA protector y territorial. Hablas en primera persona, "
        "en español, con seriedad y determinación. Máximo 15 palabras."
    ),
    "Vagabundo": (
        "Eres el Vagabundo, un agente IA impredecible y algo caótico. Hablas en primera persona, "
        "en español, de forma errática y sorpresiva. Máximo 15 palabras."
    ),
    "Depredador": (
        "Eres el Depredador, un agente IA estratégico y agresivo. Hablas en primera persona, "
        "en español, con frialdad calculada. Máximo 15 palabras."
    ),
    "Estudioso": (
        "Eres el Estudioso, un agente IA intelectual y metódico. Hablas en primera persona, "
        "en español, con precisión académica. Máximo 15 palabras."
    ),
}


@dataclass
class ThoughtRequest:
    """Solicitud de pensamiento para un NPC."""
    agent_name: str
    agent_id: int
    state: str
    energy: float
    visible_agents: List[str]
    current_terrain: str
    is_day: bool
    extra_context: str = ""


class NPCMindEngine:
    """
    Motor de pensamiento para los NPCs del mundo.
    Cada NPC puede pedir un pensamiento a Ollama de forma asíncrona.
    Los pensamientos se muestran como burbujas en el mundo.

    Rate limit: cada NPC puede pedir máximo 1 pensamiento cada N segundos.
    """

    def __init__(self, orchestrator, cooldown_per_agent: float = 20.0):
        self.orchestrator        = orchestrator
        self.cooldown_per_agent  = cooldown_per_agent
        self._last_call: Dict[int, float] = {}    # agent_id → timestamp
        self._pending: Dict[int, bool]    = {}    # agent_id → si hay petición pendiente
        self.total_thoughts      = 0

    def request_thought(self, agent, world_state: Dict,
                        callback) -> bool:
        """
        Solicita un pensamiento para el agente dado.
        Llama a callback(thought_text) cuando está listo.
        Retorna True si se lanzó la petición, False si está en cooldown.
        """
        agent_id = agent.agent_id
        now = time.time()

        # Cooldown per-agente
        last = self._last_call.get(agent_id, 0)
        if now - last < self.cooldown_per_agent:
            return False

        if self._pending.get(agent_id, False):
            return False

        # Si Ollama no está disponible, usar fallback
        if not self.orchestrator.available or "npc_mind" not in self.orchestrator.roles:
            thought = self._fallback_thought(agent)
            if thought:
                callback(thought)
            return False

        # Construir contexto
        request = self._build_request(agent, world_state)
        prompt  = self._build_prompt(request)
        system  = AGENT_SYSTEMS.get(agent.AGENT_NAME, AGENT_SYSTEMS["Vagabundo"])

        self._last_call[agent_id] = now
        self._pending[agent_id]   = True

        def _cb(text: str):
            self._pending[agent_id] = False
            if text:
                self.total_thoughts += 1
                text = self._clean_thought(text, agent.AGENT_NAME)
                callback(text)

        self.orchestrator.generate_async(
            role="npc_mind",
            prompt=prompt,
            system=system,
            max_tokens=40,
            temperature=0.9,
            stop=["\n", ".", "?", "!"],
            callback=_cb,
        )
        return True

    def _build_request(self, agent, world_state: Dict) -> ThoughtRequest:
        from world.physics import TERRAIN_PROPS
        visible = [
            a.AGENT_NAME for a in world_state.get("agents", [])
            if a is not agent and a.alive and
            abs(a.x-agent.x) < 200 and abs(a.y-agent.y) < 200
        ]
        terrain_name = "desconocido"
        if hasattr(agent, "current_terrain") and agent.current_terrain:
            terrain_name = TERRAIN_PROPS[agent.current_terrain].name

        state = getattr(agent, "current_goal", getattr(agent, "state", ""))
        return ThoughtRequest(
            agent_name=agent.AGENT_NAME,
            agent_id=agent.agent_id,
            state=state,
            energy=agent.energy,
            visible_agents=visible[:4],
            current_terrain=terrain_name,
            is_day=world_state.get("is_day", True),
        )

    def _build_prompt(self, req: ThoughtRequest) -> str:
        visible_str = (", ".join(req.visible_agents)
                       if req.visible_agents else "nadie")
        energy_pct = int(req.energy * 100)
        hora = "día" if req.is_day else "noche"

        return (
            f"Estoy en '{req.current_terrain}'. "
            f"Mi energía: {energy_pct}%. Es de {hora}. "
            f"Estoy en estado '{req.state}'. "
            f"Veo: {visible_str}. "
            f"¿Qué pienso ahora? (una frase corta, máximo 12 palabras):"
        )

    def _clean_thought(self, text: str, agent_name: str) -> str:
        """Limpia y recorta el pensamiento generado."""
        text = text.strip().strip('"').strip("'").strip("*")
        # Eliminar el nombre del agente si aparece al inicio
        for prefix in [f"{agent_name}:", "Yo:", "Pienso:", "Creo:"]:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        # Máximo ~60 chars para la burbuja
        if len(text) > 60:
            text = text[:57] + "..."
        return text

    def _fallback_thought(self, agent) -> Optional[str]:
        """Pensamiento de fallback sin Ollama."""
        state = getattr(agent, "current_goal", getattr(agent, "state", ""))
        fallbacks = {
            "Explorador":  ["¡Territorio nuevo!", "¿Qué habrá allá?", "Debo mapear esto."],
            "Recolector":  ["Calculando ruta óptima.", "Energía detectada.", "Eficiencia máxima."],
            "Guardián":    ["Nadie pasa.", "Mi territorio, mis reglas.", "Vigilando."],
            "Vagabundo":   ["¿Por qué no?", "¡Aquí también!", "Quizás allá..."],
            "Depredador":  ["Pronto...","Debilidad detectada.", "Paciencia."],
            "Estudioso":   ["Fascinante.", "Hipótesis confirmada.", "Datos insuficientes."],
        }
        options = fallbacks.get(agent.AGENT_NAME, ["..."])
        return random.choice(options)

    def get_stats(self) -> Dict:
        return {
            "total_thoughts": self.total_thoughts,
            "pending":        sum(1 for v in self._pending.values() if v),
            "agents_tracked": len(self._last_call),
        }
