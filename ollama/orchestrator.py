"""
Orquestador de modelos Ollama.
Asigna un modelo Ollama específico a cada ROL del mundo.
Autoia usa su propio LLM (entrenado desde cero) — esto es para todo lo demás.

Roles periféricos que usan Ollama:
  - narrator    → Narra los eventos del mundo en tiempo real
  - npc_mind    → Genera pensamientos/decisiones de los NPCs
  - event_gen   → Genera eventos inesperados en el mundo
  - env_desc    → Describe el entorno (clima, terreno, atmósfera)
  - lore        → Genera lore/historia de fondo del mundo
  - law_judge   → Juzga si se violó alguna ley del mundo
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ollama.client import OllamaClient, OllamaModel, GenerateRequest

logger = logging.getLogger("autoia.ollama.orchestrator")


# Preferencias de modelos por rol (orden de prioridad)
ROLE_MODEL_PREFERENCES = {
    "narrator":  ["llama3", "llama2", "mistral", "gemma", "phi", "qwen", "deepseek", "tinyllama"],
    "npc_mind":  ["phi", "tinyllama", "gemma", "llama3", "mistral", "qwen", "llama2"],
    "event_gen": ["mistral", "llama3", "gemma", "phi", "qwen", "llama2", "tinyllama"],
    "env_desc":  ["gemma", "llama3", "mistral", "phi", "qwen", "llama2", "tinyllama"],
    "lore":      ["llama3", "mistral", "gemma", "qwen", "llama2", "phi", "tinyllama"],
    "law_judge": ["llama3", "mistral", "gemma", "phi", "qwen", "llama2", "tinyllama"],
}

# Parámetros por defecto por rol
ROLE_PARAMS = {
    "narrator":  {"temperature": 0.8, "max_tokens": 80},
    "npc_mind":  {"temperature": 0.9, "max_tokens": 60},
    "event_gen": {"temperature": 1.0, "max_tokens": 100},
    "env_desc":  {"temperature": 0.6, "max_tokens": 70},
    "lore":      {"temperature": 0.85, "max_tokens": 120},
    "law_judge": {"temperature": 0.3, "max_tokens": 60},
}


@dataclass
class RoleAssignment:
    """Asignación de un modelo Ollama a un rol."""
    role: str
    model: str
    temperature: float
    max_tokens: int
    calls_made: int = 0
    total_latency_ms: float = 0.0
    errors: int = 0
    last_output: str = ""

    @property
    def avg_latency_ms(self) -> float:
        if self.calls_made == 0:
            return 0.0
        return self.total_latency_ms / self.calls_made


@dataclass
class OllamaConfig:
    """Configuración de la integración Ollama."""
    host: str = "localhost"
    port: int = 11434
    enabled: bool = True

    # Asignaciones manuales (si el usuario quiere forzar un modelo por rol)
    # Ejemplo: {"narrator": "llama3:8b", "npc_mind": "phi3:mini"}
    manual_assignments: Dict[str, str] = field(default_factory=dict)

    # Cooldowns para no saturar Ollama
    narrator_cooldown:  float = 8.0    # Segundos entre narraciones
    npc_mind_cooldown:  float = 15.0   # Segundos entre pensamientos NPC
    event_gen_cooldown: float = 20.0   # Segundos entre eventos generados
    env_desc_cooldown:  float = 30.0   # Segundos entre descripciones de entorno


class OllamaOrchestrator:
    """
    Gestiona los modelos Ollama asignados a cada rol periférico del mundo.

    Si Ollama no está disponible → degrada gracefully (modos fallback sin LLM).
    Si un modelo falla → intenta con el siguiente disponible.
    """

    def __init__(self, config: Optional[OllamaConfig] = None,
                 save_dir: str = "logs"):
        self.config   = config or OllamaConfig()
        self.client   = OllamaClient(self.config.host, self.config.port)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.roles:     Dict[str, RoleAssignment] = {}
        self.available: bool = False
        self.models:    List[OllamaModel] = []

        self._setup_log_path = self.save_dir / "ollama_roles.json"
        self._initialized    = False

    def initialize(self) -> bool:
        """
        Detecta Ollama, lista modelos y asigna roles automáticamente.
        Retorna True si Ollama está disponible y se asignó al menos 1 rol.
        """
        logger.info("Conectando a Ollama local...")
        self.available = self.client.is_available()

        if not self.available:
            logger.warning(
                "Ollama no disponible en "
                f"{self.config.host}:{self.config.port}. "
                "Los sistemas periféricos funcionarán en modo fallback."
            )
            return False

        self.models = self.client.list_models()
        if not self.models:
            logger.warning("Ollama disponible pero sin modelos instalados.")
            return False

        model_names = [m.name for m in self.models]
        logger.info(f"Ollama detectado — {len(model_names)} modelos: {model_names}")

        # Asignar modelos a roles
        for role, prefs in ROLE_MODEL_PREFERENCES.items():
            # Prioridad 1: asignación manual del config
            manual = self.config.manual_assignments.get(role)
            if manual and manual in model_names:
                chosen = manual
            else:
                chosen = self.client.find_best_model(prefs)

            if chosen:
                params = ROLE_PARAMS.get(role, {"temperature": 0.7, "max_tokens": 100})
                self.roles[role] = RoleAssignment(
                    role=role, model=chosen,
                    temperature=params["temperature"],
                    max_tokens=params["max_tokens"],
                )
                logger.info(f"  Rol '{role:12s}' -> {chosen}")
            else:
                logger.warning(f"  Rol '{role}' sin modelo asignado")

        self._save_assignments()
        self._initialized = True
        return bool(self.roles)

    # ─── Generación por rol ────────────────────────────────────────────────

    def generate(self, role: str, prompt: str,
                 system: str = "", **kwargs) -> str:
        """
        Genera texto para un rol específico.
        Si falla, retorna string vacío sin romper la simulación.
        """
        if not self.available or role not in self.roles:
            return ""

        assignment = self.roles[role]
        req = GenerateRequest(
            model=assignment.model,
            prompt=prompt,
            system=system,
            temperature=kwargs.get("temperature", assignment.temperature),
            max_tokens=kwargs.get("max_tokens", assignment.max_tokens),
            stop=kwargs.get("stop", ["\n\n", "###"]),
        )

        t0 = time.time()
        response = self.client.generate(req)
        latency = (time.time() - t0) * 1000

        assignment.calls_made += 1
        assignment.total_latency_ms += latency

        if response.ok:
            assignment.last_output = response.text
            logger.debug(
                f"[{role}] {assignment.model} → "
                f"{len(response.text)} chars en {latency:.0f}ms"
            )
            return response.text
        else:
            assignment.errors += 1
            logger.debug(f"[{role}] Error: {response.error}")
            return ""

    def generate_async(self, role: str, prompt: str,
                       callback, system: str = "", **kwargs):
        """Genera en background. No bloquea la simulación."""
        if not self.available or role not in self.roles:
            return

        assignment = self.roles[role]
        req = GenerateRequest(
            model=assignment.model,
            prompt=prompt,
            system=system,
            temperature=kwargs.get("temperature", assignment.temperature),
            max_tokens=kwargs.get("max_tokens", assignment.max_tokens),
            stop=kwargs.get("stop", ["\n\n", "###"]),
        )

        def _cb(response):
            assignment.calls_made += 1
            assignment.total_latency_ms += response.latency_ms
            if response.ok:
                assignment.last_output = response.text
                callback(response.text)

        self.client.generate_async(req, _cb)

    # ─── Consultas ────────────────────────────────────────────────────────

    def get_role_model(self, role: str) -> Optional[str]:
        """Retorna el modelo asignado a un rol."""
        if role in self.roles:
            return self.roles[role].model
        return None

    def get_status(self) -> Dict:
        """Estado completo de la orquestación."""
        return {
            "available":    self.available,
            "host":         f"{self.config.host}:{self.config.port}",
            "models_total": len(self.models),
            "roles_active": len(self.roles),
            "roles": {
                role: {
                    "model":       a.model,
                    "calls":       a.calls_made,
                    "avg_latency": f"{a.avg_latency_ms:.0f}ms",
                    "errors":      a.errors,
                }
                for role, a in self.roles.items()
            }
        }

    def get_ui_lines(self) -> List[str]:
        """Líneas de texto para mostrar en el UI."""
        if not self.available:
            return ["Ollama: no disponible", "Modo fallback activo"]

        lines = [f"Ollama: {len(self.models)} modelos"]
        for role, assignment in self.roles.items():
            model_short = assignment.model.split(":")[0][:12]
            calls = assignment.calls_made
            lines.append(f"  {role[:10]:10s} → {model_short} ({calls} calls)")
        return lines

    # ─── Persistencia ──────────────────────────────────────────────────────

    def _save_assignments(self):
        """Guarda las asignaciones para referencia/debug."""
        data = {
            "host":   f"{self.config.host}:{self.config.port}",
            "models": [m.name for m in self.models],
            "roles":  {
                role: {"model": a.model, "temperature": a.temperature}
                for role, a in self.roles.items()
            }
        }
        with open(self._setup_log_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Asignaciones Ollama guardadas en {self._setup_log_path}")

    def load_manual_config(self, path: str) -> bool:
        """
        Carga asignaciones manuales desde un JSON.
        Formato: {"narrator": "llama3:8b", "npc_mind": "phi3:mini", ...}
        """
        try:
            with open(path) as f:
                self.config.manual_assignments = json.load(f)
            logger.info(f"Configuración manual cargada desde {path}")
            return True
        except Exception as e:
            logger.warning(f"No se pudo cargar config manual: {e}")
            return False
