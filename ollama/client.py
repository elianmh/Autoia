"""
Cliente para la API local de Ollama.
Se conecta a http://localhost:11434 (puerto por defecto de Ollama).

NO se usa para Autoia (que tiene su propio LLM desde cero).
Se usa para los sistemas periféricos del mundo: NPCs, narrador, eventos.
"""

import json
import time
import logging
import threading
import urllib.request
import urllib.error
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass, field
from queue import Queue, Empty

logger = logging.getLogger("autoia.ollama")


@dataclass
class OllamaModel:
    """Información de un modelo disponible en Ollama."""
    name: str
    size_gb: float = 0.0
    family: str = ""
    parameters: str = ""


@dataclass
class GenerateRequest:
    """Parámetros de una petición de generación."""
    model: str
    prompt: str
    system: str = ""
    temperature: float = 0.7
    max_tokens: int = 150
    stop: List[str] = field(default_factory=list)
    stream: bool = False


@dataclass
class GenerateResponse:
    """Respuesta de una petición de generación."""
    text: str
    model: str
    done: bool = True
    latency_ms: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


class OllamaClient:
    """
    Cliente HTTP para la API de Ollama local.
    Síncrono, sin dependencias externas (solo stdlib).

    La API de Ollama es simple REST en localhost:11434:
      POST /api/generate  → genera texto
      POST /api/chat      → chat con historial
      GET  /api/tags      → lista modelos instalados
    """

    DEFAULT_HOST = "localhost"
    DEFAULT_PORT = 11434
    REQUEST_TIMEOUT = 30   # segundos

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
        self.base_url = f"http://{host}:{port}"
        self._available: Optional[bool] = None
        self._models_cache: Optional[List[OllamaModel]] = None
        self._cache_time: float = 0.0
        self._cache_ttl: float = 30.0   # Refrescar lista cada 30s

    # ─── Conectividad ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Comprueba si Ollama está corriendo localmente."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                self._available = resp.status == 200
        except Exception:
            self._available = False
        return self._available

    def list_models(self, use_cache: bool = True) -> List[OllamaModel]:
        """Lista todos los modelos instalados en Ollama."""
        now = time.time()
        if (use_cache and self._models_cache is not None and
                now - self._cache_time < self._cache_ttl):
            return self._models_cache

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            models = []
            for m in data.get("models", []):
                details = m.get("details", {})
                models.append(OllamaModel(
                    name=m["name"],
                    size_gb=m.get("size", 0) / 1e9,
                    family=details.get("family", ""),
                    parameters=details.get("parameter_size", ""),
                ))

            self._models_cache = models
            self._cache_time = now
            return models

        except Exception as e:
            logger.warning(f"No se pudo listar modelos Ollama: {e}")
            return []

    def get_model_names(self) -> List[str]:
        """Lista solo los nombres de los modelos disponibles."""
        return [m.name for m in self.list_models()]

    # ─── Generación ────────────────────────────────────────────────────────

    def generate(self, request: GenerateRequest) -> GenerateResponse:
        """
        Genera texto con el modelo especificado.
        Síncrono, bloquea hasta obtener respuesta.
        """
        t0 = time.time()

        payload = {
            "model":  request.model,
            "prompt": request.prompt,
            "stream": False,
            "options": {
                "temperature":  request.temperature,
                "num_predict":  request.max_tokens,
                "stop":         request.stop or [],
            }
        }
        if request.system:
            payload["system"] = request.system

        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())

            latency = (time.time() - t0) * 1000
            return GenerateResponse(
                text=data.get("response", "").strip(),
                model=request.model,
                done=data.get("done", True),
                latency_ms=latency,
            )

        except urllib.error.URLError as e:
            return GenerateResponse(
                text="", model=request.model,
                error=f"Ollama no disponible: {e.reason}"
            )
        except Exception as e:
            return GenerateResponse(
                text="", model=request.model,
                error=str(e)
            )

    def generate_async(self, request: GenerateRequest,
                       callback, error_callback=None):
        """
        Genera texto en un thread separado para no bloquear la simulación.
        Llama a callback(response) cuando termina.
        """
        def _run():
            response = self.generate(request)
            if response.ok:
                try:
                    callback(response)
                except Exception as e:
                    logger.debug(f"Error en callback Ollama: {e}")
            elif error_callback:
                try:
                    error_callback(response.error)
                except Exception:
                    pass

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    def chat(self, model: str, messages: List[Dict],
             temperature: float = 0.7, max_tokens: int = 200) -> GenerateResponse:
        """
        API de chat con historial de mensajes.
        messages = [{"role": "user"|"assistant"|"system", "content": "..."}]
        """
        t0 = time.time()
        payload = {
            "model":    model,
            "messages": messages,
            "stream":   False,
            "options":  {"temperature": temperature, "num_predict": max_tokens},
        }
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=body, method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=self.REQUEST_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())

            text = data.get("message", {}).get("content", "").strip()
            latency = (time.time() - t0) * 1000
            return GenerateResponse(text=text, model=model,
                                    done=data.get("done", True), latency_ms=latency)
        except Exception as e:
            return GenerateResponse(text="", model=model, error=str(e))

    # ─── Utilidades ────────────────────────────────────────────────────────

    def find_best_model(self, preferred: List[str]) -> Optional[str]:
        """
        Encuentra el mejor modelo disponible de una lista de preferencias.
        Retorna el primero que esté instalado, o None.
        """
        available = self.get_model_names()
        for pref in preferred:
            for avail in available:
                # Match por prefijo (ej: "llama3" matchea "llama3:8b")
                if avail.startswith(pref) or pref in avail:
                    return avail
        # Si hay alguno, retornar el primero
        return available[0] if available else None

    def __repr__(self) -> str:
        models = self.get_model_names()
        return f"OllamaClient({self.base_url}, {len(models)} modelos: {models[:3]})"
