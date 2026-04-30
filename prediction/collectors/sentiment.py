"""
Recolector de sentimiento de masas.

Fuentes:
1. Ollama (LLM local) para analisis de sentimiento de texto
2. Wikipedia como proxy de relevancia/popularidad
3. Analisis de patrones textuales (sin API externa)

Genera señales de Operaciones Motivadoras (MOs) a partir del sentimiento.
"""

import time
import json
import threading
import logging
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("autoia.prediction.sentiment")


SENTIMENT_SYSTEM = """Eres un analizador de sentimiento conductual especializado en masas, mercados y deportes.
Analiza el texto y responde SOLO con un JSON en este formato exacto:
{"score": 0.7, "function": "tangible", "mo_type": "EO", "keywords": ["keyword1"], "summary": "resumen breve"}

score: -1.0 (muy negativo) a 1.0 (muy positivo)
function: "tangible" | "escape" | "attention" | "automatic"
mo_type: "EO" (aumenta deseo) | "AO" (disminuye deseo)
keywords: lista de 3 palabras clave
summary: una frase corta del sentimiento principal"""


@dataclass
class SentimentSignal:
    """Señal de sentimiento procesada."""
    source:      str
    text:        str
    score:       float       # -1.0 a 1.0
    function:    str         # funcion conductual detectada
    mo_type:     str         # "EO" o "AO"
    keywords:    List[str]
    summary:     str
    domain:      str
    timestamp:   float = field(default_factory=time.time)
    confidence:  float = 0.5

    @property
    def is_positive(self) -> bool:
        return self.score > 0.1

    @property
    def is_strong(self) -> bool:
        return abs(self.score) > 0.5


class SentimentCollector:
    """
    Recolecta y analiza sentimiento de masas para un dominio.

    Sin APIs externas: usa Ollama local + patrones de texto.
    Con Ollama: analisis LLM profundo del texto.
    """

    # Patrones de sentimiento por palabras clave (fallback sin LLM)
    POSITIVE_WORDS = {
        "general":  ["sube", "gana", "victoria", "record", "exito", "positivo",
                     "crece", "supera", "lidera", "domina", "gol", "campeon"],
        "market":   ["alcista", "bullish", "compra", "rentable", "beneficio",
                     "dividendo", "expansion", "crecimiento", "inversion"],
        "sports":   ["gol", "victoria", "campeon", "invicto", "forma", "titular",
                     "recuperado", "confianza", "racha"],
        "betting":  ["favorito", "cuota baja", "confianza", "handicap positivo"],
        "masses":   ["viral", "tendencia", "popular", "apoyo", "entusiasmo", "fomo"],
    }

    NEGATIVE_WORDS = {
        "general":  ["baja", "pierde", "derrota", "crisis", "negativo", "cae",
                     "fracasa", "lesion", "sancion", "escandalo"],
        "market":   ["bajista", "bearish", "venta", "perdida", "quiebra",
                     "recesion", "inflacion", "riesgo", "volatil"],
        "sports":   ["lesion", "suspension", "derrota", "mal momento", "baja",
                     "sin gol", "presion", "crisis", "expulsado"],
        "betting":  ["outsider", "cuota alta", "dudas", "incertidumbre"],
        "masses":   ["panico", "miedo", "incertidumbre", "abandono", "critica"],
    }

    def __init__(self, orchestrator=None, domain: str = "general"):
        self.orchestrator = orchestrator
        self.domain       = domain
        self.signals:     deque = deque(maxlen=500)
        self._lock        = threading.Lock()
        self.total_analyzed = 0

    def analyze_text(self, text: str, source: str = "manual",
                     callback: Optional[Callable] = None) -> Optional[SentimentSignal]:
        """
        Analiza el sentimiento de un texto.
        Si Ollama disponible: usa LLM. Si no: usa patrones.
        callback(signal) se llama cuando termina (async si LLM).
        """
        if self.orchestrator and self.orchestrator.available:
            self._analyze_with_llm(text, source, callback)
            return None
        else:
            signal = self._analyze_with_patterns(text, source)
            if callback:
                callback(signal)
            return signal

    def _analyze_with_llm(self, text: str, source: str,
                          callback: Optional[Callable]):
        """Analisis profundo via Ollama (async)."""
        # Usar cualquier rol disponible
        roles = ["lore", "env_desc", "narrator", "law_judge"]
        role = next((r for r in roles
                     if r in self.orchestrator.roles), None)
        if not role:
            signal = self._analyze_with_patterns(text, source)
            if callback:
                callback(signal)
            return

        prompt = f"Texto a analizar:\n{text[:500]}\n\nDominio: {self.domain}"

        def _cb(response_text: str):
            signal = self._parse_llm_response(response_text, text, source)
            if signal:
                with self._lock:
                    self.signals.append(signal)
                    self.total_analyzed += 1
                if callback:
                    callback(signal)

        self.orchestrator.generate_async(
            role=role,
            prompt=prompt,
            system=SENTIMENT_SYSTEM,
            max_tokens=150,
            temperature=0.2,
            callback=_cb,
        )

    def _parse_llm_response(self, response: str, original: str,
                             source: str) -> Optional[SentimentSignal]:
        """Parsea respuesta JSON del LLM."""
        try:
            # Buscar JSON en la respuesta
            start = response.find("{")
            end   = response.rfind("}") + 1
            if start == -1 or end == 0:
                return self._analyze_with_patterns(original, source)
            data = json.loads(response[start:end])
            return SentimentSignal(
                source=source,
                text=original[:200],
                score=float(data.get("score", 0.0)),
                function=data.get("function", "automatic"),
                mo_type=data.get("mo_type", "EO"),
                keywords=data.get("keywords", []),
                summary=data.get("summary", ""),
                domain=self.domain,
                confidence=0.85,
            )
        except Exception:
            return self._analyze_with_patterns(original, source)

    def _analyze_with_patterns(self, text: str, source: str) -> SentimentSignal:
        """Analisis por patrones de palabras clave (fallback)."""
        text_lower = text.lower()
        pos_words = (self.POSITIVE_WORDS.get(self.domain, [])
                     + self.POSITIVE_WORDS["general"])
        neg_words = (self.NEGATIVE_WORDS.get(self.domain, [])
                     + self.NEGATIVE_WORDS["general"])

        pos_count = sum(1 for w in pos_words if w in text_lower)
        neg_count = sum(1 for w in neg_words if w in text_lower)
        total = pos_count + neg_count

        if total == 0:
            score = 0.0
            function = "automatic"
            mo_type = "EO"
        else:
            score = (pos_count - neg_count) / total
            # Inferir funcion
            if any(w in text_lower for w in ["ganancia", "precio", "beneficio",
                                              "gol", "punto", "rentable"]):
                function = "tangible"
            elif any(w in text_lower for w in ["riesgo", "perdida", "crisis",
                                                "lesion", "miedo"]):
                function = "escape"
            elif any(w in text_lower for w in ["viral", "tendencia", "fomo",
                                                "popular", "social"]):
                function = "attention"
            else:
                function = "automatic"

            mo_type = "EO" if score > 0 else "AO"

        found_pos = [w for w in pos_words if w in text_lower][:3]
        found_neg = [w for w in neg_words if w in text_lower][:3]
        keywords = (found_pos + found_neg)[:3]

        signal = SentimentSignal(
            source=source,
            text=text[:200],
            score=round(score, 3),
            function=function,
            mo_type=mo_type,
            keywords=keywords,
            summary=f"{'Positivo' if score > 0 else 'Negativo'} ({score:+.2f})",
            domain=self.domain,
            confidence=0.5,
        )
        with self._lock:
            self.signals.append(signal)
            self.total_analyzed += 1
        return signal

    def get_aggregate_sentiment(self, window_h: float = 24.0) -> Dict:
        """
        Agrega las ultimas señales en una metrica de sentimiento global.
        Retorna el 'estado conductual' actual del dominio.
        """
        now = time.time()
        cutoff = now - window_h * 3600
        recent = [s for s in self.signals if s.timestamp >= cutoff]

        if not recent:
            return {
                "score": 0.0, "mo_type": "neutral",
                "function": "automatic", "n_signals": 0,
                "trend": "sin datos",
            }

        # Promedio ponderado (mas peso a señales recientes)
        weights = [(now - s.timestamp) for s in recent]
        max_age = max(weights) if weights else 1
        weights = [1.0 - (w / max_age) * 0.8 for w in weights]

        weighted_score = sum(s.score * w for s, w in zip(recent, weights))
        total_w = sum(weights) or 1
        avg_score = weighted_score / total_w

        # Funcion dominante
        func_counts: Dict[str, int] = {}
        for s in recent:
            func_counts[s.function] = func_counts.get(s.function, 0) + 1
        dominant_func = max(func_counts, key=func_counts.get)

        # Tendencia (ultimas 5 vs anteriores)
        if len(recent) >= 6:
            recent_half = [s.score for s in recent[-5:]]
            older_half  = [s.score for s in recent[:-5]]
            trend_delta = sum(recent_half)/len(recent_half) - sum(older_half)/len(older_half)
            trend = "acelerando" if trend_delta > 0.1 else (
                    "desacelerando" if trend_delta < -0.1 else "estable")
        else:
            trend = "insuficientes datos"

        # MO predominante
        eo_count = sum(1 for s in recent if s.mo_type == "EO")
        ao_count = len(recent) - eo_count
        dominant_mo = "EO" if eo_count > ao_count else "AO"

        return {
            "score":     round(avg_score, 3),
            "mo_type":   dominant_mo,
            "function":  dominant_func,
            "n_signals": len(recent),
            "trend":     trend,
            "keywords":  self._get_top_keywords(recent),
        }

    def _get_top_keywords(self, signals: List[SentimentSignal]) -> List[str]:
        counts: Dict[str, int] = {}
        for s in signals:
            for kw in s.keywords:
                counts[kw] = counts.get(kw, 0) + 1
        return sorted(counts, key=counts.get, reverse=True)[:5]
