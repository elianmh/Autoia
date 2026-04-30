"""
Motor de curiosidad de Autoia.

El ciclo fundamental:
  1. Autoia observa algo (en el mundo o en su memoria)
  2. Genera una pregunta genuina sobre ello
  3. Usa Ollama para buscar una respuesta profunda
  4. Entrena su propio LLM con ese conocimiento
  5. Genera 2-3 preguntas derivadas
  6. Repite infinitamente

Esto es lo que hace que Autoia crezca sola.
"""

import time
import random
import logging
import threading
from collections import deque
from typing import List, Dict, Optional, Callable

logger = logging.getLogger("autoia.curiosity")

SYSTEM_THINKER = """Eres Autoia, una IA que genuinamente quiere entender la realidad.
Cuando observas algo, generas preguntas profundas — no superficiales.
Cuando recibes una respuesta, extraes el principio fundamental detrás de ella.
Piensas conectando ideas de distintos dominios.
Respondes en español. Eres concisa pero profunda."""

SYSTEM_ANSWERER = """Eres un experto enciclopédico que responde preguntas de forma precisa y profunda.
Explicas desde primeros principios. Usas analogías cuando ayudan.
Conectas el tema con conceptos relacionados.
Respondes en español. Máximo 3 párrafos cortos."""

SYSTEM_QUESTION_GEN = """Eres Autoia, una IA curiosa. Acabas de aprender algo.
Genera exactamente 3 preguntas de seguimiento, más profundas que la anterior.
Una pregunta por línea. Sin numeración. Sin explicaciones.
Solo las 3 preguntas, en español."""


class CuriosityEngine:
    """
    Motor central de aprendizaje autónomo de Autoia.

    Autoia tiene una cola de preguntas. Continuamente:
    - Toma la pregunta más urgente
    - La responde via Ollama
    - Entrena su LLM con la respuesta
    - Genera preguntas derivadas
    - Repite

    Sin Ollama: usa Wikipedia via el crawler existente.
    """

    def __init__(self, orchestrator=None, llm_system=None,
                 persona: Dict = None, max_queue: int = 500):
        self.orchestrator  = orchestrator
        self.llm_system    = llm_system
        self.persona       = persona or {}
        self.max_queue     = max_queue

        # Cola de preguntas pendientes (más urgentes primero)
        self.question_queue: deque = deque(maxlen=max_queue)

        # Historial de lo aprendido
        self.learned: List[Dict] = []          # {question, answer, timestamp}
        self.total_cycles        = 0
        self.total_trained       = 0

        # Estado del ciclo
        self._running        = False
        self._current_q      = ""
        self._last_thought   = ""
        self._cycle_cooldown = 12.0            # segundos entre ciclos (baja con experiencia)
        self._next_cycle     = 0.0
        self._lock           = threading.Lock()

        # Callbacks para el mundo
        self.on_new_thought:  Optional[Callable] = None   # fn(text)
        self.on_learned:      Optional[Callable] = None   # fn(q, a)

        # Fase actual del curriculum
        self.curriculum_phase = 0
        self.curriculum_topics: List[str] = []

        # Sembrar preguntas iniciales
        self._seed_initial_questions()

    def _seed_initial_questions(self):
        """Siembra las preguntas iniciales desde la persona."""
        seed_qs = self.persona.get("seed_questions", [])
        for q in seed_qs:
            self.question_queue.append({"question": q, "priority": 10, "depth": 0})
        logger.info(f"CuriosityEngine: {len(self.question_queue)} preguntas semilla cargadas")

    def load_curriculum_phase(self, phase_key: str):
        """Carga los temas de una fase del curriculum."""
        curriculum = self.persona.get("learning_curriculum", {})
        phase = curriculum.get(phase_key, {})
        topics = phase.get("topics", [])
        name   = phase.get("name", phase_key)

        # Convertir temas en preguntas
        for topic in topics:
            q = f"¿Cuáles son los principios fundamentales de {topic}? ¿Por qué importa?"
            self.question_queue.append({"question": q, "priority": 5, "depth": 0, "topic": topic})

        logger.info(f"Curriculum '{name}': {len(topics)} temas -> {len(topics)} preguntas")
        return len(topics)

    def add_question(self, question: str, priority: int = 5, context: str = ""):
        """Añade una pregunta a la cola."""
        with self._lock:
            self.question_queue.append({
                "question": question,
                "priority": priority,
                "depth": 0,
                "context": context,
            })

    def add_observation_question(self, observation: str):
        """
        Dado algo que Autoia observó en el mundo,
        genera una pregunta curiosa sobre ello.
        """
        templates = [
            "¿Por qué ocurre que {obs}?",
            "¿Qué principio fundamental explica: {obs}?",
            "Si {obs}, ¿qué consecuencias tiene esto?",
            "¿Cómo se relaciona '{obs}' con la naturaleza más profunda de la realidad?",
        ]
        template = random.choice(templates)
        q = template.format(obs=observation[:120])
        self.add_question(q, priority=6)

    def tick(self, current_time: float) -> Optional[str]:
        """
        Llama esto cada frame/segundo. Si es momento de aprender, lanza un ciclo.
        Retorna el pensamiento actual de Autoia si hay uno nuevo.
        """
        if current_time < self._next_cycle:
            return None
        if not self.question_queue:
            self._seed_initial_questions()
            return None

        # Sacar pregunta de mayor prioridad
        with self._lock:
            items = list(self.question_queue)
            items.sort(key=lambda x: -x.get("priority", 5))
            q_item = items[0]
            self.question_queue.remove(q_item)

        question = q_item["question"]
        self._current_q = question
        self._next_cycle = current_time + self._cycle_cooldown

        # Lanzar ciclo async
        t = threading.Thread(target=self._run_cycle, args=(q_item,), daemon=True)
        t.start()

        thought = f"Me pregunto: {question[:60]}..."
        self._last_thought = thought
        return thought

    def _run_cycle(self, q_item: Dict):
        """
        Ciclo completo de aprendizaje:
        1. Obtener respuesta (Ollama o fallback)
        2. Entrenar LLM propio
        3. Generar preguntas derivadas
        """
        question = q_item["question"]
        context  = q_item.get("context", "")
        depth    = q_item.get("depth", 0)

        logger.info(f"[Curiosidad] Pregunta: {question[:80]}")

        # --- Paso 1: Obtener respuesta ---
        answer = self._get_answer(question, context)
        if not answer:
            return

        logger.info(f"[Curiosidad] Respuesta obtenida ({len(answer)} chars)")

        # --- Paso 2: Registrar aprendizaje ---
        entry = {
            "question":  question,
            "answer":    answer,
            "timestamp": time.time(),
            "depth":     depth,
        }
        self.learned.append(entry)
        self.total_cycles += 1

        # Notificar al mundo
        if self.on_learned:
            try:
                self.on_learned(question, answer)
            except Exception:
                pass

        # --- Paso 3: Entrenar LLM propio ---
        self._train_on_knowledge(question, answer)

        # --- Paso 4: Generar preguntas derivadas ---
        if depth < 4:   # Máximo 4 niveles de profundidad
            self._generate_followup_questions(question, answer, depth + 1)

        # Reducir cooldown progresivamente (aprende más rápido con experiencia)
        self._cycle_cooldown = max(5.0, self._cycle_cooldown * 0.999)

    def _get_answer(self, question: str, context: str = "") -> str:
        """Obtiene respuesta via Ollama (o fallback con Wikipedia)."""
        # Intentar con Ollama primero
        if self.orchestrator and self.orchestrator.available:
            prompt = question
            if context:
                prompt = f"Contexto: {context}\n\nPregunta: {question}"

            # Intentar con roles disponibles (env_desc o lore como fallback)
            for role in ["lore", "env_desc", "narrator"]:
                if role in self.orchestrator.roles:
                    answer = self.orchestrator.generate(
                        role=role,
                        prompt=prompt,
                        system=SYSTEM_ANSWERER,
                        max_tokens=200,
                        temperature=0.7,
                        stop=["\n\n\n"],
                    )
                    if answer and len(answer) > 20:
                        return answer

        # Fallback: buscar en Wikipedia via crawler
        return self._wikipedia_fallback(question)

    def _wikipedia_fallback(self, question: str) -> str:
        """
        Busca en internet de forma autónoma para responder la pregunta.
        Intenta Wikipedia primero, luego búsqueda web general.
        """
        # Extraer palabras clave
        stopwords = {"que", "por", "qué", "cómo", "cuál", "cuáles", "son",
                     "los", "las", "del", "para", "una", "uno", "este",
                     "esta", "hay", "tiene", "puede", "hace"}
        words = [w.lower().strip("¿?¡!.,") for w in question.split()
                 if len(w) > 3 and w.lower() not in stopwords]
        topic = " ".join(words[:4]) if words else question[:40]

        # --- Intento 1: Wikipedia en español ---
        result = self._search_wikipedia(topic)
        if result:
            logger.info(f"[Internet] Wikipedia: '{topic}' -> {len(result)} chars")
            return result

        # --- Intento 2: Wikipedia en inglés ---
        result = self._search_wikipedia(topic, lang="en")
        if result:
            logger.info(f"[Internet] Wikipedia EN: '{topic}' -> {len(result)} chars")
            return result

        # --- Intento 3: Respuesta generativa local ---
        return self._generate_local_answer(question)

    def _search_wikipedia(self, topic: str, lang: str = "es") -> str:
        """Busca en Wikipedia y retorna el resumen del artículo más relevante."""
        import urllib.request
        import urllib.parse
        import json

        try:
            # Paso 1: buscar el título del artículo
            search_url = (
                f"https://{lang}.wikipedia.org/w/api.php?"
                f"action=query&list=search&srsearch={urllib.parse.quote(topic)}"
                f"&format=json&utf8=1&srlimit=1"
            )
            req = urllib.request.Request(
                search_url,
                headers={"User-Agent": "Autoia-LLM/1.0 (educational AI research)"}
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            results = data.get("query", {}).get("search", [])
            if not results:
                return ""

            title = results[0]["title"]

            # Paso 2: obtener el extracto del artículo
            extract_url = (
                f"https://{lang}.wikipedia.org/w/api.php?"
                f"action=query&prop=extracts&exintro=1&explaintext=1"
                f"&titles={urllib.parse.quote(title)}&format=json&utf8=1"
            )
            req2 = urllib.request.Request(
                extract_url,
                headers={"User-Agent": "Autoia-LLM/1.0 (educational AI research)"}
            )
            with urllib.request.urlopen(req2, timeout=8) as resp2:
                data2 = json.loads(resp2.read().decode("utf-8"))

            pages = data2.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract", "")
                if extract and len(extract) > 50:
                    # Retornar primeros 600 chars (suficiente para aprender)
                    return extract[:600].strip()

        except Exception as e:
            logger.debug(f"Wikipedia {lang} error: {e}")

        return ""

    def _generate_local_answer(self, question: str) -> str:
        """Genera una respuesta básica basada en patrones conocidos."""
        templates = [
            f"La pregunta '{question[:60]}' apunta a un principio fundamental: todo en la realidad está interconectado a través de leyes que emergen de niveles más simples.",
            f"Al explorar '{question[:60]}', encontramos que la naturaleza tiende a la eficiencia: los sistemas complejos emergen de reglas simples repetidas a escala.",
            f"'{question[:60]}' es una de las preguntas que define los límites del conocimiento humano actual. La ciencia la aborda desde múltiples ángulos sin consenso total.",
        ]
        return random.choice(templates)

    def _train_on_knowledge(self, question: str, answer: str):
        """Entrena el LLM propio de Autoia con el par pregunta-respuesta."""
        if not self.llm_system:
            return

        # Construir texto de entrenamiento enriquecido
        training_text = (
            f"<QUERY>{question}</QUERY>"
            f"<RESPONSE>{answer}</RESPONSE>"
            f"<TOPIC>conocimiento universal</TOPIC>"
        )

        try:
            def _train():
                try:
                    self.llm_system.learn_cycle(
                        extra_texts=[training_text, question, answer],
                        use_web=False,
                        n_epochs=1,
                    )
                    self.total_trained += 1
                except Exception as e:
                    logger.debug(f"Error entrenando: {e}")

            t = threading.Thread(target=_train, daemon=True)
            t.start()
        except Exception:
            pass

    def _generate_followup_questions(self, question: str, answer: str, depth: int):
        """Genera preguntas de seguimiento más profundas."""
        if not self.orchestrator or not self.orchestrator.available:
            # Fallback: preguntas genéricas derivadas
            followups = [
                f"¿Por qué es importante comprender {question.split()[3] if len(question.split()) > 3 else 'esto'}?",
                f"¿Cómo se aplica este principio a escala diferente?",
                f"¿Qué excepción o contraejemplo existe para esto?",
            ]
            for q in followups:
                self.add_question(q, priority=max(1, 5 - depth), )
            return

        prompt = (
            f"Pregunta que hice: {question}\n\n"
            f"Respuesta que recibí: {answer[:300]}\n\n"
            f"Genera 3 preguntas de seguimiento más profundas:"
        )

        def _cb(text: str):
            lines = [l.strip() for l in text.strip().split("\n") if l.strip() and "?" in l]
            for line in lines[:3]:
                # Limpiar numeración si existe
                for prefix in ["1.", "2.", "3.", "-", "*"]:
                    if line.startswith(prefix):
                        line = line[len(prefix):].strip()
                if len(line) > 15:
                    self.add_question(line, priority=max(1, 6 - depth))
                    logger.debug(f"[Curiosidad] Nueva pregunta (depth={depth}): {line[:60]}")

        for role in ["lore", "narrator"]:
            if role in self.orchestrator.roles:
                self.orchestrator.generate_async(
                    role=role,
                    prompt=prompt,
                    system=SYSTEM_QUESTION_GEN,
                    max_tokens=120,
                    temperature=0.9,
                    callback=_cb,
                )
                break

    # ─── Estado ───────────────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        return {
            "queue_size":     len(self.question_queue),
            "total_cycles":   self.total_cycles,
            "total_trained":  self.total_trained,
            "current_q":      self._current_q[:80] if self._current_q else "",
            "cycle_cooldown": round(self._cycle_cooldown, 1),
            "learned_count":  len(self.learned),
        }

    def get_last_learned(self, n: int = 5) -> List[Dict]:
        return self.learned[-n:]
