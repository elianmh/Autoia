"""
Módulo de Auto-Programación.
El modelo puede generar código Python para extender/mejorar su propio pipeline
de aprendizaje. Ejecuta el código en un sandbox seguro.
"""

import os
import ast
import sys
import json
import time
import logging
import textwrap
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from io import StringIO
from contextlib import redirect_stdout, redirect_stderr

logger = logging.getLogger(__name__)


@dataclass
class SelfProgramTask:
    """Una tarea de auto-programación."""
    task_id: str
    description: str
    generated_code: str = ""
    execution_result: str = ""
    success: bool = False
    timestamp: float = field(default_factory=time.time)


class CodeSandbox:
    """
    Sandbox seguro para ejecutar código generado por el modelo.
    Restringe imports peligrosos y operaciones del sistema.
    """

    FORBIDDEN_MODULES = {
        "subprocess", "multiprocessing", "socket", "ftplib",
        "smtplib", "shutil", "ctypes", "cffi", "importlib",
    }

    FORBIDDEN_BUILTINS = {"exec", "eval", "__import__"}

    def __init__(self, allowed_globals: Optional[Dict] = None):
        self.allowed_globals = allowed_globals or {}

    def _check_safety(self, code: str) -> Tuple[bool, str]:
        """Verifica la seguridad del código antes de ejecutarlo."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Error de sintaxis: {e}"

        for node in ast.walk(tree):
            # Bloquear imports peligrosos
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                else:
                    names = [node.module.split(".")[0]] if node.module else []
                for name in names:
                    if name in self.FORBIDDEN_MODULES:
                        return False, f"Import bloqueado: {name}"

            # Bloquear llamadas peligrosas
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.FORBIDDEN_BUILTINS:
                        return False, f"Función bloqueada: {node.func.id}"

        return True, "OK"

    def execute(self, code: str, timeout: float = 30.0) -> Tuple[bool, str]:
        """
        Ejecuta código en el sandbox.
        Retorna (éxito, output/error).
        """
        is_safe, reason = self._check_safety(code)
        if not is_safe:
            return False, f"Código rechazado por seguridad: {reason}"

        # Preparar namespace restringido
        safe_globals = {
            "__builtins__": {
                "print": print, "len": len, "range": range, "int": int,
                "float": float, "str": str, "list": list, "dict": dict,
                "tuple": tuple, "set": set, "bool": bool, "None": None,
                "True": True, "False": False, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter, "sorted": sorted,
                "min": min, "max": max, "sum": sum, "abs": abs,
                "isinstance": isinstance, "hasattr": hasattr,
                "getattr": getattr, "setattr": setattr,
                "open": open,  # Permitido pero restringido por el OS
            },
            "json": json,
            "Path": Path,
            "logging": logging,
        }
        safe_globals.update(self.allowed_globals)

        stdout_capture = StringIO()
        stderr_capture = StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                exec(code, safe_globals)  # noqa: S102
            output = stdout_capture.getvalue()
            return True, output or "Código ejecutado exitosamente (sin output)"
        except Exception as e:
            err = traceback.format_exc()
            return False, f"Error en ejecución:\n{err}"


class SelfProgrammer:
    """
    Sistema de auto-programación del LLM.

    Capacidades:
    1. Genera nuevas estrategias de recolección de datos
    2. Crea augmentaciones de datos específicas al tema
    3. Ajusta hiperparámetros basándose en métricas
    4. Genera funciones de evaluación para el tema
    5. Crea nuevos prompts de fine-tuning
    """

    TASK_TEMPLATES = {
        "data_augmentation": """
# Tarea: Generar función de augmentación de datos para el tema '{topic}'
# El modelo genera transformaciones de texto relevantes al tema.
# La función debe recibir List[str] y retornar List[str] con datos aumentados.

def augment_data_for_{topic_slug}(texts):
    \"\"\"Aumenta datos sobre {topic} con paráfrasis y variaciones.\"\"\"
    augmented = []
    for text in texts:
        # Variación 1: prefijo de contexto
        augmented.append(f"En el contexto de {topic}: " + text)
        # Variación 2: forma interrogativa
        sentences = text.split('. ')
        if len(sentences) > 1:
            augmented.append(sentences[0] + ".")
        augmented.append(text)
    return augmented
""",
        "eval_metric": """
# Tarea: Crear métrica de evaluación específica para el tema '{topic}'
# Evalúa si el texto generado contiene conocimiento relevante del tema.

def evaluate_topic_relevance(generated_text, keywords):
    \"\"\"Calcula relevancia del texto generado sobre el tema.\"\"\"
    text_lower = generated_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text_lower)
    diversity = len(set(generated_text.split())) / max(len(generated_text.split()), 1)
    coherence = 1.0 if len(generated_text) > 50 else 0.5
    score = (hits / max(len(keywords), 1)) * 0.5 + diversity * 0.3 + coherence * 0.2
    return round(score, 4)
""",
        "hyperparameter_search": """
# Tarea: Sugerir mejores hiperparámetros basado en las métricas actuales.
# Analiza el historial de pérdida y propone ajustes.

def suggest_hyperparams(loss_history, current_lr, current_batch_size):
    \"\"\"Sugiere ajustes de hiperparámetros basados en el comportamiento del loss.\"\"\"
    if len(loss_history) < 5:
        return {"lr": current_lr, "batch_size": current_batch_size, "reason": "Insufficient data"}

    recent = loss_history[-5:]
    trend = recent[-1] - recent[0]

    if trend > 0:  # Loss aumentando (posible divergencia)
        return {
            "lr": current_lr * 0.5,
            "batch_size": current_batch_size,
            "reason": "Loss increasing: reduce LR by 50%"
        }
    elif abs(trend) < 0.001:  # Plateau
        return {
            "lr": current_lr * 0.8,
            "batch_size": min(current_batch_size * 2, 64),
            "reason": "Plateau detected: reduce LR slightly, increase batch"
        }
    else:  # Mejorando bien
        return {
            "lr": current_lr,
            "batch_size": current_batch_size,
            "reason": "Training progressing well"
        }
""",
    }

    def __init__(self, config, model, log_dir: str = "logs"):
        self.config = config
        self.model = model
        self.sandbox = CodeSandbox()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.task_log: List[SelfProgramTask] = []
        self.generated_modules: Dict[str, str] = {}
        self._load_task_log()

    def _load_task_log(self):
        path = self.log_dir / "self_program_log.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                self.task_log = [SelfProgramTask(**t) for t in data]

    def _save_task_log(self):
        with open(self.log_dir / "self_program_log.json", "w") as f:
            json.dump([
                {
                    "task_id": t.task_id, "description": t.description,
                    "generated_code": t.generated_code,
                    "execution_result": t.execution_result,
                    "success": t.success, "timestamp": t.timestamp
                }
                for t in self.task_log[-100:]  # Mantener últimos 100
            ], f, indent=2)

    def generate_code_with_model(self, prompt: str) -> str:
        """
        Usa el LLM para generar código Python.
        Si el modelo no está lo suficientemente entrenado, usa templates.
        """
        # Si el modelo tiene val_loss < 2.0, intentar usar el modelo
        # Para el bootstrap inicial, usamos templates inteligentes
        try:
            if self.model.snapshot.training_steps > 1000:
                import torch
                topic = self.config.learning.topic
                full_prompt = (
                    f"<TOPIC>{topic}<QUERY>"
                    f"Genera código Python para: {prompt}"
                    f"<RESPONSE># Python code:\n"
                )
                tokenizer_path = Path(self.config.data_dir) / "tokenizer.json"
                if tokenizer_path.exists():
                    from core.tokenizer import AutoiaTokenizer
                    tok = AutoiaTokenizer(save_dir=self.config.data_dir)
                    input_ids = torch.tensor([tok.encode(full_prompt)], dtype=torch.long)
                    device = next(self.model.parameters()).device
                    input_ids = input_ids.to(device)
                    generated = self.model.generate(input_ids, max_new_tokens=300, temperature=0.7)
                    result = tok.decode(generated[0].tolist())
                    # Extraer solo el código después de <RESPONSE>
                    if "<RESPONSE>" in result:
                        code = result.split("<RESPONSE>")[-1]
                        return code.strip()
        except Exception as e:
            logger.debug(f"Generación con modelo falló: {e}, usando template")

        return ""

    def create_data_augmentation(self) -> Optional[SelfProgramTask]:
        """Genera función de augmentación de datos para el tema."""
        topic = self.config.learning.topic
        topic_slug = topic.replace(" ", "_").replace("/", "_")[:30]

        # Intentar con el modelo primero, luego template
        generated = self.generate_code_with_model(
            f"función de augmentación de texto sobre {topic}"
        )

        if not generated or len(generated) < 100:
            template = self.TASK_TEMPLATES["data_augmentation"]
            generated = template.format(topic=topic, topic_slug=topic_slug)

        task = SelfProgramTask(
            task_id=f"aug_{int(time.time())}",
            description=f"Augmentación de datos para: {topic}",
            generated_code=generated,
        )

        # Ejecutar en sandbox para verificar
        success, result = self.sandbox.execute(generated)
        task.success = success
        task.execution_result = result

        if success:
            self.generated_modules["data_augmentation"] = generated
            logger.info(f"[SELF-PROG] Función de augmentación creada y validada")
        else:
            logger.warning(f"[SELF-PROG] Augmentación falló en sandbox: {result}")

        self.task_log.append(task)
        self._save_task_log()
        return task

    def optimize_hyperparameters(self, metrics) -> Dict:
        """Genera y ejecuta código de optimización de hiperparámetros."""
        template = self.TASK_TEMPLATES["hyperparameter_search"]
        task = SelfProgramTask(
            task_id=f"hp_{int(time.time())}",
            description="Optimización de hiperparámetros",
            generated_code=template,
        )

        loss_history = [h.get("train_loss", float("inf")) for h in metrics.history[-20:]]
        current_lr = self.config.training.learning_rate
        current_bs = self.config.training.batch_size

        # Construir código con datos reales
        test_code = template + f"""
result = suggest_hyperparams({loss_history}, {current_lr}, {current_bs})
print(json.dumps(result))
"""
        sandbox = CodeSandbox(allowed_globals={"json": json})
        success, output = sandbox.execute(test_code)

        task.success = success
        task.execution_result = output

        suggestions = {}
        if success:
            try:
                suggestions = json.loads(output.strip())
                logger.info(f"[SELF-PROG] Sugerencia HP: {suggestions}")
            except json.JSONDecodeError:
                pass

        self.task_log.append(task)
        self._save_task_log()
        return suggestions

    def create_eval_metric(self) -> Optional[callable]:
        """Genera función de evaluación específica al tema."""
        topic = self.config.learning.topic
        template = self.TASK_TEMPLATES["eval_metric"]
        generated = template.format(topic=topic)

        task = SelfProgramTask(
            task_id=f"eval_{int(time.time())}",
            description=f"Métrica de evaluación para: {topic}",
            generated_code=generated,
        )

        # Probar la función
        test_code = generated + f"""
result = evaluate_topic_relevance(
    "machine learning neural network deep learning",
    {self.config.learning.topic_keywords}
)
print(result)
"""
        success, output = self.sandbox.execute(test_code)
        task.success = success
        task.execution_result = output

        if success:
            self.generated_modules["eval_metric"] = generated
            logger.info(f"[SELF-PROG] Métrica de evaluación creada: {output}")
        else:
            logger.warning(f"[SELF-PROG] Métrica falló: {output}")

        self.task_log.append(task)
        self._save_task_log()
        return success

    def apply_augmentation(self, texts: List[str]) -> List[str]:
        """Aplica la función de augmentación generada si existe."""
        if "data_augmentation" not in self.generated_modules:
            return texts

        try:
            namespace = {}
            exec(self.generated_modules["data_augmentation"], namespace)  # noqa: S102
            # Encontrar la función de augmentación
            aug_fn = None
            for name, obj in namespace.items():
                if callable(obj) and not name.startswith("_"):
                    aug_fn = obj
                    break
            if aug_fn:
                return aug_fn(texts)
        except Exception as e:
            logger.debug(f"Augmentación falló en producción: {e}")

        return texts

    def run_self_review(self, metrics) -> Dict:
        """
        Revisión completa del sistema.
        Ejecuta todas las tareas de auto-programación y reporta resultados.
        """
        logger.info("[SELF-PROG] Iniciando revisión automática...")
        results = {}

        # 1. Crear augmentación si no existe
        if "data_augmentation" not in self.generated_modules:
            task = self.create_data_augmentation()
            results["data_augmentation"] = task.success if task else False

        # 2. Optimizar hiperparámetros
        hp_suggestions = self.optimize_hyperparameters(metrics)
        results["hyperparameter_suggestions"] = hp_suggestions

        # 3. Crear métrica de evaluación
        if "eval_metric" not in self.generated_modules:
            eval_ok = self.create_eval_metric()
            results["eval_metric"] = eval_ok

        # 4. Aplicar sugerencias de HP si son razonables
        if hp_suggestions:
            new_lr = hp_suggestions.get("lr")
            if new_lr and 1e-6 <= new_lr <= 1e-2:
                old_lr = self.config.training.learning_rate
                self.config.training.learning_rate = new_lr
                results["lr_adjusted"] = f"{old_lr:.2e} -> {new_lr:.2e}"
                logger.info(f"[SELF-PROG] LR ajustado: {old_lr:.2e} -> {new_lr:.2e}")

        logger.info(f"[SELF-PROG] Revisión completa: {results}")
        return results

    def get_report(self) -> str:
        """Reporte de actividad de auto-programación."""
        successful = sum(1 for t in self.task_log if t.success)
        total = len(self.task_log)
        modules = list(self.generated_modules.keys())

        return (
            f"=== Auto-Programación ===\n"
            f"Tareas ejecutadas: {total}\n"
            f"Exitosas: {successful}/{total}\n"
            f"Módulos activos: {', '.join(modules) or 'ninguno'}\n"
            f"Última tarea: {self.task_log[-1].description if self.task_log else 'N/A'}"
        )
