"""
Auto-crecimiento de la arquitectura del LLM.
Decide cuándo y cómo escalar el modelo según su rendimiento.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GrowthDecision:
    """Decisión de crecimiento del modelo."""
    should_grow: bool
    reason: str
    add_layers: int = 0
    scale_d_model: bool = False
    expand_vocab: bool = False
    details: Dict = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class ArchitectureEvolver:
    """
    Decide si el modelo debe crecer y cómo hacerlo.

    Estrategia de crecimiento:
    1. Si el loss está en plateau → añadir capas
    2. Si el vocabulario nuevo es >20% del actual → expandir vocab
    3. Si el modelo satura su capacidad → aumentar d_model
    """

    def __init__(self, config, log_dir: str = "logs"):
        self.config = config
        self.evo_config = config.evolution
        self.model_config = config.model
        self.log_dir = Path(log_dir)
        self.growth_history: List[Dict] = []
        self._load_history()

    def _load_history(self):
        path = self.log_dir / "growth_history.json"
        if path.exists():
            with open(path) as f:
                self.growth_history = json.load(f)

    def _save_history(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with open(self.log_dir / "growth_history.json", "w") as f:
            json.dump(self.growth_history, f, indent=2)

    def analyze(self, metrics, model_snapshot, corpus_size: int,
                 tokenizer_vocab: int) -> GrowthDecision:
        """
        Analiza el estado actual y decide si/cómo crecer.

        Args:
            metrics: TrainingMetrics con historial de loss
            model_snapshot: ModelSnapshot con arquitectura actual
            corpus_size: Número de documentos en el corpus
            tokenizer_vocab: Tamaño actual del vocabulario

        Returns:
            GrowthDecision con la recomendación
        """
        # Verificar límites máximos
        at_max_layers = model_snapshot.n_layers >= self.model_config.max_layers
        at_max_d_model = model_snapshot.d_model >= self.model_config.max_d_model
        at_max_vocab = tokenizer_vocab >= self.model_config.max_vocab_size

        if at_max_layers and at_max_d_model and at_max_vocab:
            return GrowthDecision(
                should_grow=False,
                reason="Arquitectura en capacidad máxima configurada"
            )

        reasons = []

        # 1. Analizar plateau en val_loss
        loss_plateau = self._detect_plateau(metrics)
        if loss_plateau and not at_max_layers:
            reasons.append("loss_plateau")

        # 2. Verificar si el loss es suficientemente bajo para justificar más capacidad
        current_val_loss = metrics.val_loss if metrics.val_loss < float("inf") else metrics.train_loss
        high_loss = current_val_loss > self.evo_config.scale_loss_threshold

        # 3. ¿Corpus creció significativamente?
        corpus_ratio = corpus_size / max(model_snapshot.training_steps / 10, 100)
        corpus_grew = corpus_ratio > 2.0
        if corpus_grew:
            reasons.append("corpus_growth")

        # 4. ¿Vocabulario nuevo supera umbral?
        vocab_ratio = tokenizer_vocab / self.model_config.vocab_size
        vocab_needs_expand = vocab_ratio > 1.2 and not at_max_vocab
        if vocab_needs_expand:
            reasons.append("vocab_expansion_needed")

        # Decisión final
        if not reasons:
            return GrowthDecision(
                should_grow=False,
                reason=f"No se requiere crecimiento (val_loss={current_val_loss:.4f})",
            )

        # Determinar tipo de crecimiento
        add_layers = 0
        scale_d_model = False
        expand_vocab = False

        if "loss_plateau" in reasons and not at_max_layers:
            add_layers = 2
        if "corpus_growth" in reasons and high_loss and not at_max_d_model:
            scale_d_model = True
        if "vocab_expansion_needed" in reasons:
            expand_vocab = True

        decision = GrowthDecision(
            should_grow=True,
            reason=", ".join(reasons),
            add_layers=add_layers,
            scale_d_model=scale_d_model,
            expand_vocab=expand_vocab,
            details={
                "current_val_loss": current_val_loss,
                "loss_plateau": loss_plateau,
                "corpus_size": corpus_size,
                "current_layers": model_snapshot.n_layers,
                "current_d_model": model_snapshot.d_model,
                "generation": model_snapshot.generation,
            }
        )

        logger.info(f"[ARCHITECT] Decisión de crecimiento: {decision.reason}")
        return decision

    def _detect_plateau(self, metrics) -> bool:
        """
        Detecta si el modelo está en plateau.
        Compara las últimas N épocas con las N anteriores.
        """
        history = metrics.history
        patience = self.evo_config.scale_plateau_patience

        if len(history) < patience * 2:
            return False

        recent = history[-patience:]
        prev = history[-patience*2:-patience]

        recent_losses = [h["val_loss"] for h in recent if h["val_loss"] < float("inf")]
        prev_losses = [h["val_loss"] for h in prev if h["val_loss"] < float("inf")]

        if not recent_losses or not prev_losses:
            return False

        recent_avg = sum(recent_losses) / len(recent_losses)
        prev_avg = sum(prev_losses) / len(prev_losses)

        # Plateau si mejora < min_improvement
        improvement = (prev_avg - recent_avg) / max(prev_avg, 1e-8)
        is_plateau = improvement < self.config.training.min_improvement

        logger.debug(f"[ARCHITECT] Plateau check: mejora={improvement:.4f}, "
                     f"plateau={is_plateau}")
        return is_plateau

    def apply_growth(self, model, tokenizer, decision: GrowthDecision,
                     corpus_texts: List[str] = None):
        """
        Aplica la decisión de crecimiento al modelo.
        Retorna (nuevo_modelo, nuevo_tokenizer).
        """
        import time

        record = {
            "timestamp": time.time(),
            "generation": model.snapshot.generation,
            "reason": decision.reason,
            "before": {
                "layers": model.snapshot.n_layers,
                "d_model": model.snapshot.d_model,
                "params": model.count_parameters(),
            }
        }

        new_model = model
        new_tokenizer = tokenizer

        # 1. Expandir vocabulario si es necesario
        if decision.expand_vocab and corpus_texts:
            new_vocab_size = min(
                int(tokenizer.actual_vocab_size * 1.5),
                self.model_config.max_vocab_size
            )
            logger.info(f"[ARCHITECT] Expandiendo vocab: {tokenizer.actual_vocab_size} -> {new_vocab_size}")
            tokenizer.retrain_expand(corpus_texts, new_vocab_size)
            new_tokenizer = tokenizer

        # 2. Crecer el modelo
        if decision.add_layers > 0 or decision.scale_d_model:
            target_layers = None
            target_d_model = None

            if decision.add_layers > 0:
                target_layers = model.snapshot.n_layers + decision.add_layers

            if decision.scale_d_model:
                target_d_model = min(
                    int(model.snapshot.d_model * self.evo_config.scale_factor),
                    self.model_config.max_d_model
                )

            new_model = model.grow(
                target_layers=target_layers,
                target_d_model=target_d_model
            )

        record["after"] = {
            "layers": new_model.snapshot.n_layers,
            "d_model": new_model.snapshot.d_model,
            "params": new_model.count_parameters(),
        }

        self.growth_history.append(record)
        self._save_history()

        logger.info(
            f"[ARCHITECT] Crecimiento aplicado: "
            f"{record['before']['params']:,} -> {record['after']['params']:,} parámetros"
        )

        return new_model, new_tokenizer

    def growth_report(self) -> str:
        """Genera un reporte del historial de crecimiento."""
        if not self.growth_history:
            return "No hay historial de crecimiento aún."

        lines = ["=== Historial de Crecimiento ==="]
        for i, record in enumerate(self.growth_history):
            before = record["before"]
            after = record["after"]
            lines.append(
                f"Gen {record['generation']+1}: "
                f"Capas {before['layers']}->{after['layers']} | "
                f"d_model {before['d_model']}->{after['d_model']} | "
                f"Params {before['params']:,}->{after['params']:,} | "
                f"Razón: {record['reason']}"
            )
        return "\n".join(lines)
