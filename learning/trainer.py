"""
Motor de entrenamiento del LLM.
Soporta entrenamiento inicial, fine-tuning continuo y monitoreo de métricas.
"""

import os
import json
import math
import time
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Métricas de una sesión de entrenamiento."""
    step: int = 0
    epoch: int = 0
    train_loss: float = float("inf")
    val_loss: float = float("inf")
    perplexity: float = float("inf")
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    tokens_per_second: float = 0.0
    history: List[Dict] = field(default_factory=list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.history.append({
            "step": self.step, "train_loss": self.train_loss,
            "val_loss": self.val_loss, "perplexity": self.perplexity,
        })

    def is_improving(self, patience: int = 5) -> bool:
        """Verifica si el modelo está mejorando en las últimas 'patience' épocas."""
        if len(self.history) < patience:
            return True
        recent = self.history[-patience:]
        losses = [h["val_loss"] for h in recent if h["val_loss"] < float("inf")]
        if len(losses) < 2:
            return True
        return losses[-1] < losses[0] - 0.005  # Mejora mínima de 0.5%

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "step": self.step, "epoch": self.epoch,
                "train_loss": self.train_loss, "val_loss": self.val_loss,
                "perplexity": self.perplexity, "history": self.history[-200:],
            }, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrainingMetrics":
        if not Path(path).exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        m = cls()
        for k, v in data.items():
            setattr(m, k, v)
        return m


class WarmupCosineScheduler:
    """Learning rate scheduler con warmup lineal + cosine decay."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int,
                 min_lr_ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def _get_lrs(self) -> List[float]:
        if self._step < self.warmup_steps:
            factor = self._step / max(self.warmup_steps, 1)
        else:
            progress = (self._step - self.warmup_steps) / max(
                self.max_steps - self.warmup_steps, 1
            )
            factor = self.min_lr_ratio + 0.5 * (1 - self.min_lr_ratio) * (
                1 + math.cos(math.pi * progress)
            )
        return [base_lr * factor for base_lr in self.base_lrs]

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


class AutoiaTrainer:
    """
    Entrenador principal del LLM.
    Gestiona: training loop, validación, checkpoints, early stopping.
    """

    def __init__(self, model, tokenizer, config, device: str = "cpu"):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.metrics = TrainingMetrics()

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._setup_optimizer()
        self._load_metrics()

    def _setup_optimizer(self):
        """Configura AdamW con weight decay selectivo."""
        # No aplicar weight decay a biases y LayerNorm
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "weight" in name and "norm" not in name and "embedding" not in name:
                decay_params.append(param)
            else:
                no_decay_params.append(param)

        self.optimizer = AdamW([
            {"params": decay_params, "weight_decay": 0.1},
            {"params": no_decay_params, "weight_decay": 0.0},
        ], lr=self.config.training.learning_rate, betas=(0.9, 0.95), eps=1e-8)

    def _load_metrics(self):
        metrics_path = self.log_dir / "metrics.json"
        if metrics_path.exists():
            self.metrics = TrainingMetrics.load(str(metrics_path))

    def train_epoch(self, dataloader, val_dataloader=None,
                    scheduler=None) -> float:
        """Entrena una época completa. Retorna loss promedio."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_idx, (input_ids, labels) in enumerate(dataloader):
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)

            # Reemplazar pad_id en labels con -100 (ignorar en loss)
            labels[labels == self.tokenizer.pad_id] = -100

            self.optimizer.zero_grad()
            out = self.model(input_ids, labels=labels)
            loss = out["loss"]

            loss.backward()

            # Gradient clipping
            grad_norm = nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clip
            )

            self.optimizer.step()
            if scheduler:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            self.metrics.step += 1
            self.model.snapshot.training_steps += 1

            # Log periódico
            if batch_idx % 50 == 0:
                elapsed = time.time() - t0
                tokens = (batch_idx + 1) * input_ids.shape[0] * input_ids.shape[1]
                tps = tokens / max(elapsed, 1)
                logger.info(
                    f"Step {self.metrics.step} | Loss: {loss.item():.4f} | "
                    f"LR: {scheduler.current_lr:.2e if scheduler else 0:.2e} | "
                    f"Tokens/s: {tps:.0f}"
                )

            # Checkpoint periódico
            if self.metrics.step % self.config.training.checkpoint_interval == 0:
                self.save_checkpoint(f"step_{self.metrics.step}")

        avg_loss = total_loss / max(n_batches, 1)
        self.metrics.train_loss = avg_loss
        return avg_loss

    @torch.no_grad()
    def evaluate(self, dataloader) -> Tuple[float, float]:
        """Evalúa el modelo. Retorna (loss, perplexity)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for input_ids, labels in dataloader:
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            labels[labels == self.tokenizer.pad_id] = -100
            out = self.model(input_ids, labels=labels)
            total_loss += out["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))
        return avg_loss, perplexity

    def fit(self, train_dataloader, val_dataloader=None,
            n_epochs: Optional[int] = None) -> TrainingMetrics:
        """
        Entrenamiento completo con warmup, evaluación y checkpoints.
        """
        n_epochs = n_epochs or self.config.training.max_epochs
        total_steps = n_epochs * len(train_dataloader)

        scheduler = WarmupCosineScheduler(
            self.optimizer,
            warmup_steps=self.config.training.warmup_steps,
            max_steps=total_steps,
        )

        logger.info(f"Iniciando entrenamiento: {n_epochs} épocas, {total_steps} pasos")
        logger.info(f"Modelo: {self.model.count_parameters():,} parámetros")
        logger.info(f"Device: {self.device}")

        best_val_loss = float("inf")

        for epoch in range(n_epochs):
            self.metrics.epoch = epoch
            logger.info(f"\n{'='*50}")
            logger.info(f"Época {epoch+1}/{n_epochs}")
            logger.info(f"{'='*50}")

            train_loss = self.train_epoch(train_dataloader, val_dataloader, scheduler)

            if val_dataloader:
                val_loss, perplexity = self.evaluate(val_dataloader)
                self.metrics.update(
                    step=self.metrics.step, epoch=epoch,
                    train_loss=train_loss, val_loss=val_loss,
                    perplexity=perplexity,
                    learning_rate=scheduler.current_lr,
                )
                logger.info(
                    f"Época {epoch+1}: Train Loss={train_loss:.4f}, "
                    f"Val Loss={val_loss:.4f}, PPL={perplexity:.1f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint("best")
                    logger.info(f"Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")
            else:
                self.metrics.update(train_loss=train_loss, step=self.metrics.step)

        self.save_checkpoint("latest")
        self.metrics.save(str(self.log_dir / "metrics.json"))
        return self.metrics

    def save_checkpoint(self, name: str):
        """Guarda checkpoint del modelo + estado del optimizador."""
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        self.model.save(str(path))
        # Guardar estado del optimizador por separado
        opt_path = self.checkpoint_dir / f"optimizer_{name}.pt"
        torch.save({"optimizer": self.optimizer.state_dict()}, str(opt_path))
        logger.debug(f"Checkpoint guardado: {path}")

    def load_checkpoint(self, name: str) -> bool:
        """Carga checkpoint si existe."""
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        if not path.exists():
            return False
        from core.model import AutoiaLLM
        self.model = AutoiaLLM.load(str(path), device=self.device)
        logger.info(f"Checkpoint cargado: {path}")
        return True
