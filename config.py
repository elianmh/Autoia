"""
Autoia - Self-Learning LLM Configuration
Configura aquí el tema específico sobre el que aprenderá el modelo.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Arquitectura inicial del Transformer."""
    # Dimensiones del modelo
    vocab_size: int = 8000          # Tamaño del vocabulario BPE
    d_model: int = 256              # Dimensión del embedding
    n_heads: int = 8                # Cabezas de atención
    n_layers: int = 4               # Capas transformer
    d_ff: int = 1024                # Dimensión feed-forward
    max_seq_len: int = 512          # Longitud máxima de secuencia
    dropout: float = 0.1

    # Límites de crecimiento automático
    max_layers: int = 16
    max_d_model: int = 1024
    max_vocab_size: int = 32000


@dataclass
class TrainingConfig:
    """Parámetros de entrenamiento."""
    batch_size: int = 8
    learning_rate: float = 3e-4
    max_epochs: int = 50
    warmup_steps: int = 100
    gradient_clip: float = 1.0
    eval_interval: int = 100        # Cada cuántos pasos evaluar
    checkpoint_interval: int = 500  # Cada cuántos pasos guardar checkpoint
    min_improvement: float = 0.005  # Mejora mínima para no escalar


@dataclass
class LearningConfig:
    """Configuración del aprendizaje continuo."""
    # TEMA ESPECÍFICO DE APRENDIZAJE - CAMBIA ESTO
    topic: str = "inteligencia artificial y machine learning"
    topic_keywords: List[str] = field(default_factory=lambda: [
        "machine learning", "deep learning", "neural network",
        "transformer", "gradient descent", "backpropagation",
        "reinforcement learning", "natural language processing"
    ])

    # Fuentes de datos
    data_dir: str = "data"
    max_corpus_size_mb: int = 100   # Límite del corpus en MB

    # Aprendizaje continuo
    continual_learning: bool = True
    replay_buffer_size: int = 10000  # Ejemplos a recordar del pasado
    new_data_threshold: int = 500    # Nuevos ejemplos para re-entrenar


@dataclass
class EvolutionConfig:
    """Configuración del auto-crecimiento."""
    # Cuándo escalar la arquitectura
    scale_loss_threshold: float = 0.3    # Si loss < umbral, no es necesario crecer
    scale_plateau_patience: int = 5      # Épocas sin mejora antes de escalar
    scale_factor: float = 1.5            # Factor de crecimiento

    # Auto-programación
    allow_self_programming: bool = True
    self_program_interval: int = 1000    # Pasos entre auto-revisiones
    sandbox_execution: bool = True       # Ejecutar código generado en sandbox


@dataclass
class AutoiaConfig:
    """Configuración global de Autoia."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)

    # Directorios
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    data_dir: str = "data"

    # Dispositivo
    device: str = "auto"  # "auto", "cpu", "cuda", "mps"

    def get_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


# Instancia global de configuración
CONFIG = AutoiaConfig()
