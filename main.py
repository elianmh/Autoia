"""
Autoia - LLM Generativo Auto-Aprendiente desde Cero
=====================================================

Sistema completo que:
1. Construye un Transformer desde cero
2. Aprende sobre un tema específico (configurable)
3. Crece su arquitectura automáticamente cuando aprende más
4. Se auto-programa para mejorar su pipeline

Uso:
    python main.py --mode train    # Entrenar por primera vez
    python main.py --mode learn    # Ciclo de aprendizaje continuo
    python main.py --mode chat     # Chat interactivo
    python main.py --mode api      # Iniciar API REST
    python main.py --mode status   # Ver estado actual
"""

import os
import sys
import json
import math
import time
import logging
import argparse
import random
from pathlib import Path
from typing import Optional, List

# Configurar logging antes de imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/autoia.log", mode="a"),
    ],
)
logger = logging.getLogger("autoia")

# Asegurar directorio de logs
Path("logs").mkdir(exist_ok=True)


class AutoiaSystem:
    """
    Sistema principal de Autoia.
    Orquesta todos los componentes: modelo, tokenizer, crawler, trainer, evolucion.
    """

    def __init__(self, config=None):
        from config import CONFIG
        self.config = config or CONFIG
        self.device = self.config.get_device()

        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.crawler = None
        self.architect = None
        self.self_programmer = None

        logger.info(f"Autoia iniciando en dispositivo: {self.device}")
        logger.info(f"Tema de aprendizaje: {self.config.learning.topic}")

    def initialize(self):
        """Inicializa todos los componentes del sistema."""
        import torch
        from core.model import AutoiaLLM, ModelSnapshot
        from core.tokenizer import AutoiaTokenizer
        from learning.crawler import TopicCrawler
        from learning.trainer import AutoiaTrainer
        from evolution.architect import ArchitectureEvolver
        from evolution.self_programmer import SelfProgrammer

        # 1. Tokenizer
        self.tokenizer = AutoiaTokenizer(
            vocab_size=self.config.model.vocab_size,
            save_dir=self.config.data_dir
        )

        # 2. Crawler
        self.crawler = TopicCrawler(
            topic=self.config.learning.topic,
            keywords=self.config.learning.topic_keywords,
            data_dir=self.config.data_dir
        )

        # 3. Modelo: cargar si existe, si no crear nuevo
        best_ckpt = Path(self.config.checkpoint_dir) / "checkpoint_best.pt"
        latest_ckpt = Path(self.config.checkpoint_dir) / "checkpoint_latest.pt"

        if best_ckpt.exists():
            logger.info("Cargando modelo guardado (best)...")
            self.model = AutoiaLLM.load(str(best_ckpt), device=self.device)
        elif latest_ckpt.exists():
            logger.info("Cargando modelo guardado (latest)...")
            self.model = AutoiaLLM.load(str(latest_ckpt), device=self.device)
        else:
            logger.info("Creando modelo nuevo desde cero...")
            snap = ModelSnapshot(
                vocab_size=self.config.model.vocab_size,
                d_model=self.config.model.d_model,
                n_heads=self.config.model.n_heads,
                n_layers=self.config.model.n_layers,
                d_ff=self.config.model.d_ff,
                max_seq_len=self.config.model.max_seq_len,
                dropout=self.config.model.dropout,
            )
            self.model = AutoiaLLM(snap)

        self.model = self.model.to(self.device)
        logger.info(f"Modelo listo: {self.model.count_parameters():,} parámetros, "
                    f"generación {self.model.snapshot.generation}")

        # 4. Trainer
        self.trainer = AutoiaTrainer(self.model, self.tokenizer, self.config, self.device)

        # 5. Evolución
        self.architect = ArchitectureEvolver(self.config, log_dir=self.config.log_dir)
        self.self_programmer = SelfProgrammer(self.config, self.model, log_dir=self.config.log_dir)

    def bootstrap_tokenizer(self):
        """
        Si el tokenizer no está entrenado, recolecta datos mínimos y lo entrena.
        """
        if self.tokenizer._tokenizer is not None:
            logger.info("Tokenizer ya entrenado.")
            return

        logger.info("Tokenizer no encontrado. Recolectando datos para entrenamiento...")

        # Intentar cargar corpus existente primero
        texts = self.crawler.load_corpus()

        if len(texts) < 10:
            logger.info("Corpus vacío. Obteniendo datos iniciales...")
            docs = self.crawler.collect_all(use_wikipedia=True, use_hf=False, use_local=True)
            if docs:
                self.crawler.save_corpus(docs)
                texts = [d.text for d in docs]
            else:
                # Fallback: textos de ejemplo sobre el tema
                texts = self._get_bootstrap_texts()

        logger.info(f"Entrenando tokenizer con {len(texts)} textos...")
        self.tokenizer.train(texts, min_frequency=2)
        logger.info(f"Tokenizer entrenado: {self.tokenizer.actual_vocab_size:,} tokens")

    def _get_bootstrap_texts(self) -> List[str]:
        """Textos de bootstrap mínimos sobre el tema (usados si no hay internet)."""
        topic = self.config.learning.topic
        keywords = self.config.learning.topic_keywords
        texts = [
            f"This is a text about {topic}. " + " ".join(keywords) + ".",
            f"{topic} is an important field with many applications.",
            f"The study of {topic} involves {', '.join(keywords[:3])}.",
            f"Learning about {topic}: " + ". ".join(
                f"{kw} is a key concept" for kw in keywords
            ) + ".",
        ] * 50  # Repetir para tener suficientes tokens
        return texts

    def learn_cycle(self, extra_texts: Optional[List[str]] = None,
                    use_web: bool = True, n_epochs: int = 10):
        """
        Ciclo completo de aprendizaje:
        1. Recolectar nuevos datos
        2. Tokenizar
        3. Entrenar con replay buffer
        4. Evaluar
        5. Decidir si crecer
        6. Auto-programar mejoras
        """
        import torch
        from learning.dataset import (
            TextDataset, ReplayBuffer, ContinualDataset, create_dataloader
        )

        logger.info("=" * 60)
        logger.info("INICIANDO CICLO DE APRENDIZAJE")
        logger.info("=" * 60)

        # 1. Recolectar datos nuevos
        new_texts = extra_texts or []
        if use_web:
            docs = self.crawler.collect_all(use_wikipedia=True, use_hf=True, use_local=True)
            if docs:
                self.crawler.save_corpus(docs)
                new_texts.extend([d.text for d in docs])

        # Cargar corpus completo
        corpus = self.crawler.load_corpus(min_relevance=0.05)
        if not corpus:
            corpus = self._get_bootstrap_texts()
            logger.warning("Corpus vacío, usando textos de bootstrap")

        logger.info(f"Corpus total: {len(corpus)} documentos")

        # 2. Bootstrap tokenizer si no existe
        if self.tokenizer._tokenizer is None:
            self.tokenizer.train(corpus)
            # Actualizar vocab_size del modelo si es necesario
            actual_vocab = self.tokenizer.actual_vocab_size
            if actual_vocab != self.model.snapshot.vocab_size:
                logger.warning(
                    f"Vocab mismatch: tokenizer={actual_vocab}, modelo={self.model.snapshot.vocab_size}"
                    " - Recomendado re-crear el modelo"
                )

        # 3. Aplicar augmentación de datos si existe
        if self.self_programmer:
            corpus = self.self_programmer.apply_augmentation(corpus)

        # 4. Crear datasets
        random.shuffle(corpus)
        split = int(len(corpus) * 0.9)
        train_texts, val_texts = corpus[:split], corpus[split:]

        # Replay buffer
        replay_buffer = ReplayBuffer(max_size=self.config.learning.replay_buffer_size)
        replay_path = Path(self.config.data_dir) / "replay_buffer.json"
        replay_buffer.load(str(replay_path))

        # Dataset de entrenamiento con datos nuevos + replay
        train_dataset = TextDataset(
            train_texts, self.tokenizer,
            max_seq_len=self.config.model.max_seq_len
        )
        val_dataset = TextDataset(
            val_texts, self.tokenizer,
            max_seq_len=self.config.model.max_seq_len
        )

        # Añadir chunks al replay buffer
        replay_buffer.add(train_dataset.chunks)
        replay_buffer.save(str(replay_path))

        if len(train_dataset) == 0:
            logger.error("Dataset de entrenamiento vacío. Abortando ciclo.")
            return

        train_loader = create_dataloader(
            train_dataset, batch_size=self.config.training.batch_size
        )
        val_loader = create_dataloader(
            val_dataset, batch_size=self.config.training.batch_size, shuffle=False
        ) if len(val_dataset) > 0 else None

        # 5. Entrenar
        logger.info(f"Entrenando por {n_epochs} épocas...")
        self.trainer.model = self.model
        self.trainer._setup_optimizer()
        metrics = self.trainer.fit(train_loader, val_loader, n_epochs=n_epochs)

        # 6. Evaluar si debe crecer
        self.check_and_evolve(corpus)

        # 7. Auto-programación
        if (self.self_programmer and
                self.config.evolution.allow_self_programming and
                self.model.snapshot.training_steps % self.config.evolution.self_program_interval < n_epochs * len(train_loader)):
            logger.info("Ejecutando revisión de auto-programación...")
            self.self_programmer.model = self.model
            self.self_programmer.run_self_review(metrics)

        # 8. Guardar estado
        self.save_state()
        logger.info("Ciclo de aprendizaje completado.")
        self.print_status()

    def check_and_evolve(self, corpus: Optional[List[str]] = None):
        """Verifica si el modelo debe crecer y lo hace si es necesario."""
        if not self.architect or not self.trainer:
            return

        metrics = self.trainer.metrics
        corpus_size = self.crawler.corpus_size() if self.crawler else 0
        vocab_size = self.tokenizer.actual_vocab_size if self.tokenizer else 0

        decision = self.architect.analyze(
            metrics, self.model.snapshot, corpus_size, vocab_size
        )

        if decision.should_grow:
            logger.info(f"CRECIMIENTO ACTIVADO: {decision.reason}")
            corpus_for_growth = corpus or (self.crawler.load_corpus() if self.crawler else [])
            new_model, new_tokenizer = self.architect.apply_growth(
                self.model, self.tokenizer, decision, corpus_for_growth
            )
            self.model = new_model.to(self.device)
            self.tokenizer = new_tokenizer
            self.trainer.model = self.model
            self.trainer._setup_optimizer()

            # Si el tokenizer creció, actualizar el modelo con el nuevo vocab
            if new_tokenizer.actual_vocab_size != self.model.snapshot.vocab_size:
                logger.info("Re-inicializando modelo con nuevo tamaño de vocabulario...")
                from core.model import AutoiaLLM, ModelSnapshot
                snap = self.model.snapshot
                snap.vocab_size = new_tokenizer.actual_vocab_size
                new_model_with_vocab = AutoiaLLM(snap)
                self.model = new_model_with_vocab.to(self.device)
                self.trainer.model = self.model

        else:
            logger.info(f"Sin crecimiento: {decision.reason}")

    def generate(self, prompt: str, max_new_tokens: int = 200,
                 temperature: float = 0.8) -> str:
        """Genera texto dado un prompt."""
        if not self.model or not self.tokenizer:
            return "Modelo no inicializado."
        if self.tokenizer._tokenizer is None:
            return "Tokenizer no entrenado. Ejecuta un ciclo de aprendizaje primero."

        import torch
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        generated = self.model.generate(
            input_tensor, max_new_tokens=max_new_tokens, temperature=temperature
        )
        return self.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    def save_state(self):
        """Guarda el estado completo del sistema."""
        self.model.save(str(Path(self.config.checkpoint_dir) / "checkpoint_latest.pt"))
        if self.trainer:
            self.trainer.metrics.save(str(Path(self.config.log_dir) / "metrics.json"))
        logger.info("Estado guardado.")

    def print_status(self):
        """Imprime el estado actual del sistema."""
        snap = self.model.snapshot
        metrics = self.trainer.metrics if self.trainer else None
        corpus = self.crawler.corpus_size() if self.crawler else 0

        print("\n" + "=" * 60)
        print("ESTADO DE AUTOIA")
        print("=" * 60)
        print(f"Tema:          {self.config.learning.topic}")
        print(f"Generación:    {snap.generation}")
        print(f"Capas:         {snap.n_layers}")
        print(f"d_model:       {snap.d_model}")
        print(f"Parámetros:    {self.model.count_parameters():,}")
        print(f"Pasos entreno: {snap.training_steps:,}")
        if metrics:
            print(f"Train Loss:    {metrics.train_loss:.4f}")
            print(f"Val Loss:      {metrics.val_loss:.4f}")
            print(f"Perplexity:    {metrics.perplexity:.1f}")
        print(f"Corpus docs:   {corpus:,}")
        if self.tokenizer:
            print(f"Vocab size:    {self.tokenizer.actual_vocab_size:,}")
        if self.architect:
            print(f"Crecimientos:  {len(self.architect.growth_history)}")
        print("=" * 60)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoia - Self-Learning LLM")
    parser.add_argument("--mode", choices=["train", "learn", "chat", "api", "status"],
                        default="train", help="Modo de operación")
    parser.add_argument("--topic", type=str, help="Tema de aprendizaje (override config)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-web", action="store_true", help="No recolectar datos web")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # Configuración
    from config import CONFIG
    if args.topic:
        CONFIG.learning.topic = args.topic
        logger.info(f"Tema override: {args.topic}")

    # Inicializar sistema
    system = AutoiaSystem(CONFIG)
    system.initialize()

    if args.mode == "train":
        print(f"\nIniciando entrenamiento inicial sobre: '{CONFIG.learning.topic}'")
        system.bootstrap_tokenizer()
        system.learn_cycle(use_web=not args.no_web, n_epochs=args.epochs)

    elif args.mode == "learn":
        print(f"\nCiclo de aprendizaje continuo sobre: '{CONFIG.learning.topic}'")
        system.learn_cycle(use_web=not args.no_web, n_epochs=args.epochs)

    elif args.mode == "chat":
        print(f"\nAutoia Chat - Tema: {CONFIG.learning.topic}")
        print("Escribe 'salir' para terminar\n")
        if system.tokenizer._tokenizer is None:
            print("AVISO: Tokenizer no entrenado. Ejecuta --mode train primero.")
            return
        while True:
            try:
                prompt = input("Tú: ").strip()
                if prompt.lower() in {"salir", "exit", "quit"}:
                    break
                if not prompt:
                    continue
                print("Autoia: ", end="", flush=True)
                response = system.generate(prompt, max_new_tokens=200)
                print(response)
                print()
            except KeyboardInterrupt:
                break
        print("\n¡Hasta luego!")

    elif args.mode == "api":
        import uvicorn
        from api.server import create_app
        app = create_app(system)
        print(f"\nAPI Autoia en http://{args.host}:{args.port}")
        print(f"Documentación: http://{args.host}:{args.port}/docs")
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")

    elif args.mode == "status":
        if system.tokenizer._tokenizer is None:
            print("Tokenizer no entrenado.")
        system.print_status()
        if system.architect:
            print("\n" + system.architect.growth_report())
        if system.self_programmer:
            print("\n" + system.self_programmer.get_report())


if __name__ == "__main__":
    main()
