"""
API REST para interactuar con el LLM de Autoia.
Endpoints: generación de texto, estado del modelo, control de aprendizaje.
"""

import logging
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ─── Modelos de request/response ─────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Texto de entrada para generar a continuación")
    max_new_tokens: int = Field(200, ge=10, le=2000)
    temperature: float = Field(0.8, ge=0.01, le=2.0)
    top_k: int = Field(50, ge=1, le=200)
    top_p: float = Field(0.95, ge=0.1, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    generated_tokens: int
    model_generation: int


class ModelStatusResponse(BaseModel):
    topic: str
    generation: int
    n_layers: int
    d_model: int
    n_params: int
    training_steps: int
    train_loss: float
    val_loss: float
    perplexity: float
    corpus_size: int
    vocab_size: int
    growth_history_count: int


class LearnRequest(BaseModel):
    texts: Optional[List[str]] = Field(None, description="Textos adicionales para aprender")
    use_web: bool = Field(False, description="Recolectar datos de Wikipedia/web")
    n_epochs: int = Field(5, ge=1, le=50)


# ─── Aplicación FastAPI ───────────────────────────────────────────────────────

def create_app(autoia_system) -> FastAPI:
    """
    Crea la aplicación FastAPI con el sistema Autoia inyectado.
    """
    app = FastAPI(
        title="Autoia - Self-Learning LLM",
        description=f"LLM generativo que aprende sobre: {autoia_system.config.learning.topic}",
        version="1.0.0",
    )

    @app.get("/")
    def root():
        return {
            "name": "Autoia",
            "topic": autoia_system.config.learning.topic,
            "status": "running",
            "generation": autoia_system.model.snapshot.generation if autoia_system.model else 0,
        }

    @app.get("/status", response_model=ModelStatusResponse)
    def get_status():
        """Estado completo del modelo y el aprendizaje."""
        if not autoia_system.model:
            raise HTTPException(503, "Modelo no inicializado")

        snap = autoia_system.model.snapshot
        metrics = autoia_system.trainer.metrics if autoia_system.trainer else None
        corpus_size = autoia_system.crawler.corpus_size() if autoia_system.crawler else 0

        return ModelStatusResponse(
            topic=autoia_system.config.learning.topic,
            generation=snap.generation,
            n_layers=snap.n_layers,
            d_model=snap.d_model,
            n_params=autoia_system.model.count_parameters(),
            training_steps=snap.training_steps,
            train_loss=metrics.train_loss if metrics else float("inf"),
            val_loss=metrics.val_loss if metrics else float("inf"),
            perplexity=metrics.perplexity if metrics else float("inf"),
            corpus_size=corpus_size,
            vocab_size=autoia_system.tokenizer.actual_vocab_size if autoia_system.tokenizer else 0,
            growth_history_count=len(autoia_system.architect.growth_history) if autoia_system.architect else 0,
        )

    @app.post("/generate", response_model=GenerateResponse)
    def generate(request: GenerateRequest):
        """Genera texto continuando el prompt dado."""
        if not autoia_system.model or not autoia_system.tokenizer:
            raise HTTPException(503, "Modelo no listo. Inicializa con /learn primero.")

        try:
            import torch
            device = autoia_system.device

            input_ids = autoia_system.tokenizer.encode(request.prompt)
            if len(input_ids) > autoia_system.model.snapshot.max_seq_len - 100:
                input_ids = input_ids[-(autoia_system.model.snapshot.max_seq_len - 100):]

            input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

            generated = autoia_system.model.generate(
                input_tensor,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_k=request.top_k,
                top_p=request.top_p,
                repetition_penalty=request.repetition_penalty,
            )

            all_ids = generated[0].tolist()
            new_ids = all_ids[len(input_ids):]
            generated_text = autoia_system.tokenizer.decode(all_ids, skip_special_tokens=True)

            return GenerateResponse(
                generated_text=generated_text,
                prompt_tokens=len(input_ids),
                generated_tokens=len(new_ids),
                model_generation=autoia_system.model.snapshot.generation,
            )
        except Exception as e:
            logger.exception("Error en generación")
            raise HTTPException(500, str(e))

    @app.post("/learn")
    def learn(request: LearnRequest, background_tasks: BackgroundTasks):
        """Inicia un ciclo de aprendizaje en background."""
        background_tasks.add_task(
            autoia_system.learn_cycle,
            extra_texts=request.texts,
            use_web=request.use_web,
            n_epochs=request.n_epochs,
        )
        return {"status": "learning_started", "message": "El aprendizaje está en progreso"}

    @app.post("/evolve")
    def evolve(background_tasks: BackgroundTasks):
        """Fuerza una revisión de crecimiento de arquitectura."""
        background_tasks.add_task(autoia_system.check_and_evolve)
        return {"status": "evolution_check_started"}

    @app.get("/growth-report")
    def growth_report():
        """Historial de crecimiento del modelo."""
        if not autoia_system.architect:
            return {"report": "Architect no inicializado"}
        return {"report": autoia_system.architect.growth_report()}

    @app.get("/self-prog-report")
    def self_prog_report():
        """Reporte de auto-programación."""
        if not autoia_system.self_programmer:
            return {"report": "SelfProgrammer no inicializado"}
        return {"report": autoia_system.self_programmer.get_report()}

    @app.post("/save")
    def save_model():
        """Guarda el estado actual del modelo."""
        try:
            autoia_system.save_state()
            return {"status": "saved"}
        except Exception as e:
            raise HTTPException(500, str(e))

    return app
