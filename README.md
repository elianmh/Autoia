# Autoia — Self-Learning LLM desde Cero

Un LLM Transformer **generativo**, construido desde cero en PyTorch, que:
- Aprende sobre **un tema específico** (configurable)
- **Crece su arquitectura** automáticamente cuando alcanza sus límites
- **Se auto-programa** para mejorar su propio pipeline de entrenamiento
- Implementa **aprendizaje continuo** con replay buffer (sin olvido catastrófico)

---

## Arquitectura

```
Autoia/
├── core/
│   ├── model.py        # Transformer GPT-style desde cero (RoPE, SwiGLU, RMSNorm)
│   └── tokenizer.py    # BPE Tokenizer entrenable
├── learning/
│   ├── crawler.py      # Recolector de datos (Wikipedia, HuggingFace, archivos locales)
│   ├── dataset.py      # Dataset + Replay Buffer anti-olvido
│   └── trainer.py      # Motor de entrenamiento con warmup cosine scheduler
├── evolution/
│   ├── architect.py    # Decisor de crecimiento automático de arquitectura
│   └── self_programmer.py  # Auto-generación de código para mejorar el pipeline
├── api/
│   └── server.py       # API REST (FastAPI)
├── config.py           # Toda la configuración del sistema
└── main.py             # Punto de entrada
```

## El LLM desde cero

El transformer implementado incluye:

| Componente | Implementación |
|---|---|
| Positional Encoding | **RoPE** (Rotary Position Embeddings) |
| Normalización | **RMSNorm** |
| Feed-Forward | **SwiGLU** (como LLaMA/Mistral) |
| Atención | **Multi-Head Causal Self-Attention** |
| Sampling | Temperature + Top-k + Top-p + Repetition Penalty |
| Entrenamiento | AdamW + Warmup Cosine LR |

## Instalación

```bash
pip install -r requirements.txt
```

## Uso rápido

### 1. Configurar el tema

Edita `config.py`:
```python
topic: str = "física cuántica"
topic_keywords: List[str] = ["quark", "entrelazamiento", "superposición", ...]
```

### 2. Entrenar desde cero

```bash
python main.py --mode train --epochs 20
```

### 3. Aprendizaje continuo

```bash
python main.py --mode learn --epochs 10
```

### 4. Chat interactivo

```bash
python main.py --mode chat
```

### 5. API REST

```bash
python main.py --mode api --port 8000
# Docs en: http://localhost:8000/docs
```

### 6. Ver estado del modelo

```bash
python main.py --mode status
```

## Cómo funciona el auto-crecimiento

```
Cada ciclo de aprendizaje:
  ┌─────────────────────────────────────┐
  │  Recolectar datos del tema          │
  │  ↓                                  │
  │  Entrenar modelo                    │
  │  ↓                                  │
  │  ¿Loss en plateau? ──→ Añadir capas │
  │  ¿Corpus creció?   ──→ Escalar dims │
  │  ¿Vocab nuevo?     ──→ Expandir BPE │
  │  ↓                                  │
  │  Auto-programación:                 │
  │    - Genera augmentación de datos   │
  │    - Ajusta hiperparámetros         │
  │    - Crea métricas de evaluación    │
  └─────────────────────────────────────┘
```

## Auto-Programación

El sistema genera código Python para mejorar su propio pipeline:

1. **Augmentación de datos**: genera variaciones del texto del tema
2. **Optimización de HPs**: ajusta LR y batch_size según el comportamiento del loss
3. **Métricas custom**: crea funciones de evaluación específicas al tema

El código generado se ejecuta en un **sandbox seguro** (imports restringidos).

## Parámetros del modelo por defecto

| Parámetro | Valor inicial | Máximo |
|---|---|---|
| Capas | 4 | 16 |
| d_model | 256 | 1024 |
| Cabezas de atención | 8 | 8 |
| Vocab | 8,000 | 32,000 |
| Contexto | 512 tokens | 512 tokens |

El modelo parte con ~7M parámetros y puede crecer hasta ~300M.

## Cambiar el tema

```python
# config.py
topic: str = "historia del Imperio Romano"
topic_keywords: List[str] = [
    "César", "Augusto", "legión", "senado",
    "República", "conquista", "gladiador"
]
```

```bash
python main.py --mode train --topic "historia del Imperio Romano"
```
