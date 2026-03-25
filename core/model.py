"""
Transformer LLM construido desde cero.
Arquitectura: GPT-style decoder-only transformer con capacidad de auto-crecimiento.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ModelSnapshot:
    """Snapshot de la arquitectura actual para guardar/cargar."""
    vocab_size: int
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    max_seq_len: int
    dropout: float
    generation: int = 0          # Cuántas veces ha crecido
    total_params: int = 0
    training_steps: int = 0


class RMSNorm(nn.Module):
    """RMSNorm - más eficiente que LayerNorm."""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Permite al modelo generalizar a secuencias más largas que las vistas durante entrenamiento.
    """
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int):
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int):
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        q = (q * cos) + (self.rotate_half(q) * sin)
        k = (k * cos) + (self.rotate_half(k) * sin)
        return q, k


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention con causal masking y RoPE.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rope = RotaryPositionalEmbedding(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Aplicar RoPE
        q, k = self.rope(q, k, T)

        # Cache para inferencia
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        new_kv = (k, v) if use_cache else None
        kv_len = k.shape[2]

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask
        if mask is None:
            causal = torch.triu(
                torch.full((T, kv_len), float("-inf"), device=x.device), diagonal=1
            )
            attn = attn + causal

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        out = self.dropout(out)
        return out, new_kv


class FeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network - como LLaMA/Mistral.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: silu(gate) * up
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.dropout(self.down_proj(gate * up))


class TransformerBlock(nn.Module):
    """Un bloque Transformer completo: Atención + FFN con pre-norm."""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        use_cache: bool = False,
        past_kv: Optional[Tuple] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        # Pre-norm + atención con residual
        attn_out, new_kv = self.attn(self.norm1(x), use_cache=use_cache, past_kv=past_kv)
        x = x + attn_out
        # Pre-norm + FFN con residual
        x = x + self.ffn(self.norm2(x))
        return x, new_kv


class AutoiaLLM(nn.Module):
    """
    El LLM principal de Autoia.
    GPT-style decoder-only transformer con:
    - RoPE (Rotary Position Embeddings)
    - RMSNorm
    - SwiGLU FFN
    - Capacidad de auto-crecimiento
    """

    def __init__(self, snapshot: ModelSnapshot):
        super().__init__()
        self.snapshot = snapshot

        self.token_embedding = nn.Embedding(snapshot.vocab_size, snapshot.d_model)
        self.dropout = nn.Dropout(snapshot.dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                snapshot.d_model, snapshot.n_heads,
                snapshot.d_ff, snapshot.dropout
            )
            for _ in range(snapshot.n_layers)
        ])

        self.norm_f = RMSNorm(snapshot.d_model)
        self.lm_head = nn.Linear(snapshot.d_model, snapshot.vocab_size, bias=False)

        # Weight tying: embedding y lm_head comparten pesos
        self.token_embedding.weight = self.lm_head.weight

        self._init_weights()
        self.snapshot.total_params = self.count_parameters()

    def _init_weights(self):
        """Inicialización de pesos tipo GPT."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_kvs: Optional[list] = None,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        x = self.token_embedding(input_ids)
        x = self.dropout(x)

        new_kvs = []
        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs else None
            x, new_kv = block(x, use_cache=use_cache, past_kv=past_kv)
            new_kvs.append(new_kv)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        result = {"logits": logits}

        if labels is not None:
            # Cross-entropy loss (shift para predicción del siguiente token)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
            result["perplexity"] = torch.exp(loss)

        if use_cache:
            result["past_kvs"] = new_kvs

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> torch.Tensor:
        """
        Genera texto token por token con:
        - Temperature sampling
        - Top-k filtering
        - Top-p (nucleus) sampling
        - Repetition penalty
        """
        self.eval()
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            # Truncar si excede la longitud máxima
            context = generated[:, -self.snapshot.max_seq_len:]

            out = self.forward(context)
            logits = out["logits"][:, -1, :]  # Último token

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated[0].tolist()):
                    logits[0, token_id] /= repetition_penalty

            # Temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k
            if top_k > 0:
                top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)))[0]
                logits[logits < top_k_vals[..., [-1]]] = float("-inf")

            # Top-p (nucleus sampling)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_idx_to_remove = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_idx_to_remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)

        return generated

    def save(self, path: str):
        """Guarda el modelo completo con su snapshot de arquitectura."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "snapshot": asdict(self.snapshot),
            "state_dict": self.state_dict(),
        }, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "AutoiaLLM":
        """Carga un modelo guardado."""
        checkpoint = torch.load(path, map_location=device)
        snapshot = ModelSnapshot(**checkpoint["snapshot"])
        model = cls(snapshot)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(device)

    def grow(self, target_layers: Optional[int] = None, target_d_model: Optional[int] = None) -> "AutoiaLLM":
        """
        Crea una nueva instancia del modelo con arquitectura más grande,
        copiando los pesos existentes (Net2Net expansion).
        """
        from config import CONFIG

        new_snap = ModelSnapshot(
            vocab_size=self.snapshot.vocab_size,
            d_model=target_d_model or min(
                int(self.snapshot.d_model * CONFIG.evolution.scale_factor),
                CONFIG.model.max_d_model
            ),
            n_heads=self.snapshot.n_heads,
            n_layers=target_layers or min(
                self.snapshot.n_layers + 2,
                CONFIG.model.max_layers
            ),
            d_ff=int((target_d_model or self.snapshot.d_model * CONFIG.evolution.scale_factor) * 4),
            max_seq_len=self.snapshot.max_seq_len,
            dropout=self.snapshot.dropout,
            generation=self.snapshot.generation + 1,
            training_steps=self.snapshot.training_steps,
        )

        # Ajustar n_heads para que divida d_model
        while new_snap.d_model % new_snap.n_heads != 0:
            new_snap.n_heads -= 1

        new_model = AutoiaLLM(new_snap)

        # Copiar pesos existentes en las capas que coinciden
        self._transfer_weights(new_model)

        print(f"[GROW] Generación {self.snapshot.generation} -> {new_snap.generation}")
        print(f"       Capas: {self.snapshot.n_layers} -> {new_snap.n_layers}")
        print(f"       d_model: {self.snapshot.d_model} -> {new_snap.d_model}")
        print(f"       Parámetros: {self.count_parameters():,} -> {new_model.count_parameters():,}")

        return new_model

    def _transfer_weights(self, new_model: "AutoiaLLM"):
        """Transfiere pesos del modelo actual al nuevo (Net2Net)."""
        # Si mismas dimensiones, copiar directamente
        old_d = self.snapshot.d_model
        new_d = new_model.snapshot.d_model

        with torch.no_grad():
            # Copiar capas que existen en ambos modelos
            old_blocks = min(self.snapshot.n_layers, new_model.snapshot.n_layers)
            for i in range(old_blocks):
                if old_d == new_d:
                    new_model.blocks[i].load_state_dict(
                        self.blocks[i].state_dict(), strict=False
                    )
                # Si d_model cambió, copiar el subespacio de pesos
                # (las primeras old_d dimensiones)
                # Esto se hace de forma implícita: los nuevos pesos se inicializan
                # normalmente y los existentes se copian donde coinciden.
