"""
Dataset para entrenamiento del LLM.
Implementa replay buffer para aprendizaje continuo sin olvidar.
"""

import random
import torch
import logging
from typing import List, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
from collections import deque

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Dataset de texto para LM, trocea textos en ventanas de contexto."""

    def __init__(self, texts: List[str], tokenizer, max_seq_len: int = 512,
                 stride: int = 256):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.chunks: List[List[int]] = []
        self._build_chunks(texts, stride)

    def _build_chunks(self, texts: List[str], stride: int):
        """Tokeniza textos y los divide en chunks con solapamiento."""
        for text in texts:
            try:
                ids = self.tokenizer.encode(text, add_special_tokens=False)
                # Dividir en chunks con stride para mejor cobertura
                for i in range(0, max(1, len(ids) - self.max_seq_len + 1), stride):
                    chunk = ids[i: i + self.max_seq_len]
                    if len(chunk) >= 32:  # Mínimo 32 tokens por chunk
                        self.chunks.append(chunk)
            except Exception as e:
                logger.debug(f"Error tokenizando texto: {e}")

        logger.info(f"Dataset: {len(self.chunks)} chunks de {self.max_seq_len} tokens")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.chunks[idx]
        # Pad si es necesario
        if len(ids) < self.max_seq_len:
            ids = ids + [self.tokenizer.pad_id] * (self.max_seq_len - len(ids))
        ids = torch.tensor(ids[:self.max_seq_len], dtype=torch.long)
        # Input: todos menos último, Labels: todos menos primero
        return ids[:-1], ids[1:]


class ReplayBuffer:
    """
    Buffer de replay para aprendizaje continuo.
    Almacena ejemplos pasados para evitar el olvido catastrófico.
    Usa reservoir sampling para mantener distribución uniforme.
    """

    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer: deque = deque(maxlen=max_size)
        self.total_seen = 0

    def add(self, chunks: List[List[int]]):
        """Añade chunks al buffer usando reservoir sampling."""
        for chunk in chunks:
            self.total_seen += 1
            if len(self.buffer) < self.max_size:
                self.buffer.append(chunk)
            else:
                # Reservoir sampling
                j = random.randint(0, self.total_seen - 1)
                if j < self.max_size:
                    self.buffer[j] = chunk

    def sample(self, n: int) -> List[List[int]]:
        """Muestra n chunks aleatorios del buffer."""
        n = min(n, len(self.buffer))
        return random.sample(list(self.buffer), n)

    def __len__(self) -> int:
        return len(self.buffer)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"buffer": list(self.buffer), "total_seen": self.total_seen}, f)

    def load(self, path: str):
        import json
        from pathlib import Path
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
                self.buffer = deque(data["buffer"], maxlen=self.max_size)
                self.total_seen = data.get("total_seen", len(self.buffer))


class ContinualDataset(Dataset):
    """
    Dataset que combina datos nuevos + replay del pasado.
    Mecanismo principal anti-olvido catastrófico.
    """

    def __init__(self, new_chunks: List[List[int]], replay_chunks: List[List[int]],
                 max_seq_len: int = 512, tokenizer=None):
        self.max_seq_len = max_seq_len
        self.pad_id = tokenizer.pad_id if tokenizer else 0

        # Mezclar nuevo + replay
        all_chunks = new_chunks + replay_chunks
        random.shuffle(all_chunks)
        self.chunks = all_chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = self.chunks[idx]
        if len(ids) < self.max_seq_len:
            ids = ids + [self.pad_id] * (self.max_seq_len - len(ids))
        ids = torch.tensor(ids[:self.max_seq_len], dtype=torch.long)
        return ids[:-1], ids[1:]


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      num_workers: int = 0) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
