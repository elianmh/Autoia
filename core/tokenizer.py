"""
Tokenizer BPE (Byte-Pair Encoding) entrenado sobre el corpus del tema específico.
Puede re-entrenarse cuando el vocabulario crece con nuevo conocimiento.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class AutoiaTokenizer:
    """
    Wrapper del tokenizer BPE de Hugging Face Tokenizers.
    Entrenado específicamente sobre el corpus del tema.
    Puede crecer su vocabulario con nuevos datos.
    """

    SPECIAL_TOKENS = {
        "pad": "<PAD>",
        "unk": "<UNK>",
        "bos": "<BOS>",
        "eos": "<EOS>",
        "sep": "<SEP>",
        "mask": "<MASK>",
        "topic": "<TOPIC>",      # Marca el inicio del contexto del tema
        "query": "<QUERY>",      # Pregunta del usuario
        "response": "<RESPONSE>",  # Respuesta del modelo
    }

    def __init__(self, vocab_size: int = 8000, save_dir: str = "data"):
        self.vocab_size = vocab_size
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer = None
        self._path = self.save_dir / "tokenizer.json"

        if self._path.exists():
            self.load()

    def train(self, texts: List[str], min_frequency: int = 2) -> None:
        """Entrena el tokenizer BPE sobre los textos dados."""
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors, decoders

        tokenizer = Tokenizer(models.BPE(unk_token=self.SPECIAL_TOKENS["unk"]))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        special_tokens = list(self.SPECIAL_TOKENS.values())

        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=min_frequency,
            special_tokens=special_tokens,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)

        # Post-procesador para BOS/EOS automáticos
        bos_id = tokenizer.token_to_id(self.SPECIAL_TOKENS["bos"])
        eos_id = tokenizer.token_to_id(self.SPECIAL_TOKENS["eos"])
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"<BOS> $A <EOS>",
            pair=f"<BOS> $A <SEP> $B:1 <EOS>:1",
            special_tokens=[
                ("<BOS>", bos_id),
                ("<EOS>", eos_id),
                ("<SEP>", tokenizer.token_to_id(self.SPECIAL_TOKENS["sep"])),
            ],
        )

        self._tokenizer = tokenizer
        self.save()
        actual_vocab = tokenizer.get_vocab_size()
        logger.info(f"Tokenizer entrenado. Vocabulario: {actual_vocab:,} tokens")

    def retrain_expand(self, new_texts: List[str], new_vocab_size: int) -> bool:
        """
        Re-entrena el tokenizer con más vocabulario cuando el modelo crece.
        Retorna True si el vocabulario realmente creció.
        """
        if self._tokenizer is None:
            self.train(new_texts)
            return True

        old_vocab = self.vocab_size
        self.vocab_size = new_vocab_size
        self.train(new_texts, min_frequency=2)
        logger.info(f"Vocabulario expandido: {old_vocab} -> {new_vocab_size}")
        return new_vocab_size > old_vocab

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Codifica texto a IDs de tokens."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer no entrenado. Llama train() primero.")
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decodifica IDs de tokens a texto."""
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer no entrenado.")
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer no entrenado.")
        encodings = self._tokenizer.encode_batch(texts)
        return [e.ids for e in encodings]

    @property
    def pad_id(self) -> int:
        return self._tokenizer.token_to_id(self.SPECIAL_TOKENS["pad"])

    @property
    def bos_id(self) -> int:
        return self._tokenizer.token_to_id(self.SPECIAL_TOKENS["bos"])

    @property
    def eos_id(self) -> int:
        return self._tokenizer.token_to_id(self.SPECIAL_TOKENS["eos"])

    @property
    def actual_vocab_size(self) -> int:
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    def save(self):
        if self._tokenizer:
            self._tokenizer.save(str(self._path))
            # Guardar metadatos
            meta = {"vocab_size": self.vocab_size, "special_tokens": self.SPECIAL_TOKENS}
            with open(self.save_dir / "tokenizer_meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    def load(self):
        from tokenizers import Tokenizer
        self._tokenizer = Tokenizer.from_file(str(self._path))
        meta_path = self.save_dir / "tokenizer_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                self.vocab_size = meta.get("vocab_size", self.vocab_size)
        logger.info(f"Tokenizer cargado desde {self._path}")
