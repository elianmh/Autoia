"""
Recolector de datos sobre el tema específico.
Obtiene texto de múltiples fuentes: Wikipedia, web scraping, datasets públicos.
"""

import re
import json
import logging
import asyncio
import hashlib
from pathlib import Path
from typing import List, Optional, Set
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Un documento de texto sobre el tema."""
    text: str
    source: str
    topic_relevance: float = 1.0
    doc_id: str = ""

    def __post_init__(self):
        if not self.doc_id:
            self.doc_id = hashlib.md5(self.text[:200].encode()).hexdigest()[:12]


class TopicCrawler:
    """
    Recolecta datos de texto sobre el tema configurado.
    Fuentes: Wikipedia API, datasets de HuggingFace, archivos locales.
    """

    def __init__(self, topic: str, keywords: List[str], data_dir: str = "data"):
        self.topic = topic
        self.keywords = keywords
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.corpus_file = self.data_dir / "corpus.jsonl"
        self.seen_ids: Set[str] = set()
        self._load_seen_ids()

    def _load_seen_ids(self):
        """Carga IDs de documentos ya procesados para evitar duplicados."""
        if self.corpus_file.exists():
            with open(self.corpus_file) as f:
                for line in f:
                    try:
                        doc = json.loads(line)
                        self.seen_ids.add(doc.get("doc_id", ""))
                    except json.JSONDecodeError:
                        pass

    def _is_relevant(self, text: str) -> float:
        """Calcula relevancia del texto con respecto al tema (0-1)."""
        text_lower = text.lower()
        hits = sum(1 for kw in self.keywords if kw.lower() in text_lower)
        return min(hits / max(len(self.keywords) * 0.3, 1), 1.0)

    def _clean_text(self, text: str) -> str:
        """Limpia el texto eliminando ruido."""
        # Eliminar múltiples espacios/saltos
        text = re.sub(r"\s+", " ", text)
        # Eliminar referencias tipo [1], [2]...
        text = re.sub(r"\[\d+\]", "", text)
        # Eliminar URLs
        text = re.sub(r"http\S+", "", text)
        return text.strip()

    def fetch_wikipedia(self, max_articles: int = 50) -> List[Document]:
        """Obtiene artículos de Wikipedia sobre el tema."""
        try:
            import requests
        except ImportError:
            logger.warning("requests no disponible. Omitiendo Wikipedia.")
            return []

        documents = []
        search_terms = [self.topic] + self.keywords[:5]

        for term in search_terms:
            try:
                # Buscar artículos
                search_url = "https://en.wikipedia.org/w/api.php"
                params = {
                    "action": "query",
                    "list": "search",
                    "srsearch": term,
                    "srlimit": max_articles // len(search_terms),
                    "format": "json",
                    "srprop": "snippet",
                }
                resp = requests.get(search_url, params=params, timeout=10)
                resp.raise_for_status()
                results = resp.json().get("query", {}).get("search", [])

                for result in results:
                    page_id = str(result["pageid"])
                    # Obtener contenido completo
                    content_params = {
                        "action": "query",
                        "pageids": page_id,
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "format": "json",
                    }
                    content_resp = requests.get(search_url, params=content_params, timeout=10)
                    page_data = content_resp.json()
                    pages = page_data.get("query", {}).get("pages", {})

                    for _, page in pages.items():
                        text = page.get("extract", "")
                        if len(text) < 200:
                            continue
                        text = self._clean_text(text)
                        relevance = self._is_relevant(text)
                        if relevance < 0.1:
                            continue

                        doc = Document(
                            text=text,
                            source=f"wikipedia:{page.get('title', '')}",
                            topic_relevance=relevance,
                        )
                        if doc.doc_id not in self.seen_ids:
                            documents.append(doc)
                            self.seen_ids.add(doc.doc_id)

            except Exception as e:
                logger.warning(f"Error al buscar '{term}' en Wikipedia: {e}")

        logger.info(f"Wikipedia: {len(documents)} documentos nuevos obtenidos")
        return documents

    def fetch_huggingface_dataset(self, dataset_name: str = "wikitext",
                                   config: str = "wikitext-2-raw-v1",
                                   split: str = "train",
                                   max_samples: int = 5000) -> List[Document]:
        """Carga un dataset de HuggingFace y filtra por tema."""
        try:
            from datasets import load_dataset
        except ImportError:
            logger.warning("datasets no disponible.")
            return []

        documents = []
        try:
            logger.info(f"Cargando dataset {dataset_name}/{config}...")
            ds = load_dataset(dataset_name, config, split=split, streaming=True)

            count = 0
            for sample in ds:
                if count >= max_samples:
                    break

                text = sample.get("text", "")
                if len(text) < 100:
                    continue

                text = self._clean_text(text)
                relevance = self._is_relevant(text)
                if relevance < 0.05:
                    continue

                doc = Document(text=text, source=f"hf:{dataset_name}", topic_relevance=relevance)
                if doc.doc_id not in self.seen_ids:
                    documents.append(doc)
                    self.seen_ids.add(doc.doc_id)
                    count += 1

        except Exception as e:
            logger.warning(f"Error al cargar dataset {dataset_name}: {e}")

        logger.info(f"HuggingFace {dataset_name}: {len(documents)} documentos")
        return documents

    def load_local_files(self, directory: Optional[str] = None) -> List[Document]:
        """Carga archivos .txt y .md del directorio de datos."""
        search_dir = Path(directory) if directory else self.data_dir / "raw"
        if not search_dir.exists():
            return []

        documents = []
        for file_path in search_dir.glob("**/*.txt"):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                text = self._clean_text(text)
                if len(text) < 100:
                    continue
                doc = Document(text=text, source=f"local:{file_path.name}", topic_relevance=1.0)
                if doc.doc_id not in self.seen_ids:
                    documents.append(doc)
                    self.seen_ids.add(doc.doc_id)
            except Exception as e:
                logger.warning(f"Error al leer {file_path}: {e}")

        for file_path in search_dir.glob("**/*.md"):
            try:
                text = file_path.read_text(encoding="utf-8", errors="ignore")
                # Eliminar sintaxis markdown
                text = re.sub(r"[#*`_\[\]()]", "", text)
                text = self._clean_text(text)
                if len(text) < 100:
                    continue
                doc = Document(text=text, source=f"local:{file_path.name}", topic_relevance=1.0)
                if doc.doc_id not in self.seen_ids:
                    documents.append(doc)
                    self.seen_ids.add(doc.doc_id)
            except Exception as e:
                logger.warning(f"Error al leer {file_path}: {e}")

        logger.info(f"Archivos locales: {len(documents)} documentos")
        return documents

    def collect_all(self, use_wikipedia: bool = True,
                    use_hf: bool = True,
                    use_local: bool = True) -> List[Document]:
        """Recolecta datos de todas las fuentes configuradas."""
        all_docs = []

        if use_local:
            all_docs.extend(self.load_local_files())

        if use_wikipedia:
            all_docs.extend(self.fetch_wikipedia(max_articles=30))

        if use_hf:
            all_docs.extend(
                self.fetch_huggingface_dataset(
                    "wikitext", "wikitext-103-raw-v1", max_samples=2000
                )
            )

        # Ordenar por relevancia
        all_docs.sort(key=lambda d: d.topic_relevance, reverse=True)
        logger.info(f"Total documentos recolectados: {len(all_docs)}")
        return all_docs

    def save_corpus(self, documents: List[Document]) -> int:
        """Guarda documentos en el corpus (append mode). Retorna total guardado."""
        saved = 0
        with open(self.corpus_file, "a", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps({
                    "doc_id": doc.doc_id,
                    "text": doc.text,
                    "source": doc.source,
                    "relevance": doc.topic_relevance,
                }, ensure_ascii=False) + "\n")
                saved += 1
        return saved

    def load_corpus(self, min_relevance: float = 0.0) -> List[str]:
        """Carga todos los textos del corpus para entrenamiento."""
        texts = []
        if not self.corpus_file.exists():
            return texts
        with open(self.corpus_file, encoding="utf-8") as f:
            for line in f:
                try:
                    doc = json.loads(line)
                    if doc.get("relevance", 0) >= min_relevance:
                        texts.append(doc["text"])
                except json.JSONDecodeError:
                    pass
        return texts

    def corpus_size(self) -> int:
        """Número de documentos en el corpus."""
        if not self.corpus_file.exists():
            return 0
        count = 0
        with open(self.corpus_file) as f:
            for _ in f:
                count += 1
        return count
