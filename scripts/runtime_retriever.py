"""
Lightweight retriever for inference-time context lookup.

Usage:
    from scripts.runtime_retriever import FaissRetriever
    retriever = FaissRetriever(index_dir="data/trc2_index",
                               model_name="sentence-transformers/all-MiniLM-L6-v2",
                               top_k=3)
    contexts = retriever(["Market outlook is improving", "Inflation remains elevated"])
"""

import json
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss is required. Install with `pip install faiss-cpu`.") from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sentence-transformers is required. Install with `pip install sentence-transformers`.") from exc


class FaissRetriever:
    """Callable retriever: sentences -> context strings."""

    def __init__(self, index_dir: str, model_name: str = None, top_k: int = 3):
        self.index_dir = Path(index_dir)
        meta_path = self.index_dir / "index_meta.json"
        if not meta_path.exists():
            raise ValueError(f"index_meta.json not found in {self.index_dir}")
        meta = json.loads(meta_path.read_text())
        self.model_name = model_name or meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
        self.normalize = bool(meta.get("normalize", True))
        self.top_k = top_k

        self.index = faiss.read_index(str(self.index_dir / "index.faiss"))
        self.chunk_texts = self._load_chunks()
        self.model = SentenceTransformer(self.model_name)

    def _load_chunks(self) -> List[str]:
        texts: List[str] = []
        chunks_path = self.index_dir / "chunks.jsonl"
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                texts.append(obj["text"])
        return texts

    def __call__(self, sentences: Iterable[str]) -> List[str]:
        sentences_list = list(sentences)
        embeddings = self.model.encode(
            sentences_list,
            batch_size=64,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=self.normalize,
        ).astype("float32")
        if self.normalize:
            faiss.normalize_L2(embeddings)
        _, indices = self.index.search(embeddings, self.top_k)
        contexts: List[str] = []
        for row in indices:
            snippets = [self.chunk_texts[i] for i in row if i < len(self.chunk_texts)]
            contexts.append(" [SEP] ".join(snippets))
        return contexts
