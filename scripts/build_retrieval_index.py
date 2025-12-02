"""
Build a FAISS retrieval index over TRC2 (or any) financial text corpus.

Steps:
1) Chunk documents into overlapping passages.
2) Embed passages with a SentenceTransformer model.
3) Save FAISS index + chunk metadata for later retrieval.

Example:
python scripts/build_retrieval_index.py \\
    --input_dir /path/to/trc2/texts \\
    --output_dir data/trc2_index \\
    --model sentence-transformers/all-MiniLM-L6-v2 \\
    --chunk_size 200 --overlap 50 --top_k 5
"""

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "faiss is required. Install with `pip install faiss-cpu` (or faiss-gpu if available)."
    ) from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover - dependency hint
    raise SystemExit(
        "sentence-transformers is required. Install with `pip install sentence-transformers`."
    ) from exc


def iter_text_files(input_dir: Path) -> Iterable[Path]:
    """Yield all .txt files under input_dir (recursively)."""
    for path in input_dir.rglob("*.txt"):
        if path.is_file():
            yield path


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[Tuple[int, int, str]]:
    """
    Split text into word-based chunks with overlap.
    Returns list of tuples: (start_idx, end_idx, chunk_text).
    """
    tokens = text.split()
    if not tokens:
        return []

    stride = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(tokens), stride):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        if not chunk_tokens:
            continue
        chunks.append((start, end, " ".join(chunk_tokens)))
        if end == len(tokens):
            break
    return chunks


def load_chunks(input_dir: Path, chunk_size: int, overlap: int) -> List[dict]:
    """Load all text files, chunk them, and return metadata dicts."""
    all_chunks: List[dict] = []
    chunk_id = 0
    for file_path in tqdm(list(iter_text_files(input_dir)), desc="Chunking documents"):
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        for idx, (start, end, chunk) in enumerate(chunk_text(text, chunk_size, overlap)):
            all_chunks.append(
                {
                    "id": chunk_id,
                    "source": str(file_path),
                    "chunk_index": idx,
                    "start_word": start,
                    "end_word": end,
                    "text": chunk,
                }
            )
            chunk_id += 1
    return all_chunks


def embed_chunks(chunks: List[dict], model_name: str, batch_size: int, normalize: bool) -> np.ndarray:
    """Embed chunk texts with SentenceTransformer."""
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=normalize,
    )
    if normalize and embeddings.ndim == 2:
        # normalize_embeddings already normalizes, but guard for consistency
        faiss.normalize_L2(embeddings)
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray, metric: str) -> faiss.Index:
    """Create a FAISS index from embeddings."""
    dim = embeddings.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")
    index.add(embeddings)
    return index


def save_outputs(output_dir: Path, chunks: List[dict], embeddings: np.ndarray, index: faiss.Index, meta: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # Chunk metadata
    chunks_path = output_dir / "chunks.jsonl"
    with chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=True) + "\n")

    # Embeddings (optional to inspect/debug)
    np.save(output_dir / "embeddings.npy", embeddings)

    # FAISS index
    faiss.write_index(index, str(output_dir / "index.faiss"))

    # Meta/config
    with (output_dir / "index_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build FAISS retrieval index over TRC2-style corpus.")
    parser.add_argument("--input_dir", type=Path, required=True, help="Directory with raw .txt files (TRC2).")
    parser.add_argument("--output_dir", type=Path, required=True, help="Where to write index and metadata.")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name.",
    )
    parser.add_argument("--chunk_size", type=int, default=200, help="Words per chunk.")
    parser.add_argument("--overlap", type=int, default=50, help="Overlapping words between chunks.")
    parser.add_argument("--batch_size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument(
        "--metric",
        type=str,
        default="ip",
        choices=["ip", "l2"],
        help="Similarity metric: 'ip' for cosine/inner-product, 'l2' for Euclidean.",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="L2-normalize embeddings (recommended for cosine / inner product).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.exists():
        raise SystemExit(f"Input dir not found: {args.input_dir}")

    print("Loading and chunking documents...")
    chunks = load_chunks(args.input_dir, args.chunk_size, args.overlap)
    if not chunks:
        raise SystemExit("No chunks created. Check input_dir and file format.")

    print(f"Embedding {len(chunks)} chunks with {args.model} ...")
    embeddings = embed_chunks(chunks, args.model, args.batch_size, args.normalize or args.metric == "ip")

    print("Building FAISS index...")
    index = build_faiss_index(embeddings, args.metric)

    meta = {
        "model": args.model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "metric": args.metric,
        "normalize": args.normalize or args.metric == "ip",
        "num_chunks": len(chunks),
        "dim": int(embeddings.shape[1]),
    }
    save_outputs(args.output_dir, chunks, embeddings, index, meta)
    print(f"Saved index to {args.output_dir}")


if __name__ == "__main__":
    main()
