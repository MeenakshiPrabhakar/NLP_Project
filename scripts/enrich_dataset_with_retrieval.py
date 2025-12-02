"""
Precompute retrieved context for supervised data (Financial PhraseBank, etc.).

Given a TSV/CSV with columns: id, sentence, label, [agree], this script:
1) Embeds each sentence.
2) Retrieves top-k chunks from a FAISS index built with build_retrieval_index.py.
3) Writes an enriched TSV with an extra "context" column.

Example:
python scripts/enrich_dataset_with_retrieval.py \
  --input_path data/sentiment_data/train.csv \
  --output_path data/sentiment_data/train_rag.csv \
  --index_dir data/trc2_index \
  --top_k 3 \
  --model sentence-transformers/all-MiniLM-L6-v2
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

try:
    import faiss  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("faiss is required. Install with `pip install faiss-cpu`.") from exc

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit("sentence-transformers is required. Install with `pip install sentence-transformers`.") from exc


def load_chunks(index_dir: Path) -> List[str]:
    """Load chunk texts from chunks.jsonl."""
    chunks_path = index_dir / "chunks.jsonl"
    texts: List[str] = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def retrieve_contexts(
    sentences: List[str],
    model: SentenceTransformer,
    index: faiss.Index,
    chunk_texts: List[str],
    top_k: int,
    normalize: bool,
) -> List[str]:
    """Embed sentences, retrieve top-k chunk texts, and concatenate them."""
    embeddings = model.encode(
        sentences,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    ).astype("float32")
    if normalize:
        faiss.normalize_L2(embeddings)

    scores, indices = index.search(embeddings, top_k)
    contexts: List[str] = []
    for row in indices:
        snippets = [chunk_texts[i] for i in row if i < len(chunk_texts)]
        contexts.append(" [SEP] ".join(snippets))
    return contexts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich supervised data with retrieved context.")
    parser.add_argument("--input_path", type=Path, required=True, help="Input TSV/CSV with columns: id, sentence, label, [agree].")
    parser.add_argument("--output_path", type=Path, required=True, help="Output TSV with added context column.")
    parser.add_argument("--index_dir", type=Path, required=True, help="Directory containing index.faiss, chunks.jsonl, index_meta.json.")
    parser.add_argument("--model", type=str, default=None, help="SentenceTransformer model name. Defaults to index_meta.json['model'].")
    parser.add_argument("--top_k", type=int, default=3, help="Top-k passages to concatenate.")
    parser.add_argument("--delimiter", type=str, default="\t", help="Delimiter for input/output files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta_path = args.index_dir / "index_meta.json"
    if not meta_path.exists():
        raise SystemExit(f"index_meta.json not found in {args.index_dir}")

    meta = json.loads(meta_path.read_text())
    model_name = args.model or meta.get("model") or "sentence-transformers/all-MiniLM-L6-v2"
    normalize = bool(meta.get("normalize", True))

    print(f"Loading FAISS index from {args.index_dir} ...")
    index = faiss.read_index(str(args.index_dir / "index.faiss"))
    chunk_texts = load_chunks(args.index_dir)
    model = SentenceTransformer(model_name)

    print(f"Reading input data: {args.input_path}")
    rows = []
    with args.input_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=args.delimiter)
        rows = list(reader)
    if not rows:
        raise SystemExit("No rows found in input.")

    header = rows[0]
    data_rows = rows[1:]

    sentences = [r[1] for r in data_rows]
    contexts = retrieve_contexts(sentences, model, index, chunk_texts, args.top_k, normalize)

    # Append context column
    header_out = header + ["context"]
    enriched = [header_out]
    for row, ctx in tqdm(zip(data_rows, contexts), total=len(data_rows), desc="Enriching"):
        enriched.append(row + [ctx])

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=args.delimiter)
        writer.writerows(enriched)

    print(f"Wrote enriched data to {args.output_path}")


if __name__ == "__main__":
    main()
