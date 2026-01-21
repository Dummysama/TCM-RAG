from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


CHUNKS_PATH = Path("outputs/knowledge_chunks.jsonl")
INDEX_DIR = Path("outputs/index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# 模型选择：中文/中英混合都能用，体积小、速度快
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
META_PATH = INDEX_DIR / "meta.jsonl"


def load_chunks(path: Path) -> List[Dict[str, Any]]:
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def build_embeddings(model: SentenceTransformer, texts: List[str], batch_size: int = 64) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # 归一化后可用 inner product 近似 cosine
    )
    return np.asarray(emb, dtype=np.float32)


def save_meta(path: Path, chunks: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            # 元数据留必要字段即可，避免太大
            out = {
                "id": c.get("id"),
                "type": c.get("type"),
                "text": c.get("text"),
                "metadata": c.get("metadata", {}),
            }
            f.write(json.dumps(out, ensure_ascii=False) + "\n")


def main():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Missing chunks file: {CHUNKS_PATH}")

    print("Loading chunks...")
    chunks = load_chunks(CHUNKS_PATH)
    texts = [c["text"] for c in chunks]

    print(f"Chunks loaded: {len(chunks)}")
    print(f"Loading embedding model: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Building embeddings...")
    emb = build_embeddings(model, texts)

    dim = emb.shape[1]
    print(f"Embedding shape: {emb.shape}")

    # 使用内积检索（因为已 normalize，相当于 cosine）
    index = faiss.IndexFlatIP(dim)
    index.add(emb)

    print("Saving index + metadata...")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    save_meta(META_PATH, chunks)

    print(f"Saved FAISS index: {FAISS_INDEX_PATH}")
    print(f"Saved metadata:  {META_PATH}")
    print("\nNext: run 05_search_demo.py to test retrieval.")


if __name__ == "__main__":
    main()