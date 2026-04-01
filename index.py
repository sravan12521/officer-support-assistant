# index.py
# Builds:
# - data/faiss.index
# - data/metadata.jsonl
# - data/bm25.json  (BM25 corpus + tokens)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

CHUNKS_FILE = Path("data/chunks.jsonl")
INDEX_FILE = Path("data/faiss.index")
META_FILE = Path("data/metadata.jsonl")
BM25_FILE = Path("data/bm25.json")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "index.log"

logger = logging.getLogger("index")
logger.setLevel(logging.INFO)
fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(fmt)
logger.addHandler(fh)
ch = logging.StreamHandler()
ch.setFormatter(fmt)
logger.addHandler(ch)

def load_chunks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Run ingest.py first. Missing: {path}")
    chunks = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks

def tokenize(text: str) -> List[str]:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\-\. ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def build_index():
    chunks = load_chunks(CHUNKS_FILE)
    logger.info(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c.get("text", "") or "" for c in chunks]

    logger.info("Generating embeddings...")
    emb = model.encode(
        texts,
        convert_to_numpy=True,
        batch_size=48,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype("float32")

    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    faiss.write_index(index, str(INDEX_FILE))
    logger.info(f"Saved FAISS -> {INDEX_FILE}")

    with META_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    logger.info(f"Saved metadata -> {META_FILE}")

    # BM25
    logger.info("Building BM25...")
    tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)

    # Save minimal BM25 payload (tokens + id mapping)
    # We re-create BM25 at query time from tokens (fast for a few thousand chunks).
    payload = {
        "tokenized": tokenized,
    }
    with BM25_FILE.open("w", encoding="utf-8") as f:
        json.dump(payload, f)
    logger.info(f"Saved BM25 -> {BM25_FILE}")

    logger.info(f"Log -> {LOG_FILE}")

if __name__ == "__main__":
    build_index()