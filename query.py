# query.py — Officer Support Assistant (CLI) — HYBRID (FAISS + BM25)

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI

from synonyms import expand_query, debug_matched_keys

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "metadata.jsonl"
BM25_PATH = DATA_DIR / "bm25.json"

LOG_FILE = LOG_DIR / "query.log"

# -----------------------------
# LOGGING
# -----------------------------
logger = logging.getLogger("rag_query")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# -----------------------------
# OPENAI + EMBEDDER
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found. Set it in .env or environment.")

client = OpenAI(api_key=api_key)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)

# -----------------------------
# LOAD INDEX + METADATA + BM25
# -----------------------------
if not INDEX_PATH.exists():
    raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run index.py first.")
if not META_PATH.exists():
    raise FileNotFoundError(f"Metadata not found at {META_PATH}. Run index.py first.")
if not BM25_PATH.exists():
    raise FileNotFoundError(f"BM25 not found at {BM25_PATH}. Run index.py first.")

print("Loading FAISS index...")
index = faiss.read_index(str(INDEX_PATH))
print(f"Index loaded with {index.ntotal} vectors.")

print("Loading metadata...")
metadata: List[Dict[str, Any]] = []
with META_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))
print(f"Loaded {len(metadata)} metadata entries.")

print("Loading BM25 corpus...")
bm25_data = json.loads(BM25_PATH.read_text(encoding="utf-8"))
tokenized_corpus = bm25_data["tokenized_corpus"]
bm25 = BM25Okapi(tokenized_corpus)
print("BM25 loaded.\n")

# -----------------------------
# HELPERS
# -----------------------------
def bm25_tokenize(text: str) -> List[str]:
    import re
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\-\.\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.split()

def source_type(src: str) -> str:
    s = (src or "").lower()
    if s.startswith("ne_statute"):
        return "statute"
    if s.startswith("case_") or " v. " in s or "caselaw" in s or "case law" in s:
        return "case_law"
    return "policy"

# -----------------------------
# INTENT DETECTION
# -----------------------------
AUTHORITY_TRIGGERS = [
    "what authority applies",
    "remind me",
    "law has changed",
    "case law",
    "controlling",
    "threshold",
    "articulate",
    "post-hoc",
    "exigent",
    "warrantless",
    "community caretaking",
]

def detect_intent(question: str) -> str:
    q = question.lower()
    for t in AUTHORITY_TRIGGERS:
        if t in q:
            return "authority_reminder"
    return "general"

# -----------------------------
# HYBRID RETRIEVAL
# -----------------------------
def hybrid_retrieve(question: str, k: int = 8, alpha: float = 0.6):
    """
    alpha: weight for FAISS (semantic). (1-alpha): weight for BM25 (keyword)
    Returns: expanded_query, matched_keys, merged_results
    """
    expanded = expand_query(question, max_expansions_per_key=8, include_keys=False)
    matched_keys = debug_matched_keys(question)

    # FAISS search
    qvec = embedder.encode([expanded], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    faiss_scores, faiss_idx = index.search(qvec, k * 4)  # pull more to merge well
    faiss_scores = faiss_scores[0]
    faiss_idx = faiss_idx[0]

    # BM25 search
    q_tokens = bm25_tokenize(expanded)
    bm25_scores = bm25.get_scores(q_tokens)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][: k * 4]

    # Normalize scores into 0..1
    def norm(arr):
        arr = np.array(arr, dtype=float)
        if len(arr) == 0:
            return arr
        mn, mx = float(arr.min()), float(arr.max())
        if mx - mn < 1e-9:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)

    faiss_norm = norm(faiss_scores)
    bm25_norm = norm(bm25_scores)

    # Merge by doc index
    combined: Dict[int, float] = {}

    for s, idx_ in zip(faiss_norm, faiss_idx):
        if idx_ < 0:
            continue
        combined[int(idx_)] = combined.get(int(idx_), 0.0) + alpha * float(s)

    for idx_ in bm25_top_idx:
        combined[int(idx_)] = combined.get(int(idx_), 0.0) + (1 - alpha) * float(bm25_norm[int(idx_)])

    # Sort
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:k]

    results = []
    for idx_, score in ranked:
        row = dict(metadata[idx_])
        row["_score"] = float(score)
        row["_type"] = source_type(row.get("source", ""))
        results.append(row)

    # logging
    logger.info("QUESTION: %s", question)
    logger.info("EXPANDED_QUERY: %s", expanded)
    logger.info("MATCHED_KEYS: %s", ", ".join(matched_keys) if matched_keys else "none")
    logger.info("TOP_K: %s", k)
    for r in results:
        logger.info("HIT score=%.4f type=%s source=%s page=%s text_preview=%s",
                    r.get("_score", 0.0),
                    r.get("_type", "unknown"),
                    r.get("source", "unknown"),
                    r.get("page", 1),
                    (r.get("text", "")[:160] + "...").replace("\n", " "))
    logger.info("----")

    return expanded, matched_keys, results

# -----------------------------
# PROMPTS
# -----------------------------
DOCTRINE_FLAG = (
    "Recent controlling case law applies. Assume training and policy may lag doctrine. "
    "If retrieved policy conflicts with controlling case law, controlling law governs."
)

ROLE_LOCK_AUTHORITY = (
    "Respond as a seasoned front-seat sergeant. Assume officer competence. "
    "Do not teach procedures. Focus on legal authority thresholds and articulation under post-hoc scrutiny."
)

OUTPUT_CONTRACT_AUTHORITY = """Format strictly as:
1) Top-line authority reminder (2–4 sentences)
2) Watch-outs (2–4 bullets)
3) “Next — what would help right now?” (menu only)

Prohibited:
- Step-by-step procedures
- Policy summaries
- Mental health guidance
- Tactical advice
- Follow-up fact-gathering questions
"""

GENERAL_SYSTEM = (
    "You are the Officer Support Assistant for Sarpy County law enforcement. "
    "Answer ONLY using the provided context. "
    "If the answer is not clearly supported by the context, you MUST say: "
    "\"I don’t see that in the documents.\" "
    "Include citations like (Policy.pdf p.3) or (NE_Statute_28-105) or (Case_Name p.2)."
)

# -----------------------------
# LLM ANSWER
# -----------------------------
def ask_llm(question: str, k: int = 8, alpha: float = 0.6):
    intent = detect_intent(question)
    mode = "Authority Reminder" if intent == "authority_reminder" else "General"

    expanded, matched_keys, chunks = hybrid_retrieve(question, k=k, alpha=alpha)

    logger.info("MODE: %s", mode)
    logger.info("AUTO_INTENT: %s", intent)
    logger.info("FINAL_INTENT: %s", intent)

    if not chunks:
        return "I don’t see that in the documents.", []

    context = "\n\n".join(
        f"Source: {c.get('source','unknown')} (p.{c.get('page',1)})\n{c.get('text','')}"
        for c in chunks
    )

    if intent == "authority_reminder":
        system_msg = f"{DOCTRINE_FLAG}\n\n{ROLE_LOCK_AUTHORITY}\n\n{OUTPUT_CONTRACT_AUTHORITY}"
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}\n\nFollow OUTPUT_CONTRACT strictly."
    else:
        system_msg = GENERAL_SYSTEM
        user_msg = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=700,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    return resp.choices[0].message.content.strip(), chunks

# -----------------------------
# CLI LOOP
# -----------------------------
if __name__ == "__main__":
    print("Officer Support Assistant — HYBRID (FAISS+BM25)")
    print("Logs saved to:", LOG_FILE)
    print("Type 'exit' to quit.\n")

    while True:
        q = input("Ask a question (or 'exit'): ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break

        print("\nThinking...\n")
        answer, retrieved = ask_llm(q, k=8, alpha=0.6)
        print("Answer:\n")
        print(answer)
        print("\n------ Top Sources ------")
        for c in retrieved[:8]:
            print("- {} (p.{}) score={:.4f} type={}".format(
                c.get("source", "unknown"),
                c.get("page", 1),
                c.get("_score", 0.0),
                c.get("_type", "unknown"),
            ))
        print("-------------------------\n")