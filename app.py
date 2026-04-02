# app.py — Officer Support Assistant (Sarpy County | Streamlit)
# UX / Retrieval upgrades (KP UX NOTES):
# - Tiered workflow display: Immediate → Evidence → Interview → Policy → Law → Documentation → Command → Checklist
# - Avoid overload: show first 3–5 bullets in each section, rest collapsible
# - Retrieval fix: FAISS + BM25 hybrid retrieval to reduce "I don’t see that..." false negatives

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import re
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi

from synonyms import expand_query, debug_matched_keys


# =============================
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Officer Support Assistant",
    page_icon="🚓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# LOGGING (app)
# =============================
BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

APP_LOG = LOG_DIR / "app.log"
QUERY_LOG_CSV = LOG_DIR / "query_eval_log.csv"

logger = logging.getLogger("rag_app")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(APP_LOG, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# =============================
# ENV + CLIENT
# =============================
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found.")
    st.stop()

client = OpenAI(api_key=api_key)

INDEX_PATH = BASE_DIR / "data" / "faiss.index"
META_PATH = BASE_DIR / "data" / "metadata.jsonl"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

TOP_K_FAISS = 18
TOP_K_BM25 = 18
TOP_K_FINAL = 12

# =============================
# SESSION STATE
# =============================
if "chat" not in st.session_state:
    st.session_state.chat = []
if "sources" not in st.session_state:
    st.session_state.sources = []
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "general"

# Panel heights
LEFT_PANEL_HEIGHT = 720
CENTER_PANEL_HEIGHT = 650
RIGHT_PANEL_HEIGHT = 720
CHAT_HISTORY_HEIGHT = 90 if len(st.session_state.get("chat", [])) == 0 else 380

# =============================
# CACHED LOADERS
# =============================
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource(show_spinner="Loading index + metadata...")
def load_index_and_meta():
    if not INDEX_PATH.exists() or not META_PATH.exists():
        st.error("Missing index/metadata. Run ingest.py then index.py first.")
        st.stop()

    idx = faiss.read_index(str(INDEX_PATH))
    meta: List[Dict[str, Any]] = []
    with META_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))
    return idx, meta

@st.cache_resource(show_spinner="Building BM25 keyword index...")
def build_bm25(meta: List[Dict[str, Any]]):
    corpus_tokens = []
    for m in meta:
        text = (m.get("text") or "")
        src = (m.get("source") or "")
        joined = f"{src}\n{text}".lower()
        tokens = re.findall(r"[a-z0-9_'-]+", joined)
        corpus_tokens.append(tokens)
    return BM25Okapi(corpus_tokens)

@st.cache_resource(show_spinner="Loading retrieval resources...")
def load_all_resources():
    embedder_local = load_embedder()
    index_local, metadata_local = load_index_and_meta()
    bm25_local = build_bm25(metadata_local)
    return embedder_local, index_local, metadata_local, bm25_local

# Lazy-loaded globals
embedder = None
index = None
metadata = None
bm25 = None

# =============================
# QUERY EVAL LOGGING
# =============================
def save_eval_row(
    question: str,
    answer: str,
    intent: str,
    mode: str,
    user_role: str,
    response_time_sec: float,
    chunks: List[Dict[str, Any]],
):
    QUERY_LOG_CSV.parent.mkdir(exist_ok=True)

    source_summary = " || ".join(
        f"{c.get('source','unknown')}|p{c.get('page',1)}|type={c.get('_type','unknown')}|faiss={c.get('_score',-1):.3f}|bm25={c.get('_bm25',0):.2f}"
        for c in (chunks or [])
    )

    row = [
        datetime.utcnow().isoformat(),
        user_role,
        mode,
        intent,
        round(response_time_sec, 3),
        question.replace("\n", " ").strip(),
        answer.replace("\n", " ").strip(),
        source_summary,
        "",
        "",
    ]

    file_exists = QUERY_LOG_CSV.exists()
    with open(QUERY_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp_utc",
                "user_role",
                "mode",
                "intent",
                "response_time_sec",
                "question",
                "answer",
                "sources",
                "human_rating",
                "notes",
            ])
        writer.writerow(row)

# =============================
# SCORE BUCKETS
# =============================
def faiss_score_category(score: float) -> str:
    try:
        s = float(score)
    except Exception:
        return "Unknown"

    if s >= 0.58:
        return "Excellent"
    if 0.48 <= s <= 0.57:
        return "Relevant"
    if 0.40 <= s <= 0.47:
        return "Weak"
    return "Likely Unrelated"

def score_badge_text(score: float) -> str:
    return f"{faiss_score_category(score)}"

# =============================
# TYPE
# =============================
def source_type(src: str) -> str:
    s = (src or "").lower()
    if s.startswith("ne_statute"):
        return "statute"
    if s.startswith("case_") or " v. " in s or "v." in s or "caselaw" in s or "case law" in s:
        return "case_law"
    return "policy"

# =============================
# INTENT DETECTION
# =============================
def detect_intent(question: str) -> str:
    q = (question or "").lower()

    authority_triggers = [
        "what authority applies", "law has changed", "remind me", "threshold",
        "articulate", "community caretaking", "warrantless", "exigent",
        "post-hoc", "controlling",
    ]
    policy_triggers = [
        "what policy", "what policies", "what statute", "what statutes",
        "policy and statute", "policies and statutes", "policies & statutes",
        "applicable policy", "applicable policies", "applicable statute", "applicable statutes",
    ]
    investigative_triggers = [
        "juvenile", "sextortion", "blackmail", "extortion",
        "instagram", "snapchat", "discord", "tiktok",
        "online threats", "harassment", "child exploitation",
        "intimate pictures", "nudes", "meet up", "outing",
        "same sex", "lgbtq", "threatened to tell", "expose",
    ]

    for t in authority_triggers:
        if t in q:
            return "authority_reminder"
    for t in policy_triggers:
        if t in q:
            return "policy_lookup"
    for t in investigative_triggers:
        if t in q:
            return "investigative_guidance"
    return "general"

# =============================
# PROMPT CONTRACTS
# =============================
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

OUTPUT_CONTRACT_POLICY = """Answer strictly as:
1) Applicable policies (list with citation)
2) Applicable statutes (list with citation)
3) Applicable case law (list with citation)  [only if present in context]

Rules:
- Only list items that explicitly govern the described scenario.
- Do NOT summarize unrelated policies.
- Do NOT infer authority.
- If nothing directly applicable exists, say exactly:
  "I don’t see a policy, statute, or case law that directly governs this scenario."
"""

ROLE_LOCK_INVESTIGATIVE = (
    "Respond as an experienced investigations sergeant. Assume officer competence. "
    "Be practical, evidence-driven, and policy/statute/case-law grounded. "
    "Your priority is preservation of evidence + proper escalation/coordination."
)

OUTPUT_CONTRACT_INVESTIGATIVE = """Format strictly as:

Immediate actions:
- (3–5 bullets max; each bullet MUST include at least one citation)

Evidence preservation:
- (3–5 bullets max; each bullet MUST include at least one citation)

Interview / notification strategy:
- (3–5 bullets max; each bullet MUST include at least one citation)

Policy/procedure requirements:
- (2–5 bullets max; each bullet MUST include at least one citation)

Criminal law hooks (statutes/case law):
- (2–5 bullets max; each bullet MUST include at least one citation)

Documentation watch-outs:
- (2–5 bullets max; each bullet MUST include at least one citation)

Chain of command / coordination:
- (1–4 bullets max; each bullet MUST include at least one citation)

Checklist (if available):
- (bullets; each bullet MUST include at least one citation)

Priority rules (must follow):
- Use ONLY the provided context.
- Every bullet MUST include at least one citation.
- If a bullet cannot be supported by citations, OMIT it (do not guess).
- If you cannot support ANY section at all, reply exactly: "I don’t see that in the documents."

Scenario priorities (apply when relevant to sextortion/cyber cases, but ONLY if supported by context):
- Preserve the communication channel; do NOT advise immediate blocking.
- Do NOT engage the suspect yet (unless the context explicitly directs it).
- Contact/loop in the On-Call Investigator early; note any “stall vs seize device” tradeoff if supported.
- Secure devices for forensics; do NOT browse extensively or attempt on-scene deleted recovery that alters metadata.
- For cryptocurrency: focus on transaction ID + destination wallet; do NOT solicit a new wallet/address.

Prohibited:
- Tactical advice
- Mental health counseling
"""

GENERAL_SYSTEM = (
    "You are the Officer Support Assistant for the Sarpy County Sheriff's Office. "
    "Answer ONLY using the provided context. "
    "If the answer is not explicitly supported, reply exactly: "
    "\"I don’t see that in the documents.\" "
    "Always include citations like (filename.pdf p.3) or (NE_Statute_28-105) or (Case_Name p.2)."
)

# =============================
# HYBRID RETRIEVAL
# =============================
def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_'-]+", (text or "").lower())

def retrieve(question: str, mode: str = "All Sources") -> Tuple[str, List[str], List[Dict[str, Any]]]:
    global embedder, index, metadata, bm25

    if embedder is None or index is None or metadata is None or bm25 is None:
        embedder, index, metadata, bm25 = load_all_resources()

    expanded = expand_query(question, max_expansions_per_key=8, include_keys=False)
    matched_keys = debug_matched_keys(question)

    vec = embedder.encode([expanded], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(vec, TOP_K_FAISS)

    faiss_hits = []
    for score, idx_ in zip(D[0], I[0]):
        if 0 <= idx_ < len(metadata):
            faiss_hits.append((int(idx_), float(score)))

    q_tokens = _tokenize_for_bm25(expanded)
    bm25_scores = bm25.get_scores(q_tokens)
    top_bm25_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:TOP_K_BM25]
    bm25_hits = [(int(i), float(bm25_scores[i])) for i in top_bm25_idx if bm25_scores[i] > 0]

    merged: Dict[int, Dict[str, float]] = {}
    for idx_, s in faiss_hits:
        merged.setdefault(idx_, {})["faiss"] = s
    for idx_, s in bm25_hits:
        merged.setdefault(idx_, {})["bm25"] = s

    def rank_key(item):
        idx_, scores = item
        return (
            scores.get("faiss", -1e9),
            scores.get("bm25", -1e9),
        )

    ranked = sorted(merged.items(), key=rank_key, reverse=True)

    results: List[Dict[str, Any]] = []
    for idx_, scores in ranked:
        item = dict(metadata[idx_])
        item["_type"] = source_type(item.get("source", ""))
        item["_score"] = float(scores.get("faiss", -1.0))
        item["_bm25"] = float(scores.get("bm25", 0.0))

        if mode == "Policies Only" and item["_type"] != "policy":
            continue
        if mode == "Statutes Only" and item["_type"] != "statute":
            continue

        results.append(item)
        if len(results) >= TOP_K_FINAL:
            break

    return expanded, matched_keys, results

# =============================
# SECTION PARSER
# =============================
SECTION_ORDER = [
    ("Immediate actions", "Immediate actions", 5),
    ("Evidence preservation", "Evidence preservation", 5),
    ("Interview / notification strategy", "Interview / notification strategy", 5),
    ("Policy/procedure requirements", "Policy/procedure requirements", 5),
    ("Criminal law hooks", "Criminal law hooks (statutes/case law)", 5),
    ("Documentation watch-outs", "Documentation watch-outs", 5),
    ("Chain of command / coordination", "Chain of command / coordination", 4),
    ("Checklist", "Checklist (if available)", 8),
]

def parse_sections(answer: str) -> Tuple[Optional[str], Dict[str, str]]:
    if not answer:
        return None, {}

    text = answer.strip().replace("\r\n", "\n")

    variants = {
        "Immediate actions": ["Immediate actions", "Immediate Actions"],
        "Evidence preservation": ["Evidence preservation", "Evidence Preservation", "Perishable evidence", "Perishable Evidence"],
        "Interview / notification strategy": [
            "Interview / notification strategy", "Interview / Notification Strategy",
            "Interview strategy", "Interview Strategy"
        ],
        "Policy/procedure requirements": [
            "Policy/procedure requirements", "Policy & procedure requirements",
            "Policy requirements", "Policy/procedures"
        ],
        "Criminal law hooks": [
            "Criminal law hooks", "Criminal law hooks (statutes)", "Criminal law hooks (statutes/case law)",
            "Criminal Law Hooks", "Criminal Law Hooks (statutes)"
        ],
        "Documentation watch-outs": [
            "Documentation watch-outs", "Documentation watch outs",
            "Documentation Watch-outs", "Documentation Watch Outs",
        ],
        "Chain of command / coordination": [
            "Chain of command / coordination", "Chain of Command / Coordination",
            "Command notification", "Command Notification", "Coordination", "Coordination / escalation"
        ],
        "Checklist": ["Checklist", "Checklists", "Checklist (if available)", "Checklists (if available)"],
    }

    all_headers = []
    for vs in variants.values():
        for v in vs:
            all_headers.append(re.escape(v))

    pattern = r"(?m)^(?P<h>(" + "|".join(all_headers) + r"))\s*:\s*$"
    matches = list(re.finditer(pattern, text))
    if not matches:
        return text, {}

    def to_canonical(h: str) -> str:
        for canon, vs in variants.items():
            if h in vs:
                return canon
        return h

    sections: Dict[str, str] = {}
    preface = text[:matches[0].start()].strip() or None

    for i, m in enumerate(matches):
        raw_h = m.group("h")
        h = to_canonical(raw_h)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        sections[h] = body

    return preface, sections

def split_bullets(text: str) -> List[str]:
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.split("\n")]
    out = []
    buf = ""
    for ln in lines:
        is_bullet = bool(re.match(r"^\s*([-•*]|\d+\.)\s+", ln))
        if is_bullet:
            if buf.strip():
                out.append(buf.strip())
            buf = re.sub(r"^\s*([-•*]|\d+\.)\s+", "", ln).strip()
        else:
            if ln.strip():
                buf = (buf + "\n" + ln.strip()).strip()
    if buf.strip():
        out.append(buf.strip())
    return out

def render_section(title: str, body: str, show_first_n: int = 5):
    bullets = split_bullets(body)
    if not bullets:
        st.markdown(body)
        return

    first = bullets[:show_first_n]
    rest = bullets[show_first_n:]

    for b in first:
        st.markdown(f"- {b}")

    if rest:
        with st.expander(f"More in {title}"):
            for b in rest:
                st.markdown(f"- {b}")

# =============================
# LLM ANSWER
# =============================
def ask_llm(question: str, mode: str, user_role: str):
    intent = detect_intent(question)

    expanded, matched_keys, chunks = retrieve(question, mode=mode)

    context = "\n\n".join(
        f"Source: {c.get('source','unknown')} (p.{c.get('page',1)})\n{c.get('text','')}"
        for c in chunks
    ) if chunks else ""

    if intent == "authority_reminder":
        system_msg = f"{DOCTRINE_FLAG}\n\n{ROLE_LOCK_AUTHORITY}\n\n{OUTPUT_CONTRACT_AUTHORITY}"
    elif intent == "policy_lookup":
        system_msg = "You are the Officer Support Assistant. Use ONLY the provided context.\n\n" + OUTPUT_CONTRACT_POLICY
    elif intent == "investigative_guidance":
        system_msg = f"{ROLE_LOCK_INVESTIGATIVE}\n\n{OUTPUT_CONTRACT_INVESTIGATIVE}"
    else:
        system_msg = GENERAL_SYSTEM

    user_msg = f"""ROLE:
{user_role}

CONTEXT:
{context}

QUESTION:
{question}

RULES:
- Use ONLY the context above.
- If missing/unclear, say: "I don’t see that in the documents."
- Provide citations in parentheses.
"""

    logger.info("==== QUERY START ====")
    logger.info("QUESTION: %s", question)
    logger.info("USER_ROLE: %s", user_role)
    logger.info("AUTO_INTENT: %s", intent)
    logger.info("EXPANDED_QUERY: %s", expanded)
    logger.info("MATCHED_KEYS: %s", ", ".join(matched_keys) if matched_keys else "none")
    logger.info("HITS_USED: %s", len(chunks))
    logger.info("SYSTEM_PROMPT_BEGIN\n%s\nSYSTEM_PROMPT_END", system_msg)
    logger.info("USER_PROMPT_BEGIN\n%s\nUSER_PROMPT_END", user_msg)

    if not chunks:
        answer = "I don’t see that in the documents."
        logger.info("FINAL_ANSWER: %s", answer)
        logger.info("==== QUERY END ====\n")
        return answer, [], intent

    logger.info("TOP_HITS:")
    for c in chunks[:TOP_K_FINAL]:
        logger.info(
            "  HIT type=%s source=%s page=%s faiss=%.4f bucket=%s bm25=%.2f preview=%s",
            c.get("_type", "unknown"),
            c.get("source", "unknown"),
            c.get("page", 1),
            c.get("_score", -1.0),
            faiss_score_category(c.get("_score", -1.0)) if c.get("_score", -1.0) >= 0 else "BM25-hit",
            c.get("_bm25", 0.0),
            (c.get("text", "")[:160] + "...").replace("\n", " ")
        )

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=950,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
    )

    answer = resp.choices[0].message.content.strip()
    logger.info("FINAL_ANSWER_BEGIN\n%s\nFINAL_ANSWER_END", answer)
    logger.info("==== QUERY END ====\n")
    return answer, chunks, intent

# =============================
# UI THEME
# =============================
st.markdown(
    """
<style>
html, body, [data-testid="stAppViewContainer"], .stApp {
    height: 100vh;
}

body {
    overflow: hidden;
}

.block-container {
    padding-top: 0.6rem !important;
    padding-bottom: 0.5rem !important;
    max-width: 1400px;
}

header[data-testid="stHeader"] { height: 0px; }
div[data-testid="stToolbar"] { visibility: hidden; height: 0px; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.stApp {
    background: radial-gradient(circle at top, #0a1733 0%, #040814 60%);
    color: white;
}

.header {
    background: linear-gradient(135deg, #081b3a, #041024);
    padding: 0.95rem 1.2rem;
    border-radius: 16px;
    margin-bottom: 0.7rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    border: 1px solid rgba(120,160,255,0.18);
    box-shadow: 0 14px 34px rgba(0,0,0,0.45);
}
.header-title { font-size: 1.45rem; font-weight: 900; letter-spacing: 0.02em; }
.header-sub { font-size: 0.85rem; color: #b8c4ff; }

.badge {
    background: rgba(16,42,86,0.7);
    padding: 0.35rem 0.8rem;
    border-radius: 999px;
    font-size: 0.8rem;
    border: 1px solid rgba(56,189,248,0.25);
}

.panel-shell {
    background: rgba(8,18,46,0.92);
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(120,160,255,0.18);
    box-shadow: 0 16px 40px rgba(0,0,0,0.50);
}

.bubble-user {
    background: linear-gradient(135deg, #1d4ed8, #38bdf8);
    color: white;
    padding: 0.70rem 0.85rem;
    border-radius: 16px;
    margin: 0.35rem 0 0.6rem 0;
    font-size: 0.95rem;
}

.bubble-bot {
    background: rgba(11, 21, 52, 0.95);
    color: #e5e7eb;
    padding: 0.70rem 0.85rem;
    border-radius: 16px;
    margin: 0.35rem 0 0.6rem 0;
    border: 1px solid rgba(110, 140, 245, 0.35);
    font-size: 0.95rem;
}

.small-muted { color: #b8c4ff; font-size: 0.82rem; }

.stTextInput>div>div>input {
    background-color: rgba(10, 16, 40, 0.9);
    border-radius: 999px;
    border: 1px solid rgba(99, 102, 241, 0.55);
    padding: 0.80rem 1rem;
    color: #f8fafc;
}
.stButton>button {
    background: linear-gradient(135deg, #1d4ed8, #38bdf8);
    color: white;
    font-weight: 800;
    border-radius: 999px;
    padding: 0.55rem 1.4rem;
    border: none;
    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.45);
}
</style>
""",
    unsafe_allow_html=True,
)

# =============================
# HEADER
# =============================
resource_badge = "Hybrid: FAISS+BM25 • Load on first query"

st.markdown(
    f"""
<div class="header">
  <div>
    <div class="header-title">🚓 Officer Support Assistant</div>
    <div class="header-sub">Sarpy County Sheriff’s Office • Policies • Statutes • Case Law</div>
  </div>
  <div class="badge">{resource_badge}</div>
</div>
""",
    unsafe_allow_html=True,
)

# =============================
# LAYOUT
# =============================
col_left, col_main, col_right = st.columns([1.2, 2.6, 1.6], gap="large")

with col_left:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    left_panel = st.container(height=LEFT_PANEL_HEIGHT, border=False)
    with left_panel:
        st.markdown("### OfficerAssistAI")
        st.caption("Operational guidance grounded in your documents")

        role = st.selectbox("User Role", ["Deputy", "Sergeant", "Supervisor"], index=0)
        mode = st.selectbox("Retrieval Filter", ["All Sources", "Policies Only", "Statutes Only"], index=0)

        st.markdown("---")
        if st.button("🧹 Clear chat"):
            st.session_state.chat = []
            st.session_state.sources = []
            st.session_state.last_intent = "general"
            st.rerun()

        st.markdown("---")
        st.markdown("**Quick examples**")
        st.caption("Click → copy/paste into the box")
        st.code("Juvenile sextortion on Instagram — suspect threatens to out victim unless they meet up.")
        st.code("What policy governs seizing a juvenile’s phone at school?")
        st.code("What authority applies for an exigent phone seizure?")

        st.markdown("---")
        st.markdown("**Source score legend (FAISS only)**")
        st.markdown(
            "- **Excellent**: ≥ 0.58\n"
            "- **Relevant**: 0.48 – 0.57\n"
            "- **Weak**: 0.40 – 0.47\n"
            "- **Likely Unrelated**: < 0.40\n"
            "- **BM25-hit**: keyword match (no FAISS score)"
        )

        st.markdown("---")
        st.markdown("**App Logs**")
        st.caption(str(APP_LOG))
        if APP_LOG.exists():
            with open(APP_LOG, "rb") as f:
                st.download_button(
                    "⬇️ Download App Log",
                    data=f,
                    file_name="app.log",
                    mime="text/plain",
                )
        else:
            st.caption("App log will appear after the first query.")

        st.markdown("---")
        st.markdown("**Evaluation Log**")
        st.caption(str(QUERY_LOG_CSV))
        if QUERY_LOG_CSV.exists():
            with open(QUERY_LOG_CSV, "rb") as f:
                st.download_button(
                    "⬇️ Download Query Log",
                    data=f,
                    file_name="query_eval_log.csv",
                    mime="text/csv",
                )
        else:
            st.caption("Query log will appear after the first query.")
    st.markdown("</div>", unsafe_allow_html=True)

with col_main:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    center_panel = st.container(height=CENTER_PANEL_HEIGHT, border=False)
    with center_panel:
        st.markdown("## Live Query")
        st.caption("Auto-routing: Authority • Policy Lookup • Investigative Workflow • General")

        chat_history = st.container(height=CHAT_HISTORY_HEIGHT, border=False)
        with chat_history:
            for msg in st.session_state.chat:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="bubble-user">👮‍♂️ <b>{role}:</b> {msg["content"]}</div>',
                        unsafe_allow_html=True
                    )
                else:
                    content = msg.get("content", "")
                    intent = msg.get("intent", "general")

                    st.markdown(
                        f'<div class="bubble-bot">🛡️ <b>Assistant</b> <span class="small-muted">({intent})</span></div>',
                        unsafe_allow_html=True
                    )

                    if intent == "investigative_guidance":
                        preface, sections = parse_sections(content)

                        if preface:
                            st.markdown(preface)

                        if sections:
                            with st.expander("Tier 1 — Immediate (first view)", expanded=True):
                                body = sections.get("Immediate actions", "")
                                render_section("Immediate actions", body, show_first_n=5)

                            for canon_key, title, n_first in SECTION_ORDER[1:]:
                                body = sections.get(canon_key)
                                if body:
                                    with st.expander(title, expanded=False):
                                        render_section(title, body, show_first_n=n_first)
                        else:
                            st.markdown(content)

                        show_raw = st.toggle(
                            "Show raw answer (debug)",
                            value=False,
                            key=f"raw_{msg.get('id','')}_{len(content)}"
                        )
                        if show_raw:
                            st.code(content, language="markdown")
                    else:
                        st.markdown(content)

        with st.form("ask_form", clear_on_submit=True):
            question = st.text_input(
                "Ask a question:",
                label_visibility="collapsed",
                placeholder="Example: Juvenile threatened to be outed unless they meet up — how should we investigate?",
            )
            submitted = st.form_submit_button("Send")

        if submitted and question.strip():
            if embedder is None or index is None or metadata is None or bm25 is None:
                st.info("Loading retrieval resources for the first query. This may take a bit on Render.")

            with st.spinner("Searching and answering..."):
                t0 = time.perf_counter()
                answer, srcs, intent = ask_llm(question.strip(), mode=mode, user_role=role)
                elapsed = time.perf_counter() - t0

                save_eval_row(
                    question=question.strip(),
                    answer=answer,
                    intent=intent,
                    mode=mode,
                    user_role=role,
                    response_time_sec=elapsed,
                    chunks=srcs,
                )

                st.session_state.chat.append({"role": "user", "content": question.strip()})
                st.session_state.chat.append({
                    "role": "assistant",
                    "content": answer,
                    "intent": intent,
                    "id": f"{len(st.session_state.chat)}"
                })
                st.session_state.sources = srcs
                st.session_state.last_intent = intent
                st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel-shell">', unsafe_allow_html=True)
    right_panel = st.container(height=RIGHT_PANEL_HEIGHT, border=False)
    with right_panel:
        st.markdown("### Sources Used")

        show_numeric_scores = st.toggle("Show raw FAISS scores", value=False)
        show_bm25 = st.toggle("Show BM25 keyword score", value=False)

        if st.session_state.get("sources"):
            for c in st.session_state.sources:
                src = c.get("source", "unknown")
                page = c.get("page", 1)
                faiss_score = c.get("_score", -1.0)
                bm25_score = c.get("_bm25", 0.0)
                stype = c.get("_type", "unknown")

                if faiss_score >= 0:
                    bucket = score_badge_text(faiss_score)
                    header = f"{src} (p.{page}) • {stype} • {bucket}"
                    if show_numeric_scores:
                        header += f" • faiss={faiss_score:.3f}"
                else:
                    header = f"{src} (p.{page}) • {stype} • BM25-hit"

                if show_bm25:
                    header += f" • bm25={bm25_score:.2f}"

                with st.expander(header):
                    st.write((c.get("text", "") or "")[:1200])
        else:
            st.caption("Ask a question to see sources.")
    st.markdown("</div>", unsafe_allow_html=True)