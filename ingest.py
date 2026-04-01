# ingest.py
# - Extract local PDFs/images from ./data
# - Scrape Nebraska statutes for selected chapters (default: 28 + 60)
# Output: data/chunks.jsonl

import re
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import requests
import pdfplumber
from bs4 import BeautifulSoup

# Optional OCR imports
try:
    from pdf2image import convert_from_path
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# --------------------------
# CONFIG
# --------------------------
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://nebraskalegislature.gov"
USER_AGENT = "Mozilla/5.0 (OfficerSupportAssistant; local) PythonRequests/2.x"

# Choose which chapters to ingest
STATUTE_CHAPTERS = ["28", "60"]   # add "29" later if you want criminal procedure

# Chunk config (better than naive char slicing)
MAX_CHARS = 1800
MIN_CHARS = 450

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "ingest.log"

# --------------------------
# LOGGING
# --------------------------
logger = logging.getLogger("ingest")
logger.setLevel(logging.INFO)

_fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
fh.setFormatter(_fmt)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setFormatter(_fmt)
logger.addHandler(ch)

# --------------------------
# HELPERS
# --------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_get(url: str, timeout: int = 30) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    last_err = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            last_err = e
            logger.warning(f"GET failed ({attempt+1}/3): {url} | {e}")
            time.sleep(1.0 + attempt)
    raise RuntimeError(f"Failed to GET: {url} | last error: {last_err}")

def extract_main_text_from_statute_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()
    main = (
        soup.find(id="content")
        or soup.find("main")
        or soup.find("article")
        or soup.find("div", {"class": "content"})
        or soup.find("div", {"id": "main"})
        or soup
    )
    text = main.get_text(" ", strip=True)
    return clean_text(text)

def is_probably_not_content(text: str) -> bool:
    if not text:
        return True
    if len(text) < 350:
        return True
    nav_phrases = ["Nebraska Legislature", "Legislative Bills", "Find Your Senator"]
    hits = sum(1 for p in nav_phrases if p.lower() in text.lower())
    return hits >= 2

def statute_browse_url(chapter: str) -> str:
    return f"{BASE_URL}/laws/browse-chapters.php?chapter={chapter}"

def parse_statute_number(text: str) -> str:
    # often like "28-1202.01" etc
    return text.strip()

# --------------------------
# LOCAL PDF EXTRACTION
# --------------------------
def extract_text_from_pdf(pdf_path: Path) -> List[Dict[str, Any]]:
    logger.info(f"Extracting PDF: {pdf_path.name}")
    docs: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = clean_text(page.extract_text() or "")
                if text:
                    docs.append({
                        "doc_type": "policy",
                        "source": pdf_path.name,
                        "page": page_num,
                        "title": None,
                        "chapter": None,
                        "statute_id": None,
                        "text": text
                    })
                    continue

                if OCR_AVAILABLE:
                    logger.info(f" {pdf_path.name} p.{page_num}: OCR fallback...")
                    try:
                        images = convert_from_path(str(pdf_path), first_page=page_num, last_page=page_num)
                        if images:
                            ocr_text = clean_text(pytesseract.image_to_string(images[0]) or "")
                            if ocr_text:
                                docs.append({
                                    "doc_type": "policy",
                                    "source": pdf_path.name,
                                    "page": page_num,
                                    "title": None,
                                    "chapter": None,
                                    "statute_id": None,
                                    "text": ocr_text
                                })
                    except Exception as e:
                        logger.warning(f"OCR failed: {pdf_path.name} p.{page_num} | {e}")
                else:
                    logger.warning(f"{pdf_path.name} p.{page_num}: no text and OCR unavailable.")
    except Exception as e:
        logger.error(f"Failed to open PDF {pdf_path.name}: {e}")
    return docs

def extract_text_from_image(img_path: Path) -> Optional[Dict[str, Any]]:
    logger.info(f"Extracting IMAGE: {img_path.name}")
    if not OCR_AVAILABLE:
        logger.warning("OCR not available; skipping image.")
        return None
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(img_path)
        text = clean_text(pytesseract.image_to_string(img) or "")
        if not text:
            return None
        return {
            "doc_type": "policy",
            "source": img_path.name,
            "page": 1,
            "title": None,
            "chapter": None,
            "statute_id": None,
            "text": text
        }
    except Exception as e:
        logger.warning(f"Failed OCR on image {img_path.name}: {e}")
        return None

# --------------------------
# STATUTE SCRAPER (multi chapter)
# --------------------------
def scrape_chapter(chapter: str, max_statutes: Optional[int] = None) -> List[Dict[str, Any]]:
    logger.info(f"Scraping Nebraska statutes: Chapter {chapter}")
    url = statute_browse_url(chapter)
    resp = safe_get(url, timeout=30)
    soup = BeautifulSoup(resp.text, "html.parser")

    docs: List[Dict[str, Any]] = []

    links = soup.select("a[href*='statutes.php?statute=']")
    statutes: List[Tuple[str, Optional[str], str]] = []
    seen = set()

    for a in links:
        num = parse_statute_number((a.get_text() or "").strip())
        href = (a.get("href") or "").strip()
        if not num or num in seen:
            continue
        seen.add(num)

        title = None
        td = a.find_parent("td")
        if td is not None:
            sib = td.find_next_sibling("td")
            if sib is not None:
                title = sib.get_text(strip=True) or None

        if href.startswith("/"):
            full_url = BASE_URL + href
        elif href.startswith("http"):
            full_url = href
        else:
            full_url = BASE_URL + "/laws/" + href

        statutes.append((num, title, full_url))

    if max_statutes:
        statutes = statutes[:max_statutes]

    logger.info(f"Found {len(statutes)} statutes in Chapter {chapter}")

    for i, (num, title, stat_url) in enumerate(statutes, start=1):
        logger.info(f"[{i}/{len(statutes)}] {num}")
        try:
            page = safe_get(stat_url, timeout=30)
            full_text = extract_main_text_from_statute_html(page.text)

            if is_probably_not_content(full_text):
                print_url = stat_url + ("&" if "?" in stat_url else "?") + "print=true"
                page2 = safe_get(print_url, timeout=30)
                full_text = extract_main_text_from_statute_html(page2.text)

            if is_probably_not_content(full_text):
                logger.warning(f"No usable statute text: {num}")
                continue

            docs.append({
                "doc_type": "statute",
                "source": f"NE_Statute_{num}",
                "page": 1,
                "title": title,
                "chapter": chapter,
                "statute_id": num,
                "text": full_text
            })

            time.sleep(0.12)
        except Exception as e:
            logger.warning(f"Error scraping {stat_url}: {e}")

    return docs

# --------------------------
# BETTER CHUNKING (paragraph blocks)
# --------------------------
def chunk_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []

    for d in docs:
        text = d.get("text", "") or ""
        text = text.strip()
        if not text:
            continue

        # split into pseudo paragraphs
        parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
        buf = ""
        for p in parts:
            p = p.strip()
            if not p:
                continue

            # start new chunk if adding would exceed max
            if len(buf) + len(p) + 1 > MAX_CHARS and len(buf) >= MIN_CHARS:
                chunk = dict(d)
                chunk["text"] = buf.strip()
                chunks.append(chunk)
                buf = p
            else:
                buf = (buf + " " + p).strip()

        if buf and len(buf) >= 120:
            chunk = dict(d)
            chunk["text"] = buf.strip()
            chunks.append(chunk)

    return chunks

# --------------------------
# MAIN
# --------------------------
def main():
    logger.info("=== START INGEST ===")

    pdfs = sorted(DATA_DIR.glob("*.pdf"))
    images = sorted(list(DATA_DIR.glob("*.png")) + list(DATA_DIR.glob("*.jpg")) + list(DATA_DIR.glob("*.jpeg")))

    docs: List[Dict[str, Any]] = []
    for p in pdfs:
        docs.extend(extract_text_from_pdf(p))
    for img in images:
        d = extract_text_from_image(img)
        if d:
            docs.append(d)

    # statutes
    for ch in STATUTE_CHAPTERS:
        docs.extend(scrape_chapter(ch, max_statutes=None))

    logger.info(f"Docs before chunking: {len(docs)}")
    chunks = chunk_docs(docs)
    logger.info(f"Chunks total: {len(chunks)}")

    out_path = DATA_DIR / "chunks.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info(f"Saved chunks -> {out_path}")
    logger.info(f"Log saved -> {LOG_FILE}")

if __name__ == "__main__":
    main()