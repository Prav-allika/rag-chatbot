"""
app/document_loader.py — document ingestion and PDF cleaning pipeline.

Responsibilities:
  - Multi-format loading  : PDF, DOCX, HTML, TXT, Markdown
  - Adaptive PDF pipeline : per-page classification → best extractor chosen:
        scanned page (low text + images)  →  OCR  (pytesseract)
        table-bearing page                →  pdfplumber  [TABLE] blocks
        text page                         →  pdfplumber  plain text
  - Header/footer strip   : frequency-based, across pages
  - Encoding normalisation: NFKC + ligature/smart-quote fixes
  - Chunking              : switchable strategies (see app/chunking.py)
  - Chunk deduplication   : MD5 hash dedup after splitting
  - Vector store          : build (embed + save) and load (FAISS)

No dependency on the QA chain or LLM.
"""

import os
import re
import json
import hashlib
import logging
import unicodedata
from pathlib import Path
from collections import Counter

import numpy as np

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader

from app.config import Config
from app import chunking

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".html", ".htm", ".txt", ".md"}

# pdfplumber's default x_tolerance (3) infers word boundaries from character
# gaps and fuses adjacent words together on PDFs with tight kerning (observed
# on this LaTeX-generated arXiv PDF: "Table1: Maximumpathlengths..." instead of
# "Table 1: Maximum path lengths..."). A lower tolerance fixes it without
# affecting normally-spaced PDFs. Verified against this document's actual pages.
_PDF_TEXT_X_TOLERANCE = 1.5


# =============================================================================
# ENCODING NORMALISATION
# =============================================================================
def _normalize_text(text: str) -> str:
    """Fix encoding artifacts common in PDF-extracted text."""
    if not text:
        return text
    text = unicodedata.normalize("NFKC", text)  # fi→fi, ﬂ→fl, fullwidth→ASCII
    text = text.replace("\x00", "").replace("�", "")
    text = (
        text.replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "--")
            .replace("…", "...")
    )
    return text.strip()


# =============================================================================
# HEADER / FOOTER STRIPPING
# =============================================================================
def _strip_headers_footers(page_texts: list) -> list:
    """
    Remove lines appearing on 30%+ of pages — running headers, footers,
    page numbers.  Examines only the first and last 3 lines per page
    where repetition is most common.
    """
    if len(page_texts) < 3:
        return page_texts

    line_counts: Counter = Counter()
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        for line in lines[:3] + lines[-3:]:
            line_counts[line] += 1

    threshold = max(3, len(page_texts) * 0.30)
    noise = {ln for ln, cnt in line_counts.items() if cnt >= threshold}
    page_num_re = re.compile(
        r"^[-\s]*\d{1,4}[-\s]*$|^page\s+\d+(\s+of\s+\d+)?$", re.IGNORECASE
    )

    cleaned, removed = [], 0
    for text in page_texts:
        filtered = []
        for line in text.split("\n"):
            s = line.strip()
            if s in noise or page_num_re.match(s):
                removed += 1
            else:
                filtered.append(line)
        cleaned.append("\n".join(filtered))

    if removed:
        logger.info(
            f"Header/footer strip: removed {removed} noisy lines "
            f"across {len(page_texts)} pages"
        )
    return cleaned


# =============================================================================
# OCR  (pytesseract + pdf2image)
# =============================================================================
def _ocr_pdf_page(file_path: str, page_num: int) -> str:
    """
    OCR a single PDF page.
    Requires: pip install pytesseract pdf2image Pillow
              + system: brew install tesseract  (macOS)
    Returns extracted text, or empty string if OCR is unavailable.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
    except ImportError:
        return ""

    try:
        images = convert_from_path(
            file_path, first_page=page_num + 1, last_page=page_num + 1, dpi=200
        )
        if images:
            return pytesseract.image_to_string(images[0]).strip()
    except Exception as e:
        logger.debug(f"OCR failed for page {page_num + 1}: {e}")
    return ""


# =============================================================================
# ADAPTIVE PDF EXTRACTION — helpers
# =============================================================================
def _classify_page(page) -> str:
    """
    Inspect a pdfplumber page and return its dominant type:
      'scanned' — very little selectable text + embedded images  → needs OCR
      'table'   — has structured tables (may also have prose)    → pdfplumber table mode
      'text'    — normal searchable text, no tables              → pdfplumber plain text
    """
    text = page.extract_text(x_tolerance=_PDF_TEXT_X_TOLERANCE) or ""
    word_count = len(text.split())
    has_tables = bool(page.find_tables())
    has_images = bool(getattr(page, "images", []))

    if word_count < 15 and has_images:
        return "scanned"
    if has_tables:
        return "table"
    return "text"


_TABLE_CAPTION_RE = re.compile(r"^Table\s+\d+\s*:", re.IGNORECASE)
_SECTION_HEADER_RE = re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\s+[A-Z]")


def _wrap_informal_tables(text: str) -> str:
    """
    Some tables in this kind of PDF (e.g. Table 1 in the Attention paper) have
    no visible ruling lines, so pdfplumber's find_tables() misses them and
    they get extracted as plain prose -- nothing then stops the chunker from
    splitting a row (or its caption) away from the rest of the table (observed
    directly: a 4-row table truncated to 3 rows mid-chunk). Detects "Table N:"
    caption lines and wraps that caption through to the next numbered-section
    header (or the next table caption) in a "[TABLE]" block, so
    app/chunking.py's atomic-table handling — which already looks for that
    marker, for pdfplumber-detected tables — keeps it together too.
    """
    lines = text.split("\n")
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if _TABLE_CAPTION_RE.match(line.strip()):
            block = [line]
            i += 1
            while i < len(lines):
                stripped = lines[i].strip()
                if _SECTION_HEADER_RE.match(stripped) or _TABLE_CAPTION_RE.match(stripped):
                    break
                block.append(lines[i])
                i += 1
            while block and not block[-1].strip():
                block.pop()
            out_lines.append("")
            out_lines.append("[TABLE]")
            out_lines.extend(block)
            out_lines.append("")
        else:
            out_lines.append(line)
            i += 1
    return "\n".join(out_lines)


def _pdfplumber_table_text(page) -> str:
    """
    Extract text + structured tables from a single pdfplumber page.
    Plain text is filtered to exclude table bounding boxes so content
    is not duplicated. Tables are rendered as pipe-separated rows under
    a [TABLE] marker so the LLM can read them.
    """
    parts = []
    table_bboxes = [t.bbox for t in page.find_tables()]

    if table_bboxes:
        def _not_in_table(obj):
            for x0, top, x1, bottom in table_bboxes:
                if (obj.get("x0", 0) >= x0 - 2 and
                        obj.get("top", 0) >= top - 2 and
                        obj.get("x1", obj.get("x0", 0)) <= x1 + 2 and
                        obj.get("bottom", obj.get("top", 0)) <= bottom + 2):
                    return False
            return True
        plain = page.filter(_not_in_table).extract_text(x_tolerance=_PDF_TEXT_X_TOLERANCE) or ""
    else:
        plain = page.extract_text(x_tolerance=_PDF_TEXT_X_TOLERANCE) or ""

    if plain.strip():
        parts.append(plain.strip())

    for table in page.extract_tables(table_settings={"text_x_tolerance": _PDF_TEXT_X_TOLERANCE}):
        if not table:
            continue
        rows = [
            " | ".join(str(c).strip() if c else "" for c in row)
            for row in table if row
        ]
        if rows:
            parts.append("[TABLE]\n" + "\n".join(rows))

    return "\n\n".join(parts).strip()


# =============================================================================
# ADAPTIVE PDF EXTRACTION — main function
# =============================================================================
def _adaptive_pdf_extract(file_path: str):
    """
    Per-page adaptive PDF extraction using pdfplumber as the analysis engine.

    Algorithm (per page):
      1. Classify page → 'scanned' | 'table' | 'text'
      2. scanned  →  OCR via pytesseract (page is an image, no selectable text)
                     fallback: pdfplumber plain text if OCR unavailable
      3. table    →  pdfplumber table-aware (text + [TABLE] markers)
                     fallback: OCR if pdfplumber yields nothing
      4. text     →  pdfplumber plain text
                     upgrade: OCR if pdfplumber yields fewer words than OCR
                     (catches partially scanned pages)

    Metadata recorded per document:
      page            — 1-indexed page number
      extract_method  — 'text' | 'table' | 'scanned' | 'ocr_upgrade' | 'ocr_fallback'

    Returns list[Document] or None if pdfplumber is not installed.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning(
            "pdfplumber not installed — using PyPDF fallback. "
            "Run: pip install pdfplumber"
        )
        return None

    from langchain_core.documents import Document

    docs = []
    method_counts: dict = {}

    try:
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_type = _classify_page(page)
                method = page_type

                if page_type == "scanned":
                    # Page is image-only → OCR is the right tool
                    content = _ocr_pdf_page(file_path, page_num)
                    if not content.strip():
                        # OCR not installed / failed → last-resort pdfplumber text
                        content = page.extract_text(x_tolerance=_PDF_TEXT_X_TOLERANCE) or ""
                        method = "ocr_fallback"

                elif page_type == "table":
                    # Structured table(s) on this page → pdfplumber table mode
                    content = _pdfplumber_table_text(page)
                    if not content.strip():
                        # pdfplumber found no content → try OCR
                        content = _ocr_pdf_page(file_path, page_num)
                        method = "ocr_fallback"

                else:  # "text"
                    content = page.extract_text(x_tolerance=_PDF_TEXT_X_TOLERANCE) or ""
                    # Upgrade: if pdfplumber got suspiciously little text, try OCR
                    if len(content.split()) < 20:
                        ocr = _ocr_pdf_page(file_path, page_num)
                        if len(ocr.split()) > len(content.split()):
                            content = ocr
                            method = "ocr_upgrade"
                    if method == "text":
                        # find_tables() missed this page (no ruling lines) but it
                        # may still contain an unbordered "Table N:" style table.
                        content = _wrap_informal_tables(content)

                method_counts[method] = method_counts.get(method, 0) + 1

                if content.strip():
                    docs.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "page": page_num + 1,
                            "extract_method": method,
                        },
                    ))

    except Exception as e:
        logger.warning(f"Adaptive PDF extraction failed ({e}) — falling back to PyPDF")
        return None

    counts_str = "  ".join(f"{m}={n}" for m, n in sorted(method_counts.items()) if n)
    logger.info(f"Adaptive PDF: {len(docs)} pages extracted  [{counts_str}]")
    return docs if docs else None


# =============================================================================
# CHUNK DEDUPLICATION
# =============================================================================
def _deduplicate_chunks(chunks: list) -> list:
    """Remove exact-duplicate chunks using MD5 content hash."""
    seen: set = set()
    unique = []
    for chunk in chunks:
        h = hashlib.md5(chunk.page_content.strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
    removed = len(chunks) - len(unique)
    if removed:
        logger.info(f"Deduplication: removed {removed} duplicate chunk(s)")
    return unique


def _deduplicate_near_duplicates(chunks: list, embeddings, threshold: float = 0.95) -> list:
    """
    Remove near-duplicate chunks (cosine similarity > threshold) that survive
    exact-hash dedup — e.g. the same information paraphrased across two source
    documents. Greedy: keeps a chunk unless it's too similar to one already kept.

    Runs after _deduplicate_chunks (exact-hash pass is a cheap pre-filter) and
    before contextual retrieval (must dedup on original chunk text, not the
    LLM-prepended situating context).
    """
    if len(chunks) < 2:
        return chunks

    vectors = np.array(embeddings.embed_documents([c.page_content for c in chunks]), dtype=float)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    unit = vectors / norms

    kept_idx: list = []
    for i in range(len(chunks)):
        if kept_idx:
            # np.errstate: BLAS matmul can transiently hit overflow/invalid on some
            # platforms (observed with macOS Accelerate) with no actual bad input data
            # (verified: raw embeddings here are always finite, normed ~1.0). Silence
            # the warning and sanitize the output rather than the (already-clean) input.
            with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
                sims = unit[kept_idx] @ unit[i]
            sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
            if float(sims.max()) > threshold:
                continue
        kept_idx.append(i)

    removed = len(chunks) - len(kept_idx)
    if removed:
        logger.info(f"Near-duplicate deduplication: removed {removed} chunk(s) (cosine similarity > {threshold})")
    return [chunks[i] for i in kept_idx]


# =============================================================================
# CONTEXTUAL RETRIEVAL  (Anthropic technique)
# =============================================================================
_CONTEXT_PROMPT = """<document>
{doc}
</document>
Here is a chunk from the document above:
<chunk>
{chunk}
</chunk>
Write a short, 1-2 sentence context that situates this chunk within the overall document, to improve search retrieval of the chunk. Answer only with the succinct context, nothing else."""


def _contextualize_chunks(chunks: list, full_text: str) -> list:
    """
    Prepend an LLM-generated situating sentence to each chunk before it is
    embedded/BM25-indexed, so retrieval isn't blind to where a chunk sits in
    the document. Reduces retrieval misses on chunks that read as ambiguous
    in isolation (e.g. "the model achieved 92% accuracy" with no subject).

    Original chunk text is preserved in metadata["original_content"] for
    citation display. Runs one LLM call per chunk, parallelized.

    No-ops (returns chunks unchanged) if disabled, or no fast LLM is
    configured (skipped for the local FLAN-T5 fallback — too slow/weak
    for this instruction-following task).
    """
    if not chunks or not Config.CONTEXTUAL_RETRIEVAL:
        return chunks
    if not (Config.use_openai() or Config.use_groq()):
        logger.info("Contextual retrieval skipped — no Groq/OpenAI LLM configured")
        return chunks

    from app.rag_pipeline import get_llm
    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from concurrent.futures import ThreadPoolExecutor, as_completed

    doc_excerpt = full_text[: Config.CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS]
    prompt = PromptTemplate(template=_CONTEXT_PROMPT, input_variables=["doc", "chunk"])
    chain = prompt | get_llm() | StrOutputParser()

    def _add_context(chunk):
        try:
            context = chain.invoke({"doc": doc_excerpt, "chunk": chunk.page_content}).strip()
            chunk.metadata["original_content"] = chunk.page_content
            chunk.page_content = f"{context}\n\n{chunk.page_content}"
        except Exception as e:
            logger.warning(f"Contextual retrieval failed for a chunk: {e}")
        return chunk

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(_add_context, c) for c in chunks]
        for f in as_completed(futures):
            f.result()

    logger.info(f"Contextual retrieval: added context to {len(chunks)} chunk(s)")
    return chunks


# =============================================================================
# MULTI-FORMAT DOCUMENT LOADER
# =============================================================================
def load_document(file_path: str) -> list:
    """
    Load a document by file extension and run the full cleaning pipeline.

    PDF pipeline:
      1. Adaptive per-page extraction (pdfplumber):
           scanned pages  →  OCR
           table pages    →  pdfplumber [TABLE] blocks
           text pages     →  pdfplumber plain text
      2. PyPDFLoader fallback if pdfplumber is not installed
         (with post-extraction OCR pass for near-empty pages)
      3. Header/footer stripping across all pages
      4. Encoding normalisation (all formats)
    """
    ext = Path(file_path).suffix.lower()
    logger.info(f"Loading document ({ext}): {file_path}")

    if ext == ".pdf":
        docs = _adaptive_pdf_extract(file_path)

        if docs is None:
            # pdfplumber not installed → PyPDF plain-text fallback
            docs = PyPDFLoader(file_path).load()
            # Still apply OCR for any near-empty pages
            ocr_applied = 0
            for i, doc in enumerate(docs):
                if len(doc.page_content.strip()) < 100:
                    ocr_text = _ocr_pdf_page(file_path, i)
                    if ocr_text:
                        doc.page_content = ocr_text
                        doc.metadata["extract_method"] = "ocr"
                        ocr_applied += 1
            if ocr_applied:
                logger.info(f"PyPDF fallback: OCR applied to {ocr_applied} page(s)")

        # Header/footer stripping (works across all extraction methods)
        page_texts = [doc.page_content for doc in docs]
        cleaned = _strip_headers_footers(page_texts)
        for doc, text in zip(docs, cleaned):
            doc.page_content = text

        # Drop completely empty pages
        docs = [d for d in docs if d.page_content.strip()]

    elif ext == ".docx":
        try:
            from langchain_community.document_loaders import Docx2txtLoader
            docs = Docx2txtLoader(file_path).load()
        except ImportError:
            raise RuntimeError("DOCX support requires: pip install docx2txt")

    elif ext in (".html", ".htm"):
        try:
            from langchain_community.document_loaders import BSHTMLLoader
            docs = BSHTMLLoader(file_path, open_encoding="utf-8").load()
        except ImportError:
            raise RuntimeError("HTML support requires: pip install beautifulsoup4 lxml")

    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        docs = TextLoader(file_path, encoding="utf-8").load()

    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )

    for doc in docs:
        doc.page_content = _normalize_text(doc.page_content)

    return docs


# =============================================================================
# INTERNET CHECK
# =============================================================================
def _has_internet(host: str = "8.8.8.8", port: int = 53, timeout: float = 2.0) -> bool:
    """TCP probe to Google DNS — returns True if we have a network connection."""
    import socket
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except Exception:
        return False


# =============================================================================
# QDRANT HELPERS
# =============================================================================
def _qdrant_collection(doc_id: str) -> str:
    """Sanitise a doc_id into a valid Qdrant collection name."""
    import re
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", doc_id)[:64]
    return ("col_" + name) if name and name[0].isdigit() else (name or "default")


def _build_qdrant(chunks: list, embeddings, doc_id: str):
    """
    Ingest chunks into Qdrant Cloud.
    Collection is created automatically if it does not exist.
    Returns (QdrantVectorStore, chunks).
    """
    from langchain_qdrant import QdrantVectorStore

    collection = _qdrant_collection(doc_id)
    logger.info(f"Qdrant: ingesting {len(chunks)} chunks → collection '{collection}'")

    vs = QdrantVectorStore.from_documents(
        chunks,
        embeddings,
        url=Config.QDRANT_URL,
        api_key=Config.QDRANT_API_KEY,
        collection_name=collection,
        force_recreate=True,   # overwrite existing collection for this doc
    )
    logger.info(f"Qdrant: ingestion complete — collection '{collection}'")
    return vs, chunks   # chunks are already in memory — pass them forward


def _load_qdrant(doc_id: str, embeddings):
    """
    Connect to an existing Qdrant collection and scroll all chunks for BM25.
    Returns (QdrantVectorStore, all_chunks).
    """
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from langchain_core.documents import Document

    collection = _qdrant_collection(doc_id)
    logger.info(f"Qdrant: connecting to collection '{collection}'")

    vs = QdrantVectorStore(
        client=QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY),
        collection_name=collection,
        embedding=embeddings,
    )

    # Scroll all stored points to rebuild BM25 index
    try:
        client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
        results, _ = client.scroll(
            collection_name=collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        all_chunks = [
            Document(
                page_content=r.payload.get("page_content", ""),
                metadata=r.payload.get("metadata", {}),
            )
            for r in results
            if r.payload.get("page_content")
        ]
        logger.info(f"Qdrant: scrolled {len(all_chunks)} chunks for BM25")
    except Exception as e:
        logger.warning(f"Qdrant scroll failed ({e}) — BM25 will be disabled")
        all_chunks = []

    return vs, all_chunks


# =============================================================================
# EMBED CACHE  (skip re-embedding a document that hasn't changed)
# =============================================================================
_MANIFEST_PATH = "artifacts/vector_store/manifest.json"


def _file_hash(file_path: str) -> str:
    """MD5 of the file's bytes — detects whether the document content changed."""
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


_PIPELINE_SOURCE_FILES = ("document_loader.py", "chunking.py")


def _pipeline_code_hash() -> str:
    """
    MD5 over the extraction/chunking source files themselves (not just config
    knobs) — included in the embed fingerprint so a CODE change (a bug fix to
    how text is extracted or split) forces a rebuild too, not only a config
    change. Config knobs alone missed exactly this: the pdfplumber x_tolerance
    fix and the atomic-table-chunking fix were both pure code changes with no
    corresponding config field, so the old fingerprint never changed and a
    stale, pre-fix store (word-fused text, truncated tables) kept getting
    silently reused across rebuilds.
    """
    h = hashlib.md5()
    this_dir = Path(__file__).resolve().parent
    for fname in _PIPELINE_SOURCE_FILES:
        path = this_dir / fname
        if path.exists():
            h.update(path.read_bytes())
    return h.hexdigest()


def _embed_fingerprint() -> dict:
    """Config knobs + pipeline code version that change what gets embedded —
    cache must invalidate if any change."""
    return {
        "embedding_model": Config.OPENAI_EMBEDDING_MODEL if Config.use_openai() else Config.EMBEDDING_MODEL,
        "chunk_size": Config.CHUNK_SIZE,
        "chunk_overlap": Config.CHUNK_OVERLAP,
        "chunking_strategy": Config.CHUNKING_STRATEGY,
        "near_dup_dedup_enabled": Config.NEAR_DUP_DEDUP_ENABLED,
        "near_dup_threshold": Config.NEAR_DUP_THRESHOLD,
        "contextual_retrieval": Config.CONTEXTUAL_RETRIEVAL,
        "pipeline_code_hash": _pipeline_code_hash(),
    }


def _load_manifest() -> dict:
    if os.path.exists(_MANIFEST_PATH):
        try:
            with open(_MANIFEST_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_manifest(manifest: dict) -> None:
    os.makedirs(os.path.dirname(_MANIFEST_PATH), exist_ok=True)
    with open(_MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def _existing_store_available(doc_id: str, store_path: str) -> bool:
    """True if a vector store for doc_id already exists on the currently active backend."""
    if bool(Config.QDRANT_URL) and _has_internet():
        try:
            from qdrant_client import QdrantClient
            client = QdrantClient(url=Config.QDRANT_URL, api_key=Config.QDRANT_API_KEY)
            return client.collection_exists(_qdrant_collection(doc_id))
        except Exception:
            return False
    return os.path.exists(store_path)


# =============================================================================
# VECTOR STORE — BUILD + LOAD
# =============================================================================
def build_vector_store(file_path: str, store_path: str, doc_id: str = "default"):
    """
    Load document → clean → chunk → deduplicate → embed → store.

    Storage backend selected automatically:
      Qdrant Cloud  — when QDRANT_URL is set in .env AND internet is reachable
      FAISS (local) — fallback (offline, no config needed)

    Skips re-embedding entirely if this exact doc_id was already embedded with
    the same file content AND the same embedding-relevant config (see
    _embed_fingerprint) — re-uploading an unchanged file, or restarting the
    app, doesn't burn embedding/contextual-retrieval tokens again.

    Returns (vector_store, all_chunks).
    all_chunks is passed to make_qa_chain() so BM25 works regardless of backend.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        from app.rag_pipeline import get_embeddings   # lazy import — avoids circular

        fingerprint = {"file_hash": _file_hash(file_path), **_embed_fingerprint()}
        manifest = _load_manifest()
        if manifest.get(doc_id) == fingerprint and _existing_store_available(doc_id, store_path):
            logger.info(f"'{doc_id}' unchanged (same content + embedding config) — reusing existing vector store")
            return load_vector_store(store_path, doc_id=doc_id)

        docs = load_document(file_path)
        if not docs:
            raise ValueError("No content loaded from document")

        logger.info(f"Loaded {len(docs)} page(s)/section(s)")
        embeddings = get_embeddings()

        chunks = chunking.split_documents(
            docs,
            Config.CHUNKING_STRATEGY,
            Config.CHUNK_SIZE,
            Config.CHUNK_OVERLAP,
            embeddings=embeddings,
            semantic_breakpoint_percentile=Config.SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE,
            semantic_min_chars=Config.SEMANTIC_CHUNK_MIN_CHARS,
        )
        chunks = _deduplicate_chunks(chunks)
        if Config.NEAR_DUP_DEDUP_ENABLED:
            chunks = _deduplicate_near_duplicates(chunks, embeddings, Config.NEAR_DUP_THRESHOLD)
        logger.info(f"Created {len(chunks)} chunks after deduplication (strategy: {Config.CHUNKING_STRATEGY})")

        full_text = "\n\n".join(d.page_content for d in docs)
        chunks = _contextualize_chunks(chunks, full_text)

        use_qdrant = bool(Config.QDRANT_URL) and _has_internet()
        if use_qdrant:
            logger.info("Backend: Qdrant Cloud (internet available)")
            result = _build_qdrant(chunks, embeddings, doc_id)
        else:
            reason = "QDRANT_URL not set" if not Config.QDRANT_URL else "no internet"
            logger.info(f"Backend: FAISS local ({reason})")
            vector_store = FAISS.from_documents(chunks, embeddings)
            os.makedirs(os.path.dirname(store_path), exist_ok=True)
            vector_store.save_local(store_path)
            logger.info(f"FAISS index saved: {store_path}")
            result = vector_store, chunks

        manifest[doc_id] = fingerprint
        _save_manifest(manifest)
        return result

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise RuntimeError(f"Vector store creation failed: {e}")


def load_vector_store(store_path: str, doc_id: str = "default"):
    """
    Load an existing vector store.

    Mirrors build_vector_store: tries Qdrant first, falls back to FAISS.
    Returns (vector_store, all_chunks).
    """
    try:
        from app.rag_pipeline import get_embeddings

        embeddings = get_embeddings()
        use_qdrant = bool(Config.QDRANT_URL) and _has_internet()

        if use_qdrant:
            logger.info("Backend: Qdrant Cloud (load)")
            return _load_qdrant(doc_id, embeddings)
        else:
            if not os.path.exists(store_path):
                raise FileNotFoundError(
                    f"FAISS vector store not found at: {store_path}\n"
                    "Run 'python run_me_once.py' first to create it."
                )
            logger.info(f"Backend: FAISS local (load from {store_path})")
            vs = FAISS.load_local(
                store_path, embeddings, allow_dangerous_deserialization=True
            )
            # Extract all chunks from FAISS docstore for BM25
            _doc_ids = list(vs.index_to_docstore_id.values())
            all_chunks = [vs.docstore.search(did) for did in _doc_ids]
            all_chunks = [d for d in all_chunks if d is not None]
            logger.info(f"FAISS: loaded {len(all_chunks)} chunks")
            return vs, all_chunks

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise RuntimeError(f"Vector store loading failed: {e}")
