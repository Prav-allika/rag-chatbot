"""
app/chunking.py — switchable chunking strategies.

Three strategies, selected via Config.CHUNKING_STRATEGY:
  fixed      — RecursiveCharacterTextSplitter, fixed size + overlap (baseline)
  structure  — split on detected section headers first, recursive-split
               within oversized sections (structure-aware)
  semantic   — split on topic boundaries detected via embedding-similarity
               drops between consecutive sentences

All three converge on the same output shape (list[Document]) and get the
same metadata tagging (tag_chunks), so downstream code — embedding, BM25,
dedup, contextual retrieval — doesn't need to know which strategy produced
a given chunk.
"""

import logging
import re
from collections import defaultdict

import numpy as np
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

_SEPARATORS = ["\n\n", "\n", " ", ""]

_HEADER_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+$"),                          # Markdown ATX: # Heading
    re.compile(r"^\d{1,2}(\.\d{1,2}){0,3}\s+[A-Z].{2,80}$"),  # Numbered: 2.1 Related Work
    re.compile(r"^[A-Z][A-Z0-9 \-:]{2,79}$"),               # ALL-CAPS short line
]


def _is_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 100:
        return False
    return any(p.match(stripped) for p in _HEADER_PATTERNS)


def _group_by_source(docs: list) -> dict:
    """Group Documents by metadata['source'], preserving input order within each group."""
    groups = defaultdict(list)
    for doc in docs:
        groups[doc.metadata.get("source", "")].append(doc)
    return groups


def _split_text_preserving_tables(text: str, chunk_size: int, sub_splitter) -> list:
    """
    Splits text via sub_splitter, but keeps any "[TABLE]..." block (as produced
    by document_loader._pdfplumber_table_text, one block per "\\n\\n"-separated
    paragraph) atomic — never split mid-table. A table row or its row-group
    label (e.g. "(C)") landing in a different chunk than the rest of its row
    makes the numbers unreadable or misattributable — observed directly: a
    4-row table truncated to 3 rows right before the row a question needed.
    """
    paragraphs = text.split("\n\n")
    out = []
    prose_buffer: list = []
    table_size_cap = chunk_size * 4   # safety net against one pathologically huge table

    def _flush_prose():
        if not prose_buffer:
            return
        joined = "\n\n".join(prose_buffer)
        prose_buffer.clear()
        if joined.strip():
            out.extend(sub_splitter.split_text(joined))

    for p in paragraphs:
        if p.lstrip().startswith("[TABLE]"):
            _flush_prose()
            if not p.strip():
                continue
            if len(p) > table_size_cap:
                logger.warning(f"Table block ({len(p)} chars) exceeds safety cap — splitting despite atomicity goal")
                out.extend(sub_splitter.split_text(p))
            else:
                out.append(p.strip())
        else:
            prose_buffer.append(p)
    _flush_prose()
    return out


# =============================================================================
# STRATEGY 1 — FIXED-SIZE (baseline)
# =============================================================================
def split_fixed(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """RecursiveCharacterTextSplitter over the raw page/section Documents,
    with [TABLE] blocks kept atomic (see _split_text_preserving_tables)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=_SEPARATORS,
    )
    out = []
    for doc in docs:
        for piece in _split_text_preserving_tables(doc.page_content, chunk_size, splitter):
            out.append(Document(page_content=piece, metadata=dict(doc.metadata)))
    return out


# =============================================================================
# STRATEGY 2 — STRUCTURE-AWARE (split on section headers)
# =============================================================================
def split_structure_aware(docs: list, chunk_size: int, chunk_overlap: int) -> list:
    """
    Splits on detected section-header boundaries (Markdown ATX, numbered
    sections, ALL-CAPS lines) first, falling back to RecursiveCharacterTextSplitter
    within any section that's still bigger than chunk_size.

    Page metadata is approximate for sections that span multiple source
    pages/Documents — acceptable here since the goal is comparing retrieval
    quality across strategies, not page-perfect citation.
    """
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap,
        length_function=len, separators=_SEPARATORS,
    )

    out = []
    for source, group in _group_by_source(docs).items():
        base_metadata = dict(group[0].metadata)
        heading = ""
        buffer_lines: list = []

        def _flush():
            text = "\n".join(buffer_lines).strip()
            if not text:
                return
            if len(text) <= chunk_size:
                out.append(Document(
                    page_content=text,
                    metadata={**base_metadata, "section_heading": heading},
                ))
            else:
                for piece in _split_text_preserving_tables(text, chunk_size, sub_splitter):
                    out.append(Document(
                        page_content=piece,
                        metadata={**base_metadata, "section_heading": heading},
                    ))

        for doc in group:
            for line in doc.page_content.split("\n"):
                if _is_header(line):
                    _flush()
                    buffer_lines = []
                    heading = line.strip()
                buffer_lines.append(line)
        _flush()

    return out


# =============================================================================
# STRATEGY 3 — SEMANTIC (topic-boundary via embedding similarity)
# =============================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_semantic(
    docs: list,
    embeddings,
    chunk_size: int,
    breakpoint_percentile: int = 90,
    min_chars: int = 200,
) -> list:
    """
    Splits each source document into sentences, embeds them, and cuts a new
    chunk wherever the cosine distance between consecutive sentences exceeds
    the `breakpoint_percentile`-th percentile of distances in that document
    (i.e. a bigger-than-usual topic jump). Small merged chunks (< min_chars)
    are folded into the next chunk; oversized ones are sub-split.

    Costs one embedding call per sentence — noticeably slower/pricier than
    the other two strategies. That tradeoff is what the comparison report
    is meant to surface.
    """
    sub_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0,
        length_function=len, separators=_SEPARATORS,
    )

    out = []
    for source, group in _group_by_source(docs).items():
        base_metadata = dict(group[0].metadata)
        full_text = "\n\n".join(d.page_content for d in group)
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(full_text) if s.strip()]

        if len(sentences) <= 3:
            if full_text.strip():
                out.append(Document(page_content=full_text.strip(), metadata=dict(base_metadata)))
            continue

        vectors = np.array(embeddings.embed_documents(sentences))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-8
        unit = vectors / norms
        sims = np.sum(unit[:-1] * unit[1:], axis=1)
        distances = 1.0 - sims

        threshold = np.percentile(distances, breakpoint_percentile)
        breakpoints = set(int(i) for i in np.where(distances > threshold)[0])

        raw_chunks, current = [], [sentences[0]]
        for i in range(1, len(sentences)):
            if (i - 1) in breakpoints:
                raw_chunks.append(" ".join(current))
                current = []
            current.append(sentences[i])
        if current:
            raw_chunks.append(" ".join(current))

        # merge tiny chunks forward so single-sentence topic slivers don't
        # become their own noisy chunk
        merged: list = []
        for chunk in raw_chunks:
            if merged and len(merged[-1]) < min_chars:
                merged[-1] = merged[-1] + " " + chunk
            else:
                merged.append(chunk)

        for chunk in merged:
            if len(chunk) <= chunk_size:
                out.append(Document(page_content=chunk, metadata=dict(base_metadata)))
            else:
                for piece in _split_text_preserving_tables(chunk, chunk_size, sub_splitter):
                    out.append(Document(page_content=piece, metadata=dict(base_metadata)))

    return out


# =============================================================================
# METADATA TAGGING (applied uniformly, regardless of strategy)
# =============================================================================
def tag_chunks(chunks: list, strategy: str) -> list:
    per_source_index: dict = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        chunk.metadata["chunking_strategy"] = strategy
        chunk.metadata["chunk_index"] = per_source_index[source]
        chunk.metadata["char_count"] = len(chunk.page_content)
        per_source_index[source] += 1
    return chunks


# =============================================================================
# DISPATCHER
# =============================================================================
def split_documents(
    docs: list,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embeddings=None,
    semantic_breakpoint_percentile: int = 90,
    semantic_min_chars: int = 200,
) -> list:
    if strategy == "structure":
        chunks = split_structure_aware(docs, chunk_size, chunk_overlap)
    elif strategy == "semantic":
        if embeddings is None:
            raise ValueError("split_documents(strategy='semantic') requires an embeddings model")
        chunks = split_semantic(
            docs, embeddings, chunk_size,
            breakpoint_percentile=semantic_breakpoint_percentile,
            min_chars=semantic_min_chars,
        )
    elif strategy == "fixed":
        chunks = split_fixed(docs, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown CHUNKING_STRATEGY '{strategy}' — expected fixed | structure | semantic")

    logger.info(f"Chunking strategy '{strategy}': produced {len(chunks)} chunk(s)")
    return tag_chunks(chunks, strategy)
