"""
FastAPI application for RAG Chatbot

Multi-document API: startup ingests the default document (data/Attention.pdf,
downloaded if missing) under doc_id "Attention.pdf" for backward compatibility
with existing /ask callers that don't pass a doc_id; POST /documents/ingest adds
more documents at runtime, GET /documents lists what's indexed.
"""

import os
import time
import logging
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import (
    build_vector_store,
    load_vector_store,
    make_qa_chain,
    _SUPPORTED_EXTENSIONS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_DOC_ID = "Attention.pdf"
UPLOAD_DIR = "data/uploads"


# ---------- Helper Functions ----------
def ensure_pdf_exists():
    """Download PDF from GitHub if it doesn't exist locally"""
    pdf_path = "data/Attention.pdf"

    if not os.path.exists(pdf_path):
        logger.warning(" PDF not found locally. Downloading from GitHub...")
        os.makedirs("data", exist_ok=True)
        pdf_url = (
            "https://github.com/Prav-allika/rag-chatbot/raw/main/data/Attention.pdf"
        )
        try:
            logger.info(f" Downloading PDF from {pdf_url}")
            urllib.request.urlretrieve(pdf_url, pdf_path)
            logger.info(
                f" PDF downloaded successfully ({os.path.getsize(pdf_path)} bytes)"
            )
        except Exception as e:
            logger.error(f" Failed to download PDF: {e}")
            raise FileNotFoundError(f"Could not download PDF from {pdf_url}")
    else:
        logger.info(
            f" PDF already exists at {pdf_path} ({os.path.getsize(pdf_path)} bytes)"
        )

    return pdf_path


# ---------- Document registry ----------
# doc_id -> {"vector_store", "qa_chain", "num_chunks", "loaded_at", "source_filename"}
_documents: dict = {}


def _register(doc_id: str, vector_store, chunks: list, source_filename: str) -> dict:
    entry = {
        "vector_store": vector_store,
        "qa_chain": make_qa_chain(vector_store, doc_id=doc_id, all_chunks=chunks),
        "num_chunks": len(chunks) if chunks else 0,
        "loaded_at": time.time(),
        "source_filename": source_filename,
    }
    _documents[doc_id] = entry
    return entry


def _public_doc_info(doc_id: str, entry: dict) -> dict:
    return {
        "doc_id": doc_id,
        "num_chunks": entry["num_chunks"],
        "loaded_at": entry["loaded_at"],
        "source_filename": entry["source_filename"],
    }


# ---------- Lifespan Events ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    logger.info("🚀 Starting RAG Chatbot API...")

    try:
        store_path = "artifacts/vector_store"
        pdf_path = ensure_pdf_exists()

        if not os.path.exists(store_path):
            logger.info(" Building vector store (first startup takes 2-3 minutes)...")
            vector_store, chunks = build_vector_store(pdf_path, store_path, doc_id=DEFAULT_DOC_ID)
        else:
            logger.info(" Loading existing vector store...")
            vector_store, chunks = load_vector_store(store_path, doc_id=DEFAULT_DOC_ID)

        _register(DEFAULT_DOC_ID, vector_store, chunks, os.path.basename(pdf_path))
        logger.info(f" QA chain ready for '{DEFAULT_DOC_ID}'!")

        yield

        logger.info(" Shutting down...")
        _documents.clear()

    except Exception as e:
        logger.error(f" Failed to initialize: {e}")
        raise


# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready document Q&A system powered by Retrieval-Augmented Generation",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Request/Response Models ----------
class QuestionRequest(BaseModel):
    """Request model for asking questions."""

    question: str = Field(
        ..., min_length=1, max_length=1000, description="Question to ask"
    )
    doc_id: Optional[str] = Field(
        default=None,
        description=f"Document to query. Defaults to '{DEFAULT_DOC_ID}' if omitted.",
    )


class AnswerResponse(BaseModel):
    """Response model for answers."""

    answer: str = Field(..., description="Generated answer")
    doc_id: str = Field(..., description="Document the answer was generated from")
    sources: Optional[list] = Field(default=None, description="Retrieved source chunks (chunk, page, score)")
    confidence: Optional[dict] = Field(
        default=None, description="Composite confidence score (retrieval, citation_coverage, completeness)"
    )
    processing_time: float = Field(
        ..., description="Time taken to process the request in seconds"
    )
    status: str = Field(default="success", description="Status of the request")


class IngestResponse(BaseModel):
    """Response model for document ingestion."""

    doc_id: str
    num_chunks: int
    status: str = "ingested"


class DocumentInfo(BaseModel):
    """Metadata about one indexed document."""

    doc_id: str
    num_chunks: int
    loaded_at: float
    source_filename: str


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    documents_loaded: int
    timestamp: float


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str
    detail: Optional[str] = None
    status: str = "error"


# ---------- Exception Handlers ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail, status="error").dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error", detail=str(exc), status="error"
        ).dict(),
    )


# ---------- Routes ----------
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.1.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint — healthy if at least one document is indexed."""
    return HealthResponse(
        status="healthy" if _documents else "unhealthy",
        documents_loaded=len(_documents),
        timestamp=time.time(),
    )


@app.get("/documents", response_model=list[DocumentInfo], tags=["Documents"])
async def list_documents():
    """List all currently indexed documents."""
    return [_public_doc_info(doc_id, entry) for doc_id, entry in _documents.items()]


@app.post("/documents/ingest", response_model=IngestResponse, status_code=201, tags=["Documents"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Index a new document for Q&A.

    Accepts PDF, DOCX, HTML, TXT, or Markdown. The returned doc_id (the
    uploaded filename) is what subsequent /ask calls should pass to target
    this document.
    """
    ext = Path(file.filename).suffix.lower()
    if ext not in _SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(_SUPPORTED_EXTENSIONS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '{ext}'. Supported: {supported}",
        )

    doc_id = file.filename
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    dest_path = os.path.join(UPLOAD_DIR, doc_id)

    try:
        contents = await file.read()
        with open(dest_path, "wb") as f:
            f.write(contents)

        logger.info(f"Ingesting '{doc_id}' ({len(contents)} bytes)...")
        vector_store, chunks = build_vector_store(
            dest_path, f"artifacts/vector_store/{doc_id}", doc_id=doc_id
        )
        entry = _register(doc_id, vector_store, chunks, doc_id)
        logger.info(f"Ingested '{doc_id}': {entry['num_chunks']} chunks")

        return IngestResponse(doc_id=doc_id, num_chunks=entry["num_chunks"])

    except Exception as e:
        logger.error(f"Ingestion failed for '{doc_id}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}",
        )


@app.post("/ask", response_model=AnswerResponse, status_code=200, tags=["QA"])
async def ask_question(payload: QuestionRequest):
    """
    Ask a question about an indexed document.

    Args:
    - **question**: The question to ask (1-1000 characters)
    - **doc_id**: Which document to query (optional — defaults to the startup document)

    Returns:
    - **answer**: Generated answer, with **sources** and a composite **confidence** score
    - **processing_time**: Time taken to process the request
    """
    doc_id = payload.doc_id or DEFAULT_DOC_ID
    entry = _documents.get(doc_id)
    if entry is None:
        available = ", ".join(_documents.keys()) or "(none)"
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"doc_id '{doc_id}' not found. Available: {available}",
        )

    start_time = time.time()

    try:
        logger.info(f"[{doc_id}] Processing question: {payload.question[:50]}...")

        result = entry["qa_chain"].invoke({"query": payload.question})
        answer = result.get("result", "No answer generated")

        processing_time = time.time() - start_time
        logger.info(f" Answer generated in {processing_time:.2f}s")

        return AnswerResponse(
            answer=answer,
            doc_id=doc_id,
            sources=result.get("sources"),
            confidence=result.get("confidence"),
            processing_time=round(processing_time, 3),
            status="success",
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """Get basic metrics about the API."""
    return {
        "documents_loaded": len(_documents),
        "doc_ids": list(_documents.keys()),
        "timestamp": time.time(),
    }
