"""
FastAPI application for RAG Chatbot
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.rag_pipeline import load_vector_store, make_qa_chain

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global variables for vector store and QA chain
vector_store = None
qa_chain = None


# ---------- Lifespan Events ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan events for startup and shutdown."""
    global vector_store, qa_chain

    logger.info(" Starting RAG Chatbot API...")

    try:
        store_path = "artifacts/vector_store"
        pdf_path = "data/Attention.pdf"

        # Check if vector store exists
        if not os.path.exists(store_path):
            logger.warning(" Vector store not found. Building now...")

            # Check if PDF exists
            if not os.path.exists(pdf_path):
                logger.error(f" PDF not found at: {pdf_path}")
                raise FileNotFoundError(f"PDF not found: {pdf_path}")

            # Build vector store
            logger.info(" Building vector store (first startup takes 2-3 minutes)...")
            from app.rag_pipeline import build_vector_store

            vector_store = build_vector_store(pdf_path, store_path)
            logger.info(" Vector store built successfully!")
        else:
            # Load existing vector store
            logger.info("📚 Loading existing vector store...")
            vector_store = load_vector_store(store_path)

        # Create QA chain
        logger.info(" Creating QA chain...")
        qa_chain = make_qa_chain(vector_store)
        logger.info(" QA chain ready!")

        yield

        # Cleanup
        logger.info(" Shutting down...")
        vector_store = None
        qa_chain = None

    except Exception as e:
        logger.error(f" Failed to initialize: {e}")
        raise


# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG Chatbot API",
    description="Production-ready document Q&A system powered by Retrieval-Augmented Generation",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
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


class AnswerResponse(BaseModel):
    """Response model for answers."""

    answer: str = Field(..., description="Generated answer")
    processing_time: float = Field(
        ..., description="Time taken to process the request in seconds"
    )
    status: str = Field(default="success", description="Status of the request")


class HealthResponse(BaseModel):
    """Response model for health check."""

    status: str
    vector_store_loaded: bool
    qa_chain_loaded: bool
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
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the health status of the API and its dependencies.
    """
    return HealthResponse(
        status="healthy" if (vector_store and qa_chain) else "unhealthy",
        vector_store_loaded=vector_store is not None,
        qa_chain_loaded=qa_chain is not None,
        timestamp=time.time(),
    )


@app.post("/ask", response_model=AnswerResponse, status_code=200, tags=["QA"])
async def ask_question(payload: QuestionRequest):
    """
    Ask a question about the document.

    This endpoint uses RAG (Retrieval-Augmented Generation) to answer questions
    based on the content of the uploaded PDF document.

    Args:
    - **question**: The question to ask (1-1000 characters)

    Returns:
    - **answer**: Generated answer
    - **processing_time**: Time taken to process the request
    """
    # Check if services are ready
    if not vector_store or not qa_chain:
        logger.error("Services not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA service is not ready. Please try again later.",
        )

    start_time = time.time()

    try:
        logger.info(f"Processing question: {payload.question[:50]}...")

        # Generate answer
        result = qa_chain.invoke({"query": payload.question})
        answer = result.get("result", "No answer generated")

        processing_time = time.time() - start_time
        logger.info(f" Answer generated in {processing_time:.2f}s")

        return AnswerResponse(
            answer=answer, processing_time=round(processing_time, 3), status="success"
        )

    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process question: {str(e)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Get basic metrics about the API.
    """
    return {
        "vector_store_status": "loaded" if vector_store else "not_loaded",
        "qa_chain_status": "loaded" if qa_chain else "not_loaded",
        "timestamp": time.time(),
    }
