"""
RAG Chatbot API
FastAPI application for document-based question answering using RAG.
"""

import logging
import time
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from app.rag_pipeline import load_vector_store, make_qa_chain
except ImportError:
    from rag_pipeline import load_vector_store, make_qa_chain

# ---------- Logging Configuration ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ---------- Global State ----------
vector_store = None
qa_chain = None

# ---------- Lifespan Events ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models on startup, cleanup on shutdown."""
    global vector_store, qa_chain
    
    logger.info("🚀 Starting RAG Chatbot API...")
    try:
        # Load vector store
        logger.info("📚 Loading vector store...")
        vector_store = load_vector_store("artifacts/vector_store")
        logger.info("✅ Vector store loaded successfully")
        
        # Build QA chain
        logger.info("🔗 Building QA chain...")
        qa_chain = make_qa_chain(vector_store)
        logger.info("✅ QA chain ready")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down RAG Chatbot API...")

# ---------- FastAPI App ----------
app = FastAPI(
    title="RAG Chatbot API",
    version="1.0.0",
    description="Retrieval-Augmented Generation chatbot for document Q&A",
    lifespan=lifespan
)

# ---------- CORS Configuration ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Request/Response Schemas ----------
class QuestionRequest(BaseModel):
    """Request model for asking questions."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Question to ask the chatbot",
        example="What is attention mechanism in transformers?"
    )

class AnswerResponse(BaseModel):
    """Response model with answer and metadata."""
    answer: str = Field(..., description="Generated answer")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(default="success", description="Response status")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    vector_store_loaded: bool
    qa_chain_loaded: bool
    timestamp: float

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    status: str = "error"

# ---------- Exception Handlers ----------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            status="error"
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc) if app.debug else None,
            status="error"
        ).dict()
    )

# ---------- Routes ----------
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Chatbot API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns the status of the API and its dependencies.
    """
    return HealthResponse(
        status="healthy" if vector_store and qa_chain else "unhealthy",
        vector_store_loaded=vector_store is not None,
        qa_chain_loaded=qa_chain is not None,
        timestamp=time.time()
    )

@app.post(
    "/ask",
    response_model=AnswerResponse,
    tags=["QA"],
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "Successfully generated answer"},
        400: {"description": "Invalid request"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable"}
    }
)
async def ask_question(payload: QuestionRequest):
    """
    Ask a question to the RAG chatbot.
    
    The chatbot retrieves relevant context from indexed documents
    and generates an answer using the language model.
    
    - **question**: Your question (3-500 characters)
    
    Returns:
    - **answer**: Generated answer
    - **processing_time**: Time taken to process the request
    """
    # Check if services are ready
    if not vector_store or not qa_chain:
        logger.error("Services not initialized")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="QA service is not ready. Please try again later."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing question: {payload.question[:50]}...")
        
        # Generate answer
        result = qa_chain.invoke({"query": payload.question})
        answer = result.get("result", "No answer generated")
        
        processing_time = time.time() - start_time
        logger.info(f"✅ Answer generated in {processing_time:.2f}s")
        
        return AnswerResponse(
            answer=answer,
            processing_time=round(processing_time, 3),
            status="success"
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate answer: {str(e)}"
        )

# ---------- Metrics Endpoint (Optional) ----------
@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Basic metrics endpoint.
    In production, integrate with Prometheus or similar.
    """
    return {
        "vector_store_loaded": vector_store is not None,
        "qa_chain_loaded": qa_chain is not None,
        "status": "operational"
    }

# ---------- Entry Point ----------
if __name__ == "__main__":
    import uvicorn
    
    logger.info("🚀 Starting development server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
