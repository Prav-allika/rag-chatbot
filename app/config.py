"""
app/config.py — centralised environment configuration.
All other modules import Config from here; nothing else lives in this file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")


class Config:
    """All settings read from environment variables (with safe defaults)."""

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Chunking ---
    CHUNK_SIZE: int       = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int    = int(os.getenv("CHUNK_OVERLAP", 50))
    CHUNKING_STRATEGY: str = os.getenv("CHUNKING_STRATEGY", "fixed")  # fixed | structure | semantic
    SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE: int = int(
        os.getenv("SEMANTIC_CHUNK_BREAKPOINT_PERCENTILE", "90")
    )
    SEMANTIC_CHUNK_MIN_CHARS: int = int(os.getenv("SEMANTIC_CHUNK_MIN_CHARS", "200"))
    NEAR_DUP_DEDUP_ENABLED: bool = os.getenv("NEAR_DUP_DEDUP_ENABLED", "true").lower() == "true"
    NEAR_DUP_THRESHOLD: float = float(os.getenv("NEAR_DUP_THRESHOLD", "0.95"))

    # --- Composite confidence score (weights are normalized, needn't sum to 1) ---
    CONFIDENCE_WEIGHT_RETRIEVAL: float   = float(os.getenv("CONFIDENCE_WEIGHT_RETRIEVAL", "0.5"))
    CONFIDENCE_WEIGHT_COVERAGE: float    = float(os.getenv("CONFIDENCE_WEIGHT_COVERAGE", "0.3"))
    CONFIDENCE_WEIGHT_COMPLETENESS: float = float(os.getenv("CONFIDENCE_WEIGHT_COMPLETENESS", "0.2"))

    # --- Retrieval ---
    RETRIEVAL_K: int         = int(os.getenv("RETRIEVAL_K", 3))
    RETRIEVAL_K_INITIAL: int = int(os.getenv("RETRIEVAL_K_INITIAL", 10))
    SEARCH_TYPE: str         = os.getenv("SEARCH_TYPE", "similarity")
    # RRF fusion weight for dense (FAISS/Qdrant) results; sparse (BM25) gets (1 - this).
    # 0.5 = unweighted (today's default). Higher favors semantic matches, lower favors exact keywords.
    RRF_DENSE_WEIGHT: float  = float(os.getenv("RRF_DENSE_WEIGHT", "0.5"))

    # --- LLM (local fallback) ---
    LLM_MODEL: str       = os.getenv("LLM_MODEL", "google/flan-t5-base")
    LLM_MAX_LENGTH: int  = int(os.getenv("LLM_MAX_LENGTH", 512))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.7))
    LLM_MAX_OUTPUT_TOKENS: int = int(os.getenv("LLM_MAX_OUTPUT_TOKENS", 800))

    # --- OpenAI (optional) ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")

    # --- Groq (recommended) ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    # --- Reranker + CRAG ---
    RERANKER_MODEL: str          = os.getenv(
        "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    GRADE_CORRECT_THRESHOLD: float   = float(os.getenv("GRADE_CORRECT_THRESHOLD", "-2.0"))
    GRADE_AMBIGUOUS_THRESHOLD: float = float(os.getenv("GRADE_AMBIGUOUS_THRESHOLD", "-5.0"))

    # --- Query processing ---
    CONDENSE_QUESTIONS: bool = os.getenv("CONDENSE_QUESTIONS", "true").lower() == "true"
    DECOMPOSE_QUERIES: bool  = os.getenv("DECOMPOSE_QUERIES", "true").lower() == "true"
    HYDE_ENABLED: bool       = os.getenv("HYDE_ENABLED", "false").lower() == "true"
    HISTORY_TURNS: int       = int(os.getenv("HISTORY_TURNS", "2"))

    # --- Contextual Retrieval (Anthropic technique) ---
    CONTEXTUAL_RETRIEVAL: bool = os.getenv("CONTEXTUAL_RETRIEVAL", "false").lower() == "true"
    CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS: int = int(
        os.getenv("CONTEXTUAL_RETRIEVAL_MAX_DOC_CHARS", "30000")
    )

    # --- Semantic cache ---
    SEMANTIC_CACHE_THRESHOLD: float = float(os.getenv("SEMANTIC_CACHE_THRESHOLD", "0.92"))
    SEMANTIC_CACHE_SIZE: int        = int(os.getenv("SEMANTIC_CACHE_SIZE", "100"))

    # --- Redis ---
    REDIS_URL: str       = os.getenv("REDIS_URL", "redis://localhost:6379")
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", str(60 * 60 * 24)))

    # --- Qdrant (online vector DB — used when internet is available) ---
    QDRANT_URL: str     = os.getenv("QDRANT_URL", "")
    QDRANT_API_KEY: str = os.getenv("QDRANT_API_KEY", "")

    # --- Evaluation ---
    RAGAS_EVAL: bool = os.getenv("RAGAS_EVAL", "false").lower() == "true"

    # --- Observability ---
    LANGCHAIN_API_KEY: str  = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str  = os.getenv("LANGCHAIN_PROJECT", "rag-chatbot-v2")

    # --- Convenience predicates ---
    @classmethod
    def use_groq(cls) -> bool:
        return bool(cls.GROQ_API_KEY)

    @classmethod
    def use_openai(cls) -> bool:
        return bool(cls.OPENAI_API_KEY)
