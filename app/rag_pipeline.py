"""
RAG Pipeline - Modern LangChain 1.0+ (LCEL Pattern)
"""

import os
import logging
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)


# ---------- Configuration ----------
class Config:
    """Configuration for RAG pipeline."""

    EMBEDDING_MODEL = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 3))
    SEARCH_TYPE = os.getenv("SEARCH_TYPE", "similarity")
    LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
    LLM_MAX_LENGTH = int(os.getenv("LLM_MAX_LENGTH", 512))
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-8b-8192")

    @classmethod
    def use_openai(cls):
        return bool(cls.OPENAI_API_KEY)

    @classmethod
    def use_groq(cls):
        return bool(cls.GROQ_API_KEY)


# ---------- Cached Model Loading ----------
@lru_cache(maxsize=1)
def get_embeddings():
    """Return embeddings model (cached)."""
    try:
        if Config.use_openai():
            logger.info("Using OpenAI Embeddings")
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(openai_api_key=Config.OPENAI_API_KEY)
        else:
            logger.info(f"Using HuggingFace Embeddings: {Config.EMBEDDING_MODEL}")
            return HuggingFaceEmbeddings(
                model_name=Config.EMBEDDING_MODEL,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")
        raise RuntimeError(f"Embeddings initialization failed: {e}")


@lru_cache(maxsize=1)
def get_llm():
    """Return LLM model (cached).

    Priority order:
      1. Groq  — fastest, generous free tier (set GROQ_API_KEY)
      2. OpenAI — best quality (set OPENAI_API_KEY)
      3. HuggingFace FLAN-T5 — fully local, no API key needed (slow on CPU)
    """
    try:
        if Config.use_groq():
            logger.info(f"Using Groq: {Config.GROQ_MODEL}")
            from langchain_groq import ChatGroq

            return ChatGroq(
                model=Config.GROQ_MODEL,
                groq_api_key=Config.GROQ_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
            )

        elif Config.use_openai():
            logger.info(f"Using OpenAI: {Config.OPENAI_MODEL}")
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=Config.OPENAI_MODEL,
                openai_api_key=Config.OPENAI_API_KEY,
                temperature=Config.LLM_TEMPERATURE,
            )
        else:
            logger.info(f"Using HuggingFace LLM: {Config.LLM_MODEL}")
            from transformers import (
                AutoTokenizer,
                AutoModelForSeq2SeqLM,
                GenerationConfig,
            )
            from langchain_community.llms import HuggingFacePipeline
            from transformers import pipeline as hf_pipeline

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(Config.LLM_MODEL)
            model = AutoModelForSeq2SeqLM.from_pretrained(Config.LLM_MODEL)

            # Create pipeline with strict generation config
            pipe = hf_pipeline(
                task="text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512,
                do_sample=False,
                temperature=0.7,
                device=-1,
            )

            return HuggingFacePipeline(
                pipeline=pipe,
                model_kwargs={
                    "max_new_tokens": 150,
                    "min_length": 20,
                    "early_stopping": True,
                },
            )
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        raise RuntimeError(f"LLM initialization failed: {e}")


# ---------- Vector Store Functions ----------
def build_vector_store(pdf_path: str, store_path: str):
    """Load PDF, split into chunks, embed, and save FAISS index."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        logger.info(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No documents loaded from PDF")

        logger.info(f"Loaded {len(docs)} pages")

        logger.info(
            f"Splitting into chunks (size={Config.CHUNK_SIZE}, overlap={Config.CHUNK_OVERLAP})"
        )
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        logger.info(f"Created {len(chunks)} chunks")

        logger.info("Creating embeddings...")
        embeddings = get_embeddings()
        vector_store = FAISS.from_documents(chunks, embeddings)

        logger.info(f"Saving vector store to: {store_path}")
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        vector_store.save_local(store_path)
        logger.info("Vector store saved successfully")

        return vector_store

    except Exception as e:
        logger.error(f"Failed to build vector store: {e}")
        raise RuntimeError(f"Vector store creation failed: {e}")


def load_vector_store(store_path: str):
    """Load existing FAISS vector store."""
    if not os.path.exists(store_path):
        raise FileNotFoundError(
            f"Vector store not found at: {store_path}\n"
            f"Please run 'python run_me_once.py' first to create it."
        )

    try:
        logger.info(f"Loading vector store from: {store_path}")
        embeddings = get_embeddings()

        vector_store = FAISS.load_local(
            store_path, embeddings, allow_dangerous_deserialization=True
        )
        logger.info("Vector store loaded successfully")
        return vector_store

    except Exception as e:
        logger.error(f"Failed to load vector store: {e}")
        raise RuntimeError(f"Vector store loading failed: {e}")


# ---------- QA Chain (Modern LCEL Pattern) ----------
def make_qa_chain(vector_store):
    """
    Build QA chain using modern LangChain 1.0+ LCEL pattern.
    Supports: source chunk retrieval, follow-up context via history.
    """
    try:
        retriever = vector_store.as_retriever(
            search_type=Config.SEARCH_TYPE, search_kwargs={"k": Config.RETRIEVAL_K}
        )

        template = """Use the following context from the document to answer the question.
If previous conversation is provided, use it to understand follow-up questions.
If you don't know the answer from the context, say so clearly. Keep your answer concise.

Document context:
{context}

Question: {question}

Answer:"""

        prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

        llm = get_llm()

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        logger.info("Building QA chain (LCEL pattern)...")
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        class QAChainWrapper:
            """
            Wrapper supporting:
            - Source chunk retrieval (returns paragraphs used for the answer)
            - Follow-up context (injects recent conversation history into question)
            """

            def __init__(self, chain, retriever):
                self._chain = chain
                self._retriever = retriever

            def invoke(self, inputs):
                question = inputs.get("query") or inputs.get("question")
                history = inputs.get("history", "").strip()

                if not question:
                    raise ValueError("No question provided in inputs")

                # Inject recent history for follow-up question support
                if history:
                    recent = history[-500:] if len(history) > 500 else history
                    full_question = (
                        f"[Conversation so far for context]\n{recent}\n\n"
                        f"[Current question]: {question}"
                    )
                else:
                    full_question = question

                # Fetch source docs separately for display
                source_docs = self._retriever.invoke(question)

                # Run chain
                result = self._chain.invoke(full_question)

                # Format source chunks
                sources = []
                for i, doc in enumerate(source_docs):
                    sources.append(
                        {
                            "chunk": i + 1,
                            "content": doc.page_content[:300],
                            "page": doc.metadata.get("page", "N/A"),
                        }
                    )

                return {"result": result, "sources": sources}

            def run(self, question):
                return self._chain.invoke(question)

        wrapped_chain = QAChainWrapper(rag_chain, retriever)
        logger.info("QA chain built successfully (LCEL)")
        return wrapped_chain

    except Exception as e:
        logger.error(f"Failed to create QA chain: {e}")
        raise RuntimeError(f"QA chain creation failed: {e}")


# ---------- Helper Functions ----------
def get_vector_store_info(store_path: str):
    """Get information about vector store."""
    if not os.path.exists(store_path):
        return {"exists": False}

    try:
        vector_store = load_vector_store(store_path)
        return {
            "exists": True,
            "num_documents": vector_store.index.ntotal,
            "path": store_path,
        }
    except Exception as e:
        return {"exists": True, "error": str(e)}
