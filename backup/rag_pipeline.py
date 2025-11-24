import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

# Check if OpenAI is available
USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))

# Embeddings
if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings, ChatOpenAI
else:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline


def get_embeddings():
    """Return embeddings: OpenAI if key is set, else HuggingFace (free)."""
    if USE_OPENAI:
        print("✅ Using OpenAI Embeddings")
        return OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        print("✅ Using HuggingFace Embeddings (free)")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )


def build_vector_store(pdf_path: str, store_path: str):
    """Load PDF, split into chunks, embed, and save FAISS index."""
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(store_path)
    return vector_store


def load_vector_store(store_path: str):
    """Load existing FAISS vector store."""
    embeddings = get_embeddings()
    return FAISS.load_local(
        store_path, embeddings, allow_dangerous_deserialization=True
    )


def make_qa_chain(vector_store):
    """Build RetrievalQA chain with retriever + appropriate LLM."""
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    if USE_OPENAI:
        print("✅ Using OpenAI ChatGPT (gpt-3.5-turbo)")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
        )
    else:
        print("✅ Using HuggingFace LLM (google/flan-t5-base)")
        model_name = "google/flan-t5-base"  # can switch to flan-t5-large, mistral, etc.
        hf_pipeline = pipeline(
            "text2text-generation", model=model_name, tokenizer=model_name
        )
        llm = HuggingFacePipeline(pipeline=hf_pipeline)

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return chain
