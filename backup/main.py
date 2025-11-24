try:
    # When running with uvicorn (module mode)
    from app.rag_pipeline import load_vector_store, make_qa_chain
except ImportError:
    # When running directly with python app/main.py
    from rag_pipeline import load_vector_store, make_qa_chain

from fastapi import FastAPI
from pydantic import BaseModel

# ---------- FastAPI app ----------
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# ---------- Load vector store + QA chain ----------
vector_store = load_vector_store("artifacts/vector_store")
qa_chain = make_qa_chain(vector_store)


# ---------- Request/Response Schemas ----------
class Question(BaseModel):
    question: str


class Answer(BaseModel):
    answer: str


# ---------- Routes ----------
@app.get("/health")
def health():
    return {"status": "ok", "vector_store_loaded": vector_store is not None}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    try:
        result = qa_chain.run(payload.question)
        return Answer(answer=result)
    except Exception as e:
        return {"error": str(e)}


# ---------- Entry point ----------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
