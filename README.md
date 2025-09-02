# 🧠 RAG Chatbot API (FastAPI + Docker)

A **Retrieval-Augmented Generation (RAG) Chatbot API** built with:
- **FastAPI** for serving endpoints
- **LangChain** for pipeline orchestration
- **HuggingFace Transformers** for free embeddings + LLMs
- **FAISS** for vector database
- **Docker** for containerization

---

## 🚀 Features
- `/health` → Health check endpoint
- `/ask` → Ask any question, chatbot retrieves context from PDF + generates an answer
- Uses HuggingFace `flan-t5-base` by default (no paid API needed!)
- Fully containerized with Docker

---

## 📂 Project Structure

