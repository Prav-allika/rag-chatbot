#!/bin/bash
set -e

echo "🚀 Starting RAG Chatbot API..."

# Check if vector store exists, if not build it
if [ ! -d "artifacts/vector_store" ]; then
    echo "📦 Vector store not found. Building..."
    python run_me_once.py
else
    echo "✅ Vector store exists"
fi

# Start the API
echo "🌐 Starting FastAPI server..."
uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
