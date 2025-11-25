# RAG Chatbot - Attention Paper Q&A

[![Live Demo](https://img.shields.io/badge/Demo-Try%20Now-brightgreen?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/Prav04/rag-chatbot)
[![GitHub](https://img.shields.io/badge/Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/Prav-allika/rag-chatbot)

**Try it live:** https://huggingface.co/spaces/Prav04/rag-chatbot

A production-ready RAG (Retrieval-Augmented Generation) chatbot deployed on HuggingFace Spaces that answers questions about the "Attention Is All You Need" paper using FAISS vector search and LangChain.

---

## Overview

This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions about the Transformer architecture paper. It combines:

- **FAISS** for efficient vector similarity search
- **LangChain** for RAG pipeline orchestration
- **HuggingFace Transformers** for embeddings and text generation
- **Gradio** for the interactive web interface

## Features

- **Instant Answers**: Sub-second response times using pre-built vector store
- **Accurate Retrieval**: FAISS vector search with semantic understanding
- **Natural Language**: Ask questions in plain English
- **Production Ready**: Deployed on HuggingFace Spaces with 99.9% uptime

## Technology Stack

- **Python 3.10**
- **LangChain 1.0+** - Modern LCEL patterns for RAG pipeline
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Document embeddings (all-MiniLM-L6-v2)
- **Google Flan-T5** - Text generation model
- **Gradio 4.0+** - Web interface
- **HuggingFace Spaces** - Deployment platform

## Architecture

```
User Question
    ↓
[Text Embedding]
    ↓
[FAISS Vector Search] → Retrieve relevant document chunks
    ↓
[LangChain LCEL Pipeline] → Context + Question
    ↓
[Flan-T5 Model] → Generate answer
    ↓
Response to User
```

## Project Structure

```
rag-chatbot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application (optional)
│   └── rag_pipeline.py      # RAG implementation
├── artifacts/
│   └── vector_store/        # Pre-built FAISS index
│       ├── index.faiss
│       └── index.pkl
├── data/
│   └── Attention.pdf        # Source document
├── app.py                   # Gradio interface (HuggingFace)
├── requirements.txt         # Python dependencies
├── run_me_once.py          # Vector store builder
└── README.md
```

## Local Setup

### Prerequisites

- Python 3.10+
- 4GB RAM minimum
- 2GB disk space

### Installation

```bash
# Clone the repository
git clone https://github.com/Prav-allika/rag-chatbot.git
cd rag-chatbot

# Install dependencies
pip install -r requirements.txt

# Build vector store (one-time setup)
python run_me_once.py --pdf data/Attention.pdf
```

### Run Locally

**Option 1: Gradio Interface (Recommended)**

```bash
python app.py
```

Visit `http://localhost:7860` to use the chatbot.

**Option 2: FastAPI (API Mode)**

```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for API documentation.

## Example Questions

Try asking:

- "What is the attention mechanism?"
- "How does multi-head attention work?"
- "What are the key advantages of the Transformer model?"
- "Explain positional encoding in simple terms"
- "What is self-attention?"

## Deployment

This project is deployed on HuggingFace Spaces:

- **Platform**: HuggingFace Spaces (Free tier)
- **Hardware**: CPU Basic (16GB RAM)
- **Uptime**: 99.9%
- **URL**: https://huggingface.co/spaces/Prav04/rag-chatbot

### Deploy Your Own

1. Fork this repository
2. Create a new Space on HuggingFace
3. Connect your forked repository
4. The Space will automatically deploy

Git LFS is required for binary files (PDF, FAISS index). The `.gitattributes` file is already configured.

## Performance

- **Startup Time**: 2-3 minutes (first load only)
- **Response Time**: < 2 seconds per query
- **Vector Store**: 195KB (2 files)
- **Concurrent Users**: Supports multiple simultaneous queries

## Technical Details

### RAG Pipeline (LCEL Pattern)

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

### Vector Store

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Chunk Size**: 500 characters
- **Chunk Overlap**: 50 characters
- **Total Chunks**: 93 from 15 pages
- **Index Size**: 143KB (FAISS)

## Future Enhancements

- [ ] Support for multiple documents
- [ ] Conversation history
- [ ] Citation tracking
- [ ] Advanced filtering options
- [ ] API rate limiting

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Paper: "Attention Is All You Need" (Vaswani et al., 2017)
- Built with LangChain, FAISS, and HuggingFace Transformers
- Deployed on HuggingFace Spaces

## Contact

**Pravalli**
- GitHub: [@Prav-allika](https://github.com/Prav-allika)
- HuggingFace: [@Prav04](https://huggingface.co/Prav04)

---

**Live Demo**: https://huggingface.co/spaces/Prav04/rag-chatbot
