import gradio as gr
import os
from app.rag_pipeline import load_vector_store, make_qa_chain

# Load pre-built vector store
print("🚀 Loading vector store...")
vector_store = load_vector_store("artifacts/vector_store")
qa_chain = make_qa_chain(vector_store)
print("✅ Ready!")

def answer_question(question):
    """Answer questions about the Attention paper"""
    if not question.strip():
        return "Please ask a question!"
    
    try:
        result = qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return f"Error: {str(e)}"

# Create Gradio interface
demo = gr.Interface(
    fn=answer_question,
    inputs=gr.Textbox(
        label="Ask a question about the 'Attention Is All You Need' paper",
        placeholder="e.g., What is the attention mechanism?",
        lines=2
    ),
    outputs=gr.Textbox(label="Answer", lines=5),
    title="🤖 RAG Chatbot - Attention Paper Q&A",
    description="Ask questions about the Transformer architecture paper using RAG (Retrieval-Augmented Generation)",
    examples=[
        "What is the attention mechanism?",
        "How does multi-head attention work?",
        "What are the key advantages of the Transformer model?",
        "Explain self-attention in simple terms"
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    demo.launch()
