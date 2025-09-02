import os
from app.rag_pipeline import build_vector_store  # ✅ note: import from app/

pdf_path = "data/Attention.pdf"  # 👈 replace with your PDF filename
store_path = "artifacts/vector_store"

os.makedirs("artifacts", exist_ok=True)

build_vector_store(pdf_path, store_path)
print("✅ Vector store created at:", store_path)
