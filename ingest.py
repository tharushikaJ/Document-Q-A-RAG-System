"""
Ingestion pipeline: load documents -> chunk -> embed -> persist to disk.

Run this whenever you add/change documents in DOCS_DIR:
    python ingest.py
"""
from config import DOCS_DIR
from document_loader import load_documents_from_dir
from chunking import chunk_documents
from vector_store import VectorStore


def main():
    print(f"Loading documents from {DOCS_DIR} ...")
    docs = load_documents_from_dir(DOCS_DIR)
    if not docs:
        print(f"No supported documents found in {DOCS_DIR}. "
              f"Add .txt, .md, .pdf, or .docx files and re-run.")
        return
    print(f"  Loaded {len(docs)} document(s).")

    print("Chunking...")
    records = chunk_documents(docs)
    print(f"  Produced {len(records)} chunk(s).")

    print("Embedding + indexing (first run downloads the embedding model)...")
    store = VectorStore()
    store.add(records)
    store.save()
    print(f"Done. Index saved with {len(store)} vectors.")


if __name__ == "__main__":
    main()
