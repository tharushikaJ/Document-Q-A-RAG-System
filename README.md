# A RAG System, Built From Scratch

This is a complete, working Retrieval-Augmented Generation (RAG) pipeline you
can run locally and extend. It answers questions using **your own documents**
instead of relying purely on a model's training data.

## How RAG works (the short version)

1. **Ingest**: split your documents into small overlapping chunks.
2. **Embed**: turn each chunk into a vector (a list of numbers capturing its meaning).
3. **Index**: store those vectors so you can search them by similarity.
4. **Retrieve**: at question time, embed the question and find the most
   similar chunks.
5. **Generate**: hand those chunks to an LLM as context and ask it to answer
   using only that context.

The value of RAG over just prompting an LLM directly: the model answers from
your specific, current, private data instead of guessing from what it was
trained on — and you get citations back to the source.

## Architecture of this project

```
rag_system/
├── config.py           # all tunable settings in one place
├── document_loader.py  # reads .txt/.md/.pdf/.docx into plain text
├── chunking.py          # splits text into overlapping token-sized chunks
├── vector_store.py      # embeddings + FAISS index + persistence
├── ingest.py             # run this to build/rebuild the index
├── rag_pipeline.py      # retrieval + Claude generation, the core logic
├── cli.py               # interactive terminal chat over your docs
├── example_docs/        # put your source files here
└── index/               # generated: FAISS index + metadata (gitignore this)
```

**Design choices, and why:**
- **Embeddings run locally** (`sentence-transformers`, model `all-MiniLM-L6-v2`)
  — free, fast, no API key needed for this step, good baseline quality.
- **FAISS** for the vector index — the standard lightweight choice; swap in
  Pinecone/Weaviate/Chroma/pgvector later without changing anything except
  `vector_store.py`.
- **Claude does generation only** — retrieval is separate from generation so
  you can inspect/debug/swap either half independently.
- **Chunks carry metadata** (source filename, chunk id, similarity score) so
  answers can cite where information came from.

## Setup

```bash
cd rag_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your_key_here   # get one at console.anthropic.com
```

## Usage

1. Drop your documents (`.txt`, `.md`, `.pdf`, `.docx`) into `example_docs/`.
   A sample handbook doc is already there so you can test immediately.

2. Build the index:
   ```bash
   python ingest.py
   ```
   Re-run this any time you add, remove, or change documents.

3. Ask questions:
   ```bash
   python cli.py
   ```
   Try, against the included sample doc: *"How many vacation days do
   employees get?"* or *"What's the remote work equipment stipend?"*

4. Or use it programmatically:
   ```python
   from rag_pipeline import RAGPipeline

   pipeline = RAGPipeline()
   result = pipeline.ask("What is the internet reimbursement limit?")
   print(result["answer"])
   print([c["source"] for c in result["chunks"]])
   ```

## Tuning

All the knobs live in `config.py`:

| Setting | What it controls | Notes |
|---|---|---|
| `CHUNK_SIZE` | tokens per chunk | smaller = more precise retrieval, more chunks; larger = more context per chunk, less precise |
| `CHUNK_OVERLAP` | token overlap between chunks | prevents facts near a chunk boundary from being lost |
| `TOP_K` | chunks retrieved per query | more = more context but more noise/cost |
| `MIN_SCORE_THRESHOLD` | similarity floor | raise this to suppress weak/irrelevant matches |
| `EMBEDDING_MODEL_NAME` | which sentence-transformers model | bigger models = better quality, slower |
| `ANTHROPIC_MODEL` | generation model | swap models via the model string |

## Common next steps (not included, but straightforward extensions)

- **Better retrieval**: add a re-ranking step (e.g. cross-encoder) after the
  initial FAISS search to reorder the top ~20 candidates before picking the
  final `TOP_K`.
- **Hybrid search**: combine vector similarity with keyword (BM25) search —
  helps a lot with exact terms, IDs, and names that embeddings can miss.
- **Chunk strategy**: for structured docs (tables, code), chunk by logical
  section instead of raw token count.
- **Evaluation**: build a small set of Q&A pairs with known correct answers
  and track retrieval accuracy as you tune chunking/embedding choices.
- **Streaming responses**: swap `messages.create` for `messages.stream` in
  `rag_pipeline.py` for token-by-token output in the CLI.
- **Persistent chat**: add conversation history to `rag_pipeline.py` so
  follow-up questions retain context.

## Troubleshooting

- `FileNotFoundError: No saved index found` → run `python ingest.py` first.
- `ANTHROPIC_API_KEY is not set` → export it in your shell before running `cli.py`.
- Empty/odd answers → check `verbose=True` output (already on in the CLI) to
  see what chunks were actually retrieved; if they're irrelevant, revisit
  `CHUNK_SIZE`/`TOP_K` or add more source documents.
