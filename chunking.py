"""
Splits document text into overlapping chunks for embedding.

Why token-aware chunking: character-based splitting can cut words/sentences
awkwardly and doesn't map predictably to embedding model limits. We use
tiktoken as a reasonably universal tokenizer proxy (it won't exactly match
the embedding model's tokenizer, but it's consistent and good enough for
sizing chunks sensibly).

Why overlap: without it, a fact split across a chunk boundary becomes
unretrievable because neither chunk contains the full context.
"""
import tiktoken
from config import CHUNK_SIZE, CHUNK_OVERLAP

_encoder = tiktoken.get_encoding("cl100k_base")


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of ~chunk_size tokens.
    Tries to break on paragraph/sentence boundaries near the target size
    so chunks read naturally rather than stopping mid-sentence.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _encoder.encode(text)
    if len(tokens) <= chunk_size:
        return [text.strip()] if text.strip() else []

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_str = _encoder.decode(chunk_tokens)

        # Try to end on a clean sentence/paragraph boundary if one exists
        # reasonably close to the end of the chunk.
        if end < len(tokens):
            for boundary in ["\n\n", ". ", "\n"]:
                idx = chunk_str.rfind(boundary)
                if idx > len(chunk_str) * 0.5:  # only trim if we don't lose too much
                    chunk_str = chunk_str[: idx + len(boundary)]
                    break

        chunk_str = chunk_str.strip()
        if chunk_str:
            chunks.append(chunk_str)

        # Advance start by (this chunk's real token length - overlap)
        consumed = len(_encoder.encode(chunk_str)) if chunk_str else chunk_size
        start += max(consumed - overlap, 1)

    return chunks


def chunk_documents(docs: list[dict]) -> list[dict]:
    """
    Chunk a list of {"source", "text"} docs into
    {"source", "chunk_id", "text"} records ready for embedding.
    """
    records = []
    for doc in docs:
        pieces = chunk_text(doc["text"])
        for i, piece in enumerate(pieces):
            records.append({
                "source": doc["source"],
                "chunk_id": i,
                "text": piece,
            })
    return records
