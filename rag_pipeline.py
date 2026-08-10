"""
The core RAG loop: retrieve relevant chunks, then ask the model to answer
the question grounded in those chunks only.
"""
from groq import Groq

from config import (
    GROQ_MODEL,
    GROQ_API_KEY,
    MAX_TOKENS,
    TOP_K,
    MIN_SCORE_THRESHOLD,
)
from vector_store import VectorStore

SYSTEM_PROMPT = """You are a helpful assistant answering questions using ONLY the \
provided context excerpts. Follow these rules strictly:

1. Answer only using information found in the context below.
2. If the context does not contain enough information to answer, say so \
plainly rather than guessing or using outside knowledge.
3. When you use a fact from the context, cite the source in brackets, \
e.g. [source: filename.pdf].
4. Be concise and direct. Do not pad the answer with filler.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (source: {chunk['source']}, chunk {chunk['chunk_id']}, "
            f"relevance: {chunk['score']:.2f})\n{chunk['text']}"
        )
    return "\n\n".join(parts)


class RAGPipeline:
    def __init__(self):
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. Run:\n"
                "  export GROQ_API_KEY=your_key_here"
            )
        self.client = Groq(api_key=GROQ_API_KEY)
        self.store = VectorStore()
        self.store.load()

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        results = self.store.search(query, top_k=top_k)
        return [r for r in results if r["score"] >= MIN_SCORE_THRESHOLD]

    def generate(self, query: str, chunks: list[dict]) -> str:
        if not chunks:
            context_block = "(No relevant context was found in the knowledge base.)"
        else:
            context_block = build_context_block(chunks)

        user_message = f"Context:\n{context_block}\n\nQuestion: {query}"

        response = self.client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return response.choices[0].message.content

    def ask(self, query: str, top_k: int = TOP_K, verbose: bool = False) -> dict:
        """
        Full RAG turn: retrieve then generate.
        Returns {"answer": str, "chunks": list[dict]} so callers can
        inspect what was retrieved (useful for debugging/citations UI).
        """
        chunks = self.retrieve(query, top_k=top_k)
        if verbose:
            print(f"\n--- Retrieved {len(chunks)} chunk(s) ---")
            for c in chunks:
                preview = c["text"][:100].replace("\n", " ")
                print(f"  [{c['score']:.3f}] {c['source']}#{c['chunk_id']}: {preview}...")
            print("---\n")

        answer = self.generate(query, chunks)
        return {"answer": answer, "chunks": chunks}