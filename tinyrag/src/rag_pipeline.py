"""
src/rag_pipeline.py
--------------------
Wires the pieces together: chunk -> embed -> store -> retrieve -> generate.

This is the entire RAG "algorithm" in about 30 lines. Everything
else in this project (chunking, embeddings, vector_store) is
replaceable plumbing; this is the part that actually defines the
technique.
"""

from src.chunking import chunk_text
from src.embeddings import get_embedding, generate_answer
from src.vector_store import TinyVectorStore

PROMPT_TEMPLATE = """You are a helpful assistant. Use ONLY the context below to answer \
the question. If the answer isn't contained in the context, say you don't know - \
do not make anything up.

Context:
{context}

Question: {question}

Answer:"""


class TinyRAG:
    def __init__(self):
        self.store = TinyVectorStore()

    def ingest(self, file_path: str, chunk_size: int = 500, overlap: int = 50) -> None:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        self.ingest_text(
            text,
            source_label=file_path,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def ingest_text(
        self,
        text: str,
        source_label: str = "uploaded_text",
        chunk_size: int = 500,
        overlap: int = 50,
    ) -> None:
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        print(f"Split document into {len(chunks)} chunks. Embedding each one...")

        for i, chunk in enumerate(chunks):
            vector = get_embedding(chunk)
            self.store.add(chunk, vector, metadata={"chunk_id": i, "source": source_label})

        print(f"Done. {len(self.store)} chunks embedded and stored in memory.\n")

    def ask(self, question: str, top_k: int = 3, show_chunks: bool = True) -> str:
        answer, _ = self.query(question, top_k=top_k, show_chunks=show_chunks)
        return answer

    def query(self, question: str, top_k: int = 3, show_chunks: bool = True) -> tuple[str, list[tuple[float, dict]]]:
        query_vector = get_embedding(question)
        results = self.store.search(query_vector, top_k=top_k)

        if show_chunks:
            print("Top retrieved chunks:")
            for score, record in results:
                preview = record["text"][:80].replace("\n", " ")
                print(f"  [{score:.3f}] {preview}...")
            print()

        context = "\n\n---\n\n".join(record["text"] for _, record in results)
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)

        return generate_answer(prompt), results
