"""
src/chunking.py
----------------
The simplest possible chunker: fixed-size character windows with overlap.

Why overlap? If a sentence straddles the boundary between chunk N and
chunk N+1, a hard cut would split it and neither chunk would contain
the full thought. Overlap re-includes the tail of the previous chunk
at the start of the next one, so boundary information isn't lost.

This is intentionally naive. Real systems often chunk by sentence or
paragraph boundaries, or even by semantic similarity between sentences.
That's a great follow-up project once this version makes sense to you.
"""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Split `text` into overlapping chunks.

    chunk_size : max characters per chunk
    overlap    : characters repeated between consecutive chunks
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # step forward by (chunk_size - overlap) so the next chunk
        # re-includes the last `overlap` characters of this one
        start += chunk_size - overlap

    return chunks
