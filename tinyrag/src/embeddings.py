"""
src/embeddings.py
------------------
Talks to your local Ollama server using plain HTTP requests.

No langchain-community, no ollama python SDK abstraction layer -
this is exactly what those libraries are doing internally: a POST
request to a JSON endpoint. Seeing it raw makes it obvious there's
no magic involved.

Requires Ollama running locally (default port 11434) with the
models you already have pulled:
    ollama pull llama3.2
    ollama pull nomic-embed-text
"""

import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list[float]:
    """
    Returns the embedding vector for `text` as a plain list of floats.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": model, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def generate_answer(prompt: str, model: str = "llama3.2") -> str:
    """
    Sends `prompt` to the generation model and returns the full text
    response. stream=False means we wait for the complete answer
    instead of handling token-by-token chunks.
    """
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["response"]
