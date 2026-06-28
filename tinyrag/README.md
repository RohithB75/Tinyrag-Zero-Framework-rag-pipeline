# TinyRAG — A Zero-Framework RAG Pipeline

A minimal Retrieval-Augmented Generation system built with nothing but
`numpy` and `requests`. No LangChain, no ChromaDB, no vector database -
every step of the pipeline is written out so you can see exactly what
those libraries are doing for you under the hood.

## Why build this if you already have a ChromaDB version?

The Chroma + Ollama project answers "can you build a working RAG app."
This one answers the question interviewers actually ask next:
"okay, but how does retrieval actually work?" After this, you'll be
able to explain cosine similarity, chunking tradeoffs, and what a
vector store is actually doing, instead of saying "Chroma handles it."

## Project structure

```
tinyrag/
├── README.md
├── requirements.txt
├── main.py                 # CLI entry point
├── streamlit_app.py        # polished Streamlit UI
├── data/
│   └── sample_doc.txt       # test document (about RAG itself)
└── src/
    ├── __init__.py
    ├── chunking.py           # splits raw text into overlapping chunks
    ├── embeddings.py         # raw HTTP calls to local Ollama
    ├── vector_store.py       # in-memory store + hand-computed cosine similarity
    └── rag_pipeline.py       # wires everything together
```

## Setup

You need Ollama running locally with the same models you already have:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

Then, from inside the `tinyrag/` folder, install dependencies:

```bash
pip install -r requirements.txt
```

## Run it

Always run from the project root (the folder containing `main.py`),
since `src` is imported as a package:

```bash
python main.py data/sample_doc.txt
```

For the Streamlit interface, run:

```bash
streamlit run streamlit_app.py
```

The UI lets you load the bundled sample document or upload your own `.txt` or `.md` file, rebuild the index with different chunk settings, and inspect the retrieved chunks under each answer.

Try asking it: "What is cosine similarity?" or "Why does chunking
matter?" — and watch the retrieved chunks print out before the answer,
so you can see exactly what the model was given to work with.

Then try it on your own document:

```bash
python main.py data/your_notes.txt
```

## What's deliberately missing (and why that's the point)

- **No persistence** — vectors live in a Python list and vanish when
  the process ends. Real vector DBs add disk persistence; that's it.
- **No approximate search** — every query compares against every
  stored vector (brute force). Fine for thousands of chunks; FAISS/HNSW
  exist purely to make this fast at millions of vectors.
- **Naive chunking** — fixed character windows, not sentence- or
  paragraph-aware. Good enough to learn the mechanics, not optimal for
  real documents.

## Natural next steps

Once this clicks, the highest-value upgrades, in order, are:

1. Compare this fixed-size chunker against paragraph-based and
   semantic chunking on the same document — measure which retrieves
   better chunks for the same questions.
2. Add a second retrieval method (BM25 keyword search) and combine it
   with the vector search — that's hybrid retrieval.
3. Write 10-15 question/answer pairs about your test document and
   score your own retrieval — that's the start of a real eval harness.

Each of those is a project on its own, and each one builds directly on
the code here rather than replacing it.
