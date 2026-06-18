"""
main.py
-------
Run from the project root (the folder containing this file):

    python main.py data/sample_doc.txt
    python main.py path/to/your_own_document.txt
"""

import sys

from src.rag_pipeline import TinyRAG


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_text_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    rag = TinyRAG()
    rag.ingest(file_path)

    print("Ask questions about the document (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue

        answer = rag.ask(question)
        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
