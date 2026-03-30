#!/usr/bin/env python3
"""One-shot build of document_indices/index_dev_assistant.json for /help RAG."""

from dev_assistant_rag import DevAssistantRAG


def main() -> None:
    rag = DevAssistantRAG()
    path = rag.ensure_index_built()
    print(f"Dev assistant index ready: {path}")


if __name__ == "__main__":
    main()
