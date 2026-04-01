#!/usr/bin/env python3
"""One-shot build of support assistant RAG index."""

from support_assistant_rag import SupportAssistantRAG


def main() -> None:
    rag = SupportAssistantRAG()
    path = rag.ensure_index()
    print(f"Support assistant index ready: {path}")


if __name__ == "__main__":
    main()
