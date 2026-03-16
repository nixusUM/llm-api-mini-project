"""Main indexing pipeline orchestrator."""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .chunker import Chunk, ChunkerStrategy
from .document_loader import Document, DocumentLoader
from .embedder import Embedder
from .index_storage import IndexEntry, IndexStorage


@dataclass
class ChunkMetadata:
    """Extended metadata for a chunk."""

    chunk_id: str
    source: str
    title: str
    section: str
    strategy: str
    token_count: int = 0
    char_count: int = 0
    word_count: int = 0
    position: int = 0
    custom: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Result of indexing pipeline."""

    chunks: List[Chunk]
    embeddings: List[List[float]]
    entries: List[IndexEntry]
    stats: Dict[str, Any]


class IndexerPipeline:
    """Orchestrates document indexing pipeline."""

    def __init__(
        self,
        chunker: ChunkerStrategy,
        embedder: Embedder,
        storage: IndexStorage,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.storage = storage
        self.loader = DocumentLoader()

    def index_document(self, document: Document) -> PipelineResult:
        """Index single document."""
        chunks = self.chunker.chunk(
            text=document.text,
            source=document.source,
            title=document.title,
        )

        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedder.embed(texts)

        entries = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            metadata = self._build_metadata(chunk, embedding, i)

            entry = IndexEntry(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                embedding=embedding,
                source=chunk.source,
                title=chunk.title,
                section=chunk.section,
                strategy=chunk.strategy,
                metadata=metadata,
            )
            entries.append(entry)

        self.storage.add_entries(entries)

        stats = self._calculate_stats(chunks, embeddings)

        return PipelineResult(
            chunks=chunks,
            embeddings=embeddings,
            entries=entries,
            stats=stats,
        )

    def index_documents(self, documents: List[Document]) -> List[PipelineResult]:
        """Index multiple documents."""
        return [self.index_document(doc) for doc in documents]

    def index_file(self, path: str) -> Optional[PipelineResult]:
        """Index single file."""
        doc = self.loader.load_file(path)
        if doc:
            return self.index_document(doc)
        return None

    def index_directory(self, dir_path: str) -> List[PipelineResult]:
        """Index all documents in directory."""
        documents = self.loader.load_directory(dir_path)
        return self.index_documents(documents)

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """Search index with query text."""
        query_embedding = self.embedder.embed_single(query)
        return self.storage.search(query_embedding, top_k)

    def save(self, path: str) -> None:
        """Save index to disk."""
        self.storage.save(path)

    def load(self, path: str) -> None:
        """Load index from disk."""
        self.storage.load(path)

    def get_stats(self) -> Dict[str, Any]:
        """Get combined pipeline stats."""
        storage_stats = self.storage.get_stats()
        return {
            "chunker": self.chunker.name,
            "embedder": {
                "model": self.embedder.model,
                "dimensions": self.embedder.dimensions,
            },
            "storage": storage_stats,
        }

    def compare_strategies(
        self, document: Document, alternative_chunker: ChunkerStrategy
    ) -> Dict[str, Any]:
        """Compare two chunking strategies on same document."""
        original_chunks = self.chunker.chunk(
            text=document.text,
            source=document.source,
            title=document.title,
        )

        alternative_chunks = alternative_chunker.chunk(
            text=document.text,
            source=document.source,
            title=document.title,
        )

        original_texts = [c.text for c in original_chunks]
        alternative_texts = [c.text for c in alternative_chunks]

        original_embeddings = self.embedder.embed(original_texts)
        alternative_embeddings = self.embedder.embed(alternative_texts)

        def calc_stats(chunks, embeddings):
            sizes = [len(c.text) for c in chunks]
            return {
                "chunk_count": len(chunks),
                "avg_chunk_size": sum(sizes) // len(sizes) if sizes else 0,
                "min_chunk_size": min(sizes) if sizes else 0,
                "max_chunk_size": max(sizes) if sizes else 0,
                "total_chars": sum(sizes),
                "sections": len(set(c.section for c in chunks)),
                "embedding_dims": len(embeddings[0]) if embeddings else 0,
            }

        return {
            "original": {
                "strategy": self.chunker.name,
                "stats": calc_stats(original_chunks, original_embeddings),
                "sample_chunks": [
                    {"id": c.chunk_id, "section": c.section, "size": len(c.text)}
                    for c in original_chunks[:3]
                ],
            },
            "alternative": {
                "strategy": alternative_chunker.name,
                "stats": calc_stats(alternative_chunks, alternative_embeddings),
                "sample_chunks": [
                    {"id": c.chunk_id, "section": c.section, "size": len(c.text)}
                    for c in alternative_chunks[:3]
                ],
            },
            "comparison": {
                "chunk_count_diff": len(original_chunks) - len(alternative_chunks),
                "avg_size_diff": (
                    calc_stats(original_chunks, original_embeddings)["avg_chunk_size"]
                    - calc_stats(alternative_chunks, alternative_embeddings)[
                        "avg_chunk_size"
                    ]
                ),
                "coverage_ratio": len(original_chunks) / max(len(alternative_chunks), 1),
            },
        }

    def _build_metadata(
        self, chunk: Chunk, embedding: List[float], position: int
    ) -> Dict[str, Any]:
        """Build metadata for chunk."""
        words = len(chunk.text.split())
        chars = len(chunk.text)

        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "title": chunk.title,
            "section": chunk.section,
            "strategy": chunk.strategy,
            "token_estimate": words * 1.3,
            "char_count": chars,
            "word_count": words,
            "position": position,
            "start_pos": chunk.start_pos,
            "end_pos": chunk.end_pos,
            "embedding_sample": embedding[:5] if embedding else [],
        }

    def _calculate_stats(
        self, chunks: List[Chunk], embeddings: List[List[float]]
    ) -> Dict[str, Any]:
        """Calculate pipeline stats."""
        if not chunks:
            return {}

        sizes = [len(c.text) for c in chunks]
        total_chars = sum(sizes)

        return {
            "chunks_created": len(chunks),
            "embeddings_created": len(embeddings),
            "total_chars_indexed": total_chars,
            "avg_chunk_size": total_chars // len(chunks),
            "min_chunk_size": min(sizes),
            "max_chunk_size": max(sizes),
            "strategy_used": chunks[0].strategy if chunks else "unknown",
        }


class MultiStrategyIndexer:
    """Run multiple indexing strategies and compare."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.results: Dict[str, PipelineResult] = {}
        self.storages: Dict[str, IndexStorage] = {}

    def index_with_strategy(
        self,
        name: str,
        chunker: ChunkerStrategy,
        storage: IndexStorage,
        documents: List[Document],
    ) -> PipelineResult:
        """Index with specific strategy."""
        pipeline = IndexerPipeline(
            chunker=chunker,
            embedder=self.embedder,
            storage=storage,
        )

        all_results = pipeline.index_documents(documents)

        total_chunks = sum(len(r.chunks) for r in all_results)
        total_embeddings = sum(len(r.embeddings) for r in all_results)

        combined = PipelineResult(
            chunks=[c for r in all_results for c in r.chunks],
            embeddings=[e for r in all_results for e in r.embeddings],
            entries=[e for r in all_results for e in r.entries],
            stats={
                "documents_indexed": len(all_results),
                "total_chunks": total_chunks,
                "total_embeddings": total_embeddings,
                "strategy": name,
            },
        )

        self.results[name] = combined
        self.storages[name] = storage

        return combined

    def compare_all(self) -> Dict[str, Any]:
        """Compare all indexed strategies."""
        comparison = {
            "strategies": list(self.results.keys()),
            "details": {},
            "ranking": [],
        }

        for name, result in self.results.items():
            comparison["details"][name] = {
                "chunk_count": len(result.chunks),
                "embedding_count": len(result.embeddings),
                "avg_chunk_size": result.stats.get("avg_chunk_size", 0),
                "total_chars": result.stats.get("total_chars_indexed", 0),
            }

        sorted_by_chunks = sorted(
            comparison["details"].items(),
            key=lambda x: x[1]["chunk_count"],
        )
        comparison["ranking"] = [
            {"strategy": name, "chunks": data["chunk_count"]}
            for name, data in sorted_by_chunks
        ]

        return comparison

    def search_all(self, query: str, top_k: int = 5) -> Dict[str, List[tuple]]:
        """Search across all strategies."""
        query_embedding = self.embedder.embed_single(query)
        results = {}

        for name, storage in self.storages.items():
            results[name] = storage.search(query_embedding, top_k)

        return results
