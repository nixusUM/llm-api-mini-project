"""Storage backends for document indices."""

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IndexEntry:
    """Single index entry with embedding and metadata."""

    chunk_id: str
    text: str
    embedding: List[float]
    source: str
    title: str
    section: str
    strategy: str
    metadata: Dict[str, Any]


class IndexStorage(ABC):
    """Abstract base for index storage."""

    @abstractmethod
    def add_entries(self, entries: List[IndexEntry]) -> None:
        """Add entries to index."""
        pass

    @abstractmethod
    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[IndexEntry, float]]:
        """Search index by embedding similarity."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save index to disk."""
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load index from disk."""
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all entries."""
        pass


class FAISSStorage(IndexStorage):
    """FAISS-based vector storage."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.entries: List[IndexEntry] = []
        self._index = None
        self._faiss_available = False
        self._try_import_faiss()

    def _try_import_faiss(self):
        """Try to import FAISS."""
        try:
            import faiss

            self._faiss = faiss
            self._faiss_available = True
        except ImportError:
            self._faiss_available = False

    def _build_index(self):
        """Build FAISS index."""
        if not self._faiss_available or not self.entries:
            return

        embeddings = np.array([e.embedding for e in self.entries], dtype=np.float32)

        if self._index is None:
            self._index = self._faiss.IndexFlatIP(self.dimension)

        self._index.reset()
        self._index.add(embeddings)

    def add_entries(self, entries: List[IndexEntry]) -> None:
        """Add entries to index."""
        self.entries.extend(entries)
        self._build_index()

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[IndexEntry, float]]:
        """Search by cosine similarity."""
        if not self.entries:
            return []

        query = np.array([query_embedding], dtype=np.float32)

        if self._faiss_available and self._index is not None:
            scores, indices = self._index.search(query, min(top_k, len(self.entries)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.entries):
                    results.append((self.entries[idx], float(score)))
            return results

        return self._brute_force_search(query_embedding, top_k)

    def _brute_force_search(
        self, query_embedding: List[float], top_k: int
    ) -> List[Tuple[IndexEntry, float]]:
        """Fallback brute force search."""
        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []

        query_unit = np.array(query_embedding) / query_norm

        scores = []
        for entry in self.entries:
            emb_norm = np.linalg.norm(entry.embedding)
            if emb_norm == 0:
                continue
            emb_unit = np.array(entry.embedding) / emb_norm
            similarity = float(np.dot(query_unit, emb_unit))
            scores.append((entry, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save to JSON (FAISS index not portable)."""
        data = {
            "dimension": self.dimension,
            "entries": [
                {
                    "chunk_id": e.chunk_id,
                    "text": e.text,
                    "embedding": e.embedding,
                    "source": e.source,
                    "title": e.title,
                    "section": e.section,
                    "strategy": e.strategy,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self.dimension)
        self.entries = [
            IndexEntry(
                chunk_id=e["chunk_id"],
                text=e["text"],
                embedding=e["embedding"],
                source=e["source"],
                title=e["title"],
                section=e["section"],
                strategy=e["strategy"],
                metadata=e.get("metadata", {}),
            )
            for e in data.get("entries", [])
        ]
        self._build_index()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "type": "FAISS",
            "faiss_available": self._faiss_available,
            "entry_count": len(self.entries),
            "dimension": self.dimension,
            "strategies": list(set(e.strategy for e in self.entries)),
            "sources": list(set(e.source for e in self.entries)),
        }

    def clear(self) -> None:
        """Clear all."""
        self.entries = []
        self._index = None


class SQLiteStorage(IndexStorage):
    """SQLite-based storage with vector extension simulation."""

    def __init__(self, db_path: Optional[str] = None, dimension: int = 1536):
        self.db_path = db_path or ":memory:"
        self.dimension = dimension
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self):
        """Initialize database."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS index_entries (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                source TEXT NOT NULL,
                title TEXT NOT NULL,
                section TEXT NOT NULL,
                strategy TEXT NOT NULL,
                metadata TEXT
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_source ON index_entries(source)"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_strategy ON index_entries(strategy)"
        )
        self._conn.commit()

    def add_entries(self, entries: List[IndexEntry]) -> None:
        """Add entries."""
        for entry in entries:
            embedding_blob = json.dumps(entry.embedding).encode()
            metadata_json = json.dumps(entry.metadata, ensure_ascii=False)

            self._conn.execute(
                """
                INSERT OR REPLACE INTO index_entries
                (chunk_id, text, embedding, source, title, section, strategy, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    entry.chunk_id,
                    entry.text,
                    embedding_blob,
                    entry.source,
                    entry.title,
                    entry.section,
                    entry.strategy,
                    metadata_json,
                ),
            )
        self._conn.commit()

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[IndexEntry, float]]:
        """Brute force search in SQLite."""
        cursor = self._conn.execute("SELECT * FROM index_entries")
        rows = cursor.fetchall()

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_unit = np.array(query_embedding) / query_norm

        scores = []
        for row in rows:
            entry = self._row_to_entry(row)
            emb_norm = np.linalg.norm(entry.embedding)
            if emb_norm == 0:
                continue
            emb_unit = np.array(entry.embedding) / emb_norm
            similarity = float(np.dot(query_unit, emb_unit))
            scores.append((entry, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _row_to_entry(self, row) -> IndexEntry:
        """Convert DB row to IndexEntry."""
        embedding = json.loads(row[2].decode())
        metadata = json.loads(row[7]) if row[7] else {}
        return IndexEntry(
            chunk_id=row[0],
            text=row[1],
            embedding=embedding,
            source=row[3],
            title=row[4],
            section=row[5],
            strategy=row[6],
            metadata=metadata,
        )

    def save(self, path: str) -> None:
        """Backup to file if in-memory."""
        if self.db_path == ":memory:":
            backup_conn = sqlite3.connect(path)
            self._conn.backup(backup_conn)
            backup_conn.close()
        else:
            import shutil
            shutil.copy(self.db_path, path)

    def load(self, path: str) -> None:
        """Load from file."""
        if self._conn:
            self._conn.close()

        if os.path.exists(path):
            import shutil
            shutil.copy(path, self.db_path)

        self._init_db()

    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM index_entries")
        count = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT DISTINCT strategy FROM index_entries"
        )
        strategies = [row[0] for row in cursor.fetchall()]

        cursor = self._conn.execute(
            "SELECT DISTINCT source FROM index_entries"
        )
        sources = [row[0] for row in cursor.fetchall()]

        return {
            "type": "SQLite",
            "entry_count": count,
            "dimension": self.dimension,
            "strategies": strategies,
            "sources": sources,
        }

    def clear(self) -> None:
        """Clear all."""
        self._conn.execute("DELETE FROM index_entries")
        self._conn.commit()

    def close(self):
        """Close connection."""
        if self._conn:
            self._conn.close()
            self._conn = None


class JSONStorage(IndexStorage):
    """Simple JSON file storage."""

    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.entries: List[IndexEntry] = []

    def add_entries(self, entries: List[IndexEntry]) -> None:
        """Add entries."""
        existing_ids = {e.chunk_id for e in self.entries}
        for entry in entries:
            if entry.chunk_id not in existing_ids:
                self.entries.append(entry)
                existing_ids.add(entry.chunk_id)

    def search(
        self, query_embedding: List[float], top_k: int = 5
    ) -> List[Tuple[IndexEntry, float]]:
        """Brute force search."""
        if not self.entries:
            return []

        query_norm = np.linalg.norm(query_embedding)
        if query_norm == 0:
            return []
        query_unit = np.array(query_embedding) / query_norm

        scores = []
        for entry in self.entries:
            emb_norm = np.linalg.norm(entry.embedding)
            if emb_norm == 0:
                continue
            emb_unit = np.array(entry.embedding) / emb_norm
            similarity = float(np.dot(query_unit, emb_unit))
            scores.append((entry, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str) -> None:
        """Save to JSON."""
        data = {
            "dimension": self.dimension,
            "entries": [
                {
                    "chunk_id": e.chunk_id,
                    "text": e.text,
                    "embedding": e.embedding,
                    "source": e.source,
                    "title": e.title,
                    "section": e.section,
                    "strategy": e.strategy,
                    "metadata": e.metadata,
                }
                for e in self.entries
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        """Load from JSON."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.dimension = data.get("dimension", self.dimension)
        self.entries = [
            IndexEntry(
                chunk_id=e["chunk_id"],
                text=e["text"],
                embedding=e["embedding"],
                source=e["source"],
                title=e["title"],
                section=e["section"],
                strategy=e["strategy"],
                metadata=e.get("metadata", {}),
            )
            for e in data.get("entries", [])
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get stats."""
        return {
            "type": "JSON",
            "entry_count": len(self.entries),
            "dimension": self.dimension,
            "strategies": list(set(e.strategy for e in self.entries)),
            "sources": list(set(e.source for e in self.entries)),
        }

    def clear(self) -> None:
        """Clear all."""
        self.entries = []
