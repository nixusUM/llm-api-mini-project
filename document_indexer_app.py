#!/usr/bin/env python3
"""Web UI для тестирования пайплайна индексации документов."""

import os
import time
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

# Import our document indexer module
from document_indexer import (
    FixedSizeChunker,
    StructureBasedChunker,
    Embedder,
    FAISSStorage,
    SQLiteStorage,
    JSONStorage,
    DocumentLoader,
    IndexerPipeline,
)

load_dotenv()

app = Flask(__name__)

# Configuration
PROJECT_ROOT = Path(__file__).parent
SAMPLE_DOCS_DIR = PROJECT_ROOT / "sample_documents"
INDEX_DIR = PROJECT_ROOT / "document_indices"
INDEX_DIR.mkdir(exist_ok=True)

# Global state
indexing_state = {
    "pipelines": {},
    "last_result": None,
    "comparison_result": None,
}


def _collect_corpus_paths() -> list[Path]:
    """Build assignment-ready corpus: README + articles + code + pdf."""
    paths: list[Path] = []

    readme_path = PROJECT_ROOT / "README.md"
    if readme_path.exists():
        paths.append(readme_path)

    if SAMPLE_DOCS_DIR.exists():
        for ext in ("*.md", "*.txt", "*.pdf"):
            paths.extend(sorted(SAMPLE_DOCS_DIR.glob(ext)))

    for py_file in sorted(PROJECT_ROOT.glob("*.py")):
        paths.append(py_file)

    indexer_dir = PROJECT_ROOT / "document_indexer"
    if indexer_dir.exists():
        paths.extend(sorted(indexer_dir.glob("*.py")))

    # De-duplicate while preserving order
    unique: list[Path] = []
    seen = set()
    for item in paths:
        key = str(item.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _relative_path(path: Path) -> str:
    """Path displayed in UI and used by API."""
    return str(path.relative_to(PROJECT_ROOT))


def _resolve_selected_path(relative_path: str) -> Path | None:
    """Resolve selected document path safely inside project root."""
    try:
        resolved = (PROJECT_ROOT / relative_path).resolve()
    except Exception:
        return None
    if not str(resolved).startswith(str(PROJECT_ROOT.resolve())):
        return None
    if not resolved.exists() or not resolved.is_file():
        return None
    return resolved


def get_or_create_pipeline(strategy: str, storage_type: str = "faiss"):
    """Get or create indexing pipeline with specified strategy."""
    cache_key = f"{strategy}_{storage_type}"

    if cache_key in indexing_state["pipelines"]:
        return indexing_state["pipelines"][cache_key]

    # Create chunker based on strategy
    if strategy == "fixed_size":
        chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    elif strategy == "structure_based":
        chunker = StructureBasedChunker(max_chunk_size=1000)
    else:
        chunker = FixedSizeChunker(chunk_size=500)

    # Create embedder
    embedder = Embedder(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )

    # Create storage
    if storage_type == "faiss":
        storage = FAISSStorage(dimension=embedder.dimensions)
    elif storage_type == "sqlite":
        db_path = INDEX_DIR / f"index_{strategy}.db"
        storage = SQLiteStorage(
            db_path=str(db_path), dimension=embedder.dimensions
        )
    elif storage_type == "json":
        storage = JSONStorage(dimension=embedder.dimensions)
    else:
        storage = FAISSStorage(dimension=embedder.dimensions)

    pipeline = IndexerPipeline(
        chunker=chunker,
        embedder=embedder,
        storage=storage,
    )

    indexing_state["pipelines"][cache_key] = pipeline
    return pipeline


@app.route("/document_indexer")
def document_indexer():
    """Main page for document indexer."""
    docs = [
        {"name": _relative_path(path), "size": path.stat().st_size}
        for path in _collect_corpus_paths()
    ]

    # Get pipeline stats
    stats = {}
    for key, pipeline in indexing_state["pipelines"].items():
        try:
            stats[key] = pipeline.get_stats()
        except Exception:
            pass

    return render_template(
        "document_indexer.html",
        documents=docs,
        pipelines=list(indexing_state["pipelines"].keys()),
        stats=stats,
        last_result=indexing_state.get("last_result"),
        comparison=indexing_state.get("comparison_result"),
    )


@app.route("/")
def home():
    """Default route for standalone app."""
    return document_indexer()


@app.route("/api/documents")
def api_documents():
    """List available documents."""
    loader = DocumentLoader()
    documents = []
    for path in _collect_corpus_paths():
        doc = loader.load_file(str(path))
        if doc:
            documents.append(doc)
    stats = loader.get_stats(documents)

    return jsonify(
        {
            "documents": [
                {"source": d.source, "title": d.title, "type": d.doc_type}
                for d in documents
            ],
            "stats": stats,
        }
    )


@app.route("/api/index", methods=["POST"])
def api_index():
    """Index documents with selected strategy."""
    data = request.json
    strategy = data.get("strategy", "fixed_size")
    storage_type = data.get("storage", "faiss")
    documents = data.get("documents", [])

    start_time = time.time()

    pipeline = get_or_create_pipeline(strategy, storage_type)

    loader = DocumentLoader()
    docs_to_index = []
    if documents:
        for doc_path in documents:
            full_path = _resolve_selected_path(doc_path)
            if not full_path:
                continue
            doc = loader.load_file(str(full_path))
            if doc:
                docs_to_index.append(doc)
    else:
        for path in _collect_corpus_paths():
            doc = loader.load_file(str(path))
            if doc:
                docs_to_index.append(doc)

    # Index documents
    all_results = pipeline.index_documents(docs_to_index)

    # Calculate total stats
    total_chunks = sum(len(r.chunks) for r in all_results)
    total_embeddings = sum(len(r.embeddings) for r in all_results)
    total_chars = sum(r.stats.get("total_chars_indexed", 0) for r in all_results)

    elapsed = time.time() - start_time

    result = {
        "success": True,
        "strategy": strategy,
        "storage": storage_type,
        "documents_indexed": len(docs_to_index),
        "chunks_created": total_chunks,
        "embeddings_generated": total_embeddings,
        "total_chars": total_chars,
        "elapsed_seconds": round(elapsed, 2),
        "chunks_per_second": round(total_chunks / elapsed, 2) if elapsed > 0 else 0,
        "pipeline_stats": pipeline.get_stats(),
    }

    indexing_state["last_result"] = result
    return jsonify(result)


@app.route("/api/compare", methods=["POST"])
def api_compare():
    """Compare two chunking strategies."""
    data = request.json
    doc_source = data.get("document")

    if not doc_source:
        return jsonify({"error": "Document not specified"}), 400

    # Load document
    loader = DocumentLoader()
    doc_path = _resolve_selected_path(doc_source)
    if not doc_path:
        return jsonify({"error": "Document not found"}), 404

    document = loader.load_file(str(doc_path))
    if not document:
        return jsonify({"error": "Could not load document"}), 400

    embedder = Embedder()

    # Create chunkers
    fixed_chunker = FixedSizeChunker(chunk_size=500, overlap=50)
    structure_chunker = StructureBasedChunker(max_chunk_size=1000)

    # Index with both strategies
    fixed_pipeline = IndexerPipeline(
        chunker=fixed_chunker,
        embedder=embedder,
        storage=FAISSStorage(),
    )

    structure_pipeline = IndexerPipeline(
        chunker=structure_chunker,
        embedder=embedder,
        storage=FAISSStorage(),
    )

    fixed_result = fixed_pipeline.index_document(document)
    structure_result = structure_pipeline.index_document(document)

    # Compare
    comparison = fixed_pipeline.compare_strategies(
        document, structure_chunker
    )

    # Format for display
    fixed_chunks = [
        {"id": c.chunk_id, "section": c.section, "size": len(c.text), "text": c.text[:200] + "..."}
        for c in fixed_result.chunks[:5]
    ]

    structure_chunks = [
        {"id": c.chunk_id, "section": c.section, "size": len(c.text), "text": c.text[:200] + "..."}
        for c in structure_result.chunks[:5]
    ]

    result = {
        "document": document.source,
        "title": document.title,
        "total_chars": len(document.text),
        "fixed_size": {
            "chunks": len(fixed_result.chunks),
            "avg_chunk_size": comparison["original"]["stats"]["avg_chunk_size"],
            "min_size": comparison["original"]["stats"]["min_chunk_size"],
            "max_size": comparison["original"]["stats"]["max_chunk_size"],
            "sample_chunks": fixed_chunks,
        },
        "structure_based": {
            "chunks": len(structure_result.chunks),
            "avg_chunk_size": comparison["alternative"]["stats"]["avg_chunk_size"],
            "min_size": comparison["alternative"]["stats"]["min_chunk_size"],
            "max_size": comparison["alternative"]["stats"]["max_chunk_size"],
            "sample_chunks": structure_chunks,
        },
        "comparison": comparison["comparison"],
    }

    indexing_state["comparison_result"] = result
    return jsonify(result)


@app.route("/api/search", methods=["POST"])
def api_search():
    """Search indexed documents."""
    data = request.json
    query = data.get("query", "")
    strategy = data.get("strategy", "fixed_size")
    storage_type = data.get("storage", "faiss")
    top_k = data.get("top_k", 5)

    if not query:
        return jsonify({"error": "Query is empty"}), 400

    pipeline = get_or_create_pipeline(strategy, storage_type)

    start_time = time.time()
    results = pipeline.search(query, top_k=top_k)
    elapsed = time.time() - start_time

    formatted_results = [
        {
            "chunk_id": entry.chunk_id,
            "text": entry.text[:300] + "..." if len(entry.text) > 300 else entry.text,
            "source": entry.source,
            "title": entry.title,
            "section": entry.section,
            "score": round(float(score), 4),
        }
        for entry, score in results
    ]

    return jsonify(
        {
            "query": query,
            "strategy": strategy,
            "results": formatted_results,
            "search_time_ms": round(elapsed * 1000, 2),
            "result_count": len(formatted_results),
        }
    )


@app.route("/api/save_index", methods=["POST"])
def api_save_index():
    """Save index to disk."""
    data = request.json
    strategy = data.get("strategy", "fixed_size")
    storage_type = data.get("storage", "json")

    pipeline = get_or_create_pipeline(strategy, storage_type)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = "json"
    if storage_type == "sqlite":
        extension = "db"
    filename = f"index_{strategy}_{storage_type}_{timestamp}.{extension}"
    filepath = INDEX_DIR / filename

    pipeline.save(str(filepath))

    return jsonify(
        {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "stats": pipeline.get_stats(),
        }
    )


@app.route("/api/stats")
def api_stats():
    """Get indexing statistics."""
    all_stats = {}
    for key, pipeline in indexing_state["pipelines"].items():
        try:
            all_stats[key] = pipeline.get_stats()
        except Exception as e:
            all_stats[key] = {"error": str(e)}

    return jsonify(
        {
            "pipelines": all_stats,
            "available_indices": [
                f.name for f in INDEX_DIR.glob("*.json") if f.is_file()
            ],
        }
    )


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all indices."""
    indexing_state["pipelines"].clear()
    indexing_state["last_result"] = None
    indexing_state["comparison_result"] = None

    return jsonify({"success": True, "message": "All indices cleared"})


if __name__ == "__main__":
    port = int(os.getenv("INDEXER_PORT", "5052"))
    app.run(debug=True, host="127.0.0.1", port=port)
