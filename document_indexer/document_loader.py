"""Document loading from various sources."""

import os
import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Document:
    """Loaded document."""

    text: str
    source: str
    title: str
    doc_type: str


class DocumentLoader:
    """Load documents from files and directories."""

    SUPPORTED_EXTENSIONS = {
        ".md": "markdown",
        ".txt": "text",
        ".py": "code",
        ".js": "code",
        ".ts": "code",
        ".jsx": "code",
        ".tsx": "code",
        ".java": "code",
        ".cpp": "code",
        ".c": "code",
        ".h": "code",
        ".go": "code",
        ".rs": "code",
        ".rb": "code",
        ".php": "code",
        ".swift": "code",
        ".kt": "code",
        ".sql": "code",
        ".json": "data",
        ".yaml": "data",
        ".yml": "data",
        ".xml": "data",
        ".html": "markup",
        ".css": "code",
        ".pdf": "pdf",
    }

    def load_file(self, path: str) -> Optional[Document]:
        """Load single file."""
        if not os.path.exists(path):
            return None

        ext = os.path.splitext(path)[1].lower()
        doc_type = self.SUPPORTED_EXTENSIONS.get(ext, "unknown")

        if ext == ".pdf":
            text = self._load_pdf_text(path)
            if not text:
                return None
            title = self._extract_title(text, path)
            return Document(
                text=text,
                source=os.path.basename(path),
                title=title,
                doc_type=doc_type,
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except UnicodeDecodeError:
            try:
                with open(path, "r", encoding="latin-1") as f:
                    text = f.read()
            except Exception:
                return None
        except Exception:
            return None

        title = self._extract_title(text, path)

        return Document(
            text=text,
            source=os.path.basename(path),
            title=title,
            doc_type=doc_type,
        )

    def _load_pdf_text(self, path: str) -> Optional[str]:
        """Extract text from PDF file."""
        try:
            from pypdf import PdfReader
        except ImportError:
            return None

        try:
            reader = PdfReader(path)
            pages = []
            for page in reader.pages:
                pages.append((page.extract_text() or "").strip())
            text = "\n\n".join(part for part in pages if part)
            return text.strip() or None
        except Exception:
            return None

    def load_directory(
        self, dir_path: str, recursive: bool = True
    ) -> List[Document]:
        """Load all supported files from directory."""
        documents = []

        if not os.path.isdir(dir_path):
            return documents

        for entry in os.listdir(dir_path):
            full_path = os.path.join(dir_path, entry)

            if os.path.isfile(full_path):
                ext = os.path.splitext(entry)[1].lower()
                if ext in self.SUPPORTED_EXTENSIONS or self._is_text_file(full_path):
                    doc = self.load_file(full_path)
                    if doc:
                        documents.append(doc)

            elif recursive and os.path.isdir(full_path):
                documents.extend(self.load_directory(full_path, recursive))

        return documents

    def load_multiple(self, paths: List[str]) -> List[Document]:
        """Load multiple files/directories."""
        documents = []
        for path in paths:
            if os.path.isfile(path):
                doc = self.load_file(path)
                if doc:
                    documents.append(doc)
            elif os.path.isdir(path):
                documents.extend(self.load_directory(path))
        return documents

    def _extract_title(self, text: str, path: str) -> str:
        """Extract document title."""
        header_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if header_match:
            return header_match.group(1).strip()

        header_match = re.search(
            r'^"""\s*\n?(?:#\s*)?(.+?)\n', text, re.MULTILINE
        )
        if header_match:
            return header_match.group(1).strip()

        basename = os.path.basename(path)
        name = os.path.splitext(basename)[0]
        return name.replace("_", " ").replace("-", " ").title()

    def _is_text_file(self, path: str) -> bool:
        """Check if file is likely text."""
        try:
            with open(path, "rb") as f:
                sample = f.read(1024)
                if b"\x00" in sample:
                    return False
                return True
        except Exception:
            return False

    def get_stats(self, documents: List[Document]) -> dict:
        """Get document collection stats."""
        if not documents:
            return {"count": 0, "total_chars": 0, "total_lines": 0}

        total_chars = sum(len(d.text) for d in documents)
        total_lines = sum(d.text.count("\n") + 1 for d in documents)

        by_type = {}
        for d in documents:
            by_type[d.doc_type] = by_type.get(d.doc_type, 0) + 1

        return {
            "count": len(documents),
            "total_chars": total_chars,
            "total_lines": total_lines,
            "avg_chars": total_chars // len(documents),
            "avg_lines": total_lines // len(documents),
            "by_type": by_type,
        }
