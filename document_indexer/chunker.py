"""Chunking strategies for document indexing."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import re


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""

    text: str
    chunk_id: str
    source: str
    title: str
    section: str
    start_pos: int
    end_pos: int
    strategy: str


class ChunkerStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, text: str, source: str, title: str) -> List[Chunk]:
        """Split text into chunks."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass


class FixedSizeChunker(ChunkerStrategy):
    """Fixed-size chunking with optional overlap."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    @property
    def name(self) -> str:
        return "fixed_size"

    def chunk(self, text: str, source: str, title: str) -> List[Chunk]:
        """Split text into fixed-size chunks."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))

            if end < len(text):
                break_point = self._find_break_point(text, end)
                end = break_point if break_point > start else end

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk = Chunk(
                    text=chunk_text,
                    chunk_id=f"{source}_fixed_{chunk_index}",
                    source=source,
                    title=title,
                    section=f"chunk_{chunk_index}",
                    start_pos=start,
                    end_pos=end,
                    strategy=self.name,
                )
                chunks.append(chunk)
                chunk_index += 1

            start = end - self.overlap if end < len(text) else end

        return chunks

    def _find_break_point(self, text: str, target_pos: int) -> int:
        """Find nearest sentence or word boundary."""
        search_range = min(50, target_pos // 2)
        best_pos = target_pos

        for offset in range(search_range):
            pos = target_pos - offset
            if pos > 0 and text[pos - 1 : pos + 1] in [". ", "! ", "? ", "\n"]:
                return pos

        for offset in range(search_range):
            pos = target_pos - offset
            if pos > 0 and text[pos] == " ":
                return pos

        return best_pos


class StructureBasedChunker(ChunkerStrategy):
    """Structure-based chunking using headers and sections."""

    HEADER_PATTERNS = [
        r"^#{1,6}\s+(.+)$",  # Markdown headers
        r"^(.+)\n={3,}$",  # Underlined headers
        r"^(.+)\n-{3,}$",  # Dash underlined
        r"^\[?section\s+\d+\]?\s*:?\s*(.+)$",  # Section labels
        r"^\d+\.\s+(.+)$",  # Numbered sections
        r"^class\s+\w+|^def\s+\w+",  # Python class/function
    ]

    def __init__(self, max_chunk_size: int = 1000):
        self.max_chunk_size = max_chunk_size

    @property
    def name(self) -> str:
        return "structure_based"

    def chunk(self, text: str, source: str, title: str) -> List[Chunk]:
        """Split text based on structure (headers/sections)."""
        sections = self._extract_sections(text)
        chunks = []

        for section_idx, (section_title, section_text) in enumerate(sections):
            section_chunks = self._split_section(
                section_text, source, title, section_title, section_idx
            )
            chunks.extend(section_chunks)

        return chunks

    def _extract_sections(self, text: str) -> List[tuple]:
        """Extract sections based on headers."""
        lines = text.split("\n")
        sections = []
        current_title = "Introduction"
        current_lines = []

        for i, line in enumerate(lines):
            header = self._detect_header(line, lines, i)

            if header:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines)))
                current_title = header
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines)))

        if not sections:
            sections = [("Full Document", text)]

        return sections

    def _detect_header(self, line: str, lines: List[str], index: int) -> Optional[str]:
        """Detect if line is a header."""
        stripped = line.strip()

        for pattern in self.HEADER_PATTERNS[:3]:
            match = re.match(pattern, stripped, re.MULTILINE)
            if match:
                return match.group(1).strip()

        for pattern in self.HEADER_PATTERNS[3:]:
            match = re.match(pattern, stripped, re.IGNORECASE)
            if match:
                title = match.group(1) if "(" in pattern else stripped
                return title.strip()

        if index + 1 < len(lines):
            next_line = lines[index + 1]
            if re.match(r"^={3,}$", next_line.strip()):
                return stripped
            if re.match(r"^-{3,}$", next_line.strip()):
                return stripped

        return None

    def _split_section(
        self, text: str, source: str, title: str, section_title: str, section_idx: int
    ) -> List[Chunk]:
        """Split section into chunks if too large."""
        if len(text) <= self.max_chunk_size:
            return [
                Chunk(
                    text=text.strip(),
                    chunk_id=f"{source}_struct_s{section_idx}_c0",
                    source=source,
                    title=title,
                    section=section_title,
                    start_pos=0,
                    end_pos=len(text),
                    strategy=self.name,
                )
            ]

        chunks = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current_chunk = []
        current_size = 0
        chunk_index = 0

        for paragraph in paragraphs:
            para_size = len(paragraph)

            if current_size + para_size > self.max_chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        chunk_id=f"{source}_struct_s{section_idx}_c{chunk_index}",
                        source=source,
                        title=title,
                        section=f"{section_title} (part {chunk_index + 1})",
                        start_pos=0,
                        end_pos=len(chunk_text),
                        strategy=self.name,
                    )
                )
                current_chunk = [paragraph]
                current_size = para_size
                chunk_index += 1
            else:
                current_chunk.append(paragraph)
                current_size += para_size

        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    chunk_id=f"{source}_struct_s{section_idx}_c{chunk_index}",
                    source=source,
                    title=title,
                    section=f"{section_title} (part {chunk_index + 1})",
                    start_pos=0,
                    end_pos=len(chunk_text),
                    strategy=self.name,
                )
            )

        return chunks
