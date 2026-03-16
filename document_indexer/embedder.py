"""Embedding generation for document chunks."""

import hashlib
import os
from typing import List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


class Embedder:
    """Generate embeddings for text using OpenAI API."""

    DEFAULT_MODEL = "text-embedding-3-small"
    DEFAULT_DIMENSIONS = 1536

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or self.DEFAULT_MODEL
        self.dimensions = dimensions or self.DEFAULT_DIMENSIONS
        self._cache = {}

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for list of texts."""
        if not texts:
            return []

        results = []
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if cache_key in self._cache:
                results.append((i, self._cache[cache_key]))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)

        if uncached_texts:
            embeddings = self._api_call(uncached_texts)
            for idx, emb in zip(uncached_indices, embeddings):
                cache_key = self._get_cache_key(texts[idx])
                self._cache[cache_key] = emb
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return [emb for _, emb in results]

    def embed_single(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        embeddings = self.embed([text])
        return embeddings[0] if embeddings else []

    def _api_call(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI API for embeddings."""
        if not self.api_key:
            return self._generate_mock_embeddings(texts)

        try:
            response = requests.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "input": texts,
                    "model": self.model,
                    "dimensions": self.dimensions,
                },
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            print(f"Embedding API error: {e}")
            return self._generate_mock_embeddings(texts)

    def _generate_mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic mock embeddings for testing."""
        embeddings = []
        for text in texts:
            hash_val = hashlib.md5(text.encode()).hexdigest()
            float_vals = [
                int(hash_val[i : i + 8], 16) / 2**32 - 0.5
                for i in range(0, min(len(hash_val), 384), 8)
            ]

            while len(float_vals) < self.dimensions:
                float_vals.extend(float_vals[: self.dimensions - len(float_vals)])

            float_vals = float_vals[: self.dimensions]

            norm = sum(x * x for x in float_vals) ** 0.5
            if norm > 0:
                float_vals = [x / norm for x in float_vals]

            embeddings.append(float_vals)

        return embeddings

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        return hashlib.md5(text.encode()).hexdigest()

    def clear_cache(self):
        """Clear embedding cache."""
        self._cache.clear()
