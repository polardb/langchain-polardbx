"""Shared test helpers for LangChain PolarDBXVectorStore tests."""

import os
import hashlib
from urllib.parse import urlparse

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_polardbx import PolarDBXVectorStore


# ---- .env loading (search upward from this file) ----

def _load_dotenv():
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        env_path = os.path.join(d, ".env")
        if os.path.exists(env_path):
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip())
            return
        d = os.path.dirname(d)

_load_dotenv()

_uri = urlparse(os.environ.get("POLARDBX_URI", ""))
DB_HOST = _uri.hostname or "localhost"
DB_PORT = _uri.port or 3306
DB_USER = _uri.username or "root"
DB_PASS = _uri.password or ""
DB_NAME = _uri.path.lstrip("/") or "test"


# ---- Fake embeddings ----

class FakeEmbeddings(Embeddings):
    """Deterministic embedding using MD5 hash — same text always yields same vector."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        h = hashlib.md5(text.encode()).digest()
        vec = np.frombuffer(h * (self.dim // 16 + 1), dtype=np.uint8)[:self.dim]
        return (vec / 255.0).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]


EMB = FakeEmbeddings(dim=128)

# ---- Test data ----

TEXTS = [
    "PolarDB-X is a distributed database",
    "LangChain is a framework for LLM applications",
    "Vector search uses cosine similarity",
    "HNSW index provides fast approximate search",
    "Python is a popular programming language",
]

METADATAS = [
    {"category": "database", "lang": "en"},
    {"category": "framework", "lang": "en"},
    {"category": "search", "lang": "en"},
    {"category": "index", "lang": "en"},
    {"category": "language", "lang": "en"},
]


# ---- Store factory ----

def make_store(
    table_name: str = "test_polardbx_langchain",
    distance_strategy: str = "cosine",
    pre_delete: bool = True,
    **kwargs,
) -> PolarDBXVectorStore:
    return PolarDBXVectorStore(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, embedding=EMB, table_name=table_name,
        distance_strategy=distance_strategy, pre_delete_table=pre_delete,
        **kwargs,
    )
