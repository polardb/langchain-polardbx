# 🦜️🔗 LangChain PolarDB-X

This repository contains 1 package with PolarDB-X integrations with LangChain:

- [langchain-polardbx](https://pypi.org/project/langchain-polardbx/)

## Installation

```bash
pip install langchain-polardbx
```

## Quick Start

```python
from langchain_polardbx import PolarDBXVectorStore
from langchain_core.embeddings import DeterministicFakeEmbedding

vectorstore = PolarDBXVectorStore(
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=DeterministicFakeEmbedding(embedding_size=128),
    table_name="my_vectors",
)

# Add texts
vectorstore.add_texts(["Hello world", "PolarDB-X is great"])

# Search
results = vectorstore.similarity_search("Hello", k=2)
```

## Features

- Native PolarDB-X VECTOR data type and HNSW index support
- Cosine and Euclidean distance metrics
- Dynamic vector index management (create/drop at runtime)
- ef_search tuning per query
- ANN/KNN search mode switching
- Vector index runtime monitoring (get_stats)
- OPTIMIZE TABLE for index rebuild
- Full async support (aadd_texts, asimilarity_search, etc.)
- Metadata filtering with JSON path operators
- MMR (Maximal Marginal Relevance) search
- Connection pooling with retry logic

## Development

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync --group test --group test_integration

# Run unit tests
make test

# Run integration tests
make integration_tests

# Lint
make lint
```

## License

MIT
