# 🦜️🔗 LangChain PolarDB-X

A powerful integration between LangChain and PolarDB-X, enabling native vector search and SQL query capabilities for AI applications.

## Overview

LangChain PolarDB-X provides seamless integration between LangChain, a framework for building applications with large language models (LLMs), and PolarDB-X with native vector search support. This integration enables efficient vector storage and retrieval for AI applications like semantic search, recommendation systems, and RAG (Retrieval Augmented Generation).

PolarDB-X is a cloud-native distributed database system developed by Alibaba Cloud, featuring native HNSW-based vector index support that delivers high-performance approximate nearest neighbor (ANN) search directly within the database engine.

## Requirements

- Python 3.9+
- PolarDB-X with vector index support
- MySQL connector: `mysql-connector-python>=8.0.0` (included in package dependencies)
- Async support: `pip install langchain-polardbx[async]` (optional)
- MMR search: `pip install langchain-polardbx[mmr]` (optional)
- SQL database: `pip install langchain-polardbx[sql]` (optional)

### Enable Vector Index

PolarDB-X disables the vector index feature by default (`vidx_disabled = ON`). You need to enable it before using this package:

```sql
-- Enable vector index (run as admin/root on DN node)
SET GLOBAL vidx_disabled = OFF;
```

This setting takes effect immediately for new connections. No restart required.

All transaction isolation levels (READ-COMMITTED, REPEATABLE-READ, SERIALIZABLE) are supported — choose according to your business needs.

> **Note**: Some advanced features (e.g., inner product distance, index monitoring, `EF_CONSTRUCTION` parameter) require newer PolarDB-X versions. The package automatically detects available capabilities and adapts accordingly.

## Features

- **Native Vector Storage**: Store embeddings using PolarDB-X's native `VECTOR(N)` data type
- **HNSW Index**: Efficient approximate nearest neighbor search with configurable `M` and `EF_CONSTRUCTION` parameters
- **Multiple Distance Metrics**: Support for Cosine, Euclidean, and Inner Product distance
- **Similarity Search**: Perform efficient similarity searches with score thresholds
- **MMR Search**: Maximal Marginal Relevance search for diverse results
- **Metadata Filtering**: Filter search results by metadata with rich operators (`$eq`, `$ne`, `$gt`, `$gte`, `$lt`, `$lte`, `$in`, `$nin`, `$like`)
- **Dynamic Index Management**: Create, drop, and rebuild vector indexes at runtime without recreating tables
- **Search Mode Control**: Switch between ANN (index-accelerated) and KNN (full-scan) modes per query
- **Per-Query Tuning**: Adjust `ef_search` on a per-query basis for accuracy/latency trade-offs
- **Index Health Monitoring**: Runtime statistics, index health diagnostics, and preload checks
- **Batch Operations**: Efficient batch insert and bulk upsert with configurable batch size
- **Full Async Support**: All public methods have async equivalents (`aadd_texts`, `asimilarity_search`, etc.)
- **Dual-Version Compatibility**: Automatically detects database capabilities and adapts SQL accordingly
- **Partitioned Table Support**: Create partitioned vector tables with HASH/KEY/RANGE/LIST strategies, broadcast tables, and LOCALITY node assignment
- **Connection Pooling**: Built-in connection pool with automatic retry logic
- **SQL Database Integration**: Use PolarDB-X as a SQL database for LangChain agents with automatic DDL reflection compatibility (tab indentation, ENUM spacing, VECTOR type support)

## Installation

```bash
pip install langchain-polardbx
```

### Optional Dependencies

For async support:

```bash
pip install langchain-polardbx[async]
```

For MMR search support:

```bash
pip install langchain-polardbx[mmr]
```

For SQL database support (enables LangChain SQL agents):

```bash
pip install langchain-polardbx[sql]
```

For using OpenAI embeddings:

```bash
pip install langchain-openai
```

For using DashScope embeddings (Alibaba Cloud):

```bash
pip install langchain-community dashscope
```

## Quick Start

### Basic Usage

```python
from langchain_polardbx import PolarDBXVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

vectorstore = PolarDBXVectorStore(
    host="your-polardbx-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=embeddings,
    table_name="my_vectors",
    distance_strategy="cosine",  # or "euclidean", "inner_product"
    hnsw_m=16,  # HNSW index M parameter (3-200)
)

# Add texts
ids = vectorstore.add_texts(["Hello world", "PolarDB-X is great"])

# Similarity search
results = vectorstore.similarity_search("Hello", k=2)
for doc in results:
    print(f"- {doc.page_content}")

# Search with scores
results_with_scores = vectorstore.similarity_search_with_score("Hello", k=2)
for doc, score in results_with_scores:
    print(f"[Score: {score:.4f}] {doc.page_content}")
```

### Using DashScope Embeddings

```python
from langchain_polardbx import PolarDBXVectorStore
from langchain_community.embeddings import DashScopeEmbeddings

embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key="your-dashscope-api-key",
)

vectorstore = PolarDBXVectorStore(
    host="your-polardbx-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
    embedding=embeddings,
    table_name="langchain_vectors",
)
```

### SQL Database

`PolarDBXSQLDatabase` wraps LangChain's `SQLDatabase` with PolarDB-X-specific DDL reflection fixes, enabling seamless use with LangChain SQL agents. It automatically:

- Normalizes tab indentation in `SHOW CREATE TABLE` output (PolarDB-X uses tabs, standard MySQL uses two spaces)
- Fixes ENUM/SET value list spacing (`enum('A', 'B')` → `enum('A','B')`)
- Registers a custom `VECTOR` type so tables with vector columns don't crash reflection
- Auto-swaps `mysql+pymysql://` URIs to use the PolarDB-X dialect

```python
from langchain_polardbx import PolarDBXSQLDatabase

db = PolarDBXSQLDatabase.from_uri(
    "mysql+pymysql://user:password@host:3306/your-database"
)

# List tables
tables = db.get_usable_table_names()

# Get table schema info for SQL agents
info = db.get_table_info(["your_table"])

# Run SQL queries
result = db.run("SELECT COUNT(*) FROM your_table")
```

Use with LangChain SQL agent:

```python
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0)
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit)

agent.invoke({"input": "How many records are in your_table?"})
```

> **Note**: `VECTOR INDEX` definitions in `SHOW CREATE TABLE` are not parsed by SQLAlchemy and will be silently skipped with a warning. This is expected — the index info is not needed for SQL query generation. Tables with `VECTOR` columns are fully supported.

## Usage Examples

### Creating from Documents

```python
from langchain_core.documents import Document

documents = [
    Document(page_content="Hello world", metadata={"source": "greeting"}),
    Document(page_content="LangChain is great", metadata={"source": "review"}),
]

vectorstore = PolarDBXVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    host="your-host",
    port=3306,
    user="your-user",
    password="your-password",
    database="your-database",
)
```

### Search with Metadata Filter

```python
# Add texts with metadata
texts = ["Apple is a fruit", "Banana is yellow", "Car is a vehicle"]
metadatas = [
    {"category": "fruit", "price": 5},
    {"category": "fruit", "price": 3},
    {"category": "vehicle", "price": 20000},
]
vectorstore.add_texts(texts, metadatas=metadatas)

# Simple equality filter
results = vectorstore.similarity_search("yellow things", k=2, filter={"category": "fruit"})

# Operator filter: price > 2 AND category = fruit
results = vectorstore.similarity_search(
    "fresh fruit",
    k=2,
    filter={"category": "fruit", "price": {"$gt": 2}},
)
```

Supported filter operators:

| Operator | SQL | Description |
|----------|-----|-------------|
| `$eq` | `=` | Equal (default for simple values) |
| `$ne` | `!=` | Not equal |
| `$gt` | `>` | Greater than |
| `$gte` | `>=` | Greater than or equal |
| `$lt` | `<` | Less than |
| `$lte` | `<=` | Less than or equal |
| `$in` | `IN` | In a list of values |
| `$nin` | `NOT IN` | Not in a list of values |
| `$like` | `LIKE` | Pattern match |

### Score Threshold Filtering

```python
# Only return results with distance <= 0.8 (lower distance = more similar)
results = vectorstore.similarity_search_with_score(
    "Hello",
    k=10,
    score_threshold=0.8,
)
```

### Maximal Marginal Relevance (MMR) Search

```python
# MMR search for diverse results
results = vectorstore.max_marginal_relevance_search(
    "technology",
    k=4,
    fetch_k=20,
    lambda_mult=0.5,  # 0 = max diversity, 1 = max relevance
)
```

### Search Mode Control

```python
# Force ANN (use vector index for HNSW acceleration)
results = vectorstore.similarity_search("query", k=10, search_type="ann")

# Force KNN (full table scan, bypass vector index)
results = vectorstore.similarity_search("query", k=10, search_type="knn")

# Let the optimizer decide (default)
results = vectorstore.similarity_search("query", k=10, search_type="auto")

# Tune ef_search per query (higher = more accurate, slower)
results = vectorstore.similarity_search("query", k=10, ef_search=100)
```

### Dynamic Vector Index Management

```python
# Create a vector index at runtime
vectorstore.apply_vector_index(
    index_name="my_vi",
    m=16,
    distance="COSINE",
    ef_construction=200,
)

# Drop the vector index
vectorstore.drop_vector_index()

# Rebuild the index to reclaim space and improve recall
vectorstore.optimize()
```

### Index Monitoring

```python
# Get runtime statistics
stats = vectorstore.get_stats()
print(stats)  # e.g. {"Vidx_query_count": 100, "Vidx_load_node_hits": 950, "Vidx_load_node_misses": 50, ...}

# Preload HNSW index into memory cache to eliminate cold-start latency
vectorstore.preload_index()

# Check if preloading would fit in cache
check_result = vectorstore.preload_check()
print(check_result)

# Diagnose index health (combines VECTOR_INDEXES view + EXPLAIN)
health = vectorstore.explain_index_health()
print(health)
```

### Delete and Manage Vectors

```python
# Delete by IDs
vectorstore.delete(ids=["id1", "id2"])

# Get documents by IDs
docs = vectorstore.get_by_ids(["id1", "id2"])

# Count vectors
count = vectorstore.count()

# Clear all data (TRUNCATE TABLE)
vectorstore.clear()

# Drop the entire table
vectorstore.drop_table()

# Search documents by metadata only (no vector similarity)
docs = vectorstore.search_by_metadata(filter={"category": "fruit"}, limit=10)

# Delete documents matching metadata conditions
deleted_count = vectorstore.delete_by_metadata(filter={"status": {"$eq": "deleted"}})
```

### Bulk Upsert

```python
# Upsert multiple texts with pre-computed embeddings, metadata, and custom IDs
texts = ["doc1", "doc2", "doc3"]
embeddings = [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...]]
metadatas = [{"src": "web"}, {"src": "pdf"}, {"src": "api"}]
ids = ["a1", "a2", "a3"]

vectorstore.bulk_upsert(
    texts=texts,
    embeddings=embeddings,
    ids=ids,
    metadatas=metadatas,
    batch_size=100,
)
```

### Async API

All public methods have async equivalents:

```python
import asyncio

async def main():
    # Add texts
    ids = await vectorstore.aadd_texts(["Hello", "World"])

    # Search
    results = await vectorstore.asimilarity_search("Hello", k=2)

    # MMR search
    results = await vectorstore.amax_marginal_relevance_search("Hello", k=4)

    # Delete
    await vectorstore.adelete(ids=["id1"])

    # Get by IDs
    docs = await vectorstore.aget_by_ids(["id1", "id2"])

    # Count
    count = await vectorstore.acount()

    # Clear
    await vectorstore.aclear()

    # Dynamic index management
    await vectorstore.aapply_vector_index(index_name="vi", m=16)
    await vectorstore.adrop_vector_index()
    await vectorstore.aoptimize()

    # Index monitoring
    stats = await vectorstore.aget_stats()
    await vectorstore.apreload_index()
    health = await vectorstore.aexplain_index_health()

asyncio.run(main())
```

### Partitioned Tables

PolarDB-X is a distributed database that supports table partitioning for scalability. This package supports creating partitioned vector tables and standalone partitioned tables.

#### Vector Store with Partitioning

```python
from langchain_polardbx import PolarDBXVectorStore

# HASH partitioning (8 partitions on the id column)
vectorstore = PolarDBXVectorStore(
    host="your-host", port=3306, user="your-user", password="your-password",
    database="your-database", embedding=embeddings,
    table_name="partitioned_vectors",
    partition_by="HASH",          # "HASH", "KEY", "RANGE", or "LIST"
    partition_column="id",        # column to partition on (default: "id")
    partitions=8,                 # number of partitions (HASH/KEY only)
)

# Broadcast table (full copy on every DN node)
vectorstore = PolarDBXVectorStore(
    ..., broadcast=True,
)

# RANGE partitioning
vectorstore = PolarDBXVectorStore(
    ..., partition_by="RANGE", partition_column="id",
    partition_defs=[
        {"name": "p0", "values_less_than": 1000},
        {"name": "p1", "values_less_than": "MAXVALUE"},
    ],
)

# With LOCALITY (pin table to a specific DN node)
vectorstore = PolarDBXVectorStore(
    ..., locality="dn=your-dn-node-name",
)
```

> **Note**: Partitioned vector tables are not supported on PolarDB-X v3 instances. The package automatically detects this and raises `NotSupportedError` if you attempt to use partitioning on an incompatible version.

#### Standalone Partitioned Table (Non-Vector)

For non-vector tables (e.g., for SQL agents), use `create_partitioned_table`:

```python
from langchain_polardbx import create_partitioned_table

# HASH partitioning
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="orders",
    columns=[
        "id BIGINT NOT NULL AUTO_INCREMENT",
        "user_id BIGINT NOT NULL",
        "amount DECIMAL(10,2)",
        "created_at DATETIME",
        "PRIMARY KEY (id)",
    ],
    partition_by="HASH",
    partition_column="user_id",
    partitions=16,
)

# Broadcast table (dimension table, full copy on every DN)
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="dim_currency",
    columns=["code VARCHAR(10)", "name VARCHAR(100)", "PRIMARY KEY (code)"],
    broadcast=True,
)

# RANGE partitioning
create_partitioned_table(
    uri="mysql+pymysql://user:password@host:3306/database",
    table_name="logs",
    columns=["id BIGINT NOT NULL", "ts DATETIME", "PRIMARY KEY (id)"],
    partition_by="RANGE",
    partition_column="id",
    partition_defs=[
        {"name": "p0", "values_less_than": 1000000},
        {"name": "p1", "values_less_than": 2000000},
        {"name": "p2", "values_less_than": "MAXVALUE"},
    ],
)
```

Supported partition strategies:

| Strategy | Parameters | Description |
|----------|------------|-------------|
| `HASH` | `partition_column`, `partitions` | Hash partitioning by column value |
| `KEY` | `partition_column`, `partitions` | Key partitioning (supports multi-column) |
| `RANGE` | `partition_column`, `partition_defs` | Range partitioning with explicit boundaries |
| `LIST` | `partition_column`, `partition_defs` | List partitioning with explicit value lists |
| `BROADCAST` | (none) | Full table copy on every DN node |
| `LOCALITY` | `locality` | Pin table to a specific storage node |

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `host` | str | - | PolarDB-X host address |
| `port` | int | - | PolarDB-X port number |
| `user` | str | - | Username |
| `password` | str | - | Password |
| `database` | str | - | Database name |
| `embedding` | Embeddings | - | LangChain embedding model |
| `table_name` | str | `"polardbx_vectors"` | Table name for vector storage |
| `distance_strategy` | str | `"cosine"` | Distance function: `"cosine"`, `"euclidean"`, or `"inner_product"` |
| `hnsw_m` | int | 6 | HNSW index M parameter (3-200) |
| `pool_size` | int | 5 | Connection pool size |
| `pre_delete_table` | bool | False | Drop table before creating |
| `embedding_dimension` | int | None | Embedding dimension (auto-inferred if not provided) |
| `ef_construction` | int | None | HNSW build-time candidate list size (5-1000) |
| `connection_retries` | int | 3 | Number of connection retry attempts |
| `retry_delay` | float | 1.0 | Delay between retries in seconds |
| `vector_index_name` | str | None | Vector index name for FORCE INDEX hints (auto-detected if None) |
| `partition_by` | str | None | Partition strategy: `"HASH"`, `"KEY"`, `"RANGE"`, or `"LIST"` |
| `partitions` | int | 0 | Number of partitions (required for HASH/KEY) |
| `partition_column` | str | `"id"` | Column to partition on |
| `broadcast` | bool | False | Create a broadcast table (full copy on every DN) |
| `locality` | str | None | Pin table to a specific DN node, e.g. `"dn=node-name"` |
| `partition_defs` | list | None | Partition definitions for RANGE/LIST (see examples above) |
| `**kwargs` | - | - | Additional connection arguments (e.g. `ssl_ca`, `ssl_cert`, `ssl_key`, `ssl_disabled`) |

## PolarDB-X Vector Functions Used

This integration uses PolarDB-X's native vector functions:

- `VECTOR(N)` — Vector column data type with N dimensions
- `VEC_FROMTEXT('[1,2,3]')` — Convert JSON array string to vector
- `VEC_TOTEXT(vector)` — Convert vector to JSON array string
- `VEC_DISTANCE(v1, v2)` — Auto-inferred distance function
- `VEC_DISTANCE_COSINE(v1, v2)` — Cosine distance
- `VEC_DISTANCE_EUCLIDEAN(v1, v2)` — Euclidean distance
- `VEC_DISTANCE_INNER_PRODUCT(v1, v2)` — Inner product distance
- `VECTOR_DIM(v)` — Get vector dimension
- `VECTOR INDEX (col) M=N DISTANCE=COSINE` — HNSW vector index DDL
- `EF_CONSTRUCTION=N` — HNSW build-time parameter in DDL
- `SET SESSION vidx_hnsw_ef_search = N` — Per-session search width tuning
- `SHOW GLOBAL STATUS LIKE 'Vidx%'` — Runtime index statistics
- `CALL dbms_vidx.preload(db, table, col)` — Preload index into cache
- `CALL dbms_vidx.preload_check(db, table, col)` — Check preload feasibility
- `information_schema.VECTOR_INDEXES` — Vector index metadata view

## Development

This package uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
cd libs/langchain-polardbx

# Install dependencies
uv sync --group test --group test_integration

# Run unit tests
make test

# Run integration tests (requires a running PolarDB-X instance)
make integration_tests

# Lint
make lint
```

## License

MIT
