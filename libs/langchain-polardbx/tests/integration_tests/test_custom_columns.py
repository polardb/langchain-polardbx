"""Integration tests for custom column support.

Requires a live PolarDB-X instance (configured via .env).
Tests the full lifecycle: create table → add → search → filter →
upsert → delete, with custom column names and mapped metadata columns.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER, EMB, make_store

from langchain_polardbx import Column

# ---------------------------------------------------------------------------
# Test data — e-commerce product catalog style
# ---------------------------------------------------------------------------

PRODUCT_TEXTS = [
    "Wireless Bluetooth Headphones with Noise Cancellation",
    "Smart Watch with Heart Rate Monitor and GPS",
    "Stainless Steel Water Bottle 500ml",
    "Mechanical Keyboard with Cherry MX Switches",
    "USB-C Hub with 4K HDMI and Power Delivery",
]

PRODUCT_METADATAS = [
    {"category": "audio", "price": 299, "brand": "Sony", "tags": "wireless"},
    {"category": "wearable", "price": 399, "brand": "Apple", "tags": "fitness"},
    {"category": "accessory", "price": 25, "brand": "Thermos", "tags": "outdoor"},
    {"category": "peripheral", "price": 129, "brand": "Ducky", "tags": "gaming"},
    {"category": "accessory", "price": 49, "brand": "Anker", "tags": "charging"},
]

TABLE = "test_custom_cols_it"


def _make_custom_store(**kwargs):
    """Create a store with custom columns for IT."""
    return make_store(
        table_name=TABLE,
        id_column="product_id",
        content_column="description",
        embedding_column="embed",
        metadata_json_column="extra",
        metadata_columns=[
            Column("category", "VARCHAR(50)", nullable=False),
            Column("price", "DECIMAL(10,2)"),
            Column("brand", "VARCHAR(50)"),
        ],
        **kwargs,
    )


@pytest.fixture(autouse=True)
def _cleanup():
    """Ensure table is dropped before and after each test."""
    try:
        vs = make_store(table_name=TABLE, pre_delete=False)
        vs.drop_table()
        vs.close()
    except Exception:
        pass
    yield
    try:
        vs = make_store(table_name=TABLE, pre_delete=False)
        vs.drop_table()
        vs.close()
    except Exception:
        pass


# ===========================================================================
# 1. Table creation with custom DDL
# ===========================================================================


def test_table_created_with_custom_columns():
    """Verify the table is created with the correct custom columns."""
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS[:1], PRODUCT_METADATAS[:1])

    # Verify table structure via SHOW CREATE TABLE
    import pymysql

    conn = pymysql.connect(
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS, database=DB_NAME
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"SHOW CREATE TABLE `{TABLE}`")
            row = cur.fetchone()
            ddl = row[1] if row else ""
    finally:
        conn.close()

    # Core custom columns should be present
    assert "`product_id`" in ddl
    assert "`description`" in ddl
    assert "`embed`" in ddl
    assert "`extra`" in ddl  # JSON column
    # Metadata columns
    assert "`category`" in ddl
    assert "`price`" in ddl
    assert "`brand`" in ddl
    # VECTOR INDEX on embed column
    assert "VECTOR INDEX" in ddl
    assert "`embed`" in ddl

    vs.drop_table()
    vs.close()


# ===========================================================================
# 2. Add + similarity search (sync)
# ===========================================================================


def test_add_and_search_sync():
    vs = _make_custom_store()
    ids = vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)
    assert len(ids) == 5

    # similarity_search should return results with reconstructed metadata
    results = vs.similarity_search("wireless headphones audio", k=2)
    assert len(results) >= 1

    # Verify metadata reconstruction on the first result
    doc = results[0]
    meta = doc.metadata
    assert "category" in meta
    assert "price" in meta
    assert "brand" in meta
    assert "tags" in meta  # this went to JSON column

    # All 5 docs should be searchable
    all_results = vs.similarity_search("product", k=5)
    assert len(all_results) == 5

    # Each result should have complete metadata
    for d in all_results:
        assert "category" in d.metadata
        assert "price" in d.metadata
        assert "brand" in d.metadata
        assert "tags" in d.metadata

    vs.drop_table()
    vs.close()


# ===========================================================================
# 3. get_by_ids with custom columns
# ===========================================================================


def test_get_by_ids_sync():
    vs = _make_custom_store()
    ids = vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    docs = vs.get_by_ids([ids[0], ids[2]])
    assert len(docs) == 2
    # get_by_ids does not guarantee order — match by ID
    docs_by_id = {d.id: d for d in docs}
    assert ids[0] in docs_by_id
    assert ids[2] in docs_by_id

    # Verify metadata reconstruction
    meta0 = docs_by_id[ids[0]].metadata
    assert meta0["category"] == "audio"
    assert float(meta0["price"]) == 299
    assert meta0["brand"] == "Sony"
    assert meta0["tags"] == "wireless"

    vs.drop_table()
    vs.close()


# ===========================================================================
# 4. exists with custom id column
# ===========================================================================


def test_exists_sync():
    vs = _make_custom_store()
    ids = vs.add_texts(PRODUCT_TEXTS[:2], metadatas=PRODUCT_METADATAS[:2])

    assert vs.exists(ids[0]) is True
    assert vs.exists("nonexistent-id") is False

    vs.drop_table()
    vs.close()


# ===========================================================================
# 5. Filter on mapped column (direct column reference)
# ===========================================================================


def test_filter_on_mapped_column_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    # Filter on category (mapped column)
    results = vs.similarity_search(
        "product", k=5, filter={"category": "accessory"}
    )
    assert len(results) >= 1
    assert all(d.metadata["category"] == "accessory" for d in results)

    vs.drop_table()
    vs.close()


def test_filter_on_mapped_column_numeric_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    # Filter on price (mapped numeric column)
    results = vs.similarity_search(
        "product", k=5, filter={"price": {"$gt": 100}}
    )
    assert len(results) >= 1
    for d in results:
        assert float(d.metadata["price"]) > 100

    vs.drop_table()
    vs.close()


# ===========================================================================
# 6. Filter on unmapped key (JSON_EXTRACT on custom JSON column)
# ===========================================================================


def test_filter_on_json_column_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    # tags is stored in JSON column 'extra'
    results = vs.similarity_search(
        "product", k=5, filter={"tags": "wireless"}
    )
    assert len(results) >= 1
    assert all(d.metadata.get("tags") == "wireless" for d in results)

    vs.drop_table()
    vs.close()


# ===========================================================================
# 7. Upsert (update existing document)
# ===========================================================================


def test_upsert_sync():
    vs = _make_custom_store()
    ids = vs.add_texts(PRODUCT_TEXTS[:1], metadatas=PRODUCT_METADATAS[:1])

    # Upsert with updated content
    from langchain_core.documents import Document

    vs.upsert(
        [
            Document(
                page_content="UPDATED: Premium Wireless Headphones",
                metadata={
                    "category": "audio",
                    "price": 349,
                    "brand": "Sony",
                    "tags": "premium",
                },
            )
        ],
        ids=[ids[0]],
    )

    docs = vs.get_by_ids([ids[0]])
    assert len(docs) == 1
    assert "UPDATED" in docs[0].page_content
    assert float(docs[0].metadata["price"]) == 349
    assert docs[0].metadata["tags"] == "premium"

    vs.drop_table()
    vs.close()


# ===========================================================================
# 8. Delete by IDs
# ===========================================================================


def test_delete_sync():
    vs = _make_custom_store()
    ids = vs.add_texts(PRODUCT_TEXTS[:3], metadatas=PRODUCT_METADATAS[:3])

    assert vs.count() == 3
    vs.delete([ids[0]])
    assert not vs.exists(ids[0])
    assert vs.count() == 2

    vs.drop_table()
    vs.close()


# ===========================================================================
# 9. search_by_metadata with custom columns
# ===========================================================================


def test_search_by_metadata_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    # Filter on mapped column
    results = vs.search_by_metadata(filter={"category": "audio"}, limit=5)
    assert len(results) >= 1
    assert all(d.metadata["category"] == "audio" for d in results)

    # Filter on JSON column (unmapped key)
    results = vs.search_by_metadata(filter={"tags": "gaming"}, limit=5)
    assert len(results) >= 1
    assert all(d.metadata.get("tags") == "gaming" for d in results)

    vs.drop_table()
    vs.close()


# ===========================================================================
# 10. delete_by_metadata with custom columns
# ===========================================================================


def test_delete_by_metadata_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    assert vs.count() == 5
    deleted = vs.delete_by_metadata({"category": "accessory"})
    assert deleted >= 2  # Thermos + Anker are accessories

    remaining = vs.search_by_metadata(filter={"category": "accessory"}, limit=10)
    assert len(remaining) == 0

    vs.drop_table()
    vs.close()


# ===========================================================================
# 11. bulk_upsert with custom columns
# ===========================================================================


def test_bulk_upsert_sync():
    vs = _make_custom_store()

    embeddings = [EMB.embed_query(t) for t in PRODUCT_TEXTS[:3]]
    ids = vs.bulk_upsert(
        texts=PRODUCT_TEXTS[:3],
        embeddings=embeddings,
        metadatas=PRODUCT_METADATAS[:3],
    )
    assert len(ids) == 3
    assert vs.count() == 3

    # Verify data
    docs = vs.get_by_ids([ids[0]])
    assert docs[0].metadata["category"] == "audio"

    vs.drop_table()
    vs.close()


# ===========================================================================
# 12. Async: add + search + get + delete
# ===========================================================================


async def test_async_lifecycle():
    vs = _make_custom_store()

    # aadd_texts
    ids = await vs.aadd_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)
    assert len(ids) == 5

    # asimilarity_search
    results = await vs.asimilarity_search("watch fitness", k=2)
    assert len(results) >= 1
    doc = results[0]
    assert "category" in doc.metadata
    assert "price" in doc.metadata
    assert "brand" in doc.metadata
    assert "tags" in doc.metadata

    # aget_by_ids
    docs = await vs.aget_by_ids([ids[1]])
    assert len(docs) == 1
    assert docs[0].metadata["category"] == "wearable"

    # aexists
    assert await vs.aexists(ids[1]) is True
    assert await vs.aexists("no-such-id") is False

    # adelete
    await vs.adelete([ids[0]])
    assert not await vs.aexists(ids[0])

    # asimilarity_search with filter on mapped column
    results = await vs.asimilarity_search(
        "product", k=5, filter={"category": "peripheral"}
    )
    assert all(d.metadata["category"] == "peripheral" for d in results)

    vs.drop_table()
    vs.close()


# ===========================================================================
# 13. Async: upsert + search_by_metadata
# ===========================================================================


async def test_async_upsert_and_search_by_metadata():
    vs = _make_custom_store()

    ids = await vs.aadd_texts(PRODUCT_TEXTS[:2], metadatas=PRODUCT_METADATAS[:2])

    from langchain_core.documents import Document

    await vs.aupsert(
        [
            Document(
                page_content="UPDATED Headphones Pro Max",
                metadata={
                    "category": "audio",
                    "price": 499,
                    "brand": "Sony",
                    "tags": "premium",
                },
            )
        ],
        ids=[ids[0]],
    )

    docs = await vs.aget_by_ids([ids[0]])
    assert "UPDATED" in docs[0].page_content
    assert float(docs[0].metadata["price"]) == 499

    # search_by_metadata on mapped column
    results = await vs.asearch_by_metadata(
        filter={"category": "audio"}, limit=5
    )
    assert len(results) >= 1
    assert results[0].metadata["category"] == "audio"

    # delete_by_metadata on mapped column
    deleted = await vs.adelete_by_metadata({"category": "wearable"})
    assert deleted >= 1

    vs.drop_table()
    vs.close()


# ===========================================================================
# 14. Default schema backward compatibility
# ===========================================================================


def test_default_schema_still_works():
    """Ensure the default schema (no custom columns) still works."""
    vs = make_store(table_name=TABLE)
    ids = vs.add_texts(PRODUCT_TEXTS[:3], metadatas=PRODUCT_METADATAS[:3])
    assert len(ids) == 3

    results = vs.similarity_search("headphones", k=2)
    assert len(results) >= 1

    docs = vs.get_by_ids([ids[0]])
    assert docs[0].metadata["category"] == "audio"

    vs.drop_table()
    vs.close()


# ===========================================================================
# 15. MMR search with custom columns
# ===========================================================================


def test_mmr_search_sync():
    vs = _make_custom_store()
    vs.add_texts(PRODUCT_TEXTS, metadatas=PRODUCT_METADATAS)

    results = vs.max_marginal_relevance_search("product", k=3, fetch_k=5)
    assert len(results) <= 3

    vs.drop_table()
    vs.close()
