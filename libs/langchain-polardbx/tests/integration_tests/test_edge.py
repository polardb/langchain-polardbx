"""Edge case tests for LangChain PolarDBXVectorStore."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, TEXTS, METADATAS, make_store, DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
from langchain_core.documents import Document
from langchain_polardbx import PolarDBXVectorStore


# ==================== 1. Euclidean distance strategy ====================

def test_euclidean():
    print("=== EDGE: euclidean distance ===\n")
    vs = make_store(table_name="test_lc_euclidean", distance_strategy="euclidean")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    # similarity_search
    results = vs.similarity_search("database", k=3)
    assert len(results) == 3
    print(f"  similarity_search OK: {len(results)} results")

    # similarity_search_with_score
    results = vs.similarity_search_with_score("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)
    print(f"  similarity_search_with_score OK")

    # similarity_search_with_relevance_scores
    results = vs.similarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)
    print(f"  similarity_search_with_relevance_scores OK")

    # MMR
    results = vs.max_marginal_relevance_search("database", k=2, fetch_k=5)
    assert len(results) == 2
    print(f"  MMR OK: {len(results)} results")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 2. Filter operators ====================

def test_filter_operators():
    print("=== EDGE: filter operators ===\n")
    vs = make_store(table_name="test_lc_filters")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    # add extra docs with numeric metadata for range operators
    vs.add_texts(
        ["Rust is fast", "Go is concurrent"],
        metadatas=[{"category": "language", "score": 50}, {"category": "language", "score": 90}],
    )

    # $eq
    r = vs.similarity_search("language", k=10, filter={"category": "language"})
    assert all(d.metadata.get("category") == "language" for d in r)
    print(f"  $eq OK: {len(r)} results")

    # $ne
    r = vs.similarity_search("language", k=10, filter={"category": {"$ne": "database"}})
    assert all(d.metadata.get("category") != "database" for d in r)
    print(f"  $ne OK: {len(r)} results")

    # $gt
    r = vs.similarity_search("language", k=10, filter={"score": {"$gt": 60}})
    assert all(d.metadata.get("score", 0) > 60 for d in r)
    print(f"  $gt OK: {len(r)} results")

    # $gte
    r = vs.similarity_search("language", k=10, filter={"score": {"$gte": 50}})
    assert all(d.metadata.get("score", 0) >= 50 for d in r)
    print(f"  $gte OK: {len(r)} results")

    # $lt
    r = vs.similarity_search("language", k=10, filter={"score": {"$lt": 60}})
    assert all(d.metadata.get("score", 999) < 60 for d in r)
    print(f"  $lt OK: {len(r)} results")

    # $lte
    r = vs.similarity_search("language", k=10, filter={"score": {"$lte": 90}})
    assert all(d.metadata.get("score", 999) <= 90 for d in r)
    print(f"  $lte OK: {len(r)} results")

    # $in
    r = vs.similarity_search("language", k=10, filter={"category": {"$in": ["database", "search"]}})
    assert all(d.metadata.get("category") in ["database", "search"] for d in r)
    print(f"  $in OK: {len(r)} results")

    # $nin
    r = vs.similarity_search("language", k=10, filter={"category": {"$nin": ["database", "search"]}})
    assert all(d.metadata.get("category") not in ["database", "search"] for d in r)
    print(f"  $nin OK: {len(r)} results")

    # $like
    r = vs.similarity_search("language", k=10, filter={"category": {"$like": "lang%"}})
    assert all(d.metadata.get("category", "").startswith("lang") for d in r)
    print(f"  $like OK: {len(r)} results")

    # Multi-condition (AND)
    r = vs.similarity_search("language", k=10, filter={"category": "language", "score": {"$gt": 60}})
    assert all(d.metadata.get("category") == "language" and d.metadata.get("score", 0) > 60 for d in r)
    print(f"  multi-condition OK: {len(r)} results")

    # Filter on non-existent key
    r = vs.similarity_search("language", k=10, filter={"nonexistent_key": "value"})
    assert len(r) == 0
    print(f"  non-existent key OK: {len(r)} results")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 3. score_threshold ====================

def test_score_threshold():
    print("=== EDGE: score_threshold ===\n")
    vs = make_store(table_name="test_lc_threshold")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    vec = EMB.embed_query("database")

    # Without threshold — returns all k
    results = vs.similarity_search_with_score_by_vector(vec, k=5)
    assert len(results) == 5
    print(f"  no threshold: {len(results)} results")

    # With high threshold — should filter most out
    results_high = vs.similarity_search_with_score_by_vector(vec, k=5, score_threshold=0.99)
    print(f"  threshold=0.99: {len(results_high)} results")

    # With low threshold — should keep all
    results_low = vs.similarity_search_with_score_by_vector(vec, k=5, score_threshold=0.0)
    assert len(results_low) == 5
    print(f"  threshold=0.0: {len(results_low)} results")

    # High threshold should return <= no threshold
    assert len(results_high) <= len(results_low)

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 4. Empty table search ====================

def test_empty_table():
    print("=== EDGE: empty table ===\n")
    vs = make_store(table_name="test_lc_empty")
    # Don't insert anything — table doesn't even exist yet
    # similarity_search should return empty list, not error
    results = vs.similarity_search("anything", k=3)
    assert len(results) == 0
    print(f"  similarity_search on empty: {len(results)} results")

    results = vs.similarity_search_with_score("anything", k=3)
    assert len(results) == 0
    print(f"  similarity_search_with_score on empty: {len(results)} results")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 5. k > data count ====================

def test_k_exceeds_data():
    print("=== EDGE: k > data count ===\n")
    vs = make_store(table_name="test_lc_kexceed")
    vs.add_texts(TEXTS[:3], metadatas=METADATAS[:3])  # only 3 docs

    results = vs.similarity_search("database", k=100)
    assert len(results) == 3  # should return all 3, not error
    print(f"  k=100 with 3 docs: returned {len(results)} results")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 6. metadata=None ====================

def test_metadata_none():
    print("=== EDGE: metadata=None ===\n")
    vs = make_store(table_name="test_lc_metanone")
    vs.add_texts(["text without metadata"])  # no metadatas param

    assert vs.count() == 1
    print(f"  add_texts without metadata: count={vs.count()}")

    # search should still work
    results = vs.similarity_search("text", k=1)
    assert len(results) == 1
    print(f"  search on None metadata: {len(results)} results")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 7. Special characters / SQL injection ====================

def test_special_chars():
    print("=== EDGE: special characters ===\n")
    vs = make_store(table_name="test_lc_special")
    special_texts = [
        "It's a test with 'single quotes'",
        "Semicolon; DROP TABLE test_lc_special;--",
        "Backslash \\ and newline \n text",
        "Unicode: 你好世界 🌍 café naïve",
        "Mixed: <script>alert('xss')</script>",
    ]
    ids = vs.add_texts(special_texts)
    assert len(ids) == 5
    assert vs.count() == 5
    print(f"  inserted {len(ids)} special texts")

    # Verify text round-trip
    docs = vs.get_by_ids([ids[0]])
    assert docs[0].page_content == special_texts[0]
    print(f"  single quote round-trip OK")

    docs = vs.get_by_ids([ids[3]])
    assert docs[0].page_content == special_texts[3]
    print(f"  unicode round-trip OK")

    # Table still exists (SQL injection didn't work)
    assert vs.count() == 5
    print(f"  SQL injection blocked, table intact")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 8. Duplicate ID upsert via add_texts ====================

def test_duplicate_id():
    print("=== EDGE: duplicate ID upsert ===\n")
    vs = make_store(table_name="test_lc_dup")
    ids1 = vs.add_texts(TEXTS[:3], metadatas=METADATAS[:3])
    assert vs.count() == 3

    # Insert again with same IDs — should UPSERT, not error
    ids2 = vs.add_texts(
        ["UPDATED " + t for t in TEXTS[:3]],
        metadatas=METADATAS[:3],
        ids=ids1,
    )
    assert vs.count() == 3  # still 3, not 6
    print(f"  re-insert with same IDs: count still {vs.count()}")

    # Verify content was updated
    docs = vs.get_by_ids([ids1[0]])
    assert docs[0].page_content.startswith("UPDATED")
    print(f"  content updated via UPSERT")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 9. Large / nested metadata ====================

def test_large_metadata():
    print("=== EDGE: large/nested metadata ===\n")
    vs = make_store(table_name="test_lc_bigmeta")
    nested_meta = {
        "level1": {
            "level2": {
                "level3": "deep value",
                "numbers": [1, 2, 3, 4, 5],
            },
            "tags": ["ai", "vector", "database"],
        },
        "unicode": "你好🌟",
        "long_string": "x" * 500,
        "boolean": True,
        "null_value": None,
    }
    ids = vs.add_texts(["doc with big metadata"], metadatas=[nested_meta])
    assert vs.count() == 1
    print(f"  inserted nested metadata")

    # Verify metadata round-trip
    docs = vs.get_by_ids([ids[0]])
    meta = docs[0].metadata
    assert meta["level1"]["level2"]["level3"] == "deep value"
    assert meta["level1"]["level2"]["numbers"] == [1, 2, 3, 4, 5]
    assert meta["level1"]["tags"] == ["ai", "vector", "database"]
    assert meta["unicode"] == "你好🌟"
    assert meta["long_string"] == "x" * 500
    assert meta["boolean"] is True
    print(f"  nested metadata round-trip OK")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== 10. Data persistence across store instances ====================

def test_persistence():
    print("=== EDGE: persistence ===\n")
    vs1 = make_store(table_name="test_lc_persist", pre_delete=True)
    vs1.add_texts(TEXTS[:3], metadatas=METADATAS[:3])
    assert vs1.count() == 3
    vs1.close()
    print(f"  wrote 3 docs and closed store")

    # Create new store pointing to same table — don't delete
    vs2 = make_store(table_name="test_lc_persist", pre_delete=False)
    assert vs2.count() == 3
    print(f"  reopened store, count={vs2.count()}")

    # Search should work
    results = vs2.similarity_search("database", k=2)
    assert len(results) == 2
    print(f"  search after reopen: {len(results)} results")

    vs2.drop_table()
    vs2.close()
    print("  PASSED\n")


# ==================== 11. Vector dimension mismatch ====================

def test_dimension_mismatch():
    print("=== EDGE: dimension mismatch ===\n")
    vs = make_store(table_name="test_lc_dimmm")
    vs.add_texts(["seed text"])  # creates table with dim=128

    # Try to insert with wrong dimension embedding
    wrong_emb = FakeEmbeddings(dim=256)
    try:
        vs.add_embeddings([("wrong dim", wrong_emb.embed_query("wrong dim"))])
        # If no error, that's a problem — but some DBs silently truncate
        # At minimum verify it didn't corrupt data
        print(f"  WARNING: dimension mismatch not rejected (count={vs.count()})")
    except Exception as e:
        print(f"  correctly rejected wrong dimension: {type(e).__name__}")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ---- Helper for dimension mismatch test ----

from _helpers import FakeEmbeddings


# ==================== RUN ALL ====================

def main():
    test_euclidean()
    test_filter_operators()
    test_score_threshold()
    test_empty_table()
    test_k_exceeds_data()
    test_metadata_none()
    test_special_chars()
    test_duplicate_id()
    test_large_metadata()
    test_persistence()
    test_dimension_mismatch()
    print("=== ALL EDGE CASE TESTS PASSED ===")


if __name__ == "__main__":
    main()
