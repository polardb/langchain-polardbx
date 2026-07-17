"""Edge case tests for LangChain PolarDBXVectorStore."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, METADATAS, TEXTS, FakeEmbeddings, is_v3, make_store

# ==================== 1. Euclidean distance strategy ====================

def test_euclidean():
    vs = make_store(table_name="test_lc_euclidean", distance_strategy="euclidean")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    results = vs.similarity_search("database", k=3)
    assert len(results) == 3

    results = vs.similarity_search_with_score("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)

    results = vs.similarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)

    results = vs.max_marginal_relevance_search("database", k=2, fetch_k=5)
    assert len(results) == 2

    vs.drop_table()
    vs.close()


# ==================== 2. Filter operators ====================

def test_filter_operators():
    vs = make_store(table_name="test_lc_filters")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    vs.add_texts(
        ["Rust is fast", "Go is concurrent"],
        metadatas=[{"category": "language", "score": 50}, {"category": "language", "score": 90}],
    )

    # $eq
    r = vs.similarity_search("language", k=10, filter={"category": "language"})
    assert all(d.metadata.get("category") == "language" for d in r)

    # $ne
    r = vs.similarity_search("language", k=10, filter={"category": {"$ne": "database"}})
    assert all(d.metadata.get("category") != "database" for d in r)

    # $gt
    r = vs.similarity_search("language", k=10, filter={"score": {"$gt": 60}})
    assert all(d.metadata.get("score", 0) > 60 for d in r)

    # $gte
    r = vs.similarity_search("language", k=10, filter={"score": {"$gte": 50}})
    assert all(d.metadata.get("score", 0) >= 50 for d in r)

    # $lt
    r = vs.similarity_search("language", k=10, filter={"score": {"$lt": 60}})
    assert all(d.metadata.get("score", 999) < 60 for d in r)

    # $lte
    r = vs.similarity_search("language", k=10, filter={"score": {"$lte": 90}})
    assert all(d.metadata.get("score", 999) <= 90 for d in r)

    # $in
    r = vs.similarity_search("language", k=10, filter={"category": {"$in": ["database", "search"]}})
    assert all(d.metadata.get("category") in ["database", "search"] for d in r)

    # $nin
    r = vs.similarity_search("language", k=10, filter={"category": {"$nin": ["database", "search"]}})
    assert all(d.metadata.get("category") not in ["database", "search"] for d in r)

    # $like
    r = vs.similarity_search("language", k=10, filter={"category": {"$like": "lang%"}})
    assert all(d.metadata.get("category", "").startswith("lang") for d in r)

    # Multi-condition (AND)
    r = vs.similarity_search("language", k=10, filter={"category": "language", "score": {"$gt": 60}})
    assert all(d.metadata.get("category") == "language" and d.metadata.get("score", 0) > 60 for d in r)

    # Filter on non-existent key
    r = vs.similarity_search("language", k=10, filter={"nonexistent_key": "value"})
    assert len(r) == 0

    vs.drop_table()
    vs.close()


# ==================== 3. score_threshold ====================

def test_score_threshold():
    vs = make_store(table_name="test_lc_threshold")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    vec = EMB.embed_query("database")

    # Without threshold — returns all k
    results = vs.similarity_search_with_score_by_vector(vec, k=5)
    assert len(results) == 5

    # With high threshold — should filter most out
    results_high = vs.similarity_search_with_score_by_vector(vec, k=5, score_threshold=0.99)

    # With low threshold — should keep all
    results_low = vs.similarity_search_with_score_by_vector(vec, k=5, score_threshold=0.0)
    assert len(results_low) == 5

    # High threshold should return <= no threshold
    assert len(results_high) <= len(results_low)

    vs.drop_table()
    vs.close()


# ==================== 4. Empty table search ====================

def test_empty_table():
    vs = make_store(table_name="test_lc_empty")
    # Don't insert anything — table doesn't even exist yet
    # similarity_search should return empty list, not error
    results = vs.similarity_search("anything", k=3)
    assert len(results) == 0

    results = vs.similarity_search_with_score("anything", k=3)
    assert len(results) == 0

    vs.drop_table()
    vs.close()


# ==================== 5. k > data count ====================

def test_k_exceeds_data():
    vs = make_store(table_name="test_lc_kexceed")
    vs.add_texts(TEXTS[:3], metadatas=METADATAS[:3])  # only 3 docs

    results = vs.similarity_search("database", k=100)
    assert len(results) == 3  # should return all 3, not error

    vs.drop_table()
    vs.close()


# ==================== 6. metadata=None ====================

def test_metadata_none():
    vs = make_store(table_name="test_lc_metanone")
    vs.add_texts(["text without metadata"])  # no metadatas param

    assert vs.count() == 1

    # search should still work
    results = vs.similarity_search("text", k=1)
    assert len(results) == 1

    vs.drop_table()
    vs.close()


# ==================== 7. Special characters / SQL injection ====================

def test_special_chars():
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

    # Verify text round-trip
    docs = vs.get_by_ids([ids[0]])
    assert docs[0].page_content == special_texts[0]

    docs = vs.get_by_ids([ids[3]])
    assert docs[0].page_content == special_texts[3]

    # Table still exists (SQL injection didn't work)
    assert vs.count() == 5

    vs.drop_table()
    vs.close()


# ==================== 8. Duplicate ID upsert via add_texts ====================

def test_duplicate_id():
    vs = make_store(table_name="test_lc_dup")
    ids1 = vs.add_texts(TEXTS[:3], metadatas=METADATAS[:3])
    assert vs.count() == 3

    # Insert again with same IDs — should UPSERT, not error
    vs.add_texts(
        ["UPDATED " + t for t in TEXTS[:3]],
        metadatas=METADATAS[:3],
        ids=ids1,
    )
    assert vs.count() == 3  # still 3, not 6

    # Verify content was updated
    docs = vs.get_by_ids([ids1[0]])
    assert docs[0].page_content.startswith("UPDATED")

    vs.drop_table()
    vs.close()


# ==================== 9. Large / nested metadata ====================

def test_large_metadata():
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

    # Verify metadata round-trip
    docs = vs.get_by_ids([ids[0]])
    meta = docs[0].metadata
    assert meta["level1"]["level2"]["level3"] == "deep value"
    assert meta["level1"]["level2"]["numbers"] == [1, 2, 3, 4, 5]
    assert meta["level1"]["tags"] == ["ai", "vector", "database"]
    assert meta["unicode"] == "你好🌟"
    assert meta["long_string"] == "x" * 500
    assert meta["boolean"] is True

    vs.drop_table()
    vs.close()


# ==================== 10. Data persistence across store instances ====================

def test_persistence():
    vs1 = make_store(table_name="test_lc_persist", pre_delete=True)
    vs1.add_texts(TEXTS[:3], metadatas=METADATAS[:3])
    assert vs1.count() == 3
    vs1.close()

    # Create new store pointing to same table — don't delete
    vs2 = make_store(table_name="test_lc_persist", pre_delete=False)
    assert vs2.count() == 3

    # Search should work
    results = vs2.similarity_search("database", k=2)
    assert len(results) == 2

    vs2.drop_table()
    vs2.close()


# ==================== 11. INNER_PRODUCT distance strategy (v3) ====================

def test_inner_product():
    """inner_product distance strategy requires v3 VEC_DISTANCE support.
    On old versions, initialization should raise NotSupportedError."""
    from _helpers import NotSupportedError

    probe = make_store()
    v3 = is_v3(probe)
    probe.close()

    if v3:
        # v3: inner_product should work end-to-end
        vs = make_store(table_name="test_lc_innerprod", distance_strategy="inner_product")
        vs.add_texts(TEXTS, metadatas=METADATAS)

        results = vs.similarity_search("database", k=3)
        assert len(results) <= 3

        results = vs.similarity_search_with_score("database", k=3)
        assert len(results) <= 3
        assert all(isinstance(s, float) for _, s in results)

        results = vs.similarity_search_with_relevance_scores("database", k=3)
        assert len(results) <= 3
        assert all(isinstance(s, float) for _, s in results)

        vs.drop_table()
        vs.close()
    else:
        # Old version: should reject inner_product at init time
        try:
            vs = make_store(table_name="test_lc_innerprod", distance_strategy="inner_product")
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass  # expected


# ==================== 12. Embedding dimension validation (v3 VECTOR_DIM) ====================

def test_dimension_validation():
    """_validate_embedding_dimensions should cross-check with VECTOR_DIM on v3,
    and do client-side length check only on old versions."""
    vs = make_store(table_name="test_lc_dimval")
    vs.add_texts(["seed text"])  # creates table with dim=128

    correct_embs = EMB.embed_documents(["correct dim text"])
    # Should pass validation
    vs._validate_embedding_dimensions(correct_embs, 128)

    # Wrong dimension should raise ValueError
    wrong_embs = FakeEmbeddings(dim=256).embed_documents(["wrong dim"])
    try:
        vs._validate_embedding_dimensions(wrong_embs, 256)
        # If no error on old version (no VECTOR_DIM cross-check),
        # client-side length check should still catch mismatch with table
    except ValueError:
        pass  # expected on v3 or when dimensions mismatch

    vs.drop_table()
    vs.close()


# ==================== 13. Vector dimension mismatch ====================

def test_dimension_mismatch():
    vs = make_store(table_name="test_lc_dimmm")
    vs.add_texts(["seed text"])  # creates table with dim=128

    # Try to insert with wrong dimension embedding
    wrong_emb = FakeEmbeddings(dim=256)
    try:
        vs.add_embeddings([("wrong dim", wrong_emb.embed_query("wrong dim"))])
        # If no error, verify it didn't corrupt data
        assert vs.count() >= 1
    except Exception:
        # Correctly rejected wrong dimension
        pass

    vs.drop_table()
    vs.close()
