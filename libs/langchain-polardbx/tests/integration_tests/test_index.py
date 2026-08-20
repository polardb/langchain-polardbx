"""Tests for LangChain PolarDBXVectorStore — index management & enhanced search (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, METADATAS, TEXTS, is_v3, make_store


def _drop_existing_vi(vs):
    """Drop the default vector index created with the table."""
    existing = vs._detect_vector_index_name()
    if existing:
        vs.drop_vector_index(index_name=existing)
        return existing
    return None


# ==================== SYNC ====================


def test_sync():
    vs = make_store()
    vs.add_texts(TEXTS, metadatas=METADATAS)

    # apply_vector_index
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_test", m=8)

    # get_stats
    stats = vs.get_stats()
    assert isinstance(stats, dict)

    # ef_search
    docs = vs.similarity_search("database", k=3, ef_search=50)
    assert len(docs) <= 3

    # search_type=knn
    docs = vs.similarity_search("database", k=3, search_type="knn")
    assert len(docs) <= 3

    # search_type=ann
    docs = vs.similarity_search("database", k=3, search_type="ann")
    assert len(docs) <= 3

    # ef_search + search_type combined
    docs = vs.similarity_search("database", k=3, ef_search=80, search_type="ann")
    assert len(docs) <= 3

    # similarity_search_with_score with ef_search
    results = vs.similarity_search_with_score("database", k=3, ef_search=100)
    assert len(results) <= 3
    for _, score in results:
        assert isinstance(score, float)

    # MMR with score
    emb = EMB.embed_query("database")
    mmr = vs.max_marginal_relevance_search_with_score_by_vector(
        embedding=emb, k=3, fetch_k=10, lambda_mult=0.5
    )
    assert len(mmr) <= 3
    for _, score in mmr:
        assert isinstance(score, float)

    # optimize
    vs.optimize()

    # drop_vector_index
    vs.drop_vector_index(index_name="vi_test")

    vs.drop_table()
    vs.close()


# ==================== ASYNC ====================


async def test_async():
    vs = make_store()
    await vs.aadd_texts(TEXTS, metadatas=METADATAS)

    # aapply_vector_index
    old = vs._detect_vector_index_name()
    if old:
        await vs.adrop_vector_index(index_name=old)
    await vs.aapply_vector_index(index_name="vi_async", m=8)

    # aget_stats
    stats = await vs.aget_stats()
    assert isinstance(stats, dict)

    # asimilarity_search with ef_search
    docs = await vs.asimilarity_search("database", k=3, ef_search=50)
    assert len(docs) <= 3

    # asimilarity_search with search_type=knn
    docs = await vs.asimilarity_search("database", k=3, search_type="knn")
    assert len(docs) <= 3

    # asimilarity_search_with_score + ef_search
    results = await vs.asimilarity_search_with_score("database", k=3, ef_search=100)
    assert len(results) <= 3

    # amax_marginal_relevance_search_with_score_by_vector
    emb = EMB.embed_query("database")
    mmr = await vs.amax_marginal_relevance_search_with_score_by_vector(
        embedding=emb, k=3, fetch_k=10, lambda_mult=0.5
    )
    assert len(mmr) <= 3

    # aoptimize
    await vs.aoptimize()

    # adrop_vector_index
    await vs.adrop_vector_index(index_name="vi_async")

    vs.drop_table()
    await vs.aclose()


# ==================== ENHANCED: ef_search boundary values ====================


def test_ef_search_boundary():
    """ef_search at extreme values — 1 (minimum) and 10000 (very large)."""
    vs = make_store(table_name="test_lc_idx_efbd")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_efbd", m=8)

    # ef_search=1 — minimum, should still return results
    docs = vs.similarity_search("database", k=3, ef_search=1)
    assert len(docs) <= 3

    # ef_search=10000 — very large, should work fine
    docs = vs.similarity_search("database", k=3, ef_search=10000)
    assert len(docs) <= 3

    # with score
    results = vs.similarity_search_with_score("database", k=3, ef_search=1)
    assert len(results) <= 3

    results = vs.similarity_search_with_score("database", k=3, ef_search=10000)
    assert len(results) <= 3

    vs.drop_vector_index(index_name="vi_efbd")
    vs.drop_table()
    vs.close()


# ==================== ENHANCED: drop_vector_index auto-detect ====================


def test_drop_vector_index_auto_detect():
    """drop_vector_index without index_name should auto-detect the existing index."""
    vs = make_store(table_name="test_lc_idx_autodrop")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    # The table was created with a default vector index
    detected = vs._detect_vector_index_name()
    assert detected is not None, "Should detect default vector index"

    # Drop without passing index_name — should auto-detect
    vs.drop_vector_index()  # no index_name

    # Verify index is gone
    detected_after = vs._detect_vector_index_name()
    assert detected_after is None, "Index should be gone after drop"

    # Re-apply and drop again to confirm
    vs.apply_vector_index(index_name="vi_recreated", m=8)
    detected2 = vs._detect_vector_index_name()
    assert detected2 == "vi_recreated"

    vs.drop_vector_index()  # auto-detect again
    assert vs._detect_vector_index_name() is None

    vs.drop_table()
    vs.close()


# ==================== ENHANCED: apply_vector_index with different M values ====================


def test_apply_vector_index_variants():
    """apply_vector_index with different M values and distance strategies."""
    # M=4
    vs = make_store(table_name="test_lc_idx_m4")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m4", m=4)
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m4"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    vs.drop_vector_index(index_name="vi_m4")
    vs.drop_table()
    vs.close()

    # M=16
    vs = make_store(table_name="test_lc_idx_m16")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m16", m=16)
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m16"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    vs.drop_vector_index(index_name="vi_m16")
    vs.drop_table()
    vs.close()

    # M=32 with distance=EUCLIDEAN
    vs = make_store(table_name="test_lc_idx_m32", distance_strategy="euclidean")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m32", m=32, distance="EUCLIDEAN")
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m32"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    vs.drop_vector_index(index_name="vi_m32")
    vs.drop_table()
    vs.close()


# ==================== v3: EF_CONSTRUCTION in CREATE TABLE DDL ====================


def test_ef_construction_in_ddl():
    """ef_construction parameter should appear in CREATE TABLE DDL on v3,
    and be silently ignored on old versions."""
    vs = make_store(table_name="test_lc_idx_efcddl", ef_construction=40)
    vs.add_texts(TEXTS[:2], metadatas=METADATAS[:2])

    if is_v3(vs):
        # v3: DDL should contain EF_CONSTRUCTION=40
        with vs._get_cursor() as cursor:
            cursor.execute("SHOW CREATE TABLE `test_lc_idx_efcddl`")
            row = cursor.fetchone()
            create_sql = row.get("Create Table", "") if row else ""
        assert "EF_CONSTRUCTION=40" in create_sql.upper(), (
            f"EF_CONSTRUCTION=40 not found in DDL: {create_sql[-120:]}"
        )
    else:
        # Old version: ef_construction silently ignored, table still works
        assert vs.count() == 2

    vs.drop_table()
    vs.close()


# ==================== v3: EF_CONSTRUCTION in apply_vector_index ====================


def test_apply_vector_index_with_ef_construction():
    """apply_vector_index should accept ef_construction on v3,
    and silently ignore it on old versions."""
    vs = make_store(table_name="test_lc_idx_efcapply")
    vs.add_texts(TEXTS[:2], metadatas=METADATAS[:2])
    _drop_existing_vi(vs)

    if is_v3(vs):
        vs.apply_vector_index(index_name="vi_efc", m=8, ef_construction=64)
        detected = vs._detect_vector_index_name()
        assert detected == "vi_efc"
    else:
        # Old version: ef_construction ignored, index still created
        vs.apply_vector_index(index_name="vi_efc", m=8)
        detected = vs._detect_vector_index_name()
        assert detected == "vi_efc"

    vs.drop_vector_index(index_name="vi_efc")
    vs.drop_table()
    vs.close()


# ==================== v3: VECTOR_INDEXES view for index detection ====================


def test_detect_vector_index_name():
    """_detect_vector_index_name should find the index via VECTOR_INDEXES
    view on v3, or via SHOW CREATE TABLE regex on old versions."""
    vs = make_store(table_name="test_lc_idx_detect")
    vs.add_texts(TEXTS[:1], metadatas=METADATAS[:1])

    # Table was created with a default VECTOR INDEX — should be detectable
    vs._vector_index_name = None  # force re-detection
    detected = vs._detect_vector_index_name()
    assert detected is not None, "Should detect default vector index"

    vs.drop_table()
    vs.close()


# ==================== v3: preload_index / preload_check ====================


def test_preload_index():
    """preload_index should succeed on v3, raise NotSupportedError on old."""
    from langchain_polardbx import NotSupportedError

    vs = make_store(table_name="test_lc_idx_preload")
    vs.add_texts(TEXTS[:1], metadatas=METADATAS[:1])

    if is_v3(vs):
        vs.preload_index()  # should not raise
    else:
        try:
            vs.preload_index()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass  # expected

    vs.drop_table()
    vs.close()


def test_preload_check():
    """preload_check should return a dict on v3, raise NotSupportedError on old."""
    from langchain_polardbx import NotSupportedError

    vs = make_store(table_name="test_lc_idx_plchk")
    vs.add_texts(TEXTS[:1], metadatas=METADATAS[:1])

    if is_v3(vs):
        result = vs.preload_check()
        assert isinstance(result, dict)
    else:
        try:
            vs.preload_check()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass  # expected

    vs.drop_table()
    vs.close()


# ==================== v3: explain_index_health ====================


def test_explain_index_health():
    """explain_index_health should return metadata on v3, raise NotSupportedError on old."""
    from langchain_polardbx import NotSupportedError

    vs = make_store(table_name="test_lc_idx_health")
    vs.add_texts(TEXTS[:1], metadatas=METADATAS[:1])

    if is_v3(vs):
        result = vs.explain_index_health()
        assert "index_info" in result
        assert "explain" in result
        assert result["index_info"] is not None
    else:
        try:
            vs.explain_index_health()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass  # expected

    vs.drop_table()
    vs.close()


# ==================== v3: async preload / explain_index_health ====================


async def test_async_preload_and_health():
    """Async preload_index/preload_check/explain_index_health on v3,
    NotSupportedError on old."""
    from langchain_polardbx import NotSupportedError

    vs = make_store(table_name="test_lc_idx_asyncv3")
    await vs.aadd_texts(TEXTS[:1], metadatas=METADATAS[:1])

    if is_v3(vs):
        await vs.apreload_index()  # should not raise
        result = await vs.apreload_check()
        assert isinstance(result, dict)
        health = await vs.aexplain_index_health()
        assert "index_info" in health
    else:
        try:
            await vs.apreload_index()
            assert False, "Expected NotSupportedError on old version"
        except NotSupportedError:
            pass

    vs.drop_table()
    await vs.aclose()
