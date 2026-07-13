"""Tests for LangChain PolarDBXVectorStore — index management & enhanced search (sync + async)."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, TEXTS, METADATAS, make_store


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
    print("=== INDEX SYNC ===\n")

    # apply_vector_index
    print("--- apply_vector_index ---")
    old = _drop_existing_vi(vs)
    print(f"  Dropped default index '{old}'")
    vs.apply_vector_index(index_name="vi_test", m=8)
    print("  OK: created 'vi_test'\n")

    # get_stats
    print("--- get_stats ---")
    stats = vs.get_stats()
    assert isinstance(stats, dict)
    print(f"  OK: {len(stats)} Vidx stats\n")

    # ef_search
    print("--- ef_search=50 ---")
    docs = vs.similarity_search("database", k=3, ef_search=50)
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # search_type=knn
    print("--- search_type=knn ---")
    docs = vs.similarity_search("database", k=3, search_type="knn")
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # search_type=ann
    print("--- search_type=ann ---")
    docs = vs.similarity_search("database", k=3, search_type="ann")
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # ef_search + search_type combined
    print("--- ef_search=80 + search_type=ann ---")
    docs = vs.similarity_search("database", k=3, ef_search=80, search_type="ann")
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # similarity_search_with_score with ef_search
    print("--- similarity_search_with_score + ef_search=100 ---")
    results = vs.similarity_search_with_score("database", k=3, ef_search=100)
    assert len(results) <= 3
    for _, score in results:
        assert isinstance(score, float)
    print(f"  OK: {len(results)} pairs\n")

    # MMR with score
    print("--- max_marginal_relevance_search_with_score_by_vector ---")
    emb = EMB.embed_query("database")
    mmr = vs.max_marginal_relevance_search_with_score_by_vector(
        embedding=emb, k=3, fetch_k=10, lambda_mult=0.5
    )
    assert len(mmr) <= 3
    for _, score in mmr:
        assert isinstance(score, float)
    print(f"  OK: {len(mmr)} pairs\n")

    # optimize
    print("--- optimize ---")
    vs.optimize()
    print("  OK\n")

    # drop_vector_index
    print("--- drop_vector_index ---")
    vs.drop_vector_index(index_name="vi_test")
    print("  OK: dropped 'vi_test'\n")

    vs.drop_table()
    vs.close()
    print("=== INDEX SYNC PASSED ===\n")


# ==================== ASYNC ====================

async def test_async():
    vs = make_store()
    await vs.aadd_texts(TEXTS, metadatas=METADATAS)
    print("=== INDEX ASYNC ===\n")

    # aapply_vector_index
    print("--- aapply_vector_index ---")
    old = vs._detect_vector_index_name()
    if old:
        await vs.adrop_vector_index(index_name=old)
        print(f"  Dropped default index '{old}'")
    await vs.aapply_vector_index(index_name="vi_async", m=8)
    print("  OK: created 'vi_async'\n")

    # aget_stats
    print("--- aget_stats ---")
    stats = await vs.aget_stats()
    assert isinstance(stats, dict)
    print(f"  OK: {len(stats)} stats\n")

    # asimilarity_search with ef_search
    print("--- asimilarity_search + ef_search=50 ---")
    docs = await vs.asimilarity_search("database", k=3, ef_search=50)
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # asimilarity_search with search_type=knn
    print("--- asimilarity_search + search_type=knn ---")
    docs = await vs.asimilarity_search("database", k=3, search_type="knn")
    assert len(docs) <= 3
    print(f"  OK: {len(docs)} docs\n")

    # asimilarity_search_with_score + ef_search
    print("--- asimilarity_search_with_score + ef_search=100 ---")
    results = await vs.asimilarity_search_with_score("database", k=3, ef_search=100)
    assert len(results) <= 3
    print(f"  OK: {len(results)} pairs\n")

    # amax_marginal_relevance_search_with_score_by_vector
    print("--- amax_marginal_relevance_search_with_score_by_vector ---")
    emb = EMB.embed_query("database")
    mmr = await vs.amax_marginal_relevance_search_with_score_by_vector(
        embedding=emb, k=3, fetch_k=10, lambda_mult=0.5
    )
    assert len(mmr) <= 3
    print(f"  OK: {len(mmr)} pairs\n")

    # aoptimize
    print("--- aoptimize ---")
    await vs.aoptimize()
    print("  OK\n")

    # adrop_vector_index
    print("--- adrop_vector_index ---")
    await vs.adrop_vector_index(index_name="vi_async")
    print("  OK: dropped 'vi_async'\n")

    vs.drop_table()
    await vs.aclose()
    print("=== INDEX ASYNC PASSED ===\n")


# ==================== ENHANCED: ef_search boundary values ====================

def test_ef_search_boundary():
    """ef_search at extreme values — 1 (minimum) and 10000 (very large)."""
    print("=== INDEX: ef_search boundary ===\n")
    vs = make_store(table_name="test_lc_idx_efbd")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    old = _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_efbd", m=8)
    print(f"  Dropped '{old}', created 'vi_efbd'")

    # ef_search=1 — minimum, should still return results
    docs = vs.similarity_search("database", k=3, ef_search=1)
    assert len(docs) <= 3
    print(f"  ef_search=1: {len(docs)} docs")

    # ef_search=10000 — very large, should work fine
    docs = vs.similarity_search("database", k=3, ef_search=10000)
    assert len(docs) <= 3
    print(f"  ef_search=10000: {len(docs)} docs")

    # with score
    results = vs.similarity_search_with_score("database", k=3, ef_search=1)
    assert len(results) <= 3
    print(f"  ef_search=1 with_score: {len(results)} pairs")

    results = vs.similarity_search_with_score("database", k=3, ef_search=10000)
    assert len(results) <= 3
    print(f"  ef_search=10000 with_score: {len(results)} pairs")

    vs.drop_vector_index(index_name="vi_efbd")
    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== ENHANCED: drop_vector_index auto-detect ====================

def test_drop_vector_index_auto_detect():
    """drop_vector_index without index_name should auto-detect the existing index."""
    print("=== INDEX: drop_vector_index auto-detect ===\n")
    vs = make_store(table_name="test_lc_idx_autodrop")
    vs.add_texts(TEXTS, metadatas=METADATAS)

    # The table was created with a default vector index
    detected = vs._detect_vector_index_name()
    assert detected is not None, "Should detect default vector index"
    print(f"  Detected default index: '{detected}'")

    # Drop without passing index_name — should auto-detect
    vs.drop_vector_index()  # no index_name
    print(f"  Dropped via auto-detect")

    # Verify index is gone
    detected_after = vs._detect_vector_index_name()
    assert detected_after is None, "Index should be gone after drop"
    print(f"  Verified: no index detected after drop")

    # Re-apply and drop again to confirm
    vs.apply_vector_index(index_name="vi_recreated", m=8)
    detected2 = vs._detect_vector_index_name()
    assert detected2 == "vi_recreated"
    print(f"  Re-applied as 'vi_recreated', detected: '{detected2}'")

    vs.drop_vector_index()  # auto-detect again
    assert vs._detect_vector_index_name() is None
    print(f"  Auto-detect drop again OK")

    vs.drop_table()
    vs.close()
    print("  PASSED\n")


# ==================== ENHANCED: apply_vector_index with different M values ====================

def test_apply_vector_index_variants():
    """apply_vector_index with different M values and distance strategies."""
    print("=== INDEX: apply_vector_index variants ===\n")

    # M=4
    vs = make_store(table_name="test_lc_idx_m4")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    old = _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m4", m=4)
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m4"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    print(f"  M=4: index='{detected}', search returned {len(docs)} docs")
    vs.drop_vector_index(index_name="vi_m4")
    vs.drop_table()
    vs.close()

    # M=16
    vs = make_store(table_name="test_lc_idx_m16")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    old = _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m16", m=16)
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m16"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    print(f"  M=16: index='{detected}', search returned {len(docs)} docs")
    vs.drop_vector_index(index_name="vi_m16")
    vs.drop_table()
    vs.close()

    # M=32 with distance=EUCLIDEAN
    vs = make_store(table_name="test_lc_idx_m32", distance_strategy="euclidean")
    vs.add_texts(TEXTS, metadatas=METADATAS)
    old = _drop_existing_vi(vs)
    vs.apply_vector_index(index_name="vi_m32", m=32, distance="EUCLIDEAN")
    detected = vs._detect_vector_index_name()
    assert detected == "vi_m32"
    docs = vs.similarity_search("database", k=3)
    assert len(docs) <= 3
    print(f"  M=32 EUCLIDEAN: index='{detected}', search returned {len(docs)} docs")
    vs.drop_vector_index(index_name="vi_m32")
    vs.drop_table()
    vs.close()

    print("  PASSED\n")


def main():
    test_sync()
    asyncio.run(test_async())
    test_ef_search_boundary()
    test_drop_vector_index_auto_detect()
    test_apply_vector_index_variants()
    print("=== ALL INDEX TESTS PASSED ===")


if __name__ == "__main__":
    main()
