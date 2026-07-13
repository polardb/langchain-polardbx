"""Tests for LangChain PolarDBXVectorStore — search & MMR (sync + async)."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, TEXTS, METADATAS, make_store


def _setup(vs):
    """Insert standard test data."""
    vs.add_texts(TEXTS, metadatas=METADATAS)


# ==================== SYNC ====================

def test_sync():
    vs = make_store()
    _setup(vs)
    print("=== SEARCH SYNC ===\n")

    vec = EMB.embed_query("database")

    # similarity_search
    print("--- similarity_search ---")
    results = vs.similarity_search("database", k=3)
    assert len(results) == 3
    print(f"  OK: {len(results)} results\n")

    # similarity_search_with_score
    print("--- similarity_search_with_score ---")
    results = vs.similarity_search_with_score("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)
    print(f"  OK: {len(results)} results with scores\n")

    # similarity_search_by_vector
    print("--- similarity_search_by_vector ---")
    results = vs.similarity_search_by_vector(vec, k=2)
    assert len(results) == 2
    print(f"  OK: {len(results)} results\n")

    # similarity_search_with_score_by_vector
    print("--- similarity_search_with_score_by_vector ---")
    results = vs.similarity_search_with_score_by_vector(vec, k=2)
    assert len(results) == 2
    print(f"  OK: {len(results)} results with scores\n")

    # similarity_search_with_relevance_scores
    print("--- similarity_search_with_relevance_scores ---")
    results = vs.similarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)
    print(f"  OK: {len(results)} results with relevance scores\n")

    # similarity_search with filter
    print("--- similarity_search with filter ---")
    results = vs.similarity_search("language", k=3, filter={"category": "language"})
    assert all(d.metadata.get("category") == "language" for d in results)
    print(f"  OK: filter returned {len(results)} results\n")

    # max_marginal_relevance_search
    print("--- max_marginal_relevance_search ---")
    results = vs.max_marginal_relevance_search("database", k=3, fetch_k=5)
    assert len(results) == 3
    print(f"  OK: {len(results)} MMR results\n")

    # max_marginal_relevance_search_by_vector
    print("--- max_marginal_relevance_search_by_vector ---")
    results = vs.max_marginal_relevance_search_by_vector(vec, k=2, fetch_k=5)
    assert len(results) == 2
    print(f"  OK: {len(results)} MMR results\n")

    # search (dispatcher)
    print("--- search dispatcher ---")
    results = vs.search("database", search_type="similarity", k=2)
    assert len(results) == 2
    print(f"  OK: search(similarity) returned {len(results)}\n")

    # search_by_metadata
    print("--- search_by_metadata ---")
    results = vs.search_by_metadata(filter={"category": "language"}, limit=5)
    assert len(results) >= 1
    assert all(d.metadata.get("category") == "language" for d in results)
    print(f"  OK: {len(results)} results\n")

    vs.drop_table()
    vs.close()
    print("=== SEARCH SYNC PASSED ===\n")


# ==================== ASYNC ====================

async def test_async():
    vs = make_store()
    await vs.aadd_texts(TEXTS, metadatas=METADATAS)
    print("=== SEARCH ASYNC ===\n")

    vec = EMB.embed_query("database")

    # asimilarity_search
    print("--- asimilarity_search ---")
    results = await vs.asimilarity_search("database", k=3)
    assert len(results) == 3
    print(f"  OK: {len(results)} results\n")

    # asimilarity_search_with_score
    print("--- asimilarity_search_with_score ---")
    results = await vs.asimilarity_search_with_score("database", k=3)
    assert len(results) == 3
    print(f"  OK: {len(results)} results with scores\n")

    # asimilarity_search_by_vector
    print("--- asimilarity_search_by_vector ---")
    results = await vs.asimilarity_search_by_vector(vec, k=2)
    assert len(results) == 2
    print(f"  OK: {len(results)} results\n")

    # asimilarity_search_with_score_by_vector
    print("--- asimilarity_search_with_score_by_vector ---")
    results = await vs.asimilarity_search_with_score_by_vector(vec, k=2)
    assert len(results) == 2
    print(f"  OK: {len(results)} results with scores\n")

    # asimilarity_search_with_relevance_scores
    print("--- asimilarity_search_with_relevance_scores ---")
    results = await vs.asimilarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3
    print(f"  OK: {len(results)} results with relevance scores\n")

    # amax_marginal_relevance_search
    print("--- amax_marginal_relevance_search ---")
    results = await vs.amax_marginal_relevance_search("database", k=3, fetch_k=5)
    assert len(results) == 3
    print(f"  OK: {len(results)} MMR results\n")

    # amax_marginal_relevance_search_by_vector
    print("--- amax_marginal_relevance_search_by_vector ---")
    results = await vs.amax_marginal_relevance_search_by_vector(vec, k=2, fetch_k=5)
    assert len(results) == 2
    print(f"  OK: {len(results)} MMR results\n")

    # asearch
    print("--- asearch ---")
    results = await vs.asearch("database", search_type="similarity", k=2)
    assert len(results) == 2
    print(f"  OK: {len(results)} results\n")

    vs.drop_table()
    await vs.aclose()
    print("=== SEARCH ASYNC PASSED ===\n")


def main():
    test_sync()
    asyncio.run(test_async())
    print("=== ALL SEARCH TESTS PASSED ===")


if __name__ == "__main__":
    main()
