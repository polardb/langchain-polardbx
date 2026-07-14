"""Tests for LangChain PolarDBXVectorStore — search & MMR (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, METADATAS, TEXTS, make_store


def _setup(vs):
    """Insert standard test data."""
    vs.add_texts(TEXTS, metadatas=METADATAS)


# ==================== SYNC ====================

def test_sync():
    vs = make_store()
    _setup(vs)

    vec = EMB.embed_query("database")

    # similarity_search
    results = vs.similarity_search("database", k=3)
    assert len(results) == 3

    # similarity_search_with_score
    results = vs.similarity_search_with_score("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)

    # similarity_search_by_vector
    results = vs.similarity_search_by_vector(vec, k=2)
    assert len(results) == 2

    # similarity_search_with_score_by_vector
    results = vs.similarity_search_with_score_by_vector(vec, k=2)
    assert len(results) == 2
    assert all(isinstance(s, float) for _, s in results)

    # similarity_search_with_relevance_scores
    results = vs.similarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3
    assert all(isinstance(s, float) for _, s in results)

    # similarity_search with filter
    results = vs.similarity_search("language", k=3, filter={"category": "language"})
    assert all(d.metadata.get("category") == "language" for d in results)

    # max_marginal_relevance_search
    results = vs.max_marginal_relevance_search("database", k=3, fetch_k=5)
    assert len(results) == 3

    # max_marginal_relevance_search_by_vector
    results = vs.max_marginal_relevance_search_by_vector(vec, k=2, fetch_k=5)
    assert len(results) == 2

    # search (dispatcher)
    results = vs.search("database", search_type="similarity", k=2)
    assert len(results) == 2

    # search_by_metadata
    results = vs.search_by_metadata(filter={"category": "language"}, limit=5)
    assert len(results) >= 1
    assert all(d.metadata.get("category") == "language" for d in results)

    vs.drop_table()
    vs.close()


# ==================== ASYNC ====================

async def test_async():
    vs = make_store()
    await vs.aadd_texts(TEXTS, metadatas=METADATAS)

    vec = EMB.embed_query("database")

    # asimilarity_search
    results = await vs.asimilarity_search("database", k=3)
    assert len(results) == 3

    # asimilarity_search_with_score
    results = await vs.asimilarity_search_with_score("database", k=3)
    assert len(results) == 3

    # asimilarity_search_by_vector
    results = await vs.asimilarity_search_by_vector(vec, k=2)
    assert len(results) == 2

    # asimilarity_search_with_score_by_vector
    results = await vs.asimilarity_search_with_score_by_vector(vec, k=2)
    assert len(results) == 2

    # asimilarity_search_with_relevance_scores
    results = await vs.asimilarity_search_with_relevance_scores("database", k=3)
    assert len(results) == 3

    # amax_marginal_relevance_search
    results = await vs.amax_marginal_relevance_search("database", k=3, fetch_k=5)
    assert len(results) == 3

    # amax_marginal_relevance_search_by_vector
    results = await vs.amax_marginal_relevance_search_by_vector(vec, k=2, fetch_k=5)
    assert len(results) == 2

    # asearch
    results = await vs.asearch("database", search_type="similarity", k=2)
    assert len(results) == 2

    vs.drop_table()
    await vs.aclose()
