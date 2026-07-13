"""Tests for LangChain PolarDBXVectorStore — basic CRUD operations (sync + async)."""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import EMB, TEXTS, METADATAS, make_store
from langchain_core.documents import Document


# ==================== SYNC ====================

def test_sync():
    vs = make_store()
    print("=== BASIC SYNC ===\n")

    # add_texts
    print("--- add_texts ---")
    ids = vs.add_texts(TEXTS, metadatas=METADATAS)
    assert len(ids) == 5
    print(f"  OK: inserted {len(ids)} texts\n")

    # add_documents
    print("--- add_documents ---")
    docs = [Document(page_content="Java is also popular", metadata={"category": "language", "lang": "en"})]
    doc_ids = vs.add_documents(docs)
    assert len(doc_ids) == 1
    assert vs.count() == 6
    print(f"  OK: added 1 doc, total={vs.count()}\n")

    # add_embeddings
    print("--- add_embeddings ---")
    emb_pairs = [("Go is concise", EMB.embed_query("Go is concise"))]
    emb_ids = vs.add_embeddings(emb_pairs, metadatas=[{"category": "language", "lang": "en"}])
    assert len(emb_ids) == 1
    assert vs.count() == 7
    print(f"  OK: added 1 embedding, total={vs.count()}\n")

    # embeddings property
    print("--- embeddings property ---")
    assert vs.embeddings is EMB
    print("  OK\n")

    # count
    print("--- count ---")
    assert vs.count() == 7
    print(f"  OK: count={vs.count()}\n")

    # exists
    print("--- exists ---")
    assert vs.exists(ids[0]) is True
    assert vs.exists("nonexistent-id") is False
    print(f"  OK\n")

    # get_by_ids
    print("--- get_by_ids ---")
    docs = vs.get_by_ids([ids[0], ids[1]])
    assert len(docs) == 2
    print(f"  OK: fetched {len(docs)} docs\n")

    # upsert
    print("--- upsert ---")
    vs.upsert([Document(page_content="UPDATED text", metadata={"category": "updated", "lang": "en"})], ids=[ids[0]])
    docs = vs.get_by_ids([ids[0]])
    assert docs[0].page_content == "UPDATED text"
    assert docs[0].metadata["category"] == "updated"
    print("  OK: upsert verified\n")

    # delete
    print("--- delete ---")
    vs.delete([ids[2]])
    assert not vs.exists(ids[2])
    print(f"  OK: deleted {ids[2]}\n")

    # delete_by_metadata
    print("--- delete_by_metadata ---")
    deleted = vs.delete_by_metadata({"category": "language"})
    assert deleted >= 1
    from _helpers import PolarDBXVectorStore
    results = vs.search_by_metadata(filter={"category": "language"}, limit=10)
    assert len(results) == 0
    print(f"  OK: deleted {deleted} by metadata\n")

    # clear
    print("--- clear ---")
    vs.clear()
    assert vs.count() == 0
    print(f"  OK: count after clear={vs.count()}\n")

    # from_texts
    print("--- from_texts ---")
    vs.drop_table()
    vs2 = make_store()
    from _helpers import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
    vs2 = PolarDBXVectorStore.from_texts(
        TEXTS[:3], EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs2.count() == 3
    print(f"  OK: from_texts count={vs2.count()}\n")
    vs2.drop_table()
    vs2.close()

    # from_documents
    print("--- from_documents ---")
    docs = [Document(page_content=t, metadata=m) for t, m in zip(TEXTS[:3], METADATAS[:3])]
    vs3 = PolarDBXVectorStore.from_documents(
        docs, EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs3.count() == 3
    print(f"  OK: from_documents count={vs3.count()}\n")
    vs3.drop_table()
    vs3.close()

    # drop_table + close
    print("--- drop_table + close ---")
    vs4 = make_store()
    vs4.add_texts(["temp"])
    vs4.drop_table()
    vs4.close()
    print("  OK\n")

    print("=== BASIC SYNC PASSED ===\n")


# ==================== ASYNC ====================

async def test_async():
    vs = make_store()
    print("=== BASIC ASYNC ===\n")

    # aadd_texts
    print("--- aadd_texts ---")
    ids = await vs.aadd_texts(TEXTS, metadatas=METADATAS)
    assert len(ids) == 5
    print(f"  OK: inserted {len(ids)}\n")

    # aadd_documents
    print("--- aadd_documents ---")
    docs = [Document(page_content="Java is also popular", metadata={"category": "language", "lang": "en"})]
    doc_ids = await vs.aadd_documents(docs)
    assert len(doc_ids) == 1
    print(f"  OK\n")

    # aadd_embeddings
    print("--- aadd_embeddings ---")
    emb_pairs = [("Go is concise", EMB.embed_query("Go is concise"))]
    emb_ids = await vs.aadd_embeddings(emb_pairs, metadatas=[{"category": "language", "lang": "en"}])
    assert len(emb_ids) == 1
    print(f"  OK\n")

    # acount
    print("--- acount ---")
    cnt = await vs.acount()
    assert cnt == 7
    print(f"  OK: count={cnt}\n")

    # aget_by_ids
    print("--- aget_by_ids ---")
    docs = await vs.aget_by_ids([ids[0], ids[1]])
    assert len(docs) == 2
    print(f"  OK\n")

    # aupsert
    print("--- aupsert ---")
    await vs.aupsert([Document(page_content="ASYNC UPDATED", metadata={"category": "updated", "lang": "en"})], ids=[ids[0]])
    docs = await vs.aget_by_ids([ids[0]])
    assert docs[0].page_content == "ASYNC UPDATED"
    print("  OK\n")

    # adelete
    print("--- adelete ---")
    await vs.adelete([ids[2]])
    assert not vs.exists(ids[2])
    print(f"  OK: deleted {ids[2]}\n")

    # aclear
    print("--- aclear ---")
    await vs.aclear()
    cnt = await vs.acount()
    assert cnt == 0
    print(f"  OK: count after aclear={cnt}\n")

    # afrom_texts
    print("--- afrom_texts ---")
    from _helpers import DB_HOST, DB_PORT, DB_USER, DB_PASS, DB_NAME
    from langchain_polardbx import PolarDBXVectorStore
    vs2 = await PolarDBXVectorStore.afrom_texts(
        TEXTS[:3], EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs2.count() == 3
    print(f"  OK: afrom_texts count={vs2.count()}\n")
    vs2.drop_table()
    vs2.close()

    # afrom_documents
    print("--- afrom_documents ---")
    docs = [Document(page_content=t, metadata=m) for t, m in zip(TEXTS[:3], METADATAS[:3])]
    vs3 = await PolarDBXVectorStore.afrom_documents(
        docs, EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs3.count() == 3
    print(f"  OK: afrom_documents count={vs3.count()}\n")
    vs3.drop_table()
    vs3.close()

    # aclose
    print("--- aclose ---")
    vs4 = make_store()
    await vs4.aclose()
    print("  OK\n")

    print("=== BASIC ASYNC PASSED ===\n")


def main():
    test_sync()
    asyncio.run(test_async())
    print("=== ALL BASIC TESTS PASSED ===")


if __name__ == "__main__":
    main()
