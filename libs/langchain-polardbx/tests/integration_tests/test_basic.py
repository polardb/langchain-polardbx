"""Tests for LangChain PolarDBXVectorStore — basic CRUD operations (sync + async)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _helpers import (
    DB_HOST,
    DB_NAME,
    DB_PASS,
    DB_PORT,
    DB_USER,
    EMB,
    METADATAS,
    TEXTS,
    make_store,
)
from langchain_core.documents import Document

from langchain_polardbx import PolarDBXVectorStore

# ==================== SYNC ====================

def test_sync():
    vs = make_store()

    # add_texts
    ids = vs.add_texts(TEXTS, metadatas=METADATAS)
    assert len(ids) == 5

    # add_documents
    docs = [Document(page_content="Java is also popular", metadata={"category": "language", "lang": "en"})]
    doc_ids = vs.add_documents(docs)
    assert len(doc_ids) == 1
    assert vs.count() == 6

    # add_embeddings
    emb_pairs = [("Go is concise", EMB.embed_query("Go is concise"))]
    emb_ids = vs.add_embeddings(emb_pairs, metadatas=[{"category": "language", "lang": "en"}])
    assert len(emb_ids) == 1
    assert vs.count() == 7

    # embeddings property
    assert vs.embeddings is EMB

    # count
    assert vs.count() == 7

    # exists
    assert vs.exists(ids[0]) is True
    assert vs.exists("nonexistent-id") is False

    # get_by_ids
    docs = vs.get_by_ids([ids[0], ids[1]])
    assert len(docs) == 2

    # upsert
    vs.upsert([Document(page_content="UPDATED text", metadata={"category": "updated", "lang": "en"})], ids=[ids[0]])
    docs = vs.get_by_ids([ids[0]])
    assert docs[0].page_content == "UPDATED text"
    assert docs[0].metadata["category"] == "updated"

    # delete
    vs.delete([ids[2]])
    assert not vs.exists(ids[2])

    # delete_by_metadata
    deleted = vs.delete_by_metadata({"category": "language"})
    assert deleted >= 1
    results = vs.search_by_metadata(filter={"category": "language"}, limit=10)
    assert len(results) == 0

    # clear
    vs.clear()
    assert vs.count() == 0

    # from_texts
    vs.drop_table()
    vs2 = PolarDBXVectorStore.from_texts(
        TEXTS[:3], EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs2.count() == 3
    vs2.drop_table()
    vs2.close()

    # from_documents
    docs = [Document(page_content=t, metadata=m) for t, m in zip(TEXTS[:3], METADATAS[:3])]
    vs3 = PolarDBXVectorStore.from_documents(
        docs, EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs3.count() == 3
    vs3.drop_table()
    vs3.close()

    # drop_table + close
    vs4 = make_store()
    vs4.add_texts(["temp"])
    vs4.drop_table()
    vs4.close()


# ==================== ASYNC ====================

async def test_async():
    vs = make_store()

    # aadd_texts
    ids = await vs.aadd_texts(TEXTS, metadatas=METADATAS)
    assert len(ids) == 5

    # aadd_documents
    docs = [Document(page_content="Java is also popular", metadata={"category": "language", "lang": "en"})]
    doc_ids = await vs.aadd_documents(docs)
    assert len(doc_ids) == 1

    # aadd_embeddings
    emb_pairs = [("Go is concise", EMB.embed_query("Go is concise"))]
    emb_ids = await vs.aadd_embeddings(emb_pairs, metadatas=[{"category": "language", "lang": "en"}])
    assert len(emb_ids) == 1

    # acount
    cnt = await vs.acount()
    assert cnt == 7

    # aget_by_ids
    docs = await vs.aget_by_ids([ids[0], ids[1]])
    assert len(docs) == 2

    # aupsert
    await vs.aupsert([Document(page_content="ASYNC UPDATED", metadata={"category": "updated", "lang": "en"})], ids=[ids[0]])
    docs = await vs.aget_by_ids([ids[0]])
    assert docs[0].page_content == "ASYNC UPDATED"

    # adelete
    await vs.adelete([ids[2]])
    assert not vs.exists(ids[2])

    # aclear
    await vs.aclear()
    cnt = await vs.acount()
    assert cnt == 0

    # afrom_texts
    vs2 = await PolarDBXVectorStore.afrom_texts(
        TEXTS[:3], EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs2.count() == 3
    vs2.drop_table()
    vs2.close()

    # afrom_documents
    docs = [Document(page_content=t, metadata=m) for t, m in zip(TEXTS[:3], METADATAS[:3])]
    vs3 = await PolarDBXVectorStore.afrom_documents(
        docs, EMB,
        host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASS,
        database=DB_NAME, table_name="test_polardbx_langchain", pre_delete_table=True,
    )
    assert vs3.count() == 3
    vs3.drop_table()
    vs3.close()

    # aclose
    vs4 = make_store()
    await vs4.aclose()
