"""Standard LangChain interface tests.

Subclasses VectorStoreIntegrationTests from langchain-tests to verify
that PolarDBXVectorStore conforms to the LangChain VectorStore interface.

These tests are independent from our custom integration tests and cover
the standard contract that all LangChain vector store integrations must
satisfy.
"""

import os
import sys
import uuid
from collections.abc import Generator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from _helpers import DB_HOST, DB_NAME, DB_PASS, DB_PORT, DB_USER
from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_polardbx import PolarDBXVectorStore


class TestPolarDBXStandard(VectorStoreIntegrationTests):
    """Standard test suite for PolarDBXVectorStore."""

    @pytest.fixture()
    def vectorstore(self) -> Generator[VectorStore, None, None]:  # type: ignore[override]
        """Get an empty vectorstore for standard tests.

        Each test gets a fresh store with a unique table name.
        The table is dropped in the finally block to ensure cleanup.
        """
        table_name = f"test_std_{uuid.uuid4().hex[:8]}"
        store = PolarDBXVectorStore(
            host=DB_HOST,
            port=DB_PORT,
            user=DB_USER,
            password=DB_PASS,
            database=DB_NAME,
            embedding=self.get_embeddings(),
            table_name=table_name,
            pre_delete_table=True,
        )
        try:
            yield store
        finally:
            try:
                store._drop_table()
            except Exception:
                pass
            store.close()
