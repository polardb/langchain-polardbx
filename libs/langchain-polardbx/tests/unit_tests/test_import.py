"""Unit tests that don't require a database connection."""


def test_import() -> None:
    """Test that the package can be imported."""
    from langchain_polardbx import PolarDBXVectorStore

    assert PolarDBXVectorStore is not None


def test_import_vectorstores() -> None:
    """Test that the vectorstores module can be imported."""
    from langchain_polardbx.vectorstores import PolarDBXVectorStore

    assert PolarDBXVectorStore is not None
