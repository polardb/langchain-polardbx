"""Unit tests that don't require a database connection."""


def test_import() -> None:
    """Test that the package can be imported."""
    from langchain_polardbx import PolarDBXVectorStore

    assert PolarDBXVectorStore is not None


def test_import_vectorstores() -> None:
    """Test that the vectorstores module can be imported."""
    from langchain_polardbx.vectorstores import PolarDBXVectorStore

    assert PolarDBXVectorStore is not None


def test_import_not_supported_error() -> None:
    """Test that NotSupportedError can be imported from both paths."""
    from langchain_polardbx import NotSupportedError as NSE1
    from langchain_polardbx.vectorstores import NotSupportedError as NSE2

    assert NSE1 is not None
    assert NSE2 is not None
    assert NSE1 is NSE2  # same class
    assert issubclass(NSE1, NotImplementedError)


def test_distance_strategy_literal() -> None:
    """Test that distance_strategy accepts 'inner_product' in type hints."""
    import inspect

    from langchain_polardbx import PolarDBXVectorStore

    sig = inspect.signature(PolarDBXVectorStore.__init__)
    param = sig.parameters.get("distance_strategy")
    assert param is not None
    # The annotation should be a Literal that includes "inner_product"
    annotation_str = str(param.annotation)
    assert "inner_product" in annotation_str
