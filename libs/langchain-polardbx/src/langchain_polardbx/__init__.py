from langchain_polardbx.vectorstores import (
    Column,
    NotSupportedError,
    PolarDBXVectorStore,
)

# PolarDBXSQLDatabase requires [sql] extra; import lazily
try:
    from langchain_polardbx.sql import PolarDBXSQLDatabase, create_partitioned_table

    __all__ = [
        "PolarDBXVectorStore",
        "PolarDBXSQLDatabase",
        "NotSupportedError",
        "Column",
        "create_partitioned_table",
    ]
except ImportError:
    __all__ = ["PolarDBXVectorStore", "NotSupportedError", "Column"]

try:
    from importlib.metadata import version as _version

    __version__ = _version("langchain-polardbx")
except Exception:
    __version__ = "0.4.2"
