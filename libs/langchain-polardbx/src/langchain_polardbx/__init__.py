from langchain_polardbx.vectorstores import NotSupportedError, PolarDBXVectorStore

# PolarDBXSQLDatabase requires [sql] extra; import lazily
try:
    from langchain_polardbx.sql import PolarDBXSQLDatabase, create_partitioned_table

    __all__ = [
        "PolarDBXVectorStore",
        "PolarDBXSQLDatabase",
        "NotSupportedError",
        "create_partitioned_table",
    ]
except ImportError:
    __all__ = ["PolarDBXVectorStore", "NotSupportedError"]

__version__ = "0.3.0"
