"""PolarDB-X SQL Database integration with DDL reflection compatibility.

This module provides:
- PolarDBXDialect: A custom SQLAlchemy dialect that fixes PolarDB-X DDL
  reflection issues (tab indentation, ENUM value spacing) via subclassing,
  with zero global side effects.
- PolarDBXSQLDatabase: A thin wrapper around langchain_community's
  SQLDatabase that auto-swaps the connection URI to use our dialect.

Requires the ``sql`` extra: ``pip install langchain-polardbx[sql]``
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from sqlalchemy.dialects.mysql.pymysql import MySQLDialect_pymysql
from sqlalchemy.dialects.mysql.reflection import MySQLTableDefinitionParser
from sqlalchemy.types import UserDefinedType
from sqlalchemy.util import memoized_property


class PolarDBXVector(UserDefinedType):
    """PolarDB-X VECTOR type for SQLAlchemy schema reflection.

    This type exists so that PolarDBXDialect can correctly reflect tables
    containing VECTOR columns without crashing. It does NOT provide vector
    operations — use raw SQL for vector similarity search.
    """

    cache_ok = True

    def __init__(self, dimension: Optional[int] = None):
        self.dimension = dimension

    def get_col_spec(self, **kw: Any) -> str:
        if self.dimension is not None:
            return f"VECTOR({self.dimension})"
        return "VECTOR"

    @property
    def python_type(self) -> type:  # type: ignore[override]
        return list


class PolarDBXTableDefinitionParser(MySQLTableDefinitionParser):
    """MySQLTableDefinitionParser with PolarDB-X DDL format fixes.

    PolarDB-X SHOW CREATE TABLE output has two format differences
    from standard MySQL:
    1. Tab indentation instead of two-space indentation
    2. ENUM/SET value lists have spaces after commas

    This parser normalizes both before delegating to the parent parser.

    Note: PolarDB-X ``VECTOR INDEX`` lines (e.g.
    ``VECTOR INDEX `vi`(`embedding`) M=6 DISTANCE=COSINE``) are not
    recognized by the upstream MySQL parser and will be skipped with a
    warning. This is expected — the index info is not needed for SQL
    query generation via SQLDatabase.
    """

    def parse(self, show_create: str, charset: Optional[str]) -> Any:
        # Fix 1: Tab indentation -> two spaces (standard MySQL format)
        show_create = show_create.replace("\n\t", "\n  ")
        # Fix 2: ENUM/SET value list spacing normalization
        #  enum('A', 'B') -> enum('A','B')
        #
        # This regex matches the pattern '<quote>,<space><quote>' which
        # only occurs between ENUM/SET values in valid DDL. String literals
        # like DEFAULT 'hello, world' are unaffected because the comma is
        # inside the quotes, not between them.
        show_create = re.sub(r"',\s+'", "','", show_create)
        return super().parse(show_create, charset)


class PolarDBXDialect(MySQLDialect_pymysql):
    """MySQL pymysql dialect with PolarDB-X DDL reflection fixes.

    This dialect overrides _tabledef_parser to return a
    PolarDBXTableDefinitionParser instead of the default parser.
    Only connections using this dialect (polardbx+pymysql://)
    are affected; other MySQL connections are completely unaffected.
    """

    # Enable SQLAlchemy SQL compilation caching. This dialect only
    # overrides DDL reflection logic, not SQL compilation, so caching
    # is safe and avoids repeated deprecation warnings.
    supports_statement_cache = True

    ischema_names = {
        **MySQLDialect_pymysql.ischema_names,
        "vector": PolarDBXVector,
    }

    @memoized_property
    def _tabledef_parser(self) -> PolarDBXTableDefinitionParser:
        preparer = self.identifier_preparer
        return PolarDBXTableDefinitionParser(self, preparer)


def _register_dialect() -> None:
    """Register the polardbx dialect with SQLAlchemy's registry.

    This is a fallback for the entry-point in pyproject.toml which also
    registers ``polardbx``. The registry call specifically registers
    ``polardbx.pymysql`` (the dialect+driver form), ensuring URIs work
    even when the package is imported without being pip-installed.
    """
    from sqlalchemy.dialects import registry

    registry.register(
        "polardbx.pymysql",
        "langchain_polardbx.sql",
        "PolarDBXDialect",
    )


# Register at import time so polardbx+pymysql:// URIs work
_register_dialect()


# Lazy import: only require langchain_community when user actually
# uses PolarDBXSQLDatabase, not at module import time.
#
# Note: langchain_community is being sunset by LangChain. When the
# SQLDatabase class is migrated to a standalone package, update this
# import and the [sql] extra in pyproject.toml accordingly.
try:
    from langchain_community.utilities.sql_database import SQLDatabase

    _SQL_AVAILABLE = True
except ImportError:
    SQLDatabase = None  # type: ignore[assignment,misc]
    _SQL_AVAILABLE = False


class PolarDBXSQLDatabase(SQLDatabase if _SQL_AVAILABLE else object):  # type: ignore[misc]
    """SQLDatabase with PolarDB-X DDL reflection compatibility.

    Requires the ``sql`` extra: ``pip install langchain-polardbx[sql]``

    Automatically applies DDL format fixes when reflecting table
    schemas from PolarDB-X. Usage is identical to SQLDatabase:

        from langchain_polardbx import PolarDBXSQLDatabase
        db = PolarDBXSQLDatabase.from_uri(
            "mysql+pymysql://user:pass@host:3306/db"
        )
        db.run("SELECT * FROM my_table")
        db.get_table_info(["my_table"])
    """

    @classmethod
    def from_uri(
        cls,
        database_uri: str,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> "PolarDBXSQLDatabase":
        if not _SQL_AVAILABLE:
            raise ImportError(
                "PolarDBXSQLDatabase requires the 'sql' extra. "
                "Install it with: pip install langchain-polardbx[sql]"
            )

        # Ensure dialect is registered
        _register_dialect()

        # Auto-swap mysql+pymysql:// -> polardbx+pymysql://
        if database_uri.startswith("mysql+pymysql://"):
            database_uri = "polardbx+pymysql://" + database_uri[
                len("mysql+pymysql://") :
            ]
        elif database_uri.startswith("mysql://"):
            database_uri = "polardbx+pymysql://" + database_uri[
                len("mysql://") :
            ]

        return super().from_uri(database_uri, engine_args, **kwargs)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Standalone DDL helpers — no dependency on langchain_community
# ---------------------------------------------------------------------------


def _build_partition_clause(
    partition_by: Optional[str] = None,
    partition_column: str = "id",
    partitions: int = 0,
    broadcast: bool = False,
    locality: Optional[str] = None,
    partition_defs: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the PARTITION/BROADCAST/LOCALITY clause for CREATE TABLE.

    Returns an empty string for a single (non-partitioned) table.
    """
    parts: List[str] = []

    if broadcast:
        parts.append("BROADCAST")
    elif partition_by:
        pby = partition_by.upper()
        if pby in ("HASH", "KEY"):
            parts.append(
                f"PARTITION BY {pby}({partition_column}) "
                f"PARTITIONS {partitions}"
            )
        elif pby == "RANGE":
            items = []
            for d in partition_defs or []:
                vlt = d["values_less_than"]
                if isinstance(vlt, str) and vlt.upper() == "MAXVALUE":
                    items.append(
                        f"PARTITION {d['name']} VALUES LESS THAN (MAXVALUE)"
                    )
                else:
                    items.append(
                        f"PARTITION {d['name']} VALUES LESS THAN ({vlt})"
                    )
            parts.append(
                f"PARTITION BY RANGE({partition_column}) ("
                + ", ".join(items) + ")"
            )
        elif pby == "LIST":
            items = []
            for d in partition_defs or []:
                vals = d["values_in"]
                val_str = ", ".join(
                    repr(v) if isinstance(v, str) else str(v)
                    for v in vals
                )
                items.append(
                    f"PARTITION {d['name']} VALUES IN ({val_str})"
                )
            parts.append(
                f"PARTITION BY LIST({partition_column}) ("
                + ", ".join(items) + ")"
            )

    if locality:
        parts.append(f"LOCALITY='{locality}'")

    return "".join(f" {p}" for p in parts) if parts else ""


def create_partitioned_table(
    uri: str,
    table_name: str,
    columns: List[str],
    partition_by: Optional[str] = None,
    partition_column: str = "id",
    partitions: int = 0,
    broadcast: bool = False,
    locality: Optional[str] = None,
    partition_defs: Optional[List[Dict[str, Any]]] = None,
    if_not_exists: bool = True,
) -> None:
    """Create a table on PolarDB-X with optional partition clauses.

    This function executes raw DDL via SQLAlchemy and does NOT require
    langchain_community. It only needs the ``sqlalchemy`` and ``pymysql``
    dependencies (part of the core install).

    Args:
        uri: Connection URI, e.g.
            ``"mysql+pymysql://user:pass@host:3306/db"``.
            Will be auto-swapped to ``polardbx+pymysql://``.
        table_name: The table name.
        columns: Column definitions as SQL strings, e.g.
            ``["id BIGINT NOT NULL AUTO_INCREMENT",
            "name VARCHAR(255)", "PRIMARY KEY (id)"]``.
        partition_by: Partition strategy: "HASH", "KEY", "RANGE", or
            "LIST". None for single table.
        partition_column: Column to partition on. Defaults to "id".
        partitions: Number of partitions (HASH/KEY only).
        broadcast: If True, create a broadcast table.
        locality: Storage node, e.g. "dn=xxx".
        partition_defs: Partition definitions (RANGE/LIST only).
        if_not_exists: If True, add IF NOT EXISTS.

    Example:
        .. code-block:: python

            from langchain_polardbx import create_partitioned_table

            create_partitioned_table(
                uri="mysql+pymysql://user:pass@host:3306/db",
                table_name="orders",
                columns=[
                    "id BIGINT NOT NULL AUTO_INCREMENT",
                    "user_id BIGINT NOT NULL",
                    "amount DECIMAL(10,2)",
                    "created_at DATETIME",
                    "PRIMARY KEY (id)",
                ],
                partition_by="HASH",
                partition_column="user_id",
                partitions=16,
            )
    """
    _register_dialect()

    # Auto-swap mysql:// to polardbx://
    if uri.startswith("mysql+pymysql://"):
        uri = "polardbx+pymysql://" + uri[len("mysql+pymysql://"):]
    elif uri.startswith("mysql://"):
        uri = "polardbx+pymysql://" + uri[len("mysql://"):]

    # Validate params
    if partition_by:
        partition_by = partition_by.upper()
        if partition_by not in ("HASH", "KEY", "RANGE", "LIST"):
            raise ValueError(
                f"Invalid partition_by: {partition_by}. "
                "Must be 'HASH', 'KEY', 'RANGE', or 'LIST'."
            )
        if partition_by in ("HASH", "KEY") and partitions <= 0:
            raise ValueError(
                "partitions must be > 0 for HASH/KEY partitioning."
            )
        if partition_by in ("RANGE", "LIST") and not partition_defs:
            raise ValueError(
                "partition_defs required for RANGE/LIST partitioning."
            )
    if broadcast and partition_by:
        raise ValueError(
            "broadcast and partition_by are mutually exclusive."
        )

    # Build DDL
    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    col_defs = ",\n    ".join(columns)
    partition_clause = _build_partition_clause(
        partition_by=partition_by,
        partition_column=partition_column,
        partitions=partitions,
        broadcast=broadcast,
        locality=locality,
        partition_defs=partition_defs,
    )

    ddl = (
        f"CREATE TABLE {exists_clause}`{table_name}` (\n"
        f"    {col_defs}\n"
        f"){partition_clause}"
    )

    from sqlalchemy import create_engine, text

    engine = create_engine(uri)
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
    engine.dispose()
